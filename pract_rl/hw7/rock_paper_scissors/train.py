#!/usr/bin/env python3
import gym
import ptan
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from rockpaperscissors import RockPaperScissors


RUN_PREFIX = "runs/linear"

GAMMA = 0.99
BATCH_SIZE = 16
EXP_STEPS_COUNT = 10
EXP_BUFFER_SIZE = 100
EXP_BUFFER_POPULATE = 16

ITERATIONS = 10000
TEST_EVERY_ITER = 10
TEST_RUNS = 10


def make_env():
    env = RockPaperScissors()
    return gym.wrappers.TimeLimit(env, max_episode_steps=100)


def play_episode(env, model):
    obs = env.reset()
    total_reward = 0.0
    while True:
        qvals = model(Variable(torch.from_numpy(np.array([obs], dtype=np.float32))))
        qvals = qvals.data.cpu().numpy()[0]
        action = np.argmax(qvals)
        obs, reward, is_done, _ = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward


if __name__ == "__main__":
    env = make_env()
    params = ptan.common.env_params.EnvParams.from_env(env)
    ptan.common.env_params.register(params)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = nn.Sequential(nn.Linear(observation_shape[0], 32),
                          nn.ELU(),
                          nn.Linear(32, n_actions))

    action_selector = ptan.actions.epsilon_greedy.ActionSelectorEpsilonGreedy(epsilon=0.05, params=params)
    agent = ptan.agent.DQNAgent(model, action_selector=action_selector)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=EXP_STEPS_COUNT)
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=EXP_BUFFER_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    writer = SummaryWriter()

    def calc_loss(batch):
        """
        Calculate loss variable from batch
        """
        result = Variable(torch.FloatTensor(1).zero_())
        for exps in batch:
            R = 0.0
            for exp in reversed(exps):
                R *= GAMMA
                R += exp.reward

            # here we take first experience in the chain
            state = Variable(torch.from_numpy(np.array([exps[0].state], dtype=np.float32)))
            q_vals = model(state)[0]
            q_val = q_vals[exps[0].action]
            result += (q_val - R) ** 2
        return result / len(batch)

    for iter in range(ITERATIONS):
        losses = []
        exp_buffer.populate(EXP_BUFFER_POPULATE)
        for batch in exp_buffer.batches(BATCH_SIZE):
            optimizer.zero_grad()
            loss = calc_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy()[0])
        total_rewards = exp_source.pop_total_rewards()
        if total_rewards:
            mean_reward = np.mean(total_rewards)
            print("%d: mean reward %.3f in %d games, loss %.3f" % (iter, mean_reward, len(total_rewards), np.mean(losses)))
            writer.add_scalar(RUN_PREFIX + "/reward_train", mean_reward, iter)
        writer.add_scalar(RUN_PREFIX + "/loss", np.mean(losses), iter)
        if iter % TEST_EVERY_ITER == 0:
            test_env = make_env()
            rewards = [play_episode(test_env, model) for _ in range(TEST_RUNS)]
            print("Test run: %.2f mean reward" % np.mean(rewards))
            writer.add_scalar(RUN_PREFIX + "/reward_test", np.mean(rewards), iter)


    writer.close()
    pass
