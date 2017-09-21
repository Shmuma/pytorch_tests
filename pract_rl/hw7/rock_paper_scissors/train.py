#!/usr/bin/env python3
import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from rockpaperscissors import RockPaperScissors


GAMMA = 0.99
BATCH_SIZE = 16
EXP_STEPS_COUNT = 10
EXP_BUFFER_SIZE = 100
EXP_BUFFER_POPULATE = 16


def make_env():
    env = RockPaperScissors()
    return gym.wrappers.TimeLimit(env, max_episode_steps=100)


if __name__ == "__main__":
    env = make_env()
    params = ptan.common.env_params.EnvParams.from_env(env)
    ptan.common.env_params.register(params)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = nn.Sequential(nn.Linear(observation_shape[0], 32),
                          nn.ELU(),
                          nn.Linear(32, n_actions),
                          nn.Softmax())

    agent = ptan.agent.PolicyAgent(model)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=EXP_STEPS_COUNT)
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=EXP_BUFFER_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            probs = model(state)[0]
            prob = probs[exps[0].action]
            result += -prob.log() * R
        return result / len(batch)

    for iter in range(1000):
        losses = []
        exp_buffer.populate(EXP_BUFFER_POPULATE)
        for batch in exp_buffer.batches(BATCH_SIZE):
            optimizer.zero_grad()
            loss = calc_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy()[0])
        total_rewards = exp_source.pop_total_rewards()
        mean_reward = np.mean(total_rewards) if total_rewards else 0.0

        print("%d: mean reward %.3f in %d games, loss %.3f" % (iter, mean_reward, len(total_rewards), np.mean(losses)))

    pass
