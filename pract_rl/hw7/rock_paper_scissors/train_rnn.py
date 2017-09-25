#!/usr/bin/env python3
import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from rockpaperscissors import RockPaperScissors

RNN_HIDDEN = 32

GAMMA = 0.99
BATCH_SIZE = 16
POOL_SIZE = 4
EXP_STEPS_COUNT = 20
EXP_BUFFER_SIZE = 100
EXP_BUFFER_POPULATE = 16

ITERATIONS = 10000
TEST_EVERY_ITER = 10
TEST_RUNS = 10

RUN_PREFIX = "runs/rnn"


def make_env():
    env = RockPaperScissors()
    return gym.wrappers.TimeLimit(env, max_episode_steps=100)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, actions):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.nonlin = nn.ELU()
        self.out = nn.Linear(hidden_size, actions)

    def forward(self, x, h=None):
        out, _ = self.forward_hidden(x, h)
        return out

    def forward_hidden(self, x, h):
        # add an extra time dimension
        x = x.unsqueeze(dim=1)
        rnn_out, rnn_hid = self.rnn(x, h)
        out = self.nonlin(rnn_out)
        out = self.out(out)
        # remove time dimension back
        out = out.squeeze(dim=1)
        return out, rnn_hid

    def forward_multistep(self, x, h=None):
        """
        Called by our code to calculate many timestemps at once
        """
        rnn_out, _ = self.rnn(x, h)
        out = self.nonlin(rnn_out)
        out = self.out(out)
        return out


def play_episode(env, model):
    obs = env.reset()
    total_reward = 0.0
    hidden = None
    while True:
        qvals, hidden = model.forward_hidden(Variable(torch.from_numpy(np.array([obs], dtype=np.float32))), h=hidden)
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

    model = Model(observation_shape[0], RNN_HIDDEN, n_actions)

    action_selector = ptan.actions.epsilon_greedy.ActionSelectorEpsilonGreedy(epsilon=0.05, params=params)
    agent = ptan.agent.DQNAgent(model, action_selector=action_selector)
    exp_source = ptan.experience.ExperienceSource(env=[make_env() for _ in range(POOL_SIZE)], agent=agent, steps_count=2*EXP_STEPS_COUNT)
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=EXP_BUFFER_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    writer = SummaryWriter()

    def calc_loss(batch):
        """
        Calculate loss variable from batch
        """
        # prepare tensor
        max_len = max(map(len, batch))
        batch_input = np.zeros(shape=(len(batch), max_len, observation_shape[0]), dtype=np.float32)
        for i, batch_item in enumerate(batch):
            for j, e in enumerate(batch_item):
                batch_input[i, j] = e.state
        batch_input_v = Variable(torch.from_numpy(batch_input))
        out = model.forward_multistep(batch_input_v)

        result = Variable(torch.FloatTensor(1).zero_())
        count = 0
        for batch_idx, exps in enumerate(batch):
            if len(exps) < EXP_STEPS_COUNT:
                continue
            R = 0.0
            for idx, exp in reversed(list(enumerate(exps))):
                R *= GAMMA
                R += exp.reward

                if idx < EXP_STEPS_COUNT:
                    break
                qval = out[batch_idx, idx, exp.action]
                result += (qval - R) ** 2
                count += 1
        return result / count

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
            mean_reward = np.mean(total_rewards) if total_rewards else 0.0
            print("%d: mean reward %.3f in %d games, loss %.3f" % (iter, mean_reward, len(total_rewards), np.mean(losses)))
            writer.add_scalar(RUN_PREFIX + "/total_rewards", mean_reward, iter)
        writer.add_scalar(RUN_PREFIX + "/loss", np.mean(losses), iter)
        if iter % TEST_EVERY_ITER == 0:
            test_env = make_env()
            rewards = [play_episode(test_env, model) for _ in range(TEST_RUNS)]
            print("Test run: %.2f mean reward" % np.mean(rewards))
            writer.add_scalar(RUN_PREFIX + "/reward_test", np.mean(rewards), iter)


    writer.close()

    pass
