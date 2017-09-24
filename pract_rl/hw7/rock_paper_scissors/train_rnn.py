#!/usr/bin/env python3
import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from rockpaperscissors import RockPaperScissors

RNN_HIDDEN = 32

GAMMA = 0.99
BATCH_SIZE = 16
EXP_STEPS_COUNT = 10
EXP_BUFFER_SIZE = 100
EXP_BUFFER_POPULATE = 16


def make_env():
    env = RockPaperScissors()
    return gym.wrappers.TimeLimit(env, max_episode_steps=100)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, actions):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.nonlin = nn.ELU()
        self.out = nn.Linear(hidden_size, actions)
        self.sm = nn.Softmax()

    def forward(self, x, h=None):
        # add an extra time dimension
        x = x.unsqueeze(dim=1)
        rnn_out, _ = self.rnn(x, h)
        out = self.nonlin(rnn_out)
        out = self.out(out)
        # remove time dimension back
        out = out.squeeze(dim=1)
        out = self.sm(out)
        return out

    def forward_multistep(self, x, h=None):
        """
        Called by our code to calculate many timestemps at once
        """
        rnn_out, _ = self.rnn(x, h)
        out = self.nonlin(rnn_out)
        out = self.out(out)
        # reshape tensor to be able to apply softmax
        old_shape = out.size()
        out = out.view(-1, old_shape[-1])
        out = self.sm(out)
        # reshape back
        out = out.view(old_shape)
        return out


if __name__ == "__main__":
    env = make_env()
    params = ptan.common.env_params.EnvParams.from_env(env)
    ptan.common.env_params.register(params)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = Model(observation_shape[0], RNN_HIDDEN, n_actions)

    agent = ptan.agent.PolicyAgent(model)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=2*EXP_STEPS_COUNT)
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=EXP_BUFFER_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
                prob = out[batch_idx, idx, exp.action]
                result += -prob.log() * R
                count += 1
        return result / count

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
