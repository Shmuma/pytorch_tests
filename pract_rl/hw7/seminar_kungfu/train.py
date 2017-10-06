#!/usr/bin/env python3
import gym
import ptan
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.utils as vutils

import cv2

log = gym.logger

LSTM_SIZE = 512


class StateNet(nn.Module):
    """
    Accepts observation and hidden RNN state and returns RNN output and new hidden state
    """
    def __init__(self, input_shape, output_size):
        super(StateNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
        )
        conv_out_size = self.get_conv_out_size(input_shape)
        self.rnn = nn.LSTM(conv_out_size, hidden_size=output_size, batch_first=True)

    def forward(self, x, rnn_state):
        conv_out = self.conv(x)
        flat_out = conv_out.view((x.size()[0], 1, -1))
        rnn_out, rnn_hid = self.rnn(flat_out, rnn_state)
        return rnn_out.squeeze(dim=1), rnn_hid

    def get_conv_out_size(self, input_shape):
        v = Variable(torch.zeros((1,) + input_shape))
        out = self.conv(v).view((-1, ))
        return out.size()[0]


class PolicyNet(nn.Module):
    def __init__(self, state_size, actions_n):
        super(PolicyNet, self).__init__()
        self.policy_out = nn.Linear(state_size, actions_n)

    def forward(self, state):
        return self.policy_out(state)


class ValueNet(nn.Module):
    def __init__(self, state_size):
        super(ValueNet, self).__init__()
        self.value_out = nn.Linear(state_size, 1)

    def forward(self, state):
        return self.value_out(state)


class StatefulAgent(ptan.agent.BaseAgent):
    def __init__(self, state_net, policy_net):
        super(StatefulAgent, self).__init__()
        self.state_net = state_net
        self.policy_net = policy_net
        self.sm = nn.Softmax()
        self.action_selector = ptan.actions.ProbabilityActionSelector()

    def initial_state(self):
        """
        Create zero state of lstm
        :return: 
        """
        c0 = Variable(torch.zeros((1, 1, LSTM_SIZE)))
        h0 = Variable(torch.zeros((1, 1, LSTM_SIZE)))
        return c0, h0

    def pack_agent_states(self, agent_states):
        """
        Result is tuple suitable to be passed to LSTM
        :param agent_states: list of agent states
        :return: tuple of tensors
        """
        c0_list, h0_list = zip(*agent_states)
        c0 = torch.cat(c0_list, dim=1)
        h0 = torch.cat(h0_list, dim=1)
        return c0, h0

    def unpack_agent_states(self, agent_states):
        """
        Unpack back from LSTM into list of individual states
        :param agent_states: tuple of tensors
        :return: list of tuples of tensors 
        """
        c0, h0 = agent_states
        c0_list = torch.split(c0, 1, dim=1)
        h0_list = torch.split(h0, 1, dim=1)
        return list(zip(c0_list, h0_list))

    def __call__(self, states, agent_states):
        packed_states = self.pack_agent_states(agent_states)
        states_v = Variable(torch.from_numpy(states))
        state_out, packed_states = self.state_net(states_v, packed_states)
        policy_out = self.policy_net(state_out)
        prob_out = self.sm(policy_out)
        actions = self.action_selector(prob_out.data.cpu().numpy())
        return actions, self.unpack_agent_states(packed_states)


if __name__ == "__main__":
    env = ptan.common.wrappers.AtariWrapper(gym.make("KungFuMaster-v0"))

    log.info("Observations: %s", env.observation_space)
    log.info("Actions: %s", env.action_space)

    obs = env.reset()
    state_net = StateNet(env.observation_space.shape, LSTM_SIZE)
    value_net = ValueNet(LSTM_SIZE)
    policy_net = PolicyNet(LSTM_SIZE, env.action_space.n)

    agent = StatefulAgent(state_net, policy_net)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=10)

    for exp in exp_source:
        print(exp)
        break
    pass
