#!/usr/bin/env python3
import gym
import ptan
import itertools
import collections
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

log = gym.logger

GAMES_COUNT = 10
LSTM_SIZE = 512
LEARNING_RATE = 0.001
GAMMA = 0.99
VALUE_STEPS = 4
EXPERIENCE_LEN = 10
STATE_WARMUP_LEN = 5

ENTROPY_BETA = 0.1
REPORT_ITERS = 100
BATCH_ITERS = 4
SYNC_ITERS = 10
CUDA = True


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
    def __init__(self, state_net, policy_net, cuda=False):
        super(StatefulAgent, self).__init__()
        self.state_net = state_net
        self.policy_net = policy_net
        self.sm = nn.Softmax()
        selector = ptan.actions.ProbabilityActionSelector()
        self.action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.25, selector=selector)
        self.cuda = cuda

    def initial_state(self):
        """
        Create zero state of lstm
        :return: 
        """
        c0 = torch.zeros((1, 1, LSTM_SIZE))
        h0 = torch.zeros((1, 1, LSTM_SIZE))
        if self.cuda:
            return c0.cuda(), h0.cuda()
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
        return Variable(c0), Variable(h0)

    def unpack_agent_states(self, agent_states):
        """
        Unpack back from LSTM into list of individual states
        :param agent_states: tuple of tensors
        :return: list of tuples of tensors 
        """
        c0, h0 = agent_states
        c0_list = torch.split(c0.data, 1, dim=1)
        h0_list = torch.split(h0.data, 1, dim=1)
        return list(zip(c0_list, h0_list))

    def __call__(self, states, agent_states):
        packed_states = self.pack_agent_states(agent_states)
        states_v = Variable(torch.from_numpy(states))
        if self.cuda:
            states_v = states_v.cuda()
        state_out, packed_states = self.state_net(states_v, packed_states)
        policy_out = self.policy_net(state_out)
        prob_out = self.sm(policy_out)
        actions = self.action_selector(prob_out.data.cpu().numpy())
        return actions, self.unpack_agent_states(packed_states)


# TODO: we can optimize for speed by calculating convolutions for the whole experience chain,
# but this will require further split of state_net for conv_net and rnn_net
def calculate_loss(exp, state_net, policy_net, value_net, stats_dict, cuda=False):
    # calculate states, values and policy for our experience sequence
    rnn_state = None
    values, policies, log_policies = [], [], []
    for exp_idx, exp_item in enumerate(exp):
        obs_v = Variable(torch.from_numpy(np.expand_dims(exp_item.state, axis=0)))
        if cuda:
            obs_v = obs_v.cuda()
        state_v, rnn_state = state_net(obs_v, rnn_state)
        if exp_idx >= STATE_WARMUP_LEN:
            value_v = value_net(state_v)
            raw_logits_v = policy_net(state_v)
            log_policy_v = F.log_softmax(raw_logits_v)
            policy_v = F.softmax(raw_logits_v)
            values.append(value_v[0])
            policies.append(policy_v[0])
            log_policies.append(log_policy_v[0])

    # calculate accumulated discounted rewards using VALUE_STEPS Bellman approximation
    rewards = []
    vals_window = []
    for exp_item, value_v in reversed(list(zip(exp, values))):
        vals_window = [exp_item.reward + GAMMA * val for val in vals_window]
        reward = exp_item.reward
        if not exp_item.done:
            reward += value_v.data.cpu().numpy()[0]
        vals_window.insert(0, reward)
        vals_window = vals_window[:VALUE_STEPS]
        rewards.append(vals_window[-1])

    # observation ended prematurely
    if not rewards:
        return None

    # can calculate loss using precomputed approximation of total reward and our policies
    loss_v = Variable(torch.FloatTensor([0]))
    if cuda:
        loss_v = loss_v.cuda()
    for exp_item, total_reward, value_v, policy_idx in zip(reversed(exp), rewards, reversed(values),
                                                           range(len(policies)-1, -1, -1)):
        policy_v = policies[policy_idx]
        log_policy_v = log_policies[policy_idx]
        # value loss
        total_reward_v = Variable(torch.FloatTensor([total_reward]))
        if cuda:
            total_reward_v = total_reward_v.cuda()
        loss_value_v = F.mse_loss(value_v, total_reward_v)
        # policy loss
        advantage = total_reward - value_v.data.cpu().numpy()[0]
        loss_policy_v = torch.mul(log_policy_v[exp_item.action], -advantage)
        # entropy loss
        loss_entropy_v = ENTROPY_BETA * torch.sum(policy_v * log_policy_v)
        loss_v += loss_value_v + loss_policy_v + loss_entropy_v
        stats_dict['advantage'] += advantage
        stats_dict['reward_disc'] += total_reward
        stats_dict['loss_count'] += 1
        stats_dict['loss_value'] += loss_value_v.data.cpu().numpy()[0]
        stats_dict['loss_policy'] += loss_policy_v.data.cpu().numpy()[0]
        stats_dict['loss_entropy'] += loss_entropy_v.data.cpu().numpy()[0]
        stats_dict['loss'] += loss_v.data.cpu().numpy()[0]

    return loss_v


def report(writer, iter_idx, stats_dict):
    mean_reward = stats_dict.get('rewards')
    if mean_reward is not None:
        mean_reward /= stats_dict['rewards_count']
    l_count = stats_dict['loss_count']
    loss = stats_dict['loss'] / l_count
    loss_value = stats_dict['loss_value'] / l_count
    loss_policy = stats_dict['loss_policy'] / l_count
    loss_entropy = stats_dict['loss_entropy'] / l_count
    advantage = stats_dict['advantage'] / l_count
    reward_disc = stats_dict['reward_disc'] / l_count
    log.info("%d: mean_reward=%s, done_games=%d", iter_idx, mean_reward, stats_dict.get('rewards_count', 0))
    if mean_reward is not None:
        writer.add_scalar("reward", mean_reward, iter_idx)
    writer.add_scalar("reward_disc", reward_disc, iter_idx)
    writer.add_scalar("loss_total", loss, iter_idx)
    writer.add_scalar("loss_value", loss_value, iter_idx)
    writer.add_scalar("loss_policy", loss_policy, iter_idx)
    writer.add_scalar("loss_entropy", loss_entropy, iter_idx)
    writer.add_scalar("advantage", advantage, iter_idx)
    stats_dict.clear()


def make_env():
    return ptan.common.wrappers.AtariWrapper(gym.make("KungFuMaster-v0"))


if __name__ == "__main__":
    env = make_env()

    log.info("Observations: %s", env.observation_space)
    log.info("Actions: %s", env.action_space)

    state_net = StateNet(env.observation_space.shape, LSTM_SIZE)
    value_net = ValueNet(LSTM_SIZE)
    policy_net = PolicyNet(LSTM_SIZE, env.action_space.n)
    params = itertools.chain(state_net.parameters(), value_net.parameters(), policy_net.parameters())
    optimizer = optim.RMSprop(params, lr=LEARNING_RATE)
    writer = SummaryWriter(comment="-lr=0.001")

    target_state_net = ptan.agent.TargetNet(state_net)
    target_policy_net = ptan.agent.TargetNet(policy_net)

    if CUDA:
        state_net.cuda()
        value_net.cuda()
        policy_net.cuda()

    agent = StatefulAgent(target_state_net.target_model, target_policy_net.target_model, cuda=False)
    exp_source = ptan.experience.ExperienceSource([make_env() for _ in range(GAMES_COUNT)],
                                                  agent, steps_count=EXPERIENCE_LEN, steps_delta=5)
    stats_dict = collections.Counter()

    for idx, exp in enumerate(exp_source):
        loss_v = calculate_loss(exp, state_net, policy_net, value_net, stats_dict, cuda=CUDA)
        if loss_v is None:
            continue
        loss_v.backward()
        # perform update of gradients in batches. It's the same as do minibatches, but less parallel in GPU
        if (idx+1) % BATCH_ITERS == 0:
            params = itertools.chain(state_net.parameters(), value_net.parameters(), policy_net.parameters())
            nn.utils.clip_grad_norm(params, max_norm=10.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

        if (idx+1) % SYNC_ITERS == 0:
            target_state_net.sync()
            target_policy_net.sync()

        rewards = exp_source.pop_total_rewards()
        if rewards:
            stats_dict['rewards'] += np.sum(rewards)
            stats_dict['rewards_count'] += len(rewards)

        if (idx+1) % REPORT_ITERS == 0:
            report(writer, idx+1, stats_dict)
    pass
