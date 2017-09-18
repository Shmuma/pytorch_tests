#!/usr/bin/env python3
import gym
import ptan

import torch.nn as nn


from rockpaperscissors import RockPaperScissors


def make_env():
    env = RockPaperScissors()
    return gym.wrappers.TimeLimit(env, max_episode_steps=100)


if __name__ == "__main__":
    env = make_env()
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = nn.Sequential(nn.Linear(observation_shape[0], 32),
                          nn.ELU(),
                          nn.Linear(32, n_actions),
                          nn.Softmax())

    agent = ptan.agent.PolicyAgent(model)

    pass
