#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. move color channel axis to a first place
    2. strip top and bottom strides of the field
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self._observation(old_space.low), self._observation(old_space.high))

    def _observation(self, observation):
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(observation, 2, 0)
        # transform (3, 160, 160)
        other_dim = new_obs.shape[2]
        to_strip = (new_obs.shape[1] - other_dim) // 2
        new_obs = new_obs[:, to_strip:to_strip + other_dim, :]
        return new_obs.astype(np.float32) / 255.0


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0)
        )

        self.conv_elems = self._get_conv_elems(input_shape)

        self.out_pipe = nn.Sequential(
            nn.Linear(in_features=self.conv_elems, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        conv_out = conv_out.view(-1, self.conv_elems)
        return self.out_pipe(conv_out).squeeze(dim=1)


    def _get_conv_elems(self, input_shape):
        v = self.conv_pipe(Variable(torch.FloatTensor(1, *input_shape).zero_()))
        total = 1
        for s in v.size():
            total *= s
        return total


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=3, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=output_shape[0], out_channels=output_shape[0],
                               kernel_size=4, stride=4, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    env = InputWrapper(gym.make("Breakout-v0"))
    input_shape = env.observation_space.shape
    log.info("Observations: %s", input_shape)

    input_v = Variable(torch.from_numpy(np.array([env.reset()], dtype=np.float32)))

    net_discr = Discriminator(input_shape=input_shape)
    net_gener = Generator(output_shape=input_shape)
    log.info("Discriminator: %s", net_discr)
    log.info("Generator: %s", net_gener)

    out = net_discr(input_v)
    print(out.size())

    input_v = Variable(torch.FloatTensor(1, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1))
    out = net_gener(input_v)
    print(out.size())

    #vutils.save_image(out.data[0], "out.png", normalize=False)

    pass

