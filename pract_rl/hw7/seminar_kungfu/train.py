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


if __name__ == "__main__":
    env = gym.make("KungFuMaster-v0")
    obs = torch.from_numpy(np.moveaxis(env.reset(), 2, 0))
    vutils.save_image(obs, "obs.png")
    log.info("Observations: %s", env.observation_space)

    pass
