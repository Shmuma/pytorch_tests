import gym

from rockpaperscissors import RockPaperScissors


def make_env():
    env = RockPaperScissors()
    return gym.wrappers.TimeLimit(env, max_episode_steps=100)


if __name__ == "__main__":
    env = make_env()
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.reset()
    obs = env.step(env.action_space.sample())[0]
    print(observation_shape, n_actions, obs)
    pass
