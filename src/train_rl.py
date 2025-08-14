import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import RoadEnv


class GymRoadEnv(gym.Env):
    """Gym wrapper for the RoadEnv implemented in `env.py`."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, distances, cycles, offsets, g_durs, v_max, a_max, dt=1.0):
        super().__init__()
        self._env = RoadEnv(distances, cycles, offsets, g_durs, v_max, a_max, dt=dt)
        # observation: pos, vel, dist_to_next, phase
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        obs_high = np.array([np.sum(distances), v_max, np.sum(distances), max(cycles)], dtype=float)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=float)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def render(self, mode="human"):
        self._env.render()

    def close(self):
        pass


def make_env():
    distances = [300.0, 450.0, 280.0, 500.0, 350.0, 420.0, 380.0, 600.0]
    cycles = [65.0, 72.0, 58.0, 80.0, 70.0, 60.0, 75.0, 90.0]
    offsets = [10.0, 20.0, 5.0, 30.0, 12.0, 25.0, 15.0, 40.0]
    g_durs = [25.0, 22.0, 18.0, 28.0, 20.0, 18.0, 24.0, 30.0]
    v_max = 60.0 / 3.6
    a_max = 2.0
    return GymRoadEnv(distances, cycles, offsets, g_durs, v_max, a_max, dt=1.0)


if __name__ == "__main__":
    # Create vectorized environment for Stable-Baselines3
    env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs")
    model.learn(total_timesteps=200_000)
    model.save("./models/ppo_road")

    # quick evaluation
    eval_env = make_env()
    obs = eval_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rew, done, info = eval_env.step(int(action))
        total_reward += rew
        eval_env.render()
    print("Eval done, total_reward=", total_reward, "info=", info)
