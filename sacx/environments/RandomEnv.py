import numpy as np

import gymnasium as gym
from typing import List, Optional

class RandomEnv(gym.Env):
    def __init__(self, envs: List[gym.Env], p = None, use_one_hot=True):
        self.envs = envs
        self.task_names = [env.spec.kwargs["env_name"] for env in envs]
        self.n_envs = len(envs) # bei mt3 ist n_envs = 3

        self.active_task = None
        self.p = p
        self.use_one_hot = use_one_hot
        env = self.envs[0]

        if self.use_one_hot:
            self.observation_space = gym.spaces.Box(
                low=np.append(env.observation_space.low, [0]*self.n_envs),
                high=np.append(env.observation_space.high, [1]*self.n_envs),
                shape=(env.observation_space.shape[0]+self.n_envs,),
                dtype=np.float32
            )
        else:
            self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _get_active_env(self) -> gym.Env:
        return self.envs[self.active_task]
    
    def _transform_obs(self, orig):
        if self.use_one_hot:
            return np.append(orig, (np.array(range(self.n_envs)) == self.active_task).astype(orig.dtype))
        else:
            return orig

    def set_p(self, value):
        self.p = value

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # set seed of random number generator
        super().reset(seed=seed)

        # select a new task
        #print("test", end="\t")
        self.active_task = self.np_random.choice(range(self.n_envs), 1, p=self.p)[0]
        #print(self.active_task)

        # reset the chosen task
        observation, info = self._get_active_env().reset()
        observation = self._transform_obs(observation)
        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # do the original step
        observation, reward, terminated, truncated, info = self._get_active_env().step(action)

        # extend observation with one-hot encoding
        observation = self._transform_obs(observation)

        return observation, reward, terminated, truncated, info
        