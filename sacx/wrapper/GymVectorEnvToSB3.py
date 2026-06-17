import numpy as np

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

class GymVectorEnvToSB3(VecEnv):
    #https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
    def __init__(self, gym_vec_env: gym.vector.VectorEnv):
        self.gym_env = gym_vec_env
        self.task_names = gym_vec_env.spec.kwargs.get("envs_list", None)
        super().__init__(
            num_envs=gym_vec_env.num_envs,
            observation_space=gym_vec_env.single_observation_space,
            action_space=gym_vec_env.single_action_space,
        )

    def reset(self):
        obs, _ = self.gym_env.reset()
        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.gym_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)

        infos = [{} for _ in range(self.num_envs)]

        for key, value in info.items():
            # Case 1: per-env values (array-like)
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_envs:
                for i in range(self.num_envs):
                    infos[i][key] = value[i]
            # Case 2: scalar / shared values
            else:
                for i in range(self.num_envs):
                    infos[i][key] = value

        return obs, rewards, dones, infos

    def close(self):
        self.gym_env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.gym_env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.gym_env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        method = getattr(self.gym_env, method_name)
        return [method(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]