import gymnasium as gym
from sacx.wrapper.GymVectorEnvToSB3 import GymVectorEnvToSB3
from stable_baselines3.common.vec_env import VecMonitor

from . import RandomEnv

def make_env(tasks, seed, max_episode_steps, render_mode=None, vector_strategy="async", use_one_hot=True, **kwargs):
    env = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy=vector_strategy,
        envs_list=tasks,
        use_one_hot=use_one_hot,
        reward_function_version="v2",
        max_episode_steps=max_episode_steps,
        seed=seed,
        terminate_on_success=False,
        render_mode=render_mode,
        **kwargs
    )
    env = GymVectorEnvToSB3(env)
    return VecMonitor(env)

def make_cl_env(tasks, seed, max_episode_steps, p=None, use_one_hot=True, **kwargs):
    envs = [
        gym.make(
            "Meta-World/MT1",
            env_name=task,
            seed=seed+ii,
            reward_function_version="v2",
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
            **kwargs
        )
        for ii, task in enumerate(tasks)
    ]
    return RandomEnv.RandomEnv(envs, p=p, use_one_hot=use_one_hot)


def make_mt1_env(task, seed, max_episode_steps, render_mode, vector_strategy="async"):
    return make_env([task], seed, max_episode_steps, render_mode, vector_strategy)

def make_mt3_env(seed, max_episode_steps, render_mode, vector_strategy="async"):
    return make_env(["reach-v3", "push-v3", "pick-place-v3"], seed, max_episode_steps, render_mode, vector_strategy)

def make_mt10_env(seed, max_episode_steps, render_mode, vector_strategy="async"):
    return make_env(
        [
            "reach-v3",
            "push-v3",
            "pick-place-v3",
            "door-open-v3",
            "drawer-open-v3",
            "drawer-close-v3",
            "button-press-topdown-v3",
            "peg-insert-side-v3",
            "window-open-v3",
            "window-close-v3",
        ],
        seed, max_episode_steps, render_mode, vector_strategy
    )