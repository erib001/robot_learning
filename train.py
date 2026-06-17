import gymnasium as gym
import metaworld

from sacx.environments import environments
from sacx import callback
from sacx.sac import SAC, SAC_PC_Grad, SAC_Alpha
from sacx.sac.multihead_phv1 import SAC_MultiheadSB, SAC_MultiheadMB, SAC_Multihead

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

import yaml
import argparse

ALGO_TABLE = {
    "sac": SAC,
    "sac_pc_grad": SAC_PC_Grad,
    "sac_pcgrad": SAC_PC_Grad,
    "sac_alpha": SAC_Alpha,
    "sac_multihead_sb": SAC_MultiheadSB,
    "sac_multihead_mb": SAC_MultiheadMB,
    "sac_multihead_actor": SAC_Multihead

}

def make(rank, settings):
    def _make():
        settings["seed"] += rank
        envs = environments.make_cl_env(**settings)
        settings["seed"] -= rank
        envs = Monitor(envs)
        return envs

    return _make


if __name__ == "__main__":

    # load config
    parser = argparse.ArgumentParser(description="training SAC")
    parser.add_argument("config", nargs=1, help="filename of the paramters to train. f.e.: template would train param/template.yaml")
    args = parser.parse_args()


    #method = args.config[0].split("_")[1]
    method_temp = args.config[0].split("_")[1]
    method = f"{method_temp}"

    with open(f"./param/{args.config[0]}.yaml") as f:
        config = yaml.safe_load(f)

    ID = config["id"]
    NAMESPACE = config["ns"]
    num_tasks = len(config["env_settings"]["tasks"])

    try:
        max_elements = config["misc"]["max_elements"]
        dynamicP = config["misc"]["dynamicP"]
        updateP_eval_calls = config["misc"]["updateP_eval_calls"]
    except:
        max_elements = 10
        dynamicP = False
        updateP_eval_calls = 10
        
    num_envs = 16
    envs = SubprocVecEnv([make(ii, config["env_settings"]) for ii in range(num_envs)], start_method="spawn")

    config["env_settings"]["seed"] += num_envs
    eval_env = environments.make_env(**config["env_settings"])
    config["env_settings"]["seed"] -= num_envs

    logger = configure(
                f"./metaworld_logs_final/{NAMESPACE}/{ID}_{method}_SAC",
                ["tensorboard"]
            )
    
    agent = ALGO_TABLE[config["algo"].lower()](
        env=envs,
        seed=config["env_settings"]["seed"],
        **config["sac_settings"]
    )
    agent.set_logger(logger)

    # define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["misc"]["CHECKPOINT_FREQ"] // num_envs, 1),
        save_replay_buffer=False,
        save_path=f"./metaworld_models_final/{NAMESPACE}/checkpoints/",
        name_prefix=f"{ID}_{method}",
        verbose=2,
    )
    eval_callback = callback.CustomEvalCallback_CL(
        eval_env,
        best_model_save_path=f"./metaworld_models_final/{NAMESPACE}/{ID}_{method}_best/",
        log_path=f"./metaworld_logs_final/{NAMESPACE}/{ID}_{method}_eval",
        eval_freq=max(config["misc"]["EVAL_FREQ"] // num_envs, 1),
        n_eval_episodes=config["misc"]["N_EVAL_EPISODES"],
        deterministic=True,
        render=False,
        verbose=1,
        warn=False,
        max_episode_steps=config["env_settings"]["max_episode_steps"],
        max_elements_rates = max_elements,
        env = envs,
        dynamicP = dynamicP,
        updateP_eval_calls = updateP_eval_calls
    )


    # learn!
    agent.learn(
        total_timesteps=config["misc"]["TIMESTEPS"]*num_tasks,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    # save and clean up
    agent.save(f"./metaworld_models_final/{NAMESPACE}/{ID}_{method}_final")
    agent.save_replay_buffer(f"./metaworld_models_final/{NAMESPACE}/{ID}_{method}_replay_buffer_final")

    envs.close()
    eval_env.close()

