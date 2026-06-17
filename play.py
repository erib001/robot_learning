import os
import gymnasium as gym
import metaworld
import numpy as np

from sacx import SAC, eval
from sacx.environments import environments

import yaml
import argparse

if __name__ == "__main__":
    # load config
    parser = argparse.ArgumentParser(description="training SAC")
    parser.add_argument("config", nargs=1, help="filename of the paramters to train. f.e.: template would train param/template.yaml")
    args = parser.parse_args()
    
    with open(f"./param/{args.config[0]}.yaml") as f:
        config = yaml.safe_load(f)

    ID = config["id"]
    NAMESPACE = config["ns"]
    method = f"{args.config[0].split("_")[1]}"

    # Create environment with rendering

    num_of_envs = len(config["env_settings"]["tasks"])
    config["env_settings"]["seed"] += 2 # 0 for learning, 1 for eval during training, 2 for eval
    #config["env_settings"]["seed"] += num_of_envs # 0 for learning, 1 for eval during training, 2 for eval
    
    envs = environments.make_env(**config["env_settings"], render_mode=None, camera_id=0)

    # Load the trained model
    model_path = f"./metaworld_models_final/{NAMESPACE}/{ID}_{method}_best/best_model_reward.zip"
    #model_path = f"./metaworld_models_final/{NAMESPACE}/{ID}_{method}_best/best_model_success_rate.zip"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Trying final model instead...")
        model_path = f"./metaworld_models_final/{NAMESPACE}/{ID}_{method}_final.zip"

        if not os.path.exists(model_path):
            print(f"No trained model found!")
            print(f"Please train the model first using train_metaworld_sb3.py")
            exit(1)

    print(f"Loading model from: {model_path}")

    # create agent to act
    agent = SAC.load(model_path, env=None)

    # Run evaluation episodes
    num_episodes = 10

    success_rate, rewards = eval.evaluate(agent, envs, num_episodes, num_of_envs, True, True)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("=== Evaluation Complete ===")
    print(f"Success rate: {success_rate}")
    print(f"Mean Success rate: {100*np.mean(success_rate):.3f}%")
    print("=" * 60)
