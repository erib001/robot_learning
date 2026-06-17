from . import environments

import numpy as np

def evaluate(agent, envs, n_eval_episodes, n_envs, render, verbose):
    total_rewards = np.zeros((n_eval_episodes, n_envs))
    success_count = np.zeros(n_envs)

    print(f"\nRunning {n_eval_episodes} evaluation episodes...")
    print("=" * 60)

    for episode in range(n_eval_episodes):
        obs = envs.reset()
        dones = np.array([False]*n_envs)
        episode_success = np.array([False]*n_envs)

        if verbose:
            print(f"\n--- Episode {episode + 1}/{n_eval_episodes} ---")

        while not dones.all():
            # Get action from policy (deterministic for evaluation)
            actions, _states = agent.predict(obs, deterministic=True)

            # Step environment
            obs, rewards, dones, infos = envs.step(actions)

            total_rewards[episode, :] += rewards

            # Check for success (Meta-World provides success info)
            for ii, info in enumerate(infos):
                if 'success' in info and info['success']:
                    episode_success[ii] = True

            # Render
            if render:
                envs.render()

        if verbose:
            print(f"Total reward: {total_rewards[episode, :]}")
            print(f"Success: {episode_success}")

        success_count += episode_success.astype(np.float32)

    return success_count / n_eval_episodes, total_rewards
