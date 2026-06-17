import os
import warnings

import gymnasium as gym
import numpy as np

from . import eval

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy


class CustomEvalCallback(EvalCallback):
    def __init__(
        self, 
        eval_env, 
        callback_on_new_best = None, 
        callback_after_eval = None, 
        n_eval_episodes = 5, 
        eval_freq = 10000, 
        log_path = None, 
        best_model_save_path = None, 
        deterministic = True, 
        render = False, 
        verbose = 1, 
        warn = True
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.n_tasks = len(eval_env.task_names)


    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            #print(len(episode_rewards), episode_rewards)
            #print(len(episode_lengths), episode_lengths)
            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval_reward/mean_reward", float(mean_reward))
            self.logger.record("eval_reward/std_reward", float(std_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                print(self._is_success_buffer)
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval_success_rate/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class CustomEvalCallback_CL(EvalCallback):
    def __init__(
            self, 
            eval_env, 
            callback_on_new_best = None, 
            callback_after_eval = None, 
            n_eval_episodes = 5, 
            eval_freq = 10000, 
            log_path = None, 
            best_model_save_path = None, 
            deterministic = True, 
            render = False, 
            verbose = 1, 
            warn = True,
            max_episode_steps = 200,
            goal_success_rates = None,
            max_elements_rates = 10,
            env = None,
            dynamicP = False,
            updateP_eval_calls = 10,
            save_best_mean_success = False
        ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.n_tasks = len(eval_env.task_names)
        self.task_names = eval_env.task_names
        self.max_episode_steps = max_episode_steps
        self.goal_success_rates = goal_success_rates
        self.max_elements = max_elements_rates
        self.evaluations_successes = list()
        self.env = env
        self.dynamicP = dynamicP
        self.updateP_eval_calls = updateP_eval_calls
        self.eval_calls = 0
        self.rates = list()
        self.best_mean_success = -1
        self.save_best_mean_success = save_best_mean_success


    def _on_step(self):
        # Set continue_training to false if training should be aborted early
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.eval_calls == 0 and self.dynamicP:
                ps = [1/self.n_tasks]*self.n_tasks
                for i, p in enumerate(ps):
                    print(f"Probability of choosing {self.task_names[i]}: {p}")
                    self.logger.record(f"eval_probability/probability_{self.task_names[i]}", p)
            self.eval_calls += 1 # Update eval call, so that I can update P every self.updateP_eval_calls time
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            success_rates, episode_rewards = eval.evaluate(
                self.model,
                self.eval_env,
                n_envs = self.n_tasks,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                verbose=0,
            )
            episode_lengths = [np.int32(self.max_episode_steps)]*self.n_eval_episodes
            if self.log_path is not None:
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(success_rates) > 0:
                    self.evaluations_successes.append(success_rates)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )
            mean_reward_task = np.mean(episode_rewards, axis=0)
            std_reward_task = np.std(episode_rewards, axis=0)
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval_reward/mean_reward", float(mean_reward))
            self.logger.record("eval_reward/std_reward", float(std_reward))
            for i, mr in enumerate(mean_reward_task):
                self.logger.record(f"eval_reward/mean_reward_{self.task_names[i]}", mr)
            for i, sr in enumerate(std_reward_task):
                self.logger.record(f"eval_reward/std_reward_{self.task_names[i]}", sr)

            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(success_rates) > 0:
                success_rate = np.mean(success_rates)
                self.rates.append(success_rates)
                if len(self.rates) > self.max_elements:
                    del self.rates[0]
                if self.verbose >= 1:
                    print(f"Success rate: {success_rates}, {100 * success_rate:.2f}%")
                self.logger.record("eval_success_rate/success_rate", success_rate)
                for i, s_r in enumerate(success_rates):
                    print(self.task_names[i], s_r)
                    self.logger.record(f"eval_success_rate/success_rate_{self.task_names[i]}", s_r)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model_reward"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if success_rate >= self.best_mean_success and self.save_best_mean_success:
                if self.verbose >= 1: 
                    print("New best mean success rate!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model_success_rate"))
                self.best_mean_success = float(success_rate)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
            
            
            # Curriculum Learning (Update the probability of the task)
            if self.dynamicP and self.eval_calls % self.updateP_eval_calls == 0:
                msr = np.array(self.rates).mean(axis=0)
                inv_msr = 1 - msr
                ps = inv_msr / np.sum(inv_msr, dtype=float)
                #ps = 2*ps + [1/self.n_tasks]*self.n_tasks
                #ps = ps/np.sum(ps, dtype=float)
                print(f"Mean success rate {msr}")
                self.env.env_method("set_p", ps)
                for i, p in enumerate(ps):
                    print(f"Probability of choosing {self.task_names[i]}: {p}")
                    self.logger.record(f"eval_probability/probability_{self.task_names[i]}", p)

            # Check if success rate is good enough (Is not used at the moment, because I update the probability instead of aborting and starting a new env)
            if self.goal_success_rates is not None and np.all(success_rates >= self.goal_success_rates):
                print(f"Abort training after {self.num_timesteps} timesteps")
                continue_training = False

        return continue_training
