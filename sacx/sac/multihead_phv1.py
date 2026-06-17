import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from gymnasium import spaces
from typing import Any, ClassVar, TypeVar, Union

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp, BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

# Only one TypeVar needed for the algorithm's learn method
SelfSACSB = TypeVar("SelfSACSB", bound="SAC_MultiheadSB")
SelfSACMB = TypeVar("SelfSACMB", bound="SAC_MultiheadMB")


# ------------------------------------ SINGLE-BRANCHING ------------------------------------
class MultiHeadCriticSB(BaseModel):
    features_extractor: BaseFeaturesExtractor 

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch_shared,    
        net_arch_head,      
        features_extractor,
        features_dim,
        num_tasks, 
        n_critics=2,
        activation_fn=nn.ReLU,
        share_features_extractor=True, 
        **kwargs
    ):
        super().__init__(observation_space, action_space, features_extractor=features_extractor, **kwargs)
        
        self.share_features_extractor = share_features_extractor
        self.num_tasks = num_tasks
        self.n_critics = n_critics
        action_dim = get_action_dim(self.action_space)
        
        self.backbones = nn.ModuleList()
        self.heads = nn.ModuleList()
        
        for _ in range(self.n_critics):
            # Trunk: Shared MLP
            trunk = create_mlp(features_dim + action_dim, net_arch_shared[-1], net_arch_shared, activation_fn)
            self.backbones.append(nn.Sequential(*trunk))
            
            # Heads: 'num_tasks' specialized Q-outputs
            task_heads = nn.ModuleList([
                nn.Sequential(*create_mlp(net_arch_shared[-1], 1, net_arch_head, activation_fn))
                for _ in range(self.num_tasks)
            ])
            self.heads.append(task_heads)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        
        task_ids = obs[:, -self.num_tasks:].argmax(dim=1)
        q_input = th.cat([features, actions], dim=1)
        
        # print("Single branching")

        final_q_values = []
        for i in range(self.n_critics):
            bb_output = self.backbones[i](q_input)
            
            # Parallel compute all heads
            all_outputs = th.cat([head(bb_output) for head in self.heads[i]], dim=1)
            
            # Select the Q-value corresponding to the active task_id
            q_selected = all_outputs.gather(1, task_ids.unsqueeze(-1))
            final_q_values.append(q_selected)
            
        return tuple(final_q_values)
    
class MultiHeadPolicySB(SACPolicy):

    critic: MultiHeadCriticSB
    critic_target: MultiHeadCriticSB

    def __init__(self, *args, **kwargs):
        # Pop them so they are NOT passed to the parent SACPolicy, which would cause a TypeError
        self.num_tasks = kwargs.pop("num_tasks", 10) 
        self.net_arch_head = kwargs.pop("net_arch_head", [512, 256])
        
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None) -> MultiHeadCriticSB:
        # SACPolicy splits the main net_arch into actor/critic parts internally.
        # Retrieve the shared backbone arch (qf) from self.critic_kwargs
        net_arch_shared = self.critic_kwargs.get("net_arch", [512,1024, 1024, 512])

        features_extractor = features_extractor or self.make_features_extractor()
        features_dim = features_extractor.features_dim
        
        return MultiHeadCriticSB(
            self.observation_space,
            self.action_space,
            net_arch_shared=net_arch_shared,
            net_arch_head=self.net_arch_head,
            features_extractor=features_extractor or self.make_features_extractor(),
            features_dim=features_dim,
            num_tasks=self.num_tasks, 
            share_features_extractor=self.share_features_extractor
        ).to(self.device)
    
class SAC_MultiheadSB(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) with Multi-Head Critic Support
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "SACPolicy_Multihead": MultiHeadPolicySB,
    }
    policy: MultiHeadPolicySB
    actor: Actor
    critic: MultiHeadCriticSB
    critic_target: MultiHeadCriticSB

    def __init__(
        self,
        policy: str | type[MultiHeadPolicySB],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: th.optim.Adam | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSACSB,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC_Multihead",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSACSB:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

# ------------------------------------ MULTI-BRANCHING ------------------------------------
class MultiHeadCriticMB(BaseModel):
    features_extractor: BaseFeaturesExtractor 

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch_shared,    
        net_arch_head,      
        features_extractor,
        features_dim,
        num_tasks,  
        n_critics=2,
        activation_fn=nn.ReLU,
        share_features_extractor=True, 
        **kwargs
    ):
        super().__init__(observation_space, action_space, features_extractor=features_extractor, **kwargs)
        
        self.share_features_extractor = share_features_extractor
        self.num_tasks = num_tasks
        self.n_critics = n_critics
        action_dim = get_action_dim(self.action_space)
        
        self.backbones = nn.ModuleList()
        self.heads = nn.ModuleList()
        
        for _ in range(self.n_critics):
            # Trunk: Shared MLP
            trunk = create_mlp(features_dim + action_dim, net_arch_shared[-1], net_arch_shared, activation_fn)
            self.backbones.append(nn.Sequential(*trunk))
            
            # Heads: 'num_tasks' specialized Q-outputs
            task_heads = nn.ModuleList([
                nn.Sequential(*create_mlp(net_arch_shared[-1], 1, net_arch_head, activation_fn))
                for _ in range(self.num_tasks)
            ])
            self.heads.append(task_heads)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        
        task_ids = obs[:, -self.num_tasks:].argmax(dim=1)
        q_input = th.cat([features, actions], dim=1)

        # print("Multi branching")
        
        final_q_values = []
        for i in range(self.n_critics):
            
            bb_output = self.backbones[i](q_input)
            
            q_values = th.zeros((q_input.shape[0], 1), device=q_input.device)

            for task_idx in range(self.num_tasks):
                mask = (task_ids == task_idx)
                
                if mask.any():
                    head_input = bb_output[mask]
                    
                    head_output = self.heads[i][task_idx](head_input)
                    
                    q_values[mask] = head_output
            
            final_q_values.append(q_values)
            
        return tuple(final_q_values)

class MultiHeadPolicyMB(SACPolicy):

    critic: MultiHeadCriticMB
    critic_target: MultiHeadCriticMB

    def __init__(self, *args, **kwargs):
        # Pop them so they are NOT passed to the parent SACPolicy, which would cause a TypeError
        self.num_tasks = kwargs.pop("num_tasks", 10) 
        self.net_arch_head = kwargs.pop("net_arch_head", [512, 256])
        
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None) -> MultiHeadCriticMB:
        # SACPolicy splits the main net_arch into actor/critic parts internally.
        # Retrieve the shared backbone arch (qf) from self.critic_kwargs
        net_arch_shared = self.critic_kwargs.get("net_arch", [512,1024, 1024, 512])

        features_extractor = features_extractor or self.make_features_extractor()
        features_dim = features_extractor.features_dim
        
        return MultiHeadCriticMB(
            self.observation_space,
            self.action_space,
            net_arch_shared=net_arch_shared,
            net_arch_head=self.net_arch_head,
            features_extractor=features_extractor or self.make_features_extractor(),
            features_dim=features_dim,
            num_tasks=self.num_tasks, 
            share_features_extractor=self.share_features_extractor
        ).to(self.device)
    
class SAC_MultiheadMB(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) with Multi-Head Critic Support
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "SACPolicy_Multihead": MultiHeadPolicyMB,
    }
    policy: MultiHeadPolicyMB
    actor: Actor
    critic: MultiHeadCriticMB
    critic_target: MultiHeadCriticMB

    def __init__(
        self,
        policy: str | type[MultiHeadPolicyMB],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: th.optim.Adam | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSACMB,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC_Multihead",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSACMB:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
    
# ------------------------------------ MULTIHEAD ACTOR ------------------------------------
class MultiHeadActor(Actor):
    def __init__(self, *args, num_tasks=1, net_arch_head=None, **kwargs):
        self.num_tasks = num_tasks
        self.net_arch_head = net_arch_head if net_arch_head is not None else []
        
        super().__init__(*args, **kwargs)

        # I hard coded the clamping interval to the global constants used by the parent class for now.
        self.log_std_min = -20
        self.log_std_max = 2

        input_dim = self.mu.in_features
        action_dim = get_action_dim(self.action_space)

        del self.mu
        del self.log_std

        self.mu_heads = nn.ModuleList()
        self.log_std_heads = nn.ModuleList()

        for _ in range(self.num_tasks):
            mu_layers = create_mlp(input_dim, action_dim, self.net_arch_head)
            log_std_layers = create_mlp(input_dim, action_dim, self.net_arch_head)
            
            self.mu_heads.append(nn.Sequential(*mu_layers))
            self.log_std_heads.append(nn.Sequential(*log_std_layers))

    def get_action_dist_params(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Overriding the generation of mean and log_std to use Multi-Branching
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        
        task_ids = obs[:, -self.num_tasks:].argmax(dim=1)
        
        batch_size = latent_pi.shape[0]
        action_dim = get_action_dim(self.action_space)
        
        mean_actions = th.zeros((batch_size, action_dim), device=latent_pi.device)
        log_std = th.zeros((batch_size, action_dim), device=latent_pi.device)
        
        unique_tasks = th.unique(task_ids)
        
        for task_idx in unique_tasks:
            task_idx = task_idx.item()
            mask = (task_ids == task_idx)
            
            head_input = latent_pi[mask]
            
            mu_out = self.mu_heads[task_idx](head_input)
            log_std_out = self.log_std_heads[task_idx](head_input)
            
            mean_actions.index_put_((mask,), mu_out)
            log_std.index_put_((mask,), log_std_out)

        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean_actions, log_std, {}    


class MultiHeadPolicy(SACPolicy):

    critic: MultiHeadCriticMB
    critic_target: MultiHeadCriticMB
    actor: MultiHeadActor

    def __init__(self, *args, **kwargs):
        # Extract custom parameters so they aren't passed to SACPolicy (which would crash)
        self.num_tasks = kwargs.pop("num_tasks", 10) 
        self.net_arch_head = kwargs.pop("net_arch_head", [512, 256])
        
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None) -> MultiHeadCriticMB:
        # SACPolicy splits the main net_arch into actor/critic parts internally.
        # We retrieve the shared backbone arch (qf) from self.critic_kwargs
        net_arch_shared = self.critic_kwargs.get("net_arch", [512,1024, 1024, 512])

        features_extractor = features_extractor or self.make_features_extractor()
        features_dim = features_extractor.features_dim
        
        return MultiHeadCriticMB(
            self.observation_space,
            self.action_space,
            net_arch_shared=net_arch_shared,
            net_arch_head=self.net_arch_head,
            features_extractor=features_extractor or self.make_features_extractor(),
            features_dim=features_dim,
            num_tasks=self.num_tasks, 
            share_features_extractor=self.share_features_extractor
        ).to(self.device)
    
    def make_actor(self, features_extractor=None) -> MultiHeadActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        
        actor_kwargs["num_tasks"] = self.num_tasks
        actor_kwargs["net_arch_head"] = self.net_arch_head
        
        return MultiHeadActor(**actor_kwargs).to(self.device)
    
class SAC_Multihead(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) with Multi-Head Critic Support
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "SACPolicy_Multihead": MultiHeadPolicy,
    }
    policy: MultiHeadPolicy
    actor: MultiHeadActor
    critic: MultiHeadCriticMB
    critic_target: MultiHeadCriticMB

    def __init__(
        self,
        policy: str | type[MultiHeadPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: th.optim.Adam | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSACMB,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC_Multihead",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSACMB:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


