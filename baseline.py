import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
import os
import argparse
import random
import csv
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from partner_agents import StationaryPartner, GreedyChefAgent, SpecialistAgent, NoisyGreedyAgent
from evaluation import evaluate, evaluation_result, save_agent_gameplay

class RandomPartner:
    """Partner agent that samples a random action at every step."""

    def predict(self, obs, deterministic=False):
        return np.random.randint(0, len(Action.ALL_ACTIONS)), None


def build_training_partner_pool(noisy_epsilon=0.25):
    """Build the partner pool used by random-pool training.

    Args:
        noisy_epsilon: Exploration rate for ``NoisyGreedyAgent``.

    Returns:
        List of initialized partner agents with diverse behaviors.
    """
    return [
        RandomPartner(),
        StationaryPartner(),
        GreedyChefAgent(),
        SpecialistAgent(role='fetcher'),
        SpecialistAgent(role='plater'),
        NoisyGreedyAgent(epsilon=noisy_epsilon),
    ]


def make_eval_partner(partner_type, trained_model, noisy_epsilon=0.25):
    """Create the evaluation partner.

    Args:
        partner_type: Partner key from supported choices.
        trained_model: PPO model to use when ``partner_type`` is ``'ppo'``.
        noisy_epsilon: Exploration rate for ``'noisy_greedy'``.

    Returns:
        Partner policy object exposing ``predict``.

    Raises:
        ValueError: If ``partner_type`` is unknown.
    """
    if partner_type == 'ppo':
        return trained_model
    if partner_type == 'random':
        return RandomPartner()
    if partner_type == 'stationary':
        return StationaryPartner()
    if partner_type == 'greedy':
        return GreedyChefAgent()
    if partner_type == 'fetcher':
        return SpecialistAgent(role='fetcher')
    if partner_type == 'plater':
        return SpecialistAgent(role='plater')
    if partner_type == 'noisy_greedy':
        return NoisyGreedyAgent(epsilon=noisy_epsilon)
    raise ValueError(f"Unsupported partner_type: {partner_type}")


class OvercookedSelfPlayWrapper(gym.Env):
    """Gym wrapper exposing one Overcooked player as the learning ego agent.

    The wrapper randomizes whether ego controls player 0 or 1 each episode,
    delegates the other player to a configurable partner policy, and returns
    flattened per-player observations.
    """

    def __init__(self, layout_name="cramped_room", partner_model=None):
        super(OvercookedSelfPlayWrapper, self).__init__()
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.num_actions = len(Action.ALL_ACTIONS)
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        self.base_env.reset()
        dummy_obs = self.base_env.featurize_state_mdp(self.base_env.state)[0]
        flat_obs_size = np.prod(dummy_obs.shape)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(flat_obs_size,), dtype=np.float32)
        
        self.partner_model = partner_model
        self.partner_pool = []
        self.ego_idx = 0 

        self.current_obs_tuple = None
        self.deterministic_partner = False

    def set_deterministic_partner(self, is_deterministic):
        self.deterministic_partner = is_deterministic
    
    def set_partner_model(self, model):
        self.partner_model = model

    def set_partner_pool(self, pool):
        self.partner_pool = pool
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.base_env.reset()
        self.ego_idx = np.random.choice([0, 1])

        # Choose a random partner model from the pool if available, otherwise use the default partner_model
        if self.partner_pool:
            self.partner_model = random.choice(self.partner_pool)

        self.current_obs_tuple = self.base_env.featurize_state_mdp(self.base_env.state)
        
        return self.current_obs_tuple[self.ego_idx].flatten(), {} 

    def step(self, action):
        ego_action_str = Action.INDEX_TO_ACTION[action]
        partner_idx = 1 - self.ego_idx
        
        if self.partner_model is not None:
            partner_obs = self.current_obs_tuple[partner_idx].flatten()
            predict_kwargs = dict(deterministic=self.deterministic_partner)
            if getattr(self.partner_model, 'needs_state', False):
                predict_kwargs.update(
                    state=self.base_env.state,
                    player_idx=partner_idx,
                    mdp=self.mdp,
                )

            partner_action_idx, _ = self.partner_model.predict(
                partner_obs,
                **predict_kwargs
            )
            partner_action_str = Action.INDEX_TO_ACTION[partner_action_idx]
        else:
            partner_action_str = Action.STAY
            
        if self.ego_idx == 0:
            joint_action = (ego_action_str, partner_action_str)
        else:
            joint_action = (partner_action_str, ego_action_str)
        
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)
        
        self.current_obs_tuple = self.base_env.featurize_state_mdp(self.base_env.state)
        
        step_dense_rewards = info.get('shaped_r_by_agent', [0.0, 0.0])
        total_reward = sparse_reward + sum(step_dense_rewards)
        
        info["joint_action"] = joint_action
        info["ego_idx"] = self.ego_idx 
        
        ego_obs = self.current_obs_tuple[self.ego_idx].flatten()
        
        return ego_obs, total_reward, done, False, info
    
def train_baseline(total_timesteps = 2000000, zip_filename="overcooked_baseline",
                   train_partner_mode="curriculum", train_noisy_epsilon=0.25,
                   layout_name="cramped_room"):
    """Train a PPO baseline with curriculum or random-pool partner training.

    Args:
        total_timesteps: Total training timesteps across all iterations.
        zip_filename: Output filename used when saving the PPO model.
        train_partner_mode: ``'curriculum'`` or ``'random_pool'``.
        train_noisy_epsilon: Epsilon for noisy partner in random pool.
        layout_name: Overcooked layout to train on.

    Returns:
        Tuple ``(trained_model, eval_env)`` ready for evaluation.
    """
    raw_env = DummyVecEnv([lambda: Monitor(OvercookedSelfPlayWrapper(layout_name=layout_name))])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01, 
        verbose=1,
        device="cpu"
    )
    
    iterations = 10
    timesteps_per_iteration = total_timesteps // iterations
    training_partner_pool = build_training_partner_pool(noisy_epsilon=train_noisy_epsilon)

    if train_partner_mode == "random_pool":
        env.venv.envs[0].env.set_partner_pool(training_partner_pool)
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        if train_partner_mode == "random_pool":
            pass
        else:
            if i == 0:
                env.venv.envs[0].env.set_partner_model(RandomPartner())
                env.venv.envs[0].env.set_deterministic_partner(False)
            else:
                model.save("temp_partner_model")
                partner_model = PPO.load("temp_partner_model")
                env.venv.envs[0].env.set_partner_model(partner_model)
        
        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
        
    model.save(zip_filename)
    
    eval_env = OvercookedSelfPlayWrapper(layout_name=layout_name)
    eval_env.set_partner_model(PPO.load(zip_filename))
    return model, eval_env

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000, 
                        help='The number of time steps per iteration')
    parser.add_argument('--model', type=str, default=None, 
                        help='The filename of an already trained model')
    parser.add_argument('--layout_name', type=str, default='cramped_room',
                        help='Overcooked layout/room to train/evaluate on')
    parser.add_argument('--gif_filename', type=str, default="aasma_ego_agent.gif", 
                        help='The filename of the gif output')
    parser.add_argument('--zip_filename', type=str, default="overcooked_baseline", 
                        help='The filename of the model')
    parser.add_argument('--train_partner_mode', type=str, default='curriculum',
                        choices=['curriculum', 'random_pool'],
                        help='Training partner schedule: curriculum (default) or random_pool')
    parser.add_argument('--train_noisy_epsilon', type=float, default=0.25,
                        help='Epsilon used for NoisyGreedyAgent in training partner pool')
    parser.add_argument('--eval_partner', type=str, default='ppo',
                        choices=['ppo', 'random', 'stationary', 'greedy', 'fetcher', 'plater', 'noisy_greedy'],
                        help='Partner to use during evaluation and gameplay rendering')
    parser.add_argument('--eval_partner_epsilon', type=float, default=0.25,
                        help='Epsilon for noisy_greedy evaluation partner')
    parser.add_argument('--deterministic_partner', type=str, default='true',
                        choices=['true', 'false'],
                        help='Whether partner uses deterministic actions during eval/rendering')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='Number of episodes used during evaluation')
    parser.add_argument('--results_csv', type=str, default='evaluation_results.csv',
                        help='CSV file where evaluation summaries are stored')
    args = parser.parse_args()
    deterministic_partner = args.deterministic_partner.lower() == 'true'
    seed = 42
    np.random.seed(seed)

    if args.model is None:
        trained_model, env = train_baseline(
            args.timesteps,
            args.zip_filename,
            train_partner_mode=args.train_partner_mode,
            train_noisy_epsilon=args.train_noisy_epsilon,
            layout_name=args.layout_name,
        )
    else:
        trained_model = PPO.load(args.model)
        env = OvercookedSelfPlayWrapper(layout_name=args.layout_name) 

    env.set_partner_model(
        make_eval_partner(args.eval_partner, trained_model, noisy_epsilon=args.eval_partner_epsilon)
    )

    summary = evaluate(
        trained_model,
        env,
        num_episodes=args.eval_episodes,
        deterministic_partner=deterministic_partner,
    )

    result_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "layout_name": args.layout_name,
        "eval_partner": args.eval_partner,
        "num_episodes": args.eval_episodes,
        "deterministic_partner": deterministic_partner,
        "model_file": args.model if args.model is not None else args.zip_filename,
        "train_partner_mode": args.train_partner_mode if args.model is None else "loaded_model",
        **summary,
    }
    evaluation_result(args.results_csv, result_row)

    save_agent_gameplay(trained_model, env, output_file=args.gif_filename, deterministic_partner=deterministic_partner)