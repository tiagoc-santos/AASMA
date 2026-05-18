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
import ast
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from partner_agents import RandomPartner, StationaryPartner, GreedyChefAgent, SpecialistAgent, NoisyGreedyAgent
from evaluation import evaluate, evaluation_result, save_agent_gameplay

def build_training_partner_pool(noisy_epsilon=0.25):
    """Build a pool of partner TEAMS (2 agents each) for the 3-player setup.
    
    Returns:
        List of lists, where each sublist contains exactly two initialized agents.
    """
    return [
        # Team 1: The Perfect Kitchen (Optimal coordination)
        [SpecialistAgent(role='fetcher'), SpecialistAgent(role='plater')],
        
        # Team 2: Standard Greedy Coordination
        [GreedyChefAgent(), GreedyChefAgent()],
        
        # Team 3: One good chef, one clumsy chef (Forces ego to adapt)
        [GreedyChefAgent(), NoisyGreedyAgent(epsilon=noisy_epsilon)],
        
        # Team 4: Two noisy chefs (Higher difficulty for ego)
        [NoisyGreedyAgent(epsilon=noisy_epsilon), NoisyGreedyAgent(epsilon=noisy_epsilon)],
    ]


def make_eval_team(partner_type, trained_model, noisy_epsilon=0.25):
    """Create the evaluation partner team.

    Returns:
        A list containing exactly two partner agents.
    """
    if partner_type == 'ppo':
        return [trained_model, trained_model]
    if partner_type == 'random':
        return [RandomPartner(), RandomPartner()]
    if partner_type == 'stationary':
        return [StationaryPartner(), StationaryPartner()]
    if partner_type == 'greedy':
        return [GreedyChefAgent(), GreedyChefAgent()]
    if partner_type == 'specialists':
        return [SpecialistAgent(role='fetcher'), SpecialistAgent(role='plater')]
    if partner_type == 'noisy_greedy':
        return [NoisyGreedyAgent(epsilon=noisy_epsilon), NoisyGreedyAgent(epsilon=noisy_epsilon)]
    raise ValueError(f"Unsupported partner_type: {partner_type}")

def load_layout(layout_name):
    local_layout_path = os.path.join("layouts", f"{layout_name}.layout")

    if os.path.exists(local_layout_path):
        with open(local_layout_path, "r", encoding="utf-8") as f:
            layout_dict = ast.literal_eval(f.read())

        grid = layout_dict["grid"]
        del layout_dict["grid"]

        layout_dict["layout_name"] = layout_name

        layout_grid = [
            row.strip()
            for row in grid.split("\n")
            if row.strip() != ""
        ]

        return OvercookedGridworld.from_grid(
            layout_grid,
            base_layout_params=layout_dict
        )

    return OvercookedGridworld.from_layout_name(layout_name)

class OvercookedSelfPlayWrapper(gym.Env):
    """Gym wrapper exposing one Overcooked player as the learning ego agent.

    The wrapper randomizes whether ego controls player 0 or 1 each episode,
    delegates the other player to a configurable partner policy, and returns
    flattened per-player observations.
    """

    def __init__(self, layout_name="cramped_room", partner_models=None):
        super(OvercookedSelfPlayWrapper, self).__init__()
        self.mdp = load_layout(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.num_actions = len(Action.ALL_ACTIONS)
        
        self.num_players = self.mdp.num_players
        
        self.action_space = spaces.Discrete(self.num_actions)
        self.base_env.reset()

        obs_shape = self.mdp.get_lossless_state_encoding_shape()
        flat_obs_size = int(np.prod(obs_shape))
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(flat_obs_size,),
            dtype=np.float32
        )
        
        self.partner_models = partner_models or [None] * self.num_players
        self.partner_pool = []
        self.ego_idx = 0

        self.current_obs = None
        self.deterministic_partner = False

    def set_deterministic_partner(self, is_deterministic):
        self.deterministic_partner = is_deterministic
    
    def set_partner_models(self, models):
        self.partner_models = models

    def set_partner_pool(self, pool):
        self.partner_pool = pool
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset()

        self.ego_idx = np.random.choice(self.num_players)

        if self.partner_pool:
            self.partner_models = [None] * self.num_players
            
            selected_team = random.choice(self.partner_pool)
            team_member_idx = 0

            for i in range(self.num_players):
                if i != self.ego_idx:
                    self.partner_models[i] = selected_team[team_member_idx]
                    team_member_idx += 1

        self.current_obs = self.make_simple_obs(self.ego_idx)

        return self.current_obs, {}

    def step(self, action):
        ego_action_str = Action.INDEX_TO_ACTION[int(action)]

        partner_indices = [
            i for i in range(self.num_players)
            if i != self.ego_idx
        ]

        joint_action = [Action.STAY] * self.num_players
        joint_action[self.ego_idx] = ego_action_str

        for partner_idx in partner_indices:
            partner_model = self.partner_models[partner_idx]

            if partner_model is not None:
                partner_obs = self.make_simple_obs(partner_idx)

                predict_kwargs = dict(deterministic=self.deterministic_partner)

                if getattr(partner_model, 'needs_state', False):
                    predict_kwargs.update(
                        state=self.base_env.state,
                        player_idx=partner_idx,
                        mdp=self.mdp,
                    )

                partner_action_idx, _ = partner_model.predict(
                    partner_obs,
                    **predict_kwargs
                )

                partner_action_str = Action.INDEX_TO_ACTION[int(partner_action_idx)]
            else:
                partner_action_str = Action.STAY

            joint_action[partner_idx] = partner_action_str

        joint_action = tuple(joint_action)

        joint_agent_action_info = [{} for _ in range(self.num_players)]

        next_state, sparse_reward, done, info = self.base_env.step(joint_action, joint_agent_action_info)

        self.current_obs = self.make_simple_obs()

        step_dense_rewards = info.get(
            'shaped_r_by_agent',
            [0.0] * self.num_players
        )

        total_reward = sparse_reward + sum(step_dense_rewards)

        info["joint_action"] = joint_action
        info["ego_idx"] = self.ego_idx
        info["partner_indices"] = partner_indices

        ego_obs = self.current_obs

        return ego_obs, total_reward, done, False, info
    
    def make_simple_obs(self, controlled_idx=None):
        if controlled_idx is None:
            controlled_idx = self.ego_idx
            
        obs_tuple = self.mdp.lossless_state_encoding(
            self.base_env.state, 
            horizon=self.base_env.horizon
        )
        player_grid = obs_tuple[controlled_idx]
        
        return player_grid.flatten().astype(np.float32)
 
def make_env(layout_name, rank, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        env = OvercookedSelfPlayWrapper(layout_name=layout_name)
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init 
    
def train_baseline(total_timesteps=2000000, zip_filename="overcooked_baseline",
                   train_partner_mode="curriculum", train_noisy_epsilon=0.25,
                   layout_name="cramped_room"):
    
    num_cpu = 8
    raw_env = SubprocVecEnv([make_env(layout_name, i) for i in range(num_cpu)])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=2048 // num_cpu,
        batch_size=64,
        ent_coef=0.01, 
        verbose=1,
        device="cpu" 
    )
    
    iterations = 10
    timesteps_per_iteration = total_timesteps // iterations
    training_partner_pool = build_training_partner_pool(noisy_epsilon=train_noisy_epsilon)

    if train_partner_mode == "random_pool":
        env.env_method("set_partner_pool", training_partner_pool)
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")

        if train_partner_mode == "random_pool":
            pass
        else:
            num_players = env.get_attr("num_players")[0]
            
            if i == 0:
                env.env_method("set_partner_models", [RandomPartner()] * num_players)
                env.env_method("set_deterministic_partner", False)
            else:
                model.save("temp_partner_model")
                partner_model = PPO.load("temp_partner_model")
                env.env_method("set_partner_models", [partner_model] * num_players)

        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
        
    model.save(zip_filename)
    
    eval_env = OvercookedSelfPlayWrapper(layout_name=layout_name)
    loaded_partner = PPO.load(zip_filename)
    eval_env.set_partner_models([loaded_partner] * eval_env.num_players)
    
    return model, eval_env

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=2000000, 
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
                        choices=['ppo', 'random', 'stationary', 'greedy', 'specialists', 'noisy_greedy'],
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

    eval_team = make_eval_team(
        args.eval_partner,
        trained_model,
        noisy_epsilon=args.eval_partner_epsilon
    )

    env.set_partner_pool([eval_team])

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