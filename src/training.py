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
from pathlib import Path
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
from collections import Counter
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def build_training_partner_pool(num_partners, noisy_epsilon=0.25):
    """Build a pool of partner teams dynamically scaled to the required number of partners."""
    return [
        [SpecialistAgent(role='fetcher') for _ in range(num_partners)],
        [SpecialistAgent(role='plater') for _ in range(num_partners)],
        [GreedyChefAgent() for _ in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else NoisyGreedyAgent(epsilon=noisy_epsilon) for i in range(num_partners)],
        [NoisyGreedyAgent(epsilon=noisy_epsilon) for _ in range(num_partners)],
        [StationaryPartner() for _ in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else StationaryPartner() for i in range(num_partners)],
        [RandomPartner() for _ in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else RandomPartner() for i in range(num_partners)],
        [SpecialistAgent(role='fetcher') if i % 2 == 0 else RandomPartner() for i in range(num_partners)],
    ]


def make_eval_team(partner_type, num_partners, trained_model, noisy_epsilon=0.25):
    """Create the evaluation partner team scaled dynamically."""
    if partner_type == 'ppo':
        return [trained_model for _ in range(num_partners)]
    if partner_type == 'random':
        return [RandomPartner() for _ in range(num_partners)]
    if partner_type == 'stationary':
        return [StationaryPartner() for _ in range(num_partners)]
    if partner_type == 'greedy':
        return [GreedyChefAgent() for _ in range(num_partners)]
    if partner_type == 'specialists':
        return [SpecialistAgent(role='fetcher') if i % 2 == 0 else SpecialistAgent(role='plater') for i in range(num_partners)]
    if partner_type == 'noisy_greedy':
        return [NoisyGreedyAgent(epsilon=noisy_epsilon) for _ in range(num_partners)]
    raise ValueError(f"Unsupported partner_type: {partner_type}")

def load_layout(layout_name):
    local_layout_path = os.path.join("../layouts", f"{layout_name}.layout")

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

class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class OvercookedSelfPlayWrapper(gym.Env):
    """Gym wrapper exposing one Overcooked player as the learning ego agent.

    The wrapper randomizes whether ego controls player 0 or 1 each episode,
    delegates the other player to a configurable partner policy, and returns
    flattened per-player observations.
    """

    def __init__(self, layout_name="cramped_room", partner_models=None, architecture='mlp'):
        super(OvercookedSelfPlayWrapper, self).__init__()
        self.architecture = architecture
        self.mdp = load_layout(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.num_actions = len(Action.ALL_ACTIONS)
        
        self.num_players = self.mdp.num_players
        
        self.action_space = spaces.Discrete(self.num_actions)
        self.base_env.reset()

        self.lossless_channels = 17 + self.num_players * 5

        if architecture == 'mlp':
            obs_size = (self.lossless_channels * self.mdp.width * self.mdp.height)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
            
        elif architecture == 'cnn':
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(self.lossless_channels, self.mdp.width, self.mdp.height),
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

        step_sparse_rewards = info.get(
            "sparse_r_by_agent",
            [0.0] * self.num_players
        )

        ego_sparse_reward = step_sparse_rewards[self.ego_idx]
        ego_dense_reward = step_dense_rewards[self.ego_idx]

        total_reward = ego_sparse_reward + ego_dense_reward

        info["joint_action"] = joint_action
        info["ego_idx"] = self.ego_idx
        info["partner_indices"] = partner_indices

        ego_obs = self.current_obs

        timestep = self.base_env.state.timestep
        terminated = bool(done) and timestep < self.base_env.horizon
        truncated = bool(done) and timestep >= self.base_env.horizon

        return ego_obs, total_reward, terminated, truncated, info
    
    def make_simple_obs(self, controlled_idx=None):
        if controlled_idx is None:
            controlled_idx = self.ego_idx

        player_grid = self.lossless_state_encoding_3p(
            self.base_env.state,
            controlled_idx,
            horizon=self.base_env.horizon
        )
        
        cnn_obs = np.transpose(player_grid, (2, 0, 1)).astype(np.float32)
        
        if self.architecture == 'cnn':
            return cnn_obs
        
        elif self.architecture == 'mlp':
            return cnn_obs.flatten()
    
    def lossless_state_encoding_3p(self, overcooked_state, primary_agent_idx, horizon=400):
        """Lossless-style grid encoding that supports 2+ players."""

        base_map_features = [
            "pot_loc",
            "counter_loc",
            "onion_disp_loc",
            "tomato_disp_loc",
            "dish_disp_loc",
            "serve_loc",
        ]

        variable_map_features = [
            "onions_in_pot",
            "tomatoes_in_pot",
            "onions_in_soup",
            "tomatoes_in_soup",
            "soup_cook_time_remaining",
            "soup_done",
            "dishes",
            "onions",
            "tomatoes",
        ]

        urgency_features = ["urgency"]

        ordered_player_indices = [primary_agent_idx] + [
            i for i in range(self.num_players)
            if i != primary_agent_idx
        ]

        ordered_player_features = [
            f"player_{i}_loc"
            for i in ordered_player_indices
        ] + [
            f"player_{i}_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
            for i in ordered_player_indices
            for d in Direction.ALL_DIRECTIONS
        ]

        layers = (
            ordered_player_features
            + base_map_features
            + variable_map_features
            + urgency_features
            + ["agent_id"]
        )

        state_mask_dict = {
            layer_name: np.zeros(self.mdp.shape, dtype=np.float32)
            for layer_name in layers
        }

        def make_layer(position, value):
            layer = np.zeros(self.mdp.shape, dtype=np.float32)
            layer[position] = value
            return layer

        # Urgency near the end of the episode
        if horizon - overcooked_state.timestep < 40:
            state_mask_dict["urgency"] = np.ones(self.mdp.shape, dtype=np.float32)

        # Map Layers
        for loc in self.mdp.get_counter_locations():
            state_mask_dict["counter_loc"][loc] = 1.0

        for loc in self.mdp.get_pot_locations():
            state_mask_dict["pot_loc"][loc] = 1.0

        for loc in self.mdp.get_onion_dispenser_locations():
            state_mask_dict["onion_disp_loc"][loc] = 1.0

        for loc in self.mdp.get_tomato_dispenser_locations():
            state_mask_dict["tomato_disp_loc"][loc] = 1.0

        for loc in self.mdp.get_dish_dispenser_locations():
            state_mask_dict["dish_disp_loc"][loc] = 1.0

        for loc in self.mdp.get_serving_locations():
            state_mask_dict["serve_loc"][loc] = 1.0

        # Player Layers
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]

            state_mask_dict[f"player_{i}_loc"] = make_layer(
                player.position,
                1.0
            )

            state_mask_dict[f"player_{i}_orientation_{orientation_idx}"] = make_layer(
                player.position,
                1.0
            )

        # Object and pots layers
        for obj in overcooked_state.all_objects_list:
            if obj.name == "soup":
                ingredients_count = Counter(obj.ingredients)

                num_onions = ingredients_count["onion"]
                num_tomatoes = ingredients_count["tomato"]

                if obj.position in self.mdp.get_pot_locations():
                    if obj.is_idle:
                        state_mask_dict["onions_in_pot"] += make_layer(
                            obj.position,
                            num_onions
                        )
                        state_mask_dict["tomatoes_in_pot"] += make_layer(
                            obj.position,
                            num_tomatoes
                        )
                    else:
                        state_mask_dict["onions_in_soup"] += make_layer(
                            obj.position,
                            num_onions
                        )
                        state_mask_dict["tomatoes_in_soup"] += make_layer(
                            obj.position,
                            num_tomatoes
                        )
                        state_mask_dict["soup_cook_time_remaining"] += make_layer(
                            obj.position,
                            obj.cook_time - obj._cooking_tick
                        )

                        if obj.is_ready:
                            state_mask_dict["soup_done"] += make_layer(
                                obj.position,
                                1.0
                            )
                else:
                    state_mask_dict["onions_in_soup"] += make_layer(
                        obj.position,
                        num_onions
                    )
                    state_mask_dict["tomatoes_in_soup"] += make_layer(
                        obj.position,
                        num_tomatoes
                    )
                    state_mask_dict["soup_done"] += make_layer(
                        obj.position,
                        1.0
                    )

            elif obj.name == "dish":
                state_mask_dict["dishes"] += make_layer(obj.position, 1.0)

            elif obj.name == "onion":
                state_mask_dict["onions"] += make_layer(obj.position, 1.0)

            elif obj.name == "tomato":
                state_mask_dict["tomatoes"] += make_layer(obj.position, 1.0)

        state_mask_dict["agent_id"] = np.full(
            self.mdp.shape, 
            float(primary_agent_idx), 
            dtype=np.float32
        )
        
        state_mask_stack = np.array(
            [state_mask_dict[layer_name] for layer_name in layers],
            dtype=np.float32
        )

        state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))

        return state_mask_stack
 
def make_env(layout_name, rank, architecture, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        env = OvercookedSelfPlayWrapper(layout_name=layout_name, architecture=architecture)
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init 
    
def train_baseline(total_timesteps=2000000,
                   train_partner_mode="curriculum", train_noisy_epsilon=0.25,
                   layout_name="cramped_room", num_cpu=4, architecture='mlp'):
    
    raw_env = SubprocVecEnv([make_env(layout_name, i, architecture=architecture) for i in range(num_cpu)])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    if architecture == 'mlp':
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=3e-4,
            n_steps=2048 // num_cpu,
            batch_size=64,
            ent_coef=0.01, 
            verbose=1,
            device="cpu",
        )
    elif architecture == 'cnn':
        model = PPO(
            "CnnPolicy", 
            env, 
            learning_rate=3e-4,
            n_steps=2048 // num_cpu,
            batch_size=64,
            ent_coef=0.01, 
            verbose=1,
            device="auto",
            policy_kwargs=dict(
                features_extractor_class=SmallGridCNN,
                features_extractor_kwargs=dict(features_dim=128),
                normalize_images=False,
            )
        )
    
    iterations = 10
    timesteps_per_iteration = total_timesteps // iterations
    num_partners = env.get_attr("num_players")[0] - 1
    training_partner_pool = build_training_partner_pool(num_partners, noisy_epsilon=train_noisy_epsilon)

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
    
    output_dir = "../models/"  
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_filename = f"{architecture}_{train_partner_mode}_{total_timesteps}.zip"
    output_path = os.path.join(output_dir, output_filename)
    model.save(output_path)
    
    eval_env = OvercookedSelfPlayWrapper(layout_name=layout_name, architecture=architecture)
    loaded_partner = PPO.load(output_path)
    eval_env.set_partner_models([loaded_partner] * eval_env.num_players)
    env.close()
    
    return model, eval_env, output_filename

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=2000000, 
                        help='The number of time steps per iteration')
    parser.add_argument('--model', type=str, default=None, 
                        help='The filename of an already trained model')
    parser.add_argument('--layout_name', type=str, default='three_chefs',
                        help='Overcooked layout/room to train/evaluate on')
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
    parser.add_argument('--deterministic_partner', type=str, default='false',
                        choices=['true', 'false'],
                        help='Whether partner uses deterministic actions during eval/rendering')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='Number of episodes used during evaluation')
    parser.add_argument('--results_csv', type=str, default='evaluation_results.csv',
                        help='CSV file where evaluation summaries are stored')
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='Number of cpu cores used during training')
    parser.add_argument('--architecture', type=str, default='mlp',
                        help='Type of architecture to be used',
                        choices=['mlp', 'cnn'])
    
    args = parser.parse_args()
    deterministic_partner = args.deterministic_partner.lower() == 'true'
    seed = 42
    np.random.seed(seed)

    model_stem_parts = None

    if args.model is None:
        trained_model, env, model_filename = train_baseline(
            args.timesteps,
            train_partner_mode=args.train_partner_mode,
            train_noisy_epsilon=args.train_noisy_epsilon,
            layout_name=args.layout_name,
            num_cpu=args.num_cpu,
            architecture=args.architecture
        )
    else:
        model_stem_parts = Path(args.model).stem.split("_")
        if len(model_stem_parts) >= 3:
            args.architecture = model_stem_parts[0]
        trained_model = PPO.load(args.model)
        env = OvercookedSelfPlayWrapper(layout_name=args.layout_name, architecture=args.architecture) 
        model_filename = Path(args.model).name

    eval_team = make_eval_team(
        args.eval_partner,
        env.num_players - 1,
        trained_model,
        noisy_epsilon=args.eval_partner_epsilon
    )

    env.set_partner_pool([eval_team])

    if args.model is None:
        train_mode_label = args.train_partner_mode
    else:
        if model_stem_parts is None:
            model_stem_parts = Path(args.model).stem.split("_")
        if len(model_stem_parts) >= 3:
            train_mode_label = "_".join(model_stem_parts[1:-1])
        else:
            train_mode_label = "loaded_model"

    heatmap_filename = f"{args.architecture}_{train_mode_label}_{args.eval_partner}.pdf"

    summary = evaluate(
        trained_model,
        env,
        num_episodes=args.eval_episodes,
        deterministic_partner=deterministic_partner,
        heatmap_output_file=heatmap_filename,
    )
    gif_filename = f"{args.architecture}_{train_mode_label}_{args.eval_partner}.gif"

    result_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "layout_name": args.layout_name,
        "eval_partner": args.eval_partner,
        "num_episodes": args.eval_episodes,
        "deterministic_partner": deterministic_partner,
        "architecture": args.architecture,
        "train_partner_mode": train_mode_label,
        **summary,
    }
    evaluation_result("../" + args.results_csv, result_row)

    save_agent_gameplay(trained_model, env, output_file=gif_filename, deterministic_partner=deterministic_partner)