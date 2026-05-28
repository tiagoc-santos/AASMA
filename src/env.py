import ast
import os
import random
from collections import Counter, deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

class OvercookedSelfPlayWrapper(gym.Env):
    """Gym wrapper exposing one Overcooked player as the learning ego agent."""

    def __init__(self, layout_name="three_chefs", partner_models=None, architecture='mlp', history_len=15):
        super(OvercookedSelfPlayWrapper, self).__init__()
        self.architecture = architecture
        self.mdp = load_layout(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.num_actions = len(Action.ALL_ACTIONS)
        
        self.num_players = self.mdp.num_players
        
        self.action_space = spaces.Discrete(self.num_actions)
        self.base_env.reset()

        self.lossless_channels = 17 + self.num_players * 5
        # Feed-forward models keep explicit history.
        # RecurrentPPO receives one current frame/action and stores history in its LSTM.
        self.history_len = 1 if architecture == "rnn" else history_len

        self.action_history_size = (self.history_len * self.num_actions * self.num_players)
        self.latest_joint_action_size = self.num_actions * self.num_players
        if architecture == "mlp":
            grid_shape = (self.lossless_channels * self.mdp.width * self.mdp.height
                * self.history_len,)
            action_shape = (self.action_history_size,)

        elif architecture == "cnn":
            grid_shape = (
                self.lossless_channels * self.history_len,
                self.mdp.width,
                self.mdp.height,)
            action_shape = (self.action_history_size,)

        elif architecture == "rnn":
            # Current spatial observation; temporal memory is maintained by the LSTM.
            grid_shape = (
                self.lossless_channels,
                self.mdp.width,
                self.mdp.height,)
            action_shape = (self.latest_joint_action_size,)

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture!r}. "
                "Expected 'cnn', 'mlp' or 'rnn'.")

        self.observation_space = spaces.Dict({
            "grid_obs": spaces.Box(
                low=0.0,
                high=np.inf,
                shape=grid_shape,
                dtype=np.float32,
            ),
            "action_history": spaces.Box(
                low=0.0,
                high=1.0,
                shape=action_shape,
                dtype=np.float32,
            ),
        })
        
        self.partner_models = partner_models or [None] * self.num_players
        self.partner_lstm_states = [None] * self.num_players
        self.partner_episode_starts = np.ones(self.num_players, dtype=bool,)
        self.partner_pool = []
        self.ego_idx = 0

        self.current_obs = None
        self.obs_history = {}
        self.last_joint_action = tuple([Action.STAY] * self.num_players)
        self.action_history = deque(maxlen=self.history_len)
        self.deterministic_partner = False
        self.dense_reward_scale = 1.0

    def set_dense_reward_scale(self, scale):
        self.dense_reward_scale = float(scale)
        
    def set_deterministic_partner(self, is_deterministic):
        self.deterministic_partner = is_deterministic
    
    def set_partner_models(self, models):
        if len(models) != self.num_players:
            raise ValueError("partner_models must contain one slot per player position. "
                f"Expected {self.num_players}, got {len(models)}.")

        self.partner_models = list(models)
        self.partner_lstm_states = [None] * self.num_players
        self.partner_episode_starts = np.ones(self.num_players, dtype=bool,)

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

        self.partner_lstm_states = [None] * self.num_players
        self.partner_episode_starts = np.ones(self.num_players, dtype=bool,)
        self.obs_history = {i: deque(maxlen=self.history_len) for i in range(self.num_players)}
        
        for i in range(self.num_players):
            initial_frame = self._get_single_frame(i)
            for _ in range(self.history_len):
                self.obs_history[i].append(initial_frame)

        self.last_joint_action = tuple([Action.STAY] * self.num_players)
        self.action_history = deque(
            [self.last_joint_action for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        self.current_obs = self.make_simple_obs(self.ego_idx)

        return self.current_obs, {}

    def step(self, action):
        ego_action_idx = int(np.asarray(action).item())
        ego_action_str = Action.INDEX_TO_ACTION[ego_action_idx]

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
                if isinstance(partner_model, RecurrentPPO):
                    partner_action_idx, next_lstm_state = partner_model.predict(
                        partner_obs,
                        state=self.partner_lstm_states[partner_idx],
                        episode_start=np.array(
                            [self.partner_episode_starts[partner_idx]],
                            dtype=bool,),
                        deterministic=self.deterministic_partner,)
                    self.partner_lstm_states[partner_idx] = next_lstm_state
                    self.partner_episode_starts[partner_idx] = False

                else:
                    predict_kwargs = dict(deterministic=self.deterministic_partner)

                    if getattr(partner_model, "needs_state", False):
                        predict_kwargs.update(
                            state=self.base_env.state,
                            player_idx=partner_idx,
                            mdp=self.mdp,
                        )
                    partner_action_idx, _ = partner_model.predict(
                        partner_obs,
                        **predict_kwargs,)
                partner_action_idx = int(np.asarray(partner_action_idx).item())
                partner_action_str = Action.INDEX_TO_ACTION[partner_action_idx]

            else:
                partner_action_str = Action.STAY

            joint_action[partner_idx] = partner_action_str

        joint_action = tuple(joint_action)
        self.last_joint_action = joint_action
        self.action_history.append(joint_action)

        joint_agent_action_info = [{} for _ in range(self.num_players)]

        next_state, sparse_reward, done, info = self.base_env.step(joint_action, joint_agent_action_info)

        for i in range(self.num_players):
            new_frame = self._get_single_frame(i)
            self.obs_history[i].append(new_frame)

        self.current_obs = self.make_simple_obs()

        step_dense_rewards = info.get(
            'shaped_r_by_agent',
            [0.0] * self.num_players
        )

        step_sparse_rewards = info.get(
            "sparse_r_by_agent",
            [0.0] * self.num_players
        )

        team_sparse_reward = float(sparse_reward)
        ego_dense_reward = float(step_dense_rewards[self.ego_idx])

        total_reward = team_sparse_reward + self.dense_reward_scale * ego_dense_reward

        info["joint_action"] = joint_action
        info["ego_idx"] = self.ego_idx
        info["partner_indices"] = partner_indices

        ego_obs = self.current_obs

        timestep = self.base_env.state.timestep
        terminated = bool(done) and timestep < self.base_env.horizon
        truncated = bool(done) and timestep >= self.base_env.horizon

        return ego_obs, total_reward, terminated, truncated, info
    
    def _get_single_frame(self, controlled_idx):
        player_grid = self.lossless_state_encoding_3p(
            self.base_env.state,
            controlled_idx,
            horizon=self.base_env.horizon
        )
        
        cnn_obs = np.transpose(player_grid, (2, 0, 1)).astype(np.float32)
        
        if self.architecture in {"cnn", "rnn"}:
            return cnn_obs

        elif self.architecture == "mlp":
            return cnn_obs.flatten()

        raise ValueError(f"Unsupported architecture: {self.architecture!r}")

    def make_simple_obs(self, controlled_idx=None):
        if controlled_idx is None:
            controlled_idx = self.ego_idx
            
        if self.architecture == "cnn":
            grid_obs = np.concatenate(list(self.obs_history[controlled_idx]),axis=0,)
        elif self.architecture == "mlp":
            grid_obs = np.concatenate(list(self.obs_history[controlled_idx]),axis=0,)
        elif self.architecture == "rnn":
            # Current frame only; LSTM stores temporal information.
            grid_obs = self.obs_history[controlled_idx][-1]
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture!r}")

        if self.architecture == "rnn":
            action_features = np.zeros(
                self.latest_joint_action_size,
                dtype=np.float32,
            )

            latest_joint_action = self.action_history[-1]
            for player_idx, act in enumerate(latest_joint_action):
                act_idx = (Action.ALL_ACTIONS.index(act) if act in Action.ALL_ACTIONS
                    else Action.ALL_ACTIONS.index(Action.STAY))
                offset = player_idx * self.num_actions + act_idx
                action_features[offset] = 1.0

        else:
            action_features = np.zeros(self.action_history_size, dtype=np.float32,)

            for time_idx, joint_action in enumerate(self.action_history):
                for player_idx, act in enumerate(joint_action):
                    act_idx = (Action.ALL_ACTIONS.index(act) if act in Action.ALL_ACTIONS
                        else Action.ALL_ACTIONS.index(Action.STAY))

                    offset = (time_idx * self.num_players * self.num_actions
                        + player_idx * self.num_actions
                        + act_idx)
                    action_features[offset] = 1.0

        return {"grid_obs": grid_obs, "action_history": action_features,}
    
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
            state_mask_dict[f"player_{i}_loc"] = make_layer(player.position, 1.0)
            state_mask_dict[f"player_{i}_orientation_{orientation_idx}"] = make_layer(player.position, 1.0)

        # Object and pots layers
        for obj in overcooked_state.all_objects_list:
            if obj.name == "soup":
                ingredients_count = Counter(obj.ingredients)

                num_onions = ingredients_count["onion"]
                num_tomatoes = ingredients_count["tomato"]

                if obj.position in self.mdp.get_pot_locations():
                    if obj.is_idle:
                        state_mask_dict["onions_in_pot"] += make_layer(obj.position, num_onions)
                        state_mask_dict["tomatoes_in_pot"] += make_layer(obj.position, num_tomatoes)
                    else:
                        state_mask_dict["onions_in_soup"] += make_layer(obj.position, num_onions)
                        state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, num_tomatoes)
                        state_mask_dict["soup_cook_time_remaining"] += make_layer(obj.position, obj.cook_time - obj._cooking_tick)
                        if obj.is_ready:
                            state_mask_dict["soup_done"] += make_layer(obj.position, 1.0)
                else:
                    state_mask_dict["onions_in_soup"] += make_layer(obj.position, num_onions)
                    state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, num_tomatoes)
                    state_mask_dict["soup_done"] += make_layer(obj.position, 1.0)

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

class PartnerAwareExtractor(BaseFeaturesExtractor):
    """
    Multi-input spatial/action feature extractor.

    For architecture='rnn', this remains a CNN spatial encoder.
    Temporal memory is handled by RecurrentPPO's LSTM policy.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim=256,
                 architecture="cnn",):
        super().__init__(observation_space, features_dim)

        self.architecture = architecture
        grid_space = observation_space.spaces["grid_obs"]
        action_space = observation_space.spaces["action_history"]

        if architecture in {"cnn", "rnn"}:
            n_input_channels = grid_space.shape[0]
            self.grid_net = nn.Sequential(
                nn.Conv2d(
                    n_input_channels,
                    32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    32,
                    64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                n_flatten = self.grid_net(
                    th.zeros(1, *grid_space.shape)).shape[1]

        elif architecture == "mlp":
            n_input = grid_space.shape[0]

            self.grid_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_input, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            n_flatten = 256

        else:
            raise ValueError(f"Unsupported architecture: {architecture!r}")

        action_dim = action_space.shape[0]

        self.act_net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        grid_feat = self.grid_net(observations["grid_obs"].float())
        act_feat = self.act_net(observations["action_history"].float())

        return self.linear(th.cat([grid_feat, act_feat], dim=1))

def make_env(layout_name, rank, architecture, seed=0):
    """Utility function for multiprocessed environments."""
    def _init():
        worker_seed = seed + rank
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        th.manual_seed(worker_seed)
        env = OvercookedSelfPlayWrapper(layout_name=layout_name,architecture=architecture)
        env.reset(seed=worker_seed)
        return Monitor(env)

    return _init

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