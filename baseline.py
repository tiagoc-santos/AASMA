import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
import os
import argparse
import random
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from partner_agents import StationaryPartner, GreedyChefAgent, SpecialistAgent, NoisyGreedyAgent

class RandomPartner:
    def predict(self, obs, deterministic=False):
        return np.random.randint(0, len(Action.ALL_ACTIONS)), None


def build_training_partner_pool(noisy_epsilon=0.25):
    return [
        RandomPartner(),
        StationaryPartner(),
        GreedyChefAgent(),
        SpecialistAgent(role='fetcher'),
        SpecialistAgent(role='plater'),
        NoisyGreedyAgent(epsilon=noisy_epsilon),
    ]


def make_eval_partner(partner_type, trained_model, noisy_epsilon=0.25):
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
                   train_partner_mode="curriculum", train_noisy_epsilon=0.25):
    raw_env = DummyVecEnv([lambda: Monitor(OvercookedSelfPlayWrapper(layout_name="cramped_room"))])
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
    
    eval_env = OvercookedSelfPlayWrapper(layout_name="cramped_room")
    eval_env.set_partner_model(PPO.load(zip_filename))
    return model, eval_env


def check_behavioral_events(agent_action, prev_player, curr_player, mdp):
    bumped = 0
    misplaced = 0
    
    # Bumped / Failed Movements 
    if agent_action in Direction.ALL_DIRECTIONS and prev_player.position == curr_player.position:
        bumped = 1
        
    # Misplaced Items 
    if agent_action == Action.INTERACT and prev_player.has_object() and not curr_player.has_object():
        pos_x, pos_y = prev_player.position
        dir_x, dir_y = prev_player.orientation
        facing_pos = (pos_x + dir_x, pos_y + dir_y)

        if mdp.get_terrain_type_at_pos(facing_pos) == 'X':
            misplaced = 1
            
    return bumped, misplaced

def run_single_episode(model, gym_env, episode_seed):
    np.random.seed(episode_seed)
    obs, _ = gym_env.reset()
    base_env = gym_env.base_env
    mdp = base_env.mdp
    
    done = False
    step_count = 0
    
    ep_metrics = {
        'total_score': 0,
        'dish_delivery_times': [],
        'stood_still_count': 0,
        'bump_count': 0,
        'misplaced_count': 0
    }
    heatmap_updates = []
    
    prev_state = base_env.state

    while not done:
        ego_action_idx, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
        done = terminated or truncated
        step_count += 1
        
        current_state = base_env.state
        joint_action = info.get("joint_action", (Action.STAY, Action.STAY))
        
        # Stood Still Metric
        if Action.STAY in joint_action:
            ep_metrics['stood_still_count'] += 1
            
        # Score & Delivery Tracking
        step_sparse_reward = sum(info.get("sparse_r_by_agent", [0.0, 0.0]))
        if step_sparse_reward > 0:
            ep_metrics['total_score'] += step_sparse_reward
            ep_metrics['dish_delivery_times'].append(step_count)

        # Behavior Metrics Loop
        for agent_idx in range(2):
            agent_action = joint_action[agent_idx]
            prev_player = prev_state.players[agent_idx]
            curr_player = current_state.players[agent_idx]
            
            heatmap_updates.append(curr_player.position)
            
            bumped, misplaced = check_behavioral_events(agent_action, prev_player, curr_player, mdp)
            ep_metrics['bump_count'] += bumped
            ep_metrics['misplaced_count'] += misplaced

        prev_state = current_state
        
    return ep_metrics, step_count, heatmap_updates

def print_evaluation_summary(agg_metrics):
    print("\n================================================")
    print("FINAL BASELINE METRICS (Average over episodes)")
    print("================================================")
    print(f"Avg Total Score: {np.mean(agg_metrics['scores']):.2f}")
    print(f"Avg Time to First Dish: {np.mean(agg_metrics['time_to_first']):.2f} steps")
    print(f"Avg Time Between Dishes: {np.mean(agg_metrics['avg_time_between']):.2f} steps")
    print(f"Avg Times Stood Still: {np.mean(agg_metrics['stood_still']):.2f}")
    print(f"Avg Bumps/Failed Moves: {np.mean(agg_metrics['bumps']):.2f}")
    print(f"Avg Misplaced Items: {np.mean(agg_metrics['misplaced']):.2f}")
    print("================================================")

def evaluate(model, gym_env, num_episodes=5, deterministic_partner=True):
    gym_env.set_deterministic_partner(deterministic_partner)
    mdp = gym_env.base_env.mdp
    
    agg_metrics = {
        'scores': [], 'time_to_first': [], 'avg_time_between': [],
        'stood_still': [], 'bumps': [], 'misplaced': []
    }
    
    heatmap = np.zeros((mdp.width, mdp.height))

    for episode in range(num_episodes):
        ep_metrics, final_step_count, heatmap_updates = run_single_episode(model, gym_env, episode)
        
        for x, y in heatmap_updates:
            heatmap[x][y] += 1
            
        agg_metrics['scores'].append(ep_metrics['total_score'])
        agg_metrics['stood_still'].append(ep_metrics['stood_still_count'])
        agg_metrics['bumps'].append(ep_metrics['bump_count'])
        agg_metrics['misplaced'].append(ep_metrics['misplaced_count'])
        
        deliveries = ep_metrics['dish_delivery_times']
        time_to_first = deliveries[0] if deliveries else final_step_count
        agg_metrics['time_to_first'].append(time_to_first)
        
        avg_interval = np.mean(np.diff(deliveries)) if len(deliveries) > 1 else 0
        agg_metrics['avg_time_between'].append(avg_interval)

    print_evaluation_summary(agg_metrics)
    render_heatmap(heatmap)

def render_heatmap(heatmap, output_file = "baseline_heatmap.pdf"):
    plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')
    plt.title("Agent Movement Heatmap (Most Visited Tiles)")
    plt.colorbar(label="Visits")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def save_agent_gameplay(model, gym_env, output_file="aasma_ego_agent.mp4", fps=5, deterministic_partner=True):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    visualizer = StateVisualizer()

    gym_env.set_deterministic_partner(deterministic_partner)
    
    obs, _ = gym_env.reset()
    done = False
    initial_mdp = gym_env.base_env.mdp
    
    frames = []
    
    current_state = gym_env.base_env.state
    surface = visualizer.render_state(current_state, initial_mdp.terrain_mtx)
    frame = pygame.surfarray.pixels3d(surface)
    frame_actual = np.transpose(frame, (1, 0, 2)).copy()
    frames.append(frame_actual)

    while not done:
        ego_action_idx, _ = model.predict(obs, deterministic=False)
        
        obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
        done = terminated or truncated

        current_state = gym_env.base_env.state
        surface = visualizer.render_state(current_state, initial_mdp.terrain_mtx)
        
        frame = pygame.surfarray.pixels3d(surface)
        frame_actual = np.transpose(frame, (1, 0, 2)).copy() 
        frames.append(frame_actual)

    pygame.quit()

    if output_file.endswith('.gif'):
        duration_ms = 1000 / fps
        imageio.mimsave(output_file, frames, format='GIF', duration=duration_ms, loop=0)
    else:
        imageio.mimsave(output_file, frames, fps=fps, macro_block_size=None)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000, 
                        help='The number of time steps per iteration')
    parser.add_argument('--model', type=str, default=None, 
                        help='The filename of an already trained model')
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
        )
    else:
        trained_model = PPO.load(args.model)
        env = OvercookedSelfPlayWrapper() 

    env.set_partner_model(
        make_eval_partner(args.eval_partner, trained_model, noisy_epsilon=args.eval_partner_epsilon)
    )
    evaluate(trained_model, env, num_episodes=5, deterministic_partner=deterministic_partner)
    save_agent_gameplay(trained_model, env, output_file=args.gif_filename, deterministic_partner=deterministic_partner)