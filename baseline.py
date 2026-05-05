import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

class RandomPartner:
    def predict(self, obs, deterministic=False):
        return np.random.randint(0, len(Action.ALL_ACTIONS)), None
class OvercookedSelfPlayWrapper(gym.Env):
    def __init__(self, layout_name="forced_coordination", partner_model=None):
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
        self.ego_idx = 0 

        self.current_obs_tuple = None
        self.deterministic_partner = False

    def set_deterministic_partner(self, is_deterministic):
        self.deterministic_partner = is_deterministic
    
    def set_partner_model(self, model):
        self.partner_model = model
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.base_env.reset()
        self.ego_idx = np.random.choice([0, 1])
        self.current_obs_tuple = self.base_env.featurize_state_mdp(self.base_env.state)
        
        return self.current_obs_tuple[self.ego_idx].flatten(), {} 

    def step(self, action):
        ego_action_str = Action.INDEX_TO_ACTION[action]
        partner_idx = 1 - self.ego_idx
        
        if self.partner_model is not None:
            partner_obs = self.current_obs_tuple[partner_idx].flatten()
            partner_action_idx, _ = self.partner_model.predict(
                partner_obs, 
                deterministic=self.deterministic_partner
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
    
def train_baseline(total_timesteps = 2000000):
    raw_env = DummyVecEnv([lambda: Monitor(OvercookedSelfPlayWrapper(layout_name="forced_coordination"))])
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
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        if i == 0:
            env.venv.envs[0].env.set_partner_model(RandomPartner())
            env.venv.envs[0].env.set_deterministic_partner(False)
        else:
            model.save("temp_partner_model")
            partner_model = PPO.load("temp_partner_model")
            env.venv.envs[0].env.set_partner_model(partner_model)
        
        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
        
    model.save("overcooked_baseline")
    
    eval_env = OvercookedSelfPlayWrapper(layout_name="forced_coordination")
    eval_env.set_partner_model(PPO.load("overcooked_baseline"))
    return model, eval_env


def evaluate(model, gym_env, num_episodes=5):
    base_env = gym_env.base_env 
    gym_env.set_deterministic_partner(True)
    mdp = base_env.mdp
    
    all_ep_scores = []
    all_ep_time_to_first = []
    all_ep_avg_time_between = []
    all_ep_stood_still = []
    all_ep_bumps = []
    all_ep_misplaced = []

    heatmap = np.zeros((mdp.width, mdp.height))

    for episode in range(num_episodes):
        np.random.seed(episode)
        obs, _ = gym_env.reset()
        done = False
        step_count = 0
        
        total_score = 0
        dish_delivery_times = []
        stood_still_count = 0
        bump_count = 0
        misplaced_count = 0

        prev_state = base_env.state

        while not done:
            ego_action_idx, _ = model.predict(obs, deterministic=False)
            
            obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
            done = terminated or truncated
            step_count += 1
            current_state = base_env.state
            
            joint_action = info.get("joint_action", (Action.STAY, Action.STAY))
            a0, a1 = joint_action
            
            # Stood Still Metric
            if a0 == Action.STAY or a1 == Action.STAY:
                stood_still_count += 1
            
            step_sparse_reward = sum(info.get("sparse_r_by_agent", [0.0, 0.0]))
            
            if step_sparse_reward > 0:
                total_score += step_sparse_reward
                dish_delivery_times.append(step_count)

            # Behavior Metrics Loop
            for agent_idx in range(2):
                agent_action = joint_action[agent_idx]
                prev_player = prev_state.players[agent_idx]
                curr_player = current_state.players[agent_idx]
                
                # Most Visited Tiles 
                x, y = curr_player.position
                heatmap[x][y] += 1
                
                # Bumped / Failed Movements 
                if agent_action in Direction.ALL_DIRECTIONS:
                    if prev_player.position == curr_player.position:
                        bump_count += 1
                
                # Misplaced Items 
                if agent_action == Action.INTERACT and prev_player.has_object() and not curr_player.has_object():
                    pos_x, pos_y = prev_player.position
                    dir_x, dir_y = prev_player.orientation
                    facing_pos = (pos_x + dir_x, pos_y + dir_y)

                    terrain_type = mdp.get_terrain_type_at_pos(facing_pos)

                    if terrain_type == 'X':
                        misplaced_count += 1

            prev_state = current_state

        all_ep_scores.append(total_score)
        all_ep_stood_still.append(stood_still_count)
        all_ep_bumps.append(bump_count)
        all_ep_misplaced.append(misplaced_count)
        
        time_to_first = dish_delivery_times[0] if dish_delivery_times else step_count
        all_ep_time_to_first.append(time_to_first)
        
        avg_interval = np.mean(np.diff(dish_delivery_times)) if len(dish_delivery_times) > 1 else 0
        all_ep_avg_time_between.append(avg_interval)

    print("\n================================================")
    print("FINAL BASELINE METRICS (Average over episodes)")
    print("================================================")
    print(f"[Task] Avg Total Score: {np.mean(all_ep_scores):.2f}")
    print(f"[Task] Avg Time to First Dish: {np.mean(all_ep_time_to_first):.2f} steps")
    print(f"[Task] Avg Time Between Dishes: {np.mean(all_ep_avg_time_between):.2f} steps")
    print(f"[Behavior] Avg Times Stood Still: {np.mean(all_ep_stood_still):.2f}")
    print(f"[Behavior] Avg Bumps/Failed Moves: {np.mean(all_ep_bumps):.2f}")
    print(f"[Behavior] Avg Misplaced Items: {np.mean(all_ep_misplaced):.2f}")
    print("================================================")
    
    render_heatmap(heatmap)

def render_heatmap(heatmap, output_file = "baseline_heatmap.pdf"):
    plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')
    plt.title("Agent Movement Heatmap (Most Visited Tiles)")
    plt.colorbar(label="Visits")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def save_agent_gameplay(model, gym_env, output_file="aasma_ego_agent.mp4", fps=5):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    visualizer = StateVisualizer()

    gym_env.set_deterministic_partner(True)
    
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
        
    print(f"Done! Gameplay saved with {len(frames)} frames.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000, 
                        help='The number of time steps per iteration')
    parser.add_argument('--model', type=str, default=None, 
                        help='The filename of an already trained model')
    args = parser.parse_args()
    seed = 42
    np.random.seed(seed)

    if args.model is None:
        trained_model, env = train_baseline(args.timesteps)
    else:
        trained_model = PPO.load(args.model)
        env = OvercookedSelfPlayWrapper() 

    env.set_partner_model(trained_model)
    evaluate(trained_model, env, num_episodes=5)
    save_agent_gameplay(trained_model, env, output_file="aasma_ego_agent.gif")