import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class OvercookedSelfPlayWrapper(gym.Env):
    def __init__(self, layout_name="cramped_room", partner_model=None):
        super(OvercookedSelfPlayWrapper, self).__init__()
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.num_actions = len(Action.ALL_ACTIONS)
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        # featurize_state_mdp returns a tuple of (obs_0, obs_1)
        dummy_obs = self.base_env.featurize_state_mdp(self.base_env.state)[0]
        flat_obs_size = np.prod(dummy_obs.shape)
        self.observation_space = spaces.Box(low=0, high=1, shape=(flat_obs_size,), dtype=np.float32)
        
        self.partner_model = partner_model
        self.ego_idx = 0

    def set_partner_model(self, model):
        self.partner_model = model
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.base_env.reset()
        
        return self._get_obs(self.ego_idx), {} 

    def step(self, action):
        ego_action_str = Action.INDEX_TO_ACTION[action]
        
        partner_idx = 1 - self.ego_idx
        if self.partner_model is not None:
            # Look at the board from the partner's perspective
            partner_obs = self._get_obs(partner_idx)
            # The partner model predicts its move
            partner_action_idx, _ = self.partner_model.predict(partner_obs, deterministic=True)
            partner_action_str = Action.INDEX_TO_ACTION[partner_action_idx]
        else:
            # If no partner model exists yet (Iteration 1), partner does nothing
            partner_action_str = Action.STAY
            
        if self.ego_idx == 0:
            joint_action = (ego_action_str, partner_action_str)
        else:
            joint_action = (partner_action_str, ego_action_str)
        
        next_state, sparse_reward, dense_reward, info = self.base_env.step(joint_action)
        terminated = self.base_env.is_done()
        truncated = False
    
        total_step_reward = sparse_reward + dense_reward
        
        info = {"joint_action": joint_action}
        
        return self._get_obs(self.ego_idx), total_step_reward, terminated, truncated, info

    def _get_obs(self, agent_idx):
        obs_tuple = self.base_env.featurize_state_mdp(self.base_env.state)
        return obs_tuple[agent_idx].flatten()
    
def train_baseline():
    env = DummyVecEnv([lambda: Monitor(OvercookedSelfPlayWrapper(layout_name="cramped_room"))])
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        device="cpu"
    )
    
    iterations = 3
    timesteps_per_iteration = 300000
    
    for i in range(iterations):
        model.save("temp_partner_model")
        partner_model = PPO.load("temp_partner_model")
        env.envs[0].env.set_partner_model(partner_model)
        
        # Train the ego agent against the current partner
        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
        
    model.save("overcooked_baseline")

    eval_env = env.envs[0].env
    return model, eval_env

def evaluate(model, gym_env, num_episodes=5):
    base_env = gym_env.base_env 
    mdp = base_env.mdp
    
    all_ep_scores = []
    all_ep_time_to_first = []
    all_ep_avg_time_between = []
    all_ep_stood_still = []
    all_ep_bumps = []
    all_ep_misplaced = []

    heatmap = np.zeros((mdp.width, mdp.height))

    for episode in range(num_episodes):
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
            ego_action_idx, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
            done = terminated or truncated
            step_count += 1
            current_state = base_env.state
            
            # Default to STAY if missing just to be safe
            joint_action = info.get("joint_action", (Action.STAY, Action.STAY))
            a0, a1 = joint_action
            
            # Stood Still Metric
            if a0 == Action.STAY or a1 == Action.STAY:
                stood_still_count += 1
            
            # Primary Task Performance (Score & Timing)
            if reward > 0:
                total_score += reward
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
                    facing_pos = mdp.get_offset(*prev_player.position, prev_player.orientation)
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

def render_heatmap(heatmap):
    plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')
    plt.title("Agent Movement Heatmap (Most Visited Tiles)")
    plt.colorbar(label="Visits")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def save_agent_gameplay(model, gym_env, output_file="aasma_ego_agent.gif", fps= 5):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    visualizer = StateVisualizer()
    
    obs, _ = gym_env.reset()
    done = False
    step = 0
    total_reward = 0
    initial_mdp = gym_env.base_env.mdp
    
    frames = []
    
    current_state = gym_env.base_env.state
    surface = visualizer.render_state(current_state, initial_mdp.terrain_mtx)
    frame = pygame.surfarray.pixels3d(surface)
    frame_actual = np.transpose(frame, (1, 0, 2)).copy()
    frames.append(frame_actual)

    while not done:
        ego_action_idx, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
        done = terminated or truncated
        total_reward += reward

        current_state = gym_env.base_env.state
        surface = visualizer.render_state(current_state, initial_mdp.terrain_mtx)
        
        frame = pygame.surfarray.pixels3d(surface)
        frame_actual = np.transpose(frame, (1, 0, 2)).copy() 
        frames.append(frame_actual)

    pygame.quit()
    imageio.mimsave(output_file, frames, fps=fps, loop=0)
    
if __name__ == "__main__":
    trained_model, env = train_baseline()
    env.set_partner_model(trained_model)
    evaluate(trained_model, env, num_episodes=5)
    save_agent_gameplay(trained_model, env)