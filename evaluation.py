import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

def check_behavioral_events(agent_action, prev_player, curr_player, mdp):
    """Detect simple behavior events for one agent transition."""
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
    """Run one evaluation episode and collect task/behavior metrics."""
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
        ego_action_idx, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
        done = terminated or truncated
        step_count += 1
        
        current_state = base_env.state
        num_players = len(current_state.players)
        joint_action = info.get("joint_action", tuple([Action.STAY] * num_players))

        # Stood Still Metric
        ep_metrics['stood_still_count'] += sum(
            1 for action in joint_action
            if action == Action.STAY
        )
            
        # Score & Delivery Tracking
        step_sparse_reward = sum(info.get("sparse_r_by_agent", [0.0] * num_players))
        if step_sparse_reward > 0:
            ep_metrics['total_score'] += step_sparse_reward
            ep_metrics['dish_delivery_times'].append(step_count)

        # Behavior Metrics Loop
        for agent_idx in range(num_players):
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
    """Print averaged evaluation metrics to stdout."""
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
    """Evaluate a trained model for multiple episodes."""
    gym_env.set_deterministic_partner(deterministic_partner)
    mdp = gym_env.base_env.mdp
    
    agg_metrics = {
        'scores': [], 'time_to_first': [], 'avg_time_between': [],
        'stood_still': [], 'bumps': [], 'misplaced': [], 'deliveries': []
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
        agg_metrics['deliveries'].append(len(deliveries))
        time_to_first = deliveries[0] if deliveries else final_step_count
        agg_metrics['time_to_first'].append(time_to_first)
        
        avg_interval = np.mean(np.diff(deliveries)) if len(deliveries) > 1 else 0
        agg_metrics['avg_time_between'].append(avg_interval)

    print_evaluation_summary(agg_metrics)
    render_heatmap(heatmap)

    summary = {
        "avg_total_score": float(np.mean(agg_metrics["scores"])),
        "std_total_score": float(np.std(agg_metrics["scores"])),
        "avg_time_to_first": float(np.mean(agg_metrics["time_to_first"])),
        "avg_time_between": float(np.mean(agg_metrics["avg_time_between"])),
        "avg_stood_still": float(np.mean(agg_metrics["stood_still"])),
        "avg_bumps": float(np.mean(agg_metrics["bumps"])),
        "avg_misplaced": float(np.mean(agg_metrics["misplaced"])),
        "avg_deliveries": float(np.mean(agg_metrics["deliveries"])),
        "success_rate": float(np.mean([score > 0 for score in agg_metrics["scores"]])),
    }
    return summary


def evaluation_result(csv_file, result_row):
    """Stores one summary row per (layout_name, eval_partner)."""
    key_fields = ("layout_name", "eval_partner")
    fieldnames = [
        "timestamp", "layout_name", "eval_partner", "num_episodes",
        "deterministic_partner", "model_file", "train_partner_mode",
        "avg_total_score", "std_total_score", "avg_time_to_first",
        "avg_time_between", "avg_stood_still", "avg_bumps",
        "avg_misplaced", "avg_deliveries", "success_rate",
    ]

    rows = []
    previous_score = None

    if os.path.exists(csv_file):
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                same_key = all(row.get(k) == str(result_row.get(k)) for k in key_fields)

                if same_key:
                    previous_score = float(row.get("avg_total_score", 0) or 0)
                else:
                    rows.append(row)

    clean_row = {field: result_row.get(field, "") for field in fieldnames}
    rows.append(clean_row)

    rows.sort(key=lambda row: float(row.get("avg_total_score", 0) or 0), reverse=True)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    current_score = float(result_row.get("avg_total_score", 0) or 0)

    print(f"Results saved/updated in {csv_file} for layout='{result_row['layout_name']}' and partner='{result_row['eval_partner']}'\n")

    if previous_score is None:
        print(f"No previous result for this layout/partner. Current average score: {current_score:.2f}\n")
    elif current_score > previous_score:
        print(f"Average score improved: {previous_score:.2f} -> {current_score:.2f}\n")
    elif current_score < previous_score:
        print(f"Average score got worse: {previous_score:.2f} -> {current_score:.2f}\n")
    else:
        print(f"Average score stayed the same: {current_score:.2f}\n")

def render_heatmap(heatmap, output_file="baseline_heatmap.pdf"):
    """Render and save a heatmap of visited grid tiles."""
    plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')
    plt.title("Agent Movement Heatmap (Most Visited Tiles)")
    plt.colorbar(label="Visits")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

def save_agent_gameplay(model, gym_env, output_file="aasma_ego_agent.mp4", fps=5, deterministic_partner=True):
    """Record one episode and save it as GIF or video."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    visualizer = StateVisualizer()

    extra_colors = ["red", "yellow", "purple", "orange", "cyan", "magenta", "brown"]

    while len(visualizer.player_colors) < gym_env.num_players:
        next_color = extra_colors[
            (len(visualizer.player_colors) - 2) % len(extra_colors)
        ]
        visualizer.player_colors.append(next_color)

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
        ego_action_idx, _ = model.predict(obs, deterministic=True)
        
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