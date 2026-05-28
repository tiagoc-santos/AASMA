import os
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
from sb3_contrib import RecurrentPPO
from pathlib import Path
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

def check_behavioral_events(agent_action, prev_player, curr_player, mdp):
    """Detect simple behavior events for one agent transition.
    A bump is counted whenever the ego attempts to move but remains
    in the same position after the transition.
    """
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

def run_single_episode(model, gym_env, episode_seed, deterministic_ego=True,):
    """Run one evaluation episode and collect task/behavior metrics."""
    np.random.seed(episode_seed)
    obs, _ = gym_env.reset(seed=episode_seed)
    base_env = gym_env.base_env
    mdp = base_env.mdp
    is_recurrent = isinstance(model, RecurrentPPO)
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    done = False
    step_count = 0
    
    ep_metrics = {
        "soup_score": 0,
        "dish_delivery_times": [],
        "stood_still_count": 0,
        "bump_count": 0,
        "misplaced_count": 0,
        "coordination_score": 0.0,
    }

    heatmap_updates = []
    prev_state = copy.deepcopy(base_env.state)

    while not done:
        if is_recurrent:
            ego_action_idx, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=deterministic_ego,)
        else:
            ego_action_idx, _ = model.predict(obs, deterministic=deterministic_ego,)

        obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)
        done = terminated or truncated
        episode_start = np.array([done], dtype=bool)
        step_count += 1

        current_state = base_env.state
        num_players = len(current_state.players)
        joint_action = info.get("joint_action", tuple([Action.STAY] * num_players),)
        ego_idx = info["ego_idx"]

        if joint_action[ego_idx] == Action.STAY:
            ep_metrics["stood_still_count"] += 1

        step_sparse_reward = sum(info.get("sparse_r_by_agent", [0.0] * num_players))

        if step_sparse_reward > 0:
            ep_metrics["soup_score"] += step_sparse_reward
            ep_metrics["dish_delivery_times"].append(step_count)

        agent_action = joint_action[ego_idx]
        prev_player = prev_state.players[ego_idx]
        curr_player = current_state.players[ego_idx]

        bumped, misplaced = check_behavioral_events(
            agent_action,
            prev_player,
            curr_player,
            mdp,)

        ep_metrics["bump_count"] += bumped
        ep_metrics["misplaced_count"] += misplaced

        heatmap_updates.append(curr_player.position)
        prev_state = copy.deepcopy(current_state)

    deliveries = len(ep_metrics["dish_delivery_times"])

    friction_penalty = (0.01 * ep_metrics["bump_count"] + 0.01 * ep_metrics["stood_still_count"])

    denominator = deliveries + friction_penalty

    if denominator > 0:
        ep_metrics["coordination_score"] = deliveries / denominator
    else:
        ep_metrics["coordination_score"] = 0.0

    return ep_metrics, step_count, heatmap_updates

def print_evaluation_summary(agg_metrics):
    """Print averaged evaluation metrics to stdout."""
    print("\n================================================")
    print("FINAL BASELINE METRICS (Average over episodes)")
    print("================================================")
    print(f"Avg Soup Score: {np.mean(agg_metrics['soup_scores']):.2f}")
    print(f"Avg Coordination Score: {np.mean(agg_metrics['coordination_scores']):.4f}")
    print(f"Avg Time to First Dish: {np.mean(agg_metrics['time_to_first']):.2f} steps")
    print(f"Avg Time Between Dishes: {np.mean(agg_metrics['avg_time_between']):.2f} steps")
    print(f"Avg Times Stood Still: {np.mean(agg_metrics['stood_still']):.2f}")
    print(f"Avg Bumps/Failed Moves: {np.mean(agg_metrics['bumps']):.2f}")
    print(f"Avg Misplaced Items: {np.mean(agg_metrics['misplaced']):.2f}")
    print("================================================")

def evaluate(model, gym_env, num_episodes=5, deterministic_partner=True, deterministic_ego=True, 
             heatmap_output_file=None, train_mode="loaded_model", seed=42):
    """Evaluate a trained model for multiple episodes."""
    gym_env.set_deterministic_partner(deterministic_partner)
    mdp = gym_env.base_env.mdp
    
    agg_metrics = {
        'soup_scores': [], 'coordination_scores': [], 'time_to_first': [], 
        'avg_time_between': [], 'stood_still': [], 'bumps': [], 
        'misplaced': [], 'deliveries': []
    }
    
    heatmap = np.zeros((mdp.width, mdp.height))

    for episode in range(num_episodes):
        ep_metrics, final_step_count, heatmap_updates = run_single_episode(model, gym_env, episode, deterministic_ego)
        
        for x, y in heatmap_updates:
            heatmap[x][y] += 1
            
        agg_metrics['soup_scores'].append(ep_metrics['soup_score'])
        agg_metrics['coordination_scores'].append(ep_metrics['coordination_score'])
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
    render_heatmap(heatmap, output_file=heatmap_output_file, train_mode=train_mode, seed=seed)

    summary = {
        "avg_soup_score": float(np.mean(agg_metrics["soup_scores"])),
        "avg_coordination_score": float(np.mean(agg_metrics["coordination_scores"])),
        "std_total_score": float(np.std(agg_metrics["soup_scores"])),
        "avg_time_to_first": float(np.mean(agg_metrics["time_to_first"])),
        "avg_time_between": float(np.mean(agg_metrics["avg_time_between"])),
        "avg_stood_still": float(np.mean(agg_metrics["stood_still"])),
        "avg_bumps": float(np.mean(agg_metrics["bumps"])),
        "avg_misplaced": float(np.mean(agg_metrics["misplaced"])),
        "avg_deliveries": float(np.mean(agg_metrics["deliveries"])),
        "success_rate": float(np.mean([score > 0 for score in agg_metrics["soup_scores"]])),
    }
    return summary


def evaluation_result(csv_file, result_row):
    """Store one summary row per complete evaluation configuration."""

    primary_metric = "avg_soup_score"

    key_fields = (
        "layout_name",
        "eval_partner",
        "train_partner_mode",
        "architecture",
        "deterministic_partner",
        "deterministic_ego",
        "seed"
    )

    fieldnames = [
        "timestamp",
        "layout_name",
        "eval_partner",
        "num_episodes",
        "deterministic_partner",
        "deterministic_ego",
        "seed",
        "architecture",
        "train_partner_mode",
        "avg_soup_score",
        "avg_coordination_score",
        "std_total_score",
        "avg_time_to_first",
        "avg_time_between",
        "avg_stood_still",
        "avg_bumps",
        "avg_misplaced",
        "avg_deliveries",
        "success_rate",
    ]

    rows = []
    previous_score = None
    best_competitor_score = -float("inf")
    best_competitor_mode = None

    if os.path.exists(csv_file):
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                same_key = all(row.get(k) == str(result_row.get(k)) for k in key_fields)
                same_scenario = (
                    row.get("layout_name") == str(result_row.get("layout_name"))
                    and row.get("eval_partner") == str(result_row.get("eval_partner"))
                    and row.get("architecture") == str(result_row.get("architecture"))
                    and row.get("deterministic_partner")
                    == str(result_row.get("deterministic_partner"))
                    and row.get("deterministic_ego")
                    == str(result_row.get("deterministic_ego")))

                if same_key:
                    previous_score = float(row.get(primary_metric, 0) or 0)
                else:
                    rows.append(row)
                    if same_scenario:
                        comp_score = float(row.get(primary_metric, 0) or 0)
                        if comp_score > best_competitor_score:
                            best_competitor_score = comp_score
                            best_competitor_mode = (f"{row.get('train_partner_mode')} " f"({row.get('architecture')})")

    clean_row = {field: result_row.get(field, "") for field in fieldnames}
    rows.append(clean_row)
    rows.sort(key=lambda row: float(row.get(primary_metric, 0) or 0),reverse=True)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    current_score = float(result_row.get(primary_metric, 0) or 0)
    coordination_score = float(result_row.get("avg_coordination_score", 0) or 0)

    print("\n================================================")
    print(" CSV LEADERBOARD UPDATE")
    print("================================================")
    print(f" Layout:              {result_row['layout_name']}")
    print(f" Eval Partner:        {result_row['eval_partner']}")
    print(f" Config:              {result_row['train_partner_mode']} | {result_row['architecture']}")
    print(f" Deterministic Ego:   {result_row['deterministic_ego']}")
    print(f" Deterministic Team:  {result_row['deterministic_partner']}")
    print(f" Seed:                {result_row['seed']}")
    print("------------------------------------------------")

    if previous_score is None:
        print(f" Personal Best: First run for this config. Soup Score: {current_score:.2f}")
    elif current_score > previous_score:
        print(f" Personal Best: [IMPROVED] {previous_score:.2f} -> {current_score:.2f}")
    elif current_score < previous_score:
        print(f" Personal Best: [DROPPED] {previous_score:.2f} -> {current_score:.2f}")
    else:
        print(f" Personal Best: [MAINTAINED] at {current_score:.2f}")

    if best_competitor_mode is not None:
        if current_score > best_competitor_score:
            print(
                f" Rival Status:  [NEW RECORD] Beat "f"{best_competitor_mode} ({best_competitor_score:.2f})!")
        elif current_score < best_competitor_score:
            print(f" Rival Status:  [LAGGING] Behind "f"{best_competitor_mode} ({best_competitor_score:.2f})")
        else:
            print(f" Rival Status:  [TIED] With "f"{best_competitor_mode} ({best_competitor_score:.2f})")
    else:
        print(" Rival Status:  No other training modes tested for this scenario.")

    print(f" Coordination:  {coordination_score:.4f}")
    print(f" File Saved:    {csv_file}")

def render_heatmap(heatmap, output_file="baseline_heatmap.pdf", train_mode="loaded_model", seed=42):
    """
    Render and save a heatmap as:
        ../heatmaps/<train_mode>_seed<seed>/<output_file>.pdf
    """
    output_dir = Path("../heatmaps") / f"{train_mode}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file is None or str(output_file).strip() == "":
        output_filename = "baseline_heatmap.pdf"
    else:
        output_filename = Path(output_file).with_suffix(".pdf").name
    output_path = output_dir / output_filename
    
    plt.imshow(heatmap.T, cmap='hot', interpolation='nearest')
    plt.title("Ego Agent Movement Heatmap (Most Visited Tiles)")
    plt.colorbar(label="Visits")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_agent_gameplay(model, gym_env, output_file="aasma_ego_agent.gif", train_mode="loaded_model",
    seed=42, fps=5, deterministic_partner=False, deterministic_ego=True,):
    """
    Record one episode and save it as:
        ../gameplay_gifs/<train_mode>_seed<seed>/<output_file>.gif
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    output_dir = Path("../gameplay_gifs") / f"{train_mode}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = Path(output_file).with_suffix(".gif").name
    output_path = output_dir / output_filename

    visualizer = StateVisualizer()
    extra_colors = [
        "red",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "brown",
    ]

    while len(visualizer.player_colors) < gym_env.num_players:
        next_color = extra_colors[
            (len(visualizer.player_colors) - 2) % len(extra_colors)
        ]
        visualizer.player_colors.append(next_color)

    gym_env.set_deterministic_partner(deterministic_partner)

    obs, _ = gym_env.reset()
    done = False

    is_recurrent = isinstance(model, RecurrentPPO)
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    initial_mdp = gym_env.base_env.mdp
    frames = []

    current_state = gym_env.base_env.state
    surface = visualizer.render_state(current_state, initial_mdp.terrain_mtx,)
    frame = pygame.surfarray.pixels3d(surface)
    frames.append(np.transpose(frame, (1, 0, 2)).copy())

    while not done:
        if is_recurrent:
            ego_action_idx, lstm_state = model.predict(obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=deterministic_ego,)
        else:
            ego_action_idx, _ = model.predict(obs, deterministic=deterministic_ego,)

        obs, reward, terminated, truncated, info = gym_env.step(ego_action_idx)

        done = terminated or truncated
        episode_start = np.array([done], dtype=bool)

        current_state = gym_env.base_env.state
        surface = visualizer.render_state(current_state, initial_mdp.terrain_mtx,)
        frame = pygame.surfarray.pixels3d(surface)
        frames.append(np.transpose(frame, (1, 0, 2)).copy())

    pygame.quit()

    duration_ms = 1000 / fps
    imageio.mimsave(str(output_path),frames, format="GIF", duration=duration_ms, loop=0,)

    print(f"Gameplay GIF saved to: {output_path}")