import argparse
import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env import OvercookedSelfPlayWrapper, PartnerAwareExtractor, make_env
from evaluation import evaluate, evaluation_result, save_agent_gameplay
from partner_agents import RandomPartner, StationaryPartner, GreedyChefAgent, SpecialistAgent, NoisyGreedyAgent

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


def build_pretrain_partner_pool(num_partners):
    """Competent partners used to learn the task before adaptation training."""
    return [
        [GreedyChefAgent() for _ in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else SpecialistAgent(role='fetcher') for i in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else SpecialistAgent(role='plater') for i in range(num_partners)],
    ]


def build_role_partner_pool(num_partners):
    return [
        [GreedyChefAgent() for _ in range(num_partners)],
        [GreedyChefAgent() for _ in range(num_partners)],
        [SpecialistAgent(role='fetcher') for _ in range(num_partners)],
        [SpecialistAgent(role='plater') for _ in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else SpecialistAgent(role='fetcher') for i in range(num_partners)],
        [GreedyChefAgent() if i % 2 == 0 else SpecialistAgent(role='plater') for i in range(num_partners)],
        ]

def build_robustness_partner_pool(num_partners):
    """Selected Stage 3 pool: greedy retention, role diversity and noisy robustness."""

    greedy_team = [GreedyChefAgent() for _ in range(num_partners)]
    fetcher_team = [SpecialistAgent(role="fetcher") for _ in range(num_partners)]
    plater_team = [SpecialistAgent(role="plater") for _ in range(num_partners)]

    greedy_fetcher_team = [GreedyChefAgent() if i % 2 == 0 else SpecialistAgent(role="fetcher") for i in range(num_partners)]
    greedy_plater_team = [GreedyChefAgent() if i % 2 == 0 else SpecialistAgent(role="plater") for i in range(num_partners)]

    noisy_015_team = [NoisyGreedyAgent(epsilon=0.15) for _ in range(num_partners)]
    noisy_025_team = [NoisyGreedyAgent(epsilon=0.25) for _ in range(num_partners)]
    noisy_035_team = [NoisyGreedyAgent(epsilon=0.35) for _ in range(num_partners)]

    greedy_noisy_025_team = [GreedyChefAgent() if i % 2 == 0 else NoisyGreedyAgent(epsilon=0.25) for i in range(num_partners)]
    greedy_noisy_035_team = [GreedyChefAgent() if i % 2 == 0 else NoisyGreedyAgent(epsilon=0.35) for i in range(num_partners)]

    return [greedy_team, greedy_team, greedy_team, greedy_team, greedy_team, greedy_team, fetcher_team, plater_team,
        greedy_fetcher_team, greedy_plater_team, noisy_015_team, noisy_025_team, noisy_035_team, noisy_035_team,
        greedy_noisy_025_team, greedy_noisy_035_team, greedy_noisy_035_team,]


def train_adhoc_curriculum(model, env, total_timesteps, num_partners, checkpoint_dir):
    """Selected three-stage ad hoc curriculum."""

    pretrain_steps = total_timesteps // 4
    role_steps = total_timesteps // 8
    robustness_steps = total_timesteps - pretrain_steps - role_steps

    print(f"Ad hoc stage 1/3 - task pretraining ({pretrain_steps} steps)")
    env.env_method("set_dense_reward_scale", 1.0)
    env.env_method("set_partner_pool", build_pretrain_partner_pool(num_partners))
    model.learn(total_timesteps=pretrain_steps, reset_num_timesteps=False)

    low_path = checkpoint_dir / "adhoc_low_checkpoint.zip"
    low_vec_path = checkpoint_dir / "adhoc_low_vecnormalize.pkl"
    model.save(str(low_path))
    env.save(str(low_vec_path))

    print(f"Ad hoc stage 2/3 - role diversity ({role_steps} steps)")
    env.env_method("set_dense_reward_scale", 0.5)
    env.env_method("set_partner_pool", build_role_partner_pool(num_partners))
    model.learn(total_timesteps=role_steps, reset_num_timesteps=False)

    mid_path = checkpoint_dir / "adhoc_mid_checkpoint.zip"
    mid_vec_path = checkpoint_dir / "adhoc_mid_vecnormalize.pkl"
    model.save(str(mid_path))
    env.save(str(mid_vec_path))
    print(f"Saved Stage 2 model: {mid_path}")
    print(f"Saved Stage 2 VecNormalize: {mid_vec_path}")

    print(f"Ad hoc stage 3/3 - partner robustness ({robustness_steps} steps)")
    env.env_method("set_dense_reward_scale", 0.5)
    env.env_method("set_partner_pool", build_robustness_partner_pool(num_partners))
    model.learn(total_timesteps=robustness_steps, reset_num_timesteps=False)

    final_path = checkpoint_dir / "adhoc_final_checkpoint.zip"
    final_vec_path = checkpoint_dir / "adhoc_final_vecnormalize.pkl"
    model.save(str(final_path))
    env.save(str(final_vec_path))

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
    if partner_type == 'heldout_noisy_greedy':
        return [NoisyGreedyAgent(epsilon=0.40) for _ in range(num_partners)]
    if partner_type == 'heldout_greedy_noisy':
        return [GreedyChefAgent() if i % 2 == 0 else NoisyGreedyAgent(epsilon=0.40) for i in range(num_partners)]
    raise ValueError(f"Unsupported partner_type: {partner_type}")


def train_baseline(total_timesteps=2000000,
                   train_partner_mode="curriculum", train_noisy_epsilon=0.25,
                   layout_name="three_chefs", num_cpu=4, architecture='cnn', seed=42):
    
    raw_env = SubprocVecEnv([make_env(layout_name, i, architecture=architecture, seed=seed) for i in range(num_cpu)])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=1e-4,
        n_steps=2048 // num_cpu,
        batch_size=256 if architecture == 'cnn' else 512,
        n_epochs=5,
        ent_coef=0.01,
        target_kl=0.03,
        seed=seed, 
        verbose=1,
        device="auto",
        policy_kwargs=dict(
            features_extractor_class=PartnerAwareExtractor,
            features_extractor_kwargs=dict(features_dim=256, architecture=architecture),
        )
    )
    
    iterations = 10
    timesteps_per_iteration = total_timesteps // iterations
    num_partners = env.get_attr("num_players")[0] - 1
    training_partner_pool = build_training_partner_pool(num_partners, noisy_epsilon=train_noisy_epsilon)

    output_dir = Path("../models") / f"{architecture}_{train_partner_mode}_seed{seed}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if train_partner_mode == "adhoc_curriculum":
        train_adhoc_curriculum(model, env, total_timesteps, num_partners,Path(output_dir))
    else:
        if train_partner_mode == "random_pool":
            env.env_method("set_partner_pool", training_partner_pool)

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            if train_partner_mode == "random_pool":
                pass
            # self play
            else: 
                num_players = env.get_attr("num_players")[0]
                if i == 0:
                    env.env_method("set_partner_models", [RandomPartner()] * num_players)
                    env.env_method("set_deterministic_partner", False)
                else:
                    temp_partner_path = output_dir / f"temp_partner_iter_{i}.zip"
                    model.save(str(temp_partner_path))
                    partner_model = PPO.load(str(temp_partner_path))

            model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)

    output_filename = f"{architecture}_{train_partner_mode}_{total_timesteps}_seed{seed}.zip"
    output_path = output_dir / output_filename
    vecnormalize_path = output_dir / f"{architecture}_{train_partner_mode}_{total_timesteps}_seed{seed}_vecnormalize.pkl"

    model.save(str(output_path))
    env.save(str(vecnormalize_path))

    print(f"Saved final model: {output_path}")
    
    eval_env = OvercookedSelfPlayWrapper(layout_name=layout_name, architecture=architecture)
    loaded_partner = PPO.load(str(output_path))
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
                        choices=['curriculum', 'random_pool', 'adhoc_curriculum'],
                        help='Training partner schedule: self-play curriculum, static random_pool or three-stage adhoc_curriculum')
    parser.add_argument('--train_noisy_epsilon', type=float, default=0.25,
                        help='Epsilon used for NoisyGreedyAgent in training partner pool')
    parser.add_argument('--eval_partner', type=str, default='ppo',
                        choices=['ppo', 'random', 'stationary', 'greedy', 'specialists', 'noisy_greedy',
                                 'heldout_noisy_greedy', 'heldout_greedy_noisy'],
                        help='Partner to use during single-team evaluation and gameplay rendering')
    parser.add_argument('--eval_partner_epsilon', type=float, default=0.25,
                        help='Epsilon for noisy_greedy evaluation partner')
    parser.add_argument('--eval_suite', type=str, default='single',
                        choices=['single', 'common'],
                        help='Use single eval_partner or the fixed common external test suite')
    parser.add_argument('--deterministic_partner', type=str, default='false',
                        choices=['true', 'false'],
                        help='Whether partner uses deterministic actions during eval/rendering')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of episodes used during evaluation')
    parser.add_argument('--results_csv', type=str, default='evaluation_results.csv',
                        help='CSV file where evaluation summaries are stored')
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='Number of cpu cores used during training')
    parser.add_argument('--architecture', type=str, default='cnn',
                        help='Type of architecture to be used',
                        choices=['mlp', 'cnn'])
    parser.add_argument('--deterministic_ego', type=str, default='true',
                        choices=['true', 'false'], help='Whether the ego policy selects deterministic actions during evaluation')
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training")  
    
    args = parser.parse_args()
    deterministic_partner = args.deterministic_partner.lower() == 'true'
    deterministic_ego = args.deterministic_ego.lower() == 'true'
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    set_random_seed(seed)
    model_stem_parts = None
    if args.model is None:
        trained_model, env, model_filename = train_baseline(
            args.timesteps,
            train_partner_mode=args.train_partner_mode,
            train_noisy_epsilon=args.train_noisy_epsilon,
            layout_name=args.layout_name,
            num_cpu=args.num_cpu,
            architecture=args.architecture,
            seed=args.seed)
    else:
        model_stem_parts = Path(args.model).stem.split("_")
        if len(model_stem_parts) >= 3:
            args.architecture = model_stem_parts[0]
        trained_model = PPO.load(args.model)
        env = OvercookedSelfPlayWrapper(layout_name=args.layout_name, architecture=args.architecture) 
        model_filename = Path(args.model).name

    if args.model is None:
        train_mode_label = args.train_partner_mode
    else:
        if model_stem_parts is None:
            model_stem_parts = Path(args.model).stem.split("_")
        if len(model_stem_parts) >= 3:
            train_mode_label = "_".join(model_stem_parts[1:-1])
        else:
            train_mode_label = "loaded_model"

    if args.eval_suite == "common":
        eval_partner_types = ["greedy", "specialists", "heldout_noisy_greedy", "heldout_greedy_noisy",]
    else:
        eval_partner_types = [args.eval_partner]

    ego_eval_label = "ego_det" if deterministic_ego else "ego_stoch"
    partner_eval_label = "partner_det" if deterministic_partner else "partner_stoch"
    for eval_partner_type in eval_partner_types:
        eval_team = make_eval_team(
            eval_partner_type,
            env.num_players - 1,
            trained_model,
            noisy_epsilon=args.eval_partner_epsilon
        )
        env.set_partner_pool([eval_team])

        heatmap_filename = (f"{args.architecture}_{train_mode_label}_{eval_partner_type}_"f"seed{args.seed}_\
                            {ego_eval_label}_{partner_eval_label}.pdf")
        summary = evaluate(
            trained_model,
            env,
            num_episodes=args.eval_episodes,
            deterministic_partner=deterministic_partner,
            deterministic_ego = deterministic_ego,
            heatmap_output_file=heatmap_filename,)

        result_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "layout_name": args.layout_name,
            "eval_partner": eval_partner_type,
            "num_episodes": args.eval_episodes,
            "deterministic_partner": deterministic_partner,
            "deterministic_ego": deterministic_ego,
            "seed": args.seed,
            "architecture": args.architecture,
            "train_partner_mode": train_mode_label,
            **summary,
        }
        evaluation_result("../" + args.results_csv, result_row)

        gif_filename = (f"{args.architecture}_{train_mode_label}_{eval_partner_type}_"f"seed{args.seed}_{ego_eval_label}_{partner_eval_label}.gif")
        save_agent_gameplay(trained_model, env, output_file=gif_filename,deterministic_partner=deterministic_partner, deterministic_ego=deterministic_ego,)