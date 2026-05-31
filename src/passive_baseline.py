import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from env import OvercookedSelfPlayWrapper
from evaluation import evaluate
from partner_agents import (RandomPartner, StationaryPartner, GreedyChefAgent, SpecialistAgent, NoisyGreedyAgent, 
YieldingGeneralistAgent, TimedRoleSwitchingAgent, PrepositioningServerAgent, AlternatingCookerAgent,)

UNSEEN_ONE_POT_TEAM_NAMES = ["unseen_alternating_cookers", "unseen_prepositioning_servers", 
                             "unseen_role_switchers", "unseen_yielding_generalists",]

def load_trained_policy(path, architecture, env=None, device="auto"):
    algorithm_class = RecurrentPPO if architecture == "rnn" else PPO

    return algorithm_class.load(str(path),env=env,device=device,)


def make_unseen_one_pot_team(team_name, num_partners):
    """
    Build one of the final unseen evaluation teams.

    This project's three-player setting is expected to have exactly two
    teammate slots in addition to the ego.
    """
    if num_partners != 2:
        raise ValueError(
            "The one-pot unseen test suite is defined for two teammate slots "
            f"(three players total), but got num_partners={num_partners}."
        )

    if team_name == "unseen_alternating_cookers":
        return [
            AlternatingCookerAgent(worker_slot=0),
            AlternatingCookerAgent(worker_slot=1),
        ]

    if team_name == "unseen_prepositioning_servers":
        return [
            PrepositioningServerAgent(server_slot=0),
            PrepositioningServerAgent(server_slot=1),
        ]

    if team_name == "unseen_role_switchers":
        return [
            TimedRoleSwitchingAgent(worker_slot=0, switch_step=200),
            TimedRoleSwitchingAgent(worker_slot=1, switch_step=200),
        ]

    if team_name == "unseen_yielding_generalists":
        return [
            YieldingGeneralistAgent(worker_slot=0),
            YieldingGeneralistAgent(worker_slot=1),
        ]

    raise ValueError(f"Unknown unseen one-pot team: {team_name}")

def make_partner_team(team_name, num_partners):
    """Create fixed teammate teams for contribution-focused evaluation."""

    if team_name == "greedy_pair":
        return [GreedyChefAgent() for _ in range(num_partners)]

    if team_name == "specialist_complete":
        return [
            SpecialistAgent(role="fetcher") if i % 2 == 0
            else SpecialistAgent(role="plater")
            for i in range(num_partners)
        ]

    if team_name == "noisy_pair":
        return [NoisyGreedyAgent(epsilon=0.40) for _ in range(num_partners)]

    if team_name == "greedy_noisy":
        return [
            GreedyChefAgent() if i % 2 == 0
            else NoisyGreedyAgent(epsilon=0.40)
            for i in range(num_partners)
        ]

    # Contribution tests: these teams need the ego to fill a missing role.
    if team_name == "fetcher_pair":
        return [SpecialistAgent(role="fetcher") for _ in range(num_partners)]

    if team_name == "plater_pair":
        return [SpecialistAgent(role="plater") for _ in range(num_partners)]

    if team_name == "fetcher_stationary":
        return [
            SpecialistAgent(role="fetcher") if i % 2 == 0
            else StationaryPartner()
            for i in range(num_partners)
        ]

    if team_name == "plater_stationary":
        return [
            SpecialistAgent(role="plater") if i % 2 == 0
            else StationaryPartner()
            for i in range(num_partners)
        ]

    if team_name == "greedy_stationary":
        return [
            GreedyChefAgent() if i % 2 == 0
            else StationaryPartner()
            for i in range(num_partners)
        ]

    if team_name == "noisy_stationary":
        return [
            NoisyGreedyAgent(epsilon=0.35) if i % 2 == 0
            else StationaryPartner()
            for i in range(num_partners)
        ]
    if team_name in UNSEEN_ONE_POT_TEAM_NAMES:
        return make_unseen_one_pot_team(team_name, num_partners)

    raise ValueError(f"Unsupported team_name: {team_name}")


def evaluate_controller(
    controller,
    controller_label,
    team_name,
    layout_name,
    architecture,
    num_episodes,
    deterministic_ego,
    deterministic_partner,
    seed,
):
    """Evaluate one controlled ego policy with one fixed partner team."""

    env = OvercookedSelfPlayWrapper(
        layout_name=layout_name,
        architecture=architecture,
    )

    partner_team = make_partner_team(
        team_name,
        env.num_players - 1,
    )

    env.set_partner_pool([partner_team])

    heatmap_name = (
        f"passive_{controller_label}_{team_name}_"
        f"{'ego_det' if deterministic_ego else 'ego_stoch'}.pdf"
    )

    summary = evaluate(
        controller,
        env,
        num_episodes=num_episodes,
        deterministic_partner=deterministic_partner,
        deterministic_ego=deterministic_ego,
        heatmap_output_file=heatmap_name,
        train_mode=controller_label,
        seed=seed
    )

    env.close()
    return summary


def write_results(output_csv, rows):
    fieldnames = [
        "model_label",
        "model_seed",
        "team_name",
        "trained_ego_score",
        "passive_ego_score",
        "marginal_contribution",
        "trained_ego_deliveries",
        "passive_ego_deliveries",
        "trained_ego_bumps",
        "passive_ego_bumps",
        "trained_ego_time_to_first",
        "passive_ego_time_to_first",
    ]

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_label", type=str, required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--layout_name", type=str, default="three_chefs")
    parser.add_argument("--architecture", type=str, default="rnn", choices=["cnn", "mlp", "rnn"])
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument(
        "--suite",
        type=str,
        default="both",
        choices=["compatibility", "contribution", "unseen_final", "both"],
    )
    parser.add_argument(
        "--deterministic_ego",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--deterministic_partner",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="../passive_baseline_results.csv",
    )

    args = parser.parse_args()
    
    model_stem_parts = Path(args.model).stem.split("_")

    if model_stem_parts[0] in {"cnn", "mlp", "rnn"}:
        args.architecture = model_stem_parts[0]

    seed_token = model_stem_parts[-1]
    if seed_token.startswith("seed") and seed_token[4:].isdigit():
        args.model_seed = int(seed_token[4:])
    else:
        print(f"Could not infer model seed from {Path(args.model).name}; "f"using --model_seed={args.model_seed}.")

    deterministic_ego = args.deterministic_ego.lower() == "true"
    deterministic_partner = args.deterministic_partner.lower() == "true"

    compatibility_teams = [
        "greedy_pair",
        "specialist_complete",
        "noisy_pair",
        "greedy_noisy",
    ]

    contribution_teams = [
        "fetcher_pair",
        "plater_pair",
        "fetcher_stationary",
        "plater_stationary",
        "greedy_stationary",
        "noisy_stationary",
    ]

    if args.suite == "compatibility":
        teams = compatibility_teams
    elif args.suite == "contribution":
        teams = contribution_teams
    elif args.suite == "unseen_final":
        teams = UNSEEN_ONE_POT_TEAM_NAMES
    else:
        teams = compatibility_teams + contribution_teams

    trained_ego = load_trained_policy(args.model, args.architecture, device="auto")
    passive_ego = StationaryPartner()

    result_rows = []

    print("\n================================================")
    print(f"PASSIVE-EGO BASELINE: {args.model_label} | seed {args.model_seed}")
    print("================================================")

    for team_name in teams:
        print(f"\nEvaluating team: {team_name}")

        trained_summary = evaluate_controller(
            trained_ego,
            controller_label=f"{args.model_label}_seed{args.model_seed}",
            team_name=team_name,
            layout_name=args.layout_name,
            architecture=args.architecture,
            num_episodes=args.eval_episodes,
            deterministic_ego=deterministic_ego,
            deterministic_partner=deterministic_partner,
            seed=args.model_seed
        )

        passive_summary = evaluate_controller(
            passive_ego,
            controller_label="stationary_ego",
            team_name=team_name,
            layout_name=args.layout_name,
            architecture=args.architecture,
            num_episodes=args.eval_episodes,
            deterministic_ego=True,
            deterministic_partner=deterministic_partner,
            seed=0
        )

        marginal_contribution = (
            trained_summary["avg_soup_score"]
            - passive_summary["avg_soup_score"]
        )

        print("\n------------------------------------------------")
        print(f"Team:                  {team_name}")
        print(f"Trained ego score:     {trained_summary['avg_soup_score']:.2f}")
        print(f"Passive ego score:     {passive_summary['avg_soup_score']:.2f}")
        print(f"Marginal contribution: {marginal_contribution:+.2f}")
        print("------------------------------------------------")

        result_rows.append({
            "model_label": args.model_label,
            "model_seed": args.model_seed,
            "team_name": team_name,
            "trained_ego_score": trained_summary["avg_soup_score"],
            "passive_ego_score": passive_summary["avg_soup_score"],
            "marginal_contribution": marginal_contribution,
            "trained_ego_deliveries": trained_summary["avg_deliveries"],
            "passive_ego_deliveries": passive_summary["avg_deliveries"],
            "trained_ego_bumps": trained_summary["avg_bumps"],
            "passive_ego_bumps": passive_summary["avg_bumps"],
            "trained_ego_time_to_first": trained_summary["avg_time_to_first"],
            "passive_ego_time_to_first": passive_summary["avg_time_to_first"],
        })

    write_results(args.output_csv, result_rows)

    print("\n================================================")
    print(f"Saved results to: {args.output_csv}")
    print("================================================")


if __name__ == "__main__":
    main()