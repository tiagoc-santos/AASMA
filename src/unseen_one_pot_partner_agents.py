"""
Evaluation-only unseen teammate policies for a one-pot three-player Overcooked layout.

IMPORTANT:
    Keep this module out of all training partner pools.
    These agents are intended only for the final unseen-partner evaluation suite.

Interface:
    Each class implements:
        predict(obs, state=None, player_idx=None, mdp=None, deterministic=False)
            -> (action_index, None)

The implementation reuses the BFS motion helper from partner_agents.py, but
defines new single-pot conventions that were not used by the training partners.
"""

import numpy as np
from overcooked_ai_py.mdp.actions import Action

from partner_agents import ACTION_TO_IDX, N_ACTIONS, bfs_next_action, closest


POT_NEEDS_ONION = {"empty", "1_items", "2_items"}


def _held_name(player):
    return player.held_object.name if player.held_object is not None else None


def _only_pot(mdp):
    """Return the single pot location and fail clearly on a different layout."""
    pots = mdp.get_pot_locations()
    if len(pots) != 1:
        raise ValueError(
            "The one-pot unseen test suite expects exactly one pot, "
            f"but found {len(pots)}."
        )
    return pots[0]


def _pot_status(state, mdp, pot_pos):
    """Return the status name of the shared pot."""
    pot_states = mdp.get_pot_states(state)
    for status in (
        "empty",
        "1_items",
        "2_items",
        "3_items",
        "cooking",
        "ready",
        "both_ready",
    ):
        if pot_pos in pot_states.get(status, []):
            return status
    return None


def _ingredient_count_from_status(status):
    return {
        "empty": 0,
        "1_items": 1,
        "2_items": 2,
        "3_items": 3,
    }.get(status, None)


class _OnePotStateAgent:
    """Base class for deterministic, evaluation-only, state-aware partners."""

    needs_state = True

    def _random_fallback(self):
        return np.random.randint(0, N_ACTIONS), None

    def _act_toward(self, target_pos, state, player_idx, mdp):
        if target_pos is None:
            return ACTION_TO_IDX[Action.STAY], None

        action = bfs_next_action(
            state.players[player_idx].position,
            target_pos,
            mdp,
            state,
            player_idx,
        )
        return ACTION_TO_IDX[action], None


class AlternatingCookerAgent(_OnePotStateAgent):
    """
    Novel upstream-only partner for a shared one-pot layout.

    Two instances follow a turn convention rather than both chasing onions:
        worker_slot=0 supplies ingredient 1 and ingredient 3
        worker_slot=1 supplies ingredient 2 and starts cooking

    They never fetch a dish or deliver a soup. Therefore, an ego paired with
    two AlternatingCookerAgents must provide the serving/delivery role.
    """

    def __init__(self, worker_slot):
        if worker_slot not in (0, 1):
            raise ValueError("worker_slot must be 0 or 1")
        self.worker_slot = worker_slot

    def _choose_target(self, state, player, mdp):
        pot = _only_pot(mdp)
        status = _pot_status(state, mdp, pot)
        held_name = _held_name(player)

        # Finish an already-started onion commitment.
        if held_name == "onion":
            return pot if status in POT_NEEDS_ONION else None

        # This policy never performs serving tasks.
        if held_name is not None:
            return None

        ingredient_count = _ingredient_count_from_status(status)

        if ingredient_count is not None and ingredient_count < 3:
            active_worker = ingredient_count % 2
            if self.worker_slot == active_worker:
                return closest(
                    mdp.get_onion_dispenser_locations(),
                    player.position,
                )
            # Clear the shared-pot area while the other cooker is active.
            return closest(mdp.get_onion_dispenser_locations(), player.position)

        # One designated partner initiates cooking once the pot is full.
        if status == "3_items" and self.worker_slot == 1:
            return pot

        # Once no cooking work is available, keep clear of the shared pot.
        if held_name is None:
            return closest(mdp.get_onion_dispenser_locations(), player.position)

        return None

    def predict(self, obs, state=None, player_idx=None, mdp=None, deterministic=False):
        if state is None or mdp is None:
            return self._random_fallback()
        target = self._choose_target(state, state.players[player_idx], mdp)
        return self._act_toward(target, state, player_idx, mdp)


class PrepositioningServerAgent(_OnePotStateAgent):
    """
    Novel downstream-only partner for a shared one-pot layout.

    This partner never collects onions. Unlike the training plater, it gets
    a dish while the soup is cooking, waits with it, then picks up/delivers
    the soup when the pot is ready.

    Pairing an ego with two of these agents tests whether it can perform the
    missing ingredient/cooking role for an unseen server convention.
    """

    def __init__(self, server_slot):
        if server_slot not in (0, 1):
            raise ValueError("server_slot must be 0 or 1")
        self.server_slot = server_slot

    def _choose_target(self, state, player, mdp):
        pot = _only_pot(mdp)
        status = _pot_status(state, mdp, pot)
        held_name = _held_name(player)

        if held_name == "soup":
            return closest(mdp.get_serving_locations(), player.position)

        if held_name == "dish":
            return pot if status in {"ready", "both_ready"} else None

        # This policy never handles onions or other carried objects.
        if held_name is not None:
            return None

        # Both servers may preposition a dish; BFS treats one another as
        # temporary obstacles, making this a distinct coordination convention.
        if status in {"3_items", "cooking", "ready", "both_ready"}:
            return closest(mdp.get_dish_dispenser_locations(), player.position)

        return None

    def predict(self, obs, state=None, player_idx=None, mdp=None, deterministic=False):
        if state is None or mdp is None:
            return self._random_fallback()
        target = self._choose_target(state, state.players[player_idx], mdp)
        return self._act_toward(target, state, player_idx, mdp)


class TimedRoleSwitchingAgent(_OnePotStateAgent):
    """
    Dynamic unseen teammate.

    Before switch_step it follows the AlternatingCooker convention.
    From switch_step onward it follows the PrepositioningServer convention.
    If it is already carrying an object when the switch happens, it finishes
    that held-object commitment first.

    With two instances, the ego initially needs to serve; after the switch it
    needs to supply/cook ingredients.
    """

    def __init__(self, worker_slot, switch_step=200):
        if worker_slot not in (0, 1):
            raise ValueError("worker_slot must be 0 or 1")
        if switch_step <= 0:
            raise ValueError("switch_step must be positive")
        self.worker_slot = worker_slot
        self.switch_step = switch_step
        self._cooker = AlternatingCookerAgent(worker_slot)
        self._server = PrepositioningServerAgent(worker_slot)

    def predict(self, obs, state=None, player_idx=None, mdp=None, deterministic=False):
        if state is None or mdp is None:
            return self._random_fallback()

        held_name = _held_name(state.players[player_idx])

        # Honour current cargo across the switch boundary.
        if held_name == "onion":
            agent = self._cooker
        elif held_name in {"dish", "soup"}:
            agent = self._server
        elif state.timestep < self.switch_step:
            agent = self._cooker
        else:
            agent = self._server

        return agent.predict(
            obs,
            state=state,
            player_idx=player_idx,
            mdp=mdp,
            deterministic=deterministic,
        )


class YieldingGeneralistAgent(_OnePotStateAgent):
    """
    Novel complete-task partner using a low-contention shared-pot convention.

    Two instances can complete the whole recipe without the ego:
        - onion supply alternates by ingredient count;
        - worker_slot=1 starts cooking;
        - worker_slot=0 alone handles dish pickup and delivery.

    This differs from GreedyChefAgent, where multiple partners independently
    chase the same highest-priority task.
    """

    def __init__(self, worker_slot):
        if worker_slot not in (0, 1):
            raise ValueError("worker_slot must be 0 or 1")
        self.worker_slot = worker_slot

    def _choose_target(self, state, player, mdp):
        pot = _only_pot(mdp)
        status = _pot_status(state, mdp, pot)
        held_name = _held_name(player)

        if held_name == "soup":
            return closest(mdp.get_serving_locations(), player.position)

        if held_name == "dish":
            return pot if status in {"ready", "both_ready"} else None

        if held_name == "onion":
            return pot if status in POT_NEEDS_ONION else None

        if held_name is not None:
            return None

        ingredient_count = _ingredient_count_from_status(status)

        if ingredient_count is not None and ingredient_count < 3:
            active_worker = ingredient_count % 2
            if self.worker_slot == active_worker:
                return closest(
                    mdp.get_onion_dispenser_locations(),
                    player.position,
                )
            # Yield away from the single shared pot while the other worker acts.
            return closest(mdp.get_onion_dispenser_locations(), player.position)

        if status == "3_items" and self.worker_slot == 1:
            return pot

        # Only one partner approaches dish/soup stations, avoiding duplication.
        if (
            status in {"cooking", "ready", "both_ready"}
            and self.worker_slot == 0
        ):
            return closest(mdp.get_dish_dispenser_locations(), player.position)

        return None

    def predict(self, obs, state=None, player_idx=None, mdp=None, deterministic=False):
        if state is None or mdp is None:
            return self._random_fallback()
        target = self._choose_target(state, state.players[player_idx], mdp)
        return self._act_toward(target, state, player_idx, mdp)


UNSEEN_ONE_POT_TEAM_NAMES = [
    "unseen_alternating_cookers",
    "unseen_prepositioning_servers",
    "unseen_role_switchers",
    "unseen_yielding_generalists",
]


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
