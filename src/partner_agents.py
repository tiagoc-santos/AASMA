"""
Each agent implements:
    predict(obs, state=None, player_idx=None, mdp=None, deterministic=False)
             -> (action_idx: int, None)

The `state`, `player_idx`, and `mdp` kwargs are only used by heuristic agents.
The observation-only interface (RandomPartner, NoisyPPOPartner) remains
compatible with the original wrapper without any changes.
"""

import numpy as np
from collections import deque
from overcooked_ai_py.mdp.actions import Action, Direction

ACTION_TO_IDX = {a: i for i, a in enumerate(Action.ALL_ACTIONS)}
N_ACTIONS     = len(Action.ALL_ACTIONS)
POT_NEEDS_ONION = {"empty", "1_items", "2_items"}

# Direction vectors (x-right, y-down)
DIR_VECTORS = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0,  1),
    Direction.EAST:  (1,  0),
    Direction.WEST:  (-1, 0),
}


# Used to create a trajectory of actions for the agents that follow an heuristic
def bfs_next_action(start, target_pos, mdp, state, player_idx):
    """
    Goal:
    Return one immediate action that moves this agent toward a chosen target tile.

    Behavior:
    1) If already next to target, either rotate to face it or INTERACT.
    2) Otherwise, run BFS on walkable floor cells to find the shortest route
       to any cell adjacent to target.
    3) Return only the first step of that shortest route.
    4) If unreachable, STAY.
    """
    # Terrain is indexed as terrain[y][x]
    terrain = mdp.terrain_mtx
    height  = len(terrain)
    width   = len(terrain[0]) if height else 0

    tx, ty = target_pos

    # Treat the teammate as a temporary blocking cell to avoid collisions.
    other_pos = set()
    for i, p in enumerate(state.players):
        if i != player_idx:
            other_pos.add(p.position)

    px, py = start

    # Fast path: if already adjacent to the target tile,
    # face it first, then interact on the next call.
    for direction, (dx, dy) in DIR_VECTORS.items():
        if (px + dx, py + dy) == (tx, ty):
            curr_orientation = state.players[player_idx].orientation
            if curr_orientation == direction:
                return Action.INTERACT
            else:
                return direction  # rotate now

    # BFS queue stores:
    # ((current_x, current_y), first_action_taken_from_start)
    # We propagate the same first action along each branch.
    queue   = deque()
    visited = {start}

    # Initialize BFS frontier with valid 1-step moves from current position.
    for direction, (dx, dy) in DIR_VECTORS.items():
        nx, ny = px + dx, py + dy
        if (0 <= nx < width and 0 <= ny < height
                and terrain[ny][nx] == ' '
                and (nx, ny) not in other_pos):
            queue.append(((nx, ny), direction))
            visited.add((nx, ny))

    while queue:
        (cx, cy), first_action = queue.popleft()

        # If this reached cell is adjacent to target, we are done:
        # the best immediate move is the branch's first action.
        for direction, (dx, dy) in DIR_VECTORS.items():
            if (cx + dx, cy + dy) == (tx, ty):
                return first_action

        # Expand BFS neighbors through walkable, unvisited, unblocked cells.
        for direction, (dx, dy) in DIR_VECTORS.items():
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < width and 0 <= ny < height
                    and terrain[ny][nx] == ' '
                    and (nx, ny) not in visited
                    and (nx, ny) not in other_pos):
                visited.add((nx, ny))
                queue.append(((nx, ny), first_action))

    return Action.STAY

# Gives the closest objective when there are multiple
def closest(positions, reference):
    """Return the position in `positions` closest (L1) to `reference`."""
    if not positions:
        return None
    rx, ry = reference
    return min(positions, key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))

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

class RandomPartner:
    """Uniformly random actions. Worst-case coordination partner."""

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        return np.random.randint(0, N_ACTIONS), None
    
class StationaryPartner:
    """
    Always stays still.
    Useful to verify the ego agent can still complete tasks alone and to
    stress-test collision / deadlock avoidance.
    """

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        return ACTION_TO_IDX[Action.STAY], None

class GreedyChefAgent:
    """
    BFS-based greedy agent.  Always pursues the single highest-priority task:

        Priority (highest → lowest)
        1. Holding soup  → deliver to serving station
        2. Holding dish  + a pot is ready  → walk to ready pot & pick up soup
        3. Holding onion + pot needs onions → walk to pot & add onion
        4. Nothing held  + pot is ready    → fetch a dish
        5. Nothing held  + pot needs onion → fetch an onion
        6. Otherwise: STAY

    This agent is *greedy* (no look-ahead) so it can deadlock with another
    greedy agent but generally outperforms random play.
    """

    needs_state = True

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        if state is None or mdp is None:
            return np.random.randint(0, N_ACTIONS), None

        player     = state.players[player_idx]
        target_pos = self._choose_target(state, player, mdp)

        if target_pos is None:
            return ACTION_TO_IDX[Action.STAY], None

        action = bfs_next_action(player.position, target_pos, mdp, state, player_idx)
        return ACTION_TO_IDX[action], None

    def _choose_target(self, state, player, mdp):
        held        = player.held_object
        pot_states  = mdp.get_pot_states(state)

        ready_pots  = pot_states.get('ready', []) + pot_states.get('both_ready', [])
        full_pots = pot_states.get('3_items', [])
        empty_pots  = (pot_states.get('empty', [])
                       + pot_states.get('1_items', [])
                       + pot_states.get('2_items', []))

        held_name = held.name if held else None

        # Deliver soup
        if held_name == 'soup':
            serving = mdp.get_serving_locations()
            return closest(serving, player.position)

        # Holding dish + pot ready → go to pot to pick up soup
        if held_name == 'dish' and ready_pots:
            return closest(ready_pots, player.position)

        # Holding onion + pot needs filling
        if held_name == 'onion' and empty_pots:
            return closest(empty_pots, player.position)

        # Pot full + hands free → start cooking
        if held is None and full_pots:
            return closest(full_pots, player.position)

        # Pot ready + hands free → grab a dish first
        if held is None and ready_pots:
            dishes = mdp.get_dish_dispenser_locations()
            return closest(dishes, player.position)
        
        # Pot needs onions + hands free → fetch an onion
        if held is None and empty_pots:
            onions = mdp.get_onion_dispenser_locations()
            return closest(onions, player.position)

        return None

class SpecialistAgent:
    """
    A role-locked agent that performs *only one* half of the pipeline.

    role='fetcher'  → only fetches onions and fills pots; never plates.
    role='plater'   → only fetches dishes, picks up soups, and delivers.

    Using two SpecialistAgents (one per role) is the optimal human strategy,
    so pairing the ego with just one forces it to cover the other role.
    """

    needs_state = True

    def __init__(self, role: str = 'fetcher'):
        assert role in ('fetcher', 'plater'), "role must be 'fetcher' or 'plater'"
        self.role = role

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        if state is None or mdp is None:
            return np.random.randint(0, N_ACTIONS), None

        player     = state.players[player_idx]
        target_pos = self._choose_target(state, player, mdp)

        if target_pos is None:
            return ACTION_TO_IDX[Action.STAY], None

        action = bfs_next_action(player.position, target_pos, mdp, state, player_idx)
        return ACTION_TO_IDX[action], None

    def _choose_target(self, state, player, mdp):
        held       = player.held_object
        held_name  = held.name if held else None
        pot_states = mdp.get_pot_states(state)
        ready_pots = pot_states.get('ready', []) + pot_states.get('both_ready', [])
        full_pots = pot_states.get('3_items', [])
        empty_pots = (pot_states.get('empty', [])
                      + pot_states.get('1_items', [])
                      + pot_states.get('2_items', []))

        if self.role == 'fetcher':
            # Only cares about onions -> pots
            if held_name == 'onion' and empty_pots:
                return closest(empty_pots, player.position)

            if held is None and full_pots:
                return closest(full_pots, player.position)

            if held is None and empty_pots:
                onions = mdp.get_onion_dispenser_locations()
                return closest(onions, player.position)
            
            # If there is nothing useful to do, move away from the pot
            if held is None:
                onions = mdp.get_onion_dispenser_locations()
                return closest(onions, player.position)

            return None

        else:  # 'plater'
            # Deliver soup if carrying it
            if held_name == 'soup':
                serving = mdp.get_serving_locations()
                return closest(serving, player.position)
            # Holding dish -> go to ready pot
            if held_name == 'dish' and ready_pots:
                return closest(ready_pots, player.position)
            # Nothing held + pot ready -> fetch dish
            if held is None and ready_pots:
                dishes = mdp.get_dish_dispenser_locations()
                return closest(dishes, player.position)
            return None

class NoisyGreedyAgent(GreedyChefAgent):
    """
    GreedyChefAgent with ε-random noise injected.

    Simulates a slightly suboptimal human who occasionally fumbles.
    Vary `epsilon` to create a spectrum of difficulty.

        epsilon=0.0  → pure greedy (same as GreedyChefAgent)
        epsilon=0.3  → 30 % random actions
        epsilon=1.0  → same as RandomPartner
    """

    needs_state = True

    def __init__(self, epsilon: float = 0.25):
        super().__init__()
        self.epsilon = epsilon

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, N_ACTIONS), None
        return super().predict(obs, state=state, player_idx=player_idx,
                               mdp=mdp, deterministic=deterministic)

class NoisyPPOPartner:
    """
    Wraps a frozen PPO model and injects action noise.

    Useful for creating a *suboptimal self-play* partner without retraining.
    Pair several of these at different epsilon levels with different early
    checkpoints to build a diverse pool cheaply.

        partner = NoisyPPOPartner(PPO.load("checkpoint_500k"), epsilon=0.2)
    """

    def __init__(self, ppo_model, epsilon: float = 0.15):
        self.model   = ppo_model
        self.epsilon = epsilon

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, N_ACTIONS), None
        action_idx, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action_idx), None

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