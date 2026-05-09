"""
partner_agents.py
-----------------
Diverse partner agents for Overcooked Ad Hoc Teamwork project.

Each agent implements:
    predict(obs, state=None, player_idx=None, mdp=None, deterministic=False)
             -> (action_idx: int, None)

The `state`, `player_idx`, and `mdp` kwargs are only used by heuristic agents.
The observation-only interface (RandomPartner, NoisyPPOPartner) remains
compatible with the original wrapper without any changes.

To enable state-based agents, apply the small patch shown at the bottom of
this file to OvercookedSelfPlayWrapper.step().
"""

import numpy as np
from collections import deque
from overcooked_ai_py.mdp.actions import Action, Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTION_TO_IDX = {a: i for i, a in enumerate(Action.ALL_ACTIONS)}
N_ACTIONS     = len(Action.ALL_ACTIONS)

# Overcooked direction vectors (x-right, y-down)
DIR_VECTORS = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0,  1),
    Direction.EAST:  (1,  0),
    Direction.WEST:  (-1, 0),
}


#Usada para criar uma trajetoria de acoes para os agentes que seguem uma heuristica
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

    # No route to a tile adjacent to target.
    return Action.STAY

#Gives the closest objective when there are multiple
def closest(positions, reference):
    """Return the position in `positions` closest (L1) to `reference`."""
    if not positions:
        return None
    rx, ry = reference
    return min(positions, key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))


# ---------------------------------------------------------------------------
# 1. RandomPartner  (already in baseline – reproduced here for completeness)
# ---------------------------------------------------------------------------

class RandomPartner:
    """Uniformly random actions. Worst-case coordination partner."""

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        return np.random.randint(0, N_ACTIONS), None


# ---------------------------------------------------------------------------
# 2. StationaryPartner
# ---------------------------------------------------------------------------

class StationaryPartner:
    """
    Always stays still.
    Useful to verify the ego agent can still complete tasks alone and to
    stress-test collision / deadlock avoidance.
    """

    def predict(self, obs, state=None, player_idx=None, mdp=None,
                deterministic=False):
        return ACTION_TO_IDX[Action.STAY], None


# ---------------------------------------------------------------------------
# 3. GreedyChefAgent
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    def _choose_target(self, state, player, mdp):
        held        = player.held_object
        pot_states  = mdp.get_pot_states(state)

        ready_pots  = pot_states.get('ready', []) + pot_states.get('both_ready', [])
        full_pots = pot_states.get('3_items', [])
        empty_pots  = (pot_states.get('empty', [])
                       + pot_states.get('1_items', [])
                       + pot_states.get('2_items', []))

        held_name = held.name if held else None

        # 1. Deliver soup
        if held_name == 'soup':
            serving = mdp.get_serving_locations()
            return closest(serving, player.position)

        # 2. Holding dish + pot ready → go to pot to pick up soup
        if held_name == 'dish' and ready_pots:
            return closest(ready_pots, player.position)

        # 3. Holding onion + pot needs filling
        if held_name == 'onion' and empty_pots:
            return closest(empty_pots, player.position)

        # 4. Pot full + hands free → start cooking
        if held is None and full_pots:
            return closest(full_pots, player.position)

        # 5. Pot ready + hands free → grab a dish first
        if held is None and ready_pots:
            dishes = mdp.get_dish_dispenser_locations()
            return closest(dishes, player.position)
        
        # 6. Pot needs onions + hands free → fetch an onion
        if held is None and empty_pots:
            onions = mdp.get_onion_dispenser_locations()
            return closest(onions, player.position)

        return None


# ---------------------------------------------------------------------------
# 4. SpecialistAgent
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
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
            # Only cares about onions → pots
            if held_name == 'onion' and empty_pots:
                return closest(empty_pots, player.position)

            if held is None and full_pots:
                return closest(full_pots, player.position)

            if held is None and empty_pots:
                onions = mdp.get_onion_dispenser_locations()
                return closest(onions, player.position)

            return None

        else:  # 'plater'
            # Deliver soup if carrying it
            if held_name == 'soup':
                serving = mdp.get_serving_locations()
                return closest(serving, player.position)
            # Holding dish → go to ready pot
            if held_name == 'dish' and ready_pots:
                return closest(ready_pots, player.position)
            # Nothing held + pot ready → fetch dish
            if held is None and ready_pots:
                dishes = mdp.get_dish_dispenser_locations()
                return closest(dishes, player.position)
            return None


# ---------------------------------------------------------------------------
# 5. NoisyGreedyAgent
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. NoisyPPOPartner  (wraps a pre-trained PPO checkpoint)
# ---------------------------------------------------------------------------

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

