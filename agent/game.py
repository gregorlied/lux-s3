import numpy as np
from copy import deepcopy
from scipy.signal import convolve2d
from sys import stderr

from .constants import *
from .utils import *


class Global:
    
    ### Exploration flags
    
    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    ENERGY_MOVEMENT_FOUND = False
    OBSTACLE_MOVEMENT_FOUND = False
    OBSTACLE_DIRECTION_FOUND = False

    ### Exploration params

    ENERGY_MOVEMENT_PERIOD = None
    OBSTACLE_MOVEMENT_DIRECTION = None 
    NEBULA_TILE_DRIFT_SPEED = None 

    ### Game logs
    
    REWARD_RESULTS = []
    ENERGY_MOVEMENT_STATUS = []
    OBSTACLES_MOVEMENT_STATUS = []

    @classmethod
    def reset(cls):
        """Reset all class attributes to their default values."""
        cls.ALL_RELICS_FOUND = False
        cls.ALL_REWARDS_FOUND = False
        cls.ENERGY_MOVEMENT_FOUND = False
        cls.OBSTACLE_MOVEMENT_FOUND = False
        cls.OBSTACLE_DIRECTION_FOUND = False

        cls.ENERGY_MOVEMENT_PERIOD = None
        cls.OBSTACLE_MOVEMENT_DIRECTION = None
        cls.NEBULA_TILE_DRIFT_SPEED = None

        cls.REWARD_RESULTS = []
        cls.ENERGY_MOVEMENT_STATUS = []
        cls.OBSTACLES_MOVEMENT_STATUS = []


class GameState:
    def __init__(self):
        self.step = 0
        self.match = 0
        self.match_step = 0
        self.global_step = 0

    def update(self, step):
        self.step = get_step(step)
        self.match = get_match(step)
        self.match_step = get_match_step(step)
        self.global_step = step


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.is_visible = False
        
        self.energy = None
        self.tile_type = UNKNOWN_TILE
        self.prev_tile_type = UNKNOWN_TILE
        self.next_tile_type = UNKNOWN_TILE

        self.is_relic = False
        self.explored_relic = False

        self.is_reward = False
        self.explored_reward = False

    def __repr__(self):
        return f"Node(({self.x}, {self.y}), {self.tile_type})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Space:
    def __init__(self):
        self.nodes = [[Node(x, y) for x in range(SPACE_SIZE)] for y in range(SPACE_SIZE)]
        self.relic_nodes = set()
        self.reward_nodes = set()
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        self.reward_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)

    def __iter__(self):
        for row in self.nodes:
            yield from row

    def get_node(self, x, y):
        return self.nodes[y][x]

    def reset_explored(self, state):
        if state.match >= 3:
            return

        Global.REWARD_RESULTS = []

        for node in self:
            if not node.is_relic:
                node.explored_relic = False
            
            if not node.is_reward:
                node.explored_reward = False
        
    
    def update_map(self, obs, state):

        ### Detect movements
        obstacles_shifted = False
        energy_shifted = False
        for node in self:
            x, y = node.x, node.y
            is_visible = obs["sensor_mask"][x][y]

            if energy_shifted and obstacles_shifted:
                break

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs["map_features"]["energy"][x][y]
            ):
                energy_shifted = True

            if (
                is_visible
                and node.tile_type != UNKNOWN_TILE
                and node.tile_type != obs["map_features"]["tile_type"][x][y]
            ):
                obstacles_shifted = True


        Global.ENERGY_MOVEMENT_STATUS.append(obstacles_shifted)
        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        ### Infer movement params
        if not Global.ENERGY_MOVEMENT_FOUND and energy_shifted:
            pass
        
        if not Global.OBSTACLE_MOVEMENT_FOUND and obstacles_shifted:
            found = False
            for nebula_tile_drift_speed in [0.025, 0.05, 0.1, 0.15]: # 40 20 10 
                val_a = (state.step - 2) * abs(nebula_tile_drift_speed) % 1
                val_b = (state.step - 1) * abs(nebula_tile_drift_speed) % 1
                if val_a > val_b:
                    found = True
                    break
            if found:
                Global.OBSTACLE_MOVEMENT_FOUND = True    
                Global.NEBULA_TILE_DRIFT_SPEED = nebula_tile_drift_speed

        if not Global.OBSTACLE_DIRECTION_FOUND and obstacles_shifted:
            suitable_directions = []
            for (dx, dy) in [(1, -1), (-1, 1)]:
                moved_space = deepcopy(self)
                for node in deepcopy(self):
                    x, y = warp_point(node.x + dx, node.y + dy)
                    moved_space.get_node(x, y).tile_type = node.tile_type
                
                match = True
                for node in moved_space:
                    x, y = node.x, node.y
                    is_visible = obs["sensor_mask"][x][y]
            
                    if (
                        is_visible
                        and node.tile_type != UNKNOWN_TILE
                        and node.tile_type != obs["map_features"]["tile_type"][x][y]
                    ):
                        match = False
                        break

                if match:
                    suitable_directions.append((dx, dy))

            if len(suitable_directions) == 1:
                Global.OBSTACLE_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = suitable_directions[0]
        
        ### Update obstacles
        if (
            Global.OBSTACLE_MOVEMENT_FOUND 
            and Global.OBSTACLE_DIRECTION_FOUND
            and (state.step - 2) * abs(Global.NEBULA_TILE_DRIFT_SPEED) % 1 > (state.step - 1) * abs(Global.NEBULA_TILE_DRIFT_SPEED) % 1
        ):
            dx, dy = Global.OBSTACLE_MOVEMENT_DIRECTION

            for node in deepcopy(self):
                x, y = node.x, node.y
                self.get_node(x, y).prev_tile_type = node.tile_type
            
            for node in deepcopy(self):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).tile_type = node.tile_type
                
            for node in deepcopy(self):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).next_tile_type = node.tile_type

        ### Update map
        for node in self:
            x, y = node.x, node.y
            node.is_visible = obs["sensor_mask"][x][y]

            if node.is_visible:
                node.energy = obs["map_features"]["energy"][x][y]
                self.get_node(*get_opposite(x, y)).energy = node.energy
            
            if node.is_visible and node.tile_type == UNKNOWN_TILE:
                node.tile_type = obs["map_features"]["tile_type"][x][y]
                self.get_node(*get_opposite(x, y)).tile_type = node.tile_type

            elif not node.is_visible and energy_shifted:
                node.energy = None

    def update_exploration_map(self, obs, state, team_id, reward_delta):
        # check for visible relic nodes
        for relic_id, (x, y) in enumerate(obs["relic_nodes"]):
            is_visible = obs["relic_nodes_mask"][relic_id]
            
            if not is_visible:
                continue
            
            self._update_relic_status(x, y, status=True)
        
        # check if the maximum amount of reward nodes has spawened
        match_relic_node_spawned = len(self.relic_nodes) == (2 * min(state.match + 1, 3)) or state.match >= 3 or (state.match < 3 and state.match_step >= 50)
        
        # all other unexplored visible nodes are not a relic node
        for node in self:
            if match_relic_node_spawned and node.is_visible and not node.explored_relic:
                self._update_relic_status(node.x, node.y, status=False)

        # all relics found, mark all nodes as explored for relics
        if len(self.relic_nodes) == (2 * min(state.match + 1, 3)):
            Global.ALL_RELICS_FOUND = True
            for node in self:
                if node.explored_relic: continue
                self._update_relic_status(node.x, node.y, status=False)

        # check if all nodes are explored for rewards
        Global.ALL_REWARDS_FOUND = True
        for node in self:
            if not node.explored_reward:
                Global.ALL_REWARDS_FOUND = False
                break

        # if not all rewards are explored for reward, update rewards
        if match_relic_node_spawned and not Global.ALL_REWARDS_FOUND:
            self._update_reward_status_from_relics_distribution()
            self._update_reward_status_from_reward_results(obs, state, team_id, reward_delta)

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.           
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.is_relic or not node.explored_relic:
                self.relic_map[node.y][node.x] = 1

        self.reward_map = convolve2d(
            self.relic_map,
            np.ones((RELIC_CONFIG_SIZE, RELIC_CONFIG_SIZE), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if self.reward_map[node.y][node.x] == 0: 
                # no relics in range RELIC_REWARD_RANGE
                node.is_reward = False
                node.explored_reward = True

    def _update_reward_status_from_reward_results(self, obs, state, team_id, reward_delta):        
        ship_nodes = set()
        for active, energy, (x, y) in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                node = self.get_node(x, y)
                ship_nodes.add(node)

        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": reward_delta})
            
        for i, result in enumerate(Global.REWARD_RESULTS):

            known_reward = 0
            unknown_nodes = set()
            for node in result["nodes"]:

                if node.is_reward:
                    known_reward += 1

                if node.explored_reward:
                    continue

                unknown_nodes.add(node)

            if not unknown_nodes:
                # all nodes already explored, nothing to do here
                continue

            reward = result["reward"] - known_reward  # reward from unknown_nodes
            
            if reward == 0:
                # all nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(node.x, node.y, status=False)

            elif reward == len(unknown_nodes):
                # all nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(node.x, node.y, status=True)

            elif reward > len(unknown_nodes):
                # we shouldn't be here
                print(
                    f"Something wrong with reward result: {state.step} \n",
                    f"total : {result['reward']} known: {known_reward} unknown: {reward} \n",
                    f"{unknown_nodes} \n {result}",
                    ", this result will be ignored.",
                    file=stderr,
                )
    
    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.is_relic = status
        node.explored_relic = True

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.is_relic = status
        opp_node.explored_relic = True

        if not status:
            return
            
        self.relic_nodes.add(node)
        self.relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.is_reward = status
        node.explored_reward = True

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.is_reward = status
        opp_node.explored_reward = True

        if not status:
            return

        self.reward_nodes.add(node)
        self.reward_nodes.add(opp_node)


class Ship:
    def __init__(self, ship_id):
        self.ship_id = ship_id
        self.reset()

    def __repr__(self):
        return f"Ship({self.ship_id}, {self.node}, {self.energy})"

    def reset(self):
        self.node = None
        self.energy = -1

    def update(self, node, energy):
        self.node = node
        self.energy = energy


class Fleet:
    def __init__(self, team_id):
        self.team_id = team_id
        self.reset()

    def reset(self):
        self.points = 0
        self.reward_deltas = [0]
        self.ships = [Ship(ship_id) for ship_id in range(MAX_UNITS)]

    def update(self, obs, space):
        points = obs["team_points"][self.team_id]
        self.reward_deltas.append(points - self.points)
        self.points = points
        for ship, active, (x, y), energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            node = space.get_node(x, y)
            if active:
                ship.update(node, energy)
            else:
                ship.reset()
