import torch
from luxai_s3.params import EnvParams

from scipy.special import softmax
from scipy.signal import convolve2d

from luxai_s3.state import (
    EMPTY_TILE,    # 0
    NEBULA_TILE,   # 1
    ASTEROID_TILE, # 2
)

import math
from collections import defaultdict

from .constants import *
from .utils import *
from .game import *


DIRECTIONS = [
    (0, 0),  # Do nothing
    (0, -1), # Move up
    (1, 0),  # Move right
    (0, 1),  # Move down
    (-1, 0), # Move left
    (0, 0),  # Sap
]


class Agent:
    def __init__(self, player, env_cfg, agent_sub_id, agent_sub_ids):
        self.env_cfg = env_cfg
        self.flip = player == "player_0"
        self.team_id = 0 if player == "player_0" else 1
        self.other_team_id = 1 if self.team_id == 0 else 0

        self.state = GameState()
        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.other_fleet = Fleet(self.other_team_id)

        self.agent_id = max([i if agent_sub_id in xs else -100 for i, xs in enumerate(agent_sub_ids)])
        assert self.agent_id >= 0
        self.agent_sub_id = agent_sub_id
        self.agent_sub_ids = agent_sub_ids

    def add_models(self, model, sap_model, model2=None, sap_model2=None):
        self.model = model
        self.sap_model = sap_model
        self.model2 = model2
        self.sap_model2 = sap_model2

    def process(self, step, obs, remainingOverageTime):
        self.state.update(step)
        
        if self.state.match_step == 0: 
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous matc
            self.space.reset_explored(self.state)
            self.space.update_map(obs, self.state)
            self.fleet.reset()
            self.other_fleet.reset()
            return

        # how many points did we score in the last step
        points = obs["team_points"][self.team_id]
        reward_delta = max(0, points - self.fleet.points)

        self.space.update_map(obs, self.state)
        self.space.update_exploration_map(obs, self.state, self.team_id, reward_delta)
        self.fleet.update(obs, self.space)
        self.other_fleet.update(obs, self.space)
    
    @torch.no_grad()
    def act(self, step, obs, remainingOverageTime):
        actions = np.zeros((MAX_UNITS, 3), dtype=int)
        
        if self.state.match_step == 0:
            return actions

        feature_map, _ = self.get_feature_map()
        global_feature_map, _ = self.get_global_feature_map()

        policy = self.get_policy(obs, feature_map, global_feature_map)

        other_positons = []
        other_actions = defaultdict(list)
        for i, (x, y) in enumerate(obs["units"]["position"][self.team_id]):
            # Invalid position
            if x == -1 and y == -1: continue
            unit_policy = policy[:, y, x]
            
            actions[i, :] = self.get_action(unit_policy, x, y, feature_map, global_feature_map, other_positons, other_actions)
            action = actions[i, 0]
            
            is_start_player_0 = x == 0 and y == 0
            is_start_player_1 = x == (SPACE_SIZE - 1) and y == (SPACE_SIZE - 1)
            if action == 0 and (is_start_player_0 or is_start_player_1):
                continue
                                       
            other_actions[(x, y)].append(action)
            
            dx, dy = DIRECTIONS[action]
            other_positons.append((x + dx, y + dy))
        
        return actions

    def get_dist_from_center_x(self, map_height: int, map_width: int) -> np.ndarray:
        pos = np.linspace(0, 2, map_width, dtype=np.float32)[None, :].repeat(map_height, axis=0)
        return np.abs(1 - pos)[None, None, :, :]

    def get_dist_from_center_y(self, map_height: int, map_width: int) -> np.ndarray:
        pos = np.linspace(0, 2, map_height)[:, None].repeat(map_width, axis=1)
        return np.abs(1 - pos)[None, None, :, :]

    def get_feature_map(self):
        env_cfg = self.env_cfg
        
        num_features = 18
        feature_names = []
        feature_map = np.zeros((num_features, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    
        # ================ MY TEAM - FEATURES ================
        # - my_unit_positions {0, 1}
        # - my_unit_energy [0, 1]
        # - my_unit_sap_range_map {0, 1}
        feature_names += ["my_unit_positions", "my_unit_energy"]
        for ship in self.fleet.ships:
            if ship.node:
                x, y = ship.node.x, ship.node.y
                energy = ship.energy
                feature_map[0, y, x] = 1
                feature_map[1, y, x] = energy / EnvParams.max_unit_energy

        feature_names += ["my_unit_sap_range_map"]
        my_robot_position = feature_map[0]
        sap_range_filter = np.ones((2 * env_cfg["unit_sap_range"] + 1, 2 * env_cfg["unit_sap_range"] + 1), dtype=np.int32)
        sap_range_map = convolve2d(my_robot_position, sap_range_filter, mode="same", boundary="fill", fillvalue=0) 
        sap_range_map[sap_range_map > 0] = 1
        feature_map[2, :] = sap_range_map

        # ================ OPP TEAM - FEATURES ================
        # - opp_unit_positions {0, 1}
        # - opp_unit_energy [0, 1]
        feature_names += ["opp_unit_positions", "opp_unit_energy"]
        for ship in self.other_fleet.ships:
            if ship.node:
                x, y = ship.node.x, ship.node.y
                energy = ship.energy
                feature_map[3, y, x] = 1
                feature_map[4, y, x] = energy / EnvParams.max_unit_energy

        # ================ NODE FEATURES ================
        # is_visible {0, 1}
        # energy -1 or [0, 1]
        # curr_tile_type_empty {-1, 0, 1}
        # curr_tile_type_nebula {-1, 0, 1}
        # curr_tile_type_asteroid {-1, 0, 1}
        # next_tile_type_empty {-1, 0, 1}
        # next_tile_type_nebula {-1, 0, 1}
        # next_tile_type_asteroid {-1, 0, 1}
        feature_names += ["is_visible", "energy", "curr_tile_type_empty", "curr_tile_type_nebula", "curr_tile_type_asteroid"]
        for node in self.space:
            x, y = node.x, node.y

            if node.is_visible:
                feature_map[5, y, x] = 1 
            else: 
                feature_map[5, y, x] = 0
            
            if node.energy:
                feature_map[6, y, x] = (node.energy - MIN_ENERGY_PER_TILE) / (MAX_ENERGY_PER_TILE - MIN_ENERGY_PER_TILE)
            else:
                feature_map[6, y, x] = -1

            if node.tile_type == UNKNOWN_TILE:
                feature_map[7, y, x] = -1
                feature_map[8, y, x] = -1
                feature_map[9, y, x] = -1
            elif node.tile_type == EMPTY_TILE:
                feature_map[7, y, x] = 1
                feature_map[8, y, x] = 0
                feature_map[9, y, x] = 0
            elif node.tile_type == NEBULA_TILE:
                feature_map[7, y, x] = 0
                feature_map[8, y, x] = 1
                feature_map[9, y, x] = 0
            elif node.tile_type == ASTEROID_TILE:
                feature_map[7, y, x] = 0
                feature_map[8, y, x] = 0
                feature_map[9, y, x] = 1

        feature_names += ["next_tile_type_empty", "next_tile_type_nebula", "next_tile_type_asteroid"]
        if Global.OBSTACLE_MOVEMENT_FOUND and Global.OBSTACLE_DIRECTION_FOUND:
            for node in self.space:
                x, y = node.x, node.y
                
                if node.next_tile_type == UNKNOWN_TILE:
                    feature_map[10, y, x] = -1
                    feature_map[11, y, x] = -1
                    feature_map[12, y, x] = -1
                elif node.next_tile_type == EMPTY_TILE:
                    feature_map[10, y, x] = 1
                    feature_map[11, y, x] = 0
                    feature_map[12, y, x] = 0
                elif node.next_tile_type == NEBULA_TILE:
                    feature_map[10, y, x] = 0
                    feature_map[11, y, x] = 1
                    feature_map[12, y, x] = 0
                elif node.next_tile_type == ASTEROID_TILE:
                    feature_map[10, y, x] = 0
                    feature_map[11, y, x] = 0
                    feature_map[12, y, x] = 1
        else:
            feature_map[10, :] = -1
            feature_map[11, :] = -1
            feature_map[12, :] = -1

        # ================ DISTANCE FEATURES ================
        # dist_from_center_x [0, 1]
        # dist_from_center_y [0, 1]
        feature_names += ['dist_from_center_x', 'dist_from_center_y']
        feature_map[13, :] = self.get_dist_from_center_x(SPACE_SIZE, SPACE_SIZE)
        feature_map[14, :] = self.get_dist_from_center_y(SPACE_SIZE, SPACE_SIZE)

        # ================ RELIC FEATURES ================
        # relic_nodes {-1, 0, 1}
        # reward_nodes {-1, 0, 1}
        # reward_map [0, 1]
        feature_names += ["relic_nodes", "reward_nodes", "reward_map"]
        for node in self.space:
            x, y = node.x, node.y

            if not node.explored_relic:
                feature_map[15, y, x] = -1
            else:
                feature_map[15, y, x] = node.is_relic

            if not node.explored_reward:
                feature_map[16, y, x] = -1
            else:
                feature_map[16, y, x] = node.is_reward

        feature_map[17, :] = self.space.reward_map / MAX_RELIC_NODES    
        
        return feature_map, feature_names

    def get_global_features(self):
        env_cfg = self.env_cfg
        nebula_tile_drift_speeds = [-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]
        
        num_features = 19 + len(self.agent_sub_ids) + len(nebula_tile_drift_speeds)
        global_feature_names = []
        global_features = np.zeros((num_features), dtype=np.float32)

        # ================ GAME STATS ================
        # - steps [0, 1]
        global_feature_names += ["steps"] 
        global_features[0] = self.state.step / (MATCH_COUNT_PER_EPISODE * MAX_STEPS_IN_MATCH)
        # - match_steps [0, 1]
        global_feature_names += ["match_steps"]
        global_features[1] = self.state.match_step / MAX_STEPS_IN_MATCH
        # - next_map_update -1 or [0, 1]
        global_feature_names += ["next_map_update"]
        if Global.OBSTACLE_MOVEMENT_FOUND and Global.OBSTACLE_DIRECTION_FOUND:
            max_steps = math.ceil(1 / Global.NEBULA_TILE_DRIFT_SPEED)
            steps_until = None
            for i, step in enumerate(range(self.state.step, self.state.step + max_steps)):
                if (step - 2) * Global.NEBULA_TILE_DRIFT_SPEED % 1 > (step - 1) * Global.NEBULA_TILE_DRIFT_SPEED % 1:
                    steps_until = i
                    break
            global_features[2] = steps_until / max_steps
        else:
            global_features[2] = -1
        # unit_move_cost [0, 1]
        global_feature_names += ["unit_move_cost"]
        global_features[3] = (env_cfg["unit_move_cost"] - MIN_UNIT_MOVE_COST) / (MAX_UNIT_MOVE_COST - MIN_UNIT_MOVE_COST)
        # unit_sap_cost [0, 1]
        global_feature_names += ["unit_sap_cost"]
        global_features[4] = (env_cfg["unit_sap_cost"] - MIN_UNIT_SAP_COST) / (MAX_UNIT_SAP_COST - MIN_UNIT_SAP_COST)
        # unit_sap_range [0, 1]
        global_feature_names += ["unit_sap_range"]
        global_features[5] = (env_cfg["unit_sap_range"] - MIN_UNIT_SAP_RANGE) / (MAX_UNIT_SAP_RANGE - MIN_UNIT_SAP_RANGE)
        # unit_sensor_range [0, 1]
        global_feature_names += ["unit_sensor_range"]
        global_features[6] = (env_cfg["unit_sensor_range"] - MIN_UNIT_SENSOR_RANGE) / (MAX_UNIT_SENSOR_RANGE - MIN_UNIT_SENSOR_RANGE)

        # ================ REWARD INFORMATION ================
        # - all_relics_found {0, 1}
        global_feature_names += ["all_relics_found"]
        global_features[7] = Global.ALL_RELICS_FOUND
        # - all_rewards_found {0, 1}
        global_feature_names += ["all_rewards_found"]
        global_features[8] = Global.ALL_REWARDS_FOUND
        # - gained_reward_last_1 {0, 1}
        global_feature_names += ["gained_reward_last_1"]
        global_features[9] = self.gained_reward_last_x(1)
        # - gained_reward_last_3 {0, 1}
        global_feature_names += ["gained_reward_last_3"]
        global_features[10] = self.gained_reward_last_x(3)
        # - gained_reward_last_5 {0, 1}
        global_feature_names += ["gained_reward_last_5"]
        global_features[11] = self.gained_reward_last_x(5)
        # - gained_reward_last_10 {0, 1}
        global_feature_names += ["gained_reward_last_10"]
        global_features[12] = self.gained_reward_last_x(10)
        # - reward_last_1 [0, 1]
        global_feature_names += ["reward_last_1"]
        global_features[13] = self.get_reward_last_x(1)
        # - reward_last_3 [0, 1]
        global_feature_names += ["reward_last_3"]
        global_features[14] = self.get_reward_last_x(3)
        # - reward_last_5 [0, 1]
        global_feature_names += ["reward_last_5"]
        global_features[15] = self.get_reward_last_x(5)
        # - reward_last_10 [0, 1]
        global_feature_names += ["reward_last_10"]
        global_features[16] = self.get_reward_last_x(10)
        # team_points [0, 1]
        global_feature_names += ["team_points"]
        global_features[17] = self.fleet.points / 500
        # opp_team_points [0, 1]
        global_feature_names += ["opp_team_points"]
        global_features[18] = self.other_fleet.points / 500

        # agent_id
        for offset, (sub_id) in enumerate(self.agent_sub_ids):
            global_feature_names += [f"agent_{sub_id}"]
            global_features[19 + offset] = 1 if self.agent_sub_id in sub_id else 0

        # nebula_tile_drift_speed
        if Global.OBSTACLE_MOVEMENT_FOUND and Global.OBSTACLE_DIRECTION_FOUND:
            game_speed = Global.OBSTACLE_MOVEMENT_DIRECTION[0] * round(Global.NEBULA_TILE_DRIFT_SPEED, 2)
            for offset, speed in enumerate(nebula_tile_drift_speeds):
                global_feature_names += [f"nebula_tile_drift_speed_{speed}"]
                global_features[num_features - len(nebula_tile_drift_speeds) + offset] = 1 if game_speed == speed else 0
        
        return global_features, global_feature_names

    def get_global_feature_map(self):

        global_features, global_feature_names = self.get_global_features()
        
        num_features = global_features.shape[0]
        global_feature_map = np.zeros((num_features, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)

        global_feature_map[:] = global_features[:, None, None]

        return global_feature_map, global_feature_names

    def get_reward_last_x(self, x):
        x = min(x, len(self.fleet.reward_deltas))
        fleet_points = sum([self.fleet.reward_deltas[-i] for i in range(1, x + 1)])
        other_fleet_points = sum([self.other_fleet.reward_deltas[-i] for i in range(1, x + 1)])
        total_points = fleet_points + other_fleet_points
        return (fleet_points / total_points) if total_points else -1

    def gained_reward_last_x(self, x):
        x = min(x, len(self.fleet.reward_deltas))
        return any([self.fleet.reward_deltas[-i] > 0 for i in range(1, x  + 1)])

    def get_policy(self, obs, feature_map, global_feature_map):

        mask = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
        x_coords, y_coords = zip(*obs["units"]["position"][self.team_id])
        mask[y_coords, x_coords] = 1
        mask = mask[None, :]
        if self.flip:
            mask = mask[:, ::-1, ::-1].transpose(0, 2, 1).copy()
            mask = torch.from_numpy(mask).long()
        else:
            mask = torch.from_numpy(mask).long()
        
        state = np.concatenate((feature_map, global_feature_map), axis=0)
        if self.flip:
            mirror_state = state[:, ::-1, ::-1].transpose(0, 2, 1).copy()
            mirror_state = torch.from_numpy(mirror_state).unsqueeze(0)
        else:
            state = torch.from_numpy(state).unsqueeze(0)
        
        if self.flip:
            policy = self.model(mirror_state, torch.tensor([self.agent_id]).long())
            policy = policy.squeeze(0).numpy()
            policy = policy[:, ::-1, ::-1].transpose(0, 2, 1).copy()
            policy = policy[[0,2,1,4,3,5]]
        else:
            policy = self.model(state, torch.tensor([self.agent_id]).long())
            policy = policy.squeeze(0).numpy()

        if self.model2 is None:
            return policy

        if self.flip:
            policy2 = self.model2(mirror_state, mask, torch.tensor([self.agent_id]).long())
            policy2 = policy2.squeeze(0).numpy()
            policy2 = policy2[:, ::-1, ::-1].transpose(0, 2, 1).copy()
            policy2 = policy2[[0,2,1,4,3,5]]
        else:
            policy2 = self.model2(state, mask, torch.tensor([self.agent_id]).long())
            policy2 = policy2.squeeze(0).numpy()

        policy = (policy + policy2) / 2

        return policy
    
    def get_sap_policy(self, x, y, feature_map, global_feature_map):
        env_cfg = self.env_cfg
        
        unit_position = np.zeros((1, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        unit_position[0, y, x] = 1

        sap_range_map = np.zeros((1, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        sap_range_filter = np.ones((2 * env_cfg["unit_sap_range"] + 1, 2 * env_cfg["unit_sap_range"] + 1), dtype=np.int32)
        sap_range_map[0] = convolve2d(unit_position[0], sap_range_filter, mode="same", boundary="fill", fillvalue=0)
            
        sap_state = np.concatenate((unit_position, sap_range_map, feature_map, global_feature_map), axis=0)
        
        sap_state = torch.from_numpy(sap_state).unsqueeze(0)
        sap_policy = self.sap_model(sap_state, torch.tensor([self.agent_id]).long())
        sap_policy = sap_policy.squeeze(0).numpy()

        if self.sap_model2 is None:
            return sap_policy, sap_range_map[0]

        sap_policy2 = self.sap_model2(sap_state, torch.tensor([self.agent_id]).long())
        sap_policy2 = sap_policy2.squeeze(0).numpy()

        sap_policy = (sap_policy + sap_policy2) / 2

        return sap_policy, sap_range_map[0]

    def get_action(self, unit_policy, x, y, feature_map, global_feature_map, other_positions, other_actions):
        unit_policy = softmax(unit_policy, axis=0)
        for action in np.argsort(unit_policy)[::-1]:
            dx, dy = DIRECTIONS[action]

            # Don't perform bad action
            if unit_policy[action] < 0.15:
                break

            # Can't move OOB
            if not ((0 <= x + dx < SPACE_SIZE) and (0 <= y + dy < SPACE_SIZE)):
                continue

            # Avoid stacking
            if (x + dx, y + dy) in other_positions:
                continue

            # Don't do an action that is performed already, since we don't want stacked ships
            if action != 5 and action in other_actions[(x, y)]:
                continue
            
            # Don't move to asteroid tile
            node = self.space.get_node(x + dx, y + dy)
            if (action in [1, 2, 3, 4] or not node.is_reward) and node.tile_type == ASTEROID_TILE:
                continue

            # Do sap action
            if action == 5:

                sap_policy, sap_range_map = self.get_sap_policy(x, y, feature_map, global_feature_map)
                
                sap_policy = softmax(sap_policy, axis=0)
                targets = [(x, y) for y, x in np.argwhere(sap_policy[1] > sap_policy[0])]
                sap_range = [(x, y) for y, x in np.argwhere(sap_range_map > 0)]
                targets = [target for target in targets if target in sap_range]
                
                # No targets -> skip
                if len(targets) == 0: 
                    continue
                    
                # Find nearest target
                (x_target, y_target), _ = find_closest_target((x, y), targets)

                # Get relative position w.r.t current node
                dx, dy = x_target - x, y_target - y

            return np.array([action, dx, dy])

        return np.array([0, 0, 0])
