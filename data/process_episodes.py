import os
import io
import glob
import json
import pickle
import zipfile
import numpy as np
from luxai_s3.params import EnvParams

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.game import Global
from agent.agent import Agent

MAX_STEPS_IN_MATCH = EnvParams.max_steps_in_match

def get_sample(path, positions, actions, team_id, agent_id_map):
    coordinates = []
    action_types = [] 
    agent_id = agent_id_map[path.split("_")[0]]
    episode = path.split("_")[1]
    flip = team_id == 0
    for (x, y), (action_type, _, _) in zip(positions, actions):
        # Invalid position
        if x == -1 and y == -1: continue
        coordinates.append((x, y))
        action_types.append(action_type)
    return {
        "path": path,
        "coordinates": coordinates,
        "action_types": action_types,
        "agent_id": agent_id,
        "flip": flip,
        "episode": episode,
    }

def get_sap_sample(path, positions, actions, team_id, agent_id_map):
    samples = []
    agent_id = agent_id_map[path.split("_")[0]]
    episode = path.split("_")[1]
    flip = team_id == 0
    for (x, y), (action_type, dx, dy) in zip(positions, actions):
        # Invalid position
        if x == -1 and y == -1: continue
        # Invalid action
        if action_type != 5: continue
        x_target, y_target = x + dx, y + dy
        samples.append({
            "path": path, 
            "x": x, 
            "y": y, 
            "x_target": x_target,
            "y_target": y_target,
            "agent_id": agent_id,
            "flip": flip,
            "episode": episode,
        })
    return samples

def create_dataset(sub_ids, agent_sub_ids, agent_id_map, team_names):

    samples = []
    sap_samples = []
    
    for sub_id, team_name in zip(sub_ids, team_names):
        os.chdir(f"./raw-episodes/{sub_id}")

        zf = zipfile.ZipFile(f"../../dataset/{sub_id}.zip", 'w')
    
        for file in [file for file in glob.glob("*.json") if 'info' not in file]: 
            
            # Get episode_id
            sub_id, episode_id = file.split("_")
            episode_id = episode_id[:-5]
        
            # Load data
            with open(file) as f:
                json_load = json.load(f)

            # Skip sample
            if None in json_load['rewards']:
                continue

            # Get index of winning team 
            tmp_team_name = team_name
            if team_name == 'Frog Parade' and team_name not in json_load['info']['TeamNames']:
                tmp_team_name = team_name
                team_name = 'IsaiahP'
            elif team_name == 'Flat Neurons' and team_name not in json_load['info']['TeamNames']:
                tmp_team_name = team_name
                team_name = 'TonyK'
            team_id = json_load['info']['TeamNames'].index(team_name)
            team_name = tmp_team_name
            other_team_id = 0 if team_id else 1
            other_team_name = json_load['info']['TeamNames'][other_team_id]
            winner_team_id = np.argmax(json_load['rewards'])

            # Skip all all episodes where `team_name` is not the winner
            is_game_lost = team_id != winner_team_id
            if is_game_lost:
                continue

            # Process episdode
            Global.reset()
            player = f"player_{team_id}"
            env_cfg = json_load['configuration']['env_cfg']
            agent = Agent(player, env_cfg, sub_id, agent_sub_ids)
            for step in range(len(json_load['steps'])-1):     
                # This case is triggered, when we play against 'Boey', 'aDg4b' and have won >= 3 games
                if json_load['steps'][step][team_id] == {}:
                    break
                
                # Stop: Game over
                if json_load['steps'][step][team_id]['status'] != 'ACTIVE':
                    break
                
                obs = json.loads(json_load['steps'][step][team_id]['observation']['obs'])
                
                agent.process(step, obs, 60)
                if agent.state.match_step == 0:
                    continue

                if obs['team_wins'][team_id] >= 3 and other_team_name in ['Boey', 'aDg4b']:
                    break

                reward_step = (MAX_STEPS_IN_MATCH + 1) * (agent.state.match + 1)
                reward = json_load['steps'][reward_step-1][agent.team_id]['observation']['reward']
                next_reward = json_load['steps'][reward_step][agent.team_id]['observation']['reward']
                is_match_lost = reward == next_reward
                
                if agent.state.match == 0 and (is_match_lost or is_game_lost):
                    continue
                
                if agent.state.match != 0 and is_match_lost:
                     continue

                path = f"{sub_id}_{episode_id}_{agent.state.match}_{agent.state.match_step}.npz"

                state, _ = agent.get_feature_map()
                global_feats, _ = agent.get_global_features()

                npz_buffer = io.BytesIO()
                np.savez_compressed(
                    npz_buffer,
                    state=state,
                    global_feats=global_feats,
                )

                npz_buffer.seek(0)

                zf.writestr(f'{path}', npz_buffer.read())

                positions = obs["units"]["position"][team_id]
                actions = json_load['steps'][step+1][team_id]['action']
                
                samples.append(get_sample(path, positions, actions, team_id, agent_id_map))
                sap_samples.extend(get_sap_sample(path, positions, actions, team_id, agent_id_map))

        os.chdir(f"../..")

    with open('./dataset/samples.pkl', 'wb') as fp:
        pickle.dump(samples, fp)

    with open('./dataset/sap_samples.pkl', 'wb') as fp:
        pickle.dump(sap_samples, fp)

if __name__ == "__main__":
    
    sub_ids = [
        "43330490", "43330358", "43320130", "43317109", 
        "43294433", "43293811", 
        "43276830", 
        "43212846", "43212163", 
        "42704976"
    ]

    agent_sub_ids = [
        ["43330490", "43330358", "43320130", "43317109"], 
        ["43294433", "43293811"], 
        ["43276830"], 
        ["43212846", "43212163"], 
        ["42704976"]
    ]

    agent_id_map = {
        "43330490": 0, 
        "43330358": 0, 
        "43320130": 0, 
        "43317109": 0,
        "43294433": 1, 
        "43293811": 1,
        "43276830": 2,
        "43212846": 3, 
        "43212163": 3,
        "42704976": 4,
    }

    team_names = [
        "Flat Neurons", "Flat Neurons", "Flat Neurons", "Flat Neurons",
        "Flat Neurons", "Flat Neurons",
        "Frog Parade",
        "Frog Parade", "Frog Parade",
        "Frog Parade",
    ]

    create_dataset(sub_ids, agent_sub_ids, agent_id_map, team_names)
