import os
import json
import torch
import numpy as np
from argparse import Namespace

from agent.agent import *
from agent.constants import *
from agent.utils import *
from agent.game import *

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '/kaggle/working/agent'
model = torch.jit.load(f'{path}/model.pth')
model.eval()

model2 = torch.jit.load(f'{path}/model2.pth')
model2.eval()

sap_model = torch.jit.load(f'{path}/sap-model.pth')
sap_model.eval()

sap_model2 = torch.jit.load(f'{path}/sap-model2.pth')
sap_model2.eval()

Global.reset()

### DO NOT REMOVE THE FOLLOWING CODE ###
# store potentially multiple dictionaries as kaggle imports code directly
agent_dict = dict()
agent_prev_obs = dict()


def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_sub_id = "43276830"
        agent_sub_ids = [
            ["43330490", "43330358", "43320130", "43317109"], 
            ["43294433", "43293811"], 
            ["43276830"], 
            ["43212846", "43212163"], 
            ["43277936"]
        ]
        agent_dict[player] = Agent(player, configurations["env_cfg"], agent_sub_id, agent_sub_ids)
        agent_dict[player].add_models(
            model=model,
            sap_model=sap_model,
            model2=model2,
            sap_model2=sap_model2,
        )
    agent = agent_dict[player]
    obs = from_json(obs)
    agent.process(step, obs, remainingOverageTime)
    actions = agent.act(step, obs, remainingOverageTime)
    return dict(action=actions.tolist())


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        observation = Namespace(
            **dict(
                step=raw_input["step"],
                obs=raw_input["obs"],
                remainingOverageTime=raw_input["remainingOverageTime"],
                player=raw_input["player"],
                info=raw_input["info"],
            )
        )
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))
