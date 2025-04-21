import os
import numpy as np
from scipy.signal import convolve2d
from torch.utils.data import Dataset
from luxai_s3.params import EnvParams, env_params_ranges

SPACE_SIZE = EnvParams.map_width
MIN_UNIT_SAP_RANGE = env_params_ranges["unit_sap_range"][0]
MAX_UNIT_SAP_RANGE = env_params_ranges["unit_sap_range"][-1]


def get_state(state, global_feats):
    num_features = global_feats.shape[0]
    feature_map = np.zeros((num_features, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    feature_map[:] = global_feats[:, None, None]

    state = np.concatenate((state, feature_map), axis=0)
    return state


def get_unit_state(x, y, state, global_feats):
    num_features = global_feats.shape[0]

    unit_position = np.zeros((1, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    unit_position[0, y, x] = 1

    my_robot_position = unit_position[0]
    unit_sap_range = int(global_feats[5] * (MAX_UNIT_SAP_RANGE - MIN_UNIT_SAP_RANGE) + MIN_UNIT_SAP_RANGE)
    sap_range_filter = np.ones((2 * unit_sap_range + 1, 2 * unit_sap_range + 1), dtype=np.int32)
    action_mask = convolve2d(my_robot_position, sap_range_filter, mode="same", boundary="fill", fillvalue=0)

    sap_range_map = np.zeros((1, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    sap_range_map[0] = action_mask

    feature_map = np.zeros((num_features, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    feature_map[:] = global_feats[:, None, None]

    state = np.concatenate((unit_position, sap_range_map, state, feature_map), axis=0)
    return state, action_mask


def get_action(coordinates, action_types):
    action = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
    action_mask = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)

    x_coords, y_coords = zip(*coordinates)
    action[y_coords, x_coords] = action_types
    action_mask[y_coords, x_coords] = 1

    return action, action_mask


def get_sap_action(targets):
    action = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)

    if targets:
        x_coords, y_coords = zip(*targets)
        action[y_coords, x_coords] = 1
        
    return action


class LuxFlipTransform:
    def __init__(self):
        self.one_hot = np.eye(6)
        self.permute = [0,2,1,4,3,5]

    def __call__(self, sample):
        state, action, action_mask, flip = sample

        if flip:
            state = state[:, ::-1, ::-1].transpose(0, 2, 1).copy()
            action = self.one_hot[action][:, :, self.permute].argmax(-1)
            action = np.flip(action.T, axis=(0, 1)).copy()
            action_mask = np.flip(action_mask.T, axis=(0, 1)).copy()

        return state, action, action_mask


class LuxDataset(Dataset):
    def __init__(self, data_dir, samples, transform=None):
        self.data_dir = data_dir
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, coordinates, action_types, agent_id, flip = self.samples[idx]

        data_path = os.path.join(self.data_dir, path)
        data = np.load(data_path)

        state, global_feats = data['state'], data['global_feats']
        state = get_state(state, global_feats)
        action, action_mask = get_action(coordinates, action_types)

        if self.transform is not None:
            state, action, action_mask = self.transform((state, action, action_mask, flip))

        return state, agent_id, action, action_mask
    

class LuxSapDataset(Dataset):
    def __init__(self, data_dir, samples, transform=None):
        self.data_dir = data_dir
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x, y, x_target, y_target, agent_id, flip = self.samples[idx]

        data_path = os.path.join(self.data_dir, path)
        data = np.load(data_path)

        state, action_mask = get_unit_state(x, y, data['state'], data['global_feats'])
        targets = [(x_target, y_target)] if x_target != -100 else []
        action = get_sap_action(targets)

        if self.transform is not None:
            state, action, action_mask = self.transform((state, action, action_mask, flip))

        return state, agent_id, action, action_mask
