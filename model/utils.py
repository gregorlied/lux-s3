import os
import yaml
import random
import torch
import numpy as np


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class Accuracy:
    def __init__(self, num_agents, num_classes):
        self.num_agents = num_agents
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.per_class_n_samples = [0 for _ in range(self.num_classes)]
        self.per_class_n_correctly_classified = [0 for _ in range(self.num_classes)]
        self.per_agent_per_class_n_samples = {i: [0 for _ in range(self.num_classes)] for i in range(self.num_agents)}
        self.per_agent_per_class_n_correctly_classified = {i: [0 for _ in range(self.num_classes)] for i in range(self.num_agents)}

    def update(self, policy, actions, masks, agent_ids):
        preds = policy.argmax(1)
        for index in range(self.num_classes):
            self.per_class_n_samples[index] += ((actions == index) * masks).sum().item()
            self.per_class_n_correctly_classified[index] += ((preds == actions) * (actions == index) * masks).sum().item()

        for agent in range(self.num_agents):
            agent_mask = (agent_ids == agent).reshape(-1, 1, 1)
            for index in range(self.num_classes):
                self.per_agent_per_class_n_samples[agent][index] += ((actions == index) * masks * agent_mask).sum().item()
                self.per_agent_per_class_n_correctly_classified[agent][index] += ((preds == actions) * (actions == index) * masks * agent_mask).sum().item()

    def print(self):
        accuracy = sum(self.per_class_n_correctly_classified) / sum(self.per_class_n_samples)
        print(f" === Overall === ")
        print(f'Accuracy {accuracy:.4f}')
        for index in range(self.num_classes):
            per_class_accuracy = self.per_class_n_correctly_classified[index] / self.per_class_n_samples[index]
            print(f'Class {index} {per_class_accuracy:.4f}')
        for agent in range(self.num_agents):
            print(f" === Agent ID {agent} === ")
            accuracy = sum(self.per_agent_per_class_n_correctly_classified[agent]) / sum(self.per_agent_per_class_n_samples[agent])
            print(f'Accuracy {accuracy:.4f}')
            for index in range(self.num_classes):
                per_class_accuracy = self.per_agent_per_class_n_correctly_classified[agent][index] / self.per_agent_per_class_n_samples[agent][index]
                print(f'Class {index} {per_class_accuracy:.4f}')
