import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.stats as st

from recommendation import RecommendationEngine


def get_optimal_approx_percent(approx_dict, approx_steps, solver, lower_bound):
    optimization_array = np.ones(approx_steps)
    for i in range(approx_steps):
        step = i + 1
        quality = approx_dict[str(step)][solver]
        percent = step / (approx_steps + 1)
        if quality > lower_bound:
            optimization_array[i] = (1 - quality) + (1 - percent)
    return (np.max(np.argmin(optimization_array)) + 1) / approx_steps


class DatabaseSetup:
    def __init__(self):
        self.db = RecommendationEngine().get_database()

    def get_data_for_simple_learning(self, problem_number, solver, lower_bound):
        X_classes = []
        Y_target = []
        for _, metadata in self.db.iter_metadata():
            # print(metadata)
            problem_one_hot = np.zeros(problem_number)
            problem_one_hot[metadata.problem] = 1
            X_classes.append(problem_one_hot)
            Y_target.append(
                get_optimal_approx_percent(metadata.approx_solution_quality, metadata.approx, solver, lower_bound))
        return np.array(X_classes), np.array(Y_target)


class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
