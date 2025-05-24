import os, sys
import time
import random
import math
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from scipy.stats import ttest_rel


# 所有VRP变体
VRP_VARIANTS = [
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"
]

# Define the list of non-TW VRP variants
ALL_NO_TW_VARIANTS = [
    pt for pt in VRP_VARIANTS if 'TW' not in pt
]

# 每种变体的问题数据包含的内容 - 用于解析输入元组
# 注意：顺序很重要！
VRP_DATA_FORMAT = {
    "CVRP": ["depot_xy", "node_xy", "demand", "capacity"],
    "OVRP": ["depot_xy", "node_xy", "demand", "capacity"],
    "VRPB": ["depot_xy", "node_xy", "demand", "capacity"], # demand distinguishes linehaul/backhaul
    "OVRPB": ["depot_xy", "node_xy", "demand", "capacity"],
    "VRPTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "OVRPTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "VRPL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "OVRPL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "VRPBL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "OVRPBL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "VRPBTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "OVRPBTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "VRPLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "OVRPLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "VRPBLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "OVRPBLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"]
}

# --- Dataset Paths Table ---
# Maps (problem_type, num_nodes) to main data file and optional solution file.
# Ensure these paths are correct for your environment.
DATASET_PATHS = {
    "CVRP": {
        20: {"data": "data/CVRP/cvrp20_uniform.pkl", "solution": "data/CVRP/or_tools_10s_cvrp20_uniform.pkl"},
        50: {"data": "data/CVRP/cvrp50_uniform.pkl", "solution": "data/CVRP/hgs_cvrp50_uniform.pkl"},
        100: {"data": "data/CVRP/cvrp100_uniform.pkl", "solution": "data/CVRP/hgs_cvrp100_uniform.pkl"},
        200: {"data": "data/CVRP/cvrp200_uniform.pkl", "solution": "data/CVRP/or_tools_1s_cvrp200_uniform.pkl"},
        500: {"data": "data/CVRP/cvrp500_uniform.pkl", "solution": "data/CVRP/or_tools_5s_cvrp500_uniform.pkl"},
        1000: {"data": "data/CVRP/cvrp1000_uniform.pkl", "solution": "data/CVRP/or_tools_10s_cvrp1000_uniform.pkl"},
        2000: {"data": "data/CVRP/cvrp2000_uniform.pkl"},
        5000: {"data": "data/CVRP/cvrp5000_uniform.pkl"},
    },
    "OVRP": {
        20: {"data": "data/OVRP/ovrp20_uniform.pkl", "solution": "data/OVRP/or_tools_10s_ovrp20_uniform.pkl"},
        50: {"data": "data/OVRP/ovrp50_uniform.pkl", "solution": "data/OVRP/or_tools_200s_ovrp50_uniform.pkl"},
        100: {"data": "data/OVRP/ovrp100_uniform.pkl", "solution": "data/OVRP/lkh_ovrp100_uniform.pkl"},
        200: {"data": "data/OVRP/ovrp200_uniform.pkl"},
        500: {"data": "data/OVRP/ovrp500_uniform.pkl"},
        1000: {"data": "data/OVRP/ovrp1000_uniform.pkl"},
        2000: {"data": "data/OVRP/ovrp2000_uniform.pkl"},
        5000: {"data": "data/OVRP/ovrp5000_uniform.pkl"},
    },
    "VRPB": {
        20: {"data": "data/VRPB/vrpb20_uniform.pkl", "solution": "data/VRPB/or_tools_10s_vrpb20_uniform.pkl"},
        50: {"data": "data/VRPB/vrpb50_uniform.pkl", "solution": "data/VRPB/or_tools_200s_vrpb50_uniform.pkl"},
        100: {"data": "data/VRPB/vrpb100_uniform.pkl", "solution": "data/VRPB/or_tools_400s_vrpb100_uniform.pkl"},
        200: {"data": "data/VRPB/vrpb200_uniform.pkl"},
        500: {"data": "data/VRPB/vrpb500_uniform.pkl"},
        1000: {"data": "data/VRPB/vrpb1000_uniform.pkl"},
        2000: {"data": "data/VRPB/vrpb2000_uniform.pkl"},
        5000: {"data": "data/VRPB/vrpb5000_uniform.pkl"},
    },
    "VRPL": {
        20: {"data": "data/VRPL/vrpl20_uniform.pkl", "solution": "data/VRPL/or_tools_10s_vrpl20_uniform.pkl"},
        50: {"data": "data/VRPL/vrpl50_uniform.pkl", "solution": "data/VRPL/or_tools_200s_vrpl50_uniform.pkl"},
        100: {"data": "data/VRPL/vrpl100_uniform.pkl", "solution": "data/VRPL/lkh_vrpl100_uniform.pkl"},
        200: {"data": "data/VRPL/vrpl200_uniform.pkl"},
        500: {"data": "data/VRPL/vrpl500_uniform.pkl"},
        1000: {"data": "data/VRPL/vrpl1000_uniform.pkl"},
        2000: {"data": "data/VRPL/vrpl2000_uniform.pkl"},
        5000: {"data": "data/VRPL/vrpl5000_uniform.pkl"},
    },
    "VRPTW": {
        20: {"data": "data/VRPTW/vrptw20_uniform.pkl", "solution": "data/VRPTW/or_tools_10s_vrptw20_uniform.pkl"},
        50: {"data": "data/VRPTW/vrptw50_uniform.pkl", "solution": "data/VRPTW/hgs_vrptw50_uniform.pkl"},
        100: {"data": "data/VRPTW/vrptw100_uniform.pkl", "solution": "data/VRPTW/hgs_vrptw100_uniform.pkl"},
        200: {"data": "data/VRPTW/vrptw200_uniform.pkl"},
        500: {"data": "data/VRPTW/vrptw500_uniform.pkl"},
        1000: {"data": "data/VRPTW/vrptw1000_uniform.pkl"},
        2000: {"data": "data/VRPTW/vrptw2000_uniform.pkl"},
        5000: {"data": "data/VRPTW/vrptw5000_uniform.pkl"},
    },
    "OVRPTW": {
        20: {"data": "data/OVRPTW/ovrptw20_uniform.pkl", "solution": "data/OVRPTW/or_tools_10s_ovrptw20_uniform.pkl"},
        50: {"data": "data/OVRPTW/ovrptw50_uniform.pkl", "solution": "data/OVRPTW/or_tools_200s_ovrptw50_uniform.pkl"},
        100: {"data": "data/OVRPTW/ovrptw100_uniform.pkl", "solution": "data/OVRPTW/or_tools_400s_ovrptw100_uniform.pkl"},
        200: {"data": "data/OVRPTW/ovrptw200_uniform.pkl"},
        500: {"data": "data/OVRPTW/ovrptw500_uniform.pkl"},
        1000: {"data": "data/OVRPTW/ovrptw1000_uniform.pkl"},
        2000: {"data": "data/OVRPTW/ovrptw2000_uniform.pkl"},
        5000: {"data": "data/OVRPTW/ovrptw5000_uniform.pkl"},
    },
    "OVRPB": {
        20: {"data": "data/OVRPB/ovrpb20_uniform.pkl", "solution": "data/OVRPB/or_tools_10s_ovrpb20_uniform.pkl"},
        50: {"data": "data/OVRPB/ovrpb50_uniform.pkl", "solution": "data/OVRPB/or_tools_200s_ovrpb50_uniform.pkl"},
        100: {"data": "data/OVRPB/ovrpb100_uniform.pkl", "solution": "data/OVRPB/or_tools_400s_ovrpb100_uniform.pkl"},
        200: {"data": "data/OVRPB/ovrpb200_uniform.pkl"},
        500: {"data": "data/OVRPB/ovrpb500_uniform.pkl"},
        1000: {"data": "data/OVRPB/ovrpb1000_uniform.pkl"},
        2000: {"data": "data/OVRPB/ovrpb2000_uniform.pkl"},
        5000: {"data": "data/OVRPB/ovrpb5000_uniform.pkl"},
    },
    "OVRPL": {
        20: {"data": "data/OVRPL/ovrpl20_uniform.pkl", "solution": "data/OVRPL/or_tools_10s_ovrpl20_uniform.pkl"},
        50: {"data": "data/OVRPL/ovrpl50_uniform.pkl", "solution": "data/OVRPL/or_tools_200s_ovrpl50_uniform.pkl"},
        100: {"data": "data/OVRPL/ovrpl100_uniform.pkl", "solution": "data/OVRPL/or_tools_400s_ovrpl100_uniform.pkl"},
        200: {"data": "data/OVRPL/ovrpl200_uniform.pkl"},
        500: {"data": "data/OVRPL/ovrpl500_uniform.pkl"},
        1000: {"data": "data/OVRPL/ovrpl1000_uniform.pkl"},
        2000: {"data": "data/OVRPL/ovrpl2000_uniform.pkl"},
        5000: {"data": "data/OVRPL/ovrpl5000_uniform.pkl"},
    },
    "VRPBL": {
        20: {"data": "data/VRPBL/vrpbl20_uniform.pkl", "solution": "data/VRPBL/or_tools_10s_vrpbl20_uniform.pkl"},
        50: {"data": "data/VRPBL/vrpbl50_uniform.pkl", "solution": "data/VRPBL/or_tools_200s_vrpbl50_uniform.pkl"},
        100: {"data": "data/VRPBL/vrpbl100_uniform.pkl", "solution": "data/VRPBL/or_tools_400s_vrpbl100_uniform.pkl"},
        200: {"data": "data/VRPBL/vrpbl200_uniform.pkl"},
        500: {"data": "data/VRPBL/vrpbl500_uniform.pkl"},
        1000: {"data": "data/VRPBL/vrpbl1000_uniform.pkl"},
        2000: {"data": "data/VRPBL/vrpbl2000_uniform.pkl"},
        5000: {"data": "data/VRPBL/vrpbl5000_uniform.pkl"},
    },
    "VRPBTW": {
        20: {"data": "data/VRPBTW/vrpbtw20_uniform.pkl", "solution": "data/VRPBTW/or_tools_10s_vrpbtw20_uniform.pkl"},
        50: {"data": "data/VRPBTW/vrpbtw50_uniform.pkl", "solution": "data/VRPBTW/or_tools_200s_vrpbtw50_uniform.pkl"},
        100: {"data": "data/VRPBTW/vrpbtw100_uniform.pkl", "solution": "data/VRPBTW/or_tools_400s_vrpbtw100_uniform.pkl"},
        200: {"data": "data/VRPBTW/vrpbtw200_uniform.pkl"},
        500: {"data": "data/VRPBTW/vrpbtw500_uniform.pkl"},
        1000: {"data": "data/VRPBTW/vrpbtw1000_uniform.pkl"},
        2000: {"data": "data/VRPBTW/vrpbtw2000_uniform.pkl"},
        5000: {"data": "data/VRPBTW/vrpbtw5000_uniform.pkl"},
    },
    "VRPLTW": {
        20: {"data": "data/VRPLTW/vrpltww20_uniform.pkl", "solution": "data/VRPLTW/or_tools_10s_vrpltww20_uniform.pkl"},
        50: {"data": "data/VRPLTW/vrpltww50_uniform.pkl", "solution": "data/VRPLTW/or_tools_200s_vrpltww50_uniform.pkl"},
        100: {"data": "data/VRPLTW/vrpltww100_uniform.pkl", "solution": "data/VRPLTW/or_tools_400s_vrpltww100_uniform.pkl"},
        200: {"data": "data/VRPLTW/vrpltww200_uniform.pkl"},
        500: {"data": "data/VRPLTW/vrpltww500_uniform.pkl"},
        1000: {"data": "data/VRPLTW/vrpltww1000_uniform.pkl"},
        2000: {"data": "data/VRPLTW/vrpltww2000_uniform.pkl"},
        5000: {"data": "data/VRPLTW/vrpltww5000_uniform.pkl"},
    },
    "OVRPBL": {
        20: {"data": "data/OVRPBL/ovrpbl20_uniform.pkl", "solution": "data/OVRPBL/or_tools_10s_ovrpbl20_uniform.pkl"},
        50: {"data": "data/OVRPBL/ovrpbl50_uniform.pkl", "solution": "data/OVRPBL/or_tools_200s_ovrpbl50_uniform.pkl"},
        100: {"data": "data/OVRPBL/ovrpbl100_uniform.pkl", "solution": "data/OVRPBL/or_tools_400s_ovrpbl100_uniform.pkl"},
        200: {"data": "data/OVRPBL/ovrpbl200_uniform.pkl"},
        500: {"data": "data/OVRPBL/ovrpbl500_uniform.pkl"},
        1000: {"data": "data/OVRPBL/ovrpbl1000_uniform.pkl"},
        2000: {"data": "data/OVRPBL/ovrpbl2000_uniform.pkl"},
        5000: {"data": "data/OVRPBL/ovrpbl5000_uniform.pkl"},
    },
    "OVRPBTW": {
        20: {"data": "data/OVRPBTW/ovrpbtw20_uniform.pkl", "solution": "data/OVRPBTW/or_tools_10s_ovrpbtw20_uniform.pkl"},
        50: {"data": "data/OVRPBTW/ovrpbtw50_uniform.pkl", "solution": "data/OVRPBTW/or_tools_200s_ovrpbtw50_uniform.pkl"},
        100: {"data": "data/OVRPBTW/ovrpbtw100_uniform.pkl", "solution": "data/OVRPBTW/or_tools_400s_ovrpbtw100_uniform.pkl"},
        200: {"data": "data/OVRPBTW/ovrpbtw200_uniform.pkl"},
        500: {"data": "data/OVRPBTW/ovrpbtw500_uniform.pkl"},
        1000: {"data": "data/OVRPBTW/ovrpbtw1000_uniform.pkl"},
        2000: {"data": "data/OVRPBTW/ovrpbtw2000_uniform.pkl"},
        5000: {"data": "data/OVRPBTW/ovrpbtw5000_uniform.pkl"},
    },
    "OVRPLTW": {
        20: {"data": "data/OVRPLTW/ovrpltww20_uniform.pkl", "solution": "data/OVRPLTW/or_tools_10s_ovrpltww20_uniform.pkl"},
        50: {"data": "data/OVRPLTW/ovrpltww50_uniform.pkl", "solution": "data/OVRPLTW/or_tools_200s_ovrpltww50_uniform.pkl"},
        100: {"data": "data/OVRPLTW/ovrpltww100_uniform.pkl", "solution": "data/OVRPLTW/or_tools_400s_ovrpltww100_uniform.pkl"},
        200: {"data": "data/OVRPLTW/ovrpltww200_uniform.pkl"},
        500: {"data": "data/OVRPLTW/ovrpltww500_uniform.pkl"},
        1000: {"data": "data/OVRPLTW/ovrpltww1000_uniform.pkl"},
        2000: {"data": "data/OVRPLTW/ovrpltww2000_uniform.pkl"},
        5000: {"data": "data/OVRPLTW/ovrpltww5000_uniform.pkl"},
    },
    "VRPBLTW": {
        20: {"data": "data/VRPBLTW/vrpbltw20_uniform.pkl", "solution": "data/VRPBLTW/or_tools_10s_vrpbltw20_uniform.pkl"},
        50: {"data": "data/VRPBLTW/vrpbltw50_uniform.pkl", "solution": "data/VRPBLTW/or_tools_200s_vrpbltw50_uniform.pkl"},
        100: {"data": "data/VRPBLTW/vrpbltw100_uniform.pkl", "solution": "data/VRPBLTW/or_tools_400s_vrpbltw100_uniform.pkl"},
        200: {"data": "data/VRPBLTW/vrpbltw200_uniform.pkl"},
        500: {"data": "data/VRPBLTW/vrpbltw500_uniform.pkl"},
        1000: {"data": "data/VRPBLTW/vrpbltw1000_uniform.pkl"},
        2000: {"data": "data/VRPBLTW/vrpbltw2000_uniform.pkl"},
        5000: {"data": "data/VRPBLTW/vrpbltw5000_uniform.pkl"},
    },
    "OVRPBLTW": {
        20: {"data": "data/OVRPBLTW/ovrpbltw20_uniform.pkl", "solution": "data/OVRPBLTW/or_tools_10s_ovrpbltw20_uniform.pkl"},
        50: {"data": "data/OVRPBLTW/ovrpbltw50_uniform.pkl", "solution": "data/OVRPBLTW/or_tools_200s_ovrpbltw50_uniform.pkl"},
        100: {"data": "data/OVRPBLTW/ovrpbltw100_uniform.pkl", "solution": "data/OVRPBLTW/or_tools_400s_ovrpbltw100_uniform.pkl"},
        200: {"data": "data/OVRPBLTW/ovrpbltw200_uniform.pkl"},
        500: {"data": "data/OVRPBLTW/ovrpbltw500_uniform.pkl"},
        1000: {"data": "data/OVRPBLTW/ovrpbltw1000_uniform.pkl"},
        2000: {"data": "data/OVRPBLTW/ovrpbltw2000_uniform.pkl"},
        5000: {"data": "data/OVRPBLTW/ovrpbltw5000_uniform.pkl"},
    }
}  

# --- Helper Function for Distance ---
def calculate_distance(coord1, coord2):
    "Calculates Euclidean distance between two coordinate points."
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(count, total, elapsed_time_str, remain_time_str))


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(args):
    """
        Occupy GPU memory in advance.
    """
    torch.cuda.set_device(args.gpu_id)
    total, used = check_mem(args.gpu_id)
    total, used = int(total), int(used)
    block_mem = int((total-used) * args.occ_gpu)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_env(problem):
    from envs import CVRPEnv, OVRPEnv, VRPBEnv, VRPLEnv, VRPTWEnv, OVRPTWEnv, OVRPBEnv, OVRPLEnv, VRPBLEnv, VRPBTWEnv, VRPLTWEnv, OVRPBLEnv, OVRPBTWEnv, OVRPLTWEnv, VRPBLTWEnv, OVRPBLTWEnv
    training_problems = ['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW']
    all_problems = {
        'CVRP': CVRPEnv,
        'OVRP': OVRPEnv,
        'VRPB': VRPBEnv,
        'VRPL': VRPLEnv,
        'VRPTW': VRPTWEnv,
        'OVRPTW': OVRPTWEnv,
        'OVRPB': OVRPBEnv,
        'OVRPL': OVRPLEnv,
        'VRPBL': VRPBLEnv,
        'VRPBTW': VRPBTWEnv,
        'VRPLTW': VRPLTWEnv,
        'OVRPBL': OVRPBLEnv,
        'OVRPBTW': OVRPBTWEnv,
        'OVRPLTW': OVRPLTWEnv,
        'VRPBLTW': VRPBLTWEnv,
        'OVRPBLTW': OVRPBLTWEnv,
    }
    if problem == "Train_ALL":
        return [all_problems[i] for i in training_problems]
    elif problem == "ALL":
        return list(all_problems.values())
    else:
        return [all_problems[problem]]


def get_model(model_type):
    from models import SINGLEModel, MTLModel, MOEModel, MOEModel_Light
    if model_type == "MTL":
        return MTLModel
    elif model_type == "MOE":
        return MOEModel
    elif model_type == "MOE_LIGHT":
        return MOEModel_Light
    elif model_type == "SINGLE":
        return SINGLEModel
    else:
        return NotImplementedError


def get_opt_sol_path(dir, problem, size):
    all_opt_sol = {
        'CVRP': {20: 'or_tools_10s_cvrp20_uniform.pkl', 50: 'hgs_cvrp50_uniform.pkl', 100: 'hgs_cvrp100_uniform.pkl'},
        'OVRP': {20: 'or_tools_10s_ovrp20_uniform.pkl', 50: 'or_tools_200s_ovrp50_uniform.pkl', 100: 'lkh_ovrp100_uniform.pkl'},
        'VRPB': {20: 'or_tools_10s_vrpb20_uniform.pkl', 50: 'or_tools_200s_vrpb50_uniform.pkl', 100: 'or_tools_400s_vrpb100_uniform.pkl'},
        'VRPL': {20: 'or_tools_10s_vrpl20_uniform.pkl', 50: 'or_tools_200s_vrpl50_uniform.pkl', 100: 'lkh_vrpl100_uniform.pkl'},
        'VRPTW': {20: 'or_tools_10s_vrptw20_uniform.pkl', 50: 'hgs_vrptw50_uniform.pkl', 100: 'hgs_vrptw100_uniform.pkl'},
        'OVRPTW': {20: 'or_tools_10s_ovrptw20_uniform.pkl', 50: 'or_tools_200s_ovrptw50_uniform.pkl', 100: 'or_tools_400s_ovrptw100_uniform.pkl'},
        'OVRPB': {20: 'or_tools_10s_ovrpb20_uniform.pkl', 50: 'or_tools_200s_ovrpb50_uniform.pkl', 100: 'or_tools_400s_ovrpb100_uniform.pkl'},
        'OVRPL': {20: 'or_tools_10s_ovrpl20_uniform.pkl', 50: 'or_tools_200s_ovrpl50_uniform.pkl', 100: 'or_tools_400s_ovrpl100_uniform.pkl'},
        'VRPBL': {20: 'or_tools_10s_vrpbl20_uniform.pkl', 50: 'or_tools_200s_vrpbl50_uniform.pkl', 100: 'or_tools_400s_vrpbl100_uniform.pkl'},
        'VRPBTW': {20: 'or_tools_10s_vrpbtw20_uniform.pkl', 50: 'or_tools_200s_vrpbtw50_uniform.pkl', 100: 'or_tools_400s_vrpbtw100_uniform.pkl'},
        'VRPLTW': {20: 'or_tools_10s_vrpltww20_uniform.pkl', 50: 'or_tools_200s_vrpltww50_uniform.pkl', 100: 'or_tools_400s_vrpltww100_uniform.pkl'},
        'OVRPBL': {20: 'or_tools_10s_ovrpbl20_uniform.pkl', 50: 'or_tools_200s_ovrpbl50_uniform.pkl', 100: 'or_tools_400s_ovrpbl100_uniform.pkl'},
        'OVRPBTW': {20: 'or_tools_10s_ovrpbtw20_uniform.pkl', 50: 'or_tools_200s_ovrpbtw50_uniform.pkl', 100: 'or_tools_400s_ovrpbtw100_uniform.pkl'},
        'OVRPLTW': {20: 'or_tools_10s_ovrpltww20_uniform.pkl', 50: 'or_tools_200s_ovrpltww50_uniform.pkl', 100: 'or_tools_400s_ovrpltww100_uniform.pkl'},
        'VRPBLTW': {20: 'or_tools_10s_vrpbltw20_uniform.pkl', 50: 'or_tools_200s_vrpbltw50_uniform.pkl', 100: 'or_tools_400s_vrpbltw100_uniform.pkl'},
        'OVRPBLTW': {20: 'or_tools_10s_ovrpbltw20_uniform.pkl', 50: 'or_tools_200s_ovrpbltw50_uniform.pkl', 100: 'or_tools_400s_ovrpbltw100_uniform.pkl'},
    }
    return os.path.join(dir, all_opt_sol[problem][size])


def num_param(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param / 1e6))


def check_null_hypothesis(a, b):
    print(len(a), a)
    print(len(b), b)
    alpha_threshold = 0.05
    t, p = ttest_rel(a, b)  # Calc p value
    print(t, p)
    p_val = p / 2  # one-sided
    # assert t < 0, "T-statistic should be negative"
    print("p-value: {}".format(p_val))
    if p_val < alpha_threshold:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Rejected.")
    else:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Accepted.")


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))


def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True, disable_tqdm=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval, disable=disable_tqdm))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def show(x, y, label, title, xdes, ydes, path, min_y=None, max_y=None, x_scale="linear", dpi=300):
    plt.style.use('fast')  # bmh, fivethirtyeight, Solarize_Light2
    plt.figure(figsize=(8, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'lightpink', 'lightgreen', 'linen', 'slategray', 'darkviolet', 'darkcyan']

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            # plt.scatter(x[i], y[i], color=colors[i], s=50)  # label=label[i]
            plt.plot(x[i], y[i], color=colors[i], label=label[i], linewidth=3)
        else:
            # plt.scatter(x[i], y[i], color=colors[i % len(label)])
            plt.plot(x[i], y[i], color=colors[i % len(label)], linewidth=3)

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    if min_y and max_y:
        plt.ylim((min_y, max_y))

    plt.title(title, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='upper right', fontsize=16)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")
