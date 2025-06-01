import os
import math
import numpy as np

from collections import deque
import threading

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common import utils

import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete


device = None
step_check = False
prev_data = None
prev_result = None
env = None
model = None
pretrained = True
model_path = 'ppo_custom_model-2-204800'
data_stack = deque()
training_lock = threading.Lock()
# 이동 관련 변수
final_destination = [0,0]
prev_distance = 0

command_to_number = {'W': 0, 'S' : 1, 'A': 2, 'D': 3}
number_to_command = {0: 'W', 1 : 'S', 2: 'A', 3: 'D'}
weight_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

# 환경 선언 
class TankEnv(gym.Env):
    def __init__(self, max_steps = 1024):
        super().__init__() 
        # 연속형 환경 관측
        self.observation_space = Dict({
            "sensor_data": Box(low=-1, high=1, shape=(10,), dtype=np.float32),  # 7개의 센서 값
            "goal_position" : Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        })
        # 이산형 행동 출력
        self.action_space = MultiDiscrete([4, 11])
        self.steps = 0
        self.max_steps = max_steps
        self.prev_distance = 0

        print('Tank Env initialized')
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 시뮬레이터 초기화 및 초기 관측값 반환
        # options를 통해서 각종 자료를 flask 서버에서 넘겨보자
        if options:
            sensor_data = options['sensor_data']  # 더미 센서 데이터
            destination = options['goal_position']
        self.step_count = 0
        print('Environment has been reset')
        return {"sensor_data": sensor_data, "goal_position": destination}, {}
    
    def step(self, action):
        data = data_stack.pop()
        new_data = data['data']
        result = data['result']
        distance = round(data['distance'],2)
        # 데이터 추출
        sensor_data = new_data['sensor_data']
        goal_position = new_data['goal_position']
        # 바운더리 산출을 위한 좌표
        x = int(sensor_data.numpy()[0, 0] * 300)
        z = int(sensor_data.numpy()[0, 2] * 300)
        at_boundary = x == 300 or x == 0 or z == 300 or z == 0

        self.step_count += 1
        reward = 0
        # 타임 패널티
        reward -= 0.0005

        # 이전 거리 대비 감소해야 보상 수여
        if self.prev_distance > distance:
            reward += 0.002
        else:
            reward -= 0.002
        
        self.prev_distance = distance

        terminated = False
        truncated = False

        # 경계 위치 시 패널티
        if at_boundary:
            reward -= 0.001
        # 성공시 보상 
        if result:
            reward += 2
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True
        info = {}
        return {"sensor_data": sensor_data, "goal_position": goal_position}, reward, terminated, truncated, info
    

# 커스텀 피처 추출기 (이전 질문 참조)
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # goal_position용 MLP
        self.goal_mlp = nn.Sequential(
            nn.Linear(4, 32),  # 상대적 위치(2) + goal_position(2)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 결합된 피처 처리
        self.combined_mlp = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
            # sensor_data 처리
            sensor_data = observations["sensor_data"]  # shape: (batch_size, 7)
            sensor_features = self.mlp(sensor_data)  # shape: (batch_size, 64)
            
            # goal_position과 상대적 위치 처리
            goal_position = observations["goal_position"]  # shape: (batch_size, 2)
            current_position = sensor_data[:, [0, 2]]  # 가정: sensor_data의 첫 2차원이 위치
            relative_position = goal_position - current_position  # shape: (batch_size, 2)
            goal_input = torch.cat([goal_position, relative_position], dim=-1)  # shape: (batch_size, 4)
            goal_features = self.goal_mlp(goal_input)  # shape: (batch_size, 32)
            
            # 피처 결합
            combined_features = torch.cat([sensor_features, goal_features], dim=-1)  # shape: (batch_size, 96)
            return self.combined_mlp(combined_features)

# 커스텀 DummyVecEnv
class CustomDummyVecEnv(DummyVecEnv):
    def reset(self, seed=None, options=None):
        # 배치 차원 포함한 버퍼 초기화
        self.buf_obs = {
            key: np.zeros((self.num_envs,) + self.observation_space[key].shape, dtype=self.observation_space[key].dtype)
            for key in self.observation_space.spaces.keys()
        }
        infos = []
        for env_idx, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed, options=options)
            for key in self.buf_obs:
                self.buf_obs[key][env_idx] = obs[key]
            infos.append(info)
        # print(f"Reset buf_obs: image={self.buf_obs['image'].shape}, sensor_data={self.buf_obs['sensor_data'].shape}")
        return self.buf_obs.copy(), infos[0] if infos else {}

    def step_async(self, actions):
        self.step_results = []
        for env_idx, env in enumerate(self.envs):
            # Call env.step() directly, store results
            result = env.step(actions[env_idx])
            self.step_results.append(result)

    def step_wait(self):
        self.buf_obs = {
            "sensor_data": np.zeros((self.num_envs, 10), dtype=np.float32),
            "goal_position": np.zeros((self.num_envs, 2), dtype=np.float32)
        }
        rewards, dones, infos = [], [], []
        for i, (obs, rew, terminated, truncated, info) in enumerate(self.step_results):
            done = terminated or truncated
            
            self.buf_obs["sensor_data"][i] = np.copy(obs["sensor_data"])
            self.buf_obs["goal_position"][i] = np.copy(obs["goal_position"])
            
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        return self.buf_obs.copy(), np.array(rewards), np.array(dones), infos
    
def init_device():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# PPO 초기화
def initialize_ppo():
    global model, env
    env = TankEnv(1000000)
    env = CustomDummyVecEnv([lambda: env])
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        env=env,
        policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
        verbose=1,
        device=device,
    )
    if pretrained:
        params = PPO.load(model_path).get_parameters()
        model.set_parameters(params)
        print('Model Loaded: ',model_path)
    # Explicitly set logger
    model._logger = utils.configure_logger(verbose=model.verbose, tensorboard_log=None, tb_log_name="PPO")
    return model, env

def stack_data(data_np, result, distance):
    data_stack.append({'data':data_np, 'result':result, 'distance': distance})