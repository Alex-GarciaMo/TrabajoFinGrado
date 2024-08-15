import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
from collections import deque


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(torch.relu(self.fc2(x)))
        actions = self.fc3(x)
        return actions

    def save(self, file_name='model.pth'):
        model_folder_path = 'model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = 'model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
        else:
            raise FileNotFoundError(f"No model found at {file_name}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_action = torch.argmax(self.model(next_state[idx]))

                Q_new = reward[idx] + self.gamma * self.target_model(next_state[idx])[next_action]

            action_index = (action[idx] == 1).nonzero(as_tuple=True)[0].item()
            target[idx][action_index] = Q_new

        self.optimizer.zero_grad()  # Limpiar gradientes anteriores
        loss = self.criterion(target, pred)  # Calcular pérdida
        loss.backward()  # Calcular gradientes
        self.optimizer.step()  # Actualizar los pesos


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
