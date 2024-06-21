import math
import torch
import random
import pygame
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from model import Linear_QNet, QTrainer
from Snake.game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Predator:

    def __init__(self):
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(19, 256, 3)
        #self.model.load_state_dict(torch.load('model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.head = Point(0, 0)
        self.direction = Direction.RIGHT
        self.cuadrante = 0

    def get_state(self, game):

        # Calcular número de agentes en cada cuadrante
        n_agentes_cuad_1 = 0
        n_agentes_cuad_2 = 0
        n_agentes_cuad_3 = 0
        n_agentes_cuad_4 = 0

        for agent in game.agents:
            if agent.cuadrante == 1:
                n_agentes_cuad_1 += 1
            elif agent.cuadrante == 2:
                n_agentes_cuad_2 += 1
            elif agent.cuadrante == 3:
                n_agentes_cuad_3 += 1
            elif agent.cuadrante == 4:
                n_agentes_cuad_4 += 1

        # Calcular número de comidas en cada cuadrante
        n_comidas_cuad_1 = 0
        n_comidas_cuad_2 = 0
        n_comidas_cuad_3 = 0
        n_comidas_cuad_4 = 0

        for comida in game.food:
            if comida.x <= 320 and comida.y <= 240:
                n_comidas_cuad_1 = 1
            elif comida.x > 320 and comida.y <= 240:
                n_comidas_cuad_2 = 2
            elif comida.x <= 320 and comida.y > 240:
                n_comidas_cuad_3 = 3
            else:
                n_comidas_cuad_4 = 4

        # Puntos a cada lado del agente
        point_l = Point(self.head.x - 20, self.head.y)
        point_r = Point(self.head.x + 20, self.head.y)
        point_u = Point(self.head.x, self.head.y - 20)
        point_d = Point(self.head.x, self.head.y + 20)

        # Dirección a la que va
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Calcular la comida más próxima al agente
        chosen_food = Point(999, 999)
        chosen_distance = 999

        for food in game.food:
            food_dist = math.sqrt((food.x - self.head.x) ** 2 + (food.y - self.head.y) ** 2)
            if food_dist < chosen_distance:
                chosen_food = food
                chosen_distance = food_dist

        state = [
            # Danger straight
            (dir_r and game.is_collision(self, point_r)) or
            (dir_l and game.is_collision(self, point_l)) or
            (dir_u and game.is_collision(self, point_u)) or
            (dir_d and game.is_collision(self, point_d)),

            # Danger right
            (dir_u and game.is_collision(self, point_r)) or
            (dir_d and game.is_collision(self, point_l)) or
            (dir_l and game.is_collision(self, point_u)) or
            (dir_r and game.is_collision(self, point_d)),

            # Danger left
            (dir_d and game.is_collision(self, point_r)) or
            (dir_u and game.is_collision(self, point_l)) or
            (dir_r and game.is_collision(self, point_u)) or
            (dir_l and game.is_collision(self, point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            chosen_food.x < self.head.x,  # food left
            chosen_food.x > self.head.x,  # food right
            chosen_food.y < self.head.y,  # food up
            chosen_food.y > self.head.y,  # food down

            n_agentes_cuad_1 > 1,   # agents in quadrant 1
            n_agentes_cuad_2 > 1,   # agents in quadrant 2
            n_agentes_cuad_3 > 1,   # agents in quadrant 3
            n_agentes_cuad_4 > 1,   # agents in quadrant 4

            n_comidas_cuad_1 > 0,   # food in quadrant 1
            n_comidas_cuad_2 > 0,   # food in quadrant 2
            n_comidas_cuad_3 > 0,   # food in quadrant 3
            n_comidas_cuad_4 > 0    # food in quadrant 4
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        self.epsilon = max(100, 1500 - game.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 1500) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
