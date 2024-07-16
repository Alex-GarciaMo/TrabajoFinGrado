import math
import torch
import random
import pygame
import numpy as np
import pandas as pd
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
from model import Linear_QNet, QTrainer
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001



class Agent:

    def __init__(self, type):
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(23, 256, 4)
        # self.model.load_state_dict(torch.load('model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.type = type
        self.head = Point(0, 0)
        self.random_games = 200
        self.direction = Direction.RIGHT

    def get_state(self, game):

        # Dirección a la que va
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Calculas casillas de visión
        coor_casillas_frente = []

        # Orientación abajo
        if dir_d:
            for i in range(1, 4):
                coor_casillas_frente.append(Point(self.head.x, self.head.y + game.block_size * i))
                for j in range(1, i):
                    coor_casillas_frente.append(Point(self.head.x - game.block_size * j, self.head.y + game.block_size * i))
                    coor_casillas_frente.append(Point(self.head.x + game.block_size * j, self.head.y + game.block_size * i))

            # Casilla de su izquierda
            coor_casillas_frente.append(Point(self.head.x + game.block_size, self.head.y))
            # Casilla de su derecha
            coor_casillas_frente.append(Point(self.head.x - game.block_size, self.head.y))

        # Orientación arriba
        elif dir_u:
            for i in range(1, 4):
                coor_casillas_frente.append(Point(self.head.x, self.head.y - game.block_size * i))
                for j in range(1, i):
                    coor_casillas_frente.append(
                        Point(self.head.x - game.block_size * j, self.head.y - game.block_size * i))
                    coor_casillas_frente.append(
                        Point(self.head.x + game.block_size * j, self.head.y - game.block_size * i))

            # Casilla de su izquierda
            coor_casillas_frente.append(Point(self.head.x - game.block_size, self.head.y))
            # Casilla de su derecha
            coor_casillas_frente.append(Point(self.head.x + game.block_size, self.head.y))
            # Orientación derecha

        elif dir_r:
            for i in range(1, 4):
                coor_casillas_frente.append(Point(self.head.x + game.block_size * i, self.head.y))
                for j in range(1, i):
                    coor_casillas_frente.append(
                        Point(self.head.x + game.block_size * i, self.head.y - game.block_size * j))
                    coor_casillas_frente.append(
                        Point(self.head.x + game.block_size * i, self.head.y + game.block_size * j))

            # Casilla de su izquierda
            coor_casillas_frente.append(Point(self.head.x, self.head.y + game.block_size))
            # Casilla de su derecha
            coor_casillas_frente.append(Point(self.head.x, self.head.y - game.block_size))

        # Orientación izquierda
        elif dir_l:
            for i in range(1, 4):
                coor_casillas_frente.append(Point(self.head.x - game.block_size * i, self.head.y))
                for j in range(1, i):
                    coor_casillas_frente.append(
                        Point(self.head.x - game.block_size * i, self.head.y - game.block_size * j))
                    coor_casillas_frente.append(
                        Point(self.head.x - game.block_size * i, self.head.y + game.block_size * j))

            # Casilla de su izquierda
            coor_casillas_frente.append(Point(self.head.x, self.head.y - game.block_size))
            # Casilla de su derecha
            coor_casillas_frente.append(Point(self.head.x, self.head.y + game.block_size))

        # Calcular la comida más próxima al agente
        chosen_prey = Point(999, 999)
        chosen_distance = 999

        for prey in game.preys:
            prey_dist = math.sqrt((prey.head.x - self.head.x) ** 2 + (prey.head.y - self.head.y) ** 2)
            if prey_dist < chosen_distance:
                chosen_prey = prey.head
                chosen_distance = prey_dist

        # El estado tiene que tener 9 casillas en frente desde la posición del agente
        # El estado de las casillas de su izquierda y derecha
        # Move direction
        # Y te diría que ya está.
        state = [
            # Type
            self.type,

            # Position
            int(self.head.x),
            int(self.head.y),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Number of preys left
            len(game.preys),

            # Food location
            chosen_prey.x < self.head.x,  # food left
            chosen_prey.x > self.head.x,  # food right
            chosen_prey.y < self.head.y,  # food up
            chosen_prey.y > self.head.y,  # food down

            # Vision cone
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]

        # Puntos del tablero
        c = 10
        for coor in coor_casillas_frente:
            if 0 <= coor.x < game.w and 0 <= coor.y < game.h:
                state[c] = (game.board.casillas[int(coor.y // game.block_size), int(coor.x // game.block_size)])
                c += 1
            else:
                state[c] = -1
                c += 1

        # print(state)

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
        self.epsilon = max(50, self.random_games - game.n_games)
        final_move = [0, 0, 0, 0]
        if random.randint(0, self.random_games) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
