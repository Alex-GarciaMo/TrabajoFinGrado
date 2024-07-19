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
        self.model = Linear_QNet(24, 256, 4)
        # self.model.load_state_dict(torch.load('model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.type = type
        self.state = []
        self.head = Point(0, 0)
        self.random_games = 200
        self.direction = Direction.RIGHT

    def get_state(self, game):

        # Dirección a la que va
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Calculas cono de visión
        coord_casillas_frente = []

        # Orientación abajo
        if dir_d:
            for i in range(1, 4):
                coord_casillas_frente.append(Point(self.head.x, self.head.y + game.block_size * i))
                for j in range(1, i):
                    coord_casillas_frente.append(Point(self.head.x - game.block_size * j, self.head.y + game.block_size * i))
                    coord_casillas_frente.append(Point(self.head.x + game.block_size * j, self.head.y + game.block_size * i))

            # Casilla de su izquierda
            coord_casillas_frente.append(Point(self.head.x + game.block_size, self.head.y))
            # Casilla de su derecha
            coord_casillas_frente.append(Point(self.head.x - game.block_size, self.head.y))

        # Orientación arriba
        elif dir_u:
            for i in range(1, 4):
                coord_casillas_frente.append(Point(self.head.x, self.head.y - game.block_size * i))
                for j in range(1, i):
                    coord_casillas_frente.append(Point(self.head.x - game.block_size * j, self.head.y - game.block_size * i))
                    coord_casillas_frente.append(Point(self.head.x + game.block_size * j, self.head.y - game.block_size * i))

            # Casilla de su izquierda
            coord_casillas_frente.append(Point(self.head.x - game.block_size, self.head.y))
            # Casilla de su derecha
            coord_casillas_frente.append(Point(self.head.x + game.block_size, self.head.y))
            # Orientación derecha

        elif dir_r:
            for i in range(1, 4):
                coord_casillas_frente.append(Point(self.head.x + game.block_size * i, self.head.y))
                for j in range(1, i):
                    coord_casillas_frente.append(Point(self.head.x + game.block_size * i, self.head.y - game.block_size * j))
                    coord_casillas_frente.append(Point(self.head.x + game.block_size * i, self.head.y + game.block_size * j))

            # Casilla de su izquierda
            coord_casillas_frente.append(Point(self.head.x, self.head.y + game.block_size))
            # Casilla de su derecha
            coord_casillas_frente.append(Point(self.head.x, self.head.y - game.block_size))

        # Orientación izquierda
        elif dir_l:
            for i in range(1, 4):
                coord_casillas_frente.append(Point(self.head.x - game.block_size * i, self.head.y))
                for j in range(1, i):
                    coord_casillas_frente.append(Point(self.head.x - game.block_size * i, self.head.y - game.block_size * j))
                    coord_casillas_frente.append(Point(self.head.x - game.block_size * i, self.head.y + game.block_size * j))

            # Casilla de su izquierda
            coord_casillas_frente.append(Point(self.head.x, self.head.y - game.block_size))
            # Casilla de su derecha
            coord_casillas_frente.append(Point(self.head.x, self.head.y + game.block_size))

        # Calcular el oponente más próximo al agente
        chosen_one = Point(999, 999)
        chosen_distance = 999

        if self.type:
            for prey in game.preys:
                prey_dist = math.sqrt((prey.head.x - self.head.x) ** 2 + (prey.head.y - self.head.y) ** 2)
                if prey_dist < chosen_distance:
                    chosen_one = prey.head
                    chosen_distance = prey_dist
            n_opponents = len(game.preys)
        else:
            for predator in game.predators:
                predator_dist = math.sqrt((predator.head.x - self.head.x) ** 2 + (predator.head.y - self.head.y) ** 2)
                if predator_dist < chosen_distance:
                    chosen_one = predator.head
                    chosen_distance = predator_dist
            n_opponents = len(game.predators)

        # Tamaño de estado = 24 compuesto por: tipo, posición x e y del agente, las 4 direcciones donde solo una es 1,
        # la cantidad de oponentes restantes, la posición relativa del oponente más cercano usando 4 variables,
        # el cono de visión con 11 casillas y el tiempo transcurrido del juego.
        self.state = [
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
            n_opponents,

            # Opponent location
            chosen_one.x < self.head.x,  # Opponent left
            chosen_one.x > self.head.x,  # Opponent right
            chosen_one.y < self.head.y,  # Opponent up
            chosen_one.y > self.head.y,  # Opponent down

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
            0,
            # Time elapsed
            game.seconds
        ]

        # Actualizar el estado con el cono de visión del agente.
        c = 12
        for coord in coord_casillas_frente:
            # Si esa casilla es colisión para el agente o no:
            if game.is_collision(self, coord):
                self.state[c] = -1
            else:
                self.state[c] = (game.board.casillas[int(coord.y // game.block_size), int(coord.x // game.block_size)])
            c += 1

        return np.array(self.state, dtype=int)

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
        # [up, right, left, down]
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
            print(final_move)
            print(self.state)
        return final_move
