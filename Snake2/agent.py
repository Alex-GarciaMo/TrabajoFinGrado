import os
import csv
import math
import torch
import random
import pygame
import numpy as np
import pandas as pd
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
from model import DeepQNetwork, QTrainer
from game import PillaPillaGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01

metrics_folder_path = "metrics"

class Agent:

    def __init__(self, type):
        self.epsilon = 0.001  # randomness
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = DeepQNetwork(17, 128, 4)  # 25, 256, 4
        if type:
            self.file_name = "predator.pth"
        else:
            self.file_name = "prey.pth"
        #self.model.load_state_dict(torch.load('model/' + self.file_name))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.type = type
        self.state = []
        self.metrics = {'Game': [], 'Score': [], 'Epsilon': [], 'Reward': [], 'Loss': [], 'Q_value': []}
        self.head = Point(0, 0)
        self.random_games = 1000
        self.direction = Direction.RIGHT


    def metrics_manager(self, game):

        # Calcular las medias de cada métrica
        avg_game = sum(self.metrics['Game']) / len(self.metrics['Game']) if self.metrics['Game'] else 0
        avg_score = round(sum(self.metrics['Score']) / len(self.metrics['Score'])) if self.metrics['Score'] else 0
        avg_epsilon = sum(self.metrics['Epsilon']) / len(self.metrics['Epsilon']) if self.metrics['Epsilon'] else 0
        avg_reward = sum(self.metrics['Reward']) / len(self.metrics['Reward']) if self.metrics['Reward'] else 0
        avg_loss = sum(self.metrics['Loss']) / len(self.metrics['Loss']) if self.metrics['Loss'] else 0
        avg_q_value = sum(self.metrics['Q_value']) / len(self.metrics['Q_value']) if self.metrics['Q_value'] else 0

        # Datos a guardar en el CSV
        agent_metrics = [avg_game, avg_score, avg_epsilon, avg_reward, avg_loss, avg_q_value]

        if self.type:
            game.predators_metrics.append(agent_metrics)
        else:
            game.preys_metrics.append(agent_metrics)


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
        if self.type:
            chosen_opponent = self.find_closest_opponent(game.preys)
        else:
            chosen_opponent = self.find_closest_opponent(game.predators)

        # Estado de 17 valores, los 4 primeros son la dirección del agente. Los 2 siguientes son la diferencia de
        # distancia entre el agente y el oponente más cercano. Finalmente, el cono de visión.

        self.state = [

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Closest Opponent distance
            chosen_opponent.x - self.head.x,
            chosen_opponent.y - self.head.y,

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

        # Actualizar el estado con el cono de visión del agente.
        c = 6

        if game.is_collision(self, self.head):
            for i, coord in enumerate(coord_casillas_frente, start=c):
                self.state[i] = -1

        for coord in coord_casillas_frente:
            # Si esa casilla es colisión para el agente o no:
            if game.is_collision(self, coord):
                self.state[c] = -1
            else:
                self.state[c] = (game.board.casillas[int(coord.y // game.block_size), int(coord.x // game.block_size)])
            c += 1

        # print(self.state)
        return np.array(self.state, dtype=int)

    def find_closest_opponent(self, agents):
        chosen_opponent = Point(999, 999)
        chosen_distance = 999

        for agent in agents:
            opponent_dist = math.sqrt((agent.head.x - self.head.x) ** 2 + (agent.head.y - self.head.y) ** 2)
            if opponent_dist < chosen_distance:
                chosen_opponent = agent.head
                chosen_distance = opponent_dist

        return chosen_opponent

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        Q_value, loss_value = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.metrics['Q_value'].append(Q_value)
        self.metrics['Loss'].append(loss_value)

    def train_short_memory(self, state, action, reward, next_state, done):
        Q_value, loss_value = self.trainer.train_step(state, action, reward, next_state, done)
        self.metrics['Q_value'].append(Q_value)
        self.metrics['Loss'].append(loss_value)

    def get_action(self, state, game):
        # [up, right, left, down]
        self.epsilon = max(50, self.random_games - game.n_games)
        self.metrics['Epsilon'].append(self.epsilon)
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
