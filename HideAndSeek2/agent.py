# Código creado por Alejandro García Moreno.
# TFG 2023-2024: Desarrollo de un modelo de Aprendizaje por Refuerzo para el juego del Escondite

import os
import math
import torch
import random
import numpy as np
from collections import deque
from game import Direction, Point
from model import DeepQNetwork, DQNTrainer

# Fichero del agente. Hay dos tipos de agentes, depredadores (representados con el tipo 1) y presas (representados
# con el 0). Aquí se encuentran la información de su posición, dirección y fichero donde guardar las méticas.
# Además, el agente tiene una memoria, un modelo DQN y un entrenador.

MAX_MEMORY = 400_000
BATCH_SIZE = 100
LR = 0.001

metrics_folder_path = "metrics"


# Función estática que borra los ficheros de métricas si se comienza el entrenamiento de cero.
def clear_metrics_files():

    # Definir los nombres de los archivos CSV de las métricas
    predator_metrics_file = os.path.join(metrics_folder_path, 'predator_metrics.csv')
    prey_metrics_file = os.path.join(metrics_folder_path, 'prey_metrics.csv')

    # Eliminar el archivo de depredadores si existe
    if os.path.exists(predator_metrics_file):
        os.remove(predator_metrics_file)
        print(f'Archivo eliminado: {predator_metrics_file}')

    # Eliminar el archivo de presas si existe
    if os.path.exists(prey_metrics_file):
        os.remove(prey_metrics_file)
        print(f'Archivo eliminado: {prey_metrics_file}')


class Agent:

    def __init__(self, agent_type, load):
        self.epsilon = 0.001
        self.gamma = 0.99  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DeepQNetwork(11, 128, 4)
        self.trainer = None
        self.type = agent_type
        self.state = []
        self.file_name = None
        self.metrics = {'Game': [], 'Score': [], 'Reward': [], 'Loss': [], 'Q_value': []}
        self.head = Point(0, 0)
        self.random_games = 500
        self.direction = Direction.RIGHT
        self.x_dist_opp = 0
        self.y_dist_opp = 0
        self.load_model(load)

    # Si se desea cargar un modelo ya entrenado.
    def load_model(self, load):
        if self.type:
            self.file_name = "predator.pth"
        else:
            self.file_name = "prey.pth"
        if load:
            self.model.load_state_dict(torch.load('model/' + self.file_name))
            self.trainer = DQNTrainer(self.model, lr=LR, gamma=self.gamma)
        else:
            clear_metrics_files()
            self.trainer = DQNTrainer(self.model, lr=LR, gamma=self.gamma)

    # Gestiona las métricas que el agente ha recopilado durante la partida.
    # Como puede haber varios agentes del mismo tipo, todos sus datos se almacenan en la clase Game.
    # Al final de cada partida, se recogen esos datos y se almacenan en el CSV correspondiente.
    def metrics_manager(self, game):

        # Calcular las medias de cada métrica
        avg_game = round(sum(self.metrics['Game']) / len(self.metrics['Game']) if self.metrics['Game'] else 0)
        avg_score = round(sum(self.metrics['Score']) / len(self.metrics['Score'])) if self.metrics['Score'] else 0
        avg_reward = sum(self.metrics['Reward']) / len(self.metrics['Reward']) if self.metrics['Reward'] else 0
        avg_loss = sum(self.metrics['Loss']) / len(self.metrics['Loss']) if self.metrics['Loss'] else 0
        avg_q_value = sum(self.metrics['Q_value']) / len(self.metrics['Q_value']) if self.metrics['Q_value'] else 0

        # Datos a guardar en la clase Game
        agent_metrics = [avg_game, avg_score, avg_reward, avg_loss, avg_q_value]

        if self.type:
            game.predators_metrics.append(agent_metrics)
        else:
            game.preys_metrics.append(agent_metrics)

    # Método que calcula el estado actual del agente
    def get_state(self, game):

        # Dirección a la que va
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        point_l = Point(self.head.x - game.blck_sz, self.head.y)
        point_r = Point(self.head.x + game.blck_sz, self.head.y)
        point_u = Point(self.head.x, self.head.y - game.blck_sz)
        point_d = Point(self.head.x, self.head.y + game.blck_sz)

        # Calcular el oponente más próximo al agente
        if self.type:
            chosen_opponent = self.find_closest_opponent(game.preys)
        else:
            chosen_opponent = self.find_closest_opponent(game.predators)

        # Closest Opponent distance normalized
        # Esto sirve más adelante como recompensa dirigida. En función de la distancia al oponente.
        self.x_dist_opp = ((chosen_opponent.x // game.blck_sz - self.head.x // game.blck_sz) / (game.h // game.blck_sz))
        self.y_dist_opp = ((chosen_opponent.y // game.blck_sz - self.head.y // game.blck_sz) / (game.w // game.blck_sz))

        # Estado de 11 valores. Los 3 primeros muestran si la posición directamente contigua (en frente, derecha
        # o izquierda) es peligrosa para el agente. Los siguientes cuatro son la dirección que está llevando el agente.
        # Finalmente, los últimos 4 representan la posición relativa del oponente más cercano identificando en qué eje
        # y en qué sentido está el oponente.
        # Todos los valores del estado son binarios facilitando así el aprendizaje.
        self.state = [

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

            # Closest opponent orientate location
            chosen_opponent.x < self.head.x,  # Opponent left
            chosen_opponent.x > self.head.x,  # Opponent right
            chosen_opponent.y < self.head.y,  # Opponent up
            chosen_opponent.y > self.head.y  # Opponent down
        ]

        return np.array(self.state, dtype=int)

    # Método para encontrar el oponente más cercano al agente
    def find_closest_opponent(self, agents):
        chosen_opponent = Point(999, 999)
        chosen_distance = 999

        for agent in agents:
            opponent_dist = math.sqrt((agent.head.x - self.head.x) ** 2 + (agent.head.y - self.head.y) ** 2)
            if opponent_dist < chosen_distance:
                chosen_opponent = agent.head
                chosen_distance = opponent_dist

        return chosen_opponent

    # Método usado para obtener la acción predicha por el modelo
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

        return final_move

    # Método para actualizar la memoria
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Método de entrenamiento a corto plazo. Cada movimiento que el agente realiza es usado en el entrenamiento
    # de la red neuronal.
    def train_short_memory(self, state, action, reward, next_state, done, game):
        q_value, loss_value = self.trainer.train_step(state, action, reward, next_state, done)

        self.metrics['Game'].append(game.n_games)
        self.metrics['Reward'].append(reward)
        self.metrics['Q_value'].append(q_value)
        self.metrics['Loss'].append(loss_value)
        if self.type:
            self.metrics['Score'].append(game.score)
        else:
            self.metrics['Score'].append(len(game.preys) - game.score)

        if game.n_games % 50 == 0:
            self.trainer.target_model = self.trainer.model

    # Método de entrenamiento a largo plazo. DQN requiere de un método replay_memory donde recoge un conjunto fijo
    # de la memoria para evitar X <¡¡¡¡FALTAAAAAA!!!>
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        q_value, loss_value = self.trainer.train_step(states, actions, rewards, next_states, dones)
