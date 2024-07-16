import math
import torch
import random
import pygame
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from model import Linear_QNet, QTrainer
from game import SnakeGameAI, Direction, Point
from agent import Agent

global predators
global preys

def metrics_manager(metrics):
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Actualizar las gráficas
    axs[0].cla()
    axs[0].plot(metrics['Game'], metrics['Score'], label='Score')
    axs[0].plot(metrics['Game'], metrics['Record'], label='Record')
    axs[0].legend()

    axs[1].cla()
    axs[1].plot(metrics['Game'], metrics['Time'], label='Time')
    axs[1].legend()

    plt.pause(0.01)

    # Guardar métricas en un archivo CSV
    df = pd.DataFrame(metrics)
    df.to_csv('metrics.csv', index=False)


def Movement(agents, game):
    if agents:
        for agent in agents:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score, time = game.play_step(final_move, agent)
            state_new = agent.get_state(game)

        reward = reward if reward > 0 else -1

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done and len(agents) > 0 and time < game.match_time:
            agent.train_long_memory()
            agent.model.save()
            if agent.type:
                game.predators.remove(agent)
            else:
                game.preys.remove(agent)

        return state_old, final_move, state_new, reward, done, score, time

SPEED = 15

def train():
    record = 0
    score = 0
    n_predators = 1
    n_preys = 10
    metrics = {'Game': [], 'Score': [], 'Record': [], 'Time': []}

    predators = [Agent(1) for _ in range(n_predators)]
    preys = [Agent(0) for _ in range(n_preys)]
    game = SnakeGameAI(predators, preys)

    while True:
        # Movimiento de los agentes
        state_old, final_move, state_new, reward, done, score, time = Movement(game.predators, game)
        game.score += score
        state_old, final_move, state_new, reward, done, score, time = Movement(game.preys, game)

        game.update_ui()
        game.clock.tick(SPEED)

        # Se acaba el tiempo
        if time >= game.match_time and len(predators) > 0:
            reward = -10
            for predator in predators:
                predator.train_short_memory(state_old, final_move, reward, state_new, done)
                predator.remember(state_old, final_move, reward, state_new, done)
                predator.train_long_memory()
                predator.model.save()

            game.n_games += 1
            if game.score > record:
                record = game.score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            # metrics_manager(metrics)

            # Resetear juego
            predators = [Agent(1) for _ in range(n_predators)]  # Se reinicializan los agentes
            preys = [Agent(0) for _ in range(n_preys)]
            game.predators = predators
            game.preys = preys

            game.reset()

        # Se mueren todos los depredadores
        elif not predators:
            game.n_games += 1
            if game.score > record:
                record = game.score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            # metrics_manager(metrics)

            # Resetear juego
            predators = [Agent(1) for _ in range(n_predators)]  # Se reinicializan los agentes
            preys = [Agent(0) for _ in range(n_preys)]
            game.predators = predators
            game.preys = preys

            game.reset()

            # Se mueren todos los depredadores
        elif not preys:
            game.n_games += 1
            if game.score > record:
                record = game.score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            # metrics_manager(metrics)

            # Resetear juego
            predators = [Agent(1) for _ in range(n_predators)]  # Se reinicializan los agentes
            preys = [Agent(0) for _ in range(n_preys)]
            game.predators = predators
            game.preys = preys

            game.reset()

if __name__ == '__main__':
    train()
