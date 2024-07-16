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
from depredador import Agent

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


SPEED = 15

def train():
    record = 0
    score = 0
    n_predators = 1
    n_preys = 1
    metrics = {'Game': [], 'Score': [], 'Record': [], 'Time': []}

    predators = [Agent() for _ in range(n_predators)]
    preys = [Agent() for _ in range(n_preys)]
    game = SnakeGameAI(predators, preys)

    while True:
        for predator in predators:
            state_old = predator.get_state(game)
            final_move = predator.get_action(state_old, game)
            reward, done, score, time = game.play_step(final_move, predator)
            state_new = predator.get_state(game)

            reward = reward if reward > 0 else -1

            predator.train_short_memory(state_old, final_move, reward, state_new, done)
            predator.remember(state_old, final_move, reward, state_new, done)

            if done and len(predators) > 0 and time < game.match_time:
                predator.train_long_memory()
                predator.model.save()
                predators.remove(predator)

        game.update_ui(predators)
        game.clock.tick(SPEED)

        if time >= game.match_time and len(predators) > 0:
            reward = -10
            predators[0].train_short_memory(state_old, final_move, reward, state_new, done)
            predators[0].remember(state_old, final_move, reward, state_new, done)
            predators[0].train_long_memory()
            predators[0].model.save()

            game.n_games += 1
            if score > record:
                record = score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            # metrics_manager(metrics)

            # Resetear juego
            game.reset()

        elif not predators:
            game.n_games += 1
            if score > record:
                record = score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            # metrics_manager(metrics)

            # Resetear juego
            game.reset()



if __name__ == '__main__':
    train()
