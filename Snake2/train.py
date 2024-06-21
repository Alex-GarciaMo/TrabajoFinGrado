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
from depredador import Predator

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


SPEED = 7

def train():
    record = 0
    score = 0
    n_agents = 1
    n_foods = 4
    metrics = {'Game': [], 'Score': [], 'Record': [], 'Time': []}

    agents = [Predator() for _ in range(n_agents)]
    game = SnakeGameAI(agents, n_foods)

    while True:
        for agent in agents:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score, time = game.play_step(final_move, agent)
            state_new = agent.get_state(game)

            reward = reward if reward > 0 else -1

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            print(agent.head.x, agent.head.y)

            if done and len(agents) > 0 and time < game.match_time:
                agent.train_long_memory()
                agent.model.save()
                agents.remove(agent)
                print("AGENT DEATH")

        game.update_ui(agents)
        game.clock.tick(SPEED)

        if time >= game.match_time and len(agents) > 0:
            reward = -10
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            agent.train_long_memory()
            agent.model.save()

            game.n_games += 1
            if score > record:
                record = score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')
            game.board.Print_Tablero()

            agents = [Predator() for _ in range(n_agents)]
            game.agents = agents

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            metrics_manager(metrics)

            game.reset()

        elif not agents:
            game.n_games += 1
            if score > record:
                record = score

            print(f'Game {game.n_games}, Score {score}, Record: {record}, Time: {game.last_time}s')
            game.board.Print_Tablero()

            agents = [Predator() for _ in range(n_agents)]
            game.agents = agents

            # Guardar las métricas
            metrics['Game'].append(game.n_games)
            metrics['Score'].append(score)
            metrics['Record'].append(record)
            metrics['Time'].append(game.last_time)

            metrics_manager(metrics)

            game.reset()



if __name__ == '__main__':
    train()
