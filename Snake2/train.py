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
        end_time = False
        for agent in agents:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score = game.play_step(final_move, agent)
            state_new = agent.get_state(game)

            if score:
                game.score += score
                SharedReward(game, game.predators, reward)
                SharedReward(game, game.preys, game.reward - reward)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            # Si ha colisionado, eliminar agente
            if done and len(agents) > 0 and game.seconds < game.match_time:
                agent.train_long_memory()
                agent.model.save()
                if agent.type:
                    game.predators.remove(agent)
                else:
                    game.preys.remove(agent)

            # Comprobar si se ha acabado el tiempo del juego
            if EndTime(game):
                end_time = True

        return end_time

def SharedReward(game, agents, reward):
    for agent in agents:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, game)
        agent.train_short_memory(state_old, final_move, reward, state_old, True)
        agent.remember(state_old, final_move, reward, state_old, True)
        agent.train_long_memory()
        agent.model.save()


def EndTime(game):
    # Se acaba el tiempo y sigue habiendo depredadores
    if game.seconds >= game.match_time and len(game.predators) > 0:

        # Castigar a los depredadores
        SharedReward(game, game.predators, -game.reward)
        # Recompensar a las presas
        SharedReward(game, game.preys, game.reward)

        return True

    return False

def ResetGame(game, record, n_predators, n_preys, metrics):
    game.n_games += 1
    if game.score > record:
        record = game.score

    print(f'Game {game.n_games}, Score {game.score}, Record: {record}, Time: {game.last_time}s')

    # Guardar las métricas
    metrics['Game'].append(game.n_games)
    metrics['Score'].append(game.score)
    metrics['Record'].append(record)
    metrics['Time'].append(game.last_time)

    # metrics_manager(metrics)

    # Resetear juego
    predators = [Agent(1) for _ in range(n_predators)]  # Se reinicializan los agentes
    preys = [Agent(0) for _ in range(n_preys)]
    game.predators = predators
    game.preys = preys

    game.reset()


SPEED = 15

def train():
    record = 0
    score = 0
    n_predators = 1
    n_preys = 1
    metrics = {'Game': [], 'Score': [], 'Record': [], 'Time': []}

    predators = [Agent(1) for _ in range(n_predators)]
    preys = [Agent(0) for _ in range(n_preys)]
    game = SnakeGameAI(predators, preys)

    while True:
        # Movimiento de los agentes
        if Movement(game.predators, game):
            ResetGame(game, record, n_predators, n_preys, metrics)
        # Se mueren todos los depredadores
        if not game.predators:
            ResetGame(game, record, n_predators, n_preys, metrics)

        Movement(game.preys, game)

        # Se mueren todos los depredadores
        if not game.preys:
            ResetGame(game, record, n_predators, n_preys, metrics)

        game.update_ui()
        game.clock.tick(SPEED)

if __name__ == '__main__':
    train()
