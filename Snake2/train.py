import math
import torch
import random
import pygame
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from model import Linear_QNet, QTrainer
from game import PillaPillaGameAI, Direction, Point
from agent import Agent


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
            reward, done, score = game.play_step(final_move, agent)
            state_new = agent.get_state(game)

            # Si ha habido un encuentro con un oponente entonces score > 0
            if score:
                game.score += score  # Actualizar el score del juego
            if agent.type:
                # print(reward)
                pass
            # print(state_old, final_move, reward, state_new, done)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            # Si ha colisionado, guardar modelo y eliminar agente
            if done and len(agents) > 0 and game.seconds < game.match_time:
                agent.train_long_memory()
                agent.model.save()
                # Eliminar del juego el agente en función del tipo
                if agent.type:
                    game.predators.remove(agent)
                else:
                    game.preys.remove(agent)



def ResetGame(game, n_predators, n_preys, metrics):
    game.n_games += 1
    if game.score > game.record:
        game.record = game.score

    print(f'Game {game.n_games}, Score {game.score}, Record: {game.record}, Time: {game.last_time}s')

    # Guardar las métricas
    metrics['Game'].append(game.n_games)
    metrics['Score'].append(game.score)
    metrics['Record'].append(game.record)
    metrics['Time'].append(game.last_time)

    # metrics_manager(metrics)

    # Resetear juego
    game.predators = [Agent(1) for _ in range(n_predators)]  # Se reinicializan los agentes
    game.preys = [Agent(0) for _ in range(n_preys)]

    game.reset()


SPEED = 15


def train():
    n_predators = 2
    n_preys = 4
    metrics = {'Game': [], 'Score': [], 'Record': [], 'Time': []}

    predators = [Agent(1) for _ in range(n_predators)]
    preys = [Agent(0) for _ in range(n_preys)]
    game = PillaPillaGameAI(predators, preys)

    while True:
        # game.board.Print_Tablero()

        # Movimiento de los depredadores
        Movement(game.predators, game)
        # Se mueren todos los depredadores
        if not game.predators:
            ResetGame(game, n_predators, n_preys, metrics)

        # Movimiento de las presas
        Movement(game.preys, game)
        # Se mueren todos las presas
        if not game.preys:
            game.preys = [Agent(0) for _ in range(n_preys)]
            game.place_prey()

        if game.seconds >= game.match_time and len(game.predators) > 0:
            ResetGame(game, n_predators, n_preys, metrics)

        # Actualizamos el juego
        game.update_ui()
        game.clock.tick(SPEED)


if __name__ == '__main__':
    train()
