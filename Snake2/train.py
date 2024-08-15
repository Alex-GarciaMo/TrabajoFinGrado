import os
import csv
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from game import PillaPillaGameAI
from agent import Agent

metrics_folder_path = "metrics"

def save_metrics(game, n_preys, n_predators):
    if not os.path.exists(metrics_folder_path):
        os.makedirs(metrics_folder_path)

    avg_data_preys = [0, 0, 0, 0, 0, 0]
    avg_data_predators = [0, 0, 0, 0, 0, 0]

    for agent_data in range(len(game.preys_metrics)):
        for index in range(len(game.preys_metrics[agent_data])):
            if isinstance(game.preys_metrics[agent_data][index], torch.Tensor):  # Si el valor es un tensor, extraer el valor numérico
                avg_data_preys[index] += game.preys_metrics[agent_data][index].item()
            else:
                avg_data_preys[index] += game.preys_metrics[agent_data][index]

    for agent_data in range(len(game.predators_metrics)):
        for index in range(len(game.predators_metrics[agent_data])):
            if isinstance(game.predators_metrics[agent_data][index], torch.Tensor):
                avg_data_predators[index] += game.predators_metrics[agent_data][index].item()
            else:
                avg_data_predators[index] += game.predators_metrics[agent_data][index]

    for data in range(len(avg_data_preys)):
        avg_data_preys[data] = avg_data_preys[data] // n_preys
        avg_data_predators[data] = avg_data_predators[data] // n_predators

    # Guardar los datos en un archivo CSV dentro de la carpeta metrics
    file_path = os.path.join(metrics_folder_path, 'prey_metrics.csv')
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es el primer registro, incluir el encabezado
        if file.tell() == 0:
            writer.writerow(['Game', 'Score', 'Epsilon', 'Reward', 'Loss', 'Q_value'])
        writer.writerow(avg_data_preys)

    file_path = os.path.join(metrics_folder_path, 'predator_metrics.csv')
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es el primer registro, incluir el encabezado
        if file.tell() == 0:
            writer.writerow(['Game', 'Score', 'Epsilon', 'Reward', 'Loss', 'Q_value'])
        writer.writerow(avg_data_predators)


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
                agent.model.save(agent.file_name)

                # Eliminar del juego el agente en función del tipo
                if agent.type:
                    agent.metrics_manager(game)
                    game.predators.remove(agent)
                else:
                    agent.metrics_manager(game)
                    game.preys.remove(agent)


def ResetGame(game, n_predators, n_preys, metrics):
    game.n_games += 1
    if game.score > game.record:
        game.record = game.score

    # Guardar modelos?
    if game.predators:
        for predator in game.predators:
            predator.model.save(predator.file_name)
            predator.metrics_manager(game)
            if game.n_games % 50 == 0:
                predator.trainer.target_model = predator.trainer.model
    if game.preys:
        for prey in game.preys:
            prey.model.save(prey.file_name)
            prey.metrics_manager(game)
            if game.n_games % 50 == 0:
                prey.trainer.target_model = prey.trainer.model

    print(f'Game {game.n_games}, Score {game.score}, Record: {game.record}, Time: {game.last_time}s')

    # Guardar las métricas
    metrics['Game'].append(game.n_games)
    metrics['Score'].append(game.score)
    metrics['Record'].append(game.record)
    metrics['Time'].append(game.last_time)

    # metrics_manager(metrics)
    if game.preys_metrics:
        save_metrics(game, n_preys, n_predators)
        game.preys_metrics = []
        game.predators_metrics = []

    # Resetear juego
    game.predators = [Agent(1) for _ in range(n_predators)]  # Se reinicializan los agentes
    game.preys = [Agent(0) for _ in range(n_preys)]

    game.reset()


SPEED = 25


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
        if not game.predators or not game.preys:
            ResetGame(game, n_predators, n_preys, metrics)

        # Movimiento de las presas
        Movement(game.preys, game)
        # Se mueren todos las presas
        # if not game.preys:
        #     game.preys = [Agent(0) for _ in range(n_preys)]
        #     game.place_prey()

        if game.seconds >= game.match_time and len(game.predators) > 0:
            ResetGame(game, n_predators, n_preys, metrics)

        # Actualizamos el juego
        game.update_ui()
        game.clock.tick(SPEED)


if __name__ == '__main__':
    train()
