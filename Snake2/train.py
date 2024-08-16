# Código creado por Alejandro García Moreno.
# TFG 2023-2024: Desarrollo de un modelo de Aprendizaje por Refuerzo para el juego del Pilla Pilla

import os
import csv
import math
import torch
import pandas as pd
from agent import Agent
import matplotlib.pyplot as plt
from game import PillaPillaGameAI

# Fichero de entrenamiento.
# Bucle principal del juego con el movimiento de los agentes y la gestión y almacenamiento de métricas.
# Este proyecto consta del entorno de juego de un Pilla Pilla en pygame.
# Consta de dos tipos de agentes, depredadores y presas (predators and preys).
# Los depredadores deben de cazar a las presas y las presas evitar ser cazadas.

metrics_folder_path = "metrics"


# Función que almacena las métricas para ambos tipos de agentes en sus respectivos CSV en cada partida ejecutada.
# Al final de la partida, los agentes tienen almacenados todos sus datos. Esta función recupera esos datos,
# calcula su media y lo almacena en un CSV en función del tipo de agente.
def save_metrics(game, n_preys, n_predators):
    if not os.path.exists(metrics_folder_path):
        os.makedirs(metrics_folder_path)

    # Variables de almacenamientod e las medias
    avg_data_preys = [0, 0, 0, 0, 0, 0]
    avg_data_predators = [0, 0, 0, 0, 0, 0]

    # Al poder haber varios agentes de cada tipo, se debe calcular la media del conjunto

    # Recuperar datos de la partida de las presas
    for agent_data in range(len(game.preys_metrics)):
        for index in range(len(game.preys_metrics[agent_data])):
            # Si el valor es un tensor, extraer el valor numérico
            if isinstance(game.preys_metrics[agent_data][index], torch.Tensor):
                avg_data_preys[index] += game.preys_metrics[agent_data][index].item()
            else:
                avg_data_preys[index] += game.preys_metrics[agent_data][index]

    # Recuperar datos de la partida de los depredadores
    for agent_data in range(len(game.predators_metrics)):
        for index in range(len(game.predators_metrics[agent_data])):
            # Si el valor es un tensor, extraer el valor numérico
            if isinstance(game.predators_metrics[agent_data][index], torch.Tensor):
                avg_data_predators[index] += game.predators_metrics[agent_data][index].item()
            else:
                avg_data_predators[index] += game.predators_metrics[agent_data][index]

    # Cálculo de la media de para cada tipo de agente
    for data in range(len(avg_data_preys)):
        avg_data_preys[data] = avg_data_preys[data] // n_preys
        avg_data_predators[data] = avg_data_predators[data] // n_predators

    # Guardar los datos en un archivo CSV dentro de la carpeta metrics
    file_path = os.path.join(metrics_folder_path, 'prey_metrics.csv')  # Presas
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es el primer registro, incluir el encabezado
        if file.tell() == 0:
            writer.writerow(['Game', 'Score', 'Epsilon', 'Reward', 'Loss', 'Q_value'])
        writer.writerow(avg_data_preys)

    file_path = os.path.join(metrics_folder_path, 'predator_metrics.csv')  # Depredadores
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es el primer registro, incluir el encabezado
        if file.tell() == 0:
            writer.writerow(['Game', 'Score', 'Epsilon', 'Reward', 'Loss', 'Q_value'])
        writer.writerow(avg_data_predators)


# Para poder visualizar los datos en una gráfica, se muestra la media de un conjunto de partidas
# Esta función divide el fichero de métricas en conjuntos/bloques.
def read_data_in_blocks(file_path, block_size):
    # Leer el archivo CSV
    df = pd.read_csv(file_path)

    # Dividir en bloques de 'block_size' partidas
    num_blocks = math.ceil(len(df) / block_size)
    blocks = [df.iloc[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]

    return blocks


# Esta función calcula la media de las métricas por cada bloque.
def calculate_average_metrics_per_block(blocks):
    avg_rewards = []
    avg_losses = []
    avg_q_values = []

    for block in blocks:
        avg_rewards.append(block['Reward'].mean())
        avg_losses.append(block['Loss'].mean())
        avg_q_values.append(block['Q_value'].mean())

    return avg_rewards, avg_losses, avg_q_values


# Esta función es la que se encarga de recopilar las métricas y mostrarlas en la gráfica.
# Se ejecuta iterativamente cada tamaño de bloque
def update_evolutionary_plot(file_path, agent_type, block_size):
    plt.ion()  # Activar el modo interactivo de matplotlib

    # Calcular bloques
    blocks = read_data_in_blocks(file_path, block_size)
    # Calcular medias
    avg_rewards, avg_losses, avg_q_values = calculate_average_metrics_per_block(blocks)

    # Limpiar la figura antes de redibujarla
    plt.clf()

    # Configuración de la gráfica
    plt.figure(figsize=(12, 6))

    # Graficar Reward
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, label=f'Avg Reward per {block_size} Games', color='blue')
    # Graficar Loss
    plt.plot(range(1, len(avg_losses) + 1), avg_losses, label=f'Avg Loss per {block_size} Games', color='red')
    # Graficar Q_value
    plt.plot(range(1, len(avg_q_values) + 1), avg_q_values, label=f'Avg Q Value per {block_size} Games', color='green')

    # Configuraciones de la gráfica
    plt.xlabel(f'Block of {block_size} Games')
    plt.ylabel('Value')
    if agent_type:
        plt.title('Evolution of Metrics Over Time for Predators')
    else:
        plt.title('Evolution of Metrics Over Time for Preys')
    plt.legend()
    plt.grid(True)

    # Actualizar la gráfica sin bloquear el flujo del código
    plt.draw()
    plt.pause(0.001)  # Pequeña pausa para actualizar la gráfica


def Movement(agents, game):
    if agents:
        for agent in agents:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score = game.play_step(final_move, agent)
            state_new = agent.get_state(game)

            if game.seconds < game.match_time:
                # Si ha habido un encuentro con un oponente entonces score > 0
                if score:
                    game.score += score  # Actualizar el score del juego

                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)

                # Si ha colisionado, guardar modelo y eliminar agente
                if done and len(agents) > 0:
                    agent.train_long_memory()
                    agent.model.save(agent.file_name)

                    # Eliminar del juego el agente en función del tipo
                    if agent.type:
                        agent.metrics_manager(game)   # Guardar métricas antes de ser removido
                        game.predators.remove(agent)  # Remover agente
                    else:
                        agent.metrics_manager(game)   # Guardar métricas antes de morir
                        game.preys.remove(agent)      # Remover agente


def ResetGame(game, n_predators, n_preys, metrics, block_size):
    game.n_games += 1
    if game.score > game.record:
        game.record = game.score

    # Guardar modelos
    if game.predators:
        for predator in game.predators:
            predator.model.save(predator.file_name)
            predator.train_long_memory()
            predator.metrics_manager(game)
            if game.n_games % 50 == 0:
                predator.trainer.target_model = predator.trainer.model
    if game.preys:
        for prey in game.preys:
            prey.model.save(prey.file_name)
            prey.train_long_memory()
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
    game.predators = [Agent(1, 1) for _ in range(n_predators)]  # Se reinicializan los agentes
    game.preys = [Agent(0, 1) for _ in range(n_preys)]

    if not game.n_games % block_size:
        file_path = 'metrics/predator_metrics.csv'

        # Actualizar el gráfico después de cada bloque de 20 partidas
        update_evolutionary_plot(file_path, 1, block_size)

        file_path = 'metrics/prey_metrics.csv'
        update_evolutionary_plot(file_path, 0, block_size)

    game.reset()


SPEED = 20


def train():
    n_predators = 2  # Número de depredadores
    n_preys = 4  # Número de presas
    load = 1  # Si se utiliza un modelo entrenado o se empieza de cero
    metrics_block_size = 15  # El tamaño de bloques para las métricas
    metrics = {'Game': [], 'Score': [], 'Record': [], 'Time': []}

    # Preparar agentes
    predators = [Agent(1, load) for _ in range(n_predators)]
    preys = [Agent(0, load) for _ in range(n_preys)]

    # Si se usa el mismo modelo, continuar por la última partida
    # En caso contrario, volver a empezar
    if load:
        file_path = 'metrics/predator_metrics.csv'
        df = pd.read_csv(file_path)

        # Obtener la última fila del DataFrame
        last_row = df.iloc[-1]

        # Obtener el primer valor de la última fila
        n_games = last_row.iloc[0]

        game = PillaPillaGameAI(predators, preys, n_games)
    else:
        game = PillaPillaGameAI(predators, preys, 0)

    # Comienzo del bucle del entrenamiento
    while True:

        # Movimiento de los depredadores y las presas
        Movement(game.predators, game)
        Movement(game.preys, game)

        # Se mueren todos los depredadores o todas las presas reseteamos juego
        if not game.predators or not game.preys:
            ResetGame(game, n_predators, n_preys, metrics, metrics_block_size)

        # Si se acaba el tiempo, penalizar depredadores, recompensar presas y resetear juego
        if game.seconds >= game.match_time:
            game.end_time()  # Penalizar depredadores y recompensar presas
            ResetGame(game, n_predators, n_preys, metrics, metrics_block_size)

        # Actualizamos el juego
        game.update_ui()
        game.clock.tick(SPEED)


if __name__ == '__main__':
    train()
