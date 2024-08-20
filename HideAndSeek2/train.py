# Código creado por Alejandro García Moreno.
# TFG 2023-2024: Desarrollo de un modelo de Aprendizaje por Refuerzo para el juego del Escondite

import os
import csv
import math
import torch
import pandas as pd
from agent import Agent
import matplotlib.pyplot as plt
from game import HideAndSeekGameAI

# Fichero de entrenamiento.
# Bucle principal del juego y la gestión y almacenamiento de métricas.
# Este proyecto consta del entorno de juego de un Escondite en pygame.
# Consta de dos tipos de agentes, depredadores y presas (predators and preys).
# Los depredadores deben de cazar a las presas y las presas evitar ser cazadas.

metrics_folder_path = "metrics"


# Función que almacena las métricas para ambos tipos de agentes en sus respectivos CSV en cada partida ejecutada.
# Al final de la partida, los agentes tienen almacenados todos sus datos. Esta función recupera esos datos,
# calcula su media y lo almacena en un CSV en función del tipo de agente.
def save_metrics(game, n_preys, n_predators):
    if not os.path.exists(metrics_folder_path):
        os.makedirs(metrics_folder_path)

    # Variables de almacenamiento de las medias
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
        avg_data_preys[data] = avg_data_preys[data] / n_preys
        avg_data_predators[data] = avg_data_predators[data] / n_predators

    # Guardar los datos en un archivo CSV dentro de la carpeta metrics
    file_path = os.path.join(metrics_folder_path, 'prey_metrics.csv')  # Presas
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es el primer registro, incluir el encabezado
        if file.tell() == 0:
            writer.writerow(['Game', 'Score', 'Reward', 'Loss', 'Q_value'])
        writer.writerow(avg_data_preys)

    file_path = os.path.join(metrics_folder_path, 'predator_metrics.csv')  # Depredadores
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es el primer registro, incluir el encabezado
        if file.tell() == 0:
            writer.writerow(['Game', 'Score', 'Reward', 'Loss', 'Q_value'])
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


# Esta función calcula la media de las métricas que se quieren visualizar por cada bloque.
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


# Función que resetea el juego para comenzar una nueva partida
def reset_game(game, n_predators, n_preys, block_size):
    game.n_games += 1  # Aumentar contador de partida
    winner = None
    if game.score > game.record:  # Actualizar record
        game.record = game.score

    # Si queda algún agente vivo, guardar modelos
    if game.predators:
        winner = "Predators"
        for predator in game.predators:
            predator.model.save(predator.file_name)
            predator.train_long_memory()
            predator.metrics_manager(game)
            game.save_predators.append(predator)

    if game.preys:
        winner = "Preys"
        for prey in game.preys:
            prey.model.save(prey.file_name)
            prey.train_long_memory()
            prey.metrics_manager(game)
            game.save_preys.append(prey)


    # Imprimir resultados de la partida
    print(f'Game {int(game.n_games)}, Score {game.score}, Record: {game.record}, Time: {game.last_time}s')
    print("Winner:", winner)

    # Guardar métricas en el CSV y limpiarlas del juego
    if game.preys_metrics:
        save_metrics(game, n_preys, n_predators)
        game.preys_metrics = []
        game.predators_metrics = []

    # Resetear agentes del juego
    game.predators = game.save_predators  # Se recuperan los agentes
    game.preys = game.save_preys

    game.save_predators = []
    game.save_preys = []

    print(game.preys[0].memory)

    # Cada bloque, actualizar gráficas
    if not game.n_games % block_size:
        file_path = 'metrics/predator_metrics.csv'

        # Actualizar el gráfico después de cada bloque de 20 partidas
        update_evolutionary_plot(file_path, 1, block_size)

        file_path = 'metrics/prey_metrics.csv'
        update_evolutionary_plot(file_path, 0, block_size)

    # Resetear
    game.reset()


# Velocidad de acciones por segundo
SPEED = 20


# Función principal de entrenamiento.
# Inicialización de variables y agentes y bucle de entrenamiento
def train():
    n_predators = 1  # Número de depredadores
    n_preys = 1  # Número de presas
    load = 0  # Si se utiliza un modelo entrenado o se empieza de cero
    metrics_block_size = 15  # El tamaño de bloque para las métricas

    # Crear agentes
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

        game = HideAndSeekGameAI(predators, preys, n_games)
    else:
        game = HideAndSeekGameAI(predators, preys, 0)

    # Comienzo del bucle del entrenamiento
    while True:
        # Movimiento de los depredadores y las presas
        game.movement(game.predators)
        game.movement(game.preys)

        # Si se mueren todos los depredadores o todas las presas se resetea el juego
        if not game.predators or not game.preys:
            reset_game(game, n_predators, n_preys, metrics_block_size)

        # Si se acaba el tiempo, penalizar depredadores, recompensar presas y resetear juego
        if game.seconds >= game.match_time:
            game.end_time()  # Penalizar depredadores y recompensar presas
            reset_game(game, n_predators, n_preys, metrics_block_size)

        # Actualizar juego
        game.update_ui()
        game.clock.tick(SPEED)


if __name__ == '__main__':
    train()
