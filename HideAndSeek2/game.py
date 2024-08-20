# Código creado por Alejandro García Moreno.
# TFG 2023-2024: Desarrollo de un modelo de Aprendizaje por Refuerzo para el juego del Escondite

import math
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# En este fichero se encuentra el juego desarrollado con pygame.
# Aquí se gestiona la parte visual del juego y las interacciones de los agentes con el juego.
# El juego es un tablero cuadricular definido por coordenadas.
# Cada movimiento mueve al agente un cuadrado adyacente obligatoriamente (no se pueden quedar quietos)

# Inicialización de pygame
pygame.init()
# font = pygame.font.Font('../HideAndSeek/arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)


# Esta clase es usada para definir la dirección a la que se mueven los agentes
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Este es el formato con el que se representa las coordenadas de los agentes.
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Tamaño del bloque del tablero del juego
BLOCK_SIZE = 20


# Esta clase es una representación reducida del tablero del juego donde se representan los diferentes
# agentes en sus posiciones correspondientes. Las posiciones vacías se representan con 0, las casillas con
# presas se representan con 1 y los depredadores con 2.
# Fue usado para un intento de cono de visión demasiado complejo y fue descartado, pero actualmente, solo es usado
# para comprobar si un depredador ha cazado una presa.
class Tablero:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
        self.block_size = 20
        self.boxes = np.zeros((self.alto // BLOCK_SIZE, self.ancho // BLOCK_SIZE), dtype=int)

    def resetear_tablero(self):
        self.boxes = np.zeros((self.alto // BLOCK_SIZE, self.ancho // BLOCK_SIZE), dtype=int)

    def print_tablero(self):
        print(self.boxes)


# Función estática que calcula la recompensa dirigida en función de la distancia al enemigo.
def calculate_directed_reward(agent):
    distance_to_opponent = math.sqrt(agent.state[4]**2 + agent.state[5]**2)
    if agent.type:
        reward = - distance_to_opponent
    else:
        reward = + distance_to_opponent

    return reward


# Clase del entorno de juego.
# Sus métodos se encargan de gestionar la interacción de los agentes con el entorno
class HideAndSeekGameAI:

    def __init__(self, predators, preys, n_games, w=640, h=480):
        self.blck_sz = BLOCK_SIZE
        self.score = 0
        self.record = 0
        self.frame_iteration = 0
        self.n_games = n_games  # Partida que se está jugando
        self.match_time = 10    # Tiempo máximo de partida en segundos
        self.w = w
        self.h = h
        self.fixed_reward = 10
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('IA ESCONDITE')
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()  # Inicializa el tiempo de inicio del juego
        self.seconds = 0
        self.last_time = self.seconds
        self.predators = predators
        self.preys = preys
        self.n_predators = len(predators)
        self.n_preys = len(preys)
        self.board = Tablero(w, h)
        self.preys_metrics = []      # Almacenan las métricas de esta partida
        self.predators_metrics = []
        self.save_predators = []  # Almacenar los agentes muertos pero sin destruirlos
        self.save_preys = []
        self.reset()  # Empieza el juego

    # Método para resetear el juego donde se posicionan los agentes y reinicializan las variables.
    def reset(self):
        self.board.resetear_tablero()  # Resetear tablero

        # Se recolocan los agentes
        self.place_predators()
        self.place_prey()

        # Se resetean los contadores
        self.score = 0
        self.frame_iteration = 0
        self.start_time = pygame.time.get_ticks()  # Reinicia el tiempo de inicio del juego
        self.seconds = 0
        self.last_time = self.seconds
        self.preys_metrics = []
        self.predators_metrics = []

    # Este método posiciona los depredadores en el tablero.
    def place_predators(self):
        separation = 0
        for predator in self.predators:
            # init agent state
            predator.direction = Direction.RIGHT

            predator.head = Point(self.w / 2 - separation, self.h / 2)
            self.board.boxes[int(predator.head.y // self.blck_sz), int(predator.head.x // self.blck_sz)] = 2
            separation = + 20

    # Posicionamiento de las presas en espacios libres.
    def place_prey(self):
        for prey in self.preys:
            x = random.randint(0, (self.w - self.blck_sz) // self.blck_sz) * self.blck_sz
            y = random.randint(0, (self.h - self.blck_sz) // self.blck_sz) * self.blck_sz
            direction = Direction(random.randint(1, 4))

            while self.board.boxes[y // self.blck_sz, x // self.blck_sz] != 0:
                x = random.randint(0, (self.w - self.blck_sz) // self.blck_sz) * self.blck_sz
                y = random.randint(0, (self.h - self.blck_sz) // self.blck_sz) * self.blck_sz
            prey.head = Point(x, y)
            prey.direction = direction
            self.board.boxes[y // self.blck_sz, x // self.blck_sz] = 1

    # Proceso de movimiento de todos los agentes.
    # Aquí se calcula el estado antiguo, la acción realizada, el estado nuevo y la recompensa recibida
    def movement(self, agents):
        if agents:
            for agent in agents:
                state_old = agent.get_state(self)
                final_move = agent.get_action(state_old, self)
                reward, done, score = self.play_step(final_move, agent)
                state_new = agent.get_state(self)

                # Si no se ha acabado el tiempo se siguen con las comprobaciones
                if self.seconds < self.match_time:
                    # Si ha habido un encuentro con un oponente entonces score > 0
                    if score:
                        self.score += score  # Actualizar el score del juego
                        reward = self.kill_prey(agent)  # Calcular recompensa y eliminar presa

                    # Entrenar siempre el agente y almacenarlo en su memoria
                    agent.train_short_memory(state_old, final_move, reward, state_new, done, self)
                    agent.remember(state_old, final_move, reward, state_new, done)

                    # Si ha colisionado, guardar modelo y eliminar agente
                    # if done and len(agents) > 0:
                    if done:
                        agent.train_long_memory()
                        agent.model.save(agent.file_name)

                        # Eliminar del juego el agente en función del tipo
                        if agent.type:
                            agent.metrics_manager(self)   # Guardar métricas antes de ser removido
                            self.predators.remove(agent)  # Remover agente
                            self.save_predators.append(agent)  # Guardar agente para la próxima partida
                        else:
                            agent.metrics_manager(self)   # Guardar métricas antes de morir
                            self.preys.remove(agent)  # Remover agente
                            self.save_preys.append(agent)


    # Método que realiza el movimiento de un agente con la acción dada
    def move(self, action, agent):  # [up, right, left, down]
        # Identificar acción
        if np.array_equal(action, [1, 0, 0, 0]):  # Arriba
            agent.direction = Direction.UP
        elif np.array_equal(action, [0, 1, 0, 0]):  # Derecha
            agent.direction = Direction.RIGHT
        elif np.array_equal(action, [0, 0, 1, 0]):  # Izquierda
            agent.direction = Direction.LEFT
        else:                                       # [0, 0, 0, 1] Abajo
            agent.direction = Direction.DOWN

        x = agent.head.x
        y = agent.head.y

        # Actualizar coordenada en función de la acción
        if agent.direction == Direction.RIGHT:
            x += self.blck_sz
        elif agent.direction == Direction.LEFT:
            x -= self.blck_sz
        elif agent.direction == Direction.DOWN:
            y += self.blck_sz
        elif agent.direction == Direction.UP:
            y -= self.blck_sz

        # Vaciar la antigua casilla del tablero y moverse a la siguiente
        self.board.boxes[int(agent.head.y // self.blck_sz), int(agent.head.x // self.blck_sz)] = 0
        agent.head = Point(x, y)  # Actualizar posición del agente

        # Si ha cazado a la presa o no
        catch = False

        # Comprobar que no se haya ido fuera del límite
        if not self.is_collision(agent, agent.head):
            # Comprobar si en esa casilla había un oponente
            if self.board.boxes[int(agent.head.y // self.blck_sz), int(agent.head.x // self.blck_sz)] != agent.type + 1\
                    and self.board.boxes[int(agent.head.y // self.blck_sz), int(agent.head.x // self.blck_sz)] > 0:
                catch = True
                # Actualizar la casilla con el valor depredador
                self.board.boxes[int(agent.head.y // self.blck_sz), int(agent.head.x // self.blck_sz)] = 2

            # En caso contrario, actualizar el tablero en función del tipo del agente
            else:
                self.board.boxes[int(agent.head.y // self.blck_sz), int(agent.head.x // self.blck_sz)] = agent.type + 1

        return catch

    # Método que gestiona la interacción de la acción ha realizar con el entorno.
    def play_step(self, action, agent):
        # Ver si se cierra el juego
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Realizar acción y actualizar la posición del agente
        catch = self.move(action, agent)  # Si ha ocurrido un encuentro entre agentes opuestos

        # Recompensa dirigida en función de la distancia al oponente
        reward = calculate_directed_reward(agent)
        score = 0
        done = False

        # Ver si el agente ha colisionado
        if self.is_collision(agent, agent.head):
            done = True
            reward = - self.fixed_reward  # -10
            return reward, done, score

        # o la partida ha terminado.
        elif self.seconds > self.match_time:
            done = True
            return reward, done, score

        # Al haber caza, se aumenta la puntuación y si ha sido una presa, se muere.
        if catch:
            score += 1
            if not agent.type:
                done = True

        return reward, done, score

    # Dar recompensa en función de si el agente es capturado o ha capturado
    def opponent_catch(self, agent):
        if agent.memory:
            last_memory = agent.memory[-1]
            state, action, reward, next_state, done = last_memory
            if agent.type:
                reward = self.fixed_reward
                done = False
            else:
                reward = -self.fixed_reward
                done = True
            agent.train_short_memory(state, action, reward, next_state, done, self)
            agent.remember(state, action, reward, next_state, done)

    # Método que comprueba si el cazador ha cazado una presa o si la presa se ha encontrado con el cazado.
    # Después, recompensa al depredador y castiga a la presa.
    def kill_prey(self, agent):
        reward = 0

        if agent.type:  # Si es depredador
            for prey in self.preys:  # Busca la presa que ha cazado
                if agent.head == prey.head:
                    reward = self.fixed_reward
                    self.opponent_catch(prey)  # Añadimos recompensa negativa a la presa capturada
                    prey.train_long_memory()  # Entrenar antes de ser removido
                    prey.model.save(agent.file_name)  # Guardar modelo
                    prey.metrics_manager(self)  # Guardamos las métricas
                    self.preys.remove(prey)  # Eliminar presa cazada
                    self.save_preys.append(prey)  # Guardar agente para usarlo en la próxima partida

        else:  # Si es presa
            for predator in self.predators:  # Se busca al cazador que le ha cazado
                if agent.head == predator.head:
                    reward = - self.fixed_reward
                    self.opponent_catch(predator)  # Añadimos recompensa positiva al depredador

        return reward

    # Al terminar el tiempo, recompensar a las presas y castigar a los depredadores
    def end_time(self):
        for predator in self.predators:
            last_memory = predator.memory[-1]
            state, action, reward, next_state, done = last_memory
            reward = -self.fixed_reward
            done = True
            predator.train_short_memory(state, action, reward, next_state, done, self)
            predator.remember(state, action, reward, next_state, done)

        for prey in self.preys:
            last_memory = prey.memory[-1]
            state, action, reward, next_state, done = last_memory
            reward = self.fixed_reward
            done = True
            prey.train_short_memory(state, action, reward, next_state, done, self)
            prey.remember(state, action, reward, next_state, done)

    # Método que identifica si las coordenadas recibidas están dentro de los parámetros del tablero.
    def is_collision(self, agent, pt=None):
        if pt is None:
            pt = agent.head
        # Si da al borde del tablero
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    # Aquí se actualiza el tablero visualmente con las nuevas posiciones de los agentes.
    def update_ui(self):

        # Limpiar tablero
        self.display.fill(BLACK)

        # Calcular puntos del triángulo en función de la dirección de los depredadores
        for predator in self.predators:
            x = predator.head.x
            y = predator.head.y
            if predator.direction == Direction.UP:  # Punta arriba
                vertices = ((x, y + BLOCK_SIZE), (x + BLOCK_SIZE, y + BLOCK_SIZE), (x + BLOCK_SIZE / 2, y))
            elif predator.direction == Direction.DOWN:  # Punta abajo
                vertices = ((x, y), (x + BLOCK_SIZE, y), (x + BLOCK_SIZE / 2, y + BLOCK_SIZE))
            elif predator.direction == Direction.RIGHT:  # Punta derecha
                vertices = ((x, y), (x, y + BLOCK_SIZE), (x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2))
            else:  # Punta izquierda
                vertices = ((x, y + BLOCK_SIZE / 2), (x + BLOCK_SIZE, y), (x + BLOCK_SIZE, y + BLOCK_SIZE))

            # Dibujar la cabeza del agente
            pygame.draw.polygon(self.display, RED, vertices)

        # Calcular puntos del triángulo en función de la dirección de las presas
        for prey in self.preys:
            x = prey.head.x
            y = prey.head.y
            if prey.direction == Direction.UP:  # Punta arriba
                vertices = ((x, y + BLOCK_SIZE), (x + BLOCK_SIZE, y + BLOCK_SIZE), (x + BLOCK_SIZE / 2, y))
            elif prey.direction == Direction.DOWN:  # Punta abajo
                vertices = ((x, y), (x + BLOCK_SIZE, y), (x + BLOCK_SIZE / 2, y + BLOCK_SIZE))
            elif prey.direction == Direction.RIGHT:  # Punta derecha
                vertices = ((x, y), (x, y + BLOCK_SIZE), (x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2))
            else:  # Punta izquierda
                vertices = ((x, y + BLOCK_SIZE / 2), (x + BLOCK_SIZE, y), (x + BLOCK_SIZE, y + BLOCK_SIZE))

            # Dibujar la cabeza del agente
            pygame.draw.polygon(self.display, BLUE1, vertices)

        # Textos informativos en la partida:

        # Número de la partida
        text = font.render("Game: " + str(int(self.n_games)), True, WHITE)
        self.display.blit(text, [85, 0])

        # Score de la partida actual
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        # Record global del entrenamiento
        text_record = font.render("Record: " + str(self.record), True, WHITE)
        self.display.blit(text_record, [0, 30])

        # Calcula el tiempo transcurrido en segundos y lo muestra en la pantalla
        self.seconds = (pygame.time.get_ticks() - self.start_time) // 1000
        self.last_time = self.seconds
        text_time = font.render("Time: " + str(self.seconds) + "s", True, WHITE)
        self.display.blit(text_time, [95, 30])

        pygame.display.update()



