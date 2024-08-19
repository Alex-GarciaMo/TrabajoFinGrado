# Código creado por Alejandro García Moreno.
# TFG 2023-2024: Desarrollo de un modelo de Aprendizaje por Refuerzo para el juego del Pilla Pilla

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
# font = pygame.font.Font('../HideAndSeek/arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20


class Tablero():
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
        self.casillas = np.zeros((self.alto // BLOCK_SIZE, self.ancho // BLOCK_SIZE), dtype=int)

    def Resetear_Tablero(self):
        self.casillas = np.zeros((self.alto // BLOCK_SIZE, self.ancho // BLOCK_SIZE), dtype=int)

    def Print_Tablero(self):
        print(self.casillas)


class HideAndSeekGameAI:

    def __init__(self, predators, preys, n_games, w=640, h=480):
        self.block_size = BLOCK_SIZE
        self.score = 0
        self.record = 0
        self.frame_iteration = 0
        self.n_games = n_games
        self.match_time = 10
        self.w = w
        self.h = h
        self.fixed_reward = 10
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('IA Pilla_Pilla')
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()  # Inicializa el tiempo de inicio del juego
        self.seconds = 0
        self.predators = predators
        self.preys = preys
        self.n_predators = len(predators)
        self.n_preys = len(preys)
        self.board = Tablero(w, h)
        self.preys_metrics = []
        self.predators_metrics = []
        self.reset()

    def reset(self):
        self.board.Resetear_Tablero()  # Resetear tablero

        # Se recolocan los agentes
        self.place_predators()
        self.place_prey()

        # Se resetean los contadores
        self.score = 0
        self.frame_iteration = 0
        self.start_time = pygame.time.get_ticks()  # Reinicia el tiempo de inicio del juego
        self.seconds = 0

    def place_predators(self):
        separation = 0

        for predator in self.predators:
            # init agent state
            predator.direction = Direction.RIGHT

            predator.head = Point(self.w / 2 - separation, self.h / 2)
            self.board.casillas[int(predator.head.y // BLOCK_SIZE), int(predator.head.x // BLOCK_SIZE)] = 2
            separation = + 20

    def place_prey(self):
        for prey in self.preys:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            direction = Direction(random.randint(1, 4))

            while self.board.casillas[y // BLOCK_SIZE, x // BLOCK_SIZE] != 0:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            prey.head = Point(x, y)
            prey.direction = direction
            self.board.casillas[y // BLOCK_SIZE, x // BLOCK_SIZE] = 1

    def play_step(self, action, agent):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        catch = self.move(action, agent)  # update the head

        # 3. check if game over
        reward = 4  # Valor base por moverse
        score = 0
        game_over = False
        agent.metrics['Game'].append(self.n_games)
        agent.metrics['Score'].append(self.score)

        # Hasta que colisione
        if self.is_collision(agent, agent.head):
            game_over = True
            reward = -self.fixed_reward
            agent.metrics['Reward'].append(reward)
            return reward, game_over, score
        # o hayan transcurrido X segundos
        elif self.seconds > self.match_time:
            game_over = True
            return reward, game_over, score

        # 4. Si ha habido caza
        if catch:
            if agent.type:  # Si es depredador
                for prey in self.preys:  # Busca la presa que ha cazado
                    if agent.head == prey.head:
                        score += 1
                        reward = self.fixed_reward
                        self.opponent_catch(prey)  # Añadimos recompensa negativa a la presa capturada
                        prey.train_long_memory()  # Entrenar antes de ser removido
                        prey.model.save(agent.file_name)  # Guardar modelo
                        prey.metrics_manager(self)  # Guardamos la métrica
                        self.preys.remove(prey)  # Eliminar presa cazada


            else:  # Si es presa
                for predator in self.predators:  # Se busca al cazador que le ha cazado
                    if agent.head == predator.head:
                        score += 1
                        reward = -self.fixed_reward
                        self.opponent_catch(predator)  # Añadimos recompensa positiva al depredador
                        game_over = True

        agent.metrics['Reward'].append(reward)

        return reward, game_over, score

    # Dar recompensa en función de si es capturado o si ha capturado
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
            agent.remember(state, action, reward, next_state, done)

    def end_time(self):
        for predator in self.predators:
            last_memory = predator.memory[-1]
            state, action, reward, next_state, done = last_memory
            reward = -self.fixed_reward
            done = True
            predator.remember(state, action, reward, next_state, done)

        for prey in self.preys:
            last_memory = prey.memory[-1]
            state, action, reward, next_state, done = last_memory
            reward = self.fixed_reward
            done = True
            prey.remember(state, action, reward, next_state, done)

    def is_collision(self, agent, pt=None):
        if pt is None:
            pt = agent.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    def update_ui(self):

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

        # Calcular puntos del triángulo en función de la dirección de los depredadores
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

    def move(self, action, agent):
        # [up, right, left, down]
        if np.array_equal(action, [1, 0, 0, 0]):
            agent.direction = Direction.UP
        elif np.array_equal(action, [0, 1, 0, 0]):
            agent.direction = Direction.RIGHT
        elif np.array_equal(action, [0, 0, 1, 0]):
            agent.direction = Direction.LEFT
        else:  # [0, 0, 0, 1]
            agent.direction = Direction.DOWN

        x = agent.head.x
        y = agent.head.y
        if agent.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif agent.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif agent.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif agent.direction == Direction.UP:
            y -= BLOCK_SIZE

        # Si ha cazado a la presa o no
        catch = False

        # Vaciar la antigua casilla del tablero y moverse a la siguiente
        self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] = 0
        agent.head = Point(x, y)  # Actualizar posición del agente

        # Comprobar que no se haya ido fuera del límite
        if 0 <= x < self.w and 0 <= y < self.h:
            # Comprobar si en esa casilla había un oponente
            if self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] != agent.type + 1 \
                    and self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] > 0:
                catch = True
                self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] = 2
            # Actualizar el tablero en función del tipo del agente
            else:
                if agent.type:
                    self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] = 2
                else:
                    self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] = 1
        return catch