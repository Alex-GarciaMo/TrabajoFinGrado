import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# font = pygame.font.Font('../Snake/arial.ttf', 25)
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


class SnakeGameAI:

    def __init__(self, agents, foods, w=640, h=480):
        self.score = 0
        self.food = []
        self.frame_iteration = 0
        self.n_games = 0
        self.match_time = 10
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('IA Pilla_Pilla')
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()  # Inicializa el tiempo de inicio del juego
        self.seconds = 0
        self.agents = agents
        self.n_agents = len(agents)
        self.n_foods = foods
        self.board = Tablero(w, h)
        self.reset()

    def reset(self, ):
        self.board.Resetear_Tablero() # Resetear tablero
        # Se inicializa la posición de los depredadores
        separation = 0
        if self.agents:
            for agent in self.agents:
                # init game state
                agent.direction = Direction.RIGHT

                agent.head = Point(self.w / 2 - separation, self.h / 2)
                self.board.casillas[int(agent.head.y // BLOCK_SIZE), int(agent.head.x // BLOCK_SIZE)] = 2
                separation = + 60

        # Se resetean los contadores
        self.score = 0
        self.food = []
        self._place_food()  # Se recolocan las presas
        self.frame_iteration = 0
        self.start_time = pygame.time.get_ticks()  # Reinicia el tiempo de inicio del juego
        self.seconds = 0


    def _place_food(self):
        self.food = []
        for food in range(0, self.n_foods):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            # print(y//BLOCK_SIZE,x//BLOCK_SIZE)
            while self.board.casillas[y//BLOCK_SIZE, x//BLOCK_SIZE] != 0:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food.append(Point(x, y))
            self.board.casillas[y//BLOCK_SIZE, x//BLOCK_SIZE] = 1

    def play_step(self, action, agent):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self.move(action, agent)  # update the head

        # 3. check if game over
        reward = 0
        game_over = False

        # Hasta que colisione o hayan transcurrido X segundos
        if self.is_collision(agent, agent.head) or self.seconds > self.match_time:
            game_over = True
            reward = -10

            return reward, game_over, self.score, self.seconds

        # 4. place new food or just move
        for food in self.food:
            if agent.head == food:
                self.score += 1
                reward += 10
                self.food.remove(food)
                if not self.food:
                    self._place_food()

        # 5. update ui and clock
        # self.update_ui(agent)
        # self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score, self.seconds

    def is_collision(self, agent, pt=None, ):
        if pt is None:
            pt = agent.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    def update_ui(self, agents):

        self.display.fill(BLACK)

        for agent in agents:
            # Calcular puntos del triángulo en función de la dirección
            x = agent.head.x
            y = agent.head.y
            if agent.direction == Direction.UP:  # Punta arriba
                vertices = ((x, y + BLOCK_SIZE), (x + BLOCK_SIZE, y + BLOCK_SIZE), (x + BLOCK_SIZE / 2, y))
            elif agent.direction == Direction.DOWN:  # Punta abajo
                vertices = ((x, y), (x + BLOCK_SIZE, y), (x + BLOCK_SIZE / 2, y + BLOCK_SIZE))
            elif agent.direction == Direction.RIGHT:  # Punta derecha
                vertices = ((x, y), (x, y + BLOCK_SIZE), (x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2))
            else:  # Punta izquierda
                vertices = ((x, y + BLOCK_SIZE / 2), (x + BLOCK_SIZE, y), (x + BLOCK_SIZE, y + BLOCK_SIZE))

            # Calcular cuadrante del agente
            # Distribución del tablero  w=640, h=480
            # Primer cuadrante: 1 = eje x <= 320, eje y <= 240
            # Segundo cuadrante: 2 = eje x > 320, eje y <= 240
            # Tercer cuadrante: 3 = eje x <= 320, eje y > 240
            # Cuarto cuadrante: 4 = eje x > 320, eje y > 240
            if x <= 320 and y <= 240:
                agent.cuadrante = 1
            elif x > 320 and y <= 240:
                agent.cuadrante = 2
            elif x <= 320 and y > 240:
                agent.cuadrante = 3
            else:
                agent.cuadrante = 4

            # Dibujar la cabeza del agente
            pygame.draw.polygon(self.display, BLUE1, vertices)

        # Dibujar las comidas
        for food in self.food:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        # Calcula el tiempo transcurrido en segundos y lo muestra en la pantalla
        self.seconds = (pygame.time.get_ticks() - self.start_time) // 1000
        self.last_time = self.seconds
        text_time = font.render("Time: " + str(self.seconds) + "s", True, WHITE)
        self.display.blit(text_time, [0, 30])

        pygame.display.update()

    def move(self, action, agent):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(agent.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        agent.direction = new_dir

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



        self.board.casillas[int(agent.head.y//BLOCK_SIZE), int(agent.head.x//BLOCK_SIZE)] = 0
        agent.head = Point(x, y)

        if 0 <= x < self.w and 0 <= y < self.h:
            self.board.casillas[int(agent.head.y//BLOCK_SIZE), int(agent.head.x//BLOCK_SIZE)] = 2