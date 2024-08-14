import numpy as np
from enum import Enum

class Tablero():
    def __init__(self, alto, ancho):
        self.ancho = ancho
        self.alto = alto
        self.casillas = np.zeros((alto//20, ancho//20), dtype=int)

    def Print_Casillas(self):
        print(self.casillas)



class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4







if __name__ == "__main__":
    ancho = 640
    alto = 480
    mi_tablero = Tablero(alto, ancho)
    mi_tablero.casillas[120//20, 620//20] = 1
    mi_tablero.Print_Casillas()


    print(mi_tablero.casillas[120//20, 620//20])

    x = 4
    y = 4

    lista= []

    # Abajo
    # for i in range(1, 4):
    #     lista.append([x, y + 1*i])
    #     for j in range(1, i):
    #         lista.append([x - 1*j, y + 1*i])
    #         lista.append([x + 1*j, y + 1*i])

    # Arriba
    # for i in range(1, 4):
    #     lista.append([x, y - 1*i])
    #     for j in range(1, i):
    #         lista.append([x - 1*j, y - 1*i])
    #         lista.append([x + 1*j, y - 1*i])

    # Derecha
    # for i in range(1, 4):
    #     lista.append([x + 1*i, y])
    #     for j in range(1, i):
    #         lista.append([x + 1*i, y - 1*j])
    #         lista.append([x + 1*i, y + 1*j])

    # Izquierda

    for i in range(1, 4):
        lista.append([x - 1*i, y])
        for j in range(1, i):
            lista.append([x - 1*i, y - 1*j])
            lista.append([x - 1*i, y + 1*j])

    print(lista)
    a = 1
    print(Direction(a))

    print(21%10)