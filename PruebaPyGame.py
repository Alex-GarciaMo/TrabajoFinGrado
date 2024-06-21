import numpy as np

class Tablero():
    def __init__(self, alto, ancho):
        self.ancho = ancho
        self.alto = alto
        self.casillas = np.zeros((alto//20, ancho//20), dtype=int)

    def Print_Casillas(self):
        print(self.casillas)











if __name__ == "__main__":
    ancho = 640
    alto = 480
    mi_tablero = Tablero(alto, ancho)
    mi_tablero.casillas[120//20, 620//20] = 1
    mi_tablero.Print_Casillas()


    print(mi_tablero.casillas[120//20, 620//20])