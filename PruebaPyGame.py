class Tablero():
    def __init__(self, alto, ancho):
        self.ancho = ancho
        self.alto = alto
        self.casillas = self.Generar_Casillas()

    def Generar_Casillas(self):
        total_casillas = []
        for i in range(0,self.alto,20):
            fila = []
            for j in range(0, self.ancho,20):
                fila.append(0)
            total_casillas.append(fila)

        # for fila in total_casillas:
        #     print(fila)
        return total_casillas










if __name__ == "__main__":
    ancho = 640
    alto = 480
    mi_tablero = Tablero(alto, ancho)

    mi_tablero.casillas[0][1] = 1

    altura = 0
    anchura = 0
    for fila in mi_tablero.casillas:
        altura += 1
        print(fila)
        for cuadro in fila:
            anchura += 1

    print(altura, anchura//altura)

    #Prueba