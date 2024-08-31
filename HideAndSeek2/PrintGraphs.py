import train

block_size = 1000

file_path = 'metrics/predator_metrics.csv'
# file_path = 'Graficas/Smooth, sin dropout, distancia normalizada/predator_metrics.csv'
# file_path = 'Graficas/Mala recompensa dirigida y sin recompensa indiferida/predator_metrics.csv'
# file_path = 'Graficas/Modelo final, MSE sin obstáculos/predator_metrics.csv'

# Actualizar el gráfico después de cada bloque de block_size partidas
train.update_evolutionary_plot(file_path, 1, block_size)

file_path = 'metrics/prey_metrics.csv'
# file_path = 'Graficas/Smooth, sin dropout, distancia normalizada/preys_metrics.csv'
# file_path = 'Graficas/Mala recompensa dirigida y sin recompensa indiferida/prey_metrics.csv'
# file_path = 'Graficas/Modelo final, MSE sin obstáculos/prey_metrics.csv'

train.update_evolutionary_plot(file_path, 0, block_size)

