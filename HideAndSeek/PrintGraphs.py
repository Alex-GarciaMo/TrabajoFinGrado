import train

file_path = 'metrics/predator_metrics.csv'

block_size = 100

# Actualizar el gráfico después de cada bloque de block_size partidas
train.update_evolutionary_plot(file_path, 1, block_size)

file_path = 'metrics/prey_metrics.csv'
train.update_evolutionary_plot(file_path, 0, block_size)

