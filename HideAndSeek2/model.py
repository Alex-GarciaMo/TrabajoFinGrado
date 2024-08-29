# Código creado por Alejandro García Moreno.
# TFG 2023-2024: Desarrollo de un modelo de Aprendizaje por Refuerzo para el juego del Escondite

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# El modelo DQN utiliza redes neuronales para el cálculo de Q_value.
# Mejoras implementadas:
#   - Double Q Network: Utilizamos target_network para evitar X <¡¡¡FALTAAAA!!!>
#   - Replay memory: Es el método del agente llamado train_long_memory
#   - Epsilon-greedy: Al comenzar el entrenamiento de un nuevo modelo, el agente explora de manera aleatoria
#   - Recompensas dirigidas: El agente recibe recompensas según lo cerca o lejos de su oponente.
#  ¿- Dropout?

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_actions):
        super(DeepQNetwork, self).__init__()  # Red neuronal de 3 capas
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        actions = self.fc3(x)
        return actions

    def save(self, file_name='model.pth'):
        model_folder_path = 'model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = 'model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
        else:
            raise FileNotFoundError(f"No model found at {file_name}")


# Entrenador del agente, recibe los estados y recompensas para entrenar el modelo y la red neuronal
class DQNTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # self.criterion = nn.SmoothL1Loss()  # Huber Loss
        self.criterion = nn.MSELoss()  # Huber Loss

    # Método para calcular Q_new y entrenar
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Si se llama a train_step con train_short_memory sólo se recibe el último movimiento
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()

        # Si se llama a train_step con train_long_memory, se debe recorrer todos los datos del trozo de memoria.
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx] and reward[idx] < 10:
                # with torch.no_grad():
                next_action = torch.argmax(self.model(next_state[idx]))  # Calcular mejor acción

                # Calcular Q_new con target_model
                q_new = reward[idx] + self.gamma * self.target_model(next_state[idx])[next_action]

            action_index = (action[idx] == 1).nonzero(as_tuple=True)[0].item()
            target[idx][action_index] = q_new

        # Limpiar gradientes anteriores
        self.optimizer.zero_grad()
        # Calcular la pérdida entre el objetivo y la predicción
        loss = self.criterion(target, pred)
        # Calcular gradientes para los parámetros del modelo
        loss.backward()
        # Actualizar los pesos del modelo
        self.optimizer.step()

        return q_new, loss.item()
