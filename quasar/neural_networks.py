import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class NeuralNetworkModels:
    @staticmethod
    def train_ann(X, y, hidden_size=10, epochs=100, learning_rate=0.01):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        input_size = X_tensor.shape[1]
        output_size = 1

        model = MLP(input_size, hidden_size, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        # create mesh grid
        X_np = np.array(X)
        x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
        y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))

        grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        with torch.no_grad():
            Z = model(grid_tensor)
            Z = (Z > 0.5).float().numpy()

        Z = Z.reshape(xx.shape)

        return {
            'loss_history': loss_history,
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist()
        }
