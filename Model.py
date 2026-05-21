import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, output_size):
        super().__init__()
        # Input to first hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        # First hidden layer to second hidden layer
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        # Second hidden layer to output
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)

        self.linear4 = nn.Linear(hidden_size3, output_size)
    def forward(self, x):
        # Activation at each step to handle non-linear logic
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, file_name='model.pth'):
        import os
        # Finds the folder where Model.py lives
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_folder_path = os.path.join(base_path, 'model')

        # Create the folder if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        full_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), full_path)
        print(f"Model saved")

    def load(self, file_name='model.pth'):
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_path, 'model', file_name)

        if os.path.exists(full_path):
            try:
                self.load_state_dict(torch.load(full_path, map_location='cpu'))
                self.eval()
                print(f"!!! SUCCESS: Loaded {full_path} !!!")
                return True
            except Exception as e:
                print(f"!!! ERROR: Could not load model: {e} !!!")
                return False
        else:
            print(f"!!! NOTICE: No file found at {full_path} !!!")
            return False