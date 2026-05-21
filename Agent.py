from Model import Linear_QNet
import torch
import random
import numpy as np
from collections import deque
import os
from AI_Snake_Environment import AiGameLoop
MAX_MEMORY = 200_000
BATCH_SIZE = 1024

GAMMA = 0.95


class SnakeAgent:
    def __init__(self, load_model=False):
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        # Old: (18, 256, 128, 64, 3)
        self.LR = 0.0001  # Learning Rate
        self.model = Linear_QNet(26, 512, 256, 128, 3)
        self.epsilon = 2
        self.base = 2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = torch.nn.MSELoss()
        self.gamma = GAMMA
        success = False
        if load_model:
            success = self.model.load()

        if success:
            # If we loaded a brain, skip the "dumb" phase
            print("Resuming training with a trained brain.")
        else:
            # If load failed or wasn't requested, start fresh
            self.n_games = 0
            print("Starting training from game 0.")
    def get_action(self, state, game):
        # Exponential decay is better for parallel: drops fast but stays useful
        self.epsilon = max(self.base, self.epsilon - (self.n_games // 15))
        final_move = [0, 0, 0]

        if random.randint(0, 500) < self.epsilon:

            move_idx = random.randint(0, 2)
            # Collision avoidance during exploration
            future_pt = self.get_future_point(move_idx, game)
            if game.is_collision(future_pt):
                # If random move kills us, try a different one
                for safe_idx in [0, 1, 2]:
                    if not game.is_collision(self.get_future_point(safe_idx, game)):
                        move_idx = safe_idx
                        break
            final_move[move_idx] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def update_parameters_from_file(self, games):
        control_path = 'control.txt'
        if not os.path.exists(control_path): return

        try:
            with open(control_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if ':' not in line: continue
                key, val = line.strip().split(':', 1)  # Limit split to 1 to be safe
                key, val = key.strip(), val.strip()

                # Update Agent Globals
                if key == 'lr':
                    new_lr = float(val)
                    if self.LR != new_lr:
                        self.LR = new_lr
                        for pg in self.optimizer.param_groups: pg['lr'] = self.LR
                elif key == 'epsilon_base':
                    self.base = float(val)
                elif key == 'epsilon':
                    self.epsilon = float(val)

                # Update ALL Game Environments at once
                elif key in ['apple', 'death', 'step']:
                    num_val = float(val)
                    for g in games:
                        if key == 'apple': g.reward_apple = int(num_val)
                        if key == 'death': g.reward_death = int(num_val)
                        if key == 'step': g.step_reward = num_val
        except Exception as e:
            print(f"File Sync Error: {e}")

    def train(self, state, action, reward, next_state, done):
        if len(np.array(state).shape) == 1:
            # If it's just 1 move, make it a list of 1
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = (done,)  # Make it a tuple

        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        prediction = self.model(state)
        target = prediction.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def get_future_point(self, move_idx, game):
        head = game.plr_XY[0]
        # Clockwise: Up, Right, Down, Left
        cw = ["w", "d", "s", "a"]
        curr_idx = cw.index(game.direction)

        # Figure out new direction string
        if move_idx == 0:  # Straight
            new_dir = cw[curr_idx]
        elif move_idx == 1:  # Right
            new_dir = cw[(curr_idx + 1) % 4]
        else:  # Left
            new_dir = cw[(curr_idx - 1) % 4]

        x, y = head[0], head[1]
        if new_dir == "w":
            y -= 1
        elif new_dir == "s":
            y += 1
        elif new_dir == "a":
            x -= 1
        elif new_dir == "d":
            x += 1

        return [x, y]
