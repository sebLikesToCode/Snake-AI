import psutil
import os
import torch # The main AI engine
import numpy as np # Handles the lists of numbers (states)
import random # For the "exploration" moves
from collections import deque # For the AI's memory bank
from Model import Linear_QNet
# From your Game file, import the Snake class
from AI_Snake_Environment import AiGameLoop
from Agent import SnakeAgent
import torch.nn as nn # For the loss functions (how we measure mistakes)
import torch.optim as optim # For the optimizer (how we fix the brain)
import random

def set_high_priority():
    p = psutil.Process(os.getpid())
    # HIGH_PRIORITY_CLASS ensures it runs faster than standard apps
    # but doesn't freeze your mouse/keyboard like REALTIME_PRIORITY_CLASS might.
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    print("Process priority set to HIGH.")

def train_parallel():
    num_envs = 10
    games = [AiGameLoop() for _ in range(num_envs)]
    agent = SnakeAgent(load_model=True)
    high_score = 0
    recent_scores = deque(maxlen=500)  # Track the last 10 games
    BATCH_SIZE = 1024
    current_loss = 0
    watching = False
    games_since_last_train = 0
    print(f"Parallel Training Started. Showing all game results...")
    games_since_last_save = 0
    while True:
        # print("Iterating...") # Turn this off now, we know the loop starts
        for i in range(num_envs):
            game = games[i]

            # TEST 1
            state_old = game.get_state()


            # TEST 2 - This is the most likely failure point
            final_move = agent.get_action(state_old, game)

            done = False
            # TEST 3
            reward, done, score = game.step(final_move)

            state_new = game.get_state()

            #agent.train(state_old, final_move, reward, state_new, done)
            agent.memory.append((state_old, final_move, reward, state_new, done))

            #if keyboard.is_pressed("space") and i == 0:
                #game.print_board()
                #time.sleep(0.01)

            if done:

                agent.n_games += 1
                recent_scores.append(score)
                avg_score = sum(recent_scores) / len(recent_scores)

                games_since_last_train += 1
                games_since_last_save += 1

                # 1. TRAIN ONLY EVERY 5 GAMES
                if games_since_last_train >= 5:
                    if len(agent.memory) > BATCH_SIZE:
                        mini_sample = random.sample(agent.memory, BATCH_SIZE)
                        states, actions, rewards, next_states, dones = zip(*mini_sample)

                        current_loss = agent.train(states, actions, rewards, next_states, dones)
                    games_since_last_train = 0  # Reset this here!

                # 2. SAVE EVERY 100 GAMES (Saving too often slows you down)
                if games_since_last_save >= 100:
                    agent.model.save()
                    games_since_last_save = 0

                # 3. KICK LOGIC
                if avg_score < 0.8 and agent.n_games > 3000:
                    agent.n_games = max(0, agent.n_games - 1500)
                    print(f"--- KICK ACTIVATED ---")
                if agent.n_games % 100 == 0:

                    # The real Flood Fill values (Indices 0-2)
                    print(f"Radar FF Check (S/R/L): {state_old[0:3]}")

                    # The real Line-of-Sight Radars (Indices 3-5)
                    print(f"Distance Radar (S/R/L): {state_old[3:6]}")

                    # If you want to see where it thinks the apple is (Indices 10-13)
                    print(f"Apple Binary (L/R/U/D): {state_old[10:14]}")
                # 4. PRINT ONLY EVERY 20 GAMES
                if agent.n_games % 20 == 0:
                    print(f"Game: {agent.n_games} | Avg: {avg_score:.1f} | Score: {score} | Epsilon: {agent.epsilon} | Loss: {current_loss}")


                if score > high_score:
                    high_score = score
                    agent.model.save()
                    print(f"*** NEW RECORD: {score} ***")

                # !!! REMOVED THE REDUNDANT TRAINING BLOCK FROM HERE !!!

                game.reset()


set_high_priority()
if __name__ == "__main__":
    train_parallel()
