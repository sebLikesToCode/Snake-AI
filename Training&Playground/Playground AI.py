import torch
import numpy as np
import os
import time
import psutil
from AI_Snake_Environment import AiGameLoop
from Agent import SnakeAgent


def set_low_priority():
    """ Sets to BELOW_NORMAL so it doesn't lag your trainer or PC """
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def run_visualizer():
    # 1. Setup Environment
    game = AiGameLoop()

    # 2. Setup Agent (Load the current brain)
    # Ensure load_model=True is set in your SnakeAgent __init__
    agent = SnakeAgent(load_model=True)

    print(f"Visualizer Started. Loading Model...")
    print("Press Ctrl+C to quit.")

    while True:
        # Get current state
        state_old = game.get_state()

        # 3. Get Action with Epsilon = 0
        # This forces the snake to use 100% logic, no random guesses.
        # If your get_action doesn't accept an epsilon override,
        # you can manually set: agent.epsilon = 0
        agent.epsilon = 0
        final_move = agent.get_action(state_old, game)

        # 4. Step the game
        reward, done, score = game.step(final_move)

        # 5. Visuals
        game.print_board()

        # Adjust this speed to your liking (0.05 is fast, 0.1 is relaxed)
        time.sleep(0.06)

        if done:
            print(f"Game Over! Final Score: {score}")

            # Optional: Reload the model from disk to see Trainer's progress
            # agent.model.load_state_dict(torch.load(agent.model_path))

            game.reset()


if __name__ == "__main__":
    set_low_priority()
    try:
        run_visualizer()
    except KeyboardInterrupt:
        print("\nVisualizer closed.")