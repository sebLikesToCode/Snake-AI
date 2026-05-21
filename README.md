# Snake-AI
one of my biggest projects ever, a snake ai made with PYtorch! it uses deep q learning, and takes 26 inputs. i will go over each file below

# 'AI_Snake_Environment.py'
this is the actual gameboard handled behund the scenes. it holds the board in memory, handles the movement, and decides the reward for each outcome. It also handles the apples and input vectors for the ai.
* I used a BFS to chech spaces on the board an an input, but it lagged python waaaaaaay to much. i used Numba to compile the BFS to machine code to make it run smoother.
the inputs include:
* decimal space checks using flood fill
* raycast decimals for distance to walls
* binary flags for direction to tail and apple

# 'Agent.py'
agent.py is the bridge between the neural network weights and the gameboard. it has the memory, and decides when to move randomly (epsilon) or use its brain.
* agent.py has the memory deque thta holds up to 200,000 frames of gameplay that the ai takes a random sample from and uses to train its future moves.
* safety: if the random epsilon move will kill the snake, it wont let it random mve. this prevents the ai from overfilling with crashes while it cant control its body.
* gamma math: uses the bellman equasion to weight long term survival against immediate rewards

# 'Model.py'
model.py is the brain. it uses pytorch to set up the neural network withe the parameters passes by agent.py.
* it sets up and passes the 26 inputs through 3 hidden layers of nehrons. the etup looks like '26 -> 512 -> 256 -> 128 -> 3. the big hidden layers let it fighure out complex pathfinding and have spatial reasoning.
* the final layer passes a list of 3 outputs like 0, 1, 0. where the 1 is depends if the snake will go left, right, or continue straight.
* it holds the save and load functions that saves the brain even if the script is colsed. that is what is in the included model folder.

# 'Snake AI.py'
this is the actual script you run. it sets up the training environments and manages the backpropagation.
* parelell environments: this script runs multiple parallel snakes to speed up training.
* file reDING: this script reads the control.txt on the fly and and lets me change parameters like epsilon without stopping the script.
* kick: if the ai plays 3000 games and its average is still below 0.8, it will boodt the epsilon up higher to get it unstuck.

# 'Playground AI.py'
this lets you watch the snake plays, as it doesnt show in the training script.
* overrides the epsilon to let you watch the snake play with no randomness
* renders the grid with ascii characters and nice ansi escape colors.
