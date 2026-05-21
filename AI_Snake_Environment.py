import numpy as np
import random
from numba import njit
from numba.typed import List
wall = "██"
space = "  "
plr = "\033[32m██\033[0m"
apple = "\033[31m\uE0B6\uE0B4\033[0m"


@njit(nopython=True, cache=True)
def turbo_bfs(grid, start_x, start_y, max_check):
    board_size = grid.shape[0]
    # Create a local copy of visited so we don't ruin the original grid
    visited = grid.copy()

    queue = np.empty((board_size * board_size, 2), dtype=np.int32)
    queue[0, 0] = start_x
    queue[0, 1] = start_y
    visited[start_x, start_y] = 1

    count = 0
    head = 0
    tail = 1

    while head < tail and count < max_check:
        cx, cy = queue[head, 0], queue[head, 1]
        head += 1
        count += 1

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < board_size - 1 and 0 < ny < board_size - 1:
                if visited[nx, ny] == 0:
                    visited[nx, ny] = 1
                    queue[tail, 0] = nx
                    queue[tail, 1] = ny
                    tail += 1
    return count
class AiGameLoop:
    def __init__(self):
        self.size = 29  # Hardcoded size for internal logic
        self.reset()
        self.reward_apple = 300
        self.reward_death = -1000
        self.reward_step = -0.02
    def reset(self):

        self.gameEnd = False
        self.plr_XY = [[14, 13], [14, 14], [14, 15]]  # Start with a small body
        self.snake_size = 3

        self.apple_summon()
        self.direction = "w"  # Start moving up
        self.score = 0
        self.counter = 0

    def is_collision(self, pt=None):
        if pt is None: pt = self.plr_XY[0]
        # Wall boundaries
        if pt[0] <= 0 or pt[0] >= self.size - 1 or pt[1] <= 0 or pt[1] >= self.size - 1:
            return True
        # Body collision (excluding the head)
        if pt in self.plr_XY[1:]:
            return True
        return False

    def move(self, action):
        cw = ["w", "d", "s", "a"]
        idx = cw.index(self.direction)

        # Action: [1,0,0] = Straight, [0,1,0] = Right Turn, [0,0,1] = Left Turn
        if action == [1, 0, 0]:
            new_dir = cw[idx]
        elif action == [0, 1, 0]:
            new_dir = cw[(idx + 1) % 4]
        else:
            new_dir = cw[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.plr_XY[0]
        if self.direction == "w":
            y -= 1
        elif self.direction == "s":
            y += 1
        elif self.direction == "a":
            x -= 1
        elif self.direction == "d":
            x += 1

        self.plr_XY.insert(0, [x, y])

    def step(self, action):

        self.counter += 1

        # 1. Calculate distance BEFORE moving
        # Using Manhattan distance: abs(x1-x2) + abs(y1-y2)


        # 2. MOVE ONLY ONCE
        self.move(action)

        # 3. Check for Death (Collision)
        if self.is_collision():
            self.gameEnd = True
            return self.reward_death, True, self.score

        # 4. Check for Food (Apple)
        if self.plr_XY[0] == self.apple_XY:
            self.score += 1
            self.snake_size += 1
            self.apple_summon()
            self.counter = 0
            return self.reward_apple, False, self.score  # Keep the big reward for the actual eat

        # 5. Normal Move: Trim the tail to keep length correct
        # This prevents the "doubling" / getting long for no reason
        if len(self.plr_XY) > self.snake_size:
            self.plr_XY.pop()

        # 6. Check for Starvation (based on snake length)
        if self.counter >= 200 + (len(self.plr_XY) * 5):
            self.gameEnd = True
            return self.reward_death, True, self.score



        return self.reward_step, False, self.score

    def apple_summon(self):
        while True:

            self.apple_XY = [random.randint(1, self.size - 2), random.randint(1, self.size - 2)]
            if self.apple_XY not in self.plr_XY:
                break

    def get_distance(self, direction_vector):
        head = self.plr_XY[0]
        distance = 0
        current_pt = [head[0], head[1]]
        while True:
            distance += 1
            current_pt[0] += direction_vector[0]
            current_pt[1] += direction_vector[1]
            if self.is_collision(current_pt):
                break
        # Return raw distance in grid units
        return distance

    def get_state(self):
        head = self.plr_XY[0]
        tail = self.plr_XY[-1]
        cw = ['w', 'd', 's', 'a']
        idx = cw.index(self.direction)

        # Relative direction vectors
        dir_vectors = {'w': [0, -1], 'd': [1, 0], 's': [0, 1], 'a': [-1, 0]}
        v_straight = dir_vectors[cw[idx]]
        v_right = dir_vectors[cw[(idx + 1) % 4]]
        v_left = dir_vectors[cw[(idx - 1) % 4]]

        # --- NEW: CALCULATE GHOST HEADS FOR FLOOD FILL ---
        next_s = [head[0] + v_straight[0], head[1] + v_straight[1]]
        next_r = [head[0] + v_right[0], head[1] + v_right[1]]
        next_l = [head[0] + v_left[0], head[1] + v_left[1]]

        ff_s = self.get_area_volume(next_s)
        ff_r = self.get_area_volume(next_r)
        ff_l = self.get_area_volume(next_l)

        # 1. Normalized Apple Distance
        dist_to_apple = abs(head[0] - self.apple_XY[0]) + abs(head[1] - self.apple_XY[1])
        apple_dist_norm = 1.0 - (dist_to_apple / (self.size * 2))

        # 2. Tail Location
        tail_up = 1.0 if tail[1] < head[1] else 0.0
        tail_down = 1.0 if tail[1] > head[1] else 0.0
        tail_left = 1.0 if tail[0] < head[0] else 0.0
        tail_right = 1.0 if tail[0] > head[0] else 0.0

        # 3. Global Wall Distances
        dist_wall_u = 1.0 - (head[1] / self.size)
        dist_wall_d = head[1] / self.size
        dist_wall_l = 1.0 - (head[0] / self.size)
        dist_wall_r = head[0] / self.size

        # 4. Improved Radar
        def norm_dist(v):
            d = self.get_distance(v)
            return max(0.0, 1.0 - (d / self.size))

        state = [
            # [0, 1, 2] - FLOOD FILL (3 Floats: 0.0 to 1.0)
            ff_s, ff_r, ff_l,

            # [3, 4, 5] - RADAR DISTANCE (3 Floats: 0.0 to 1.0)
            norm_dist(v_straight), norm_dist(v_right), norm_dist(v_left),

            # [6-9] - CURRENT DIRECTION (4 Binary)
            self.direction == 'a', self.direction == 'd',
            self.direction == 'w', self.direction == 's',

            # [10-13] - APPLE DIRECTION (4 Binary)
            self.apple_XY[0] < head[0], self.apple_XY[0] > head[0],
            self.apple_XY[1] < head[1], self.apple_XY[1] > head[1],

            # [14-17] - WALL PROXIMITY (4 Floats)
            dist_wall_u, dist_wall_d, dist_wall_l, dist_wall_r,

            # [18-22] - TARGETING & TAIL (5 Floats)
            apple_dist_norm,
            tail_up, tail_down, tail_left, tail_right,

            # [23-25] - OLD OPEN SPACE (Normalized to 29.0)
            self.count_free_space(head, v_straight),
            self.count_free_space(head, v_right),
            self.count_free_space(head, v_left),
        ]
        return np.array(state, dtype=float)

    def get_area_volume(self, start_node):
        if self.is_collision(start_node):
            return 0.0

        # Create a grid of the board (0 = empty, 1 = obstacle)
        # Doing this with NumPy is much faster than typed lists
        grid = np.zeros((self.size, self.size), dtype=np.int8)

        # Fill walls
        grid[0, :] = 1
        grid[self.size - 1, :] = 1
        grid[:, 0] = 1
        grid[:, self.size - 1] = 1

        # Fill snake body
        for p in self.plr_XY:
            grid[p[0], p[1]] = 1

        # Call turbo_bfs with the grid
        # We set max_check to 200 (enough for the snake to know if it's trapped)
        count = turbo_bfs(grid, start_node[0], start_node[1], 200)

        # Normalize based on snake size
        return min(1.0, count / (len(self.plr_XY) + 2))

    def print_board(self):
        # Clear the console
        print("\033[H", end="")

        # Header
        header = f" Score: {self.score} "
        print(wall * self.size + header)

        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if [x, y] == self.plr_XY[0]:  # Head
                    row += plr
                elif [x, y] in self.plr_XY[1:]:  # Body
                    row += plr
                elif [x, y] == self.apple_XY:  # Apple
                    row += apple
                elif x == 0 or x == self.size - 1 or y == 0 or y == self.size - 1:
                    row += wall
                else:
                    row += space
            print(row)

    def count_free_space(self, start_pt, direction_vector):
        # A simplified version: check a 3x3 area around the potential head
        # or look 'N' steps ahead in that direction
        free_steps = 0
        test_pt = [start_pt[0], start_pt[1]]
        # Check up to 10 steps ahead in current direction
        for _ in range(20):
            test_pt[0] += direction_vector[0]
            test_pt[1] += direction_vector[1]
            if not self.is_collision(test_pt):
                free_steps += 1
            else:
                break
        return free_steps / 29.0