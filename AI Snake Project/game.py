import random
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import deque
from dash_app import save_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

from dqn import DQN, DQN_CNN
from constants import (
CELL_SIZE, SAVE_INTERVAL, DIRS_OFFSET, TRAINING_FRAMES,
BATCH_SIZE, training_interval, current_dir, folder_path,
LR, BOARD_RANGE, LOAD_MODEL_PATH, IS_TRAINING, EPSILON_THRESHOLD, USE_EPSILON, EVALUATE_MODEL, DATABASE_FILE,
SEEDED, SEED, IS_GATHERING_DATA, USE_DISPLAY_DATA, USE_GRAPHICS, VIEW_DATA_WINDOW)

if USE_GRAPHICS:
  import pygame

from dash_app import episode_fitness, episode_steps, episode_epsilon, episode_snake_length, load_data

class Game:
  def __init__(self, use_graphics=True, strategy="dqn", frames=10, load_model_path=None, initial_epsilon=1.0):
    self.strategy = strategy.lower()
    self.frames = frames
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", self.device)

    # Warning, Message, None
    self.debug_level = "Warning"
    self.decay_interval = 10

    self.episode_count = 0
    self.max_steps = 0
    self.most_apples = 0
    self.total_apples = 0
    self.current_apples = 0
    self.total_reward = 0
    self.highest_fitness = 0
    self.use_graphics = use_graphics

    self.board_width = 2
    self.board_height = 2
    self.margin = 20
    self.cell_size = CELL_SIZE

    self.window_width = len(BOARD_RANGE) * self.cell_size
    self.window_height = len(BOARD_RANGE) * self.cell_size

    self.extra_size = self.margin

    # Initialize pygame
    if self.use_graphics:
      pygame.init()
      pygame.font.init()
      self.randomize_board_size()
      pygame.display.set_caption('AI Snake Game')
      self.clock = pygame.time.Clock()
      self.font = pygame.font.Font(None, 36)

      # Colors
      self.BLACK = (0, 0, 0)
      self.DARK_GREY = (169, 169, 169)
      self.LIGHT_GREY = (211, 211, 211)
      self.APPLE = (200, 0, 0)
      self.BODY = (0, 150, 0)
      self.HEAD = (0, 175, 0)

    # Game variables
    if self.board_width - 2 == 0 and self.board_height - 2 == 0:
      self.snake_pos = [(1, 1)]
    elif self.board_width - 2 == 0 and self.board_height - 2 != 0:
      self.snake_pos = [(1, random.randint(1, self.board_height-2))]

    elif self.board_height - 2 == 0 and self.board_width - 2 != 0:
      self.snake_pos = [(random.randint(1, self.board_width-2), 1)]
    else:
      self.snake_pos = [(random.randint(1, self.board_width-2), random.randint(1, self.board_height-2))]

    self.snake_dir = random.choice([0,1,2,3])  # Start moving to the right
    self.apple_pos = (0, 0)
    
    # Learning parameters
    input_dim = 1 + 1 + 1 # head, body, apple
    extra_dim = 16
    self.dqn = DQN_CNN(input_dim, extra_dim, output_dim=3).to(self.device)
    self.target_dqn = DQN_CNN(input_dim, extra_dim, output_dim=3).to(self.device)

    self.is_training = IS_TRAINING
    self.use_epsilon = USE_EPSILON

    # Load the pre-trained model if specified
    if load_model_path:
      self.load_model(load_model_path)

    self.target_dqn.load_state_dict(self.dqn.state_dict())
    self.target_dqn.eval()
    
    self.optimizer = optim.Adam(self.dqn.parameters(), lr=LR)
    self.criterion = nn.MSELoss()
    self.memory = deque(maxlen=TRAINING_FRAMES)
    self.gamma = 0.95  # Discount factors
    if USE_EPSILON and LOAD_MODEL_PATH == None:
      self.epsilon = initial_epsilon
    elif USE_EPSILON and LOAD_MODEL_PATH != None:
      self.epsilon = self.min_epsilon
    elif USE_EPSILON == False:
      self.epsilon = 0
    self.epsilon_decay = 0.99
    self.min_epsilon = 0.05
    self.steps = 0
    self.training_step_count = 0       # Count the number of training steps
    self.target_update_interval = 20  # Update target DQN every 100 steps
  
  def get_episode_count(self):
    return self.episode_count

  def board_filled(self):
    """ Return True if the snake occupies every cell on the board. """
    return len(self.snake_pos) == self.board_width * self.board_height
  
  def randomize_board_size(self):
    self.board_width = random.choice(BOARD_RANGE) # 5 - 6
    self.board_height = self.board_width
    self.window_width = list(BOARD_RANGE)[-1] * self.cell_size
    self.window_height = list(BOARD_RANGE)[-1] * self.cell_size

    if self.use_graphics:
      self.window = pygame.display.set_mode((self.window_width + self.extra_size, self.window_height + self.extra_size))

  def load_model(self, model_path):
    try:
        self.dqn.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.dqn.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

  def place_new_apple(self):
    """ Place the new apple in random positions not occupied by the snake. """
    free_cells = []
    for x in range(self.board_width):
      for y in range(self.board_height):
        if (x, y) not in self.snake_pos:
          free_cells.append((x, y))

    if len(free_cells) == 0:
      return
    
    self.apple_pos = random.choice(free_cells)

  def draw_cube(self, x, y, value):
    rect = pygame.Rect(x * self.cell_size + self.margin/2, y * self.cell_size + self.margin/2, self.cell_size, self.cell_size)
    if value == 1:  # Apple
        pygame.draw.rect(self.window, self.APPLE, rect.inflate(-self.cell_size // 4, -self.cell_size // 4), border_radius=4)
    elif value == 2:  # Snake body
        pygame.draw.rect(self.window, self.BODY, rect, border_radius=4)
    elif value == 3:  # Snake head
        pygame.draw.rect(self.window, self.HEAD, rect, border_radius=8)

  def draw_snake(self):
    for i, (x, y) in enumerate(self.snake_pos):
      if i == 0:
        self.draw_cube(x, y, 3)  # Head
      else:
        self.draw_cube(x, y, 2)  # Body

  def handle_collision(self, collision_type):
    reward = 0
    head_x, head_y = self.snake_pos[0]
    dx, dy = DIRS_OFFSET[self.snake_dir]
    new_head = (head_x + dx, head_y + dy)
    apple_x, apple_y = self.apple_pos
    max_possible_distance = self.board_width + self.board_height - 2
    new_distance = abs(new_head[0] - apple_x) + abs(new_head[1] - apple_y)
    previous_distance = abs(head_x - apple_x) + abs(head_y - apple_y)
    normalized_distance = new_distance / max_possible_distance

    # # Apply a sigmoid so that 0 distance => near 1.0, large distance => near 0.0
    k = 5.0  # steepness factor
    sigmoid_value = 1.0 / (1.0 + math.exp(k * (normalized_distance - 0.5)))

    # # Compare new_distance vs old_distance in a sigmoid sense
    old_sigmoid = 1.0 / (1.0 + math.exp(k * ((previous_distance / max_possible_distance) - 0.5)))
    new_sigmoid = sigmoid_value

    if new_distance < previous_distance:
      reward += 0.2 * (old_sigmoid - new_sigmoid)
    else:
      reward -= 0.1

    if collision_type == None:
      return reward, False
    elif collision_type == "apple":
      self.place_new_apple()
      self.current_apples += 1
      reward += 5
      return reward, False
    elif collision_type == "wall":
      reward -= 5
      return reward, True
    elif collision_type == "body":
      reward -= 5
      return reward, True
    elif collision_type == "steps":
      reward -= 5
      return reward, True


  def get_object_distances(self, x, y, direction):
    left_direction = (direction - 1) % 4
    right_direction = (direction + 1) % 4

    def is_obstacle(dx, dy):
      new_pos = (x+dx, y+dy)
      return (
        new_pos[0] < 0 or new_pos[0] >= self.board_width or  # Wall Collision
        new_pos[1] < 0 or new_pos[1] >= self.board_height or # Wall Collision
        new_pos in self.snake_pos                       # Body Collision
      )
    
    object_left = is_obstacle(*DIRS_OFFSET[left_direction])
    object_straight = is_obstacle(*DIRS_OFFSET[direction])
    object_right = is_obstacle(*DIRS_OFFSET[right_direction])

    return object_left, object_straight, object_right

  def is_collision(self, point):
    x, y = point
    if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
      return True
    if point in self.snake_pos:
      return True
    return False

  def calculate_max_steps(self):
    total_cells = self.board_width * self.board_height
    apples_to_fill = total_cells - 1  # Number of apples needed to fill the board
    max_steps_total = apples_to_fill * (apples_to_fill + 1) // 2
    return max_steps_total

  def max_steps_next_apple(self):
    total_cells = self.board_width * self.board_height
    free_cells = total_cells - len(self.snake_pos)
    return free_cells

  def state_to_tensor(self):
    image = np.zeros((3, self.board_height, self.board_width), dtype=np.float32)

    head_x, head_y = self.snake_pos[0]
    image[0, head_y, head_x] = 1

    for pos in self.snake_pos[1:]:
      x, y = pos
      image[1, y, x] = 1
    
    apple_x, apple_y = self.apple_pos
    image[2, apple_y, apple_x] = 1

    extra = np.zeros(16, dtype=np.float32)

    # The first 4 bits are the snakes direction
    direction = np.zeros(4, dtype=np.float32)
    direction[self.snake_dir] = 1.0
    extra[:4] = direction

    # What direction is the apple
    extra[4] = 1 if apple_x < head_x else 0
    extra[5] = 1 if apple_x > head_x else 0
    extra[6] = 1 if apple_y < head_y else 0
    extra[7] = 1 if apple_y > head_y else 0

    # Are there objects around the head of the snake
    if self.snake_dir == 0:  # Left
      left = (head_x, head_y + 1)
      straight = (head_x - 1, head_y)
      right = (head_x, head_y - 1)
    elif self.snake_dir == 1:  # Up
      left = (head_x - 1, head_y)
      straight = (head_x, head_y - 1)
      right = (head_x + 1, head_y)
    elif self.snake_dir == 2:  # Right
      left = (head_x, head_y - 1)
      straight = (head_x + 1, head_y)
      right = (head_x, head_y + 1)
    elif self.snake_dir == 3:  # Down
      left = (head_x + 1, head_y)
      straight = (head_x, head_y + 1)
      right = (head_x - 1, head_y)

    object_left = 1 if self.is_collision(left) else 0
    object_straight = 1 if self.is_collision(straight) else 0
    object_right = 1 if self.is_collision(right) else 0

    extra[8] = object_left
    extra[9] = object_straight
    extra[10] = object_right
    
    # Normalize the steps
    extra[11] = self.steps / self.calculate_max_steps()

    # Normalized wall distances
    abs_left = (head_x) / self.board_width
    abs_up = (head_y) / self.board_height
    abs_right = (self.board_width - head_x) / self.board_width
    abs_down = (self.board_height - head_y) / self.board_height

    if self.snake_dir == 0:  # Facing left
      front_wall_distance = abs_left
      left_wall_distance = abs_up
      right_wall_distance = abs_down
    elif self.snake_dir == 1:  # Facing up
      front_wall_distance = abs_up
      left_wall_distance = abs_left
      right_wall_distance = abs_right
    elif self.snake_dir == 2:  # Facing right
      front_wall_distance = abs_right
      left_wall_distance = abs_up
      right_wall_distance = abs_down
    elif self.snake_dir == 3:  # Facing down
      front_wall_distance = abs_down
      left_wall_distance = abs_right
      right_wall_distance = abs_left
    
    extra[12], extra[13], extra[14] = front_wall_distance, left_wall_distance, right_wall_distance

    extra[15] = len(self.snake_pos) / (self.board_width * self.board_height)

    image_tensor = torch.tensor(image).unsqueeze(0).to(self.device)
    extra_tensor = torch.tensor(extra).unsqueeze(0).to(self.device)

    return image_tensor, extra_tensor

  def choose_action(self, state):
    state_image, state_extra = state
    """ Epsilon-greedy action selection. """
    if random.random() < self.epsilon:
      action = random.choice([0, 1, 2]) # 0 = Left, 1 = Straight, 2 = Right
    else:
      with torch.no_grad():
        q_values = self.dqn(state_image, state_extra)
        action = torch.argmax(q_values).item()  # Exploit
    return action

  def remember(self, state, action, reward, next_state, done):
    if len(self.memory) >= TRAINING_FRAMES:
      self.memory.popleft()
    self.memory.append((state, action, reward, next_state, done))

  def update_target_network(self):
    self.target_dqn.load_state_dict(self.dqn.state_dict())

  def train_long_memory(self, batch_size=BATCH_SIZE):
    """ Sample a batch of experiences and train the DQN. """
    if len(self.memory) < batch_size:
      return

    minibatch = random.sample(self.memory, batch_size)

    image_list = [s[0] for s, a, r, ns, d in minibatch]
    next_image_list = [ns[0] for s, a, r, ns, d in minibatch]


    images = torch.cat(image_list)
    extras = torch.cat([s[1] for s, a, r, ns, d in minibatch])
    next_images = torch.cat(next_image_list)
    next_extras = torch.cat([ns[1] for s, a, r, ns, d in minibatch])

    actions = torch.tensor([a for s, a, r, ns, d in minibatch], dtype=torch.long).unsqueeze(1).to(self.device)
    rewards = torch.tensor([r for s, a, r, ns, d in minibatch], dtype=torch.float32).unsqueeze(1).to(self.device)
    dones = torch.tensor([d for s, a, r, ns, d in minibatch], dtype=torch.bool).unsqueeze(1).to(self.device)

    with torch.no_grad():
      next_actions = self.dqn(next_images, next_extras).argmax(1).unsqueeze(1)
      next_q_values = self.target_dqn(next_images, next_extras).gather(1, next_actions)

    target_q_values = rewards + self.gamma * next_q_values * (~dones)

    q_values = self.dqn(images, extras).gather(1, actions)

    loss = self.criterion(q_values, target_q_values)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Increment training step count
    self.training_step_count += 1

    # Update target network every 100 steps
    if self.training_step_count % self.target_update_interval == 0:
      self.update_target_network()
      
  def train_short_memory(self, state, action, reward, next_state, done):
    """
    Train the network on a single transition.
    """
    state_image, state_extra = state
    next_state_image, next_state_extra = next_state

    # Create tensors for the single transition
    action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
    reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
    done_tensor = torch.tensor([[done]], dtype=torch.bool).to(self.device)

    # Compute the target Q-value using the target network
    with torch.no_grad():
      next_action = self.dqn(next_state_image, next_state_extra).argmax(1).unsqueeze(1)
      next_q_value = self.target_dqn(next_state_image, next_state_extra).gather(1, next_action)
    target_q_value = reward_tensor + self.gamma * next_q_value * (~done_tensor)

    # Compute current Q-value from the main network
    current_q_value = self.dqn(state_image, state_extra).gather(1, action_tensor)

    # Calculate loss and update the network
    loss = self.criterion(current_q_value, target_q_value)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def manhattan_distance(self, pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
  
  def reconstruct_path(self, came_from, current):
    while current in came_from:
      prev = came_from[current]
      if prev == self.snake_pos[0]:
        return self.get_direction(self.snake_pos[0], current)
      current = prev
    return self.snake_dir

  def get_direction(self, from_pos, to_pos):
    dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    for direction, (dir_x, dir_y) in DIRS_OFFSET.items():
      if (dx, dy) == (dir_x, dir_y):
        return direction
    return self.snake_dir
  
  def is_safe(self, x, y):
      return (0 <= x < self.board_width and 0 <= y < self.board_height and (x, y) not in self.snake_pos)
  
  def ai_decide_direction_astar(self):
      """ A* Algorithm to decide the snake's next direction. """
      start = self.snake_pos[0]
      goal = self.apple_pos

      open_list = []
      heapq.heappush(open_list, (0, start))
      g_score = {start: 0}
      came_from = {}

      while open_list:
          _, current = heapq.heappop(open_list)

          if current == goal:
              return self.reconstruct_path(came_from, current)

          for direction, (dx, dy) in DIRS_OFFSET.items():
              neighbor = (current[0] + dx, current[1] + dy)
              if not self.is_safe(neighbor[0], neighbor[1]):
                  continue

              tentative_g_score = g_score[current] + 1
              if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                  came_from[neighbor] = current
                  g_score[neighbor] = tentative_g_score
                  f_score = tentative_g_score + self.manhattan_distance(neighbor, goal)
                  heapq.heappush(open_list, (f_score, neighbor))
      
      # No path was found using A*. Choose a safe direction from the current head position.
      safe_directions = []
      for direction, (dx, dy) in DIRS_OFFSET.items():
          new_position = (start[0] + dx, start[1] + dy)
          if self.is_safe(new_position[0], new_position[1]):
              safe_directions.append(direction)
      
      if safe_directions:
          # Optionally, choose one safe direction at random.
          return random.choice(safe_directions)

      return self.snake_dir  # Default if no path found

  def check_collision(self, direction):
    new_pos = (self.snake_pos[0][0]+DIRS_OFFSET[direction][0], self.snake_pos[0][1]+DIRS_OFFSET[direction][1])
    if new_pos[0] < 0 or new_pos[0] >= self.board_width or new_pos[1] < 0 or new_pos[1] >= self.board_height:# Wall Collision
      return "wall"
    elif new_pos in self.snake_pos: # Body Collision
      return "body"
    elif self.steps > self.current_apples * self.board_width * self.board_height + self.board_width * self.board_height:# Too many steps
      return "steps"
    elif new_pos == self.apple_pos: # Apple Collision
      return "apple"
    else:
      return None
  
  def make_move(self, move, collision):
    self.steps += 1
    self.snake_dir = move
    head_x, head_y = self.snake_pos[0]
    dx, dy = DIRS_OFFSET[self.snake_dir]
    self.snake_pos.insert(0, (head_x + dx, head_y + dy))

    if collision != "apple":
      self.snake_pos.pop()

  def gen_dir(self):
    if self.strategy == "astar":
      new_dir = self.ai_decide_direction_astar()
      return new_dir, None
    elif self.strategy == "dqn":
      self.current_state = self.state_to_tensor()
      relative_action = self.choose_action(self.current_state)

      clock_wise = [1, 2, 3, 0]  # Corresponds to [Up, Right, Down, Left]
      idx = clock_wise.index(self.snake_dir)
      if relative_action == 0:      # Turn left
        new_dir = clock_wise[(idx - 1) % 4]
      elif relative_action == 1:    # Go straight (no change)
        new_dir = clock_wise[idx]
      elif relative_action == 2:    # Turn right
        new_dir = clock_wise[(idx + 1) % 4]
      return new_dir, relative_action
    elif self.strategy == "random":
      new_dir = random.choice(list(DIRS_OFFSET.keys()))
      return new_dir, None

  def reset_game(self):
    self.episode_count += 1
    if SEEDED:
      random.seed(SEED + self.episode_count)
    if self.debug_level == "Message":
      print(f"Resetting the game for a new episode. Total Steps: {self.max_steps}")
      print(f"Episode {self.episode_count} - Epsilon: {self.epsilon:.3f}")

    if self.board_filled():
      if self.strategy == "dqn" and EVALUATE_MODEL is False:
        save_path = os.path.join(current_dir, folder_path, f"snake_dqn_episode{self.episode_count} BC_{self.board_width}x{self.board_height}.pth")
        torch.save(self.dqn.state_dict(), save_path)
        print(f"Model Saved - snake_dqn_episode{self.episode_count} BC_{self.board_width}x{self.board_height}.pth")

    if self.use_epsilon:
      if self.episode_count % 100 == 0 and np.mean(episode_fitness) < EPSILON_THRESHOLD:# and (recent_avg - previous_avg) < EPSILON_THRESHOLD:
        self.epsilon = min(1.0, self.epsilon + 0.01)
      else:
        self.epsilon = max(self.min_epsilon, self.epsilon - 0.001)
    self.snake_pos = [(self.board_width // 2, self.board_height // 2)]
    self.snake_dir = random.choice([0,1,2,3])
    self.place_new_apple()
    self.steps = 0
    self.current_apples = 0
    self.total_reward = 0

    self.training_step_count = 0 # Count the number of training steps

  def draw_board(self):
    if not self.use_graphics:
      return
    self.window.fill(self.BLACK)
    for y in range(self.board_height):
      for x in range(self.board_width):
        color = self.DARK_GREY if (x + y) % 2 == 0 else self.LIGHT_GREY
        rect = pygame.Rect(x * self.cell_size + self.margin/2, y * self.cell_size + self.margin/2, self.cell_size, self.cell_size)
        pygame.draw.rect(self.window, color, rect)
    
    # Draw the apple
    self.draw_cube(self.apple_pos[0], self.apple_pos[1], 1)

  def run(self):
    self.reset_game()
    running = True

    while running:

      if self.use_graphics and not IS_GATHERING_DATA:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            running = False  # Exit the game when window is closed

          # Toggle framerate
          if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
              self.frames = min(self.frames + 10, 120)
            elif event.key == pygame.K_DOWN:
              self.frames = max(0, self.frames - 10)
    
      if IS_GATHERING_DATA:
        strats = ["astar","dqn","random"] # 
        board_sizes = range(5, 21)
        episodes_per_size = 1000
        episodes = {}

        for strat in strats:
          self.strategy = strat

          episodes[strat] = {}
          for board_size in board_sizes:
            print(f"Running {strat} on {board_size}x{board_size}")
            episodes[strat][board_size] = []
            self.board_width = board_size
            self.board_height = board_size
            self.reset_game()
            current_episode = 0
            start_time = time.perf_counter()

            while current_episode < episodes_per_size:
              start_time = time.perf_counter()
              absolute_action, relative_action = self.gen_dir()
              collision = self.check_collision(absolute_action)
              step_reward, is_terminal = self.handle_collision(collision)

              if is_terminal:
                end_time = time.perf_counter()
                episode_duration = end_time - start_time

                episode = {"episode": current_episode, "length": len(self.snake_pos), "steps": self.steps, "time": episode_duration}
                episodes[strat][board_size].append(episode)

                current_episode += 1
                self.reset_game()
              else:
                self.make_move(absolute_action, collision)

        # Terminate Simulation
        running = False

        # Save Data
        data2 = json.dumps(episodes, indent=2)
        save_data(data2, "data.txt")

      else:
        
        absolute_action, relative_action = self.gen_dir()

        collision = self.check_collision(absolute_action)

        step_reward, is_terminal = self.handle_collision(collision)

        # Add reward to total
        self.total_reward += step_reward

        if is_terminal:
          global episode_fitness, episode_steps, episode_epsilon, episode_snake_length

          fitness = self.total_reward
          last_action = relative_action

          # Update data
          episode_steps.append(self.steps / self.calculate_max_steps())
          episode_snake_length.append(len(self.snake_pos) / (self.board_width * self.board_height))

          if self.strategy == "dqn":              
            episode_fitness.append(fitness / (self.board_width * self.board_height))
            episode_epsilon.append(self.epsilon)

          if len(episode_steps) > VIEW_DATA_WINDOW:
            episode_steps.pop(0)
          if len(episode_snake_length) > VIEW_DATA_WINDOW:
            episode_snake_length.pop(0)
          if len(episode_fitness) > VIEW_DATA_WINDOW:
            episode_fitness.pop(0)
          if len(episode_epsilon) > VIEW_DATA_WINDOW:
            episode_epsilon.pop(0)

          # Do machine learning stuff here
          if self.strategy == "dqn":    
            if len(episode_fitness) > 0:
              save_data(f"{self.episode_count} {episode_snake_length[-1]} {episode_steps[-1]} {episode_fitness[-1]} {episode_epsilon[-1]}\n",
                        DATABASE_FILE)
        
            # Set the highest score is it isn't set
            if self.highest_fitness is None:
              self.highest_fitness = fitness / (self.board_width * self.board_height)

            # Update the highest score if the current score is higher
            if fitness / (self.board_width * self.board_height) > self.highest_fitness:
              self.highest_fitness = fitness / (self.board_width * self.board_height)

              if self.is_training and EVALUATE_MODEL is False:
                save_path = os.path.join(current_dir, folder_path, f"snake_dqn_episode{self.episode_count} fitness {episode_fitness[-1]:.3f} board {self.board_width}x{self.board_height}.pth")

                torch.save(self.dqn.state_dict(), save_path)
                print(f"New highest fitness! {episode_fitness[-1]:.3f} Model saved at episode {self.episode_count}")

            if self.is_training and EVALUATE_MODEL is False:
              next_state = self.state_to_tensor()
              self.remember(self.state_to_tensor(), last_action, fitness, next_state, done=True)

              self.train_short_memory(self.current_state, last_action, fitness, next_state, done=True)

              # Learn from the episode
              self.replay()

              # Save the model periodically
              if self.episode_count % SAVE_INTERVAL == 0:
                save_path = os.path.join(current_dir, folder_path, f"snake_dqn_episode{self.episode_count} fitness {episode_fitness[-1]:.3f} board {self.board_width}x{self.board_height}.pth")

                torch.save(self.dqn.state_dict(), save_path)
                print(f"Model saved at episode {self.episode_count}")


          self.reset_game()
        else:
          self.make_move(absolute_action, collision)

      if self.use_graphics:
        self.draw_board()
        self.draw_snake()

        pygame.display.flip()

        self.clock.tick(self.frames)

    # Load JSON data from file
    if IS_GATHERING_DATA:
      data = load_data("data.txt")

      # Convert nested dictionary data into a DataFrame.
      rows = []
      for strat, boards in data.items():
          for board_size, episodes_list in boards.items():
              for ep in episodes_list:
                  rows.append({
                      "strategy": strat,
                      "board_size": int(board_size),
                      "episode": ep["episode"],
                      "length": ep["length"],
                      "steps": ep["steps"],
                      "time": ep["time"]
                  })
      df = pd.DataFrame(rows)

      # Group data by strategy and board_size, computing the mean of each metric.
      grouped = df.groupby(["strategy", "board_size"]).agg({
          "length": "mean",  # Effectiveness
          "steps": "mean",   # Efficiency (lower steps might indicate less wasted moves)
          "time": "mean"     # Total time per episode
      }).reset_index()

      # Calculate efficiency metrics.
      # For example, we can define "time per step" (lower is better) as a measure of efficiency.
      grouped["time_per_step"] = grouped["time"] / grouped["steps"]

      # Alternatively, you can compute "steps per time" (higher is better).
      grouped["steps_per_time"] = grouped["steps"] / grouped["time"]

      grouped["length_per_step"] = grouped["length"] / grouped["steps"]


      # Now we plot the three categories on subplots. Here we'll use a 2x3 grid:
      # Row 1: Raw performance metrics (Effectiveness & Efficiency)
      # Row 2: Derived efficiency measures, which help illustrate adaptability across board sizes.
      fig, axes = plt.subplots(2, 3, figsize=(18, 10))

      # --- Effectiveness: Average Snake Length vs Board Size ---
      for strat in grouped["strategy"].unique():
          subset = grouped[grouped["strategy"] == strat]
          axes[0][0].plot(subset["board_size"], subset["length"], marker='o', label=strat)
      axes[0][0].set_title("Average Snake Length vs Board Size")
      axes[0][0].set_xlabel("Board Size")
      axes[0][0].set_ylabel("Average Snake Length")
      axes[0][0].legend()
      axes[0][0].grid(True)

      # --- Raw Efficiency: Average Steps vs Board Size ---
      for strat in grouped["strategy"].unique():
          subset = grouped[grouped["strategy"] == strat]
          axes[0][1].plot(subset["board_size"], subset["steps"], marker='o', label=strat)
      axes[0][1].set_title("Average Steps vs Board Size")
      axes[0][1].set_xlabel("Board Size")
      axes[0][1].set_ylabel("Average Steps")
      axes[0][1].legend()
      axes[0][1].grid(True)

      # --- Raw Efficiency: Average Time vs Board Size ---
      for strat in grouped["strategy"].unique():
          subset = grouped[grouped["strategy"] == strat]
          axes[0][2].plot(subset["board_size"], subset["time"], marker='o', label=strat)
      axes[0][2].set_title("Average Time vs Board Size")
      axes[0][2].set_xlabel("Board Size")
      axes[0][2].set_ylabel("Average Time (sec)")
      axes[0][2].legend()
      axes[0][2].grid(True)

      # --- Derived Efficiency: Time per Step vs Board Size ---
      for strat in grouped["strategy"].unique():
          subset = grouped[grouped["strategy"] == strat]
          axes[1][2].plot(subset["board_size"], subset["time_per_step"], marker='o', label=strat)
      axes[1][2].set_title("Time per Step vs Board Size")
      axes[1][2].set_xlabel("Board Size")
      axes[1][2].set_ylabel("Time per Step (sec)")
      axes[1][2].legend()
      axes[1][2].grid(True)

      # --- Derived Efficiency: Steps per Time vs Board Size ---
      for strat in grouped["strategy"].unique():
          subset = grouped[grouped["strategy"] == strat]
          axes[1][1].plot(subset["board_size"], subset["steps_per_time"], marker='o', label=strat)
      axes[1][1].set_title("Steps per Time vs Board Size")
      axes[1][1].set_xlabel("Board Size")
      axes[1][1].set_ylabel("Steps per Time")
      axes[1][1].legend()
      axes[1][1].grid(True)

      # Plot Snake Length per Step vs Board Size (Composite Performance Metric)
      for strat in grouped["strategy"].unique():
          subset = grouped[grouped["strategy"] == strat]
          axes[1][0].plot(subset["board_size"], subset["length_per_step"], marker='o', label=strat)
      axes[1][0].set_title("Snake Length per Step vs Board Size")
      axes[1][0].set_xlabel("Board Size")
      axes[1][0].set_ylabel("Length/Step")
      axes[1][0].legend()
      axes[1][0].grid(True)


      plt.tight_layout()
      plt.show()
        
    if self.use_graphics:
      pygame.quit()
