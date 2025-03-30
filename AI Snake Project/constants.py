import os
import datetime

current_dir = os.getcwd()

# Get the current time (once, at training start)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

custom_folder = "snake_dqn"

folder_name = f"{custom_folder}_{start_time}"

# Full path to the folder in the current directory
folder_path = os.path.join(current_dir, folder_name)

DATABASE_FILE = os.path.join(folder_path, "database.txt")

SEEDED = False
SEED = 123456789

IS_GATHERING_DATA = False
USE_DISPLAY_DATA = True
USE_GRAPHICS = True

STRATEGY = "dqn"
IS_TRAINING = True
USE_EPSILON = True
EVALUATE_MODEL = False

if STRATEGY == "dqn" and EVALUATE_MODEL is True:
  path = os.path.join(current_dir, "snake_dqn_episode13014 BC_6x6.pth")
else:
  path = None
LOAD_MODEL_PATH = path
EPSILON_THRESHOLD = 0.1

VIEW_DATA_WINDOW = 500

if STRATEGY == "dqn" and IS_TRAINING is True and USE_EPSILON is True:
  DASH_PORT = 8050
elif STRATEGY == "dqn" and IS_TRAINING is True and USE_EPSILON is False:
  DASH_PORT = 8051
elif STRATEGY == "random":
  DASH_PORT = 8052
elif STRATEGY == "astar":
  DASH_PORT = 8053
else:
  DASH_PORT = 8054

# Create the folder if it does not exist
if EVALUATE_MODEL == False and STRATEGY == "dqn" and not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created folder: {folder_path}")

  # Create a database.txt file
  open(DATABASE_FILE, "w").close()

# Constants
CELL_SIZE = 30
SAVE_INTERVAL = 500  # Save the model every 500 steps

KILL_STEPS = 1000

# Directions: Left, Up, Right, Down
DIRS_OFFSET = {
    0: (-1, 0),  # Left
    1: (0, -1),  # Up
    2: (1, 0),   # Right
    3: (0, 1)    # Down
}

average_count = 1

training_interval = 5
TRAINING_FRAMES = 100000
BATCH_SIZE = 256
LR = 0.00005

BOARD_RANGE = range(6,7)

highest_fitness = None
index_values = []
