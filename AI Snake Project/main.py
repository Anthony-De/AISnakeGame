import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

from threading import Thread
from dash_app import run_dash, save_data
from game import Game
from constants import (STRATEGY, LOAD_MODEL_PATH,
USE_DISPLAY_DATA, USE_GRAPHICS)

if __name__ == "__main__":
  # Start Dash in a separate thread
  if USE_DISPLAY_DATA:
    dash_thread = Thread(target=run_dash)
    dash_thread.daemon = True
    dash_thread.start()

  # Create and run the game
  game = Game(use_graphics=USE_GRAPHICS, strategy=STRATEGY, load_model_path=LOAD_MODEL_PATH, frames=0)
  game.run()
