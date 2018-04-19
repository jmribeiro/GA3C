ATARI_GAME = 'PongDeterministic-v0'

# 0 = No Rendering;
# 1 = Render first agent;
# 2 = Render all agents (slow)
RENDER_MODE = 0

# Enable to see the trained agent in action
PLAY_MODE = True
# Load old models. Throws if the model doesn't exist
LOAD = True
# If 0, the latest checkpoint is loaded
LOAD_EPISODE = 0

ACTOR_LEARNERS = 16  # 32
PREDICTORS = 1      # 2
TRAINERS = 2        # 2

EPISODES = 400000
TIME_MAX = 5

TRAINING_QUEUE_SIZE = 100
TRAINING_MIN_BATCH = 0
PREDICTION_QUEUE_SIZE = 100
PREDICTION_MIN_BATCH = 128

STACKED_FRAMES = 4
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84

LEARNING_RATE = 0.0003
DISCOUNT_FACTOR = 0.99
RMSPROP_DECAY = 0.99
RMSPROP_MOMENTUM = 0.0
RMSPROP_EPSILON = 0.1
LOG_NOISE = 1e-6
ENTROPY_BETA = 0.01

REWARD_MIN = -1
REWARD_MAX = 1

SAVE_INTERVAL = 5*60    # 5 minutes
SAVE_DIR = "weights"