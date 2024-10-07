import numpy as np
import random
import pygame
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Hyperparameters
STATE_SIZE = 4
ACTION_SIZE = 2
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
REPLAY_MEMORY_SIZE = 2000

# Specify sprite file paths
SPRITE_PATHS = {
    'player': '/home/gabe/Desktop/5100/flappy-bird-gym/flappy_bird_gym/assets/sprites/yellowbird-midflap.png',
    'background': '/home/gabe/Desktop/5100/flappy-bird-gym/flappy_bird_gym/assets/sprites/background-day.png'
}

# Define the list of pipe sprites
PIPES_LIST = [
    '/home/gabe/Desktop/5100/flappy-bird-gym/flappy_bird_gym/assets/sprites/pipe-green.png',
    '/home/gabe/Desktop/5100/flappy-bird-gym/flappy_bird_gym/assets/sprites/pipe-red.png'  # Add more pipe paths as needed
]

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 600))

class DQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def load_sprites():
    """Load all sprites based on specified paths."""
    sprites = {}
    for key, path in SPRITE_PATHS.items():
        sprites[key] = pygame.image.load(path).convert_alpha()  # Use convert_alpha for transparency

    # Select random pipe sprites
    pipe_index = random.randint(0, len(PIPES_LIST) - 1)
    sprites['upper_pipe'] = pygame.transform.flip(pygame.image.load(PIPES_LIST[pipe_index]).convert_alpha(), False, True)
    sprites['lower_pipe'] = pygame.image.load(PIPES_LIST[pipe_index]).convert_alpha()
    
    return sprites

def draw_window(sprites, player_x, player_y, upper_pipes, lower_pipes, score):
    """Draw the game window."""
    screen.fill((255, 255, 255))  # Clear screen

    # Draw background
    screen.blit(sprites['background'], (0, 0))

    # Draw player
    screen.blit(sprites['player'], (player_x, player_y))

    # Draw pipes
    for pipe in upper_pipes:
        screen.blit(sprites['upper_pipe'], (pipe['x'], pipe['y']))

    for pipe in lower_pipes:
        screen.blit(sprites['lower_pipe'], (pipe['x'], pipe['y']))

    # Display score
    font = pygame.font.SysFont("comicsans", 50)
    text = font.render("Score: " + str(score), True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.update()

def reset_game(sprites):
    """Reset the game and return the initial state."""
    player_y = 200
    player_x = 50
    upper_pipes = [{'x': 400, 'y': 150}]
    lower_pipes = [{'x': 400, 'y': 300}]
    return player_x, player_y, upper_pipes, lower_pipes

def step(action, player_x, player_y, upper_pipes, lower_pipes, sprites):
    """Take an action and return the next state, reward, and done status."""
    player_flapped = False
    if action == 1:  # Flap
        player_y -= 9  # Adjust flap force

    # Update pipes' position
    for pipe in upper_pipes:
        pipe['x'] -= 4  # Move pipes left
    for pipe in lower_pipes:
        pipe['x'] -= 4  # Move pipes left

    # Check for collisions and scoring logic here (simplified)
    done = False
    reward = 0

    # Check if pipes are off-screen
    if upper_pipes[0]['x'] < -50:
        upper_pipes.pop(0)
        lower_pipes.pop(0)

        # Select random pipe sprites for new pipes
        pipe_index = random.randint(0, len(PIPES_LIST) - 1)
        upper_pipes.append({'x': 400, 'y': random.randint(100, 300)})  # Add new pipes
        lower_pipes.append({'x': 400, 'y': upper_pipes[-1]['y'] + 150})

    # Check for collisions
    if player_y > 600 or player_y < 0:  # Check boundaries
        done = True
        reward = -10

    # Draw the game state
    draw_window(sprites, player_x, player_y, upper_pipes, lower_pipes, 0)  # Replace 0 with actual score if needed

    return np.array([player_y, 0, 0, 0]), reward, done  # Update state representation as needed

def main():
    sprites = load_sprites()
    agent = DQNAgent()
    
    for e in range(EPISODES):
        player_x, player_y, upper_pipes, lower_pipes = reset_game(sprites)
        state = np.reshape([player_y, 0, 0, 0], [1, STATE_SIZE])
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = step(action, player_x, player_y, upper_pipes, lower_pipes, sprites)
            next_state = np.reshape(next_state, [1, STATE_SIZE])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode: {e}/{EPISODES}, score: {time}")
                break

        agent.replay()

if __name__ == "__main__":
    main()
