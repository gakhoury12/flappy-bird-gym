import pygame
import random

class FlappyBird:
    def __init__(self):
        self.screen_width = 600
        self.screen_height = 400
        self.gravity = 0.5
        self.jump_strength = -10
        self.velocity = 0
        self.bird_y = self.screen_height / 2
        self.game_over = False
        self.score = 0
        self.pipe_width = 80
        self.pipes = []
        self.spawn_pipes()

    def spawn_pipes(self):
        gap = random.randint(100, 300)
        top_pipe_height = random.randint(50, 300)
        # Store pipes as tuples: (x, top_height, bottom_height)
        self.pipes.append((self.screen_width, top_pipe_height, self.screen_height - (top_pipe_height + gap)))

    def get_state(self):
        # Return the state representation
        return [self.bird_y] + [pipe[0] for pipe in self.pipes]

    def step(self, action):
        if action == 1:  # Flap
            self.velocity = self.jump_strength
        self.velocity += self.gravity
        self.bird_y += self.velocity

        # Check if bird hits the ground or goes off-screen
        if self.bird_y > self.screen_height or self.bird_y < 0:
            self.game_over = True

        # Move pipes
        for i in range(len(self.pipes)):
            x, top_height, bottom_height = self.pipes[i]
            self.pipes[i] = (x - 5, top_height, bottom_height)  # Update x position

        # Remove off-screen pipes
        if self.pipes and self.pipes[0][0] < -self.pipe_width:
            self.pipes.pop(0)  # Remove the pipe that is off-screen
            self.spawn_pipes()  # Spawn a new pipe
            self.score += 1

        return self.get_state(), self.score, self.game_over

    def reset(self):
        self.bird_y = self.screen_height / 2
        self.velocity = 0
        self.game_over = False
        self.score = 0
        self.pipes.clear()
        self.spawn_pipes()
        return self.get_state()
