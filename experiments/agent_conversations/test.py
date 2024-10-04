import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the game window
window_width = 600
window_height = 400
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Snake Game')

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

# Snake properties
snake_block = 10
snake_speed = 15

# Initial snake position and length
snake_x = window_width / 2
snake_y = window_height / 2
snake_x_change = 0
snake_y_change = 0
snake_list = []
snake_length = 1

# Food properties
food_x = round(random.randrange(0, window_width - snake_block) / 10.0) * 10.0
food_y = round(random.randrange(0, window_height - snake_block) / 10.0) * 10.0

# Game over flag
game_over = False

# Score
score = 0

# Font for displaying the score
font_style = pygame.font.SysFont(None, 35)

# Function to display the score
def display_score(score):
    value = font_style.render("Your Score: " + str(score), True, white)
    window.blit(value, [0, 0])

# Main game loop
while not game_over:

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                snake_x_change = -snake_block
                snake_y_change = 0
            elif event.key == pygame.K_RIGHT:
                snake_x_change = snake_block
                snake_y_change = 0
            elif event.key == pygame.K_UP:
                snake_y_change = -snake_block
                snake_x_change = 0
            elif event.key == pygame.K_DOWN:
                snake_y_change = snake_block
                snake_x_change = 0

    # Update snake position
    snake_x += snake_x_change
    snake_y += snake_y_change

    # Check for game over conditions
    if snake_x >= window_width or snake_x < 0 or snake_y >= window_height or snake_y < 0:
        game_over = True

    # Check if snake collides with itself
    for x in snake_list[:-1]:
        if x == snake_head:
            game_over = True

    # Update snake list
    snake_head = []
    snake_head.append(snake_x)
    snake_head.append(snake_y)
    snake_list.append(snake_head)
    if len(snake_list) > snake_length:
        del snake_list[0]

    # Check if snake eats food
    if snake_x == food_x and snake_y == food_y:
        food_x = round(random.randrange(0, window_width - snake_block) / 10.0) * 10.0
        food_y = round(random.randrange(0, window_height - snake_block) / 10.0) * 10.0
        snake_length += 1
        score += 1

    # Clear the screen
    window.fill(black)

    # Draw the snake
    for x in snake_list:
        pygame.draw.rect(window, green, [x[0], x[1], snake_block, snake_block])

    # Draw the food
    pygame.draw.rect(window, red, [food_x, food_y, snake_block, snake_block])

    # Display the score
    display_score(score)

    # Update the display
    pygame.display.update()

    # Control the game speed
    pygame.time.Clock().tick(snake_speed)

# Game over message
message = font_style.render("You Lost! Press Q-Quit or C-Play Again", True, red)
rect = message.get_rect(center=(window_width/2, window_height/2))
window.blit(message, rect)
pygame.display.update()

# Wait for user input to quit or play again
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                quit()
            if event.key == pygame.K_c:
                # Reset game variables and start a new game
                # ... (Add code to reset game variables here)

# Quit Pygame
pygame.quit()
quit()