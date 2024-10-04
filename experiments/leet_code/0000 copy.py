import pygame
import random

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
FALL_SPEED = 0.27

# Colors
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0), (128, 0, 128), (0, 255, 255)
]

# Tetromino shapes (rotations included)  Add more if you like!
TETROMINOES = [
    [[[1, 1, 1],
      [0, 1, 0]],

     [[0, 1],
      [1, 1],
      [0, 1]],

     [[0, 1, 0],
      [1, 1, 1]],

     [[1, 0],
      [1, 1],
      [1, 0]]],

    [[[2, 2, 0],
      [0, 2, 2]],

     [[0, 2],
      [2, 2],
      [2, 0]]],

    [[[3, 3, 3, 3]],

     [[3],
      [3],
      [3],
      [3]]],

    [[[4, 4],
      [4, 4]]]
]


class Tetromino:
    def __init__(self, x, y):
        self.shape = random.choice(TETROMINOES)
        self.rotation = 0
        self.x = x
        self.y = y
        self.color = COLORS[random.randint(0, len(COLORS) - 1)]

    def draw(self, surface):
        for i, line in enumerate(self.shape[self.rotation]):
            row = list(line)
            for j, column in enumerate(row):
                if column:
                    pygame.draw.rect(surface, self.color,
                                     (self.x + j * GRID_SIZE, self.y + i * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)


def create_grid(locked_pos={}):
    grid = [[(0, 0, 0) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if (j, i) in locked_pos:
                c = locked_pos[(j, i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation]
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column:
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(GRID_WIDTH) if grid[i][j] == (0, 0, 0)] for i in range(GRID_HEIGHT)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True

    return False


def draw_grid(surface, grid):
    # ... (same as before)


def clear_rows(grid, locked):
    inc = 0
    for i in range(len(grid)-1,-1,-1): # Start from bottom
        row = grid[i]
        if (0,0,0) not in row:  # Check if row is full
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j,i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked), key = lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)

    return inc



def draw_next_shape(shape, surface):
     # ... (Implementation for displaying next shape - Add this as needed)



def draw_window(surface, grid, score=0, last_score = 0):
    surface.fill((0, 0, 0))

    pygame.font.init()
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('Tetris', 1, (255, 255, 255))

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 5))


    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Score: ' + str(score), 1, (255,255,255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100

    surface.blit(label, (sx + 10, sy + 160))


    label = font.render('High Score: ' + last_score, 1, (255,255,255))


    sx = top_left_x - 240
    sy = top_left_y + 200


    surface.blit(label, (sx + 20, sy + 160))


    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            pygame.draw.rect(surface, grid[i][j],
                             (top_left_x + j * GRID_SIZE, top_left_y + i * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)

    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 5)




    draw_grid(surface, grid)
    # pygame.display.update()


def main(win):  # Pass the window object
    locked_positions = {}
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = Tetromino(5, 0)
    next_piece = Tetromino(5, 0)
    clock = pygame.time.Clock()
    fall_time = 0
    score = 0



    while run:
        fall_speed = 0.27

        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()



        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece