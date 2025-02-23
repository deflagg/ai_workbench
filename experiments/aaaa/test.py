import pygame
import random

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH = 800
HEIGHT = 800
SQUARE_SIZE = WIDTH // 11  # 11x11 grid for Monopoly board
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Monopoly")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)

class Property:
    def __init__(self, name, price, rent, position, color):
        self.name = name
        self.price = price
        self.rent = rent
        self.owner = None
        self.position = position
        self.color = color

class Player:
    def __init__(self, name, color):
        self.name = name
        self.money = 1500
        self.position = 0
        self.properties = []
        self.color = color

class Monopoly:
    def __init__(self):
        self.players = [
            Player("Player 1", RED),
            Player("Player 2", BLUE)
        ]
        self.properties = [
            Property("Mediterranean Avenue", 60, 2, 1, BROWN),
            Property("Baltic Avenue", 60, 4, 3, BROWN),
            # Add more properties with their board positions (0-39)
        ]
        self.board_size = 40
        self.current_player = 0
        self.font = pygame.font.Font(None, 24)

    def roll_dice(self):
        return random.randint(1, 6) + random.randint(1, 6)

    def draw_board(self):
        screen.fill(WHITE)
        
        # Draw outer squares
        for i in range(11):
            pygame.draw.rect(screen, BLACK, (i * SQUARE_SIZE, 0, SQUARE_SIZE, SQUARE_SIZE), 2)
            pygame.draw.rect(screen, BLACK, (i * SQUARE_SIZE, HEIGHT - SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
            pygame.draw.rect(screen, BLACK, (0, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
            pygame.draw.rect(screen, BLACK, (WIDTH - SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)

        # Draw properties
        for prop in self.properties:
            pos = self.position_to_coords(prop.position)
            pygame.draw.rect(screen, prop.color, 
                           (pos[0], pos[1], SQUARE_SIZE, SQUARE_SIZE // 4))
            text = self.font.render(prop.name[:5], True, BLACK)
            screen.blit(text, (pos[0] + 5, pos[1] + 5))

        # Draw players
        for player in self.players:
            pos = self.position_to_coords(player.position)
            pygame.draw.circle(screen, player.color, 
                             (pos[0] + SQUARE_SIZE // 2, pos[1] + SQUARE_SIZE // 2), 10)

        # Draw info
        info = [
            f"Turn: {self.players[self.current_player].name}",
            f"Money: ${self.players[self.current_player].money}"
        ]
        for i, line in enumerate(info):
            text = self.font.render(line, True, BLACK)
            screen.blit(text, (WIDTH // 2 - 50, HEIGHT // 2 + i * 30))

    def position_to_coords(self, position):
        if position < 10:  # Bottom row
            return (WIDTH - (position + 1) * SQUARE_SIZE, HEIGHT - SQUARE_SIZE)
        elif position < 20:  # Left column
            return (0, HEIGHT - (position - 9) * SQUARE_SIZE)
        elif position < 30:  # Top row
            return ((position - 20) * SQUARE_SIZE, 0)
        else:  # Right column
            return (WIDTH - SQUARE_SIZE, (position - 30) * SQUARE_SIZE)

    def play_turn(self):
        player = self.players[self.current_player]
        roll = self.roll_dice()
        
        player.position = (player.position + roll) % self.board_size
        if player.position < roll:
            player.money += 200

        self.current_player = (self.current_player + 1) % len(self.players)

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.play_turn()
                    elif event.key == pygame.K_q:
                        running = False

            self.draw_board()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    game = Monopoly()
    game.run()