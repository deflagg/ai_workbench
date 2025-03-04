import pygame
import numpy as np
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Flight Simulator")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Constants
g = 9.81  # Gravity (m/s^2)
rho = 1.225  # Air density at sea level (kg/m^3)
dt = 0.016  # Time step (s), ~60 FPS

# Aircraft parameters
class Aircraft:
    def __init__(self):
        # Physical properties
        self.mass = 1000  # kg
        self.wing_area = 10  # m^2
        self.chord = 1  # m (mean aerodynamic chord)
        self.I = 1000  # Moment of inertia (kg·m^2), approximate
        # Aerodynamic coefficients
        self.C_L_alpha = 5.0  # Lift slope (per radian)
        self.C_D0 = 0.02  # Parasitic drag coefficient
        self.k = 0.05  # Induced drag factor
        self.C_M0 = 0.0  # Base moment coefficient
        self.C_M_alpha = -0.5  # Pitch stability (negative for stability)
        self.C_M_delta_e = -1.0  # Elevator effectiveness
        # State variables
        self.x = 0.0  # m (horizontal position)
        self.z = 1000.0  # m (altitude, positive up)
        self.v_x = 50.0  # m/s (initial forward speed)
        self.v_z = 0.0  # m/s (vertical speed)
        self.theta = 0.0  # radians (pitch angle)
        self.omega = 0.0  # rad/s (pitch rate)
        # Control inputs
        self.thrust = 5000  # N (initial thrust)
        self.delta_e = 0.0  # radians (elevator deflection)
        # Control limits
        self.thrust_max = 10000  # N
        self.thrust_min = 0  # N
        self.delta_e_max = math.radians(20)  # 20 degrees

    def update(self, dt):
        # Velocity magnitude
        v = math.sqrt(self.v_x**2 + self.v_z**2)
        if v < 1e-6:  # Avoid division by zero
            v = 1e-6
        
        # Flight path angle
        psi = math.atan2(self.v_z, self.v_x)
        # Angle of attack
        alpha = self.theta - psi
        
        # Dynamic pressure
        q = 0.5 * rho * v**2
        
        # Aerodynamic coefficients
        C_L = self.C_L_alpha * alpha
        C_D = self.C_D0 + self.k * C_L**2
        C_M = self.C_M0 + self.C_M_alpha * alpha + self.C_M_delta_e * self.delta_e
        
        # Forces
        L = q * self.wing_area * C_L  # Lift
        D = q * self.wing_area * C_D  # Drag
        # Aerodynamic forces in world coordinates
        F_aero_x = -D * (self.v_x / v) - L * (self.v_z / v)
        F_aero_z = -D * (self.v_z / v) + L * (self.v_x / v)
        # Thrust
        F_thrust_x = self.thrust * math.cos(self.theta)
        F_thrust_z = self.thrust * math.sin(self.theta)
        # Gravity
        F_gravity_z = -self.mass * g
        
        # Total forces
        F_x = F_aero_x + F_thrust_x
        F_z = F_aero_z + F_thrust_z + F_gravity_z
        
        # Accelerations
        a_x = F_x / self.mass
        a_z = F_z / self.mass
        
        # Update velocities
        self.v_x += a_x * dt
        self.v_z += a_z * dt
        
        # Update positions
        self.x += self.v_x * dt
        self.z += self.v_z * dt
        if self.z < 0:  # Simple ground collision
            self.z = 0
            self.v_z = 0
        
        # Pitch dynamics
        M = q * self.wing_area * self.chord * C_M  # Moment
        omega_dot = M / self.I  # Angular acceleration
        self.omega += omega_dot * dt
        self.theta += self.omega * dt

    def get_shape(self, scale, screen_height):
        # Define aircraft shape in local coordinates (meters)
        points = [
            (5, 0),   # Nose
            (-5, 2),  # Top tail
            (-5, -2)  # Bottom tail
        ]
        # Rotate and translate to world, then to screen
        screen_points = []
        for x_loc, z_loc in points:
            # Rotate by theta
            x_rot = x_loc * math.cos(self.theta) - z_loc * math.sin(self.theta)
            z_rot = x_loc * math.sin(self.theta) + z_loc * math.cos(self.theta)
            # Screen coordinates (center aircraft, invert z for screen y)
            x_screen = SCREEN_WIDTH // 2 + x_rot * scale
            y_screen = SCREEN_HEIGHT // 2 - z_rot * scale
            screen_points.append((x_screen, y_screen))
        return screen_points

# Simulation setup
aircraft = Aircraft()
clock = pygame.time.Clock()
scale = 10  # pixels per meter
running = True

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Handle continuous input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        aircraft.thrust = min(aircraft.thrust + 100, aircraft.thrust_max)
    if keys[pygame.K_DOWN]:
        aircraft.thrust = max(aircraft.thrust - 100, aircraft.thrust_min)
    if keys[pygame.K_LEFT]:
        aircraft.delta_e = min(aircraft.delta_e + 0.01, aircraft.delta_e_max)  # Nose down
    if keys[pygame.K_RIGHT]:
        aircraft.delta_e = max(aircraft.delta_e - 0.01, -aircraft.delta_e_max)  # Nose up

    # Update simulation
    aircraft.update(dt)
    
    # Rendering
    screen.fill(BLUE)  # Sky
    pygame.draw.rect(screen, GREEN, (0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2))  # Ground
    aircraft_shape = aircraft.get_shape(scale, SCREEN_HEIGHT)
    pygame.draw.polygon(screen, BLACK, aircraft_shape)
    
    pygame.display.flip()
    clock.tick(60)  # 60 FPS

pygame.quit()