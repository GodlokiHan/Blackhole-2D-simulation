import pygame
import numpy as np
import math
import random
from typing import List, Tuple

# Constants
C = 299792458.0  # speed of light m/s
G = 6.67430e-11  # gravitational constant

class BlackHole:
    def __init__(self, position: Tuple[float, float], mass: float):
        self.position = np.array(position, dtype=np.float64)
        self.mass = mass
        self.r_s = 2.0 * G * mass / (C * C)  # Schwarzschild radius
        
    def draw(self, screen, camera):
        # Convert world coordinates to screen coordinates
        screen_pos = camera.world_to_screen(self.position)
        radius = max(2, int(self.r_s * camera.zoom))
        
        # Draw event horizon
        pygame.draw.circle(screen, (255, 0, 0), screen_pos, radius)
        
        # Draw photon sphere (r = 1.5 * r_s)
        photon_radius = max(1, int(1.5 * self.r_s * camera.zoom))
        pygame.draw.circle(screen, (255, 100, 100), screen_pos, photon_radius, 1)

class Camera:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1e-11  # meters per pixel
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        x = int((world_pos[0] - self.offset_x) * self.zoom + self.screen_width / 2)
        y = int((world_pos[1] - self.offset_y) * self.zoom + self.screen_height / 2)
        return (x, y)
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> np.ndarray:
        x = (screen_pos[0] - self.screen_width / 2) / self.zoom + self.offset_x
        y = (screen_pos[1] - self.screen_height / 2) / self.zoom + self.offset_y
        return np.array([x, y], dtype=np.float64)

class Ray:
    def __init__(self, position: np.ndarray, direction: np.ndarray):
        self.x, self.y = position
        self.r = np.sqrt(self.x * self.x + self.y * self.y)
        self.phi = np.arctan2(self.y, self.x)
        
        # Convert cartesian velocity to polar
        v_r = direction[0] * np.cos(self.phi) + direction[1] * np.sin(self.phi)
        v_phi = (-direction[0] * np.sin(self.phi) + direction[1] * np.cos(self.phi)) / self.r
        
        self.dr = v_r
        self.dphi = v_phi
        
        # Conserved quantities
        self.L = self.r * self.r * self.dphi
        f = 1.0 - black_hole.r_s / self.r
        dt_dλ = np.sqrt((self.dr * self.dr) / (f * f) + (self.r * self.r * self.dphi * self.dphi) / f)
        self.E = f * dt_dλ
        
        self.trail: List[Tuple[float, float]] = [(float(self.x), float(self.y))]
        self.active = True
        self.color = (
            random.randint(150, 255),
            random.randint(150, 255), 
            random.randint(150, 255)
        )
    
    def geodesic_rhs(self, rs: float) -> Tuple[float, float, float, float]:
        """Right-hand side of geodesic equations"""
        f = 1.0 - rs / self.r
        
        # dr/dλ = dr
        rhs_r = self.dr
        # dφ/dλ = dphi
        rhs_phi = self.dphi
        
        # d²r/dλ² from Schwarzschild metric
        dt_dλ = self.E / f
        rhs_dr = (-(rs / (2 * self.r * self.r)) * f * (dt_dλ * dt_dλ) +
                  (rs / (2 * self.r * self.r * f)) * (self.dr * self.dr) +
                  (self.r - rs) * (self.dphi * self.dphi))
        
        # d²φ/dλ²
        rhs_dphi = -2.0 * self.dr * self.dphi / self.r
        
        return rhs_r, rhs_phi, rhs_dr, rhs_dphi
    
    def rk4_step(self, dλ: float, rs: float):
        """RK4 integration step"""
        if not self.active or self.r <= rs:
            self.active = False
            return
        
        # Store current state
        y0 = [self.r, self.phi, self.dr, self.dphi]
        
        # k1
        k1 = self.geodesic_rhs(rs)
        
        # k2
        temp = [y0[i] + k1[i] * dλ / 2.0 for i in range(4)]
        r_temp, phi_temp, dr_temp, dphi_temp = temp
        k2 = self._geodesic_rhs_temp(r_temp, phi_temp, dr_temp, dphi_temp, rs)
        
        # k3
        temp = [y0[i] + k2[i] * dλ / 2.0 for i in range(4)]
        r_temp, phi_temp, dr_temp, dphi_temp = temp
        k3 = self._geodesic_rhs_temp(r_temp, phi_temp, dr_temp, dphi_temp, rs)
        
        # k4
        temp = [y0[i] + k3[i] * dλ for i in range(4)]
        r_temp, phi_temp, dr_temp, dphi_temp = temp
        k4 = self._geodesic_rhs_temp(r_temp, phi_temp, dr_temp, dphi_temp, rs)
        
        # Update state
        for i in range(4):
            y0[i] += (dλ / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        
        self.r, self.phi, self.dr, self.dphi = y0
        
        # Convert back to cartesian
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)
        
        # Add to trail
        self.trail.append((float(self.x), float(self.y)))
        
        # Deactivate if too far away
        if self.r > 1e13:
            self.active = False
    
    def _geodesic_rhs_temp(self, r: float, phi: float, dr: float, dphi: float, rs: float):
        """Helper for intermediate RK4 steps"""
        f = 1.0 - rs / r
        dt_dλ = self.E / f
        
        rhs_r = dr
        rhs_phi = dphi
        rhs_dr = (-(rs / (2 * r * r)) * f * (dt_dλ * dt_dλ) +
                  (rs / (2 * r * r * f)) * (dr * dr) +
                  (r - rs) * (dphi * dphi))
        rhs_dphi = -2.0 * dr * dphi / r
        
        return rhs_r, rhs_phi, rhs_dr, rhs_dphi
    
    def draw(self, screen, camera):
        if not self.trail:
            return
        
        # Draw trail
        if len(self.trail) > 1:
            screen_trail = [camera.world_to_screen(np.array(point)) for point in self.trail]
            
            # Draw with fading alpha (simulated with color brightness)
            for i in range(len(screen_trail) - 1):
                alpha_ratio = i / (len(screen_trail) - 1)
                color = (
                    int(self.color[0] * alpha_ratio),
                    int(self.color[1] * alpha_ratio), 
                    int(self.color[2] * alpha_ratio)
                )
                pygame.draw.line(screen, color, screen_trail[i], screen_trail[i + 1], 1)
        
        # Draw current position
        if self.active:
            screen_pos = camera.world_to_screen(np.array([self.x, self.y]))
            pygame.draw.circle(screen, self.color, screen_pos, 2)

class BlackHoleSimulation:
    def __init__(self):
        pygame.init()  # Initialize Pygame first
        pygame.font.init()  # Initialize fonts
        
        self.screen_width = 1200
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Black Hole Ray Tracer - CLICK: Add Rays, Mouse: Navigate, R: Reset")
        
        self.camera = Camera(self.screen_width, self.screen_height)
        self.rays: List[Ray] = []
        self.auto_spawn = False  # Turn off auto-spawn since we're using mouse clicks
        self.last_spawn_time = 0
        self.spawn_interval = 500  # ms
        
        # Create font for HUD
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Create Sagittarius A* black hole
        self.black_hole = BlackHole((0.0, 0.0), 8.54e36)
        
        self.running = True
        self.clock = pygame.time.Clock()
    
    def spawn_ray_at_mouse(self, mouse_pos: Tuple[int, int], direction_type: str = "horizontal"):
        """Spawn a ray at the mouse click position with different initial directions"""
        # Convert mouse screen coordinates to world coordinates
        world_pos = self.camera.screen_to_world(mouse_pos)
        
        # Create position array
        pos = np.array([world_pos[0], world_pos[1]])
        
        # Calculate different initial directions based on position
        if direction_type == "horizontal":
            # Horizontal motion (left to right or right to left)
            if pos[0] > 0:
                direction = np.array([-C, 0.0])  # Moving left if on right side
            else:
                direction = np.array([C, 0.0])   # Moving right if on left side
                
        elif direction_type == "vertical":
            # Vertical motion (up to down or down to up)
            if pos[1] > 0:
                direction = np.array([0.0, -C])  # Moving down if above
            else:
                direction = np.array([0.0, C])   # Moving up if below
                
        elif direction_type == "tangential":
            # Tangential motion (perpendicular to radius)
            # Calculate vector from black hole to click position
            to_black_hole = self.black_hole.position - pos
            # Create perpendicular vector (rotate 90 degrees)
            direction = np.array([-to_black_hole[1], to_black_hole[0]])
            # Normalize and scale by speed of light
            direction = direction / np.linalg.norm(direction) * C
            
        elif direction_type == "radial_outward":
            # Radial motion away from black hole
            to_black_hole = self.black_hole.position - pos
            direction = -to_black_hole  # Point away from black hole
            direction = direction / np.linalg.norm(direction) * C
            
        elif direction_type == "radial_inward":
            # Radial motion toward black hole (original behavior)
            to_black_hole = self.black_hole.position - pos
            direction = to_black_hole  # Point toward black hole
            direction = direction / np.linalg.norm(direction) * C
            
        else:  # random
            # Random direction
            random_angle = random.uniform(0, 2 * math.pi)
            direction = np.array([math.cos(random_angle), math.sin(random_angle)]) * C
        
        # Add a small random perturbation to avoid perfectly symmetric paths
        random_angle = random.uniform(-0.05, 0.05)
        cos_a, sin_a = math.cos(random_angle), math.sin(random_angle)
        direction = np.array([
            direction[0] * cos_a - direction[1] * sin_a,
            direction[0] * sin_a + direction[1] * cos_a
        ])
        
        self.rays.append(Ray(pos, direction))
    
    def spawn_random_ray(self):
        """Spawn a ray from random position pointing toward black hole"""
        angle = random.uniform(0, 2 * math.pi)
        distance = 8.0e11
        pos = np.array([
            math.cos(angle) * distance,
            math.sin(angle) * distance
        ])
        
        # Use tangential motion for random rays to see interesting paths
        to_black_hole = self.black_hole.position - pos
        direction = np.array([-to_black_hole[1], to_black_hole[0]])  # Perpendicular
        direction = direction / np.linalg.norm(direction) * C
        
        self.rays.append(Ray(pos, direction))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse click - spawn photon
                    # Use different direction types based on modifier keys
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        direction_type = "vertical"
                    elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                        direction_type = "tangential"
                    elif keys[pygame.K_LALT] or keys[pygame.K_RALT]:
                        direction_type = "radial_outward"
                    else:
                        direction_type = "horizontal"
                    
                    self.spawn_ray_at_mouse(event.pos, direction_type)
                    
                elif event.button == 2:  # Middle mouse - drag camera
                    self.camera.dragging = True
                    self.camera.last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    self.camera.zoom *= 1.1
                elif event.button == 5:  # Mouse wheel down
                    self.camera.zoom /= 1.1
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self.camera.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.camera.dragging:
                    dx = event.pos[0] - self.camera.last_mouse_pos[0]
                    dy = event.pos[1] - self.camera.last_mouse_pos[1]
                    self.camera.offset_x -= dx / self.camera.zoom
                    self.camera.offset_y -= dy / self.camera.zoom
                    self.camera.last_mouse_pos = event.pos
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Spacebar for random rays with tangential motion
                    self.spawn_random_ray()
                elif event.key == pygame.K_r:
                    self.rays.clear()
                elif event.key == pygame.K_a:
                    self.auto_spawn = not self.auto_spawn
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self):
        current_time = pygame.time.get_ticks()
        
        # Auto-spawn rays (optional, turned off by default)
        if self.auto_spawn and current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_random_ray()
            self.last_spawn_time = current_time
        
        # Update rays
        for ray in self.rays[:]:
            if ray.active:
                ray.rk4_step(1.0, self.black_hole.r_s)
            # Remove inactive rays that are too old (keep last 50 rays for performance)
            elif len(self.rays) > 50 and len(ray.trail) > 1000:
                self.rays.remove(ray)
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Draw coordinate grid
        self.draw_grid()
        
        # Draw black hole
        self.black_hole.draw(self.screen, self.camera)
        
        # Draw rays
        for ray in self.rays:
            ray.draw(self.screen, self.camera)
        
        # Draw HUD
        self.draw_hud()
        
        pygame.display.flip()
    
    def draw_grid(self):
        """Draw a simple coordinate grid"""
        grid_spacing = 1e11  # 100 billion meters
        grid_color = (40, 40, 40)
        
        # Vertical lines
        x = -grid_spacing * 10
        while x <= grid_spacing * 10:
            start = self.camera.world_to_screen(np.array([x, -1e13]))
            end = self.camera.world_to_screen(np.array([x, 1e13]))
            pygame.draw.line(self.screen, grid_color, start, end, 1)
            x += grid_spacing
        
        # Horizontal lines
        y = -grid_spacing * 10
        while y <= grid_spacing * 10:
            start = self.camera.world_to_screen(np.array([-1e13, y]))
            end = self.camera.world_to_screen(np.array([1e13, y]))
            pygame.draw.line(self.screen, grid_color, start, end, 1)
            y += grid_spacing
    
    def draw_hud(self):
        """Draw heads-up display with info"""
        info_lines = [
            f"Rays: {len(self.rays)}",
            f"Auto-spawn: {'ON' if self.auto_spawn else 'OFF'}",
            f"Zoom: {self.camera.zoom:.2e}",
            "Controls: CLICK=Add ray (horizontal)",
            "SHIFT+CLICK=Vertical, CTRL+CLICK=Tangential",
            "ALT+CLICK=Radial outward, SPACE=Random ray",
            "R=Reset, ESC=Quit"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(text, (10, 10 + i * 25))
    
    def run(self):
        try:
            while self.running:
                self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(60)
        finally:
            pygame.quit()

# Global black hole reference for Ray class
black_hole = None

if __name__ == "__main__":
    # Initialize global black hole
    black_hole = BlackHole((0.0, 0.0), 8.54e36)
    
    simulation = BlackHoleSimulation()
    simulation.run()