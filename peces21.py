import cv2
import numpy as np
import random
import math
import os
import time

# Video settings
width, height = 1920, 1080
fps = 30
duration = 60 * 60 * 12  # seconds
total_frames = fps * duration

# Prepare video directory
video_dir = 'videos'
os.makedirs(video_dir, exist_ok=True)

# Define video file path with timestamp
epoch_time = int(time.time())
video_path = os.path.join(video_dir, f'fish_simulation_{epoch_time}.mp4')

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Helper functions
def angle_difference(beta, alpha):
    difference = alpha - beta
    while difference > math.pi:
        difference -= 2 * math.pi
    while difference < -math.pi:
        difference += 2 * math.pi
    return difference

def angle_in_radians(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

class Pez:
    def __init__(self):
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.a = random.uniform(0, math.pi * 2)
        self.edad = (random.uniform(2, 4)) / 2
        self.tiempo = random.uniform(0, 1)
        self.avancevida = random.uniform(0.05, 0.1)
        self.sexo = random.randint(0, 1)

        # Color aleatorio independiente del sexo
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        self.energia = random.uniform(0, 1)
        self.direcciongiro = random.choice([-1, 0, 1])
        self.numeroelementos = 10
        self.numeroelementoscola = 5
        self.colorr = [self.color[0] + random.randint(-50, 50) for _ in range(-1, self.numeroelementos)]
        self.colorg = [self.color[1] + random.randint(-50, 50) for _ in range(-1, self.numeroelementos)]
        self.colorb = [self.color[2] + random.randint(-50, 50) for _ in range(-1, self.numeroelementos)]
        self.anguloanterior = 0
        self.giro = 0
        self.max_turn_rate = random.uniform(0.005, 0.01)  # Reduced turn rate for slower, arcing movement
        self.target_angle = self.a  # Desired angle

        # New attributes for speed management
        self.initial_speed = self.avancevida
        self.max_speed = self.initial_speed * 5
        self.speed = self.initial_speed
        self.is_chasing_food = False
        self.is_avoiding_collision = False

    def dibuja(self, frame):
        # Set the main color based on energy level
        color_main = self.color if self.energia > 0 else (128, 128, 128)

        # Mouth with breathing effect
        mouth_radius = max(int(math.sin(self.tiempo * 2) * 2 + 3), 1)
        x_mouth = int(self.x + math.cos(self.a) * 5 * self.edad)
        y_mouth = int(self.y + math.sin(self.a) * 5 * self.edad)
        
        # Oscillation perpendicular to direction
        mouth_oscillation = math.sin(self.tiempo) * 2
        x_mouth += int(math.cos(self.a + math.pi / 2) * mouth_oscillation)
        y_mouth += int(math.sin(self.a + math.pi / 2) * mouth_oscillation)
        
        cv2.circle(frame, (x_mouth, y_mouth), mouth_radius, color_main, -1, cv2.LINE_AA)

        # Eyes
        for i in range(-1, self.numeroelementos):
            if i == 1:
                for sign in [-1, 1]:
                    x_eye = int(self.x + sign * math.cos(self.a + math.pi / 2) * 4 * self.edad - i * math.cos(self.a) * self.edad)
                    y_eye = int(self.y + sign * math.sin(self.a + math.pi / 2) * 4 * self.edad - i * math.sin(self.a) * self.edad)

                    # Oscillation perpendicular to direction
                    eye_oscillation = math.sin((i / 5) - self.tiempo) * 4
                    x_eye += int(math.cos(self.a + math.pi / 2) * eye_oscillation)
                    y_eye += int(math.sin(self.a + math.pi / 2) * eye_oscillation)
                    
                    radius_eye = max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) / 3), 1)
                    cv2.circle(frame, (x_eye, y_eye), radius_eye, (255, 255, 255), -1, cv2.LINE_AA)

        # Fins
        for i in range(-1, self.numeroelementos):
            if i == self.numeroelementos // 2 or i == int(self.numeroelementos / 1.1):
                for sign in [-1, 1]:
                    x_fin = int(self.x + sign * math.cos(self.a + math.pi / 2) * 0.3 * self.edad - i * math.cos(self.a) * self.edad)
                    y_fin = int(self.y + sign * math.sin(self.a + math.pi / 2) * 0.3 * self.edad - i * math.sin(self.a) * self.edad)

                    # Oscillation perpendicular to direction
                    fin_oscillation = math.sin((i / 5) - self.tiempo) * 4
                    x_fin += int(math.cos(self.a + math.pi / 2) * fin_oscillation)
                    y_fin += int(math.sin(self.a + math.pi / 2) * fin_oscillation)
                    
                    axes = (max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) * 2), 1),
                            max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1)), 1))
                    angle = math.degrees(self.a + math.pi / 2 - math.cos(self.tiempo * 2) * sign)
                    cv2.ellipse(frame, (x_fin, y_fin), axes, angle, 0, 360, color_main, -1, cv2.LINE_AA)

        # Body
        for i in range(-1, self.numeroelementos):
            x_body = int(self.x - i * math.cos(self.a) * 2 * self.edad)
            y_body = int(self.y - i * math.sin(self.a) * 2 * self.edad)
            
            # Oscillation perpendicular to direction
            body_oscillation = math.sin((i / 5) - self.tiempo) * 4
            x_body += int(math.cos(self.a + math.pi / 2) * body_oscillation)
            y_body += int(math.sin(self.a + math.pi / 2) * body_oscillation)
            
            radius_body = max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) / 1), 1)
            color_body = (self.colorr[i], self.colorg[i], self.colorb[i])
            cv2.circle(frame, (x_body, y_body), radius_body, color_body, -1, cv2.LINE_AA)

        # Tail
        for i in range(self.numeroelementos, self.numeroelementos + self.numeroelementoscola):
            x_tail = int(self.x - (i - 3) * math.cos(self.a) * 2 * self.edad)
            y_tail = int(self.y - (i - 3) * math.sin(self.a) * 2 * self.edad)
            
            # Oscillation perpendicular to direction
            tail_oscillation = math.sin((i / 5) - self.tiempo) * 4
            x_tail += int(math.cos(self.a + math.pi / 2) * tail_oscillation)
            y_tail += int(math.sin(self.a + math.pi / 2) * tail_oscillation)
            
            radius_tail = max(int(-self.edad * 0.4 * (self.numeroelementos - i) * 2 + 1), 1)
            cv2.circle(frame, (x_tail, y_tail), radius_tail, color_main, -1, cv2.LINE_AA)

    def vive(self, frame):
        if random.random() < 0.002:
            self.direcciongiro = -self.direcciongiro
        if self.energia > 0:
            self.tiempo += self.avancevida
            self.mueve()
        self.energia -= 0.00003
        self.edad += 0.00001
        if self.edad > 3:
            self.energia = 0
        if self.energia > 0:
            self.dibuja(frame)

    def mueve(self):
        self.is_avoiding_collision = False  # Reset collision avoidance flag

        # Avoidance logic
        repulsion_radius = 50
        collision_radius = 10  # Radius below which the fish must turn sharply to avoid overlapping
        repulsion_force = 0.05  # How strongly the fish should avoid nearby fish
        avg_repulsion_x, avg_repulsion_y = 0, 0
        nearby_fish_count = 0

        for other_fish in peces:
            if other_fish != self:
                dist = math.hypot(self.x - other_fish.x, self.y - other_fish.y)
                if dist < repulsion_radius:
                    # Compute repulsion vector
                    repulsion_x = self.x - other_fish.x
                    repulsion_y = self.y - other_fish.y
                    avg_repulsion_x += repulsion_x / dist  # Normalize and add
                    avg_repulsion_y += repulsion_y / dist
                    nearby_fish_count += 1

                    if dist < collision_radius:
                        # Immediate collision avoidance
                        self.is_avoiding_collision = True
                        # Adjust target angle sharply away from other fish
                        avoidance_angle = math.atan2(repulsion_y, repulsion_x)
                        self.target_angle = avoidance_angle
                        break  # Exit the loop to handle immediate collision

        # If there are nearby fish to avoid and not in immediate collision
        if nearby_fish_count > 0 and not self.is_avoiding_collision:
            avg_repulsion_x /= nearby_fish_count
            avg_repulsion_y /= nearby_fish_count
            avoidance_angle = math.atan2(avg_repulsion_y, avg_repulsion_x)
            
            # Blend avoidance angle with the current target angle
            self.target_angle = (self.target_angle + avoidance_angle * repulsion_force) % (2 * math.pi)
            self.is_avoiding_collision = True

        # Adjust speed based on behavior
        if self.is_chasing_food or self.is_avoiding_collision:
            self.speed = self.max_speed
        else:
            self.speed = self.initial_speed

        # Regular movement logic
        angle_diff = angle_difference(self.a, self.target_angle)
        if abs(angle_diff) > self.max_turn_rate:
            angle_change = self.max_turn_rate if angle_diff > 0 else -self.max_turn_rate
            self.a += angle_change
        else:
            self.a = self.target_angle

        self.a = (self.a + math.pi) % (2 * math.pi) - math.pi
        self.x += math.cos(self.a) * self.speed * self.edad * 5
        self.y += math.sin(self.a) * self.speed * self.edad * 5
        self.colisiona()

    def colisiona(self):
        if self.x < 0 or self.x > width or self.y < 0 or self.y > height:
            # If out of bounds, set a target angle to turn 180 degrees slowly
            self.target_angle = (self.a + math.pi) % (2 * math.pi)
            self.is_avoiding_collision = True  # Increase speed to avoid boundary

class Comida:
    def __init__(self, x=None, y=None, radius=None, angle=None):
        # Initialize with provided values if given (for splitting), else randomize
        self.x = x if x is not None else random.uniform(0, width)
        self.y = y if y is not None else random.uniform(0, height)
        self.radio = radius if radius is not None else random.uniform(5, 15)
        self.a = angle if angle is not None else random.uniform(0, math.pi * 2)
        self.v = random.uniform(0, 0.25)
        self.visible = True
        self.vida = 0
        self.transparencia = 1.0

    def dibuja(self, frame):
        if self.visible:
            color = (255, 255, 255)
            radius = max(int(self.radio), 1)
            cv2.circle(frame, (int(self.x), int(self.y)), radius, color, -1, cv2.LINE_AA)

    def vive(self, frame):
        # Wandering logic
        if random.random() < 0.1:
            self.a += (random.random() - 0.5) * 0.2

        # Move based on direction and speed
        self.x += math.cos(self.a) * self.v
        self.y += math.sin(self.a) * self.v

        # Boundary conditions
        if self.x < 0:
            self.x = 0
            self.a = -self.a
        elif self.x > width:
            self.x = width
            self.a = -self.a
        if self.y < 0:
            self.y = 0
            self.a = math.pi - self.a
        elif self.y > height:
            self.y = height
            self.a = math.pi - self.a

        # Handle life cycle and division
        self.vida += 1
        if self.vida % fps == 0 and self.radio >= 2:  # Divide every second if radius is >= 2
            self.divide()

        # Remove particle if too small
        if self.radio < 1:
            self.visible = False

        self.dibuja(frame)

    def divide(self):
        # Create two new particles with half the radius and opposite directions
        angle_offset = math.pi  # 180 degrees
        child_radius = self.radio / 1.4

        if child_radius >= 1:
            # Create two new particles in opposite directions
            food1 = Comida(self.x, self.y, child_radius, self.a)
            food2 = Comida(self.x, self.y, child_radius, (self.a + angle_offset) % (2 * math.pi))
            
            # Add new particles to the global list
            comidas.extend([food1, food2])

        # Mark this particle as invisible to "remove" it
        self.visible = False


# Initialize fishes and food
numeropeces = random.randint(5, 10)
peces = [Pez() for _ in range(numeropeces)]
comidas = [Comida()]

# Main loop
for frame_count in range(total_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if random.random() < 0.00002 * numeropeces:
        comidas.append(Comida())
    for comida in comidas:
        comida.vive(frame)

    # Fish behavior and food interaction
    for pez in peces:
        # Reset chasing food flag
        pez.is_chasing_food = False

        food_in_radius = [comida for comida in comidas if comida.visible and math.hypot(pez.x - comida.x, pez.y - comida.y) < 300]
        
        if food_in_radius:
            closest_food = min(food_in_radius, key=lambda comida: math.hypot(pez.x - comida.x, pez.y - comida.y))
            angleRadians = angle_in_radians(pez.x, pez.y, closest_food.x, closest_food.y)
            pez.target_angle = angleRadians
            pez.is_chasing_food = True  # Fish is chasing food

            if math.hypot(pez.x - closest_food.x, pez.y - closest_food.y) < 10:
                closest_food.visible = False
                pez.energia += closest_food.radio / 10
        else:
            if random.random() < 0.05:
                pez.target_angle += (random.random() - 0.5) * 0.05

    # Update each fish's state and spawn new fish for any that die
    for pez in peces[:]:  # Make a shallow copy of the list to safely modify it
        pez.vive(frame)
        if pez.energia <= 0:
            peces.remove(pez)
            peces.append(Pez())  # Spawn a new fish when one dies

    # Clean up invisible food items and update the frame
    comidas = [comida for comida in comidas if comida.visible]
    video_writer.write(frame)
    cv2.imshow('Fish Simulation', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if frame_count % fps == 0:
        print(f'Progress: {frame_count // fps}/{duration} seconds')

video_writer.release()
cv2.destroyAllWindows()
print('Video saved as fish_simulation.mp4')
