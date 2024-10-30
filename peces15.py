import cv2
import numpy as np
import random
import math
import os
import time

# Video settings
width, height = 1920, 1080
fps = 30
duration = 60*60*12  # seconds
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
        # Initial attributes for fish properties
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.a = random.uniform(0, math.pi * 2)
        self.edad = (random.uniform(2, 4)) / 2
        self.tiempo = random.uniform(0, 1)
        self.avancevida = random.uniform(0, 0.1)
        self.sexo = random.randint(0, 1)  # 0 = male, 1 = female
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.energia = random.uniform(0, 1)
        
        # Additional attributes for appearance and behavior
        self.direcciongiro = random.choice([-1, 0, 1])
        self.numeroelementos = 10
        self.numeroelementoscola = 5
        self.colorr = [self.color[0] + random.randint(-50, 50) for _ in range(-1, self.numeroelementos)]
        self.colorg = [self.color[1] + random.randint(-50, 50) for _ in range(-1, self.numeroelementos)]
        self.colorb = [self.color[2] + random.randint(-50, 50) for _ in range(-1, self.numeroelementos)]
        self.anguloanterior = 0
        self.giro = 0
        self.max_turn_rate = random.uniform(0.005, 0.01)
        self.target_angle = self.a

        # Reproduction control
        self.is_mating = False
        self.mate_target = None

    def mate(self, other):
        # Create two offspring with inherited traits
        offspring1 = Pez()
        offspring2 = Pez()
        
        # Inherit or mix specific traits from parents
        offspring1.color = self.color if random.random() < 0.5 else other.color
        offspring2.color = other.color if random.random() < 0.5 else self.color
        offspring1.avancevida = (self.avancevida + other.avancevida) / 2
        offspring2.avancevida = (self.avancevida + other.avancevida) / 2
        offspring1.sexo = random.randint(0, 1)
        offspring2.sexo = random.randint(0, 1)
        
        # Position offspring near parents
        offspring1.x = self.x + random.uniform(-10, 10)
        offspring1.y = self.y + random.uniform(-10, 10)
        offspring2.x = self.x + random.uniform(-10, 10)
        offspring2.y = self.y + random.uniform(-10, 10)
        
        # Add offspring to the global fish list
        peces.extend([offspring1, offspring2])

    def approach(self, target):
        # Move towards target fish
        angle_to_target = angle_in_radians(self.x, self.y, target.x, target.y)
        self.target_angle = angle_to_target

        # Mate if close enough
        if math.hypot(self.x - target.x, self.y - target.y) < 10:
            self.mate(target)
            self.energia -= 0.4  # Energy cost for mating
            target.energia -= 0.4
            self.is_mating = False
            target.is_mating = False
            self.mate_target = None
            target.mate_target = None

    def dibuja(self, frame):
        # Drawing logic for the fish (body, eyes, tail, etc.)
        color_main = self.color if self.energia > 0 else (128, 128, 128)
        mouth_radius = max(int(math.sin(self.tiempo * 2) * 2 + 3), 1)
        x_mouth = int(self.x + math.cos(self.a) * 5 * self.edad)
        y_mouth = int(self.y + math.sin(self.a) * 5 * self.edad)
        cv2.circle(frame, (x_mouth, y_mouth), mouth_radius, color_main, -1, cv2.LINE_AA)

        for i in range(-1, self.numeroelementos):
            if i == 1:
                for sign in [-1, 1]:
                    x_eye = int(self.x + sign * math.cos(self.a + math.pi / 2) * 4 * self.edad - i * math.cos(self.a) * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    y_eye = int(self.y + sign * math.sin(self.a + math.pi / 2) * 4 * self.edad - i * math.sin(self.a) * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    radius_eye = max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) / 3), 1)
                    cv2.circle(frame, (x_eye, y_eye), radius_eye, (255, 255, 255), -1, cv2.LINE_AA)

            if i == self.numeroelementos // 2 or i == int(self.numeroelementos / 1.1):
                for sign in [-1, 1]:
                    x_fin = int(self.x + sign * math.cos(self.a + math.pi / 2) * 0.3 * self.edad - i * math.cos(self.a) * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    y_fin = int(self.y + sign * math.sin(self.a + math.pi / 2) * 0.3 * self.edad - i * math.sin(self.a) * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    axes = (max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) * 2), 1), max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1)), 1))
                    angle = math.degrees(self.a + math.pi / 2 - math.cos(self.tiempo * 2) * sign)
                    cv2.ellipse(frame, (x_fin, y_fin), axes, angle, 0, 360, color_main, -1, cv2.LINE_AA)

        for i in range(-1, self.numeroelementos):
            x_body = int(self.x - i * math.cos(self.a) * 2 * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            y_body = int(self.y - i * math.sin(self.a) * 2 * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            radius_body = max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) / 1), 1)
            color_body = (self.colorr[i], self.colorg[i], self.colorb[i])
            cv2.circle(frame, (x_body, y_body), radius_body, color_body, -1, cv2.LINE_AA)

        for i in range(self.numeroelementos, self.numeroelementos + self.numeroelementoscola):
            x_tail = int(self.x - (i - 3) * math.cos(self.a) * 2 * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            y_tail = int(self.y - (i - 3) * math.sin(self.a) * 2 * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            radius_tail = max(int(-self.edad * 0.4 * (self.numeroelementos - i) * 2 + 1), 1)
            cv2.circle(frame, (x_tail, y_tail), radius_tail, color_main, -1, cv2.LINE_AA)

    def vive(self, frame):
        # Check if fish is eligible for mating
        if not self.is_mating and self.energia > 0.6 and 1.5 < self.edad < 3 and random.random() < 0.05:
            potential_mates = [other for other in peces if other != self and other.sexo != self.sexo
                               and other.energia > 0.6 and 1.5 < other.edad < 3]
            if potential_mates:
                self.mate_target = random.choice(potential_mates)
                self.is_mating = True
                self.mate_target.is_mating = True
                self.mate_target.mate_target = self

        # Approach target if in mating state
        if self.is_mating and self.mate_target:
            self.approach(self.mate_target)
        else:
            # Regular movement and survival mechanics
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
        # Existing movement and collision logic
        angle_diff = angle_difference(self.a, self.target_angle)
        if abs(angle_diff) > self.max_turn_rate:
            self.a += self.max_turn_rate if angle_diff > 0 else -self.max_turn_rate
        else:
            self.a = self.target_angle

        self.x += math.cos(self.a) * self.avancevida * self.edad * 5
        self.y += math.sin(self.a) * self.avancevida * self.edad * 5
        self.colisiona()

    def colisiona(self):
        if self.x < 0 or self.x > width or self.y < 0 or self.y > height:
            self.target_angle = (self.a + math.pi) % (2 * math.pi)


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
numeropeces = 200
peces = [Pez() for _ in range(numeropeces)]
comidas = [Comida()]

# Main loop
for frame_count in range(total_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if random.random() < 0.01:
        comidas.append(Comida())
    for comida in comidas:
        comida.vive(frame)

    for pez in peces:
        food_in_radius = [comida for comida in comidas if comida.visible and math.hypot(pez.x - comida.x, pez.y - comida.y) < 300]
        
        if food_in_radius:
            closest_food = min(food_in_radius, key=lambda comida: math.hypot(pez.x - comida.x, pez.y - comida.y))
            angleRadians = angle_in_radians(pez.x, pez.y, closest_food.x, closest_food.y)
            pez.target_angle = angleRadians
            
            if math.hypot(pez.x - closest_food.x, pez.y - closest_food.y) < 10:
                closest_food.visible = False
                pez.energia += closest_food.radio / 10
        else:
            if random.random() < 0.05:
                pez.target_angle += (random.random() - 0.5) * 0.05

    for pez in peces:
        pez.vive(frame)

    peces = [pez for pez in peces if pez.energia > 0]
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
