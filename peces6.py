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
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.a = random.uniform(0, math.pi * 2)
        self.edad = (random.uniform(2, 4)) / 2
        self.tiempo = random.uniform(0, 1)
        self.avancevida = random.uniform(0, 0.1)
        self.sexo = random.randint(0, 1)
        
        # Color aleatorio independiente del sexo
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Color específico de acuerdo al sexo para distinguir
        if self.sexo == 0:  # Macho
            self.r = random.randint(127, 254)
            self.g = self.r // 3
            self.b = self.r // 3
        else:  # Hembra
            self.g = random.randint(127, 254)
            self.r = self.g // 3
            self.b = self.g // 3
            
        self.energia = random.uniform(0, 1)
        self.direcciongiro = random.choice([-1, 0, 1])
        self.numeroelementos = 10
        self.numeroelementoscola = 5
        self.colorr = []
        self.colorg = []
        self.colorb = []
        for i in range(-1, self.numeroelementos):
            self.colorr.append(self.color[0] + random.randint(-50, 50))
            self.colorg.append(self.color[1] + random.randint(-50, 50))
            self.colorb.append(self.color[2] + random.randint(-50, 50))
        self.anguloanterior = 0
        self.giro = 0
        self.max_turn_rate = random.uniform(0.02, 0.05)  # Maximum angular speed

        self.target_angle = self.a  # Desired angle

    def dibuja(self, frame):
        if self.energia > 0:
            color_main = self.color
        else:
            color_main = (128, 128, 128)
        numeroelementos = self.numeroelementos
        numeroelementoscola = self.numeroelementoscola

        for i in range(-1, numeroelementos):
            if i == 1:
                # Eyes
                for sign in [-1, 1]:
                    x_eye = int(self.x + sign * math.cos(self.a + math.pi / 2) * 4 * self.edad - i * math.cos(self.a) * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    y_eye = int(self.y + sign * math.sin(self.a + math.pi / 2) * 4 * self.edad - i * math.sin(self.a) * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    radius_eye = max(int((self.edad * 0.4 * (numeroelementos - i) + 1) / 3), 1)
                    cv2.circle(frame, (x_eye, y_eye), radius_eye, (255, 255, 255), -1, cv2.LINE_AA)
            # Fins
            if i == numeroelementos // 2 or i == int(numeroelementos / 1.1):
                for sign in [-1, 1]:
                    x_fin = int(self.x + sign * math.cos(self.a + math.pi / 2) * 0.3 * self.edad - i * math.cos(self.a) * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    y_fin = int(self.y + sign * math.sin(self.a + math.pi / 2) * 0.3 * self.edad - i * math.sin(self.a) * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
                    axes = (max(int((self.edad * 0.4 * (numeroelementos - i) + 1) * 2), 1), max(int((self.edad * 0.4 * (numeroelementos - i) + 1)), 1))
                    angle = math.degrees(self.a + math.pi / 2 - math.cos(self.tiempo * 2) * sign)
                    cv2.ellipse(frame, (x_fin, y_fin), axes, angle, 0, 360, color_main, -1, cv2.LINE_AA)

        for i in range(-1, numeroelementos):
            x_body = int(self.x - i * math.cos(self.a) * 2 * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            y_body = int(self.y - i * math.sin(self.a) * 2 * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            radius_body = max(int((self.edad * 0.4 * (numeroelementos - i) + 1) / 1), 1)
            color_body = (self.colorr[i], self.colorg[i], self.colorb[i])
            cv2.circle(frame, (x_body, y_body), radius_body, color_body, -1, cv2.LINE_AA)

        # Tail
        for i in range(numeroelementos, numeroelementos + numeroelementoscola):
            x_tail = int(self.x - (i - 3) * math.cos(self.a) * 2 * self.edad + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            y_tail = int(self.y - (i - 3) * math.sin(self.a) * 2 * self.edad + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4)
            radius_tail = max(int(-self.edad * 0.4 * (numeroelementos - i) * 2 + 1), 1)
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
        angle_diff = angle_difference(self.a, self.target_angle)
        if abs(angle_diff) > self.max_turn_rate:
            angle_change = self.max_turn_rate if angle_diff > 0 else -self.max_turn_rate
            self.a += angle_change
        else:
            self.a = self.target_angle

        self.a = (self.a + math.pi) % (2 * math.pi) - math.pi
        self.x += math.cos(self.a) * self.avancevida * self.edad * 5
        self.y += math.sin(self.a) * self.avancevida * self.edad * 5
        self.colisiona()

    def colisiona(self):
        if self.x < 0:
            self.x = 40
            self.a = 0
        if self.y < 0:
            self.y = 40
            self.a = math.pi / 2
        if self.x > width:
            self.x = width - 40
            self.a = math.pi
        if self.y > height:
            self.y = height - 40
            self.a = -math.pi / 2
        if self.x < 200 or self.x > width - 200 or self.y < 200 or self.y > height - 200:
            self.target_angle += 0.05 * self.direcciongiro

class Comida:
    def __init__(self):
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.visible = True
        self.vida = 0
        self.radio = random.uniform(0, 10)
        self.a = random.uniform(0, math.pi * 2)
        self.v = random.uniform(0, 0.25)
        self.transparencia = 1.0

    def dibuja(self, frame):
        if self.visible:
            color = (255, 255, 255)
            radius = max(int(self.radio), 1)
            cv2.circle(frame, (int(self.x), int(self.y)), radius, color, -1, cv2.LINE_AA)

    def vive(self, frame):
        self.vida += 1
        self.dibuja(frame)

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

    numerocomidasvisibles = sum(1 for comida in comidas if comida.visible)
    for pez in peces:
        if numerocomidasvisibles > 0:
            if pez.energia < 5 and pez.energia > 0:
                distancias = []
                for comida in comidas:
                    if comida.visible:
                        distancia = math.hypot(pez.x - comida.x, pez.y - comida.y)
                        distancias.append((distancia, comida))
                if distancias:
                    distanciamenor, comida_cercana = min(distancias, key=lambda x: x[0])
                    angleRadians = angle_in_radians(pez.x, pez.y, comida_cercana.x, comida_cercana.y)
                    pez.target_angle = angleRadians
                    if distanciamenor < 10:
                        comida_cercana.visible = False
                        pez.energia += comida_cercana.radio / 10
        else:
            if random.random() < 0.1:
                pez.target_angle += (random.random() - 0.5) * 0.1

    for pez in peces:
        pez.vive(frame)

    peces = [pez for pez in peces if pez.energia > 0]
    comidas = [comida for comida in comidas if comida.visible]
    video_writer.write(frame)
    cv2.imshow('Fish Simulation', frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit early
        break
    if frame_count % fps == 0:
        print(f'Progress: {frame_count // fps}/{duration} seconds')

video_writer.release()
cv2.destroyAllWindows()
print('Video saved as fish_simulation.mp4')
