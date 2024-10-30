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

class Particle:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.visible = True
        # Add movement attributes
        self.a = random.uniform(0, math.pi * 2)  # Random initial angle
        self.v = random.uniform(0.1,0.2)  # Random speed
        self.frames_since_birth = 0  # To track time for radius decrease

    def update(self):
        # Move the particle in its current direction with some random variation
        self.a += (random.random() - 0.5) * 0.2  # Slight random turn
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

        # Decrease radius by 1 pixel per second
        self.frames_since_birth += 1
        if self.frames_since_birth % fps == 0:
            self.radius -= 1  # Decrease radius by 1 pixel per second
        if self.radius < 1:
            self.visible = False

    def draw(self, frame):
        if self.visible and self.radius >= 1:
            cv2.circle(frame, (int(self.x), int(self.y)), int(self.radius), self.color, -1, cv2.LINE_AA)

class Pez:
    def __init__(self):
        # Atributos iniciales de las propiedades del pez
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.a = random.uniform(0, math.pi * 2)
        self.edad = max((random.uniform(2, 4)) / 2, 1.0)  # Asegurar edad mínima
        self.tiempo = random.uniform(0, 1)
        self.avancevida = random.uniform(0.5, 1.0)  # Aumentar velocidad de movimiento
        self.sexo = random.randint(0, 1)  # 0 = macho, 1 = hembra
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        self.energia = random.uniform(50, 100)  # Energía inicial suficiente
        self.particles_generated = False

        # Atributos adicionales para apariencia y comportamiento
        self.direcciongiro = random.choice([-1, 0, 1])
        self.numeroelementos = 10
        self.numeroelementoscola = 5
        self.colorr = [
            min(max(self.color[0] + random.randint(-50, 50), 0), 255)
            for _ in range(-1, self.numeroelementos)
        ]
        self.colorg = [
            min(max(self.color[1] + random.randint(-50, 50), 0), 255)
            for _ in range(-1, self.numeroelementos)
        ]
        self.colorb = [
            min(max(self.color[2] + random.randint(-50, 50), 0), 255)
            for _ in range(-1, self.numeroelementos)
        ]
        self.anguloanterior = 0
        self.giro = 0
        self.max_turn_rate = random.uniform(0.1, 0.2)  # Permitir giros más rápidos
        self.target_angle = self.a

        # Control de reproducción
        self.is_mating = False
        self.mate_target = None

        # Atributos para la muerte y partículas
        self.alive = True
        self.body_parts = []  # Inicializar partes del cuerpo

    def mate(self, other):
        # Determinar el número de descendientes N
        N = random.randint(1, 5)  # Número variable de descendientes

        offspring_list = []
        for _ in range(N):
            offspring = Pez()
            # Heredar o mezclar rasgos específicos de los padres
            offspring.color = (
                self.color if random.random() < 0.5 else other.color
            )
            offspring.avancevida = (self.avancevida + other.avancevida) / 2
            if offspring.avancevida == 0:
                offspring.avancevida = random.uniform(0.5, 1.0)  # Asegurar movimiento
            offspring.sexo = random.randint(0, 1)
            offspring.energia = 0.5  # Establecer energía inicial para el descendiente

            # Posicionar al descendiente cerca de los padres
            offspring.x = (self.x + other.x) / 2 + random.uniform(-10, 10)
            offspring.y = (self.y + other.y) / 2 + random.uniform(-10, 10)

            offspring_list.append(offspring)

        # Costo de energía por apareamiento
        energy_cost = 0.2 * N  # Ajustar el costo de energía por descendiente
        self.energia -= energy_cost
        other.energia -= energy_cost

        # Agregar descendientes a la lista global de peces
        peces.extend(offspring_list)

    def approach(self, target):
        # Verificar si el compañero es válido
        if not target.alive or target not in peces:
            self.is_mating = False
            self.mate_target = None
            return
        # Moverse hacia el pez objetivo
        angle_to_target = angle_in_radians(self.x, self.y, target.x, target.y)
        self.target_angle = angle_to_target

        # Aparearse si están lo suficientemente cerca
        if math.hypot(self.x - target.x, self.y - target.y) < 10:
            self.mate(target)
            self.is_mating = False
            target.is_mating = False
            self.mate_target = None
            target.mate_target = None

    def dibuja(self, frame):
        # Lógica de dibujo del pez (cuerpo, ojos, cola, etc.)
        color_main = self.color if self.energia > 0 else (128, 128, 128)
        mouth_radius = max(int(math.sin(self.tiempo * 2) * 2 + 3), 1)
        x_mouth = int(self.x + math.cos(self.a) * 5 * self.edad)
        y_mouth = int(self.y + math.sin(self.a) * 5 * self.edad)
        if frame is not None:
            cv2.circle(
                frame,
                (x_mouth, y_mouth),
                mouth_radius,
                color_main,
                -1,
                cv2.LINE_AA,
            )
        # Guardar datos de la boca
        self.body_parts.append(
            {"x": x_mouth, "y": y_mouth, "radius": mouth_radius, "color": color_main}
        )

        for i in range(-1, self.numeroelementos):
            if i == 1:
                for sign in [-1, 1]:
                    x_eye = int(
                        self.x
                        + sign
                        * math.cos(self.a + math.pi / 2)
                        * 4
                        * self.edad
                        - i * math.cos(self.a) * self.edad
                        + math.sin(self.a)
                        * math.sin((i / 5) - self.tiempo)
                        * 4
                    )
                    y_eye = int(
                        self.y
                        + sign
                        * math.sin(self.a + math.pi / 2)
                        * 4
                        * self.edad
                        - i * math.sin(self.a) * self.edad
                        + math.cos(self.a)
                        * math.sin((i / 5) - self.tiempo)
                        * 4
                    )
                    radius_eye = max(
                        int(
                            (self.edad * 0.4 * (self.numeroelementos - i) + 1) / 3
                        ),
                        1,
                    )
                    if frame is not None:
                        cv2.circle(
                            frame,
                            (x_eye, y_eye),
                            radius_eye,
                            (255, 255, 255),
                            -1,
                            cv2.LINE_AA,
                        )
                    # Guardar datos del ojo
                    self.body_parts.append(
                        {
                            "x": x_eye,
                            "y": y_eye,
                            "radius": radius_eye,
                            "color": (255, 255, 255),
                        }
                    )

            if i == self.numeroelementos // 2 or i == int(
                self.numeroelementos / 1.1
            ):
                for sign in [-1, 1]:
                    x_fin = int(
                        self.x
                        + sign
                        * math.cos(self.a + math.pi / 2)
                        * 0.3
                        * self.edad
                        - i * math.cos(self.a) * self.edad
                        + math.sin(self.a)
                        * math.sin((i / 5) - self.tiempo)
                        * 4
                    )
                    y_fin = int(
                        self.y
                        + sign
                        * math.sin(self.a + math.pi / 2)
                        * 0.3
                        * self.edad
                        - i * math.sin(self.a) * self.edad
                        + math.cos(self.a)
                        * math.sin((i / 5) - self.tiempo)
                        * 4
                    )
                    axes = (
                        max(
                            int(
                                (self.edad * 0.4 * (self.numeroelementos - i) + 1)
                                * 2
                            ),
                            1,
                        ),
                        max(
                            int(
                                self.edad * 0.4 * (self.numeroelementos - i) + 1
                            ),
                            1,
                        ),
                    )
                    angle = math.degrees(
                        self.a + math.pi / 2 - math.cos(self.tiempo * 2) * sign
                    )
                    if frame is not None:
                        cv2.ellipse(
                            frame,
                            (x_fin, y_fin),
                            axes,
                            angle,
                            0,
                            360,
                            color_main,
                            -1,
                            cv2.LINE_AA,
                        )
                    # Guardar datos de la aleta
                    self.body_parts.append(
                        {
                            "x": x_fin,
                            "y": y_fin,
                            "radius": axes[0],
                            "color": color_main,
                        }
                    )

        for i in range(-1, self.numeroelementos):
            x_body = int(
                self.x
                - i * math.cos(self.a) * 2 * self.edad
                + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4
            )
            y_body = int(
                self.y
                - i * math.sin(self.a) * 2 * self.edad
                + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4
            )
            radius_body = max(
                int((self.edad * 0.4 * (self.numeroelementos - i) + 1) / 1), 1
            )
            color_body = (
                self.colorr[i % len(self.colorr)],
                self.colorg[i % len(self.colorg)],
                self.colorb[i % len(self.colorb)],
            )
            if frame is not None:
                cv2.circle(
                    frame,
                    (x_body, y_body),
                    radius_body,
                    color_body,
                    -1,
                    cv2.LINE_AA,
                )
            # Guardar datos del cuerpo
            self.body_parts.append(
                {
                    "x": x_body,
                    "y": y_body,
                    "radius": radius_body,
                    "color": color_body,
                }
            )

        for i in range(
            self.numeroelementos, self.numeroelementos + self.numeroelementoscola
        ):
            x_tail = int(
                self.x
                - (i - 3) * math.cos(self.a) * 2 * self.edad
                + math.sin(self.a) * math.sin((i / 5) - self.tiempo) * 4
            )
            y_tail = int(
                self.y
                - (i - 3) * math.sin(self.a) * 2 * self.edad
                + math.cos(self.a) * math.sin((i / 5) - self.tiempo) * 4
            )
            radius_tail = max(
                int(
                    -self.edad * 0.4 * (self.numeroelementos - i) * 2 + 1
                ),
                1,
            )
            if frame is not None:
                cv2.circle(
                    frame,
                    (x_tail, y_tail),
                    radius_tail,
                    color_main,
                    -1,
                    cv2.LINE_AA,
                )
            # Guardar datos de la cola
            self.body_parts.append(
                {
                    "x": x_tail,
                    "y": y_tail,
                    "radius": radius_tail,
                    "color": color_main,
                }
            )

    def vive(self, frame):
        # Limpiar partes del cuerpo anteriores si el pez está vivo
        if self.alive:
            self.body_parts = []

        # Comprobar si el pez es elegible para aparearse
        if (
            not self.is_mating
            and self.energia > 0.6
            and 1.5 < self.edad < 3
            and random.random() < 0.05
        ):
            potential_mates = [
                other
                for other in peces
                if other != self
                and other.sexo != self.sexo
                and other.energia > 0.6
                and 1.5 < other.edad < 3
                and other.alive  # Asegurar que el compañero está vivo
            ]
            if potential_mates:
                self.mate_target = random.choice(potential_mates)
                self.is_mating = True
                self.mate_target.is_mating = True
                self.mate_target.mate_target = self

        # Verificar si el compañero de apareamiento sigue siendo válido
        if self.is_mating:
            if (
                self.mate_target is None
                or not self.mate_target.alive
                or self.mate_target not in peces
            ):
                self.is_mating = False
                self.mate_target = None

        # Movimiento y supervivencia
        if self.energia > 0:
            self.edad = max(self.edad, 1.0)  # Asegurar edad mínima
            self.tiempo += self.avancevida
            if self.is_mating and self.mate_target:
                self.approach(self.mate_target)
            else:
                self.mueve()
        self.energia -= 0.00001  # Pérdida de energía por frame
        self.edad += 0.00001
        if self.edad > 3:
            self.energia = 0

        # Manejar la muerte y generación de partículas
        if self.energia <= 0 and not self.particles_generated:
            self.alive = False
            self.dibuja(None)  # Asegurar que body_parts está poblado
            # Generar partículas de body_parts
            for part in self.body_parts:
                particle = Particle(
                    part["x"], part["y"], part["radius"], part["color"]
                )
                particles.append(particle)
            self.particles_generated = True

        # Dibujar el pez si está vivo
        if self.energia > 0:
            self.dibuja(frame)

    def mueve(self):
        # Movimiento y lógica de colisión
        angle_diff = angle_difference(self.a, self.target_angle)
        if abs(angle_diff) > self.max_turn_rate:
            self.a += (
                self.max_turn_rate if angle_diff > 0 else -self.max_turn_rate
            )
        else:
            self.a = self.target_angle

        delta_x = math.cos(self.a) * self.avancevida * self.edad * 5
        delta_y = math.sin(self.a) * self.avancevida * self.edad * 5

        # Imprimir valores de depuración
##        print(f"Pez ID {id(self)}:")
##        print(f"  Posición antes: ({self.x:.2f}, {self.y:.2f})")
##        print(f"  Delta posición: ({delta_x:.5f}, {delta_y:.5f})")
##        print(f"  avancevida: {self.avancevida:.5f}, edad: {self.edad:.5f}")
##        print(f"  Ángulo actual: {self.a:.5f}, Ángulo objetivo: {self.target_angle:.5f}")
##        print(f"  max_turn_rate: {self.max_turn_rate:.5f}")
##        print(f"  color: {self.color}")

        self.x += delta_x
        self.y += delta_y

        # Comprobar colisiones o proximidad a los bordes
        self.colisiona()

    def colisiona(self):
        edge_margin = 50  # Distancia segura del borde en píxeles

        # Si está cerca de cualquier borde, dirigirse hacia el centro
        if (
            self.x < edge_margin
            or self.x > width - edge_margin
            or self.y < edge_margin
            or self.y > height - edge_margin
        ):
            # Calcular ángulo hacia el centro con ligera variación aleatoria
            center_x, center_y = width / 2, height / 2
            angle_to_center = angle_in_radians(
                self.x, self.y, center_x, center_y
            )
            random_variation = (random.random() - 0.5) * 0.1  # Ajustar según sea necesario
            self.target_angle = angle_to_center + random_variation




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


# Initialize fishes, food, and particles
numeropeces = 5
peces = [Pez() for _ in range(numeropeces)]
comidas = [Comida()]
particles = []  # List to hold particles from dead fishes

# Main loop
for frame_count in range(total_frames):
    #print("------------------")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if random.random() < 0.0004*numeropeces:
        comidas.append(Comida())
    for comida in comidas:
        comida.vive(frame)

    for pez in peces:
        if not pez.is_mating:
            food_in_radius = [
                comida for comida in comidas
                if comida.visible and math.hypot(pez.x - comida.x, pez.y - comida.y) < 300
            ]
            if food_in_radius:
                closest_food = min(
                    food_in_radius, key=lambda comida: math.hypot(pez.x - comida.x, pez.y - comida.y)
                )
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

    # Update and draw particles
    for particle in particles:
        particle.update()
        particle.draw(frame)
    particles = [p for p in particles if p.visible]

    peces = [pez for pez in peces if pez.energia > 0 or pez.alive]  # Keep dead fish until particles are generated
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
