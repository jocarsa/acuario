import cv2
import numpy as np
import random
import math
import os
import time
from collections import deque

# =========================
# Video / sim settings
# =========================
width, height = 1080, 1080
fps = 30
duration = 60*1
total_frames = fps * duration

video_dir = 'videos'
os.makedirs(video_dir, exist_ok=True)
epoch_time = int(time.time())
video_path = os.path.join(video_dir, f'fish_simulation_{epoch_time}.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# =========================
# Helpers
# =========================
def angle_difference(beta, alpha):
    difference = alpha - beta
    while difference > math.pi:
        difference -= 2 * math.pi
    while difference < -math.pi:
        difference += 2 * math.pi
    return difference

def angle_in_radians(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def clamp(val, a, b):
    return max(a, min(b, val))

def make_vertical_gradient(h, w, top_color, bottom_color):
    top = np.array(top_color, dtype=np.float32)
    bottom = np.array(bottom_color, dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    grad = (1.0 - t) * top + t * bottom
    grad = np.repeat(grad[None, :, :], w, axis=0).transpose(1,0,2)
    return grad.astype(np.uint8)

def premult_blur(img, mask, ksize):
    """
    Premultiply RGB by alpha, then blur both RGB and alpha.
    img:  uint8 HxWx3 (BGR)
    mask: uint8 HxW   (0..255)
    ksize: (k,k) odd kernel for GaussianBlur
    Returns: (src_pm_f32, alpha_f32 [0..1])
    """
    a = (mask.astype(np.float32) / 255.0)  # HxW [0..1]
    img_f = img.astype(np.float32)
    img_pm = img_f * a[..., None]          # premultiply
    img_pm_blur = cv2.GaussianBlur(img_pm, ksize, 0)
    a_blur = cv2.GaussianBlur(a, ksize, 0)
    a_blur = np.clip(a_blur, 0.0, 1.0)
    return img_pm_blur, a_blur

def alpha_over_pm(dst_bgr_u8, src_pm_f32, alpha_f32):
    """
    'Over' compositing with premultiplied src:
      out = src_pm + dst*(1-a)
    """
    dst = dst_bgr_u8.astype(np.float32)
    out = src_pm_f32 + dst * (1.0 - alpha_f32)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

# Precompute background gradient (BGR: light -> deep blue-ish)
BG_GRADIENT = make_vertical_gradient(
    height, width,
    top_color=(240, 220, 180),   # light bluish tint (B,G,R)
    bottom_color=(80, 60, 40)    # deep blue-ish
)

# =========================
# Spatial index
# =========================
class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x  # center
        self.y = y
        self.w = w  # half-width
        self.h = h  # half-height

    def contains(self, entity):
        return (self.x - self.w <= entity.x <= self.x + self.w and
                self.y - self.h <= entity.y <= self.y + self.h)

    def intersects(self, range):
        return not (range.x - range.w > self.x + self.w or
                    range.x + range.w < self.x - self.w or
                    range.y - range.h > self.y + self.h or
                    range.y + range.h < self.y - self.h)

class Quadtree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary
        self.capacity = capacity
        self.entities = []
        self.divided = False

    def subdivide(self):
        x, y = self.boundary.x, self.boundary.y
        w, h = self.boundary.w/2, self.boundary.h/2
        self.northeast = Quadtree(Rectangle(x + w, y - h, w, h), self.capacity)
        self.northwest = Quadtree(Rectangle(x - w, y - h, w, h), self.capacity)
        self.southeast = Quadtree(Rectangle(x + w, y + h, w, h), self.capacity)
        self.southwest = Quadtree(Rectangle(x - w, y + h, w, h), self.capacity)
        self.divided = True

    def insert(self, entity):
        if not self.boundary.contains(entity):
            return False
        if len(self.entities) < self.capacity:
            self.entities.append(entity)
            return True
        if not self.divided:
            self.subdivide()
        return (self.northeast.insert(entity) or
                self.northwest.insert(entity) or
                self.southeast.insert(entity) or
                self.southwest.insert(entity))

    def query(self, range, found):
        if not self.boundary.intersects(range):
            return
        for e in self.entities:
            if range.contains(e):
                found.append(e)
        if self.divided:
            self.northwest.query(range, found)
            self.northeast.query(range, found)
            self.southwest.query(range, found)
            self.southeast.query(range, found)

# =========================
# Food
# =========================
class Comida:
    def __init__(self, x=None, y=None, radius=None, angle=None, v=None, layer=None):
        self.x = x if x is not None else random.uniform(0, width)
        self.y = y if y is not None else random.uniform(0, height)
        self.layer = layer if layer is not None else random.choices([0,1,2], weights=[0.4,0.35,0.25])[0]
        base_r = radius if radius is not None else random.uniform(5, 15)
        scale_by_layer = [0.6, 0.8, 1.0][self.layer]
        self.radio = base_r * scale_by_layer

        self.a = angle if angle is not None else random.uniform(0, math.pi * 2)
        self.v = v if v is not None else random.uniform(0, 0.25) * scale_by_layer
        self.visible = True
        self.vida = 0

    def dibuja(self, img, mask):
        if not self.visible:
            return
        center = (int(self.x), int(self.y))
        radius = max(int(self.radio), 1)
        cv2.circle(img, center, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)

    def vive(self, img, mask):
        if random.random() < 0.1:
            self.a += (random.random() - 0.5) * 0.2
        self.x += math.cos(self.a) * self.v
        self.y += math.sin(self.a) * self.v

        if self.x < 0:
            self.x = 0; self.a = -self.a
        elif self.x > width:
            self.x = width; self.a = -self.a
        if self.y < 0:
            self.y = 0; self.a = math.pi - self.a
        elif self.y > height:
            self.y = height; self.a = math.pi - self.a

        self.vida += 1
        if self.vida % fps == 0 and self.radio >= 2:
            self.divide()

        if self.radio < 1:
            self.visible = False

        self.dibuja(img, mask)

    def divide(self):
        angle_offset = math.pi
        child_radius = self.radio / 1.4
        if child_radius >= 1:
            comidas.extend([
                Comida(self.x, self.y, child_radius, self.a, self.v, layer=self.layer),
                Comida(self.x, self.y, child_radius, (self.a + angle_offset) % (2*math.pi), self.v, layer=self.layer),
            ])
        self.visible = False

# =========================
# Fish
# =========================
WALL_MARGIN = 80
WALL_PUSH_MAX = 1.8
WALL_ESCAPE_BOOST = 1.8
WALL_ESCAPE_FRAMES = int(fps * 0.8)

FISH_ID_SEQ = 0
def next_fish_id():
    global FISH_ID_SEQ
    FISH_ID_SEQ += 1
    return FISH_ID_SEQ

def mutate(val, pct=0.05, minv=None, maxv=None):
    delta = val * pct * (random.random()*2 - 1)
    out = val + delta
    if minv is not None: out = max(minv, out)
    if maxv is not None: out = min(maxv, out)
    return out

class Pez:
    def __init__(self, parent=None):
        if parent is None:
            base_color = (random.randint(40, 215), random.randint(40, 215), random.randint(40, 215))
            self.genome = {
                "base_color": base_color,              # BGR
                "flapping_freq": random.uniform(0.5, 1.0),
                "max_thrust": random.uniform(0.01, 0.03),
                "drag": random.uniform(0.02, 0.04),
                "max_speed": random.uniform(1.2, 2.4),
                "max_turn": random.uniform(0.006, 0.02),
                "perception_food": random.uniform(200, 350),
                "perception_avoid": random.uniform(40, 70),
                "energy_eff": random.uniform(0.08, 0.14),
            }
            self.layer = random.choices([0,1,2], weights=[0.35,0.35,0.30])[0]
        else:
            p = parent.genome
            def mut_c(c): return int(clamp(mutate(c, 0.08), 0, 255))
            base_color = tuple(mut_c(c) for c in p["base_color"])
            self.genome = {
                "base_color": base_color,
                "flapping_freq": mutate(p["flapping_freq"], 0.08, 0.3, 1.4),
                "max_thrust": mutate(p["max_thrust"], 0.08, 0.006, 0.05),
                "drag": mutate(p["drag"], 0.08, 0.01, 0.07),
                "max_speed": mutate(p["max_speed"], 0.08, 0.8, 3.0),
                "max_turn": mutate(p["max_turn"], 0.08, 0.004, 0.03),
                "perception_food": mutate(p["perception_food"], 0.08, 120, 420),
                "perception_avoid": mutate(p["perception_avoid"], 0.08, 30, 100),
                "energy_eff": mutate(p["energy_eff"], 0.08, 0.05, 0.2),
            }
            self.layer = parent.layer if random.random() > 0.1 else random.choice([0,1,2])

        self.numeroelementos = 10
        self.numeroelementoscola = 5
        c0 = self.genome["base_color"]
        self.colorr = [int(clamp(c0[0] + random.randint(-40, 40), 0, 255)) for _ in range(-1, self.numeroelementos)]
        self.colorg = [int(clamp(c0[1] + random.randint(-40, 40), 0, 255)) for _ in range(-1, self.numeroelementos)]
        self.colorb = [int(clamp(c0[2] + random.randint(-40, 40), 0, 255)) for _ in range(-1, self.numeroelementos)]

        self.id = next_fish_id()
        self.gen = 0 if parent is None else getattr(parent, "gen", 0) + 1
        self.x = random.uniform(0, width) if parent is None else clamp(parent.x + random.uniform(-20, 20), 0, width)
        self.y = random.uniform(0, height) if parent is None else clamp(parent.y + random.uniform(-20, 20), 0, height)
        self.a = random.uniform(0, math.pi * 2)
        self.target_angle = self.a
        self.edad = random.uniform(1.0, 2.0)
        self.tiempo = random.uniform(0, 1)
        self.avancevida = random.uniform(0.05, 0.1)
        self.sexo = random.randint(0, 1)
        self.energia = random.uniform(0.6, 1.2) if parent is None else parent.energia * 0.5
        self.speed = random.uniform(0.3, 0.9)
        self.flapping_phase = 0.0
        self.base_flapping_frequency = self.genome["flapping_freq"]
        self.base_max_thrust = self.genome["max_thrust"]
        self.max_turn_rate = self.genome["max_turn"]
        self.drag_coefficient = self.genome["drag"]
        self.is_chasing_food = False
        self.is_avoiding_collision = False

        self.food_memory = deque(maxlen=3)
        self.memory_ttl_frames = fps * 6

        self.previous_positions = deque(maxlen=6)
        self.stuck_counter = 0
        self.is_stuck = False

        self.wall_escape_cooldown = 0

        self.food_eaten = 0
        self.distance_travelled = 0.0
        self.birth_frame = 0

    def color_alive(self):
        return self.genome["base_color"] if self.energia > 0 else (128, 128, 128)

    def dibuja(self, img, mask):
        color_main = self.color_alive()

        def paint_circle(cx, cy, r, col):
            cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)

        def paint_ellipse(center, axes, angle, col):
            cv2.ellipse(img, center, axes, angle, 0, 360, col, -1, cv2.LINE_AA)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1, cv2.LINE_AA)

        mouth_radius = max(int(math.sin(2 * math.pi * self.flapping_phase * 2) * 2 + 3), 1)
        x_mouth = int(self.x + math.cos(self.a) * 5 * self.edad)
        y_mouth = int(self.y + math.sin(self.a) * 5 * self.edad)
        mouth_osc = math.sin(2 * math.pi * self.flapping_phase) * 2
        x_mouth += int(math.cos(self.a + math.pi/2) * mouth_osc)
        y_mouth += int(math.sin(self.a + math.pi/2) * mouth_osc)
        paint_circle(x_mouth, y_mouth, mouth_radius, color_main)

        for i in range(-1, self.numeroelementos):
            if i == 1:
                for sign in [-1, 1]:
                    x_eye = int(self.x + sign * math.cos(self.a + math.pi/2) * 4 * self.edad - i * math.cos(self.a) * self.edad)
                    y_eye = int(self.y + sign * math.sin(self.a + math.pi/2) * 4 * self.edad - i * math.sin(self.a) * self.edad)
                    eye_osc = math.sin((i/5) - 2 * math.pi * self.flapping_phase) * 4
                    x_eye += int(math.cos(self.a + math.pi/2) * eye_osc)
                    y_eye += int(math.sin(self.a + math.pi/2) * eye_osc)
                    radius_eye = max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) / 3), 1)
                    paint_circle(x_eye, y_eye, radius_eye, (255, 255, 255))

        for i in range(-1, self.numeroelementos):
            if i == self.numeroelementos // 2 or i == int(self.numeroelementos / 1.1):
                for sign in [-1, 1]:
                    x_fin = int(self.x + sign * math.cos(self.a + math.pi/2) * 0.3 * self.edad - i * math.cos(self.a) * self.edad)
                    y_fin = int(self.y + sign * math.sin(self.a + math.pi/2) * 0.3 * self.edad - i * math.sin(self.a) * self.edad)
                    fin_osc = math.sin((i/5) - 2 * math.pi * self.flapping_phase) * 4
                    x_fin += int(math.cos(self.a + math.pi/2) * fin_osc)
                    y_fin += int(math.sin(self.a + math.pi/2) * fin_osc)
                    axes = (
                        max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1) * 2), 1),
                        max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1)), 1)
                    )
                    angle = math.degrees(self.a + math.pi/2 - math.cos(2 * math.pi * self.flapping_phase * 2) * sign)
                    paint_ellipse((x_fin, y_fin), axes, angle, color_main)

        for i in range(-1, self.numeroelementos):
            x_body = int(self.x - i * math.cos(self.a) * 2 * self.edad)
            y_body = int(self.y - i * math.sin(self.a) * 2 * self.edad)
            body_osc = math.sin((i/5) - 2 * math.pi * self.flapping_phase) * 4
            x_body += int(math.cos(self.a + math.pi/2) * body_osc)
            y_body += int(math.sin(self.a + math.pi/2) * body_osc)
            radius_body = max(int((self.edad * 0.4 * (self.numeroelementos - i) + 1)), 1)
            color_body = (self.colorr[i], self.colorg[i], self.colorb[i])
            paint_circle(x_body, y_body, radius_body, color_body)

        for i in range(self.numeroelementos, self.numeroelementos + self.numeroelementoscola):
            x_tail = int(self.x - (i - 3) * math.cos(self.a) * 2 * self.edad)
            y_tail = int(self.y - (i - 3) * math.sin(self.a) * 2 * self.edad)
            tail_osc = math.sin((i/5) - 2 * math.pi * self.flapping_phase) * 4
            x_tail += int(math.cos(self.a + math.pi/2) * tail_osc)
            y_tail += int(math.sin(self.a + math.pi/2) * tail_osc)
            radius_tail = max(int(-self.edad * 0.4 * (self.numeroelementos - i) * 2 + 1), 1)
            paint_circle(x_tail, y_tail, radius_tail, self.color_alive())

    def apply_wall_avoidance(self):
        push_x = 0.0
        push_y = 0.0
        if self.x < WALL_MARGIN:
            k = 1.0 - (self.x / WALL_MARGIN); push_x += WALL_PUSH_MAX * k
        if (width - self.x) < WALL_MARGIN:
            k = 1.0 - ((width - self.x) / WALL_MARGIN); push_x -= WALL_PUSH_MAX * k
        if self.y < WALL_MARGIN:
            k = 1.0 - (self.y / WALL_MARGIN); push_y += WALL_PUSH_MAX * k
        if (height - self.y) < WALL_MARGIN:
            k = 1.0 - ((height - self.y) / WALL_MARGIN); push_y -= WALL_PUSH_MAX * k

        has_push = (abs(push_x) + abs(push_y)) > 0.0001
        if not has_push:
            return False, self.target_angle, 0.0
        angle = math.atan2(push_y, push_x)
        strength = math.hypot(push_x, push_y)
        return True, angle, strength

    def vive(self, img, mask, quadtree, frame_count):
        if random.random() < 0.002:
            self.max_turn_rate = clamp(self.max_turn_rate + (random.random()-0.5)*0.002, 0.002, 0.04)

        if self.energia > 0:
            self.tiempo += self.avancevida
            self.mueve(quadtree, frame_count)

        self.energia -= 0.00003 + (self.speed * 0.00001)

        self.edad += 0.00001
        if self.edad > 3.0:
            self.energia = 0

        if self.energia > 0:
            self.dibuja(img, mask)

    def mueve(self, quadtree, frame_count):
        self.is_avoiding_collision = False
        self.previous_positions.append((self.x, self.y))

        avoid_r = self.genome["perception_avoid"]
        nearby = []
        quadtree.query(Rectangle(self.x, self.y, avoid_r, avoid_r), nearby)
        nearby = [f for f in nearby if f is not self]

        avg_rep_x = 0.0
        avg_rep_y = 0.0
        nclose = 0
        for other in nearby:
            dist = math.hypot(self.x - other.x, self.y - other.y)
            if dist < avoid_r and dist > 0:
                avg_rep_x += (self.x - other.x) / dist
                avg_rep_y += (self.y - other.y) / dist
                nclose += 1

        if nclose > 0:
            avg_rep_x /= nclose
            avg_rep_y /= nclose
            self.target_angle = math.atan2(avg_rep_y, avg_rep_x)
            self.is_avoiding_collision = True

        if not self.is_avoiding_collision and not self.is_chasing_food and self.food_memory:
            while self.food_memory and (frame_count - self.food_memory[0][2]) > self.memory_ttl_frames:
                self.food_memory.popleft()
            if self.food_memory:
                fx, fy, _ = self.food_memory[-1]
                self.target_angle = angle_in_radians(self.x, self.y, fx, fy)

        if len(self.previous_positions) >= 6:
            total_move = 0.0
            for i in range(1, len(self.previous_positions)):
                x0,y0 = self.previous_positions[i-1]
                x1,y1 = self.previous_positions[i]
                total_move += math.hypot(x1-x0, y1-y0)
            if total_move < 1.0:
                self.stuck_counter += 1
                if self.stuck_counter > 6:
                    self.is_stuck = True
            else:
                self.stuck_counter = 0
                self.is_stuck = False

        if self.is_stuck:
            self.target_angle += (random.random()-0.5) * 1.5
            self.is_avoiding_collision = True

        wall_hit, wall_angle, _ = self.apply_wall_avoidance()
        if wall_hit:
            self.target_angle = wall_angle
            self.is_avoiding_collision = True
            if self.wall_escape_cooldown <= 0:
                self.wall_escape_cooldown = WALL_ESCAPE_FRAMES

        starving = (self.energia < 0.2)
        escaping = (self.wall_escape_cooldown > 0)
        if self.is_chasing_food or self.is_avoiding_collision or self.is_stuck or starving or escaping:
            flapping_frequency = self.base_flapping_frequency * (1.6 if not escaping else 1.8)
            max_thrust = self.base_max_thrust * (1.5 if not escaping else WALL_ESCAPE_BOOST)
        else:
            flapping_frequency = self.base_flapping_frequency
            max_thrust = self.base_max_thrust

        self.flapping_phase += flapping_frequency * self.avancevida
        thrust = max_thrust * max(math.sin(2*math.pi*self.flapping_phase), 0)
        drag = self.drag_coefficient * self.speed
        self.speed += thrust - drag
        self.speed = clamp(self.speed, 0, self.genome["max_speed"])

        angle_diff = angle_difference(self.a, self.target_angle)
        if abs(angle_diff) > self.max_turn_rate:
            self.a += self.max_turn_rate if angle_diff > 0 else -self.max_turn_rate
        else:
            self.a = self.target_angle
        self.a = (self.a + math.pi) % (2*math.pi) - math.pi

        dx = math.cos(self.a) * self.speed * self.edad * 5
        dy = math.sin(self.a) * self.speed * self.edad * 5
        self.x += dx
        self.y += dy
        self.distance_travelled += math.hypot(dx, dy)

        out = False
        if self.x < 0: self.x = 0; out = True
        elif self.x > width: self.x = width; out = True
        if self.y < 0: self.y = 0; out = True
        elif self.y > height: self.y = height; out = True

        if out:
            cx, cy = width * 0.5, height * 0.5
            to_center = angle_in_radians(self.x, self.y, cx, cy) + (random.random()-0.5)*0.4
            self.target_angle = to_center
            self.is_avoiding_collision = True
            self.wall_escape_cooldown = max(self.wall_escape_cooldown, WALL_ESCAPE_FRAMES // 2)

        if self.wall_escape_cooldown > 0:
            self.wall_escape_cooldown -= 1

    def eat(self, food_obj):
        gain = (food_obj.radio / 10.0) * (self.genome["energy_eff"] / 0.1)
        self.energia += gain
        self.food_eaten += 1

    def maybe_reproduce(self, fishes, frame_count, max_pop):
        if len(fishes) >= max_pop:
            return
        breed_threshold = 1.6
        breed_cost = 0.7
        if self.energia > breed_threshold:
            child = Pez(parent=self)
            child.birth_frame = frame_count
            self.energia -= breed_cost
            child.energia = min(child.energia + (breed_cost * 0.6), 2.2)
            fishes.append(child)

# =========================
# Initialization
# =========================
numeropeces = random.randint(40, 100)
peces = [Pez() for _ in range(numeropeces)]
comidas = [Comida() for _ in range(14)]
MAX_POP = 220

# =========================
# Main loop
# =========================
for frame_count in range(total_frames):
    # Layers and masks
    layer_back  = np.zeros((height, width, 3), dtype=np.uint8)
    layer_mid   = np.zeros((height, width, 3), dtype=np.uint8)
    layer_front = np.zeros((height, width, 3), dtype=np.uint8)
    mask_back   = np.zeros((height, width), dtype=np.uint8)
    mask_mid    = np.zeros((height, width), dtype=np.uint8)
    mask_front  = np.zeros((height, width), dtype=np.uint8)

    LAYERS = [layer_back, layer_mid, layer_front]
    MASKS  = [mask_back, mask_mid, mask_front]

    # Random food spawn
    if random.random() < 0.00002 * max(20, len(peces)):
        comidas.append(Comida())

    # Evolve food (draw on its depth layer with mask)
    for comida in comidas:
        comida.vive(LAYERS[comida.layer], MASKS[comida.layer])

    # Quadtrees
    boundary = Rectangle(width/2, height/2, width/2, height/2)
    fish_qt = Quadtree(boundary, capacity=6)
    for f in peces:
        fish_qt.insert(f)

    food_qt = Quadtree(boundary, capacity=6)
    for c in comidas:
        food_qt.insert(c)

    # Interactions
    for pez in peces:
        pez.is_chasing_food = False
        pr = pez.genome["perception_food"]
        seen_food = []
        food_qt.query(Rectangle(pez.x, pez.y, pr, pr), seen_food)
        seen_food = [c for c in seen_food if c.visible]

        if seen_food:
            closest = min(seen_food, key=lambda c: math.hypot(pez.x - c.x, pez.y - c.y))
            pez.target_angle = angle_in_radians(pez.x, pez.y, closest.x, closest.y)
            pez.is_chasing_food = True
            pez.food_memory.append((closest.x, closest.y, frame_count))
            if math.hypot(pez.x - closest.x, pez.y - closest.y) < 10:
                closest.visible = False
                pez.eat(closest)

        pez.maybe_reproduce(peces, frame_count, MAX_POP)

    # Update fish (draw to layer & mask), remove dead
    for pez in peces[:]:
        pez.vive(LAYERS[pez.layer], MASKS[pez.layer], fish_qt, frame_count)
        if pez.energia <= 0:
            peces.remove(pez)

    # Clean up invisible food
    comidas = [c for c in comidas if c.visible]

    # ===== Compose with premultiplied alpha (no halos) =====
    frame = BG_GRADIENT.copy()

    # Back: strong blur
    back_pm, a_back = premult_blur(layer_back, mask_back, (31, 31))
    a_back *= 0.65  # global opacity tweak

    # Mid: light blur
    mid_pm, a_mid = premult_blur(layer_mid, mask_mid, (11, 11))
    a_mid *= 0.85

    # Front: no blur (kernel 1x1)
    front_pm, a_front = premult_blur(layer_front, mask_front, (1, 1))
    a_front *= 1.0

    # Alpha-over in depth order
    frame = alpha_over_pm(frame, back_pm,  a_back)
    frame = alpha_over_pm(frame, mid_pm,   a_mid)
    frame = alpha_over_pm(frame, front_pm, a_front)

    # Output
    video_writer.write(frame)
    cv2.imshow('Fish Simulation', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    if frame_count % fps == 0:
        if peces:
            top = sorted(peces, key=lambda z: (z.food_eaten, z.energia), reverse=True)[:3]
            top_txt = ", ".join([f"#{p.id}(gen{p.gen},food{p.food_eaten})" for p in top])
        else:
            top_txt = "—"
        print(f'Progress: {frame_count // fps}/{duration} sec | fish: {len(peces)} | top: {top_txt}')

video_writer.release()
cv2.destroyAllWindows()
print(f'Video saved as {video_path}')
