import cv2
import numpy as np
import random
import math
import os
import time
from collections import deque

# =========================
# Speed knobs (tune these)
# =========================
OUTPUT_W, OUTPUT_H = 1080, 1080   # 720p is MUCH faster; keep 1080x1080 if you must
FPS = 30
DURATION = 60*1
BLUR_EVERY_N = 1                 # 1=every frame; 2–3 caches blur N-1 frames
BACK_SCALE = 0.5                 # render+blur back layer at 50% size
MID_SCALE  = 0.75                # render+blur mid layer at 75% size
BACK_KSIZE = 31                  # blur kernel (odd)
MID_KSIZE  = 11

# =========================
# Collision / Physics knobs
# =========================
COLLISION_ITERATIONS = 1         # more -> stronger separation (cost ↑)
COLLISION_DAMPING = 0.8          # velocity damping after collision (0..1)
COLLISION_ESCAPE_FRAMES = int(FPS * 0.5)
COLLISION_BAUMGARTE = 0.15       # bias to eliminate resting penetration
SLIDE_KEEP = 1.0                 # keep 100% tangential velocity
NORMAL_REMOVE = 1.1              # remove >100% normal to avoid sticking
TANGENT_NUDGE = 0.25             # small forward nudge to keep flow
CAP_EXTRA_RADIUS = 1.5           # “skin” to preserve a visible gap

# --- Angular dynamics (smooth visible rotation) ---
MAX_ANGULAR_SPEED = 2.0 * math.pi / 1.2   # rad/s (≈300°/s). Lower = slower turns
MAX_ANGULAR_ACCEL = 2.5 * math.pi         # rad/s^2 cap on angular acceleration
COLLISION_ANG_IMPULSE = 1.2 * math.pi     # rad/s “bump” to start turning on collision

# =========================
# Sim params
# =========================
width, height = OUTPUT_W, OUTPUT_H
fps = FPS
duration = DURATION
total_frames = fps * duration

video_dir = 'videos'
os.makedirs(video_dir, exist_ok=True)
epoch_time = int(time.time())
video_path = os.path.join(video_dir, f'fish_simulation_{epoch_time}.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

cv2.setUseOptimized(True)
# cv2.setNumThreads(0)  # uncomment if you suspect thread oversubscription

# =========================
# Helpers
# =========================
def angle_difference(beta, alpha):
    difference = alpha - beta
    while difference > math.pi: difference -= 2 * math.pi
    while difference < -math.pi: difference += 2 * math.pi
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

def premult_blur_scaled(img, mask, ksize, scale):
    a = (mask.astype(np.float32) / 255.0)
    img_pm = img.astype(np.float32) * a[..., None]
    if scale != 1.0:
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        img_pm = cv2.resize(img_pm, (new_w, new_h), interpolation=cv2.INTER_AREA)
        a      = cv2.resize(a,      (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_pm_blur = cv2.GaussianBlur(img_pm, (ksize, ksize), 0)
    a_blur      = cv2.GaussianBlur(a,      (ksize, ksize), 0)
    a_blur = np.clip(a_blur, 0.0, 1.0)
    if scale != 1.0:
        img_pm_blur = cv2.resize(img_pm_blur, (width, height), interpolation=cv2.INTER_LINEAR)
        a_blur      = cv2.resize(a_blur,      (width, height), interpolation=cv2.INTER_LINEAR)
    return img_pm_blur, a_blur

def alpha_over_pm(dst_bgr_u8, src_pm_f32, alpha_f32):
    dst_f = dst_bgr_u8.astype(np.float32)
    inv_a = 1.0 - alpha_f32
    inv_a3 = cv2.merge([inv_a, inv_a, inv_a])
    dst_scaled = cv2.multiply(dst_f, inv_a3)
    out = cv2.add(src_pm_f32, dst_scaled)
    return np.clip(out, 0, 255).astype(np.uint8)

# Background gradient (BGR)
BG_GRADIENT = make_vertical_gradient(
    height, width,
    top_color=(245, 230, 210),
    bottom_color=(90, 70, 50)
)

# =========================
# Spatial index
# =========================
class Rectangle:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
    def contains(self, entity):
        return (self.x - self.w <= entity.x <= self.x + self.w and
                self.y - self.h <= entity.y <= self.y + self.h)
    def intersects(self, r):
        return not (r.x - r.w > self.x + self.w or
                    r.x + r.w < self.x - self.w or
                    r.y - r.h > self.y + self.h or
                    r.y + r.h < self.y - self.h)

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
    def insert(self, e):
        if not self.boundary.contains(e): return False
        if len(self.entities) < self.capacity:
            self.entities.append(e); return True
        if not self.divided: self.subdivide()
        return (self.northeast.insert(e) or self.northwest.insert(e) or
                self.southeast.insert(e) or self.southwest.insert(e))
    def query(self, r, found):
        if not self.boundary.intersects(r): return
        for e in self.entities:
            if r.contains(e): found.append(e)
        if self.divided:
            self.northwest.query(r, found); self.northeast.query(r, found)
            self.southwest.query(r, found); self.southeast.query(r, found)

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
        if not self.visible: return
        c = (int(self.x), int(self.y)); r = max(int(self.radio), 1)
        cv2.circle(img,  c, r, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(mask, c, r, 255, -1, cv2.LINE_AA)
    def vive(self, img, mask):
        if random.random() < 0.1: self.a += (random.random() - 0.5) * 0.2
        self.x += math.cos(self.a) * self.v
        self.y += math.sin(self.a) * self.v
        if self.x < 0: self.x = 0; self.a = -self.a
        elif self.x > width: self.x = width; self.a = -self.a
        if self.y < 0: self.y = 0; self.a = math.pi - self.a
        elif self.y > height: self.y = height; self.a = math.pi - self.a
        self.vida += 1
        if self.vida % fps == 0 and self.radio >= 2: self.divide()
        if self.radio < 1: self.visible = False
        self.dibuja(img, mask)
    def divide(self):
        angle_offset = math.pi
        child_r = self.radio / 1.4
        if child_r >= 1:
            comidas.extend([
                Comida(self.x, self.y, child_r, self.a, self.v, layer=self.layer),
                Comida(self.x, self.y, child_r, (self.a + angle_offset) % (2*math.pi), self.v, layer=self.layer),
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
    global FISH_ID_SEQ; FISH_ID_SEQ += 1; return FISH_ID_SEQ

def mutate(val, pct=0.05, minv=None, maxv=None):
    delta = val * pct * (random.random()*2 - 1)
    out = val + delta
    if minv is not None: out = max(minv, out)
    if maxv is not None: out = min(maxv, out)
    return out

class Pez:
    def __init__(self, parent=None):
        if parent is None:
            base_color = (random.randint(40,215), random.randint(40,215), random.randint(40,215))
            self.genome = {
                "base_color": base_color, "flapping_freq": random.uniform(0.5,1.0),
                "max_thrust": random.uniform(0.01,0.03), "drag": random.uniform(0.02,0.04),
                "max_speed": random.uniform(1.2,2.4), "max_turn": random.uniform(0.006,0.02),
                "perception_food": random.uniform(200,350), "perception_avoid": random.uniform(40,70),
                "energy_eff": random.uniform(0.08,0.14),
            }
            self.layer = random.choices([0,1,2], weights=[0.35,0.35,0.30])[0]
        else:
            p = parent.genome
            def mut_c(c): return int(clamp(mutate(c,0.08),0,255))
            base_color = tuple(mut_c(c) for c in p["base_color"])
            self.genome = {
                "base_color": base_color, "flapping_freq": mutate(p["flapping_freq"],0.08,0.3,1.4),
                "max_thrust": mutate(p["max_thrust"],0.08,0.006,0.05), "drag": mutate(p["drag"],0.08,0.01,0.07),
                "max_speed": mutate(p["max_speed"],0.08,0.8,3.0), "max_turn": mutate(p["max_turn"],0.08,0.004,0.03),
                "perception_food": mutate(p["perception_food"],0.08,120,420),
                "perception_avoid": mutate(p["perception_avoid"],0.08,30,100),
                "energy_eff": mutate(p["energy_eff"],0.08,0.05,0.2),
            }
            self.layer = parent.layer if random.random() > 0.1 else random.choice([0,1,2])

        self.numeroelementos = 10
        self.numeroelementoscola = 5
        c0 = self.genome["base_color"]
        self.colorr = [int(clamp(c0[0]+random.randint(-40,40),0,255)) for _ in range(-1,self.numeroelementos)]
        self.colorg = [int(clamp(c0[1]+random.randint(-40,40),0,255)) for _ in range(-1,self.numeroelementos)]
        self.colorb = [int(clamp(c0[2]+random.randint(-40,40),0,255)) for _ in range(-1,self.numeroelementos)]

        self.id = next_fish_id()
        self.gen = 0 if parent is None else getattr(parent,"gen",0)+1
        self.x = random.uniform(0,width) if parent is None else clamp(parent.x+random.uniform(-20,20),0,width)
        self.y = random.uniform(0,height) if parent is None else clamp(parent.y+random.uniform(-20,20),0,height)
        self.a = random.uniform(0,math.pi*2)
        self.target_angle = self.a
        self.edad = random.uniform(1.0,2.0)
        self.tiempo = random.uniform(0,1)
        self.avancevida = random.uniform(0.05,0.1)
        self.sexo = random.randint(0,1)
        self.energia = random.uniform(0.6,1.2) if parent is None else parent.energia*0.5
        self.speed = random.uniform(0.3,0.9)
        self.flapping_phase = 0.0
        self.base_flapping_frequency = self.genome["flapping_freq"]
        self.base_max_thrust = self.genome["max_thrust"]
        self.max_turn_rate = self.genome["max_turn"]  # kept but unused for visible rotation
        self.drag_coefficient = self.genome["drag"]
        self.is_chasing_food = False
        self.is_avoiding_collision = False
        self.food_memory = deque(maxlen=3)
        self.memory_ttl_frames = fps * 6
        self.previous_positions = deque(maxlen=6)
        self.stuck_counter = 0
        self.is_stuck = False
        self.wall_escape_cooldown = 0
        self.collision_escape_cooldown = 0
        self.food_eaten = 0
        self.distance_travelled = 0.0
        self.birth_frame = 0

        # kinematics
        self.vx = math.cos(self.a) * self.speed
        self.vy = math.sin(self.a) * self.speed
        self.ang_vel = 0.0  # rad/s

    # --- Capsule collider (segment + radius) ---
    def body_length(self):
        layer_scale = [0.90, 0.97, 1.05][self.layer]
        return (22.0 * self.edad + 10.0) * layer_scale
    def body_radius(self):
        layer_scale = [0.85, 0.93, 1.0][self.layer]
        return (5.5 * self.edad + 4.0) * layer_scale + CAP_EXTRA_RADIUS
    def capsule_endpoints(self):
        u = (math.cos(self.a), math.sin(self.a))
        L = self.body_length() * 0.5
        ax = self.x - u[0]*L; ay = self.y - u[1]*L
        bx = self.x + u[0]*L; by = self.y + u[1]*L
        r = self.body_radius()
        return (ax, ay), (bx, by), r

    # --- Rendering ---
    def color_alive(self):
        return self.genome["base_color"] if self.energia > 0 else (128,128,128)

    def dibuja(self, img, mask):
        cmain = self.color_alive()
        def pc(cx,cy,r,col):
            cv2.circle(img,(cx,cy),r,col,-1,cv2.LINE_AA)
            cv2.circle(mask,(cx,cy),r,255,-1,cv2.LINE_AA)
        def pe(center,axes,angle,col):
            cv2.ellipse(img,center,axes,angle,0,360,col,-1,cv2.LINE_AA)
            cv2.ellipse(mask,center,axes,angle,0,360,255,-1,cv2.LINE_AA)

        mouth_r = max(int(math.sin(2*math.pi*self.flapping_phase*2)*2+3),1)
        xm = int(self.x + math.cos(self.a)*5*self.edad)
        ym = int(self.y + math.sin(self.a)*5*self.edad)
        mo = math.sin(2*math.pi*self.flapping_phase)*2
        xm += int(math.cos(self.a+math.pi/2)*mo)
        ym += int(math.sin(self.a+math.pi/2)*mo)
        pc(xm,ym,mouth_r,cmain)

        for i in range(-1,self.numeroelementos):
            if i==1:
                for s in (-1,1):
                    xe = int(self.x + s*math.cos(self.a+math.pi/2)*4*self.edad - i*math.cos(self.a)*self.edad)
                    ye = int(self.y + s*math.sin(self.a+math.pi/2)*4*self.edad - i*math.sin(self.a)*self.edad)
                    eo = math.sin((i/5) - 2*math.pi*self.flapping_phase)*4
                    xe += int(math.cos(self.a+math.pi/2)*eo); ye += int(math.sin(self.a+math.pi/2)*eo)
                    re = max(int((self.edad*0.4*(self.numeroelementos-i)+1)/3),1)
                    pc(xe,ye,re,(255,255,255))

        for i in range(-1,self.numeroelementos):
            if i == self.numeroelementos//2 or i == int(self.numeroelementos/1.1):
                for s in (-1,1):
                    xf = int(self.x + s*math.cos(self.a+math.pi/2)*0.3*self.edad - i*math.cos(self.a)*self.edad)
                    yf = int(self.y + s*math.sin(self.a+math.pi/2)*0.3*self.edad - i*math.sin(self.a)*self.edad)
                    fo = math.sin((i/5) - 2*math.pi*self.flapping_phase)*4
                    xf += int(math.cos(self.a+math.pi/2)*fo); yf += int(math.sin(self.a+math.pi/2)*fo)
                    axes = (max(int((self.edad*0.4*(self.numeroelementos-i)+1)*2),1),
                            max(int((self.edad*0.4*(self.numeroelementos-i)+1)),1))
                    ang = math.degrees(self.a+math.pi/2 - math.cos(2*math.pi*self.flapping_phase*2)*s)
                    pe((xf,yf),axes,ang,cmain)

        for i in range(-1,self.numeroelementos):
            xb = int(self.x - i*math.cos(self.a)*2*self.edad)
            yb = int(self.y - i*math.sin(self.a)*2*self.edad)
            bo = math.sin((i/5) - 2*math.pi*self.flapping_phase)*4
            xb += int(math.cos(self.a+math.pi/2)*bo); yb += int(math.sin(self.a+math.pi/2)*bo)
            rb = max(int((self.edad*0.4*(self.numeroelementos-i)+1)),1)
            col = (self.colorr[i], self.colorg[i], self.colorb[i])
            pc(xb,yb,rb,col)

        for i in range(self.numeroelementos, self.numeroelementos + self.numeroelementoscola):
            xt = int(self.x - (i-3)*math.cos(self.a)*2*self.edad)
            yt = int(self.y - (i-3)*math.sin(self.a)*2*self.edad)
            to = math.sin((i/5) - 2*math.pi*self.flapping_phase)*4
            xt += int(math.cos(self.a+math.pi/2)*to); yt += int(math.sin(self.a+math.pi/2)*to)
            rt = max(int(-self.edad*0.4*(self.numeroelementos-i)*2 + 1),1)
            pc(xt,yt,rt,self.color_alive())

    # --- Steering / movement / energy ---
    def apply_wall_avoidance(self):
        px, py = 0.0, 0.0
        if self.x < WALL_MARGIN: px += WALL_PUSH_MAX * (1.0 - (self.x / WALL_MARGIN))
        if (width - self.x) < WALL_MARGIN: px -= WALL_PUSH_MAX * (1.0 - ((width - self.x)/WALL_MARGIN))
        if self.y < WALL_MARGIN: py += WALL_PUSH_MAX * (1.0 - (self.y / WALL_MARGIN))
        if (height - self.y) < WALL_MARGIN: py -= WALL_PUSH_MAX * (1.0 - ((height - self.y)/WALL_MARGIN))
        if abs(px)+abs(py) <= 1e-4: return False, self.target_angle, 0.0
        return True, math.atan2(py,px), math.hypot(px,py)

    def mueve(self, quadtree, frame_count):
        self.is_avoiding_collision = False
        self.previous_positions.append((self.x, self.y))

        avoid_r = self.genome["perception_avoid"]
        nearby = []
        quadtree.query(Rectangle(self.x, self.y, avoid_r, avoid_r), nearby)
        nearby = [f for f in nearby if f is not self and f.layer == self.layer]

        arx = ary = 0.0; ncl = 0
        for o in nearby:
            d = math.hypot(self.x - o.x, self.y - o.y)
            if d < avoid_r and d > 0:
                arx += (self.x - o.x) / d; ary += (self.y - o.y) / d; ncl += 1
        if ncl > 0:
            arx /= ncl; ary /= ncl
            self.target_angle = math.atan2(ary, arx)
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
                x0,y0 = self.previous_positions[i-1]; x1,y1 = self.previous_positions[i]
                total_move += math.hypot(x1-x0, y1-y0)
            if total_move < 1.0:
                self.stuck_counter += 1
                if self.stuck_counter > 6: self.is_stuck = True
            else:
                self.stuck_counter = 0; self.is_stuck = False
        if self.is_stuck:
            self.target_angle += (random.random()-0.5) * 1.5
            self.is_avoiding_collision = True

        wall_hit, wall_angle, _ = self.apply_wall_avoidance()
        if wall_hit:
            self.target_angle = wall_angle
            self.is_avoiding_collision = True
            if self.wall_escape_cooldown <= 0: self.wall_escape_cooldown = WALL_ESCAPE_FRAMES

        starving = (self.energia < 0.2)
        escaping = (self.wall_escape_cooldown > 0) or (self.collision_escape_cooldown > 0)
        if self.is_chasing_food or self.is_avoiding_collision or self.is_stuck or starving or escaping:
            flapping_frequency = self.base_flapping_frequency * (1.6 if not escaping else 1.8)
            max_thrust = self.base_max_thrust * (1.5 if not escaping else WALL_ESCAPE_BOOST)
        else:
            flapping_frequency = self.base_flapping_frequency; max_thrust = self.base_max_thrust

        self.flapping_phase += flapping_frequency * self.avancevida
        thrust = max_thrust * max(math.sin(2*math.pi*self.flapping_phase), 0)
        drag = self.drag_coefficient * abs(self.speed)
        self.speed += thrust - drag
        self.speed = clamp(self.speed, 0, self.genome["max_speed"])

        # --- Smooth turn with angular inertia ---
        ad = angle_difference(self.a, self.target_angle)  # [-pi,pi]
        escape_scale = 1.0
        if self.wall_escape_cooldown > 0 or self.collision_escape_cooldown > 0:
            escape_scale = 1.4  # turn a bit faster while escaping (still smooth)
        desired_omega = clamp(ad * 2.0 * math.pi * escape_scale,
                              -MAX_ANGULAR_SPEED*escape_scale, MAX_ANGULAR_SPEED*escape_scale)  # rad/s
        delta_omega = desired_omega - self.ang_vel
        max_delta = MAX_ANGULAR_ACCEL / fps
        delta_omega = clamp(delta_omega, -max_delta, max_delta)
        self.ang_vel += delta_omega
        self.ang_vel = clamp(self.ang_vel, -MAX_ANGULAR_SPEED*escape_scale, MAX_ANGULAR_SPEED*escape_scale)
        self.a += self.ang_vel / fps
        self.a = (self.a + math.pi) % (2*math.pi) - math.pi

        # integrate linear motion
        dx = math.cos(self.a) * self.speed * self.edad * 5
        dy = math.sin(self.a) * self.speed * self.edad * 5
        self.x += dx; self.y += dy
        self.vx = dx; self.vy = dy
        self.distance_travelled += math.hypot(dx, dy)

        out = False
        if self.x < 0: self.x = 0; out = True
        elif self.x > width: self.x = width; out = True
        if self.y < 0: self.y = 0; out = True
        elif self.y > height: self.y = height; out = True
        if out:
            cx, cy = width*0.5, height*0.5
            self.target_angle = angle_in_radians(self.x, self.y, cx, cy) + (random.random()-0.5)*0.4
            self.is_avoiding_collision = True
            self.wall_escape_cooldown = max(self.wall_escape_cooldown, WALL_ESCAPE_FRAMES//2)

        if self.wall_escape_cooldown > 0: self.wall_escape_cooldown -= 1
        if self.collision_escape_cooldown > 0: self.collision_escape_cooldown -= 1

    def eat(self, food_obj):
        self.energia += (food_obj.radio / 10.0) * (self.genome["energy_eff"] / 0.1)
        self.food_eaten += 1

    def maybe_reproduce(self, fishes, frame_count, max_pop):
        if len(fishes) >= max_pop: return
        if self.energia > 1.6:
            child = Pez(parent=self); child.birth_frame = frame_count
            self.energia -= 0.7
            child.energia = min(child.energia + 0.42, 2.2)
            fishes.append(child)

    def step(self, quadtree, frame_count):
        if random.random() < 0.002:
            self.max_turn_rate = clamp(self.max_turn_rate + (random.random()-0.5)*0.002, 0.002, 0.04)
        if self.energia > 0:
            self.tiempo += self.avancevida
            self.mueve(quadtree, frame_count)
        self.energia -= 0.00003 + (abs(self.speed) * 0.00001)
        self.edad += 0.00001
        if self.edad > 3.0: self.energia = 0

    def render(self, img, mask):
        if self.energia > 0:
            self.dibuja(img, mask)

# =========================
# Initialization
# =========================
numeropeces = random.randint(40, 80)
peces = [Pez() for _ in range(numeropeces)]
comidas = [Comida() for _ in range(14)]
MAX_POP = 220

# =========================
# Preallocate layers/masks and reuse
# =========================
layer_back  = np.zeros((height, width, 3), dtype=np.uint8)
layer_mid   = np.zeros((height, width, 3), dtype=np.uint8)
layer_front = np.zeros((height, width, 3), dtype=np.uint8)
mask_back   = np.zeros((height, width), dtype=np.uint8)
mask_mid    = np.zeros((height, width), dtype=np.uint8)
mask_front  = np.zeros((height, width), dtype=np.uint8)

# Caches for "blur every N"
cache_back_pm = None; cache_a_back = None
cache_mid_pm  = None; cache_a_mid  = None

# =========================
# Geometry: closest points between segments
# =========================
def closest_points_segments(ax,ay,bx,by, cx,cy,dx,dy):
    """
    Returns (px,py,qx,qy, t,u, dist, nx,ny)
    p=A+t*(B-A) on AB, q=C+u*(D-C) on CD
    nx,ny is (q - p) normalized (if dist>0), otherwise a symmetric fallback normal
    """
    abx, aby = (bx-ax), (by-ay)
    cdx, cdy = (dx-cx), (dy-cy)
    rpx, rpy = (ax-cx), (ay-cy)

    ab2 = abx*abx + aby*aby
    cd2 = cdx*cdx + cdy*cdy
    eps = 1e-8
    t = 0.0; u = 0.0

    if ab2 < eps and cd2 < eps:
        px, py = ax, ay
        qx, qy = cx, cy
    elif ab2 < eps:
        u = ((ax-cx)*cdx + (ay-cy)*cdy) / (cd2 + eps)
        u = clamp(u, 0.0, 1.0)
        qx, qy = (cx + cdx*u, cy + cdy*u)
        px, py = ax, ay
    elif cd2 < eps:
        t = ((cx-ax)*abx + (cy-ay)*aby) / (ab2 + eps)
        t = clamp(t, 0.0, 1.0)
        px, py = (ax + abx*t, ay + aby*t)
        qx, qy = cx, cy
    else:
        A = ab2
        B = abx*cdx + aby*cdy
        C = cd2
        D = abx*rpx + aby*rpy
        E = cdx*rpx + cdy*rpy
        denom = (A*C - B*B)
        if abs(denom) > eps:
            t = clamp((B*E - C*D) / denom, 0.0, 1.0)
        else:
            t = 0.0
        u = (B*t + E) / (C + eps)
        if u < 0.0:
            u = 0.0
            t = clamp(-D / (A + eps), 0.0, 1.0)
        elif u > 1.0:
            u = 1.0
            t = clamp((B - D) / (A + eps), 0.0, 1.0)
        px, py = (ax + abx*t, ay + aby*t)
        qx, qy = (cx + cdx*u, cy + cdy*u)

    dxn, dyn = (qx - px), (qy - py)
    dist2 = dxn*dxn + dyn*dyn
    dist = math.sqrt(dist2)
    if dist > eps:
        nx, ny = (dxn/dist, dyn/dist)
    else:
        # ===== Unbiased fallback normal (fixes CCW spin bias) =====
        # 1) Try center-to-center vector
        mx1, my1 = (ax + bx)*0.5, (ay + by)*0.5
        mx2, my2 = (cx + dx)*0.5, (cy + dy)*0.5
        cxn, cyn = (mx1 - mx2), (my1 - my2)
        clen = math.hypot(cxn, cyn)
        if clen > eps:
            nx, ny = cxn/clen, cyn/clen
        else:
            # 2) Try relative direction of the segments (as a proxy for motion)
            rvx, rvy = (abx if ab2 > eps else 0.0) - (cdx if cd2 > eps else 0.0), \
                       (aby if ab2 > eps else 0.0) - (cdy if cd2 > eps else 0.0)
            rlen = math.hypot(rvx, rvy)
            if rlen > eps:
                nx, ny = rvx/rlen, rvy/rlen
            else:
                # 3) Final tie-breaker: perpendicular with sign from cross product to avoid fixed CCW
                cross = (cx-ax)*aby - (cy-ay)*abx
                s = 1.0 if cross >= 0 else -1.0
                nx, ny = (-aby*s, abx*s)
                nlen = math.hypot(nx, ny) + eps
                nx, ny = nx/nlen, ny/nlen
    return px,py,qx,qy, t,u, dist, nx,ny

# =========================
# Collision solver (capsules + sliding + smooth rotation)
# =========================
def resolve_collisions_capsule(fishes, iterations=COLLISION_ITERATIONS):
    """
    Oriented capsule-capsule collision with sliding:
    - find closest points on segments
    - if distance < r_sum -> separate along normal (Baumgarte bias)
    - remove normal component of velocity; keep tangential (slip)
    - DO NOT snap heading; instead, nudge target + angular impulse
    """
    for _ in range(iterations):
        for layer in (0,1,2):
            boundary = Rectangle(width/2, height/2, width/2, height/2)
            qt = Quadtree(boundary, capacity=6)
            fs = [f for f in fishes if f.layer == layer and f.energia > 0]
            for f in fs: qt.insert(f)

            for f in fs:
                (ax,ay),(bx,by), rf = f.capsule_endpoints()
                search_r = f.body_length()*0.5 + f.body_radius()*2 + 8.0
                neighbors = []
                qt.query(Rectangle(f.x, f.y, search_r, search_r), neighbors)
                for o in neighbors:
                    if o is f or o.id <= f.id or o.layer != layer or o.energia <= 0:
                        continue
                    (cx,cy),(dx,dy), ro = o.capsule_endpoints()
                    px,py,qx,qy, t,u, dist, nx,ny = closest_points_segments(ax,ay,bx,by, cx,cy,dx,dy)
                    r_sum = rf + ro
                    penetration = r_sum - dist
                    if penetration > 0.0:
                        corr = (penetration * (0.5 + COLLISION_BAUMGARTE))
                        f.x -= nx * corr; f.y -= ny * corr
                        o.x += nx * corr; o.y += ny * corr
                        f.x = clamp(f.x, 0, width); f.y = clamp(f.y, 0, height)
                        o.x = clamp(o.x, 0, width); o.y = clamp(o.y, 0, height)

                        # --- Sliding velocities ---
                        vfx, vfy = f.vx, f.vy
                        vox, voy = o.vx, o.vy

                        vfn = (vfx*nx + vfy*ny)                    # scalar normal component
                        vftx, vfty = (vfx - vfn*nx, vfy - vfn*ny)   # tangential
                        vfx2 = vftx*SLIDE_KEEP - nx * max(0.0, vfn) * NORMAL_REMOVE
                        vfy2 = vfty*SLIDE_KEEP - ny * max(0.0, vfn) * NORMAL_REMOVE

                        von = (vox*nx + voy*ny)
                        votx, voty = (vox - von*nx, voy - von*ny)
                        vox2 = votx*SLIDE_KEEP + nx * max(0.0, -von) * NORMAL_REMOVE
                        voy2 = voty*SLIDE_KEEP + ny * max(0.0, -von) * NORMAL_REMOVE

                        # small forward nudge to keep flow
                        uf = (math.cos(f.a), math.sin(f.a))
                        uo = (math.cos(o.a), math.sin(o.a))
                        vfx2 += uf[0]*TANGENT_NUDGE; vfy2 += uf[1]*TANGENT_NUDGE
                        vox2 += uo[0]*TANGENT_NUDGE; voy2 += uo[1]*TANGENT_NUDGE

                        # damping
                        vfx2 *= COLLISION_DAMPING; vfy2 *= COLLISION_DAMPING
                        vox2 *= COLLISION_DAMPING; voy2 *= COLLISION_DAMPING

                        # write back velocity and speed (heading will rotate smoothly)
                        f.vx, f.vy = vfx2, vfy2
                        o.vx, o.vy = vox2, voy2

                        sf_f = max(1e-6, f.edad*5)
                        sf_o = max(1e-6, o.edad*5)
                        f.speed = clamp(math.hypot(vfx2, vfy2) / sf_f, 0.0, f.genome["max_speed"])
                        o.speed = clamp(math.hypot(vox2, voy2) / sf_o, 0.0, o.genome["max_speed"])

                        # steer targets away; do not snap orientation
                        away_f = math.atan2(f.y - o.y, f.x - o.x)
                        away_o = math.atan2(o.y - f.y, o.x - f.x)
                        f.target_angle = away_f
                        o.target_angle = away_o

                        # small angular impulses (capped) to start rotating but keep it visible
                        imp_f = clamp(angle_difference(f.a, away_f), -COLLISION_ANG_IMPULSE, COLLISION_ANG_IMPULSE)
                        imp_o = clamp(angle_difference(o.a, away_o), -COLLISION_ANG_IMPULSE, COLLISION_ANG_IMPULSE)
                        f.ang_vel += clamp(imp_f * fps * 0.15, -MAX_ANGULAR_ACCEL, MAX_ANGULAR_ACCEL) / fps
                        o.ang_vel += clamp(imp_o * fps * 0.15, -MAX_ANGULAR_ACCEL, MAX_ANGULAR_ACCEL) / fps

                        # escape boost (affects thrust/turn scaling in mover)
                        f.collision_escape_cooldown = max(f.collision_escape_cooldown, COLLISION_ESCAPE_FRAMES)
                        o.collision_escape_cooldown = max(o.collision_escape_cooldown, COLLISION_ESCAPE_FRAMES)

# =========================
# Main loop
# =========================
for frame_count in range(total_frames):
    # zero layers/masks
    layer_back.fill(0); layer_mid.fill(0); layer_front.fill(0)
    mask_back.fill(0);  mask_mid.fill(0);  mask_front.fill(0)
    LAYERS = [layer_back, layer_mid, layer_front]
    MASKS  = [mask_back, mask_mid, mask_front]

    # Random food spawn
    if random.random() < 0.0002 * max(20, len(peces)):
        comidas.append(Comida())

    # Evolve food
    for comida in comidas:
        comida.vive(LAYERS[comida.layer], MASKS[comida.layer])

    # Quadtrees for perception
    boundary = Rectangle(width/2, height/2, width/2, height/2)
    fish_qt = Quadtree(boundary, capacity=6)
    for f in peces: fish_qt.insert(f)
    food_qt = Quadtree(boundary, capacity=6)
    for c in comidas: food_qt.insert(c)

    # Interactions (food chase / eat / reproduce)
    for pez in peces:
        pez.is_chasing_food = False
        pr = pez.genome["perception_food"]
        seen_food = []
        food_qt.query(Rectangle(pez.x, pez.y, pr, pr), seen_food)
        seen_food = [c for c in seen_food if c.visible and c.layer == pez.layer]
        if seen_food:
            closest = min(seen_food, key=lambda c: math.hypot(pez.x - c.x, pez.y - c.y))
            pez.target_angle = angle_in_radians(pez.x, pez.y, closest.x, closest.y)
            pez.is_chasing_food = True
            pez.food_memory.append((closest.x, closest.y, frame_count))
            if math.hypot(pez.x - closest.x, pez.y - closest.y) < 10:
                closest.visible = False; pez.eat(closest)
        pez.maybe_reproduce(peces, frame_count, MAX_POP)

    # Move (no render yet)
    alive = []
    for pez in peces:
        pez.step(fish_qt, frame_count)
        if pez.energia > 0: alive.append(pez)
    peces = alive

    # Clean food
    comidas = [c for c in comidas if c.visible]

    # Capsule collisions + sliding + smooth rotation
    resolve_collisions_capsule(peces, iterations=COLLISION_ITERATIONS)

    # Render after collision resolution
    for pez in peces:
        pez.render(LAYERS[pez.layer], MASKS[pez.layer])

    # ===== Compose with premultiplied alpha =====
    frame = BG_GRADIENT.copy()

    if (frame_count % BLUR_EVERY_N == 0) or cache_back_pm is None:
        back_pm, a_back = premult_blur_scaled(layer_back, mask_back, BACK_KSIZE, BACK_SCALE)
        a_back *= 0.65
        cache_back_pm, cache_a_back = back_pm, a_back
    else:
        back_pm, a_back = cache_back_pm, cache_a_back

    if (frame_count % BLUR_EVERY_N == 0) or cache_mid_pm is None:
        mid_pm, a_mid = premult_blur_scaled(layer_mid, mask_mid, MID_KSIZE, MID_SCALE)
        a_mid *= 0.85
        cache_mid_pm, cache_a_mid = mid_pm, a_mid
    else:
        mid_pm, a_mid = cache_mid_pm, cache_a_mid

    front_pm, a_front = premult_blur_scaled(layer_front, mask_front, 1, 1.0)
    a_front *= 1.0

    frame = alpha_over_pm(frame, back_pm,  a_back)
    frame = alpha_over_pm(frame, mid_pm,   a_mid)
    frame = alpha_over_pm(frame, front_pm, a_front)

    # Output
    video_writer.write(frame)
    cv2.imshow('Fish Simulation', frame)
    if cv2.waitKey(1) == 27: break

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
