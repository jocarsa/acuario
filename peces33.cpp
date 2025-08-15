// peces33.cpp
// C++ OpenCV + OpenMP version of the evolutionary fish sim with survival memory,
// genetics, wall-avoidance/escape, reproduction, quadtree queries.
//
// Build:
//   g++ -O3 -fopenmp -std=c++17 peces33.cpp -o peces33 `pkg-config --cflags --libs opencv4`
//
// Run:
//   ./peces33
//
// Keys: ESC to quit.

#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <random>
#include <memory>
#include <atomic>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <system_error>

#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads(){ return 1; }
  inline int omp_get_thread_num(){ return 0; }
#endif

// =========================
// Video / sim settings
// =========================
static int WIDTH  = 1920;    // For testing try 1280
static int HEIGHT = 1080;    // For testing try 720
static int FPS    = 30;
static int DURATION_SEC = 60*60; // test with 60
static int TOTAL_FRAMES = FPS * DURATION_SEC;

// Wall avoidance
static float WALL_MARGIN        = 80.0f;
static float WALL_PUSH_MAX      = 1.8f;
static float WALL_ESCAPE_BOOST  = 1.8f;
static int   WALL_ESCAPE_FRAMES = 24; // ~0.8s at 30fps

// Population
static int MAX_POP = 220;

// RNG (thread-local)
static thread_local std::mt19937 rng{ std::random_device{}() };

inline float randf(float a, float b){
    std::uniform_real_distribution<float> dist(a,b);
    return dist(rng);
}
inline int randi(int a, int b){ // inclusive
    std::uniform_int_distribution<int> dist(a,b);
    return dist(rng);
}
inline bool randchance(float p){
    std::bernoulli_distribution d(p);
    return d(rng);
}

// Helpers
inline float clampf(float v, float a, float b){ return std::max(a, std::min(b, v)); }
inline float angle_diff(float beta, float alpha){
    float d = alpha - beta;
    while(d >  M_PI) d -= 2.0f * M_PI;
    while(d < -M_PI) d += 2.0f * M_PI;
    return d;
}
inline float angle_to(float x1, float y1, float x2, float y2){
    return std::atan2(y2-y1, x2-x1);
}

// =========================
/* Spatial index (Quadtree) */
// =========================
template<typename T>
struct Rect {
    float x, y, w, h; // center + half extents
    bool contains(const T* e) const {
        return (x - w <= e->x && e->x <= x + w &&
                y - h <= e->y && e->y <= y + h);
    }
    bool intersects(const Rect& r) const {
        return !(r.x - r.w > x + w || r.x + r.w < x - w ||
                 r.y - r.h > y + h || r.y + r.h < y - h);
    }
};

template<typename T>
struct Quadtree {
    Rect<T> boundary;
    int capacity;
    bool divided = false;
    std::vector<T*> entities;
    std::unique_ptr<Quadtree> ne, nw, se, sw;

    Quadtree(const Rect<T>& b, int cap) : boundary(b), capacity(cap) {
        entities.reserve(capacity);
    }

    void subdivide(){
        float x = boundary.x, y = boundary.y;
        float w = boundary.w/2.0f, h = boundary.h/2.0f;
        ne = std::make_unique<Quadtree>(Rect<T>{x + w, y - h, w, h}, capacity);
        nw = std::make_unique<Quadtree>(Rect<T>{x - w, y - h, w, h}, capacity);
        se = std::make_unique<Quadtree>(Rect<T>{x + w, y + h, w, h}, capacity);
        sw = std::make_unique<Quadtree>(Rect<T>{x - w, y + h, w, h}, capacity);
        divided = true;
    }

    bool insert(T* e){
        if(!boundary.contains(e)) return false;
        if((int)entities.size() < capacity){
            entities.push_back(e);
            return true;
        }
        if(!divided) subdivide();
        return ne->insert(e) || nw->insert(e) || se->insert(e) || sw->insert(e);
    }

    void query(const Rect<T>& range, std::vector<T*>& found) const {
        if(!boundary.intersects(range)) return;
        for(auto* e : entities){
            if(range.contains(e)) found.push_back(e);
        }
        if(divided){
            nw->query(range, found);
            ne->query(range, found);
            sw->query(range, found);
            se->query(range, found);
        }
    }
};

// =========================
/* Food */
// =========================
struct Comida {
    float x, y;
    float radio;
    float a;   // direction
    float v;   // speed
    std::atomic<bool> visible;
    int vida;

    Comida(float X, float Y, float R, float A, float V)
        : x(X), y(Y), radio(R), a(A), v(V), visible(true), vida(0) {}

    Comida() : x(randf(0, WIDTH)), y(randf(0, HEIGHT)),
               radio(randf(5,15)), a(randf(0, 2*M_PI)),
               v(randf(0,0.25f)), visible(true), vida(0) {}

    // ---- Copy / Move to allow std::vector operations with atomic<bool> ----
    Comida(const Comida& o)
        : x(o.x), y(o.y), radio(o.radio), a(o.a), v(o.v),
          visible(o.visible.load()), vida(o.vida) {}

    Comida& operator=(const Comida& o){
        if(this != &o){
            x = o.x; y = o.y; radio = o.radio; a = o.a; v = o.v;
            visible.store(o.visible.load());
            vida = o.vida;
        }
        return *this;
    }

    Comida(Comida&& o) noexcept
        : x(o.x), y(o.y), radio(o.radio), a(o.a), v(o.v),
          visible(o.visible.load()), vida(o.vida) {}

    Comida& operator=(Comida&& o) noexcept{
        if(this != &o){
            x = o.x; y = o.y; radio = o.radio; a = o.a; v = o.v;
            visible.store(o.visible.load());
            vida = o.vida;
        }
        return *this;
    }
    // ----------------------------------------------------------------------

    void draw(cv::Mat& frame) const {
        if(!visible.load()) return;
        int rr = std::max(1, (int)std::round(radio));
        cv::circle(frame, cv::Point((int)x,(int)y), rr, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_AA);
    }

    // Serial update (avoids race with division/spawn)
    void step(){
        if(randchance(0.1f)) a += (randf(0,1)-0.5f)*0.2f;
        x += std::cos(a)*v; y += std::sin(a)*v;
        // Reflect at walls
        if(x < 0){ x = 0; a = -a; }
        else if(x > WIDTH){ x = (float)WIDTH; a = -a; }
        if(y < 0){ y = 0; a = (float)M_PI - a; }
        else if(y > HEIGHT){ y = (float)HEIGHT; a = (float)M_PI - a; }

        vida++;
    }
};

// =========================
/* Fish (memory + genetics + walls) */
// =========================
struct Genome {
    cv::Vec3b base_color;   // B,G,R
    float flapping_freq;
    float max_thrust;
    float drag;
    float max_speed;
    float max_turn;
    float perception_food;
    float perception_avoid;
    float energy_eff;       // energy gain scale
};

static int FISH_ID_SEQ = 0;
inline int next_fish_id(){ return ++FISH_ID_SEQ; }

inline float mutate(float v, float pct, float lo, float hi){
    float delta = v * pct * (randf(-1.0f, 1.0f));
    float out = v + delta;
    return clampf(out, lo, hi);
}
inline cv::Vec3b mutate_color(const cv::Vec3b& c, float pct=0.08f){
    auto mut = [&](int ch)->uchar{
        float v = (float)ch + (float)ch * pct * (randf(-1.0f, 1.0f));
        return (uchar)clampf(v, 0.f, 255.f);
    };
    return cv::Vec3b(mut(c[0]), mut(c[1]), mut(c[2]));
}

struct FoodMemory { float x,y; int frame; };

struct Pez {
    // Position & motion
    float x, y;
    float a;             // current angle
    float target_angle;
    float edad, tiempo, avancevida;
    int   sexo;

    // Energy & gait
    float energia;
    float speed;
    float flapping_phase;
    float base_flapping_frequency;
    float base_max_thrust;
    float max_turn_rate;
    float drag_coefficient;

    // Visual body
    int numeroelementos = 10;
    int numeroelementoscola = 5;
    std::vector<int> colorr, colorg, colorb;

    // Genome / inheritance
    Genome g;
    int id;
    int gen;

    // Flags
    bool is_chasing_food = false;
    bool is_avoiding_collision = false;
    bool is_stuck = false;

    // Memory
    std::deque<FoodMemory> food_memory; // maxlen 3
    int memory_ttl_frames = FPS * 6;

    // Stuck detection
    std::deque<cv::Point2f> prev_pos;
    int stuck_counter = 0;

    // Wall escape
    int wall_escape_cooldown = 0;

    // Fitness
    int   food_eaten = 0;
    float distance_travelled = 0.f;
    int   birth_frame = 0;

    Pez(){ // founder
        id = next_fish_id();
        gen = 0;

        // Genome init
        g.base_color       = cv::Vec3b((uchar)randi(40,215),(uchar)randi(40,215),(uchar)randi(40,215));
        g.flapping_freq    = randf(0.5f, 1.0f);
        g.max_thrust       = randf(0.01f, 0.03f);
        g.drag             = randf(0.02f, 0.04f);
        g.max_speed        = randf(1.2f, 2.4f);
        g.max_turn         = randf(0.006f,0.02f);
        g.perception_food  = randf(200,350);
        g.perception_avoid = randf(40, 70);
        g.energy_eff       = randf(0.08f,0.14f);

        init_common(nullptr);
    }

    Pez(const Pez& parent){ // child
        id = next_fish_id();
        gen = parent.gen + 1;

        g.base_color       = mutate_color(parent.g.base_color);
        g.flapping_freq    = mutate(parent.g.flapping_freq, 0.08f, 0.3f, 1.4f);
        g.max_thrust       = mutate(parent.g.max_thrust,    0.08f, 0.006f, 0.05f);
        g.drag             = mutate(parent.g.drag,          0.08f, 0.01f,  0.07f);
        g.max_speed        = mutate(parent.g.max_speed,     0.08f, 0.8f,   3.0f);
        g.max_turn         = mutate(parent.g.max_turn,      0.08f, 0.004f, 0.03f);
        g.perception_food  = mutate(parent.g.perception_food, 0.08f, 120, 420);
        g.perception_avoid = mutate(parent.g.perception_avoid,0.08f, 30,  100);
        g.energy_eff       = mutate(parent.g.energy_eff,    0.08f, 0.05f,  0.2f);

        init_common(&parent);
    }

    void init_common(const Pez* p){
        if(p){
            x = clampf(p->x + randf(-20,20), 0, (float)WIDTH);
            y = clampf(p->y + randf(-20,20), 0, (float)HEIGHT);
        }else{
            x = randf(0, WIDTH);
            y = randf(0, HEIGHT);
        }
        a = randf(0, 2*M_PI);
        target_angle = a;
        edad = randf(1.0f, 2.0f);
        tiempo = randf(0,1);
        avancevida = randf(0.05f, 0.1f);
        sexo = randi(0,1);
        energia = p ? p->energia * 0.5f : randf(0.6f, 1.2f);
        speed = randf(0.3f, 0.9f);
        flapping_phase = 0.f;
        base_flapping_frequency = g.flapping_freq;
        base_max_thrust = g.max_thrust;
        max_turn_rate = g.max_turn;
        drag_coefficient = g.drag;

        // Colors along body
        colorr.resize(numeroelementos+1);
        colorg.resize(numeroelementos+1);
        colorb.resize(numeroelementos+1);
        for(int i=0;i<=numeroelementos;i++){
            colorr[i] = (int)clampf(g.base_color[0] + randi(-40,40), 0,255);
            colorg[i] = (int)clampf(g.base_color[1] + randi(-40,40), 0,255);
            colorb[i] = (int)clampf(g.base_color[2] + randi(-40,40), 0,255);
        }

        food_memory.clear();
        prev_pos.clear();
        is_chasing_food = is_avoiding_collision = is_stuck = false;
        wall_escape_cooldown = 0;
        food_eaten = 0;
        distance_travelled = 0.f;
        birth_frame = 0;
    }

    cv::Scalar color_alive() const {
        if(energia > 0) return cv::Scalar(g.base_color[0], g.base_color[1], g.base_color[2]); // BGR
        return cv::Scalar(128,128,128);
    }

    void draw(cv::Mat& frame) const {
        if(energia <= 0) return;
        cv::Scalar color_main = color_alive();

        // Mouth
        int mouth_radius = std::max(1, (int)std::round(std::sinf(2.0f*(float)M_PI*flapping_phase*2.0f)*2.0f + 3.0f));
        int x_mouth = (int)std::round(x + std::cos(a)*5*edad);
        int y_mouth = (int)std::round(y + std::sin(a)*5*edad);
        float mouth_osc = std::sinf(2.0f*(float)M_PI*flapping_phase)*2.0f;
        x_mouth += (int)std::round(std::cos(a + M_PI/2)*mouth_osc);
        y_mouth += (int)std::round(std::sin(a + M_PI/2)*mouth_osc);
        cv::circle(frame, {x_mouth, y_mouth}, mouth_radius, color_main, cv::FILLED, cv::LINE_AA);

        // Eyes
        for(int i=-1;i<numeroelementos;i++){
            if(i==1){
                for(int sign : {-1,1}){
                    int x_eye = (int)std::round(x + sign*std::cos(a+M_PI/2)*4*edad - i*std::cos(a)*edad);
                    int y_eye = (int)std::round(y + sign*std::sin(a+M_PI/2)*4*edad - i*std::sin(a)*edad);
                    float eye_osc = std::sinf((float)i/5.0f - 2.0f*(float)M_PI*flapping_phase) * 4.0f;
                    x_eye += (int)std::round(std::cos(a+M_PI/2)*eye_osc);
                    y_eye += (int)std::round(std::sin(a+M_PI/2)*eye_osc);
                    int r_eye = std::max(1, (int)std::round((edad*0.4f*(numeroelementos - i) + 1)/3.0f));
                    cv::circle(frame, {x_eye, y_eye}, r_eye, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_AA);
                }
            }
        }

        // Fins
        for(int i=-1;i<numeroelementos;i++){
            if(i==numeroelementos/2 || i==(int)(numeroelementos/1.1f)){
                for(int sign : {-1,1}){
                    int x_fin = (int)std::round(x + sign*std::cos(a+M_PI/2)*0.3f*edad - i*std::cos(a)*edad);
                    int y_fin = (int)std::round(y + sign*std::sin(a+M_PI/2)*0.3f*edad - i*std::sin(a)*edad);
                    float fin_osc = std::sinf((float)i/5.0f - 2.0f*(float)M_PI*flapping_phase) * 4.0f;
                    x_fin += (int)std::round(std::cos(a+M_PI/2)*fin_osc);
                    y_fin += (int)std::round(std::sin(a+M_PI/2)*fin_osc);
                    cv::Size axes(
                        std::max(1, (int)std::round((edad*0.4f*(numeroelementos - i) + 1)*2)),
                        std::max(1, (int)std::round( (edad*0.4f*(numeroelementos - i) + 1)))
                    );
                    double angle_deg = (a + M_PI/2 - std::cos(2*M_PI*flapping_phase*2)*sign) * 180.0/M_PI;
                    cv::ellipse(frame, {x_fin, y_fin}, axes, angle_deg, 0, 360, color_main, cv::FILLED, cv::LINE_AA);
                }
            }
        }

        // Body
        for(int i=-1;i<numeroelementos;i++){
            int x_body = (int)std::round(x - i*std::cos(a)*2*edad);
            int y_body = (int)std::round(y - i*std::sin(a)*2*edad);
            float osc = std::sinf((float)i/5.0f - 2.0f*(float)M_PI*flapping_phase) * 4.0f;
            x_body += (int)std::round(std::cos(a+M_PI/2)*osc);
            y_body += (int)std::round(std::sin(a+M_PI/2)*osc);
            int r_body = std::max(1, (int)std::round((edad*0.4f*(numeroelementos - i) + 1)));
            cv::Scalar color_body(colorr[i+1], colorg[i+1], colorb[i+1]);
            cv::circle(frame, {x_body, y_body}, r_body, color_body, cv::FILLED, cv::LINE_AA);
        }

        // Tail
        for(int i=numeroelementos; i<numeroelementos+numeroelementoscola; ++i){
            int x_tail = (int)std::round(x - (i-3)*std::cos(a)*2*edad);
            int y_tail = (int)std::round(y - (i-3)*std::sin(a)*2*edad);
            float osc = std::sinf((float)i/5.0f - 2.0f*(float)M_PI*flapping_phase) * 4.0f;
            x_tail += (int)std::round(std::cos(a+M_PI/2)*osc);
            y_tail += (int)std::round(std::sin(a+M_PI/2)*osc);
            int r_tail = std::max(1, (int)std::round(-edad*0.4f*(numeroelementos - i)*2 + 1));
            cv::circle(frame, {x_tail, y_tail}, r_tail, color_alive(), cv::FILLED, cv::LINE_AA);
        }
    }

    // Wall avoidance: returns (has_push, angle, strength)
    std::tuple<bool,float,float> apply_wall_avoidance() const {
        float px = 0.f, py = 0.f;

        if(x < WALL_MARGIN){
            float k = 1.f - (x / WALL_MARGIN);
            px += WALL_PUSH_MAX * k;
        }
        if((WIDTH - x) < WALL_MARGIN){
            float k = 1.f - ((WIDTH - x)/WALL_MARGIN);
            px -= WALL_PUSH_MAX * k;
        }
        if(y < WALL_MARGIN){
            float k = 1.f - (y / WALL_MARGIN);
            py += WALL_PUSH_MAX * k;
        }
        if((HEIGHT - y) < WALL_MARGIN){
            float k = 1.f - ((HEIGHT - y)/WALL_MARGIN);
            py -= WALL_PUSH_MAX * k;
        }
        float mag = std::hypot(px,py);
        if(mag < 1e-4f) return {false, target_angle, 0.f};
        float ang = std::atan2(py, px);
        return {true, ang, mag};
    }

    void eat(const Comida& food){
        float gain = (food.radio / 10.0f) * (g.energy_eff / 0.1f);
        energia += gain;
        food_eaten += 1;
    }

    // Physics & decisions not involving shared state mutation
    void step(const Quadtree<Pez>& fish_qt, int frame_count){
        // small random tuning drift
        if(randchance(0.002f)){
            max_turn_rate = clampf(max_turn_rate + (randf(0,1)-0.5f)*0.002f, 0.002f, 0.04f);
        }

        if(energia > 0){
            tiempo += avancevida;
            move_internal(fish_qt, frame_count);
        }

        // metabolism
        energia -= 0.00003f + (speed * 0.00001f);

        // aging
        edad += 0.00001f;
        if(edad > 3.0f) energia = 0.f; // natural death
    }

    void move_internal(const Quadtree<Pez>& fish_qt, int frame_count){
        is_avoiding_collision = false;

        // neighbor avoidance
        float avoid_r = g.perception_avoid;
        std::vector<Pez*> near;
        fish_qt.query(Rect<Pez>{x,y,avoid_r,avoid_r}, near);
        near.erase(std::remove(near.begin(), near.end(), this), near.end());

        float rx=0.f, ry=0.f; int nclose=0;
        for(auto* o : near){
            float dx = x - o->x, dy = y - o->y;
            float dist = std::hypot(dx,dy);
            if(dist < avoid_r && dist > 0.f){
                rx += dx / dist; ry += dy / dist; nclose++;
            }
        }
        if(nclose>0){
            rx /= nclose; ry /= nclose;
            target_angle = std::atan2(ry, rx);
            is_avoiding_collision = true;
        }

        // memory fallback if not avoiding & not chasing
        if(!is_avoiding_collision && !is_chasing_food && !food_memory.empty()){
            // purge stale
            while(!food_memory.empty() && (frame_count - food_memory.front().frame) > memory_ttl_frames){
                food_memory.pop_front();
            }
            if(!food_memory.empty()){
                const auto& m = food_memory.back();
                target_angle = angle_to(x,y,m.x,m.y);
            }
        }

        // stuck detection
        prev_pos.emplace_back(x,y);
        if(prev_pos.size() >= 6){
            float tot=0.f;
            for(size_t i=1;i<prev_pos.size();++i){
                tot += cv::norm(prev_pos[i]-prev_pos[i-1]);
            }
            if(tot < 1.0f){
                stuck_counter++;
                if(stuck_counter > 6) is_stuck = true;
            }else{
                stuck_counter = 0; is_stuck = false;
            }
            if(prev_pos.size()>6) prev_pos.pop_front();
        }
        if(is_stuck){
            target_angle += (randf(0,1)-0.5f)*1.5f;
            is_avoiding_collision = true;
        }

        // wall avoidance
        auto [w_hit, w_ang, w_str] = apply_wall_avoidance();
        if(w_hit){
            target_angle = w_ang;
            is_avoiding_collision = true;
            if(wall_escape_cooldown <= 0) wall_escape_cooldown = WALL_ESCAPE_FRAMES;
        }

        // energy-aware gait
        bool starving = (energia < 0.2f);
        bool escaping = (wall_escape_cooldown > 0);

        float flap = base_flapping_frequency * ( (escaping||is_chasing_food||is_avoiding_collision||is_stuck||starving) ? (escaping?1.8f:1.6f) : 1.0f );
        float thrust_max = base_max_thrust * ( (escaping)? WALL_ESCAPE_BOOST : ((is_chasing_food||is_avoiding_collision||is_stuck||starving)?1.5f:1.0f) );

        // update speed
        flapping_phase += flap * avancevida;
        float thrust = thrust_max * std::max(0.0f, std::sinf(2.0f * (float)M_PI * flapping_phase));
        float drag = drag_coefficient * speed;
        speed += thrust - drag;
        speed = clampf(speed, 0.f, g.max_speed);

        // turn toward target
        float ad = angle_diff(a, target_angle);
        if(std::fabs(ad) > max_turn_rate) a += (ad>0? max_turn_rate : -max_turn_rate);
        else a = target_angle;
        if(a >  M_PI) a -= 2*M_PI;
        if(a < -M_PI) a += 2*M_PI;

        // move
        float dx = std::cos(a) * speed * edad * 5.f;
        float dy = std::sin(a) * speed * edad * 5.f;
        x += dx; y += dy;
        distance_travelled += std::hypot(dx,dy);

        // bounds clamp + soft turn-in
        bool out = false;
        if(x < 0){ x = 0; out=true; }
        else if(x > WIDTH){ x = (float)WIDTH; out=true; }
        if(y < 0){ y = 0; out=true; }
        else if(y > HEIGHT){ y = (float)HEIGHT; out=true; }

        if(out){
            float cx = WIDTH*0.5f, cy = HEIGHT*0.5f;
            float to_center = angle_to(x,y,cx,cy) + (randf(0,1)-0.5f)*0.4f;
            target_angle = to_center;
            is_avoiding_collision = true;
            wall_escape_cooldown = std::max(wall_escape_cooldown, WALL_ESCAPE_FRAMES/2);
        }

        if(wall_escape_cooldown > 0) wall_escape_cooldown--;
    }
};

// =========================
// Main
// =========================
int main(){
    // Video writer
    std::string outDir = "videos";
    std::filesystem::create_directories(outDir);

    int64_t epoch = (int64_t)std::time(nullptr);
    std::string videoPath = outDir + "/fish_simulation_" + std::to_string(epoch) + ".mp4";

    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v'); // try 'a','v','c','1' if needed
    writer.open(videoPath, fourcc, FPS, cv::Size(WIDTH, HEIGHT));
    if(!writer.isOpened()){
        std::cerr << "Failed to open VideoWriter. Trying AVC1...\n";
        fourcc = cv::VideoWriter::fourcc('a','v','c','1');
        writer.open(videoPath, fourcc, FPS, cv::Size(WIDTH, HEIGHT));
        if(!writer.isOpened()){
            std::cerr << "Could not open video writer. Exiting.\n";
            return 1;
        }
    }

    // Init population and food
    int numeropeces = randi(40, 100);
    std::vector<Pez> peces; peces.reserve(MAX_POP*2);
    for(int i=0;i<numeropeces;i++) peces.emplace_back();

    std::vector<Comida> comidas; comidas.reserve(10000);
    for(int i=0;i<14;i++) comidas.emplace_back();

    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3);

    for(int frame_count=0; frame_count<TOTAL_FRAMES; ++frame_count){
        frame.setTo(cv::Scalar(0,0,0));

        // Spawn random food (scaled by pop)
        if(randf(0,1) < 0.00002f * std::max(20, (int)peces.size())){
            comidas.emplace_back();
        }

        // Update + draw food (serial)
        std::vector<Comida> new_foods; new_foods.reserve(32);
        for(auto& c : comidas){
            c.step();
            // divide every second if radius >=2
            if(c.vida % FPS == 0 && c.radio >= 2.f){
                float child_r = c.radio / 1.4f;
                if(child_r >= 1.f){
                    float ang = c.a;
                    new_foods.emplace_back(c.x, c.y, child_r, ang, c.v);
                    new_foods.emplace_back(c.x, c.y, child_r, std::fmod(ang + (float)M_PI, 2.0f*(float)M_PI), c.v);
                }
                c.visible.store(false);
            }
            if(c.radio < 1.f) c.visible.store(false);
            c.draw(frame);
        }
        if(!new_foods.empty()){
            comidas.insert(comidas.end(), new_foods.begin(), new_foods.end());
        }
        // prune invisible food
        comidas.erase(
            std::remove_if(comidas.begin(), comidas.end(), [](const Comida& c){ return !c.visible.load(); }),
            comidas.end()
        );

        // Build quadtrees
        Quadtree<Pez>  fish_qt (Rect<Pez>{ WIDTH/2.f, HEIGHT/2.f, WIDTH/2.f, HEIGHT/2.f }, 6);
        for(auto& f : peces) fish_qt.insert(&f);

        Quadtree<Comida> food_qt (Rect<Comida>{ WIDTH/2.f, HEIGHT/2.f, WIDTH/2.f, HEIGHT/2.f }, 6);
        for(auto& c : comidas) food_qt.insert(&c);

        // ---------- Perception & reproduction (parallel) ----------
        int nth = omp_get_max_threads();
        std::vector<std::vector<Pez>> births_tls(nth);

        #pragma omp parallel for schedule(static)
        for(int i=0; i<(int)peces.size(); ++i){
            auto& pez = peces[i];
            pez.is_chasing_food = false;

            // Perceive food
            float pr = pez.g.perception_food;
            std::vector<Comida*> seen_food;
            food_qt.query(Rect<Comida>{pez.x, pez.y, pr, pr}, seen_food);

            // filter visible
            seen_food.erase(std::remove_if(seen_food.begin(), seen_food.end(),
                [](Comida* c){ return !c->visible.load(); }), seen_food.end());

            if(!seen_food.empty()){
                Comida* closest = nullptr;
                float bestd = 1e9f;
                for(auto* c : seen_food){
                    float d = std::hypot(pez.x - c->x, pez.y - c->y);
                    if(d < bestd){ bestd = d; closest = c; }
                }
                if(closest){
                    pez.target_angle = angle_to(pez.x, pez.y, closest->x, closest->y);
                    pez.is_chasing_food = true;
                    // memory push
                    if(pez.food_memory.size() >= 3) pez.food_memory.pop_front();
                    pez.food_memory.push_back(FoodMemory{closest->x, closest->y, frame_count});

                    if(bestd < 10.f){
                        bool expected = true;
                        if(closest->visible.compare_exchange_strong(expected, false)){
                            pez.eat(*closest);
                        }
                    }
                }
            }

            // Reproduction (collect births thread-locally)
            float breed_threshold = 1.6f;
            float breed_cost = 0.7f;
            if((int)peces.size() < MAX_POP && pez.energia > breed_threshold){
                int tid = omp_get_thread_num();
                births_tls[tid].emplace_back(pez); // child from parent
                pez.energia -= breed_cost;
                // child initial energy handled when we merge
            }
        }

        // Merge births (serial)
        for(auto& bucket : births_tls){
            for(auto& child : bucket){
                child.birth_frame = frame_count;
                child.energia = std::min(child.energia + 0.7f*0.6f, 2.2f);
                peces.push_back(std::move(child));
                if((int)peces.size() >= MAX_POP) break;
            }
            if((int)peces.size() >= MAX_POP) break;
        }

        // ---------- Physics update (parallel) ----------
        #pragma omp parallel for schedule(static)
        for(int i=0; i<(int)peces.size(); ++i){
            peces[i].step(fish_qt, frame_count);
        }

        // Remove dead fish (serial)
        peces.erase(
            std::remove_if(peces.begin(), peces.end(), [](const Pez& p){ return p.energia <= 0.f; }),
            peces.end()
        );

        // Draw fish (serial)
        for(const auto& p : peces) p.draw(frame);

        // Write frame + show
        writer.write(frame);
        cv::imshow("Fish Simulation", frame);
        int key = cv::waitKey(1);
        if(key == 27) break; // ESC

        // Progress each second
        if(frame_count % FPS == 0){
            // Top 3 by food eaten (then energy)
            std::vector<const Pez*> top;
            top.reserve(peces.size());
            for(const auto& p : peces) top.push_back(&p);
            std::size_t n = std::min<std::size_t>(3, top.size());
            if(n > 0){
                std::partial_sort(top.begin(), top.begin()+n, top.end(),
                    [](const Pez* a, const Pez* b){
                        if(a->food_eaten != b->food_eaten) return a->food_eaten > b->food_eaten;
                        return a->energia > b->energia;
                    });
            }
            std::string top_txt = "—";
            if(n > 0){
                top_txt.clear();
                for(std::size_t i=0;i<n;i++){
                    if(i) top_txt += ", ";
                    top_txt += "#" + std::to_string(top[i]->id) + "(gen" + std::to_string(top[i]->gen)
                               + ",food" + std::to_string(top[i]->food_eaten) + ")";
                }
            }
            std::cout << "Progress: " << (frame_count / FPS) << "/" << DURATION_SEC
                      << " sec | fish: " << peces.size() << " | top: " << top_txt << "\n";
        }
    }

    writer.release();
    cv::destroyAllWindows();
    std::cout << "Video saved as " << videoPath << "\n";
    return 0;
}

