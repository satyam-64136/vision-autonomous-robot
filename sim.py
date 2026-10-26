"""2D simulator for the Hack in Hills robot (vision-based autonomous robot).

Sensors modelled = what the real build will have (NO 360 LiDAR):
  - forward camera: 60 deg field of view, 4 m range, noisy bearing/distance
  - 3 front distance rays (-25, 0, +25 deg), 2 m range (ToF / ultrasonic)
  - wheel encoders (4 channels, noisy, slip)
The robot is driven through the same text protocol the Arduino Uno firmware uses:
  "M <left> <right>"  (-255..255),  "S",  "RESET"
"""
import math, random, sys

W = 6.0                 # arena size (m)
DT = 0.05               # sim step (s)
WB = 0.20               # wheel base (m)
VMAX = 0.5              # m/s at PWM 255
TPM = 3000              # encoder ticks per metre
R_ROBOT = 0.12
FOV = math.radians(60)
CAM_RANGE = 4.0
TOF_RANGE = 2.0
TOF_ANGLES = [math.radians(a) for a in (-25, 0, 25)]
HARD_STOP = 0.15        # firmware-level emergency stop distance (m)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ray(x, y, a, circles, maxr):
    dx, dy = math.cos(a), math.sin(a)
    best = maxr
    for cx, cy, r in circles:
        fx, fy = x - cx, y - cy
        b = fx * dx + fy * dy
        disc = b * b - (fx * fx + fy * fy - r * r)
        if disc >= 0:
            t = -b - math.sqrt(disc)
            if 0 <= t < best:
                best = t
    for t in ((W - x) / dx if dx > 1e-9 else None, (0 - x) / dx if dx < -1e-9 else None,
              (W - y) / dy if dy > 1e-9 else None, (0 - y) / dy if dy < -1e-9 else None):
        if t is not None and 0 <= t < best:
            best = t
    return best


class World:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        r = self.rng
        self.x, self.y, self.th = 0.7, 0.7, r.uniform(-math.pi, math.pi)
        self.target = (r.uniform(3.5, 5.3), r.uniform(3.5, 5.3), 0.15)
        self.obst = []
        while len(self.obst) < 6:
            c = (r.uniform(1.2, 5.0), r.uniform(1.2, 5.0), r.uniform(0.2, 0.4))
            if math.hypot(c[0] - self.target[0], c[1] - self.target[1]) > 0.9 and \
               math.hypot(c[0] - self.x, c[1] - self.y) > 0.9:
                self.obst.append(c)
        self.gl, self.gr = 0.93, 1.0     # motor mismatch: left side is weaker
        self.cmd = (0, 0)
        self.enc_l = self.enc_r = 0.0
        self.collisions = 0
        self.estops = 0
        self.path = [(self.x, self.y)]

    # ---- "firmware" interface -------------------------------------------------
    def send(self, line):
        p = line.split()
        if not p:
            return
        if p[0] == "M":
            self.cmd = (max(-255, min(255, int(p[1]))), max(-255, min(255, int(p[2]))))
        elif p[0] == "S":
            self.cmd = (0, 0)
        elif p[0] == "RESET":
            self.enc_l = self.enc_r = 0.0

    def enc(self):
        l, r = int(self.enc_l), int(self.enc_r)
        return l, l, r, r            # m1..m4 (left pair, right pair)

    # ---- sensors --------------------------------------------------------------
    def tof(self):
        circ = self.obst + [self.target]
        return [ray(self.x, self.y, self.th + a, circ, TOF_RANGE) + self.rng.gauss(0, 0.01)
                for a in TOF_ANGLES]

    def camera(self):
        tx, ty, tr = self.target
        phi = math.atan2(ty - self.y, tx - self.x)
        bearing = wrap(phi - self.th)
        dist = math.hypot(tx - self.x, ty - self.y) - tr
        if abs(bearing) > FOV / 2 or dist > CAM_RANGE:
            return None
        if ray(self.x, self.y, phi, self.obst, dist + tr) < dist:   # blocked by obstacle
            return None
        if self.rng.random() < 0.05:                                # missed detection
            return None
        return bearing + self.rng.gauss(0, 0.03), dist * (1 + self.rng.gauss(0, 0.04))

    # ---- physics --------------------------------------------------------------
    def step(self):
        pl, pr = self.cmd
        vl, vr = pl / 255 * VMAX * self.gl, pr / 255 * VMAX * self.gr
        if min(self.tof()) < HARD_STOP and (vl + vr) > 0:           # firmware emergency stop
            vl = vr = 0
            self.estops += 1
        v, w = (vl + vr) / 2, (vr - vl) / WB
        nx = self.x + v * math.cos(self.th) * DT
        ny = self.y + v * math.sin(self.th) * DT
        if any(math.hypot(nx - cx, ny - cy) < r + R_ROBOT for cx, cy, r in self.obst) or \
           not (R_ROBOT < nx < W - R_ROBOT and R_ROBOT < ny < W - R_ROBOT):
            self.collisions += 1
            v = 0
        else:
            self.x, self.y = nx, ny
        self.th = wrap(self.th + w * DT)
        slip = lambda: 1 + self.rng.gauss(0, 0.02)
        self.enc_l += vl * DT * TPM * slip()
        self.enc_r += vr * DT * TPM * slip()
        self.path.append((self.x, self.y))


class Controller:
    """SEARCH -> APPROACH -> AVOID -> ARRIVED, built only on sensors the robot really has.
    SEARCH = spin once recording the freest heading (a 'scan'), turn to it, hop forward."""
    def __init__(self, world):
        self.w = world
        self.state, self.t_state = "SEARCH", 0.0
        self.last_bearing = 1
        self.lost_t = 0.0
        self.prev_enc = world.enc()
        self.labels = {"straight": 0, "curve": 0, "turn_in_place": 0}
        self._lab_t = 0.0
        self._lab_d = self._lab_th = 0.0
        self.avoid_dir = 1
        self.clear_t = 0.0
        self.yaw = 0.0                      # odometry yaw (encoders), rad
        self.new_search()

    def new_search(self):
        self.phase, self.yaw0, self.best = "spin", self.yaw, (-1.0, 0.0)
        self.hop_d = 0.0

    def odometry(self):
        e = self.w.enc()
        dl = (e[0] - self.prev_enc[0]) / TPM
        dr = (e[2] - self.prev_enc[2]) / TPM
        self.prev_enc = e
        return (dl + dr) / 2, (dr - dl) / WB

    def go(self, state):
        self.state, self.t_state = state, 0.0
        if state == "SEARCH":
            self.new_search()

    def tick(self, t):
        w = self.w
        d, dth = self.odometry()
        self.yaw += dth
        self._lab_t += DT; self._lab_d += d; self._lab_th += dth
        if self._lab_t >= 1.0:                       # straight / curved / turning classification
            if abs(self._lab_d) < 0.05:
                k = "turn_in_place" if abs(self._lab_th) > 0.2 else None
            else:
                k = "straight" if abs(self._lab_th) < 0.12 else "curve"
            if k:
                self.labels[k] += 1
            self._lab_t = self._lab_d = self._lab_th = 0.0
        self.t_state += DT
        cam, tof = w.camera(), w.tof()
        left, mid, right = tof
        if cam:
            self.last_bearing = 1 if cam[0] > 0 else -1

        if self.state == "ARRIVED":
            w.send("S"); return
        if cam and cam[1] < 0.45:
            self.go("ARRIVED"); w.send("S"); return

        driving = self.state == "APPROACH" or bool(cam) or \
            (self.state == "SEARCH" and self.phase == "hop")
        if self.state != "AVOID" and driving and min(tof) < 0.40 and not (cam and cam[1] < 0.7):
            self.avoid_dir = 1 if left > right else -1      # turn toward the more open side
            self.clear_t = 0.0
            self.go("AVOID")

        if self.state == "AVOID":
            if min(tof) > 0.55:                          # path is clear: bypass forward briefly
                self.clear_t += DT
                if self.clear_t < 1.2:
                    w.send("M 110 110")
                else:
                    self.go("SEARCH")
                return
            self.clear_t = 0.0
            w.send(f"M {-90 * self.avoid_dir} {90 * self.avoid_dir}")
            return

        if cam:
            self.lost_t = 0.0
            self.state = "APPROACH"
            b, dist = cam
            base = 150 if dist > 1.0 else 100
            turn = int(max(-120, min(120, 260 * b)))
            w.send(f"M {base - turn} {base + turn}")
            return

        if self.state == "APPROACH":                         # target just lost: keep turning toward it
            self.lost_t += DT
            w.send(f"M {-70 * self.last_bearing} {70 * self.last_bearing}")
            if self.lost_t > 1.5:
                self.go("SEARCH")
            return

        # SEARCH (no 360 sensor): spin once while logging the freest heading, align to it, hop
        turned = self.yaw - self.yaw0
        if self.phase == "spin":
            if mid > self.best[0]:
                self.best = (mid, turned)
            w.send("M -80 80")
            if turned >= 2 * math.pi:
                self.phase = "align"
        elif self.phase == "align":
            if turned >= 2 * math.pi + self.best[1]:
                self.phase, self.hop_d = "hop", 0.0
            else:
                w.send("M -80 80")
        else:
            self.hop_d += d
            if self.hop_d > 1.5 or mid < 0.6:
                self.new_search()
            else:
                w.send("M 140 140")


def run(seed, tmax=180.0, plot=False):
    w = World(seed)
    c = Controller(w)
    t = 0.0
    while t < tmax and c.state != "ARRIVED":
        c.tick(t)
        w.step()
        t += DT
    ok = c.state == "ARRIVED"
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - skipping the plot (pip install matplotlib)")
            return ok, t, w.collisions, w.estops, c.labels
        fig, ax = plt.subplots(figsize=(5, 5))
        for cx, cy, r in w.obst:
            ax.add_patch(plt.Circle((cx, cy), r, color="gray"))
        ax.add_patch(plt.Circle(w.target[:2], w.target[2], color="green"))
        ax.plot(*zip(*w.path), lw=1, color="tab:blue")
        ax.set_xlim(0, W); ax.set_ylim(0, W); ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: {'reached target' if ok else 'failed'} in {t:.0f}s")
        fig.savefig("sim_run.png", dpi=110)       # saved next to where you run the script
    return ok, t, w.collisions, w.estops, c.labels


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    res = [run(s, plot=(s == 0)) for s in range(n)]
    ok = [r for r in res if r[0]]
    print(f"runs: {n}  reached target: {len(ok)}/{n}")
    print(f"avg time (successes): {sum(r[1] for r in ok) / max(1, len(ok)):.1f}s")
    print(f"runs with collisions: {sum(1 for r in res if r[2] > 0)}  "
          f"total collision steps: {sum(r[2] for r in res)}  e-stop steps: {sum(r[3] for r in res)}")
    print("odometry labels (seed 0, seconds):", res[0][4])
    fails = [i for i, r in enumerate(res) if not r[0]]
    print("failed seeds:", fails)
