#!/usr/bin/env python3
"""
VISION-GUIDED AUTONOMOUS ROBOT - full software prototype (Hack in Hills '26, Track 01)

3D room (PyBullet) -> robot camera + range sensors -> AI vision -> navigation brain -> motors
Everything runs in simulation. The Brain only outputs (v, w) so the same code can drive an ESP32 later.

  python robot_sim.py --target "coke can"                 live dashboard
  python robot_sim.py --target "red bull can" --seed 3
  python robot_sim.py --headless --video demo.mp4         record, no window
  python robot_sim.py --bench 20 --detector color         20 random rooms, success table
  python robot_sim.py --bench 8 --latency 0.8 --target mix   "what if the AI needs 0.8 s per frame" (Raspberry Pi test)
  python robot_sim.py --explore legacy                    old explorer (default: map = remembers the room)
Detectors: yoloworld (open-vocabulary AI, needs internet once) | color (offline classical CV) | gt (debug oracle)
Keys: q quit | n new random room | space pause.   Type a new command in the terminal any time.
"""
import argparse, colorsys, csv, heapq, json, math, os, random, re, shutil, sys, tempfile, threading, time
from dataclasses import dataclass
import cv2
import numpy as np
import pybullet as p

# ----------------------------------------------------------------- constants
ROOM, HALF = 9.0, 4.5                    # square room, metres
HFOV = 70.0                              # camera horizontal field of view
CAM_FWD, CAM_Z = 0.12, 0.14              # camera position on the car
DT, SUB = 0.1, 6                         # control period (s), physics substeps per tick
V_MAX, W_MAX = 0.6, 1.6                  # m/s, rad/s
RMAX = 3.5                               # range sensor max distance
CELL = 0.75                              # (legacy explorer) exploration-memory grid size (m)
MRES = 0.15                              # room-map resolution: one map cell = 15 cm
MN = int(round(ROOM / MRES))             # map is MN x MN cells
MIN_GAIN, SPIN_GAIN = 120.0, 250.0         # map-explorer thresholds: smallest unsearched patch worth driving to / worth a 360-degree look
VIS_R = 3.0                              # how far the camera is trusted to spot a can (m)
ANGS = [math.radians(a) for a in range(-80, 81, 20)]   # 9 range sensors (hardware: servo-swept ultrasonic)
SUBA = [math.radians(a) for a in (-8, -4, 0, 4, 8)]    # each sensor has a ~16 deg beam, like a real ultrasonic
FONT = cv2.FONT_HERSHEY_SIMPLEX
EXP_D0, EXP_DK, EXP_P = (float(os.environ.get(k, v)) for k, v in (('EXP_D0', 25), ('EXP_DK', 45), ('EXP_P', 1.15)))   # viewpoint score = gain / (D0 + DK * path_m ** P)

@dataclass
class Safety:
    """Every safety number in one place (metres / seconds). The planner and the speed limiter both read these."""
    half_len: float = .15        # car body half length
    half_wid: float = .10        # car body half width
    sensor_sigma: float = .03    # range-sensor noise + beam-width uncertainty
    pose_sigma: float = .04      # odometry / map-position uncertainty
    cell_err: float = MRES / 4   # error from storing obstacles on a 15 cm grid
    soft_extra: float = .22      # extra clearance the planner PREFERS (slower/costlier, not forbidden)
    decel: float = .8            # braking deceleration (m/s^2)
    react: float = .35           # sensing + control reaction time (s)
    extra: float = 0.0           # user-set additional clearance (--margin)
    gap_min: float = .10         # never plan to stop closer than this to something ahead

    @property
    def robot_r(self):           # turning in place sweeps a circle of this radius
        return math.hypot(self.half_len, self.half_wid)

    @property
    def hard(self):              # centre-to-obstacle distance that is FORBIDDEN to enter
        return self.robot_r + math.hypot(self.sensor_sigma, self.pose_sigma) + self.cell_err + self.extra

    @property
    def soft(self):
        return self.hard + self.soft_extra

    def stop_dist(self, v):
        return v * self.react + v * v / (2 * self.decel)

    def v_safe(self, free):      # fastest speed from which we can still stop inside `free` metres
        s = max(free - self.gap_min, 0.0)
        return self.decel * (-self.react + math.sqrt(self.react ** 2 + 2 * s / self.decel))


SAFE = Safety()

WEIGHTS = "robot_det.pt"                 # produced by --train

CATALOG = {   # things the robot can be told to find
    "red":    dict(desc="red cola can", rgba=(.80, .06, .06, 1), stripe=(.95, .95, .95, 1),
                   words={"red", "cola"}, standoff=0.40),
    "blue":   dict(desc="blue energy drink can", rgba=(.05, .22, .80, 1), stripe=(.80, .82, .85, 1),
                   words={"blue", "energy", "drink"}, standoff=0.40),
    "green":  dict(desc="green soda can", rgba=(.05, .62, .15, 1), stripe=(.95, .95, .30, 1),
                   words={"green", "soda", "lemon"}, standoff=0.40),
    "person": dict(desc="person", rgba=(.95, .45, .05, 1), stripe=None,
                   words={"person", "human", "man", "people", "guy", "mannequin"}, standoff=0.90),
}
EXTRA_LABELS = ["cardboard box", "grey pillar"]          # obstacle classes for the AI detector


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def resolve_target(text):
    t = text.lower()
    t = re.sub(r"red\s*bull", "energy", t)
    t = re.sub(r"coca[\s-]*cola|coke|pepsi", "cola", t)
    t = re.sub(r"sprite|7up", "soda", t)
    words = set(re.findall(r"[a-z]+", t))
    best, score = None, 0
    for k, c in CATALOG.items():
        s = len(words & c["words"])
        if s > score:
            best, score = k, s
    return best


# ----------------------------------------------------------------- random world generator
MUTED = [(.62, .50, .36), (.45, .32, .20), (.50, .50, .52), (.90, .90, .88), (.15, .15, .17),
         (.25, .30, .42), (.42, .46, .30), (.70, .62, .50), (.35, .35, .38), (.80, .75, .65)]


def rnd_color(rng, vivid=.25):
    if rng.random() < vivid:
        r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(.55, .95), rng.uniform(.5, .95))
    else:
        r, g, b = [min(1, max(0, c + rng.uniform(-.06, .06))) for c in rng.choice(MUTED)]
    return (r, g, b, 1)


def make_item(rng, typ):
    """A random real-world-like object. Sizes, colours and rotation are random every time."""
    U = rng.uniform
    d = {"type": typ, "yaw": U(0, math.pi), "col": rnd_color(rng)}
    if typ == "box":         d.update(shape="rect", hx=U(.12, .35), hy=U(.12, .3), hz=U(.07, .3))
    elif typ == "crate":     d.update(shape="rect", hx=U(.2, .4), hy=U(.2, .35), hz=U(.15, .3), col=rnd_color(rng, .1))
    elif typ == "barrel":    d.update(shape="circle", r=U(.14, .27), hz=U(.2, .45))
    elif typ == "bin":       d.update(shape="circle", r=U(.12, .22), hz=U(.15, .3), col=rnd_color(rng, .6))
    elif typ == "table":     d.update(shape="rect", hx=U(.35, .6), hy=U(.25, .4), hz=U(.18, .28))
    elif typ == "shelf":     d.update(shape="rect", hx=U(.35, .6), hy=U(.12, .2), hz=U(.45, .7))
    elif typ == "plant":     d.update(shape="circle", r=U(.14, .25), hz=U(.08, .14))
    elif typ == "ball":      d.update(shape="circle", r=U(.1, .25), col=rnd_color(rng, .7))
    elif typ == "cone":      d.update(shape="circle", r=U(.09, .14), hz=U(.15, .28), col=rnd_color(rng, .8))
    elif typ == "partition": d.update(shape="rect", hx=U(.6, 1.3), hy=.05, hz=U(.25, .5))
    elif typ == "bottle":    d.update(shape="circle", r=U(.04, .055), hz=U(.11, .16), col=rnd_color(rng, .6))
    elif typ == "small":     d.update(shape="rect", hx=U(.06, .12), hy=U(.05, .1), hz=U(.045, .08))
    # ---- "novel" shapes the AI was never trained on (use --novel): they test unknown-obstacle handling
    elif typ == "pipe":      d.update(shape="rect", hx=U(.3, .6), hy=.08, hz=.08)
    elif typ == "snow":      d.update(shape="circle", r=.2, hz=0)
    elif typ == "lshape":    d.update(shape="rect", hx=.4, hy=.3, hz=.2)
    elif typ == "pole":      d.update(shape="circle", r=.035, hz=U(.4, .8))
    elif typ == "sofa":      d.update(shape="rect", hx=U(.5, .7), hy=.3, hz=.15)
    elif typ == "ottoman":   d.update(shape="circle", r=U(.2, .3), hz=.08)
    if d["shape"] == "rect" and d["hx"] < d["hy"]:
        d["hx"], d["hy"] = d["hy"], d["hx"]
    d.setdefault("r", math.hypot(d.get("hx", 0), d.get("hy", 0)))
    return d


def circles(d):
    """Footprint of an item as a few circles (used for spacing and for spawning)."""
    x, y = d["x"], d["y"]
    if d["shape"] == "circle":
        return [(x, y, d["r"])]
    hx, hy, yaw = d["hx"], d["hy"], d["yaw"]
    k = max(1, int(round(hx / hy)))
    offs = [0.0] if k == 1 else list(np.linspace(-(hx - hy), hx - hy, k))
    return [(x + o * math.cos(yaw), y + o * math.sin(yaw), hy * 1.42) for o in offs]


NOVEL = ["pipe", "snow", "lshape", "pole", "sofa", "ottoman"]
CLUTTER = (["box"] * 5 + ["crate"] * 3 + ["barrel"] * 3 + ["bin"] * 2 + ["table"] * 3 + ["shelf"] * 2 +
           ["plant"] * 3 + ["ball"] * 2 + ["cone"] * 2 + ["partition"] * 3 + ["bottle"] * 3 + ["small"] * 3)


# ----------------------------------------------------------------- simulator
class Sim:
    def __init__(self, res=(400, 300), noise=False, novel=False):
        self.W, self.H = res
        self.noise, self.novel = noise, novel
        p.connect(p.DIRECT)
        self.f = (self.W / 2) / math.tan(math.radians(HFOV / 2))
        vfov = math.degrees(2 * math.atan((self.H / 2) / self.f))
        self.proj = p.computeProjectionMatrixFOV(vfov, self.W / self.H, 0.02, 20)

    # ---- body helpers
    def _vis(self, shape, rgba, **kw):
        return p.createVisualShape(shape, rgbaColor=rgba, **kw)

    def _reg(self, uid, kind, gid, label):
        self.kind[uid] = kind
        if label:
            self.uid_group[uid] = gid
            self.group_label[gid] = label

    def _box(self, pos, half, yaw, rgba, gid=None, label=None, collide=True, kind="obstacle"):
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half) if collide else -1
        self._reg(p.createMultiBody(0, c, v, pos, p.getQuaternionFromEuler([0, 0, yaw])),
                  kind if collide else "part", gid, label)

    def _cyl(self, pos, r, h, rgba, gid=None, label=None, collide=True, kind="obstacle"):
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=rgba)
        c = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h) if collide else -1
        self._reg(p.createMultiBody(0, c, v, pos), kind if collide else "part", gid, label)

    def _sph(self, pos, r, rgba, gid=None, label=None, collide=True):
        v = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba)
        c = p.createCollisionShape(p.GEOM_SPHERE, radius=r) if collide else -1
        self._reg(p.createMultiBody(0, c, v, pos), "obstacle" if collide else "part", gid, label)

    # ---- layout
    def _layout(self, rng, start):
        want = [{"type": "can", "key": k, "shape": "circle", "r": .15, "yaw": 0, "col": None}
                for k in ("red", "blue", "green")]
        want.append({"type": "person", "shape": "circle", "r": .30, "yaw": 0, "col": None})
        want += [make_item(rng, rng.choice(CLUTTER)) for _ in range(rng.randint(28, 36))]
        if self.novel:
            want += [make_item(rng, t) for t in NOVEL]
        placed, lim = [], HALF - .6
        for d in want:
            target = d["type"] in ("can", "person")
            for attempt in range(700):
                d["x"], d["y"] = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
                gap = .5 if attempt < 500 else .3
                cs = circles(d)
                if any(math.hypot(cx - start[0], cy - start[1]) < cr + (2.0 if target else .8) for cx, cy, cr in cs):
                    continue
                if all(math.hypot(cx - ox, cy - oy) >= cr + orr + gap
                       for cx, cy, cr in cs for o in placed for ox, oy, orr in circles(o)):
                    placed.append(d)
                    break
        return placed

    def reset(self, seed):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(DT / SUB)
        self.rng = rng = random.Random(seed)
        self.kind, self.uid_group, self.group_label = {}, {}, {}
        self.items, self.trail = {}, []
        self.t, self.path, self.collisions, self.person_hits = 0.0, 0.0, 0, 0
        self.touch_prev, self.events = set(), []
        fl = rng.choice([(.74, .70, .64), (.60, .60, .62), (.55, .45, .35), (.80, .80, .78), (.50, .55, .50)])
        wl = rng.choice([(.86, .87, .90), (.9, .85, .75), (.75, .8, .85), (.8, .8, .8)])
        self._box([0, 0, -.05], [HALF + .5, HALF + .5, .05], 0, fl + (1,), kind="floor")
        for (x, y, hx, hy) in [(0, HALF + .05, HALF + .1, .05), (0, -HALF - .05, HALF + .1, .05),
                               (HALF + .05, 0, .05, HALF + .1), (-HALF - .05, 0, .05, HALF + .1)]:
            self._box([x, y, .3], [hx, hy, .3], 0, wl + (1,), kind="wall")
        start = (rng.uniform(-HALF + 1.2, HALF - 1.2), rng.uniform(-HALF + 1.2, HALF - 1.2))
        yaw0 = rng.uniform(-math.pi, math.pi)
        self.layout = self._layout(rng, start)
        for gid, d in enumerate(self.layout):
            self._build(d, gid)
        self._make_car(start, yaw0)

    def _build(self, d, gid):
        t, x, y, yaw, col = d["type"], d["x"], d["y"], d["yaw"], d["col"]
        ob = "obstacle"
        if t == "can":
            cat = CATALOG[d["key"]]
            c = p.createCollisionShape(p.GEOM_CYLINDER, radius=.033, height=.12)
            v = p.createVisualShape(p.GEOM_CYLINDER, radius=.033, length=.12, rgbaColor=cat["rgba"])
            self._reg(p.createMultiBody(0, c, v, [x, y, .06]), "target:" + d["key"], gid, cat["desc"])
            self._cyl([x, y, .06], .0345, .035, cat["stripe"], gid, cat["desc"], collide=False)
            self._cyl([x, y, .122], .030, .006, (.82, .82, .85, 1), gid, cat["desc"], collide=False)
            self.items[d["key"]] = (x, y)
        elif t == "person":
            org = CATALOG["person"]["rgba"]
            c = p.createCollisionShape(p.GEOM_CYLINDER, radius=.17, height=1.0)
            v = p.createVisualShape(p.GEOM_CYLINDER, radius=.15, length=.55, rgbaColor=org,
                                    visualFramePosition=[0, 0, .125])
            self._reg(p.createMultiBody(0, c, v, [x, y, .5]), "person", gid, "person")
            for dx, r_, h_, z_, cl in [(.07, .06, .4, .2, org), (-.07, .06, .4, .2, org),
                                       (.20, .04, .45, .65, org), (-.20, .04, .45, .65, org),
                                       (0, .10, 0, 1.0, (.90, .72, .58, 1))]:
                if h_ == 0:
                    self._sph([x + dx, y, z_], r_, cl, gid, "person", collide=False)
                else:
                    self._cyl([x + dx, y, z_], r_, h_, cl, gid, "person", collide=False)
            self.items["person"] = (x, y)
        elif t in ("box", "crate", "shelf", "small", "partition"):
            self._box([x, y, d["hz"]], [d["hx"], d["hy"], d["hz"]], yaw, col, gid, ob)
        elif t in ("barrel", "bin"):
            self._cyl([x, y, d["hz"]], d["r"], 2 * d["hz"], col, gid, ob)
            if t == "bin":
                self._cyl([x, y, 2 * d["hz"] + .01], d["r"] * 1.05, .03, tuple(.6 * c for c in col[:3]) + (1,), gid, ob, collide=False)
        elif t == "table":
            H, c_, s_ = 2 * d["hz"], math.cos(yaw), math.sin(yaw)
            self._box([x, y, H], [d["hx"], d["hy"], .02], yaw, col, gid, ob, collide=False)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lx, ly = sx * (d["hx"] - .05), sy * (d["hy"] - .05)
                    self._box([x + lx * c_ - ly * s_, y + lx * s_ + ly * c_, H / 2], [.025, .025, H / 2], yaw, col, gid, ob)
        elif t == "plant":
            self._cyl([x, y, d["hz"]], d["r"], 2 * d["hz"], (.55, .35, .25, 1), gid, ob)
            g = self.rng.uniform(.25, .5)
            self._sph([x, y, 2 * d["hz"] + d["r"] * 1.1], d["r"] * 1.5, (.1, g, .1, 1), gid, ob, collide=False)
        elif t == "ball":
            self._sph([x, y, d["r"]], d["r"], col, gid, ob)
        elif t == "cone":
            self._cyl([x, y, d["hz"]], d["r"], 2 * d["hz"], col, gid, ob)
            self._cyl([x, y, d["hz"] * 1.3], d["r"] * 1.03, d["hz"] * .4, (.95, .95, .95, 1), gid, ob, collide=False)
        elif t == "bottle":
            self._cyl([x, y, d["hz"]], d["r"], 2 * d["hz"], col, gid, ob)
            self._cyl([x, y, 2 * d["hz"] + .015], d["r"] * .5, .03, (.2, .2, .2, 1), gid, ob, collide=False)
        elif t == "pipe":                                   # a pipe lying on the floor (horizontal cylinder)
            q = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, yaw]), [0, 0, 0],
                                     p.getQuaternionFromEuler([0, math.pi / 2, 0]))[1]
            v = p.createVisualShape(p.GEOM_CYLINDER, radius=d["hy"], length=2 * d["hx"], rgbaColor=col)
            c = p.createCollisionShape(p.GEOM_CYLINDER, radius=d["hy"], height=2 * d["hx"])
            self._reg(p.createMultiBody(0, c, v, [x, y, d["hy"]], q), "obstacle", gid, ob)
        elif t == "snow":                                   # three stacked balls
            for z_, r_ in ((.2, .2), (.5, .14), (.74, .1)):
                self._sph([x, y, z_], r_, col, gid, ob)
        elif t == "lshape":                                 # an L-shaped piece of furniture
            c_, s_ = math.cos(yaw), math.sin(yaw)
            self._box([x, y, .2], [.4, .07, .2], yaw, col, gid, ob)
            self._box([x + .33 * c_ - .23 * s_, y + .33 * s_ + .23 * c_, .2], [.07, .3, .2], yaw, col, gid, ob)
        elif t == "pole":
            self._cyl([x, y, d["hz"]], d["r"], 2 * d["hz"], col, gid, ob)
        elif t == "sofa":
            c_, s_ = math.cos(yaw), math.sin(yaw)
            self._box([x, y, d["hz"]], [d["hx"], d["hy"], d["hz"]], yaw, col, gid, ob)
            self._box([x + (d["hy"] - .07) * s_, y - (d["hy"] - .07) * c_, .3], [d["hx"], .07, .3], yaw, col, gid, ob)
        elif t == "ottoman":
            self._cyl([x, y, d["hz"]], d["r"], 2 * d["hz"], col, gid, ob)

    def _make_car(self, start, yaw0):
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.15, .10, .05])
        v = self._vis(p.GEOM_BOX, (.15, .35, .55, 1), halfExtents=[.15, .10, .05])
        wv = self._vis(p.GEOM_CYLINDER, (.08, .08, .08, 1), radius=.05, length=.03)
        mv = self._vis(p.GEOM_BOX, (.9, .9, .2, 1), halfExtents=[.02, .02, .04])
        q = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        z = [0, 0, 0, 1]
        self.car = p.createMultiBody(
            3, c, v, [start[0], start[1], .07], p.getQuaternionFromEuler([0, 0, yaw0]),
            linkMasses=[0, 0, 0], linkCollisionShapeIndices=[-1] * 3, linkVisualShapeIndices=[wv, wv, mv],
            linkPositions=[[-.07, .12, -.02], [-.07, -.12, -.02], [CAM_FWD, 0, .09]],
            linkOrientations=[q, q, z], linkInertialFramePositions=[[0, 0, 0]] * 3,
            linkInertialFrameOrientations=[z] * 3, linkParentIndices=[0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED] * 3, linkJointAxis=[[0, 0, 1]] * 3)
        p.changeDynamics(self.car, -1, lateralFriction=.2, linearDamping=0, angularDamping=0)
        self.kind[self.car] = "car"
        self.trail = [start]

    # ---- things that change while the robot is working (to test replanning)
    def add_obstacle(self, x, y, typ="crate"):
        d = make_item(self.rng, typ)
        d["x"], d["y"] = x, y
        self.layout.append(d)
        self._build(d, len(self.layout) - 1)
        return d

    def move_item(self, key, x, y):
        """Pick up a can / the person and put it somewhere else."""
        ox, oy = self.items[key]
        gid = next(g for g, d in enumerate(self.layout) if d.get("key") == key or d["type"] == key)
        for uid, g in list(self.uid_group.items()):
            if g == gid:
                pos, orn = p.getBasePositionAndOrientation(uid)
                p.resetBasePositionAndOrientation(uid, [pos[0] + x - ox, pos[1] + y - oy, pos[2]], orn)
        self.layout[gid]["x"], self.layout[gid]["y"], self.items[key] = x, y, (x, y)

    def free_spot(self, rng, min_robot=2.0, gap=.6):
        circs = [c for it in self.layout for c in circles(it)]
        rx, ry, _ = self.pose()
        for _ in range(400):
            x, y = rng.uniform(-HALF + .8, HALF - .8), rng.uniform(-HALF + .8, HALF - .8)
            if math.hypot(x - rx, y - ry) > min_robot and all(math.hypot(x - cx, y - cy) > cr + gap for cx, cy, cr in circs):
                return x, y
        return None

    # ---- state
    def pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.car)
        return pos[0], pos[1], p.getEulerFromQuaternion(orn)[2]

    def step(self, v, w, dt=DT):
        x0, y0, _ = self.pose()
        p.setTimeStep(dt / SUB)
        for _ in range(SUB):
            x, y, yaw = self.pose()
            p.resetBasePositionAndOrientation(self.car, [x, y, .07], p.getQuaternionFromEuler([0, 0, yaw]))
            p.resetBaseVelocity(self.car, [v * math.cos(yaw), v * math.sin(yaw), 0], [0, 0, w])
            p.stepSimulation()
        self.t += dt
        x, y, _ = self.pose()
        self.path += math.hypot(x - x0, y - y0)
        self.trail = (self.trail + [(x, y)])[-600:]
        touch = set()
        for cp in p.getContactPoints(self.car):
            if cp[8] < 0.003:
                k = self.kind.get(cp[2], "")
                if k in ("wall", "obstacle", "person") or (k.startswith("target:")):
                    touch.add((k, cp[2]))
        for k, uid in touch - self.touch_prev:
            if k == "person":
                self.person_hits += 1
            if k in ("wall", "obstacle", "person"):
                self.collisions += 1
                self.events.append((round(self.t, 1), k))
        self.touch_prev = touch

    # ---- sensors
    def camera(self, light=None):
        x, y, yaw = self.pose()
        eye = [x + CAM_FWD * math.cos(yaw), y + CAM_FWD * math.sin(yaw), CAM_Z]
        tgt = [eye[0] + math.cos(yaw), eye[1] + math.sin(yaw), CAM_Z]
        view = p.computeViewMatrix(eye, tgt, [0, 0, 1])
        kw = {"lightDirection": light} if light else {}
        _, _, rgb, _, seg = p.getCameraImage(self.W, self.H, view, self.proj, renderer=p.ER_TINY_RENDERER, **kw)
        rgb = np.reshape(rgb, (self.H, self.W, 4))[:, :, :3].astype(np.uint8)
        seg = np.reshape(seg, (self.H, self.W))
        bgr = rgb[:, :, ::-1].copy()
        sky = seg < 0
        grad = np.linspace(0, 1, self.H)[:, None]
        sky_col = (np.array([235, 205, 165]) * (1 - grad) + np.array([250, 235, 215]) * grad)[:, None, :]
        bgr[sky] = np.broadcast_to(sky_col, (self.H, self.W, 3))[sky].astype(np.uint8)
        if self.noise:
            bgr = np.clip(bgr.astype(np.float32) * self.rng.uniform(.85, 1.15)
                          + np.random.normal(0, 4, bgr.shape), 0, 255).astype(np.uint8)
        return bgr, seg

    def ranges(self):
        x, y, yaw = self.pose()
        o = [x + .16 * math.cos(yaw), y + .16 * math.sin(yaw), .07]
        angs = [a + d for a in ANGS for d in SUBA]          # 5 sub-rays per sensor = ~16 deg beam
        to = [[o[0] + RMAX * math.cos(yaw + a), o[1] + RMAX * math.sin(yaw + a), .07] for a in angs]
        res = p.rayTestBatch([o] * len(angs), to)
        d = [(h[2] * RMAX if h[0] >= 0 else RMAX) for h in res]
        n = len(SUBA)
        return [max(.03, min(d[i * n:(i + 1) * n]) + self.rng.gauss(0, .01)) for i in range(len(ANGS))]

    # ---- ground truth (used ONLY for scoring, never by the brain)
    def judge(self, key):
        x, y, yaw = self.pose()
        cx, cy = x + CAM_FWD * math.cos(yaw), y + CAM_FWD * math.sin(yaw)
        if key not in self.items:
            return "UNKNOWN"
        dist = {k: math.hypot(px - cx, py - cy) for k, (px, py) in self.items.items()}
        reach = lambda k: dist[k] <= CATALOG[k]["standoff"] + 0.30
        if reach(key):
            return "SUCCESS"
        return "WRONG_TARGET" if any(reach(k) for k in dist) else "TOO_FAR"


# ----------------------------------------------------------------- AI vision
@dataclass
class Det:
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def merge_boxes(boxes, gap=0.7, xgap=0.0):
    boxes = [list(b) for b in boxes]          # x1,y1,x2,y2
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                hmax = max(a[3] - a[1], b[3] - b[1])
                xo = min(a[2], b[2]) - max(a[0], b[0])
                yg = max(a[1], b[1]) - min(a[3], b[3])
                if xo > -xgap * hmax and yg <= gap * hmax:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


class YoloWorldDetector:
    """Open-vocabulary AI: finds things from TEXT descriptions, no retraining."""
    name = "YOLO-World (AI)"

    def __init__(self, imgsz=640, conf=0.12):
        from ultralytics import YOLOWorld
        self.labels = [c["desc"] for c in CATALOG.values()] + EXTRA_LABELS
        self.m = YOLOWorld("yolov8s-world.pt")
        self.m.set_classes(self.labels)
        self.imgsz, self.conf = imgsz, conf

    def detect(self, bgr, seg, sim):
        r = self.m.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [Det(self.labels[int(b.cls[0])], float(b.conf[0]), *b.xyxy[0].tolist()) for b in r.boxes]


class ColorDetector:
    """Offline classical CV baseline: HSV colour segmentation + shape. No downloads."""
    name = "Colour/shape CV (offline)"
    HSV = {"red cola can": [((0, 120, 60), (5, 255, 255)), ((172, 120, 60), (180, 255, 255))],
           "blue energy drink can": [((100, 120, 40), (130, 255, 255))],
           "green soda can": [((45, 100, 40), (85, 255, 255))],
           "person": [((7, 190, 150), (22, 255, 255))]}

    def detect(self, bgr, seg, sim):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        min_a = max(6, bgr.shape[0] * bgr.shape[1] / 50000)
        out = []
        for label, rngs in self.HSV.items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in rngs:
                m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in cnts:
                if cv2.contourArea(c) >= min_a:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((x, y, x + w, y + h))
            for b in merge_boxes(boxes, xgap=0.15 if label == "person" else 0.0):
                w, h = b[2] - b[0], b[3] - b[1]
                if label != "person" and not (0.8 <= h / max(w, 1) <= 4.5):
                    continue
                fill = float((m[b[1]:b[3], b[0]:b[2]] > 0).mean())
                out.append(Det(label, min(.99, .45 + fill), *map(float, b)))
        return out


def classify_color(crop):
    """Which catalog can does this crop look like? (reuses the HSV ranges above)"""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best, best_n = None, 0
    for label, rngs in ColorDetector.HSV.items():
        if label == "person":
            continue
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in rngs:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        n = int(m.sum() / 255)
        if n > best_n:
            best, best_n = label, n
    return best if best_n >= 0.08 * crop.shape[0] * crop.shape[1] else None


class HybridDetector:
    """AI finds generic 'drink can' / 'person' / obstacles; colour then says WHICH can."""
    name = "YOLO-World + colour classifier (hybrid)"

    def __init__(self, imgsz=640, conf=0.03):
        import os
        from ultralytics import YOLOWorld
        self.labels = ["drink can", "person"] + EXTRA_LABELS
        self.m = YOLOWorld("yolov8s-world.pt")
        self.m.set_classes(self.labels)
        self.imgsz, self.conf = imgsz, conf
        self.debug = os.environ.get("DET_DEBUG") == "1"

    def detect(self, bgr, seg, sim):
        r = self.m.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b in r.boxes:
            lab, cf = self.labels[int(b.cls[0])], float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if self.debug:
                print(f"  raw: {lab} {cf:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            if lab == "drink can":
                lab = classify_color(bgr[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1])
                if lab is None:
                    continue
            out.append(Det(lab, max(cf, .25), x1, y1, x2, y2))
        return out


def floor_hazards(bgr, f, thr=30, max_d=3.0, step=4):
    """LABEL-FREE obstacle vision: anything standing on the floor that is not floor-coloured is a hazard, whatever it is.
    The floor colour is learned from the bottom of the picture; for every image column we walk up from the bottom
    to the first non-floor pixels = where the object touches the ground = its distance (ground-plane geometry).
    Returns [(bearing, distance, column)]."""
    H, W = bgr.shape[:2]
    hz = H // 2
    ref = np.median(bgr[int(H * .88):, ::4].reshape(-1, 3).astype(np.int16), axis=0)
    non = np.abs(bgr[hz + 3:].astype(np.int16) - ref).sum(axis=2) > thr
    run3 = non[:-2] & non[1:-1] & non[2:]
    has = run3.any(axis=0)
    idx = run3.shape[0] - 1 - np.argmax(run3[::-1], axis=0)
    row = hz + 3 + idx + 2
    out = []
    for x in range(0, W, step):
        if has[x]:
            d = CAM_Z * f / max(row[x] - hz, 1)
            if d < max_d:
                out.append((-math.atan2(x - W / 2, f), float(d), x))
    return out


def hazard_boxes(haz, f, H):
    """Group neighbouring hazard columns into 'unknown object' boxes (for the dashboard)."""
    boxes, cur = [], []
    for b, d, x in haz:
        if cur and (x - cur[-1][2] > 8 or abs(d - cur[-1][1]) > .25 * d + .1):
            boxes.append(cur)
            cur = []
        cur.append((b, d, x))
    if cur:
        boxes.append(cur)
    out = []
    for g in boxes:
        d = float(np.median([q[1] for q in g]))
        row = H / 2 + CAM_Z * f / d
        out.append(Det("unknown object", .5, g[0][2] - 2, row - .25 * f / d, g[-1][2] + 2, row))
    return out


def gt_boxes(sim, seg, min_small=10, min_big=60):
    """Exact boxes from the simulator's segmentation mask: [(label, (x1,y1,x2,y2)), ...]."""
    boxes = {}
    for uid in np.unique(seg):
        gid = sim.uid_group.get(int(uid))
        if gid is None:
            continue
        ys, xs = np.where(seg == uid)
        b = boxes.setdefault(gid, [10 ** 6, 10 ** 6, -1, -1, 0])
        b[0], b[1] = min(b[0], xs.min()), min(b[1], ys.min())
        b[2], b[3], b[4] = max(b[2], xs.max()), max(b[3], ys.max()), b[4] + len(xs)
    out = []
    for gid, b in boxes.items():
        lab = sim.group_label[gid]
        if b[4] >= (min_big if lab == "obstacle" else min_small):
            out.append((lab, tuple(float(v) for v in b[:4])))
    return out


class TrainedDetector:
    """YOLOv8n trained on pictures from THIS simulator (labels came free from the segmentation mask)."""
    name = "Trained YOLOv8n (sim-trained)"

    def __init__(self, weights=WEIGHTS, imgsz=416, conf=0.2):
        from ultralytics import YOLO
        self.m, self.imgsz, self.conf = YOLO(weights), imgsz, conf

    def detect(self, bgr, seg, sim):
        r = self.m.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False, agnostic_nms=True)[0]   # one box per object
        out = []
        for b in r.boxes:
            lab, cf = self.m.names[int(b.cls[0])], float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if lab in LABEL2KEY and lab != "person":        # second opinion on cans: does the PAINT agree with the AI's label?
                col = classify_color(bgr[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1])
                if col is not None and col != lab:
                    lab, cf = col, cf * .9                    # the AI found the can correctly but named it wrongly
            out.append(Det(lab, cf, x1, y1, x2, y2))
        return out


class GroundTruthDetector:
    """Debug oracle: reads the simulator's segmentation mask. Upper bound, NOT real vision."""
    name = "Ground-truth oracle (debug)"

    def detect(self, bgr, seg, sim):
        return [Det(lab, .95, *box) for lab, box in gt_boxes(sim, seg, 6, 6)]


# ----------------------------------------------------------------- room map (the robot's memory)
class WorldMap:
    """The robot's memory of the room, stored in WORLD coordinates (not relative to where it is facing),
    so turning, driving or a slow camera can never make it forget anything.
        lo      obstacle memory   (log-odds grid, filled from the range sensors)
        seen    searched memory   (cells the camera has actually looked at)
        visits  where the robot has been
        tg      remembered objects (cans / person) with an averaged position
    On real hardware the pose would come from wheel encoders (+ IMU); everything else stays the same."""

    def __init__(self):
        self.lo = np.zeros((MN, MN), np.float32)
        self.lo[0, :] = self.lo[-1, :] = self.lo[:, 0] = self.lo[:, -1] = 4.0      # the room walls are known
        self.seen = np.zeros((MN, MN), bool)
        self.vis = np.zeros((MN, MN), np.float32)          # camera-only obstacle evidence (fades; sonar can overrule it)
        self.sfree = np.zeros((MN, MN), np.uint8)          # how often the sonar has seen this cell EMPTY
        self.visits = np.zeros((MN, MN), np.float32)
        self.tg, self.sweep, self.bad, self.neg, self.sus = {}, 1, [], {}, {}
        self.room = None
        self.dist = self.prev = None
        k = np.arange(1, 24)                                                      # rays used to estimate "what would I see from there"
        th = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        self.rx = np.round(k[None, :] * np.cos(th)[:, None]).astype(int)
        self.ry = np.round(k[None, :] * np.sin(th)[:, None]).astype(int)
        self.rw = k.astype(np.float32)                                            # farther cells stand for more floor area
        self.inner = float((MN - 2) ** 2)

    # ---- helpers
    @staticmethod
    def cell(x, y):
        return (int(np.clip((y + HALF) // MRES, 0, MN - 1)), int(np.clip((x + HALF) // MRES, 0, MN - 1)))   # (row, col)

    @staticmethod
    def centre(r, c):
        return (c + .5) * MRES - HALF, (r + .5) * MRES - HALF

    def occ(self):
        """Obstacle cells: sonar evidence, or camera evidence that no sonar ray has contradicted
        (tabletops, foliage and other overhangs look solid to a camera but the sonar passes underneath)."""
        return (self.lo > 0.6) | ((self.vis > 0.6) & (self.sfree < 2))

    @staticmethod
    def grow(m, n=1):
        g = m.copy()
        for _ in range(n):
            h = g.copy()
            h[1:, :] |= g[:-1, :]
            h[:-1, :] |= g[1:, :]
            h[:, 1:] |= g[:, :-1]
            h[:, :-1] |= g[:, 1:]
            h[1:, 1:] |= g[:-1, :-1]
            h[:-1, :-1] |= g[1:, 1:]
            h[1:, :-1] |= g[:-1, 1:]
            h[:-1, 1:] |= g[1:, :-1]
            g = h
        return g

    def pct(self):
        return 100.0 * float(self.seen[1:-1, 1:-1].sum()) / self.inner

    # ---- updating the map
    def update_range(self, pose, ranges, dt=0.1):
        """Every tick, from the CURRENT pose: free space along each sonar ray, an obstacle where it ended."""
        x, y, yaw = pose
        ox, oy = x + .16 * math.cos(yaw), y + .16 * math.sin(yaw)
        self.vis *= .995
        for a, r in zip(ANGS, ranges):
            th = yaw + a
            ds = np.arange(0.0, max(r - .05, .01), MRES * .5)
            ix = np.clip(((ox + ds * math.cos(th)) + HALF) // MRES, 0, MN - 1).astype(int)
            iy = np.clip(((oy + ds * math.sin(th)) + HALF) // MRES, 0, MN - 1).astype(int)
            self.lo[iy, ix] = np.maximum(self.lo[iy, ix] - .2, -2.0)
            self.sfree[iy, ix] = np.minimum(self.sfree[iy, ix] + 1, 5)
            if r < RMAX - .08:
                ey, ex = self.cell(ox + (r + .03) * math.cos(th), oy + (r + .03) * math.sin(th))
                self.lo[ey, ex] = min(4.0, self.lo[ey, ex] + 1.2)
                self.sfree[ey, ex] = 0
        r_, c_ = self.cell(x, y)
        self.lo[max(0, r_ - 1):r_ + 2, max(0, c_ - 1):c_ + 2] = np.minimum(self.lo[max(0, r_ - 1):r_ + 2, max(0, c_ - 1):c_ + 2], 0.0)
        self.visits[r_, c_] += dt

    def update_cover(self, pose, ranges):
        """Mark what the camera looked at. Called with the pose/ranges of the frame the detector actually
        processed, so a slow detector cannot make the robot believe it searched somewhere it did not."""
        x, y, yaw = pose
        cx, cy = x + CAM_FWD * math.cos(yaw), y + CAM_FWD * math.sin(yaw)
        half = math.radians(HFOV / 2) - .03
        bs = np.linspace(-half, half, 17)
        lim = np.minimum(VIS_R, np.interp(bs, ANGS, ranges) + .25)
        for b, L in zip(bs, lim):
            ds = np.arange(.2, L, MRES * .7)
            ix = np.clip(((cx + ds * math.cos(yaw + b)) + HALF) // MRES, 0, MN - 1).astype(int)
            iy = np.clip(((cy + ds * math.sin(yaw + b)) + HALF) // MRES, 0, MN - 1).astype(int)
            self.seen[iy, ix] = True

    # ---- remembered objects
    def observe(self, label, x, y, d, t):
        """Add one sighting. Close-up sightings are more accurate, so they count more in the average."""
        if self.rejected(label, x, y):
            return
        w = 1.0 / max(d, .5) ** 2
        for c in self.tg.setdefault(label, []):
            if math.hypot(c["x"] - x, c["y"] - y) < .7:
                c["x"], c["y"] = (c["x"] * c["w"] + x * w) / (c["w"] + w), (c["y"] * c["w"] + y * w) / (c["w"] + w)
                c["w"] += w
                c["n"] += 1
                c["t"] = t
                if d < 1.5:
                    c["miss"] = 0                               # seen again from close up: it is really there
                return
        self.tg[label].append(dict(x=x, y=y, w=w, n=1, t=t, miss=0, ver=False))

    def pick_recall(self, label, x, y, yaw, min_n=3, escape=False):
        """Which remembered spot to drive to for `label`, and the optimised A* route to it.
        Ranking (NOT 'whoever was seen in the most frames' - staring at a false alarm piles up frames):
        1. spots that never failed a visit  2. spots where we really arrived and verified the object
        3. confidence bucket (many sightings)  4. shortest A* route (time cost). Returns (cluster, goal, path, margin)."""
        best = None
        for c in self.tg.get(label, []):
            if c["n"] < min_n:
                continue
            goal = self.goal_near(x, y, c["x"], c["y"], escape=escape)
            path, m = self.route((x, y), yaw, goal, escape) if goal else (None, None)
            if not path:
                continue
            length = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))
            bucket = 0 if c["n"] >= 6 else 1 if c["n"] >= 4 else 2
            key = (c.get("miss", 0), 0 if c.get("ver") else 1, bucket, length)
            if best is None or key < best[0]:
                best = (key, c, goal, path, m)
        return best[1:] if best else None

    def verify(self, label, x, y):
        """The car really got to the object and the judge confirmed it: this spot is the truth now."""
        for c in self.tg.get(label, []):
            if math.hypot(c["x"] - x, c["y"] - y) < 1.0:
                c["x"], c["y"], c["ver"], c["miss"] = x, y, True, 0
                c["n"] = max(c["n"], 8)
                return
        self.tg.setdefault(label, []).append(dict(x=x, y=y, w=4.0, n=8, t=0.0, miss=0, ver=True))

    def miss(self, label, c, hard=False):
        """A visit to this remembered spot found nothing. First time: keep it but try other spots first.
        Second time (or unreachable): it has moved or was a false alarm - forget it."""
        c["miss"] = c.get("miss", 0) + (2 if hard else 1)
        if c["miss"] >= 2:
            self.forget(label, c["x"], c["y"])
            return True
        return False

    def reset_objects(self):
        self.tg, self.sus, self.neg, self.bad = {}, {}, {}, []

    def reject(self, label, x, y):
        """Remember a FALSE ALARM: something that looked like `label` from afar but failed the close-up check."""
        for c in self.sus.setdefault(label, []):                      # one failed look proves nothing: need 3 near the same spot
            if math.hypot(c[0] - x, c[1] - y) < 1.0:
                c[2] += 1
                if c[2] >= 3:
                    self.neg.setdefault(label, []).append((c[0], c[1]))
                    self.forget(label, c[0], c[1])
                return
        self.sus[label].append([x, y, 1])

    def rejected(self, label, x, y):
        return any(math.hypot(x - a, y - b) < 1.0 for a, b in self.neg.get(label, []))

    def forget(self, label, x, y):
        self.tg[label] = [c for c in self.tg.get(label, []) if math.hypot(c["x"] - x, c["y"] - y) > .8]

    def new_sweep(self):
        self.seen[:] = False
        self.sweep += 1
        self.bad = []

    # ---- clearance + path planning ---------------------------------------------------------
    def clearance(self):
        """Metres from every cell to the nearest SONAR-confirmed obstacle (distance transform).
        Only these cells are forbidden to plan through."""
        free = (self.lo <= 0.6).astype(np.uint8)
        return cv2.distanceTransform(free, cv2.DIST_L2, 3) * MRES

    def clearance_any(self):
        """Like clearance() but ALSO counts camera-only hazards. Used where the body swings round (spinning in place):
        a plant the sonar passes through can still hit the car."""
        free = ((self.lo <= 0.6) & (self.vis <= 0.6)).astype(np.uint8)
        return cv2.distanceTransform(free, cv2.DIST_L2, 3) * MRES

    def vision_penalty(self):
        """Camera-only hazards are not trusted enough to forbid a route (a tabletop looks solid but is not), but the
        planner pays extra to stay away from them: cost multiplier 1 (far) .. 3 (inside)."""
        vo = (self.vis > .6) & (self.sfree < 2) & (self.lo <= .6)
        if not vo.any():
            return np.ones((MN, MN), np.float32)
        d = cv2.distanceTransform((~vo).astype(np.uint8), cv2.DIST_L2, 3) * MRES
        return (1.0 + 2.0 * np.clip((.45 - d) / .45, 0, 1)).astype(np.float32)

    def plan_from(self, x, y, hard=None):
        """Dijkstra cost-to-go field from the robot (used to choose WHERE to go; A* then plans the actual route)."""
        N = MN
        hard = SAFE.hard if hard is None else hard
        clear = self.clearance()
        sr, sc = self.cell(x, y)
        hard = self.relaxed(hard, clear[sr, sc])
        blocked = clear < hard
        near = (clear < hard + SAFE.soft_extra) & ~blocked
        blocked[max(0, sr - 1):sr + 2, max(0, sc - 1):sc + 2] = False
        bl, nr = blocked.ravel().tolist(), near.ravel().tolist()
        pen = self.vision_penalty().ravel().tolist()
        dist = [1e9] * (N * N)
        prev = [-1] * (N * N)
        s = sr * N + sc
        dist[s] = 0.0
        pq = [(0.0, s)]
        nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            r, c = divmod(u, N)
            for dr, dc, w in nb:
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    v = rr * N + cc
                    if bl[v]:
                        continue
                    if dr and dc and (bl[r * N + cc] or bl[rr * N + c]):      # same no-corner-cutting rule as A*
                        continue
                    nd = d + w * (1.8 if nr[v] else 1.0) * pen[v]
                    if nd < dist[v]:
                        dist[v], prev[v] = nd, u
                        heapq.heappush(pq, (nd, v))
        self.dist = np.array(dist, np.float32).reshape(N, N) * MRES
        self.prev = prev
        self.blocked = blocked

    def path_to(self, r, c):
        N, out, u = MN, [], r * MN + c
        if self.dist[r, c] >= 1e8:
            return None
        while u != -1:
            out.append(self.centre(*divmod(u, N)))
            u = self.prev[u]
        return out[::-1]

    H8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]      # (d_col, d_row); heading h = h * 45 deg

    def astar(self, start, goal, yaw, hard=None, turn_rate=.9, turn_fixed=.12):
        """A* over (cell, heading). The cost is TIME in seconds:
             driving time  = distance / speed, where speed drops as the route gets closer to obstacles
           + turning time  = angle / turn_rate (+ a small fixed cost per turn, so straighter routes win)
           + 25 % extra for cells nobody has looked at yet.
        Cells closer to an obstacle than `hard` are forbidden. Returns a list of (x, y) or None."""
        N = MN
        hard = SAFE.hard if hard is None else hard
        clear = self.clearance()
        sr, sc = self.cell(*start)
        gr, gc = self.cell(*goal)
        hard = self._hard_eff = self.relaxed(hard, clear[sr, sc])
        soft = hard + SAFE.soft_extra
        blocked = clear < hard
        blocked[max(0, sr - 1):sr + 2, max(0, sc - 1):sc + 2] = False
        blocked[gr, gc] = False
        known = (self.lo < -.05) | self.seen
        spd = np.clip(V_MAX * (.3 + .7 * (clear - hard) / max(soft - hard, 1e-3)), .15, V_MAX)
        tc = (np.where(known, MRES / spd, 1.25 * MRES / spd) * self.vision_penalty()).ravel().tolist()
        bl = blocked.ravel().tolist()
        h0 = int(round(yaw / (math.pi / 4))) % 8
        vmax_inv = MRES / V_MAX

        def hf(r, c):
            dx, dy = abs(c - gc), abs(r - gr)
            return (max(dx, dy) + .414 * min(dx, dy)) * vmax_inv

        s0 = (sr * N + sc) * 8 + h0
        g, parent = {s0: 0.0}, {s0: -1}
        pq = [(hf(sr, sc), 0.0, s0)]
        goal_state = None
        while pq:
            f, gg, s = heapq.heappop(pq)
            if gg > g.get(s, 1e9):
                continue
            cell, h = divmod(s, 8)
            r, c = divmod(cell, N)
            if r == gr and c == gc:
                goal_state = s
                break
            for h2 in range(8):
                dc, dr = self.H8[h2]
                rr, cc = r + dr, c + dc
                if not (0 <= rr < N and 0 <= cc < N):
                    continue
                v = rr * N + cc
                if bl[v]:
                    continue
                diag = dc != 0 and dr != 0
                if diag and (bl[r * N + cc] or bl[rr * N + c]):            # never cut a corner
                    continue
                turn = abs((h2 - h + 4) % 8 - 4)
                ng = gg + (tc[cell] + tc[v]) * .5 * (1.414 if diag else 1.0) + \
                    (turn * (math.pi / 4) / turn_rate + turn_fixed if turn else 0.0)
                ns = v * 8 + h2
                if ng < g.get(ns, 1e9):
                    g[ns], parent[ns] = ng, s
                    heapq.heappush(pq, (ng + hf(rr, cc), ng, ns))
        if goal_state is None:
            return None
        cells, s = [], goal_state
        while s != -1:
            cells.append(self.centre(*divmod(s // 8, N)))
            s = parent[s]
        cells = cells[::-1]
        cells[0] = (float(start[0]), float(start[1]))
        return self.densify(self.smooth(cells, hard, clear))

    def line_clear(self, a, b, hard, clear):
        n = int(max(2, math.hypot(b[0] - a[0], b[1] - a[1]) / (MRES * .4)))
        xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
        ix = np.clip((xs + HALF) // MRES, 0, MN - 1).astype(int)
        iy = np.clip((ys + HALF) // MRES, 0, MN - 1).astype(int)
        return bool((clear[iy, ix] >= hard).all())

    def smooth(self, pts, hard, clear):
        """Pull the string tight: drop waypoints whenever a straight, clear line exists (fewer turns)."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_clear(pts[i], pts[j], hard, clear):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    @staticmethod
    def densify(pts, step=.3):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
            out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
        return out

    @staticmethod
    def relaxed(hard, clear_here):
        """Never forbid what we are already standing in: if the car already sits in a tighter spot than `hard`,
        allow cells at least as roomy as this one so it can always drive back OUT."""
        return min(hard, max(clear_here - .03, .08)) if clear_here < hard else hard

    @staticmethod
    def margins(escape=False):
        """Safety margins to try, biggest first. The smaller ones are only used when nothing else gets through.
        escape=True adds the bare body radius: ONLY for getting out of a pocket, never for everyday routes."""
        m = (SAFE.hard, .5 * (SAFE.hard + SAFE.robot_r + .03), SAFE.robot_r + .03)
        return m + (SAFE.robot_r,) if escape else m

    def route(self, start, yaw, goal, escape=False):
        """Best safe route. If the full safety margin leaves no way through, squeeze: retry with smaller margins,
        never below body radius + a little. Returns (path, margin_used) or (None, None)."""
        for m in self.margins(escape):
            p_ = self.astar(start, goal, yaw, hard=m)
            if p_:
                return p_, self._hard_eff
        return None, None

    # ---- where to look next
    def gain_at(self, rows, cols):
        """For each candidate cell: how much NOT-yet-searched floor could a 360-degree look from there reveal?"""
        occ = self.lo > 0.6                               # line of sight: sonar-confirmed walls only (camera hazards can be overhangs)
        iy = np.clip(rows[:, None, None] + self.ry[None], 0, MN - 1)
        ix = np.clip(cols[:, None, None] + self.rx[None], 0, MN - 1)
        hit = occ[iy, ix]
        los = ~np.concatenate([np.zeros_like(hit[:, :, :1]), np.maximum.accumulate(hit, axis=2)[:, :, :-1]], axis=2)
        new = los & ~self.seen[iy, ix] & ~hit
        return (new * self.rw[None, None, :]).sum(axis=(1, 2))

    def best_viewpoint(self, x, y, yaw=0.0, escape=False):
        """Pick where to go next: lots of unsearched floor visible from there, and not too far to drive."""
        for m in self.margins(escape):                    # try the comfortable margin first; squeeze only if it finds nothing
            self.plan_from(x, y, hard=m)
            rr, cc = np.meshgrid(np.arange(1, MN - 1, 3), np.arange(1, MN - 1, 3), indexing="ij")
            rr, cc = rr.ravel(), cc.ravel()
            ok = self.dist[rr, cc] < 1e8
            for bx, by in self.bad:
                br, bc = self.cell(bx, by)
                ok &= np.hypot(rr - br, cc - bc) * MRES > .9
            rr, cc = rr[ok], cc[ok]
            if len(rr) == 0:
                continue
            g = self.gain_at(rr, cc)
            d = self.dist[rr, cc]
            score = g / (EXP_D0 + EXP_DK * d ** EXP_P)
            score[g < MIN_GAIN] = -1
            for i in np.argsort(-score)[:25]:             # best candidates first: the first one A* can really reach wins
                if score[i] <= 0:
                    break
                goal = self.centre(int(rr[i]), int(cc[i]))
                path = self.astar((x, y), goal, yaw, hard=m)
                if path:
                    return path, float(g[i]), self._hard_eff
        return None

    def nearest_roomy(self, x, y, yaw):
        """Closest reachable spot with room to spin the whole body (clearance > turning radius + a bit)."""
        self.plan_from(x, y)
        clear = self.clearance_any()
        ok = (clear >= SAFE.robot_r + .14) & (self.dist < 1e8)
        if not ok.any():
            return None
        r, c = np.unravel_index(int(np.argmin(np.where(ok, self.dist, 1e9))), ok.shape)
        goal = self.centre(int(r), int(c))
        path, m = self.route((x, y), yaw, goal)
        return (path, goal, m) if path else None

    def gain_here(self, x, y):
        r, c = self.cell(x, y)
        return float(self.gain_at(np.array([r]), np.array([c]))[0])

    def goal_near(self, x, y, tx, ty, lo_r=.7, hi_r=1.4, escape=False):
        """A reachable floor cell 0.7-1.4 m from a remembered object (we stop short of it, then look)."""
        self.plan_from(x, y, hard=SAFE.robot_r if escape else None)
        rr, cc = np.meshgrid(np.arange(1, MN - 1), np.arange(1, MN - 1), indexing="ij")
        dd = np.hypot((cc + .5) * MRES - HALF - tx, (rr + .5) * MRES - HALF - ty)
        reach = self.dist[rr, cc] < 1e8
        ok = (dd >= lo_r) & (dd <= hi_r) & reach
        if not ok.any():                                   # the ring is walled off: get as CLOSE as the map allows, then
            ok = (dd >= lo_r) & (dd <= 3.5) & reach        # the camera + sonar approach finishes the last stretch
            if not ok.any():
                return None
            r, c = np.unravel_index(int(np.argmin(np.where(ok, dd, 1e9))), dd.shape)
            return self.centre(int(r) + 1, int(c) + 1)
        cost = np.where(ok, self.dist[rr, cc], 1e9)
        r, c = np.unravel_index(int(np.argmin(cost)), cost.shape)
        return self.centre(int(r) + 1, int(c) + 1)           # (+1: the grids above start at cell 1)

    # ---- saving the map, vision hazards, reporting
    def update_vision(self, pose, hazards):
        """Label-free obstacles seen by the camera: [(bearing, distance), ...] -> evidence in the obstacle map."""
        x, y, yaw = pose
        cx, cy = x + CAM_FWD * math.cos(yaw), y + CAM_FWD * math.sin(yaw)
        for b, d in hazards:
            r, c = self.cell(cx + d * math.cos(yaw + b), cy + d * math.sin(yaw + b))
            self.vis[r, c] = min(3.0, self.vis[r, c] + .5)

    def save(self, path, room=None):
        """Write map + object locations + obstacles. Written to a temp name first so a crash never leaves a half file."""
        tmp = path + ".part.npz"
        np.savez_compressed(tmp, lo=self.lo, seen=self.seen, visits=self.visits, vis=self.vis, sfree=self.sfree,
                            meta=json.dumps(dict(tg=self.tg, neg=self.neg, sweep=self.sweep, room=room)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        wm = cls()
        wm.lo, wm.seen, wm.visits = z["lo"].astype(np.float32), z["seen"].astype(bool), z["visits"].astype(np.float32)
        if "vis" in z.files:
            wm.vis, wm.sfree = z["vis"].astype(np.float32), z["sfree"].astype(wm.sfree.dtype)
        meta = json.loads(str(z["meta"]))
        wm.room = meta.get("room")
        wm.tg, wm.neg, wm.sweep = meta["tg"], {k: [tuple(v) for v in vs] for k, vs in meta["neg"].items()}, meta["sweep"]
        return wm

    def summary(self):
        n, _ = cv2.connectedComponents(self.occ()[1:-1, 1:-1].astype(np.uint8), connectivity=8)
        objs = {k: [(round(c["x"], 2), round(c["y"], 2)) for c in v if c["n"] >= 3] for k, v in self.tg.items()}
        return dict(explored_pct=round(self.pct(), 1), obstacles=n - 1, objects={k: v for k, v in objs.items() if v})


# ----------------------------------------------------------------- navigation brain
class Brain:
    """SEARCH(spin) -> EXPLORE -> VERIFY -> APPROACH (gap-following obstacle avoidance) -> ARRIVED."""

    def __init__(self, standoff, wm=None, key=None, map_only=False):
        self.standoff, self.wm, self.key, self.map_only = standoff, wm, key, map_only   # wm=None -> old "legacy" explorer
        self.lat_est, self.det_period = 0.0, DT                    # measured vision delay / time between vision results
        self.reset()

    @property
    def spin_w(self):
        """How fast to spin while searching: slow enough that a can stays in view for ~4 vision results."""
        return float(np.clip(0.25 * math.radians(HFOV) / max(self.det_period, .05), .25, 1.0))

    def reset(self):
        self.path, self.goal, self.go_t, self.go_pos, self.blocked_n = [], None, 0.0, (0, 0), 0
        self.go_yaw, self.stuck_why = 0.0, ""
        self.replans, self.route_margin, self.replan_why, self.last_pt = 0, SAFE.hard, "", None
        self.rc_pending, self.recall_block, self.last_replan, self.spin_ok = None, -1.0, -99.0, False
        self.last_pick, self.look_t0, self.look_hold, self.recall_retry = -99.0, 0.0, None, 0
        self.state, self.note = "SEARCH", "starting"
        self.seen, self.lost, self.tracked, self.hist = 0, 999, False, []
        self.goal_yaw = None
        self.phase, self.spin_acc, self.prev_yaw = "SPIN", 0.0, None
        self.polar, self.visited, self.cov = {}, [], set()
        self.exp_yaw, self.exp_len, self.exp_start = 0.0, 0.0, (0, 0)
        self.est_dist, self.last_seen_t = None, -99.0
        self.detour_yaw, self.detour_until, self.last_d = 0.0, -1.0, 0.0
        self.ignore_until, self.best_d, self.prog_t = -1.0, 99.0, 0.0

    def gap_steer(self, goal, rng, mask=None):
        r = np.array(rng, dtype=float)
        a = np.array(ANGS)
        if mask:                                   # ignore range hits explained by the vision target itself
            ma, md = mask
            r[(np.abs(a - ma) < .36) & (r < md + .3)] = RMAX
        front = r[np.abs(a) <= .35].min()
        if front < 2 * SAFE.gap_min:
            side = r[a > 0].mean() - r[a < 0].mean()
            return -.18, 1.0 * (1 if side >= 0 else -1), "BACKUP"
        ok = [i for i in range(len(a)) if r[i] >= .7 and r[max(i - 1, 0)] >= .5 and r[min(i + 1, len(a) - 1)] >= .5]
        if not ok:
            side = r[a > 0].mean() - r[a < 0].mean()
            return 0.0, 1.2 * (1 if side >= 0 else -1), "BLOCKED"
        i = min(ok, key=lambda k: abs(a[k] - goal))
        d = a[i] if abs(a[i] - goal) >= .2 else goal
        d = float(np.clip(d, -1.2, 1.2))
        self.last_d = d
        v = V_MAX * max(0.0, 1 - abs(d) / 1.3) * float(np.clip(SAFE.v_safe(front) / V_MAX, .15, 1.0))   # fast enough only if we can still STOP in time
        push = 0.0                                  # side-clearance: the car's corners swing wider than the front ray
        for ai, ri in zip(a, r):
            lat, fwd = ri * abs(math.sin(ai)), ri * math.cos(ai)
            if -.05 < fwd < .55 and lat < .30 and abs(ai) > .3:
                push -= math.copysign((.30 - lat) / .20, ai)
        w = float(np.clip(2.0 * d + 1.2 * push, -W_MAX, W_MAX))
        if push:
            v *= .6
        return v, w, ("CLEAR" if d == goal and not push else "AVOID")

    def update(self, tdet, others, rng, pose, fresh=True, now=0.0):
        x, y, yaw = pose
        if self.state in ("ARRIVED", "MAPPED"):
            return 0.0, 0.0
        if self.map_only:                          # mapping mode: explore everything, chase nothing
            tdet = None
        if now < self.ignore_until:                # gave up on this approach for a while: explore, try another angle
            tdet, self.hist, self.tracked = None, [], False
        real = tdet is not None and not tdet.get("dr")     # "dr" = dead-reckoned from memory, not a new camera sighting
        if fresh:                                  # count only NEW vision results
            self.hist = (self.hist + [real])[-3:]
            self.seen = sum(self.hist)             # confirmed = 2 hits in the last 3 results (tolerates flicker)
            if real:
                self.lost, self.last_seen_t = 0, now
            else:
                self.lost += 1
        confirmed = tdet is not None and (self.seen >= 2 or tdet.get("dr", False))
        if confirmed:
            b, d = tdet["bearing"], tdet["dist"]
            self.tracked, self.goal_yaw, self.est_dist = True, wrap(yaw + b), d
            if d is not None:                      # patience: must keep getting closer
                if d < self.best_d - .25 or now - self.prog_t > 60:
                    self.best_d, self.prog_t = d, now
                elif now - self.prog_t > 15:
                    self.ignore_until, self.best_d, self.prog_t = now + 12.0, 99.0, now
                    self.recall_block, self.rc_pending = now + 15.0, None
                    self.phase, self.prev_yaw = "SPIN", None
                    self.note = "no progress: trying another angle"
                    return 0.0, 0.0
            if d is not None and d <= self.standoff:
                if abs(b) < .12:
                    self.state, self.note = "ARRIVED", f"stopped {d:.2f} m from target"
                    return 0.0, 0.0
                self.state, self.note = "ALIGN", "centering on target"
                return 0.0, float(np.sign(b) * max(.3, min(1.0, 1.8 * abs(b))))
            use = b                                # direction to steer toward (target, or a committed detour)
            if self.detour_until > now:
                ahead = rng[int(np.argmin(np.abs(np.array(ANGS) - float(np.clip(b, -1.4, 1.4)))))]
                if ahead < 1.0:
                    use = wrap(self.detour_yaw - yaw)      # still blocked toward the target: keep going around
                else:
                    self.detour_until = -1.0
            if abs(use) > 1.2:                    # target far off to the side / behind: turn on the spot
                self.state, self.note = "ALIGN", f"turning to target ({math.degrees(b):+.0f} deg)"
                return 0.0, float(np.sign(use) * max(.5, min(1.4, 1.5 * abs(use))))
            v, w, tag = self.gap_steer(use, rng, (b, d) if d else None)
            if tag == "AVOID" and use == b:           # new obstacle between us and the target: commit to this detour
                self.detour_yaw, self.detour_until = wrap(yaw + self.last_d), now + 1.5
            if d:
                v *= float(np.clip((d - self.standoff) / (.6 + V_MAX * self.lat_est), .25, 1))   # brake earlier if vision is slow
            self.state = "APPROACH" if tag == "CLEAR" else "AVOID"
            self.note = f"target {d:.2f} m away" if d else "target in view"
            return v, w
        if tdet is not None:
            self.state, self.note = "VERIFY", "confirming detection"
            return 0.0, 0.0
        if self.tracked and now - self.last_seen_t <= 4.0:       # target hidden behind something: head to last known bearing
            g = wrap(self.goal_yaw - yaw)
            self.state, self.note = "REACQUIRE", "target lost, heading to last seen spot"
            if self.detour_until > now:
                ahead = rng[int(np.argmin(np.abs(np.array(ANGS) - float(np.clip(g, -1.4, 1.4)))))]
                g = wrap(self.detour_yaw - yaw) if ahead < 1.0 else g
            if abs(g) > 1.2:
                return 0.0, float(np.sign(g) * .8)
            v, w, _ = self.gap_steer(g, rng)
            return v, w
        self.tracked = False
        if self.wm is None:
            return self._search(rng, pose)
        return self._search_map(rng, pose, now)

    # ---- map-based exploration / route execution ---------------------------------------------
    def _set_route(self, path, goal, margin, now, pose):
        self.path, self.goal, self.route_margin = path, goal, margin
        self.go_t, self.go_pos, self.blocked_n, self.last_replan, self.last_pt = now, pose[:2], 0, now, tuple(pose[:2])
        self.go_yaw = pose[2]

    def _replan(self, pose, now, why):
        """New A* route to the SAME goal from where the car is now, starting with its current heading (no needless U-turn)."""
        x, y, yaw = pose
        path, m = self.wm.route((x, y), yaw, self.goal)
        self.last_replan = now
        if path is None:
            return False
        self.path, self.route_margin, self.replans, self.replan_why = path, m, self.replans + 1, why
        self.go_t, self.go_pos, self.last_pt, self.go_yaw = now, (x, y), (x, y), yaw
        return True

    @staticmethod
    def _cross_track(x, y, pts):
        best = 1e9
        for a, b in zip(pts, pts[1:]):
            vx, vy = b[0] - a[0], b[1] - a[1]
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - a[0]) * vx + (y - a[1]) * vy) / L2))
            best = min(best, math.hypot(x - a[0] - t * vx, y - a[1] - t * vy))
        return best if len(pts) > 1 else math.hypot(x - pts[0][0], y - pts[0][1])

    def _follow(self, pose, rng, now, label):
        """Drive along self.path while watching the map. Returns (v, w, status): 'go' | 'done' | 'stuck'."""
        x, y, yaw = pose
        pts, wm = self.path, self.wm
        while len(pts) > 1 and math.hypot(pts[0][0] - x, pts[0][1] - y) < .35:
            self.last_pt = pts.pop(0)                       # remember the waypoint we just passed (the route line starts there)
        if not pts or (len(pts) == 1 and math.hypot(pts[0][0] - x, pts[0][1] - y) < .3):
            return 0.0, 0.0, "done"
        if now - self.last_replan > .5:                     # ---- continuous route monitoring
            clear, why = wm.clearance(), None
            for q in pts[:12]:
                if math.hypot(q[0] - x, q[1] - y) > .5:
                    r_, c_ = wm.cell(*q)
                    if clear[r_, c_] < .9 * self.route_margin:
                        why = "obstacle on the route"
                        break
            if why is None and self._cross_track(x, y, ([self.last_pt] if self.last_pt else []) + pts[:12]) > .5:
                why = "left the planned route"
            if why:
                if self._replan(pose, now, why):
                    pts = self.path
                else:
                    self.stuck_why = f"no route after: {why}"
                    return 0.0, 0.0, "stuck"
        if now - self.go_t > 4.0:                           # progress watchdog
            if math.hypot(x - self.go_pos[0], y - self.go_pos[1]) < .25 and abs(wrap(yaw - self.go_yaw)) < .6:
                self.stuck_why = "no progress for 4 s"
                return 0.0, 0.0, "stuck"                    # (turning in place IS progress; only no move AND no turn is stuck)
            self.go_t, self.go_pos, self.go_yaw = now, (x, y), yaw
        look = next((q for q in pts if math.hypot(q[0] - x, q[1] - y) >= .6), pts[-1])
        g = wrap(math.atan2(look[1] - y, look[0] - x) - yaw)
        self.state = "EXPLORE"
        if abs(g) > 1.0:
            self.note = f"{label}: turning toward next waypoint"
            return 0.0, float(np.sign(g) * max(.5, min(1.4, 1.5 * abs(g)))), "go"
        v, w, tag = self.gap_steer(g, rng)
        self.blocked_n = self.blocked_n + 1 if tag in ("BLOCKED", "BACKUP") else 0
        if self.blocked_n > 6:
            self.blocked_n = 0
            if not self._replan(pose, now, "blocked"):
                self.stuck_why = "blocked, no alternative route"
                return 0.0, 0.0, "stuck"
        left = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:]))
        self.note = f"{label}: {left:.1f} m to go | explored {wm.pct():.0f}% | replans {self.replans}"
        return v, w, "go"

    def _search_map(self, rng, pose, now):
        x, y, yaw = pose
        wm = self.wm
        # ---- the map remembers where this object was: go there by the best route (no re-scan)
        if (self.key and not self.map_only and now >= self.recall_block and self.phase in ("PLAN", "GO")
                and now - self.last_pick > (0.0 if self.phase == "PLAN" else 3.0)):
            self.last_pick = now
            pick = wm.pick_recall(self.key, x, y, yaw) or wm.pick_recall(self.key, x, y, yaw, escape=True)
            if pick:
                c, goal, path, m = pick
                self.rc_pending, self.phase = c, "RECALL"
                self._set_route(path, goal, m, now, pose)
        if self.phase == "RECALL":
            c = self.rc_pending
            if c is None or c not in wm.tg.get(self.key, []):       # memory changed under us (merged / forgotten)
                self.rc_pending, self.phase = None, "PLAN"
                return 0.0, 0.0
            v, w, st = self._follow(pose, rng, now, "going to where I saw it")
            self.state = "RECALL"
            if st == "go":
                return v, w
            if st == "stuck":
                if self.stuck_why.startswith("no route") and self.recall_retry < 3:
                    self.recall_retry += 1                          # the map changed under the route (a new obstacle, a noisy cell):
                    self.rc_pending, self.phase, self.last_pick = None, "PLAN", -99.0   # re-plan from HERE, maybe to another spot near it
                    return 0.0, 0.0
                wm.miss(self.key, c)                                # genuinely cannot get there: try other remembered spots first
                self.rc_pending, self.phase = None, "PLAN"
                return 0.0, 0.0
            self.phase, self.look_t0 = "LOOK", now                  # arrived
        if self.phase == "LOOK":                                    # face the remembered spot and give the detector a moment
            c = self.rc_pending
            e = wrap(math.atan2(c["y"] - y, c["x"] - x) - yaw)
            self.state, self.note = "SEARCH", "arrived: looking at where I saw it"
            if abs(e) > .12 and now - self.look_t0 < 4.0:
                self.look_hold = None
                return 0.0, float(np.sign(e) * max(.3, min(1.2, 1.8 * abs(e))))
            if self.look_hold is None:
                self.look_hold = now
            if now - self.look_hold < 0.8 + 3 * self.det_period + self.lat_est:
                return 0.0, 0.0
            self.look_hold = None
            self.phase, self.prev_yaw = "SPIN", None
        if self.phase == "SPIN":
            if self.prev_yaw is None and not self.spin_ok:
                r_, c_ = wm.cell(x, y)
                if wm.clearance_any()[r_, c_] < SAFE.robot_r + .10 or min(rng) < .15:   # turning here would swing the body into something
                    fr = wm.nearest_roomy(x, y, yaw)
                    if fr:
                        self._set_route(fr[0], fr[1], fr[2], now, pose)
                        self.phase = "FREE"
                    else:
                        wm.bad.append((x, y))
                        self.phase = "PLAN"
                        return 0.0, 0.0
            self.spin_ok = False
        if self.phase == "FREE":
            v, w, st = self._follow(pose, rng, now, "moving to a spot with room to turn")
            if st == "go":
                return v, w
            self.spin_ok = st == "done"
            self.phase = "SPIN" if st == "done" else "PLAN"
            if st != "done":
                wm.bad.append((x, y))
            self.prev_yaw = None
            return 0.0, 0.0
        if self.phase == "SPIN":
            if self.prev_yaw is None:
                self.prev_yaw, self.spin_acc = yaw, 0.0
            self.spin_acc += abs(wrap(yaw - self.prev_yaw))
            self.prev_yaw = yaw
            self.state, self.note = "SEARCH", f"scanning 360 deg ({min(100, int(100 * self.spin_acc / (2 * math.pi)))}%) | explored {wm.pct():.0f}%"
            if self.spin_acc < 2 * math.pi:
                return 0.0, self.spin_w
            wm.bad.append((x, y))                           # looked all around from here: don't pick this spot again
            if self.rc_pending:                             # looked all around and it is not there
                if wm.miss(self.key, self.rc_pending):      # second failed visit: it was moved / false alarm
                    wm.new_sweep()                          # old "already searched" knowledge is stale now
                self.rc_pending = None
            self.phase, self.prev_yaw = "PLAN", None
        if self.phase == "PLAN":
            self.last_replan = now
            best = wm.best_viewpoint(x, y, yaw)
            if best is None and wm.pct() < 85:               # boxed in? allow the tightest margin, just to get out
                best = wm.best_viewpoint(x, y, yaw, escape=True)
            if best is None:
                if self.map_only:
                    self.state, self.note = "MAPPED", f"room mapped: {wm.pct():.0f}% looked at"
                    return 0.0, 0.0
                wm.new_sweep()                              # everything reachable was looked at once: sweep again
                self.state, self.note = "SEARCH", f"whole room searched, starting sweep {wm.sweep}"
                self.phase, self.prev_yaw = "SPIN", None
                return 0.0, 0.0
            path, gain, m = best
            if len(path) < 3:                               # the best place to look from is right here
                self.phase, self.prev_yaw = "SPIN", None
                return 0.0, 0.0
            self._set_route(path, path[-1], m, now, pose)
            self.phase = "GO"
        v, w, st = self._follow(pose, rng, now, "exploring new area")
        if st == "go":
            return v, w
        if st == "done":
            self.phase, self.prev_yaw = ("SPIN", None) if wm.gain_here(x, y) >= SPIN_GAIN else ("PLAN", None)
        else:                                               # stuck
            wm.bad.append(self.goal)
            self.phase = "PLAN"
        return 0.0, 0.0

    def _search(self, rng, pose):
        x, y, yaw = pose
        if self.phase == "SPIN":
            if self.prev_yaw is None:
                self.prev_yaw, self.spin_acc, self.polar = yaw, 0.0, {}
                self.visited.append((x, y))
            self.spin_acc += abs(wrap(yaw - self.prev_yaw))
            self.prev_yaw = yaw
            for a, r in zip(ANGS, rng):
                self.polar[int(((yaw + a) % (2 * math.pi)) / (math.pi / 18))] = r
            self.state, self.note = "SEARCH", f"scanning 360 deg ({min(100, int(100 * self.spin_acc / (2 * math.pi)))}%)"
            if self.spin_acc >= 2 * math.pi:
                self._pick_heading(x, y)
                self.phase = "TURN"
            return 0.0, self.spin_w
        if self.phase == "TURN":
            e = wrap(self.exp_yaw - yaw)
            self.state, self.note = "EXPLORE", "turning toward open space"
            if abs(e) < .12:
                self.phase, self.exp_start = "GO", (x, y)
            return 0.0, float(np.sign(e) * max(.35, min(1.2, 1.8 * abs(e))))
        moved = math.hypot(x - self.exp_start[0], y - self.exp_start[1])      # GO
        v, w, tag = self.gap_steer(0.0, rng)
        self.state, self.note = "EXPLORE", f"moving to new viewpoint ({moved:.1f}/{self.exp_len:.1f} m)"
        if moved >= self.exp_len or tag in ("BLOCKED", "BACKUP"):
            self.phase, self.prev_yaw = "SPIN", None
        return v, w

    def _mark_seen(self, x, y):
        """Remember which parts of the room we have already looked at (360-degree scan, limited by walls)."""
        for b in range(36):
            ang, reach = (b + .5) * math.pi / 18, max(1.0, min(self.polar.get(b, 0.0), 3.5))
            sdist = 0.0
            while sdist <= reach:
                self.cov.add((int((x + sdist * math.cos(ang)) // CELL), int((y + sdist * math.sin(ang)) // CELL)))
                sdist += .4

    def _pick_heading(self, x, y):
        """Head toward open space that we have NOT looked at yet."""
        self._mark_seen(x, y)
        n, best, best_s, fallback = 36, None, -99, (0, -1)
        for b in range(n):
            clear = min(self.polar.get((b + k) % n, 0.0) for k in (-2, -1, 0, 1, 2))
            if clear > fallback[1]:
                fallback = (b, clear)
            if clear < .9:
                continue
            ang = (b + .5) * math.pi / 18
            L = float(np.clip(clear - .5, .4, 2.5))
            ex, ey = x + math.cos(ang) * L, y + math.sin(ang) * L
            pts = [(ex, ey)] + [(ex + 1.3 * math.cos(a2), ey + 1.3 * math.sin(a2)) for a2 in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
            pts = [q for q in pts if abs(q[0]) < HALF - .3 and abs(q[1]) < HALF - .3]
            unseen = sum((int(q[0] // CELL), int(q[1] // CELL)) not in self.cov for q in pts) / max(1, len(pts))
            pen = 1.0 if any(math.hypot(ex - vx, ey - vy) < 1.0 for vx, vy in self.visited) else 0.0
            sc = 2.0 * unseen + .5 * min(clear, 3.5) / 3.5 - pen + random.random() * .05
            if sc > best_s:
                best, best_s = (b, L), sc
        if best is None:
            best = (fallback[0], float(np.clip(fallback[1] - .5, .4, 2.5)))
        self.exp_yaw = wrap((best[0] + .5) * math.pi / 18)
        self.exp_len = best[1]


# ----------------------------------------------------------------- mission glue
def plausible(d, key, loc, H):
    """Size sanity check from monocular geometry: a can must be can-sized, a person person-sized."""
    dist = loc["dist"]
    if dist is None or d.y2 >= H - 2:               # distance unknown / box cut by the image edge: can't judge
        return True
    height_m = (d.y2 - d.y1) * dist / loc["f"]
    width_m = (d.x2 - d.x1) * dist / loc["f"]
    if key == "person":
        return d.y1 <= 2 or .5 <= height_m <= 1.4
    if width_m > .11:                       # a can is ~7 cm wide: a wider green/red blob is a plant, a box, ... (an edge-cut box is only wider in reality)
        return False
    return .04 <= height_m <= .20


def locate(d, W, H, f):
    cx = (d.x1 + d.x2) / 2
    bearing = -math.atan2(cx - W / 2, f)
    dy = d.y2 - H / 2
    dist = CAM_Z * f / dy if dy > 3 else None      # ground-plane range from the bottom edge of the box
    return {"bearing": bearing, "dist": dist, "conf": d.conf, "box": d, "f": f, "W": W}


class AsyncDetector:
    """Runs a slow detector (YOLO) on a background thread so the video/physics never wait for it."""

    def __init__(self, det):
        self.det, self.lock, self.ev = det, threading.Lock(), threading.Event()
        self.job, self.res, self.stamp = None, None, 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            self.ev.wait()
            self.ev.clear()
            with self.lock:
                job, self.job = self.job, None
            if job is None:
                continue
            frame, seg, pose, epoch, sim, t, ranges = job
            dets = self.det.detect(frame, seg, sim)
            with self.lock:
                self.stamp += 1
                self.res = (dets, pose, self.stamp, epoch, t, ranges)     # result is stamped with WHEN/WHERE its frame was taken

    def submit(self, frame, seg, pose, epoch, sim, t, ranges):
        with self.lock:
            if self.job is None:                   # newest frame wins; never queue up a backlog
                self.job = (frame, seg, pose, epoch, sim, t, ranges)
                self.ev.set()

    def latest(self, epoch):
        with self.lock:
            return self.res if self.res and self.res[3] == epoch else None


LABEL2KEY = {c["desc"]: k for k, c in CATALOG.items()}


class Mission:
    """Glue: camera + detector + map + brain + car.
    SYNC RULE: every vision result remembers the pose and time of the frame it came from. The target is stored as a
    point in the WORLD, and the bearing/distance the brain sees is recomputed from the car's CURRENT pose every tick.
    So a late result cannot point the car at where the target *used to be*."""

    def __init__(self, sim, detector, key, seed, max_t=300, use_async=False, latency=0.0, explore="map",
                 mode="find", load_map=None, hazard=True, store=None):
        self.sim, self.det, self.max_t, self.lat, self.explore = sim, detector, max_t, latency, explore
        self.mode, self.pending_map, self.hazard, self.haz_boxes = mode, load_map, hazard, []
        self.store, self.last_save = store, 0.0         # temporary on-robot storage: survives restarts, wiped by reset_map()
        self.asy = AsyncDetector(detector) if use_async else None
        self.epoch, self.seed = 0, seed
        self.start(key)

    def start(self, key=None, seed=None):
        if key:
            self.key = key
        if seed is not None:
            self.seed = seed
        self.sim.reset(self.seed)
        self.epoch += 1
        self.last_stamp, self.world_b, self.best_box = 0, None, None
        self.wm = None
        if self.explore == "map":
            self.wm = WorldMap.load(self.pending_map) if self.pending_map else WorldMap()
            if not self.pending_map and self.store and os.path.exists(self.store):
                try:
                    old = WorldMap.load(self.store)
                    if old.room == self.seed:               # only the SAME room's map is valid
                        self.wm = old
                except Exception:
                    pass
            self.wm.room = self.seed
            self.pending_map = None                         # only the first room can use a saved map
        self.brain = Brain(CATALOG[self.key]["standoff"], self.wm, self.key, map_only=self.mode == "map")
        self.outcome, self.dets, self.tdet, self.last_ranges = None, [], None, [RMAX] * len(ANGS)
        self.cmd = (0.0, 0.0)
        self.settle = 0
        self.pend, self.busy_until = None, 0.0
        self.tgt_xy, self.tgt_t, self.loc_dist, self.tgt_t_any = None, -99.0, None, -99.0
        self.run_len, self.last_run = 0, 0
        self.dets_pose, self.lat_ema, self.per_ema, self.last_arr = None, 0.0, DT, 0.0

    def reset_map(self):
        """Wipe everything the robot remembers about the room (map, obstacles, object locations) and its saved copy."""
        if self.explore != "map":
            return
        self.wm = WorldMap()
        self.wm.room = self.seed
        if self.store and os.path.exists(self.store):
            os.remove(self.store)
        self.brain = Brain(CATALOG[self.key]["standoff"], self.wm, self.key, map_only=self.mode == "map")
        self.outcome, self.best_box, self.world_b, self.tgt_xy = None, None, None, None

    def autosave(self, force=False):
        if self.store and self.wm is not None and (force or self.sim.t - self.last_save >= 5.0):
            self.last_save = self.sim.t
            try:
                self.wm.save(self.store, room=self.seed)
            except Exception:
                pass

    def retarget(self, key, mode="find"):
        self.key, self.mode = key, mode
        self.brain = Brain(CATALOG[key]["standoff"], self.wm, key, map_only=mode == "map")   # the MAP is kept: it may already know where the new target is
        self.brain.lat_est, self.brain.det_period = self.lat_ema, self.per_ema
        if self.wm is not None and self.wm.pct() > 5:
            self.brain.phase = "PLAN"                       # the map already knows the room: plan from it, do not spin first
        self.best_box, self.world_b, self.tgt_xy = None, None, None     # forget detections of the old target
        self.dets, self.pend, self.run_len, self.last_run = [], None, 0, 0
        self.outcome, self.settle = None, 0

    def _observe_all(self, cap_pose, H):
        """Put every can / person the detector saw into the map (world coordinates)."""
        s = self.sim
        for d in self.dets:
            k = LABEL2KEY.get(d.label)
            if k is None or d.conf < .3:
                continue
            loc = locate(d, s.W, s.H, s.f)
            if loc["dist"] is None or loc["dist"] > 5.0 or not plausible(d, k, loc, H):
                continue
            a = cap_pose[2] + loc["bearing"]
            cx, cy = cap_pose[0] + CAM_FWD * math.cos(cap_pose[2]), cap_pose[1] + CAM_FWD * math.sin(cap_pose[2])
            self.wm.observe(k, cx + loc["dist"] * math.cos(a), cy + loc["dist"] * math.sin(a), loc["dist"], s.t)

    def tick(self, dt=DT):
        s, desc = self.sim, CATALOG[self.key]["desc"]
        frame, seg = s.camera()
        pose = s.pose()
        ranges = s.ranges()
        if self.asy:                                # REAL background thread (live window)
            self.asy.submit(frame, seg, pose, self.epoch, s, s.t, ranges)
            r = self.asy.latest(self.epoch)
            fresh = r is not None and r[2] != self.last_stamp
            if fresh:
                self.dets, cap_pose, self.last_stamp, cap_t, cap_ranges = r[0], r[1], r[2], r[4], r[5]
        elif self.lat > 0:                          # EMULATED slow detector (benchmark / "what if this ran on a Raspberry Pi")
            if s.t >= self.busy_until:              # worker is free: grab this frame, result arrives `lat` seconds later
                self.pend = (s.t + self.lat, self.det.detect(frame, seg, s), pose, s.t, ranges)
                self.busy_until = s.t + self.lat
            fresh = self.pend is not None and s.t >= self.pend[0] - 1e-9
            if fresh:
                _, self.dets, cap_pose, cap_t, cap_ranges = self.pend
                self.pend = None
        else:
            self.dets, cap_pose, cap_t, cap_ranges, fresh = self.det.detect(frame, seg, s), pose, s.t, ranges, True
        if fresh:                                   # new vision result: pick the best target box
            self.lat_ema = .7 * self.lat_ema + .3 * (s.t - cap_t)                  # measured vision delay
            self.per_ema = .7 * self.per_ema + .3 * max(s.t - self.last_arr, .03)   # measured time between results
            self.last_arr, self.dets_pose = s.t, cap_pose
            if self.wm:
                self.wm.update_cover(cap_pose, cap_ranges)      # only what the detector really looked at counts as searched
                self._observe_all(cap_pose, s.H)
            cands = []
            for d in self.dets:
                if d.label != desc or d.conf < .2 or self.mode == "map":
                    continue
                loc = locate(d, s.W, s.H, s.f)
                wxy = None
                if loc["dist"] is not None:
                    a_ = cap_pose[2] + loc["bearing"]
                    wxy = (cap_pose[0] + CAM_FWD * math.cos(cap_pose[2]) + loc["dist"] * math.cos(a_),
                           cap_pose[1] + CAM_FWD * math.sin(cap_pose[2]) + loc["dist"] * math.sin(a_))
                if self.wm and wxy and self.wm.rejected(self.key, *wxy):
                    continue                                    # a spot we already proved is a false alarm
                if plausible(d, self.key, loc, s.H):
                    cands.append(d)
                elif self.wm and wxy and loc["dist"] < 1.5:     # close enough to judge, and it is NOT can-sized: remember that
                    self.wm.reject(self.key, *wxy)
            best = max(cands, key=lambda d: d.conf * (d.y2 - d.y1)) if cands else None
            self.best_box = best
            if best:
                loc = locate(best, s.W, s.H, s.f)
                self.world_b, self.loc_dist = wrap(cap_pose[2] + loc["bearing"]), loc["dist"]
                if s.t - self.tgt_t_any >= 2.0:                                         # a brand-new sighting, not a continuation
                    self.run_len = self.last_run = 0
                self.run_len += 1                                                       # consecutive vision results that saw it
                self.tgt_t_any = s.t
                if loc["dist"]:                       # remember WHERE (in the world) the target is, not where it was in the image
                    a = self.world_b
                    cx, cy = cap_pose[0] + CAM_FWD * math.cos(cap_pose[2]), cap_pose[1] + CAM_FWD * math.sin(cap_pose[2])
                    self.tgt_xy, self.tgt_t = (cx + loc["dist"] * math.cos(a), cy + loc["dist"] * math.sin(a)), s.t
                else:
                    self.tgt_xy = None
            else:
                self.world_b = None
                if self.run_len:                                                        # it just disappeared: remember how long we had it
                    self.last_run, self.run_len = self.run_len, 0
        # ---- what the brain gets: bearing/distance of the target from the CURRENT pose
        self.tdet = None
        cxn, cyn = pose[0] + CAM_FWD * math.cos(pose[2]), pose[1] + CAM_FWD * math.sin(pose[2])
        if self.best_box is not None and self.world_b is not None:
            if self.tgt_xy is not None:
                dx, dy = self.tgt_xy[0] - cxn, self.tgt_xy[1] - cyn
                self.tdet = {"bearing": wrap(math.atan2(dy, dx) - pose[2]), "dist": math.hypot(dx, dy),
                             "conf": self.best_box.conf, "box": self.best_box}
            else:                                   # bearing only (distance unknown)
                self.tdet = {"bearing": wrap(self.world_b - pose[2]), "dist": None,
                             "conf": self.best_box.conf, "box": self.best_box}
        elif (self.tgt_xy is not None and s.t - self.tgt_t < 2.5 and max(self.run_len, self.last_run) >= 3 and self.brain.tracked
              and self.brain.state in ("APPROACH", "AVOID", "ALIGN")):
            dx, dy = self.tgt_xy[0] - cxn, self.tgt_xy[1] - cyn
            if math.hypot(dx, dy) < 1.3:            # close-range hand-off: a can this near is cut off by the image edge,
                self.tdet = {"bearing": wrap(math.atan2(dy, dx) - pose[2]), "dist": math.hypot(dx, dy),   # so finish the
                             "conf": .5, "box": None, "dr": True}                                          # approach from memory
        others = [d for d in self.dets if d is not self.best_box]
        self.last_ranges = ranges
        if self.wm:
            self.wm.update_range(pose, ranges, dt)
        if self.hazard:                                     # label-free hazards: anything on the floor that is not floor
            haz = floor_hazards(frame, s.f)
            if self.wm:
                self.wm.update_vision(pose, [(b, d) for b, d, _ in haz])
            self.haz_boxes = [h for h in hazard_boxes(haz, s.f, s.H)
                              if not any(d.x1 < (h.x1 + h.x2) / 2 < d.x2 for d in self.dets)]     # ones the AI did not name
        self.brain.lat_est, self.brain.det_period = self.lat_ema, self.per_ema
        v, w = self.brain.update(self.tdet, others, ranges, pose, fresh, s.t)
        if self.outcome is None and self.key != "person" and any(
                o.label == "person" and o.y2 - o.y1 > .25 * s.H for o in others):
            v = min(v, .15)                          # a person is close: slow right down
        self.cmd = (v, w)
        s.step(v, w, dt)
        self.autosave()
        if self.outcome is None:
            if self.brain.state == "MAPPED":
                self.outcome = "MAPPED"
            elif self.brain.state == "ARRIVED":
                self.outcome = s.judge(self.key)
                if self.outcome == "SUCCESS" and self.wm is not None and self.tgt_xy is not None:
                    self.wm.verify(self.key, *self.tgt_xy)      # we stood next to it: this location is now certain
            elif s.t >= self.max_t:
                self.outcome = "TIMEOUT"
        return frame


# ----------------------------------------------------------------- dashboard
def draw_map(sim, mission, S=400):
    img = np.full((S, S, 3), 38, np.uint8)
    k = S / ROOM
    px = lambda x, y: (int(S / 2 + x * k), int(S / 2 - y * k))
    wm = mission.wm
    if wm is not None:                                  # the robot's MEMORY: lighter floor = already searched
        tint = np.zeros((MN, MN, 3), np.uint8)
        tint[wm.seen] = (30, 24, 0)
        img = cv2.add(img, cv2.resize(tint[::-1], (S, S), interpolation=cv2.INTER_NEAREST))
    cv2.rectangle(img, px(-HALF, HALF), px(HALF, -HALF), (70, 70, 70), 2)
    for o in sim.layout:
        t, c = o["type"], px(o["x"], o["y"])
        if t == "can":
            r_, g_, b_, _ = CATALOG[o["key"]]["rgba"]
            cv2.circle(img, c, 5, (int(b_ * 255), int(g_ * 255), int(r_ * 255)), -1)
            if o["key"] == mission.key:
                cv2.circle(img, c, 10, (255, 255, 255), 1)
        elif t == "person":
            cv2.circle(img, c, max(4, int(.17 * k)), (10, 115, 240), -1)
        else:
            r_, g_, b_, _ = o["col"]
            colr = (int(b_ * 255), int(g_ * 255), int(r_ * 255))
            if o["shape"] == "circle":
                cv2.circle(img, c, max(2, int(o["r"] * k)), colr, -1)
            else:
                cs, sn = math.cos(o["yaw"]), math.sin(o["yaw"])
                pts = [px(o["x"] + a_ * o["hx"] * cs - b2 * o["hy"] * sn, o["y"] + a_ * o["hx"] * sn + b2 * o["hy"] * cs)
                       for a_, b2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                if t == "table":
                    cv2.polylines(img, [np.array(pts, np.int32)], True, colr, 2)
                else:
                    cv2.fillPoly(img, [np.array(pts, np.int32)], colr)
    if wm is not None:
        rr, cc = np.nonzero(wm.occ()[1:-1, 1:-1])       # obstacles the robot remembers (red dots)
        for r_, c_ in zip(rr + 1, cc + 1):
            cv2.circle(img, px(*wm.centre(r_, c_)), 1, (60, 60, 230), -1)
        for k_, cl in wm.tg.items():                    # objects the robot remembers (diamonds)
            rgba = CATALOG[k_]["rgba"]
            col_ = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            for c_ in cl:
                if c_["n"] >= 3:
                    cv2.drawMarker(img, px(c_["x"], c_["y"]), col_, cv2.MARKER_DIAMOND, 14, 1)
        pts_ = mission.brain.path
        if len(pts_) > 1:
            cv2.polylines(img, [np.array([px(*q) for q in pts_], np.int32)], False, (255, 255, 0), 1)
    if mission.key == "person":
        px_, py_ = sim.items["person"]
        cv2.circle(img, px(px_, py_), 12, (255, 255, 255), 1)
    for vx, vy in mission.brain.visited:
        cv2.circle(img, px(vx, vy), 3, (255, 200, 0), 1)
    for a, b in zip(sim.trail, sim.trail[1:]):
        cv2.line(img, px(*a), px(*b), (150, 150, 150), 1)
    x, y, yaw = sim.pose()
    for a, r in zip(ANGS, mission.last_ranges):
        cv2.line(img, px(x, y), px(x + r * math.cos(yaw + a), y + r * math.sin(yaw + a)), (60, 90, 60), 1)
    pts = [(x + .22 * math.cos(yaw), y + .22 * math.sin(yaw)),
           (x + .14 * math.cos(yaw + 2.5), y + .14 * math.sin(yaw + 2.5)),
           (x + .14 * math.cos(yaw - 2.5), y + .14 * math.sin(yaw - 2.5))]
    cv2.fillPoly(img, [np.array([px(*q) for q in pts], np.int32)], (0, 210, 255))
    b = mission.brain
    if b.goal_yaw is not None and b.est_dist and b.state in ("APPROACH", "AVOID", "ALIGN", "ARRIVED"):
        gx, gy = x + b.est_dist * math.cos(b.goal_yaw), y + b.est_dist * math.sin(b.goal_yaw)
        cv2.drawMarker(img, px(gx, gy), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(img, "top-down map (god view, for judges)", (8, 18), FONT, .45, (200, 200, 200), 1)
    return img


def render_ui(sim, mission, frame, fps):
    cam = cv2.resize(frame, (640, 480))
    sx, sy = 640 / sim.W, 480 / sim.H
    desc = CATALOG[mission.key]["desc"]
    dyaw = wrap(sim.pose()[2] - mission.dets_pose[2]) if mission.dets_pose else 0.0    # how far we turned since those boxes were found
    for d in mission.dets:
        t = d.label == desc
        col = (0, 255, 0) if t else (0, 165, 255)
        sh = 0.0                                          # slide old boxes sideways so they sit on the object NOW
        if abs(dyaw) > 1e-3:
            b0 = -math.atan2((d.x1 + d.x2) / 2 - sim.W / 2, sim.f)
            sh = -sim.f * math.tan(max(-1.4, min(1.4, b0 - dyaw))) - ((d.x1 + d.x2) / 2 - sim.W / 2)
        x1, y1, x2, y2 = int((d.x1 + sh) * sx), int(d.y1 * sy), int((d.x2 + sh) * sx), int(d.y2 * sy)
        cv2.rectangle(cam, (x1, y1), (x2, y2), col, 3 if t else 1)
        txt = f"{d.label} {d.conf:.2f}"
        if t and mission.tdet and mission.tdet["dist"]:      # distance is re-computed from the CURRENT pose every tick
            txt += f"  {mission.tdet['dist']:.2f}m"
        cv2.putText(cam, txt, (x1, max(14, y1 - 5)), FONT, .5, col, 1)
    cv2.line(cam, (320, 0), (320, 480), (255, 255, 255), 1)
    cv2.rectangle(cam, (0, 0), (150, 28), (24, 24, 24), -1)
    cv2.putText(cam, "ROBOT CAMERA", (8, 20), FONT, .55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.full((570, 1040, 3), 24, np.uint8)
    canvas[:480, :640] = cam
    canvas[:400, 640:] = draw_map(sim, mission)
    b, (v, w) = mission.brain, mission.cmd
    lines = [f"state: {b.state}", f"v={v:+.2f} m/s", f"w={w:+.2f} rad/s", f"{fps:.0f} fps"]
    for i, ln in enumerate(lines):
        cv2.putText(canvas, ln, (650, 425 + 20 * i), FONT, .5, (255, 255, 255), 1)
    cv2.putText(canvas, f"target: {desc}", (800, 425), FONT, .5, (0, 255, 0), 1)
    cv2.putText(canvas, f"time {sim.t:5.1f}s  path {sim.path:4.1f} m", (800, 445), FONT, .5, (255, 255, 255), 1)
    cv2.putText(canvas, f"collisions {sim.collisions}  (person {sim.person_hits})", (800, 465), FONT, .5, (255, 255, 255), 1)
    cv2.putText(canvas, f"vision: {mission.det.name}", (800, 485), FONT, .45, (255, 220, 120), 1)
    if mission.wm is not None:
        cv2.putText(canvas, f"explored {mission.wm.pct():.0f}% | delay {mission.lat_ema * 1000:.0f} ms", (800, 505), FONT, .45, (160, 220, 160), 1)
    cv2.putText(canvas, b.note, (10, 505), FONT, .6, (255, 255, 255), 1)
    cv2.putText(canvas, "q quit | n new room | space pause | type a command in the terminal", (10, 555), FONT, .45, (150, 150, 150), 1)
    if mission.outcome:
        ok = mission.outcome == "SUCCESS"
        cv2.rectangle(canvas, (170, 36), (470, 86), (0, 120, 0) if ok else (0, 0, 160), -1)
        cv2.putText(canvas, "TARGET REACHED" if ok else mission.outcome, (182, 70), FONT, .9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ----------------------------------------------------------------- entry points
def make_detector(name, weights=WEIGHTS):
    if name == "auto":
        if os.path.exists(weights):
            name = "trained"
        else:
            print(f"[i] No trained model ({weights}) yet -> using the colour detector.")
            print("[i] For real learned vision run:  python robot_sim.py --collect 1500   then   python robot_sim.py --train\n")
            return ColorDetector()
    if name == "trained":
        try:
            return TrainedDetector(weights)
        except Exception as e:
            print(f"\n[!] Could not load {weights} ({type(e).__name__}: {str(e)[:100]}). Using colour detector.\n")
            return ColorDetector()
    if name in ("yoloworld", "hybrid"):
        try:
            return YoloWorldDetector() if name == "yoloworld" else HybridDetector()
        except Exception as e:                      # never fail silently: say it loudly
            print(f"\n[!] YOLO-World could not start ({type(e).__name__}: {str(e)[:120]})")
            print("[!] Falling back to the offline colour detector. The dashboard shows which vision is active.\n")
            return ColorDetector()
    return {"color": ColorDetector, "gt": GroundTruthDetector}[name]()


# ----------------------------------------------------------------- train your OWN detector from the simulator
def collect_dataset(n, out="dataset", res=(400, 300)):
    """Drive the camera around many random rooms; the segmentation mask gives perfect labels for free."""
    sim = Sim(res, noise=True)
    names = [c["desc"] for c in CATALOG.values()] + ["obstacle"]
    cid = {nm: i for i, nm in enumerate(names)}
    for sp in ("train", "val"):
        os.makedirs(f"{out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{out}/labels/{sp}", exist_ok=True)
    rng, k, ep = random.Random(12345), 0, 0
    while k < n:
        sim.reset(10000 + ep)
        ep += 1
        circs = [c for it in sim.layout for c in circles(it)]
        targets = list(sim.items.values())
        for _ in range(14):
            if k >= n:
                break
            for _try in range(60):
                x, y = rng.uniform(-HALF + .5, HALF - .5), rng.uniform(-HALF + .5, HALF - .5)
                if all(math.hypot(x - cx, y - cy) > cr + .3 for cx, cy, cr in circs):
                    break
            else:
                continue
            if rng.random() < .6:
                tx, ty = rng.choice(targets)
                yaw = math.atan2(ty - y, tx - x) + rng.uniform(-.5, .5)
            else:
                yaw = rng.uniform(-math.pi, math.pi)
            p.resetBasePositionAndOrientation(sim.car, [x, y, .07], p.getQuaternionFromEuler([0, 0, yaw]))
            bgr, seg = sim.camera([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(.5, 2)])
            labs = gt_boxes(sim, seg)
            if not labs:
                continue
            sp = "val" if k % 10 == 0 else "train"
            cv2.imwrite(f"{out}/images/{sp}/{k:06d}.jpg", bgr)
            with open(f"{out}/labels/{sp}/{k:06d}.txt", "w") as fh:
                for lab, (x1, y1, x2, y2) in labs:
                    fh.write(f"{cid[lab]} {(x1 + x2) / 2 / sim.W:.6f} {(y1 + y2) / 2 / sim.H:.6f} "
                             f"{(x2 - x1 + 1) / sim.W:.6f} {(y2 - y1 + 1) / sim.H:.6f}\n")
            k += 1
            if k % 100 == 0:
                print(f"  {k}/{n} images ({ep} rooms)", flush=True)
    with open(f"{out}/data.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for nm, i in cid.items():
            fh.write(f"  {i}: {nm}\n")
    print(f"dataset ready: {k} images in ./{out}")


def train_detector(epochs=25, imgsz=416, batch=16, out="dataset", weights=WEIGHTS, fast=False):
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")                              # small COCO-pretrained starting point
    m.train(data=os.path.abspath(f"{out}/data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, workers=0,
            project=os.path.abspath("runs"), name="robot", exist_ok=True, plots=False, verbose=False,
            freeze=10 if fast else None)          # fast = keep the pretrained backbone fixed (much quicker on CPU)
    shutil.copy(str(m.trainer.best), weights)
    print(f"\nSaved {weights}. Now just run:  python robot_sim.py")


def run_bench(args, sim, det):
    rows, tkeys = [], ["red", "blue", "green", "person"]
    for i in range(args.bench):
        seed, key = args.seed + i, tkeys[i % 4] if args.target == "mix" else resolve_target(args.target)
        m = Mission(sim, det, key, seed, latency=args.latency, explore=args.explore)
        while m.outcome is None:
            m.tick()
        rows.append(dict(seed=seed, target=CATALOG[key]["desc"], outcome=m.outcome, time_s=round(sim.t, 1),
                         path_m=round(sim.path, 1), collisions=sim.collisions, person_hits=sim.person_hits))
        r = rows[-1]
        print(f"seed {seed:3d} | {r['target']:22s} | {r['outcome']:12s} | {r['time_s']:6.1f}s | {r['path_m']:5.1f} m | collisions {r['collisions']}")
    ok = [r for r in rows if r["outcome"] == "SUCCESS"]
    n = len(rows)
    print(f"\nvision: {det.name}\nsuccess {len(ok)}/{n} ({100 * len(ok) / n:.0f}%) | "
          f"wrong-target {sum(r['outcome'] == 'WRONG_TARGET' for r in rows)} | timeout {sum(r['outcome'] == 'TIMEOUT' for r in rows)} | "
          f"avg time {np.mean([r['time_s'] for r in ok]) if ok else 0:.1f}s | "
          f"collisions/episode {np.mean([r['collisions'] for r in rows]):.2f} | person contacts {sum(r['person_hits'] for r in rows)}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)
        print("saved", args.csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="red cola can", help='e.g. "coke can", "red bull can", "person" ("mix" for --bench)')
    ap.add_argument("--detector", choices=["auto", "trained", "yoloworld", "hybrid", "color", "gt"], default="auto",
                    help="auto = your trained model if it exists, else colour")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--res", default="400x300", help="camera size; bigger = sharper but slower")
    ap.add_argument("--noise", action="store_true", help="camera noise + brightness jitter")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--video", help="save the dashboard to an .mp4")
    ap.add_argument("--bench", type=int, default=0, help="run N random rooms and print a results table")
    ap.add_argument("--csv", help="save bench results")
    ap.add_argument("--max-time", type=float, default=300)
    ap.add_argument("--explore", choices=["map", "legacy"], default="map", help="map = remembers the room (default), legacy = old explorer")
    ap.add_argument("--mode", choices=["find", "map"], default="find", help="map = just map the whole room (no target), find = go find the target")
    ap.add_argument("--save-map", help="write the room map here when mapping finishes (and on key s)")
    ap.add_argument("--load-map", help="start with a map saved earlier: the robot then drives straight to remembered objects")
    ap.add_argument("--store", default=os.path.join(tempfile.gettempdir(), "robot_map.npz"),
                    help="temporary on-robot map storage (auto-saved every 5 s, re-loaded on restart for the same room)")
    ap.add_argument("--no-store", action="store_true", help="do not keep the map between runs")
    ap.add_argument("--reset-map", action="store_true", help="wipe the stored map before starting")
    ap.add_argument("--novel", action="store_true", help="add odd unfamiliar obstacles (pipe, snowman, sofa...) to test unknown-object avoidance")
    ap.add_argument("--no-hazard", action="store_true", help="turn off the label-free camera hazard layer (sonar only)")
    ap.add_argument("--margin", type=float, help="extra clearance (m) added to the hard safety margin")
    ap.add_argument("--soft-margin", type=float, help="extra clearance the planner prefers (default %.2f)" % SAFE.soft_extra)
    ap.add_argument("--react", type=float, help="reaction time (s) used for stopping distance")
    ap.add_argument("--decel", type=float, help="braking deceleration (m/s^2) used for stopping distance")
    ap.add_argument("--latency", type=float, default=0.0, help="--bench only: pretend the detector needs this many seconds per frame")
    ap.add_argument("--collect", type=int, default=0, help="make N auto-labelled training images from the simulator")
    ap.add_argument("--train", action="store_true", help="train your own detector on ./dataset")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fast", action="store_true", help="with --train: faster, slightly less accurate (good for CPU-only laptops)")
    args = ap.parse_args()

    if args.collect:
        return collect_dataset(args.collect, res=tuple(map(int, args.res.lower().split("x"))))
    if args.train:
        return train_detector(args.epochs, fast=args.fast)
    key = resolve_target(args.target) if args.target != "mix" else "red"
    if key is None:
        sys.exit(f"Unknown target '{args.target}'. Try: " + ", ".join(c["desc"] for c in CATALOG.values()))
    W, H = map(int, args.res.lower().split("x"))
    if args.margin is not None:
        SAFE.extra = args.margin
    if args.soft_margin is not None:
        SAFE.soft_extra = args.soft_margin
    if args.react is not None:
        SAFE.react = args.react
    if args.decel is not None:
        SAFE.decel = args.decel
    sim, det = Sim((W, H), args.noise, args.novel), make_detector(args.detector)
    if args.bench:
        return run_bench(args, sim, det)

    slow = isinstance(det, (YoloWorldDetector, HybridDetector, TrainedDetector))
    live = not args.headless
    m = Mission(sim, det, key, args.seed, args.max_time, use_async=slow and live, explore=args.explore,
                mode=args.mode, load_map=args.load_map, hazard=not args.no_hazard,
                store=None if args.no_store else args.store)
    if args.reset_map:
        m.reset_map()
    saved = {"done": False}

    def save_map():
        if m.wm is not None and args.save_map:
            m.wm.save(args.save_map)
            print("map saved to", args.save_map)
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (1040, 570)) if args.video else None
    pending = {"cmd": None}
    if not args.headless:
        def reader():
            print("Type a command, e.g.  go to the red bull can   |   find the person")
            while True:
                try:
                    pending["cmd"] = input("command> ")
                except EOFError:
                    return
        threading.Thread(target=reader, daemon=True).start()
    print(f"Target: {CATALOG[key]['desc']} | vision: {det.name} | seed {args.seed}")
    fps, paused, last_state, settle, t_prev = 10.0, False, "", 0, time.time()
    while True:
        t0 = time.time()
        dt = float(np.clip(t0 - t_prev, .03, .12)) if live else DT      # live window = real time
        t_prev = t0
        if pending["cmd"]:
            txt = pending["cmd"].lower()
            k = resolve_target(txt)
            pending["cmd"] = None
            if re.search(r"\b(reset|forget|clear|wipe)\b", txt) and re.search(r"\b(map|room|memory)\b", txt):
                m.reset_map()
                print("Map wiped. The robot knows nothing about the room now.")
            elif re.search(r"\b(map|explore|scan)\b", txt) and not k:
                m.retarget(m.key, mode="map")
                print("Mapping the whole room...")
            elif k:
                m.retarget(k)
                print("New target:", CATALOG[k]["desc"], "(it goes straight there if the map already knows where it is)")
            else:
                print("Unknown. Try:", ", ".join(c["desc"] for c in CATALOG.values()))
        if not paused:
            frame = m.tick(dt)
            fps = .9 * fps + .1 / max(time.time() - t0, 1e-3)
        ui = render_ui(sim, m, frame, fps)
        if writer:
            writer.write(ui)
        if m.brain.state != last_state:
            print(f"[{sim.t:6.1f}s] {m.brain.state}: {m.brain.note}")
            last_state = m.brain.state
        if m.outcome:
            settle += 1
            if settle == 1:
                m.autosave(force=True)
                if m.outcome == "MAPPED":
                    print("MAP SUMMARY:", m.wm.summary())
                    if args.save_map and not saved["done"]:
                        save_map()
                        saved["done"] = True
                print(f"RESULT: {m.outcome} | time {sim.t:.1f}s | path {sim.path:.1f} m | collisions {sim.collisions} {sim.events}")
            if args.headless and settle > 15:
                break
        else:
            settle = 0
        if not args.headless:
            cv2.imshow("Vision-guided robot - Hack in Hills 26", ui)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("s"):
                save_map()
                m.autosave(force=True)
            if k == ord("r"):
                m.reset_map()
                print("Map wiped (key r)")
            if k == ord("o"):                       # drop an unknown obstacle right in front of the car
                x, y, yaw = sim.pose()
                sim.add_obstacle(x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw), "crate")
                print("Dropped an obstacle 1 m ahead")
            if k == ord("t"):                       # move the target somewhere else
                spot = sim.free_spot(np.random.default_rng())
                if spot and m.key in sim.items:
                    sim.move_item(m.key, *spot)
                    print("Moved the target")
            if k == ord("n"):
                m.start(seed=m.seed + 1)
                last_state = ""
                print(f"New room, seed {m.seed}")
        if live:                                    # hold ~10 fps: a loop that never rests starves the AI thread (it got 4x slower!)
            time.sleep(max(0.0, DT - (time.time() - t0)))
    if writer:
        writer.release()
        print("saved", args.video)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
