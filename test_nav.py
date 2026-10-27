"""Focused regression tests for robot_sim.py navigation.   Run:  python test_nav.py   (or: pytest -q test_nav.py)
They use the real PyBullet sim, the real WorldMap / Brain, and a ground-truth detector, so they need no camera model.
Set ROBOT_SIM=/path/to/other_version.py to run the same tests against another version of the file."""
import importlib.util, math, os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rs", os.environ.get("ROBOT_SIM", os.path.join(HERE, "robot_sim.py")))
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


# ------------------------------------------------------------------ helpers
def oracle_map(sim, objects=True):
    """A WorldMap that already knows every obstacle of the sim's current room (and optionally where the cans are)."""
    wm = rs.WorldMap()
    wm.lo[1:-1, 1:-1] = -2.0
    for it in sim.layout:
        for r in range(rs.MN):
            for c in range(rs.MN):
                x, y = wm.centre(r, c)
                if it["shape"] == "circle":
                    inside = math.hypot(x - it["x"], y - it["y"]) <= it["r"] + .05
                else:                                              # exact rotated rectangle
                    dx, dy = x - it["x"], y - it["y"]
                    u = dx * math.cos(it["yaw"]) + dy * math.sin(it["yaw"])
                    v = -dx * math.sin(it["yaw"]) + dy * math.cos(it["yaw"])
                    inside = abs(u) <= it["hx"] + .05 and abs(v) <= it["hy"] + .05
                if inside:
                    wm.lo[r, c] = 4.0
    wm.seen[:] = True
    if objects:
        for k, (x, y) in sim.items.items():
            wm.tg[k] = [dict(x=x, y=y, w=4.0, n=8, t=0.0, miss=0, ver=False)]
    return wm


def route_len(p):
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(p, p[1:]))


def free_pair(wm, min_d=4.5, seed=0):
    clear = wm.clearance()
    rng = np.random.default_rng(seed)
    cells = np.argwhere(clear > rs.SAFE.hard + .2)
    for _ in range(4000):
        a, b = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        A, B = wm.centre(*a), wm.centre(*b)
        if math.hypot(A[0] - B[0], A[1] - B[1]) >= min_d:
            p = wm.astar(A, B, 0.0)
            if p and route_len(p) < 1.6 * math.hypot(A[0] - B[0], A[1] - B[1]) + .5:
                return A, B
    raise RuntimeError("no start/goal pair")


# ------------------------------------------------------------------ 1. exploration stops only when nothing reachable is left
def _synthetic_room():
    wm = rs.WorldMap()
    wm.lo[1:-1, 1:-1] = -2.0

    def wall(x0, y0, x1, y1):
        for t in np.linspace(0, 1, 200):
            r, c = wm.cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            wm.lo[r, c] = 4.0
    wall(-4.4, 1.0, 1.2, 1.0)                    # a long partition with a doorway at the east end
    wall(2.6, -4.4, 2.6, -2.6)                   # a small alcove in the south-east corner with a narrow mouth
    wall(2.6, -2.6, 3.4, -2.6)
    wall(3.9, -2.6, 4.4, -2.6)
    return wm


def _look(wm, x, y, R=3.0):
    for a in np.linspace(0, 2 * math.pi, 180, endpoint=False):
        for d in np.arange(.1, R, rs.MRES * .5):
            r, c = wm.cell(x + d * math.cos(a), y + d * math.sin(a))
            if wm.lo[r, c] > .6:
                break
            wm.seen[r, c] = True


def explore_synthetic(max_iter=120):
    wm = _synthetic_room()
    pose, yaw, spots = (-3.5, -3.5), 0.0, []
    _look(wm, *pose)
    for it in range(max_iter):
        best = wm.best_viewpoint(pose[0], pose[1], yaw)
        if best is None:
            return wm, spots, it
        path = best[0]
        pose = path[-1]
        spots.append(pose)
        _look(wm, *pose)
        wm.bad.append(pose)
        wm.note_scan(*pose)
    return wm, spots, max_iter


def test_exploration_terminates_only_when_everything_reachable_was_searched():
    wm, spots, iters = explore_synthetic()
    assert iters < 120, "explorer never finished"
    wm.plan_from(*(spots[-1] if spots else (-3.5, -3.5)))
    left = wm.unsearched_reachable(rs.SAFE.hard)
    assert left.sum() == 0, f"{int(left.sum())} reachable unsearched cells left when exploration stopped"
    r, c = wm.cell(3.4, -3.6)                    # inside the alcove: tiny gain, the old MIN_GAIN cut-off skipped it
    assert wm.seen[r, c], "the small alcove was never explored"
    free = (wm.lo <= .6)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    assert wm.seen[free].mean() > .95, f"only {100 * wm.seen[free].mean():.0f}% of the free floor was searched"


def test_exploration_never_revisits_the_same_place():
    wm, spots, iters = explore_synthetic()
    for i, a in enumerate(spots):
        for b in spots[i + 1:]:
            assert math.hypot(a[0] - b[0], a[1] - b[1]) > .85, f"viewpoints {a} and {b} are the same place"


# ------------------------------------------------------------------ 2+3. shortest route, straight tracking, detour and return
def run_route(sim, wm, A, B, max_t=90.0):
    brain = rs.Brain(.4, wm, None, map_only=True)
    path, m = wm.route(A, sim.pose()[2], B)
    assert path, "no route on the oracle map"
    brain._set_route(path, B, m, sim.t, sim.pose())
    trail = [sim.pose()[:2]]
    st = "go"
    while sim.t < max_t:
        pose, rng = sim.pose(), sim.ranges()
        wm.update_range(pose, rng)
        v, w, st = brain._follow(pose, rng, sim.t, "test")
        if st != "go":
            break
        sim.step(v, w, rs.DT)
        trail.append(sim.pose()[:2])
    return brain, st, trail


def _place_robot(sim, A, B):
    import pybullet as p
    yaw = math.atan2(B[1] - A[1], B[0] - A[0])
    p.resetBasePositionAndOrientation(sim.car, [A[0], A[1], .06], p.getQuaternionFromEuler([0, 0, yaw]))
    sim.trail = []


def test_route_without_obstacles_is_followed_directly():
    sim = rs.Sim((400, 300))
    sim.reset(905)
    wm = oracle_map(sim, objects=False)
    A, B = free_pair(wm, seed=1)
    _place_robot(sim, A, B)
    ideal = route_len(wm.astar(A, B, sim.pose()[2]))
    brain, st, trail = run_route(sim, wm, A, B)
    assert st == "done", f"did not arrive ({st}: {brain.stuck_why})"
    assert sim.collisions == 0
    assert route_len(trail) < 1.25 * ideal + .5, f"drove {route_len(trail):.1f} m for a {ideal:.1f} m route"
    assert brain.replans <= 1, f"{brain.replans} replans although nothing was in the way (oracle map is nearly exact)"


def test_unexpected_obstacle_causes_one_detour_then_the_route_resumes():
    sim = rs.Sim((400, 300))
    sim.reset(905)
    wm = oracle_map(sim, objects=False)
    A, B = free_pair(wm, seed=1)
    _place_robot(sim, A, B)
    plan = wm.astar(A, B, sim.pose()[2])
    k = int(len(plan) * .45)
    sim.add_obstacle(plan[k][0], plan[k][1], "crate")           # appears ON the planned route, unknown to the map
    wm2 = oracle_map(sim, objects=False)                        # what the best possible plan looks like (crate known)
    best = route_len(wm2.astar(A, B, sim.pose()[2]))
    brain, st, trail = run_route(sim, wm, A, B)
    assert st == "done", f"did not arrive ({st}: {brain.stuck_why})"
    assert sim.collisions == 0, f"{sim.collisions} collisions"
    assert brain.replans >= 1, "never replanned around the crate"
    assert route_len(trail) < 1.4 * best + .8, f"drove {route_len(trail):.1f} m; best possible {best:.1f} m"


# ------------------------------------------------------------------ 4+5. repeated target switching, direct approach
def _mission(seed, first):
    sim = rs.Sim((400, 300))
    sim.reset(seed)
    path = os.path.join(tempfile.gettempdir(), f"test_oracle_{seed}.npz")
    oracle_map(sim).save(path, room=seed)
    return sim, rs.Mission(sim, rs.GroundTruthDetector(), first, seed, max_t=2000, load_map=path)


def _leg(m, sim, key, first=False):
    if not first:
        m.retarget(key)
        b = m.brain                                               # nothing from the previous leg may survive
        assert b.rc_pending is None and not b.path and b.ap_goal is None and b.goal is None
        assert m.tgt_xy is None and m.best_box is None and m.dets == [] and m.outcome is None
    t0, p0, start = sim.t, sim.path, sim.pose()[:2]
    m.max_t = sim.t + 150
    first_goal, n = None, 0
    while m.outcome is None:
        m.tick()
        n += 1
        if not first and n <= 3:                                  # right after the change: route from here, or already tracking it
            assert m.brain.goal is not None or m.brain.state in ("VERIFY", "APPROACH", "AVOID", "ALIGN", "REACQUIRE", "ARRIVED"), \
                f"tick {n} after retarget: state {m.brain.state}/{m.brain.phase} with no route and no target in view"
        if first_goal is None and m.brain.goal is not None:
            first_goal = m.brain.goal
    return dict(outcome=m.outcome, time=sim.t - t0, dist=sim.path - p0, goal=first_goal, start=start)


def test_switching_targets_always_goes_to_the_requested_one_by_a_short_route():
    sim, m = _mission(903, "blue")
    seq = ["blue", "red", "green", "blue", "green", "red", "blue"]
    for i, key in enumerate(seq):
        leg = _leg(m, sim, key, first=(i == 0))
        tx, ty = sim.items[key]
        assert leg["outcome"] == "SUCCESS", f"leg {i} ({key}): {leg['outcome']}"
        if leg["goal"] is not None:
            assert math.hypot(leg["goal"][0] - tx, leg["goal"][1] - ty) < 2.0, f"leg {i} ({key}) first headed to {leg['goal']}, not near {(tx, ty)}"
        straight = math.hypot(tx - leg["start"][0], ty - leg["start"][1])
        assert leg["dist"] < 1.8 * straight + 1.5, f"leg {i} ({key}): drove {leg['dist']:.1f} m for {straight:.1f} m straight line"
    assert sim.collisions == 0, f"{sim.collisions} collisions while switching"


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name)
        except Exception:
            fails += 1
            print("FAIL", name)
            traceback.print_exc()
    sys.exit(1 if fails else 0)
