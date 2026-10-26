"""Phase 1: robot looks for a water bottle, turns to it and drives up to it. No distance sensors needed.

Setup (laptop):   pip install ultralytics opencv-python pyserial
Run:
    python bottle_follow.py spin 6     # test: spin in place for 6 s (check turning / calibrate)
    python bottle_follow.py fwd 2      # test: drive straight for 2 s (check both sides spin the right way)
    python bottle_follow.py dry        # camera + YOLO only, prints what it WOULD do, motors untouched
    python bottle_follow.py            # find the bottle and drive to it   (press q in the video window to quit)
"""
import sys, time, threading
import cv2, serial

PORT = "COM5"                                # HC-05 outgoing COM port (Linux: /dev/rfcomm0, Mac: /dev/cu.HC-05-*)
BAUD = 9600                                  # HC-05 default data speed
CAM_URL = "http://192.168.1.50:8080/video"   # shown in the IP Webcam app after "Start server"

SPIN_PWM = 110      # in-place turning power. If the wheels only buzz, raise it (130-170)
FWD_PWM = 140       # driving power
KP = 120            # steering strength (PWM per unit of sideways offset, offset is -1..+1)
ARRIVE_H = 0.55     # stop when the bottle is this fraction of the frame height (about "close enough")
BOTTLE = 39         # COCO class id for "bottle"
IMGSZ = 320         # YOLO input size: smaller = faster = less delay
CONFIRM = 2         # bottle must be seen in this many NEW frames in a row before the robot reacts
LOST_GRACE = 4      # while driving, ignore this many missed frames in a row before going back to searching
SPIN_PULSE, TURN_PULSE, SETTLE = 0.25, 0.15, 0.35   # seconds: search turn, aim turn, pause for a sharp image


class Robot:
    """Sends text commands to the Arduino. dry=True only prints them."""

    def __init__(self, dry=False):
        self.dry, self.ser, self.last_print = dry, None, ""
        if dry:
            return
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=0.05, write_timeout=1)
        except serial.SerialException as e:
            sys.exit(f"Cannot open {PORT}: {e}\n"
                     "  - Is the HC-05 paired in Windows Bluetooth settings?\n"
                     "  - Use the OUTGOING COM port (Bluetooth settings > More Bluetooth options > COM Ports)\n"
                     "  - Is another program (serial monitor / phone app) still connected to it?")
        time.sleep(2)

    def _write(self, text):
        if self.dry:
            if text != self.last_print:          # print only changes, not 20 lines per second
                print("[dry]", text.strip())
                self.last_print = text
            return
        try:
            self.ser.write(text.encode())
        except (serial.SerialException, OSError) as e:
            print("serial write failed:", e)

    def send(self, left, right):
        left, right = max(-255, min(255, int(left))), max(-255, min(255, int(right)))
        self._write(f"M {left} {right}\n")

    def stop(self):
        self._write("S\n")

    def poll(self):
        """Drain whatever the Arduino sent so the input buffer never fills. Returns True if it reported ESTOP."""
        if self.dry or not self.ser:
            return False
        try:
            data = self.ser.read(self.ser.in_waiting or 0)
        except (serial.SerialException, OSError):
            return False
        return b"ESTOP" in data

    def close(self):
        try:
            self.stop()
            if self.ser:
                self.ser.close()
        except Exception:
            pass


class LatestFrame:
    """Reads the camera stream in a background thread and keeps ONLY the newest frame.
    Without this the stream buffers up and the robot reacts to what the camera saw seconds ago."""

    def __init__(self, url):
        self.url, self.frame, self.stamp = url, None, 0
        self.lock, self.running = threading.Lock(), True
        self.cap = cv2.VideoCapture(url)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:                           # stream dropped: reconnect
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                continue
            with self.lock:
                self.frame, self.stamp = f, self.stamp + 1

    def get(self):
        """Returns (frame, frame_number). frame is None until the first image arrives."""
        with self.lock:
            return self.frame, self.stamp

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def decide(box, W, H):
    """Pure steering logic. box = (x1, y1, x2, y2) in pixels. Returns (action, left_pwm, right_pwm)."""
    x1, y1, x2, y2 = box
    offset = ((x1 + x2) / 2 - W / 2) / (W / 2)           # -1 = far left, +1 = far right
    if (y2 - y1) / H >= ARRIVE_H:
        return "ARRIVED", 0, 0
    if abs(offset) > 0.35:                                # well off-centre: turn toward it first
        s = SPIN_PWM if offset > 0 else -SPIN_PWM
        return "TURN", s, -s                              # offset > 0 (bottle on the right): left wheel forward, right back
    turn = KP * offset                                    # roughly ahead: drive and steer
    return "DRIVE", FWD_PWM + turn, FWD_PWM - turn


def timed_move(left, right, seconds):
    robot = Robot()
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            robot.send(left, right)              # the Arduino stops if commands stop, so keep re-sending
            robot.poll()
            time.sleep(0.2)
    finally:
        robot.close()


def main(dry=False):
    from ultralytics import YOLO
    robot = Robot(dry)
    model = YOLO("yolov8n.pt")                   # downloads once (~6 MB)
    cam = LatestFrame(CAM_URL)
    cv2.namedWindow("robot view")
    last_n, seen, lost, driving, last_cmd, warned = -1, 0, 0, False, (0, 0), 0.0
    try:
        while True:
            frame, n = cam.get()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if robot.poll():
                print("Arduino reported ESTOP (obstacle too close)")
            if frame is None:
                robot.stop()
                if time.time() - warned > 3:
                    print("no camera frame yet - check the IP Webcam URL"); warned = time.time()
                time.sleep(0.1)
                continue
            if n == last_n:                      # no new image since last loop: don't re-process the same one
                time.sleep(0.005)
                continue
            last_n = n
            H, W = frame.shape[:2]
            res = model(frame, verbose=False, conf=0.4, classes=[BOTTLE], imgsz=IMGSZ)[0]
            box = None
            if len(res.boxes):
                b = max(res.boxes, key=lambda b: float(b.conf))
                box = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("robot view", frame)

            if box is None:
                lost += 1
                if driving and lost <= LOST_GRACE:       # probably a one-frame miss: keep going
                    robot.send(*last_cmd)
                    continue
                seen, driving = 0, False                 # SEARCH: short turn, then pause so the image settles
                robot.send(-SPIN_PWM, SPIN_PWM); time.sleep(SPIN_PULSE)
                robot.stop(); time.sleep(SETTLE)
                continue

            lost = 0
            seen += 1
            if seen < CONFIRM:                           # wait for a second sighting before reacting (kills false alarms)
                robot.stop(); driving = False
                continue
            action, l, r = decide(box, W, H)
            if action == "ARRIVED":
                robot.stop(); print("ARRIVED at the bottle")
                break
            if action == "TURN":
                driving = False
                robot.send(l, r); time.sleep(TURN_PULSE)
                robot.stop(); time.sleep(SETTLE)
            else:
                driving, last_cmd = True, (l, r)
                robot.send(l, r)
    finally:
        robot.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    if mode == "spin":
        timed_move(-SPIN_PWM, SPIN_PWM, secs)
        print(f"spun for {secs}s - note how much of a full turn that was")
    elif mode == "fwd":
        timed_move(FWD_PWM, FWD_PWM, secs)
        print(f"drove forward for {secs}s")
    elif mode == "dry":
        main(dry=True)
    else:
        main()
