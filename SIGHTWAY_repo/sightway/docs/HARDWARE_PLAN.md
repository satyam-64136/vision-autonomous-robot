# Hardware plan (PROPOSED, not built)

> **Status:** no hardware has been built or tested. This is the plan for taking the simulator brain onto a real
> low-cost car. Every number below is a placeholder to be measured.

## Proposed setup

| Part | Role |
|---|---|
| Android phone | Camera and on-device processing (detector, map, planner). Sends motion commands |
| ESP32 on an RC car chassis | Receives `(v, w)` and drives the motors |
| Motor driver and wheel encoders | Needed for odometry (the simulator provides perfect-ish pose today) |
| Optional ultrasonic sensors | Replace the 9 simulated range rays |
| Optional cloud VLM | See [VLM_DESIGN.md](VLM_DESIGN.md) |

## The interface the brain already has

The brain outputs forward speed `v` (m/s, up to 0.6) and turn rate `w` (rad/s, up to 1.6) once per tick (0.1 s).
For a differential-drive car with wheel separation `L`:

```
v_left  = v - w * L / 2
v_right = v + w * L / 2
```

The ESP32 converts these into motor PWM, using a calibration measured on the real car.

## Message format (draft)

A short UDP packet or BLE write, for example the text `"v,w\n"` (`"0.30,-0.40\n"`). The ESP32 stops the car if no packet
arrives within a timeout (for example 300 ms). That safety timeout is mandatory.

## Steps to bring it up

1. Measure real top speed, turn rate and braking distance; update `Safety` (`react`, `decel`, `half_len`, `half_wid`).
2. Build the ESP32 receiver with the stop-on-timeout rule; test with the wheels off the ground.
3. Replace simulated pose with encoder odometry; check drift over a lap.
4. Replace the 9 simulated range rays with real sensors (or camera depth), re-tune the noise values.
5. Photograph the real cans and obstacles, retrain the detector (see README), test in a cleared room.
6. Only then run the full explore / go-to-object loop, at reduced speed first.

## Known gaps to close

- The detector is trained only on simulator images.
- Margins and speeds are tuned only for the simulator.
- No wheel-slip handling, no IMU fusion.
- No hardware code exists in this repository yet.
