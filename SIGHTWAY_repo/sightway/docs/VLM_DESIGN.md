# VLM design (PROPOSED, not implemented)

> **Status:** nothing in this document is built. `robot_sim.py` has an open-vocabulary detector hook
> (`--detector yoloworld`, `YoloWorldDetector`) that has **not been tested**, and no cloud VLM is called anywhere.
> This is the design we would implement next.

## Principle

Do not run a large model on every frame. Use cheap local layers for safety and for known objects, and ask a
vision-language model (VLM) only when those layers cannot answer. The VLM **names things only; it never drives.**

![cascade](images/D2_vlm_cascade.png)

## Layers

1. **Always on (built):** sonar + camera hazard layer. Keeps the robot safe around anything, labelled or not.
2. **Every frame (built):** YOLOv8n / colour detector for the known classes.
3. **On demand (proposed):** VLM for everything else.

## When the VLM would be called

- A hazard blob has no label and the robot needs to know whether it is movable or fragile.
- The user asks for an object the local detector does not know (for example "find the stapler").
- A detection has low confidence and must be verified before the robot commits to driving to it.

## Gate (all must hold)

- The robot is slow or stopped, so latency is harmless.
- The per-object cooldown has expired.
- The per-minute call budget is not used up.
- The object is not already cached on a map cluster.

If any condition fails, the object is treated as an obstacle and the robot keeps driving.

## Token and cost savers

| Technique | Why it helps |
|---|---|
| Crop to the region of interest and downscale | A 256×256 crop has about 21% of the pixels of a 640×480 frame, and image cost grows with pixels |
| Numbered contact sheet | Several candidates are answered in one call |
| Fixed short prompt, structured reply | Few output tokens: `{"id": 2, "label": "stapler", "movable": true}` |
| Cache by map cluster | The same object is never asked about twice |
| Cooldown and budget | Bounds worst-case usage |
| Asynchronous call | The robot stays safe on layers 1 and 2 while waiting |
| Offline fallback | No network means the object stays an obstacle |

We have **not measured** token counts, latency or cost. A first experiment should log them per call.

## Example prompt (draft)

```
You see numbered crops from a small indoor robot's camera.
For each crop answer only with JSON: {"id": <n>, "label": "<2-3 words>", "movable": true|false, "fragile": true|false}.
If unsure, use "label": "unknown".
```

## Where it plugs in

`make_detector()` already returns a detector with a `detect(frame, seg, sim)` interface. A VLM client would be an extra
step after the local detector and hazard layer, writing its label onto the existing map cluster. The planner and safety
code would not change.

## Risks to test

- Wrong labels: require a second sighting before trusting a VLM label for planning.
- Latency and outages: handled by the gate and the offline fallback.
- Privacy: crops leave the device, so document what is sent and let it be switched off.
