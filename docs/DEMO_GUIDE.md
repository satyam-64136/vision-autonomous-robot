# Demo guide (for the 90 to 120 second video)

## Set up

```bash
pip install -r requirements.txt
python robot_sim.py --reset-map --seed 5 --mode map --save-map room.npz     # record this as shot 1
```

Record the screen (the dashboard window) plus your voice. `--video demo.mp4` also saves the dashboard.

## Shot list

| Time | What to show | What to say |
|---|---|---|
| 0:00 | Title card, then the dashboard | "SIGHTWAY maps a room once and drives straight to the object you name." |
| 0:10 | Mapping run (`--mode map`) | "It explores until nothing reachable is left unsearched." |
| 0:30 | Type `go to the blue energy drink can` | "Now it plans a route from the saved map. No re-scan." |
| 0:45 | Press `o` to drop an obstacle ahead | "A surprise obstacle: it replans around it." |
| 1:00 | Type `go to the red cola can` (target switch) | "Switching targets clears old goals and plans fresh from here." |
| 1:15 | Press `r` or type `reset map` | "The map is temporary storage and can be wiped." |
| 1:25 | Architecture slide (D1 or the README diagram) | "Built in simulation. Phone, ESP32 and VLM are our proposed next steps." |
| 1:40 | Results: 8/8, 0 collisions | "Simulator results only." |

## Commands to type in the terminal while it runs

```
go to the red cola can
go to the blue energy drink can
go to the green soda can
find the person
map the room
reset map
```

## Tips

- Use the same `--seed` for every take so the room looks identical.
- Close other apps so the window stays smooth, and keep the dashboard window focused when pressing keys.
- Say "simulation" out loud at least once so nobody thinks it is hardware.
