"""Where is the peg relative to the socket's hole when they touch? (socket frame)

Socket: tube along its local x, half-length 0.06, inner half-width 0.018; pin fills |x|<0.04.
Peg: half-length 0.06, half-width 0.01 -> 8 mm play per side. Tip depth past the mouth =
0.12 - |axial|; touching the pin = depth 0.02.
"""
import json
import sys

import numpy as np


def rotm(q):  # MuJoCo quat (w, x, y, z) -> rotation matrix
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


run = sys.argv[1]
idx = [json.loads(line) for line in open(f"{run}/episodes.jsonl")]
out = []
for d in idx:
    z = np.load(f"{run}/episodes/ep_{d['idx']:05d}.npz")
    c = z["trace/peg_socket_contacts"] > 0
    if not c.any():
        continue
    P, S = z["trace/peg_pos"], z["trace/socket_pos"]
    QP, QS = z["trace/peg_quat"], z["trace/socket_quat"]
    depth, lat, tilt = [], [], []
    for t in range(len(P)):
        Rs = rotm(QS[t])
        r = Rs.T @ (P[t] - S[t])
        axis = Rs.T @ rotm(QP[t])[:, 0]
        depth.append(0.12 - abs(r[0]))
        lat.append(max(abs(r[1]), abs(r[2])))  # square hole: the worse side
        tilt.append(np.degrees(np.arccos(min(1.0, abs(axis[0])))))
    depth, lat, tilt = map(np.array, (depth, lat, tilt))
    j = np.where(c)[0]
    k = j[np.argmax(depth[j])]
    out.append(dict(seed=d["seed"], label=d["label"], contact_steps=int(c.sum()), depth_mm=1000 * depth[k],
                    lat_mm=1000 * lat[k], tilt_deg=tilt[k], first_lat_mm=1000 * lat[j[0]],
                    first_tilt=tilt[j[0]], force_med=float(np.median(z["trace/peg_socket_force"][j]))))

for lab in ("success", "near_miss", "fail"):
    rows = [o for o in out if o["label"] == lab]
    if not rows:
        continue
    print(f"\n{lab}: {len(rows)} episodes with peg-socket contact")
    for key in ("depth_mm", "lat_mm", "tilt_deg", "first_lat_mm", "first_tilt", "contact_steps", "force_med"):
        v = np.array([o[key] for o in rows])
        print(f"  {key:14s} median {np.median(v):7.1f}   IQR {np.percentile(v, 25):7.1f} .. {np.percentile(v, 75):7.1f}")
    if lab == "near_miss":
        for o in sorted(rows, key=lambda o: -o["depth_mm"]):
            print(f"   seed {o['seed']:3d} depth {o['depth_mm']:6.1f} mm  off-centre {o['lat_mm']:5.1f} mm  "
                  f"tilt {o['tilt_deg']:5.1f} deg  contact {o['contact_steps']:3d} steps  force {o['force_med']:.1f} N")
