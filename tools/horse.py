# The unicorn's body: a young, soft-looking horse (foal proportions: a bigger head, shorter legs).
# axes: x = forward, y = up, z = sideways. Ground is y = 0.
import sys, numpy as np
from sdf import *
out = sys.argv[1]
EZ = 0.86
torso = [
    ((-0.95, 2.22, 0), (-0.2, 2.12, 0), 0.60, 0.62, EZ),
    ((-0.2, 2.12, 0), (0.45, 2.12, 0), 0.62, 0.64, EZ),
    ((0.45, 2.12, 0), (0.8, 2.26, 0), 0.64, 0.55, EZ),
    ((0.76, 2.4, 0), (1.12, 2.96, 0), 0.54, 0.43, 0.82),      # neck: deep from the side, slimmer from the front
    ((1.12, 2.96, 0), (1.27, 3.36, 0), 0.43, 0.38, 0.84),
]
head = [
    ((1.38, 3.62, 0), (1.40, 3.64, 0), 0.50, 0.50, 0.94),     # skull
    ((1.40, 3.40, 0), (1.42, 3.40, 0), 0.36, 0.36, 0.82),     # jaw
    ((1.56, 3.48, 0), (2.0, 3.19, 0), 0.36, 0.25, 0.9),       # muzzle
    ((2.0, 3.19, 0), (2.08, 3.13, 0), 0.25, 0.225, 0.94),
]
for sg in (1, -1):
    head.append(((1.24, 3.94, 0.25 * sg), (1.17, 4.2, 0.31 * sg), 0.1, 0.135, 0.5))      # ears: leaf-shaped
    head.append(((1.17, 4.2, 0.31 * sg), (1.11, 4.45, 0.36 * sg), 0.135, 0.03, 0.5))
def front(z):
    return [((0.72, 2.02, z), (0.74, 1.2, z), 0.31, 0.155, 1), ((0.74, 1.2, z), (0.745, 1.06, z), 0.165, 0.165, 1),
            ((0.745, 1.06, z), (0.76, 0.46, z), 0.15, 0.115, 1), ((0.76, 0.46, z), (0.77, 0.34, z), 0.14, 0.14, 1),
            ((0.77, 0.34, z), (0.82, 0.19, z), 0.125, 0.125, 1), ((0.83, 0.17, z), (0.86, 0.04, z), 0.15, 0.2, 1)]
def hind(z):
    return [((-0.8, 2.08, z), (-0.72, 1.52, z), 0.43, 0.3, 1), ((-0.72, 1.52, z), (-0.97, 1.08, z), 0.3, 0.15, 1),
            ((-0.97, 1.08, z), (-0.975, 0.96, z), 0.165, 0.165, 1), ((-0.975, 0.96, z), (-0.87, 0.46, z), 0.15, 0.115, 1),
            ((-0.87, 0.46, z), (-0.86, 0.34, z), 0.14, 0.14, 1), ((-0.86, 0.34, z), (-0.81, 0.19, z), 0.125, 0.125, 1),
            ((-0.8, 0.17, z), (-0.77, 0.04, z), 0.15, 0.2, 1)]
legs = [front(0.3), front(-0.3), hind(0.3), hind(-0.3)]

def parts(P):
    dT = group_sdf(P, torso, 0.16)
    dH = group_sdf(P, head, 0.14)
    dL = [group_sdf(P, l, 0.07) for l in legs]
    return dT, dH, dL
def field(P):
    dT, dH, dL = parts(P)
    d = smin(dT, dH, 0.2)
    for l in dL:
        d = smin(d, l, 0.26)          # legs melt into the body
    return np.maximum(d, -(P[:, 1] - 0.012))   # flat under the hooves

v, f = polygonise(field, (-1.75, -0.06, -0.86), (2.4, 4.56, 0.86), 0.034)
print('raw', len(v), len(f))
v = taubin(v, f, 18, keep=v[:, 1] < 0.03)
dT, dH, dL = parts(v)
m = 0.2
w = np.stack([smoothstep(-m, m, dT - dH)] + [smoothstep(-m, m, dT - l) for l in dL], 1)   # head, 4 legs
s = w.sum(1); over = s > 1; w[over] /= s[over, None]
# soft paint: muzzle and hooves
ax = np.array([0.44, -0.3, 0]); ax /= np.linalg.norm(ax)
t = (v - np.array([1.56, 3.48, 0])) @ ax
muzzle = smoothstep(0.18, 0.5, t) * smoothstep(0.3, 0.9, w[:, 0])
hoof = 1 - smoothstep(0.17, 0.23, v[:, 1])
info = export_js(out, 'HORSE_MESH', v, f, {'w': np.round(w * 255).astype(np.uint8), 'paint': np.round(np.stack([muzzle, hoof], 1) * 255).astype(np.uint8)})
print('verts', info['n'], 'tris', info['t'], 'bbox', np.round(v.min(0), 2), np.round(v.max(0), 2))
