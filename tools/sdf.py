# Shared tools: build one smooth, continuous animal skin from rounded "bones" of clay,
# then cut it into triangles. Legs, neck and head melt into the body instead of sticking out of it.
import numpy as np, base64, json
from skimage import measure
import scipy.sparse as sp

def seg(P, a, b, ra, rb, ez=1.0):
    """distance from points P to a rounded stick from a to b, thick ra at a and rb at b; ez squashes it sideways"""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    Q = P.copy()
    if ez != 1.0:
        Q[:, 2] = a[2] + (Q[:, 2] - a[2]) / ez
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return np.linalg.norm(Q - a, axis=1) - ra
    t = np.clip(((Q - a) @ ab) / L2, 0, 1)
    d = np.linalg.norm(Q - (a + t[:, None] * ab), axis=1)
    return d - (ra + t * (rb - ra))

def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25

def group_sdf(P, prims, k):
    d = None
    for p in prims:
        s = seg(P, *p)
        d = s if d is None else smin(d, s, k)
    return d

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def polygonise(field_fn, lo, hi, h):
    xs = np.arange(lo[0], hi[0] + h, h, dtype=np.float32)
    ys = np.arange(lo[1], hi[1] + h, h, dtype=np.float32)
    zs = np.arange(lo[2], hi[2] + h, h, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vol = np.empty(len(P), np.float32)
    step = 400000
    for i in range(0, len(P), step):
        vol[i:i + step] = field_fn(P[i:i + step])
    vol = vol.reshape(X.shape)
    v, f, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(h, h, h))
    v = v + np.asarray(lo, np.float32)
    # make sure faces point outward
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() < 0:
        f = f[:, ::-1]
    return v.astype(np.float32), f.astype(np.int64)

def taubin(v, f, iters=6, lam=0.5, mu=-0.53, keep=None):
    n = len(v)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sp.coo_matrix((np.ones(len(i), np.float32), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    x = v.astype(np.float64)
    for _ in range(iters):
        for s in (lam, mu):
            d = W @ x - x
            if keep is not None:
                d[keep] = 0
            x = x + s * d
    return x.astype(np.float32)

def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')

def export_js(path, name, v, f, extra):
    lo = v.min(0); hi = v.max(0)
    q = np.round((v - lo) / (hi - lo) * 65535).astype(np.uint16)
    idx = f.astype(np.uint16 if len(v) < 65536 else np.uint32)
    obj = {'n': int(len(v)), 't': int(len(f)), 'lo': [float(x) for x in lo], 'hi': [float(x) for x in hi],
           'pos': b64(q), 'idx': b64(idx), 'idxBytes': int(idx.dtype.itemsize)}
    for k, a in extra.items():
        obj[k] = b64(a)
    open(path, 'w').write('window.' + name + ' = ' + json.dumps(obj) + ';\n')
    return obj
