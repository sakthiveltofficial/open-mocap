# mp_skeleton.py
# ==============
# Fast engine (MediaPipe) -> the body pipeline's skeleton contract.
#
# The body pipeline (pipeline/modes.py run_body and everything after it:
# tracking repair, finger boost, camera offset, foot locking, Feet Define
# the Floor, body grounding, depth noise filtering, face anchoring and the
# BVH writer) was written against SAM 3D Body's 127-joint MHR skeleton. It
# addresses joints by MHR index (Hips = 1, LeftFoot = 5, Head = 113, ...).
#
# Rather than fork that pipeline, the Fast engine produces data in the same
# shape: a 127-slot skeleton whose KEPT joints (body_skeleton.KEPT_ORDER)
# sit at the MHR indices the pipeline expects, with every other slot a
# zero-length marker. Global rotations are world-space deltas from a T-pose
# rest (prerotations are identity, so the T-pose globals are identity), and
# joint_coords / cam_t follow SAM's camera-frame conventions. With that,
# the whole cleanup pipeline runs unchanged on MediaPipe captures, and both
# engines emit BVHs with identical bone names, so every retarget preset,
# the FK -> IK converter and the library work for either.
#
# Frames:
#   MediaPipe world:  x image-right, y down, z away from camera (metres,
#                     origin between the hips)
#   BVH / model:      x image-right (= subject's left when facing camera),
#                     y up, z toward camera
#   SAM camera:       x right, y down, z forward (away)  (joint_coords, cam_t)
#
# Part of Open Mocap, GPL-3.0-or-later.

import numpy as np

N_JOINTS = 127

# --- MHR joint indices used by the body pipeline (see body_skeleton) -------
ROOT0 = 0
HIPS = 1
SPINE, SPINE1, SPINE2, SPINE3 = 34, 35, 36, 37
NECK, HEAD, HEAD_TOP = 110, 113, 126
EYE_L, EYE_R = 122, 124

SIDES = {
    # side: dict of MHR indices
    "L": dict(upleg=2, leg=3, ankle_marker=4, foot=5, heel=6, toe=7, toe_tip=8,
              shoulder=74, arm=75, forearm=76, wrist_marker=77, hand=78,
              pinky_root=79,
              fingers={"pinky": [80, 81, 82, 83], "ring": [84, 85, 86, 87],
                       "middle": [88, 89, 90, 91], "index": [92, 93, 94, 95],
                       "thumb": [96, 97, 98, 99]}),
    "R": dict(upleg=18, leg=19, ankle_marker=20, foot=21, heel=22, toe=23, toe_tip=24,
              shoulder=38, arm=39, forearm=40, wrist_marker=41, hand=42,
              pinky_root=43,
              fingers={"pinky": [44, 45, 46, 47], "ring": [48, 49, 50, 51],
                       "middle": [52, 53, 54, 55], "index": [56, 57, 58, 59],
                       "thumb": [60, 61, 62, 63]}),
}

# MediaPipe pose landmark indices
MP = dict(nose=0, eye_l=2, eye_r=5, ear_l=7, ear_r=8,
          sho_l=11, sho_r=12, elb_l=13, elb_r=14, wri_l=15, wri_r=16,
          pinky_l=17, pinky_r=18, index_l=19, index_r=20, thumb_l=21, thumb_r=22,
          hip_l=23, hip_r=24, knee_l=25, knee_r=26, ank_l=27, ank_r=28,
          heel_l=29, heel_r=30, toe_l=31, toe_r=32)

# MediaPipe hand landmark chains (wrist = 0)
HAND_CHAINS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
               "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}

# MP world (x right, y down, z away) -> BVH model (x right, y up, z toward cam)
_MP2BVH = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# small numpy helpers
# ---------------------------------------------------------------------------

def _norm(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _basis(y, ref):
    """Rotation matrices with Y axis along `y`, X along ref (orthogonalised)."""
    y = _norm(y)
    x = ref - np.sum(ref * y, axis=-1, keepdims=True) * y
    x = _norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def _perp(v, axis):
    """Component of v perpendicular to (unit) axis."""
    a = _norm(axis)
    return v - np.sum(v * a, axis=-1, keepdims=True) * a


def _blend_ref(primary, fallback, axis, full_at=0.25):
    """Blend a primary reference direction (only meaningful when large, e.g.
    the bend direction of a limb) with a fallback, both made perpendicular
    to `axis`. Weight of the primary ramps 0..1 as its perpendicular length
    (relative to its own length) goes 0..full_at."""
    p = _perp(primary, axis)
    plen = np.linalg.norm(p, axis=-1, keepdims=True)
    rel = plen / np.maximum(np.linalg.norm(primary, axis=-1, keepdims=True), 1e-9)
    w = np.clip(rel / full_at, 0.0, 1.0)
    f = _norm(_perp(fallback, axis))
    return w * _norm(p) + (1.0 - w) * f


def fill_gaps(x, valid, max_gap=None):
    """Linear interpolation over invalid frames along axis 0 (holds at ends).
    x: (N, ...) ; valid: (N,) bool. Returns filled copy."""
    x = np.array(x, dtype=np.float64, copy=True)
    n = x.shape[0]
    valid = np.asarray(valid, bool) & np.all(np.isfinite(x.reshape(n, -1)), axis=1)
    if valid.all():
        return x
    if not valid.any():
        return np.nan_to_num(x)
    idx = np.arange(n)
    good = idx[valid]
    flat = x.reshape(n, -1)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(idx, good, flat[good, c])
    return flat.reshape(x.shape)


def gaussian(x, sigma):
    if sigma <= 0.05 or x.shape[0] < 3:
        return x
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    out = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * pad[i:i + x.shape[0]]
    return out


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

def _topology():
    """parents (127,) for the synthetic skeleton. Unused slots hang off the
    origin joint with zero length so they are pruned away."""
    p = np.zeros(N_JOINTS, dtype=np.int32)
    p[ROOT0] = -1
    p[HIPS] = ROOT0
    p[SPINE] = HIPS
    p[SPINE1] = SPINE
    p[SPINE2] = SPINE1
    p[SPINE3] = SPINE2
    p[NECK] = SPINE3
    p[HEAD] = NECK
    p[HEAD_TOP] = HEAD
    p[EYE_L] = HEAD
    p[EYE_R] = HEAD
    for s in SIDES.values():
        p[s["upleg"]] = HIPS
        p[s["leg"]] = s["upleg"]
        p[s["ankle_marker"]] = s["leg"]
        p[s["foot"]] = s["leg"]
        p[s["heel"]] = s["foot"]
        p[s["toe"]] = s["foot"]
        p[s["toe_tip"]] = s["toe"]
        p[s["shoulder"]] = SPINE3
        p[s["arm"]] = s["shoulder"]
        p[s["forearm"]] = s["arm"]
        p[s["wrist_marker"]] = s["forearm"]
        p[s["hand"]] = s["forearm"]
        p[s["pinky_root"]] = s["hand"]
        for chain in s["fingers"].values():
            prev = s["hand"]
            for j in chain:
                p[j] = prev
                prev = j
    for j in range(N_JOINTS):
        assert p[j] < j, (j, p[j])
    return p


PARENTS = _topology()

# Joints that carry their own measured rotation (everything else copies
# its parent's global rotation).
_DRIVEN = [HIPS, SPINE, SPINE1, SPINE2, SPINE3, NECK, HEAD]
for _s in SIDES.values():
    _DRIVEN += [_s["upleg"], _s["leg"], _s["foot"], _s["shoulder"], _s["arm"],
                _s["forearm"], _s["hand"]]
    for _c in _s["fingers"].values():
        _DRIVEN += _c[:3]


# ---------------------------------------------------------------------------
# Proportions and rest (T-pose) layout
# ---------------------------------------------------------------------------

DEFAULT_LENGTHS = dict(
    torso=0.50, hip_width=0.20, shoulder_width=0.36, neck=0.17, head_top=0.12,
    upper_arm=0.28, forearm=0.26, palm=0.09, thigh=0.42, shin=0.41,
    ankle_height=0.08, foot=0.20,
    finger={"thumb": [0.040, 0.033, 0.028], "index": [0.040, 0.024, 0.021],
            "middle": [0.044, 0.028, 0.023], "ring": [0.041, 0.026, 0.022],
            "pinky": [0.033, 0.020, 0.019]},
)


def measure_lengths(pw, pvalid, hw=None, hvalid=None):
    """Median segment lengths (metres) from MediaPipe world landmarks.
    pw: (N,33,3) ; pvalid: (N,) ; hw: (N,2,21,3) ; hvalid: (N,2)."""
    L = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_LENGTHS.items()}
    if pvalid is None or not np.any(pvalid):
        return L
    P = pw[pvalid]

    def seg(a, b):
        d = np.linalg.norm(P[:, a] - P[:, b], axis=-1)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else None

    def both(al, bl, ar, br, key, lo, hi):
        vals = [v for v in (seg(al, bl), seg(ar, br)) if v]
        if vals:
            L[key] = float(np.clip(np.mean(vals), lo, hi))

    both(MP["sho_l"], MP["elb_l"], MP["sho_r"], MP["elb_r"], "upper_arm", 0.18, 0.40)
    both(MP["elb_l"], MP["wri_l"], MP["elb_r"], MP["wri_r"], "forearm", 0.16, 0.36)
    both(MP["hip_l"], MP["knee_l"], MP["hip_r"], MP["knee_r"], "thigh", 0.28, 0.58)
    both(MP["knee_l"], MP["ank_l"], MP["knee_r"], MP["ank_r"], "shin", 0.28, 0.56)
    both(MP["heel_l"], MP["toe_l"], MP["heel_r"], MP["toe_r"], "foot", 0.14, 0.32)
    v = seg(MP["hip_l"], MP["hip_r"])
    if v:
        L["hip_width"] = float(np.clip(v, 0.12, 0.40))
    v = seg(MP["sho_l"], MP["sho_r"])
    if v:
        L["shoulder_width"] = float(np.clip(v, 0.24, 0.55))
    hm = (P[:, MP["hip_l"]] + P[:, MP["hip_r"]]) / 2
    sm = (P[:, MP["sho_l"]] + P[:, MP["sho_r"]]) / 2
    em = (P[:, MP["ear_l"]] + P[:, MP["ear_r"]]) / 2
    t = np.linalg.norm(sm - hm, axis=-1)
    if np.isfinite(t).any():
        L["torso"] = float(np.clip(np.nanmedian(t), 0.35, 0.75))
    nk = np.linalg.norm(em - sm, axis=-1)
    if np.isfinite(nk).any():
        L["neck"] = float(np.clip(np.nanmedian(nk), 0.10, 0.30))

    if hw is not None and hvalid is not None and np.any(hvalid):
        H = hw[hvalid]  # (M,21,3)
        palm = np.linalg.norm(H[:, 9] - H[:, 0], axis=-1)
        if palm.size:
            L["palm"] = float(np.clip(np.median(palm), 0.06, 0.13))
        for name, chain in HAND_CHAINS.items():
            segs = []
            for a, b in zip(chain[:-1], chain[1:]):
                segs.append(float(np.clip(np.median(np.linalg.norm(H[:, b] - H[:, a], axis=-1)),
                                          0.01, 0.07)))
            L["finger"][name] = segs
    return L


def rest_layout(L):
    """World rest positions (metres, BVH frame) for all 127 joints in a
    T-pose facing +Z, plus per-joint rest reference frames used by the
    solver. Returns (pos (127,3), refs dict)."""
    pos = np.zeros((N_JOINTS, 3))
    ankle_h = L["ankle_height"]
    hip_h = L["thigh"] + L["shin"] + ankle_h
    hips = np.array([0.0, hip_h, 0.0])
    pos[ROOT0] = 0.0
    pos[HIPS] = hips
    T = L["torso"]
    up = np.array([0.0, 1.0, 0.0])
    pos[SPINE] = hips + up * T * 0.10
    pos[SPINE1] = hips + up * T * 0.35
    pos[SPINE2] = hips + up * T * 0.60
    pos[SPINE3] = hips + up * T * 0.82
    neck = hips + up * T
    pos[NECK] = neck
    pos[HEAD] = neck + up * L["neck"]
    pos[HEAD_TOP] = pos[HEAD] + up * L["head_top"]
    pos[EYE_L] = pos[HEAD] + np.array([0.032, 0.015, 0.085])
    pos[EYE_R] = pos[HEAD] + np.array([-0.032, 0.015, 0.085])
    s_unit = L["palm"] / 0.09
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        S = SIDES[side]
        X = np.array([sgn, 0.0, 0.0])
        hipj = hips + X * L["hip_width"] / 2
        pos[S["upleg"]] = hipj
        pos[S["leg"]] = hipj - up * L["thigh"]
        ank = np.array([hipj[0], ankle_h, 0.0])
        pos[S["ankle_marker"]] = ank
        pos[S["foot"]] = ank
        flen = L["foot"]
        pos[S["heel"]] = np.array([hipj[0], 0.02, -0.25 * flen])
        pos[S["toe"]] = np.array([hipj[0], 0.02, 0.50 * flen])
        pos[S["toe_tip"]] = np.array([hipj[0], 0.01, 0.75 * flen])
        pos[S["shoulder"]] = neck + X * L["shoulder_width"] * 0.10 - up * 0.02
        sho = neck + X * L["shoulder_width"] / 2
        pos[S["arm"]] = sho
        elb = sho + X * L["upper_arm"]
        pos[S["forearm"]] = elb
        wri = elb + X * L["forearm"]
        pos[S["wrist_marker"]] = wri
        pos[S["hand"]] = wri
        pos[S["pinky_root"]] = wri
        fwd = np.array([0.0, 0.0, 1.0])
        mcp = {
            "index": wri + X * L["palm"] * 0.95 + fwd * 0.022 * s_unit,
            "middle": wri + X * L["palm"],
            "ring": wri + X * L["palm"] * 0.95 - fwd * 0.020 * s_unit,
            "pinky": wri + X * L["palm"] * 0.86 - fwd * 0.040 * s_unit,
            "thumb": wri + X * 0.025 * s_unit + fwd * 0.022 * s_unit - up * 0.01 * s_unit,
        }
        for fname, chain in S["fingers"].items():
            d = X if fname != "thumb" else _norm(X + fwd)
            p0 = mcp[fname]
            segs = L["finger"][fname]
            pos[chain[0]] = p0
            for k in range(1, 4):
                pos[chain[k]] = pos[chain[k - 1]] + d * segs[k - 1]
    return pos


def offsets_from_positions(pos):
    off = np.zeros_like(pos)
    for j in range(N_JOINTS):
        p = PARENTS[j]
        off[j] = pos[j] if p < 0 else pos[j] - pos[p]
    return off


def skeleton_arrays(L):
    """(parents, offsets_cm, prerot_quats_xyzw, rest_pos_m) for the pipeline."""
    pos = rest_layout(L)
    offsets_cm = offsets_from_positions(pos) * 100.0
    prerot = np.zeros((N_JOINTS, 4))
    prerot[:, 3] = 1.0
    return PARENTS.copy(), offsets_cm, prerot, pos


# ---------------------------------------------------------------------------
# Per-frame joint positions (BVH frame) from MediaPipe landmarks
# ---------------------------------------------------------------------------

def _joint_points(pw_bvh, hw_bvh, hvalid):
    """Virtual joint positions used for bone directions. Returns dict of
    (N,3) arrays (hip-centred, BVH frame) and per-side hand arrays."""
    g = lambda k: pw_bvh[:, MP[k]]  # noqa: E731
    J = {}
    J["hip_mid"] = (g("hip_l") + g("hip_r")) / 2
    J["sho_mid"] = (g("sho_l") + g("sho_r")) / 2
    J["ear_mid"] = (g("ear_l") + g("ear_r")) / 2
    for side, s in (("L", "l"), ("R", "r")):
        J["hip" + side] = g("hip_" + s)
        J["knee" + side] = g("knee_" + s)
        J["ank" + side] = g("ank_" + s)
        J["heel" + side] = g("heel_" + s)
        J["toe" + side] = g("toe_" + s)
        J["sho" + side] = g("sho_" + s)
        J["elb" + side] = g("elb_" + s)
        J["wri" + side] = g("wri_" + s)
        J["pidx" + side] = g("index_" + s)
        J["ppinky" + side] = g("pinky_" + s)
    return J


def solve_global_rotations(pw_bvh, hw_bvh, hvalid, rest_pos, finger_hold=True):
    """World-delta rotations G (N,127,3,3) relative to the T-pose rest.

    pw_bvh: (N,33,3) pose world landmarks in BVH frame (hip-centred)
    hw_bvh: (N,2,21,3) hand world landmarks in BVH frame ([0]=left, [1]=right)
    hvalid: (N,2) bool
    """
    n = pw_bvh.shape[0]
    J = _joint_points(pw_bvh, hw_bvh, hvalid)
    R = rest_pos
    G = np.broadcast_to(np.eye(3), (n, N_JOINTS, 3, 3)).copy()
    B_rest = {}
    B_pose = {}

    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])
    Z = np.array([0.0, 0.0, 1.0])

    hip_axis = J["hipL"] - J["hipR"]
    sho_axis = J["shoL"] - J["shoR"]
    torso_up = J["sho_mid"] - J["hip_mid"]
    chest_fwd = _norm(np.cross(sho_axis, torso_up))       # X x Y = +Z (forward)

    # --- pelvis and spine ---
    B_pose[HIPS] = _basis(torso_up, hip_axis)
    B_rest[HIPS] = _basis(Y[None], X[None])[0]
    for j, t in ((SPINE, 0.0), (SPINE1, 1 / 3), (SPINE2, 2 / 3), (SPINE3, 1.0)):
        ref = (1 - t) * _norm(hip_axis) + t * _norm(sho_axis)
        B_pose[j] = _basis(torso_up, ref)
        B_rest[j] = B_rest[HIPS]

    # --- neck and head ---
    ear_axis = pw_bvh[:, MP["ear_l"]] - pw_bvh[:, MP["ear_r"]]
    neck_dir = J["ear_mid"] - J["sho_mid"]
    B_pose[NECK] = _basis(neck_dir, ear_axis)
    B_rest[NECK] = _basis(Y[None], X[None])[0]
    head_fwd = pw_bvh[:, MP["nose"]] - J["ear_mid"]
    head_up = np.cross(head_fwd, ear_axis)               # Z x X = +Y
    B_pose[HEAD] = _basis(head_up, ear_axis)
    # Rest: the nose sits slightly below the ear line when looking straight
    # ahead; compensate so a level gaze maps to an upright head.
    rest_fwd = _norm(np.array([0.0, -0.25, 1.0]))
    B_rest[HEAD] = _basis(np.cross(rest_fwd, X)[None], X[None])[0]

    for side, sgn in (("L", 1.0), ("R", -1.0)):
        S = SIDES[side]
        hi = 0 if side == "L" else 1
        # hand landmarks (if available) or pose fallbacks
        hv = hvalid[:, hi]
        hand = hw_bvh[:, hi]
        wri = J["wri" + side]
        h_mid = np.where(hv[:, None], wri + (hand[:, 9] - hand[:, 0]),
                         wri + ((J["pidx" + side] + J["ppinky" + side]) / 2 - wri) * 0.8)
        h_lat = np.where(hv[:, None], hand[:, 5] - hand[:, 17],
                         J["pidx" + side] - J["ppinky" + side])

        # legs
        thigh = J["knee" + side] - J["hip" + side]
        shin = J["ank" + side] - J["knee" + side]
        foot_fwd = J["toe" + side] - J["heel" + side]
        B_pose[S["upleg"]] = _basis(thigh, _blend_ref(-shin, foot_fwd, thigh))
        B_rest[S["upleg"]] = _basis(-Y[None], Z[None])[0]
        B_pose[S["leg"]] = _basis(shin, foot_fwd)
        B_rest[S["leg"]] = B_rest[S["upleg"]]
        ball = J["heel" + side] + 0.72 * foot_fwd
        B_pose[S["foot"]] = _basis(ball - J["ank" + side], -shin)
        rest_foot_dir = R[S["toe"]] - R[S["foot"]]
        B_rest[S["foot"]] = _basis(rest_foot_dir[None], Y[None])[0]

        # arms
        sho = J["sho" + side]
        clav = J["sho_mid"] + (sho - J["sho_mid"]) * 0.2
        B_pose[S["shoulder"]] = _basis(sho - clav, torso_up)
        B_rest[S["shoulder"]] = _basis((sgn * X)[None], Y[None])[0]
        upper = J["elb" + side] - sho
        fore = wri - J["elb" + side]
        B_pose[S["arm"]] = _basis(upper, _blend_ref(fore, chest_fwd, upper))
        B_rest[S["arm"]] = _basis((sgn * X)[None], Z[None])[0]
        B_pose[S["forearm"]] = _basis(fore, h_lat)
        B_rest[S["forearm"]] = B_rest[S["arm"]]
        B_pose[S["hand"]] = _basis(h_mid - wri, h_lat)
        B_rest[S["hand"]] = B_rest[S["arm"]]

        # fingers (only meaningful where the hand was tracked)
        palm_dir = hand[:, 9] - hand[:, 0]
        lat = hand[:, 5] - hand[:, 17]
        normal = np.cross(palm_dir, lat) if side == "L" else np.cross(lat, palm_dir)
        rest_normal = -Y
        for fname, chain in S["fingers"].items():
            lm = HAND_CHAINS[fname]
            for k in range(3):
                j = chain[k]
                d = hand[:, lm[k + 1]] - hand[:, lm[k]] if k < 3 else None
                ref = normal if fname == "thumb" else lat
                B_pose[j] = _basis(d, ref)
                rest_d = R[chain[k + 1]] - R[chain[k]]
                B_rest[j] = _basis(rest_d[None], (rest_normal if fname == "thumb" else Z)[None])[0]

    # world deltas
    for j in B_pose:
        G[:, j] = B_pose[j] @ B_rest[j].T

    # fingers: where the hand was not tracked, hold the nearest tracked local
    # rotation relative to the hand (or rest when never tracked)
    for side in ("L", "R"):
        S = SIDES[side]
        hi = 0 if side == "L" else 1
        hv = hvalid[:, hi].astype(bool)
        Gh = G[:, S["hand"]]
        for chain in S["fingers"].values():
            prev = S["hand"]
            for j in chain[:3]:
                Gp = G[:, prev]
                local = np.transpose(Gp, (0, 2, 1)) @ G[:, j]
                if not hv.any():
                    local[:] = np.eye(3)
                elif not hv.all():
                    idx = np.arange(n)
                    good = idx[hv]
                    near = good[np.clip(np.searchsorted(good, idx), 0, len(good) - 1)]
                    prv = good[np.clip(np.searchsorted(good, idx) - 1, 0, len(good) - 1)]
                    choose = np.where(np.abs(idx - prv) <= np.abs(near - idx), prv, near)
                    local[~hv] = local[choose[~hv]]
                G[:, j] = Gp @ local
                prev = j
        del Gh

    # Toe follows the foot; everything not driven copies its parent.
    for j in range(N_JOINTS):
        if j in B_pose:
            continue
        p = PARENTS[j]
        if p >= 0:
            G[:, j] = G[:, p]
    # re-orthonormalise
    u, _, vt = np.linalg.svd(G)
    G = u @ vt
    return G


def forward_kinematics(G, rest_pos):
    """World positions (N,127,3) in BVH frame (root joint 0 at origin)."""
    off = offsets_from_positions(rest_pos)
    n = G.shape[0]
    pos = np.zeros((n, N_JOINTS, 3))
    for j in range(N_JOINTS):
        p = PARENTS[j]
        if p < 0:
            pos[:, j] = off[j]
        else:
            pos[:, j] = pos[:, p] + np.einsum("nij,j->ni", G[:, p], off[j])
    return pos


def solve_camera_translation(pw_mp, pimg_px, vis, focal_px, cx, cy):
    """Per-frame hip-centre camera translation (SAM camera frame: x right,
    y down, z forward, metres) by linear least squares on the perspective
    projection of the hip-centred world landmarks."""
    n = pw_mp.shape[0]
    t = np.full((n, 3), np.nan)
    for f in range(n):
        w = vis[f] > 0.5
        if w.sum() < 6:
            continue
        X, Yy, Zz = pw_mp[f, w, 0], pw_mp[f, w, 1], pw_mp[f, w, 2]
        u = pimg_px[f, w, 0] - cx
        v = pimg_px[f, w, 1] - cy
        fl = focal_px[f] if np.ndim(focal_px) else focal_px
        k = w.sum()
        A = np.zeros((2 * k, 3))
        b = np.zeros(2 * k)
        A[:k, 0] = fl
        A[:k, 2] = -u
        b[:k] = u * Zz - fl * X
        A[k:, 1] = fl
        A[k:, 2] = -v
        b[k:] = v * Zz - fl * Yy
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        if sol[2] > 0.2:
            t[f] = sol
    valid = np.all(np.isfinite(t), axis=1)
    return fill_gaps(t, valid), valid


# MHR70 2D keypoint layout (see models/.../metadata/mhr70.py)
def mhr70_keypoints(pimg_px, himg_px, hvalid):
    """(N,70,2) pixel keypoints in the MHR70 order the overlay / hand solver use."""
    n = pimg_px.shape[0]
    k = np.zeros((n, 70, 2))
    P = pimg_px
    m = lambda i: P[:, i, :2]  # noqa: E731
    body = [MP["nose"], MP["eye_l"], MP["eye_r"], MP["ear_l"], MP["ear_r"], MP["sho_l"], MP["sho_r"],
            MP["elb_l"], MP["elb_r"], MP["hip_l"], MP["hip_r"], MP["knee_l"], MP["knee_r"],
            MP["ank_l"], MP["ank_r"]]
    for i, src in enumerate(body):
        k[:, i] = m(src)
    k[:, 15] = m(MP["toe_l"])
    k[:, 16] = m(MP["toe_l"])
    k[:, 17] = m(MP["heel_l"])
    k[:, 18] = m(MP["toe_r"])
    k[:, 19] = m(MP["toe_r"])
    k[:, 20] = m(MP["heel_r"])

    def hand_block(start, hi, wrist_pose):
        H = himg_px[:, hi, :, :2]
        v = hvalid[:, hi][:, None]
        # tip, first(distal), second, third(proximal) per finger
        order = [("thumb", [4, 3, 2, 1]), ("index", [8, 7, 6, 5]), ("middle", [12, 11, 10, 9]),
                 ("ring", [16, 15, 14, 13]), ("pinky", [20, 19, 18, 17])]
        c = start
        for _, ids in order:
            for i in ids:
                k[:, c] = np.where(v, H[:, i], wrist_pose)
                c += 1
        k[:, c] = np.where(v, H[:, 0], wrist_pose)  # wrist

    hand_block(21, 1, m(MP["wri_r"]))   # right hand 21..41
    hand_block(42, 0, m(MP["wri_l"]))   # left hand 42..62
    k[:, 63] = m(MP["elb_l"])
    k[:, 64] = m(MP["elb_r"])
    k[:, 65] = m(MP["elb_l"])
    k[:, 66] = m(MP["elb_r"])
    k[:, 67] = m(MP["sho_l"])
    k[:, 68] = m(MP["sho_r"])
    k[:, 69] = (m(MP["sho_l"]) + m(MP["sho_r"])) / 2
    return k


def build_body_npz_arrays(pw_mp, pvis, pvalid, pimg_px, hw_mp, himg_px, hvalid,
                          width, height, focal_px):
    """Everything save_mp_data.py writes, from per-frame MediaPipe results.

    pw_mp  (N,33,3) pose world landmarks (MP frame), NaN where missing
    pvis   (N,33)   visibility
    pvalid (N,)     pose detected
    pimg_px(N,33,2) pose image landmarks in pixels
    hw_mp  (N,2,21,3) hand world landmarks ([0]=subject-left, [1]=right)
    himg_px(N,2,21,2) hand image landmarks in pixels
    hvalid (N,2)
    """
    n = pw_mp.shape[0]
    # repair landmark gaps (per landmark, low visibility treated as missing)
    vis_ok = (pvis > 0.35) & pvalid[:, None]
    pw = np.empty_like(pw_mp, dtype=np.float64)
    pim = np.empty((n, 33, 2))
    for j in range(33):
        good = vis_ok[:, j]
        if good.sum() < max(2, int(0.05 * n)):
            good = pvalid.copy()
        pw[:, j] = fill_gaps(pw_mp[:, j], good)
        pim[:, j] = fill_gaps(pimg_px[:, j], pvalid)
    hw = np.nan_to_num(np.asarray(hw_mp, dtype=np.float64))
    him = np.nan_to_num(np.asarray(himg_px, dtype=np.float64))

    L = measure_lengths(pw, pvalid, hw.reshape(-1, 21, 3), hvalid.reshape(-1))
    parents, offsets_cm, prerot, rest_pos = skeleton_arrays(L)

    pw_bvh = pw @ _MP2BVH.T
    hw_bvh = hw @ _MP2BVH.T
    G = solve_global_rotations(pw_bvh, hw_bvh, hvalid.astype(bool), rest_pos)

    # SAM-style joint_coords: FK positions relative to the floor-root joint,
    # in camera axes (x right, y down, z forward).
    fk = forward_kinematics(G, rest_pos)                 # BVH frame, root at origin
    jc = fk @ _MP2BVH.T                                    # flip y,z (self-inverse)

    cx, cy = width / 2.0, height / 2.0
    t_hip, t_ok = solve_camera_translation(pw, pim, np.where(pvalid[:, None], pvis, 0.0),
                                           focal_px, cx, cy)
    # cam_t locates joint 0 (floor point under the hips): hip camera position
    # minus the hips' joint_coords.
    cam_t = t_hip - jc[:, HIPS]

    kp2d = mhr70_keypoints(np.concatenate([pim, np.zeros((n, 33, 1))], -1), him, hvalid.astype(bool))
    return dict(
        global_rots=G.astype(np.float32),
        joint_coords=jc.astype(np.float32),
        cam_t=cam_t.astype(np.float32),
        keypoints_2d=kp2d.astype(np.float32),
        skel_parents=parents,
        skel_offsets_cm=offsets_cm,
        skel_prerot=prerot,
        lengths_json=np.array(__import__("json").dumps(L)),
        cam_t_valid=t_ok,
    )


def default_skeleton():
    """Skeleton with average proportions (used when no capture is at hand,
    e.g. the T-pose template export)."""
    p, o, r, _ = skeleton_arrays(
        {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_LENGTHS.items()})
    return p, o, r
