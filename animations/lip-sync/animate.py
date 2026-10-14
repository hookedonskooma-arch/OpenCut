"""Talking-character animator: Rhubarb lip sync + scene switching between two characters."""
import sys, math, random, subprocess
import numpy as np, cv2
from PIL import Image

S = sys.argv[1]
AUDIO = sys.argv[2]
OUT = sys.argv[3]
PREVIEW = len(sys.argv) > 4 and sys.argv[4] == "preview"
W, H, FPS = 1080, 1920, 30
UP = 2  # supersample characters
random.seed(7)

# ---------- character data (native cutout coords) ----------
CHARS = [
    dict(file="char1.png", mouth=(187, 157), mw=44, tilt=0,
         eyes=[(158, 107, 24, 12), (215, 107, 24, 12)], face=(187, 120)),
    dict(file="char2.png", mouth=(250, 192), mw=38, tilt=6,
         eyes=[(233, 138, 22, 9), (283, 145, 18, 8)], face=(255, 150)),
]

# Rhubarb shape -> (jaw open, width factor, teeth, tongue)
SHAPES = {
    "X": (0.00, 1.00, 0, 0), "A": (0.00, 1.00, 0, 0),
    "B": (0.16, 0.95, 1, 0), "C": (0.36, 0.95, 1, 0),
    "D": (0.62, 1.00, 1, 1), "E": (0.42, 0.72, 1, 0),
    "F": (0.22, 0.55, 0, 0), "G": (0.14, 0.90, 2, 0),
    "H": (0.40, 0.90, 1, 2),
}

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def make_variant(base, c, shape, blink):
    img = base.copy()
    h, w = img.shape[:2]
    cx, cy = c["mouth"][0] * UP, c["mouth"][1] * UP
    MW = c["mw"] * UP
    op, wf, teeth, tongue = SHAPES[shape]
    d = op * MW * 0.45
    if d > 0.5:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # rotate into mouth-local frame
        a = math.radians(c["tilt"])
        lx = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
        ly = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
        wx = 1 - smooth(0.55 * MW, 1.25 * MW, np.abs(lx))
        wy = smooth(-0.06 * MW, 0.10 * MW, ly) * (1 - smooth(0.9 * MW, 1.9 * MW, ly))
        wgt = wx * wy
        mx = xx - d * wgt * math.sin(a) * -1
        my = yy - d * wgt * math.cos(a)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # mouth interior
        ccx = cx - d / 2 * math.sin(a)
        ccy = cy + d / 2 * math.cos(a)
        rx = MW * 0.42 * wf * (0.85 + 0.15 * op)
        ry = d * 0.62 + MW * 0.05
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (int(ccx), int(ccy)), (int(rx), int(ry)), c["tilt"], 0, 360, 1.0, -1, cv2.LINE_AA)
        layer = np.zeros((h, w, 3), np.float32)
        layer[:] = (30, 12, 13)
        if teeth:
            tm = np.zeros((h, w), np.float32)
            th = ry * (0.38 if teeth == 1 else 0.7)
            cv2.rectangle(tm, (int(ccx - rx), int(ccy - ry - 2)), (int(ccx + rx), int(ccy - ry + th)), 1.0, -1)
            tm = cv2.GaussianBlur(tm, (0, 0), 1.2)
            layer = layer * (1 - tm[..., None]) + np.array([196, 184, 168], np.float32) * tm[..., None]
            if teeth == 1 and op > 0.3:  # bottom teeth hint
                bm = np.zeros((h, w), np.float32)
                cv2.rectangle(bm, (int(ccx - rx * 0.7), int(ccy + ry * 0.72)), (int(ccx + rx * 0.7), int(ccy + ry + 2)), 0.8, -1)
                bm = cv2.GaussianBlur(bm, (0, 0), 1.5)
                layer = layer * (1 - bm[..., None]) + np.array([150, 138, 125], np.float32) * bm[..., None]
        if tongue:
            gm = np.zeros((h, w), np.float32)
            ty = ccy + ry * (0.45 if tongue == 1 else -0.05)
            cv2.ellipse(gm, (int(ccx), int(ty)), (int(rx * 0.6), int(ry * 0.5)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            gm = cv2.GaussianBlur(gm, (0, 0), 1.5)
            layer = layer * (1 - gm[..., None]) + np.array([128, 52, 56], np.float32) * gm[..., None]
        # inner shading: darker toward edges
        inner = cv2.GaussianBlur(mask, (0, 0), max(1.0, ry * 0.35))
        layer *= (0.55 + 0.45 * inner[..., None])
        mask = cv2.GaussianBlur(mask, (0, 0), 1.1)
        rgb = img[..., :3].astype(np.float32)
        rgb = rgb * (1 - mask[..., None]) + layer * mask[..., None]
        # subtle lip edge darkening around opening
        ring = np.clip(cv2.GaussianBlur(mask, (0, 0), 3.0) - mask, 0, 1) * 0.35
        rgb *= (1 - ring[..., None])
        img[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    elif shape == "F" or shape == "E":
        pass
    if blink:
        rgb = img[..., :3].astype(np.float32)
        for (ex, ey, ew, eh) in c["eyes"]:
            ex, ey, ew, eh = ex * UP, ey * UP, ew * UP, eh * UP
            # skin colour from just above the eye
            patch = img[int(ey + eh * 1.1):int(ey + eh * 1.7), int(ex - ew * 0.3):int(ex + ew * 0.3), :3]
            col = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32) * 0.88
            m = np.zeros(rgb.shape[:2], np.float32)
            cv2.ellipse(m, (int(ex), int(ey)), (int(ew * 0.62), int(eh * 0.85)), c["tilt"], 0, 360, 1.0, -1, cv2.LINE_AA)
            m = cv2.GaussianBlur(m, (0, 0), 1.6)
            rgb = rgb * (1 - m[..., None]) + col * m[..., None]
            l = np.zeros(rgb.shape[:2], np.float32)
            cv2.ellipse(l, (int(ex), int(ey + eh * 0.25)), (int(ew * 0.58), int(eh * 0.35)), c["tilt"], 20, 160, 1.0, 2, cv2.LINE_AA)
            l = cv2.GaussianBlur(l, (0, 0), 0.8) * 0.75
            rgb = rgb * (1 - l[..., None]) + np.array([40, 25, 20], np.float32) * l[..., None]
        img[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return img

# ---------- load characters & build variants ----------
for c in CHARS:
    im = Image.open(f"{S}/{c['file']}").convert("RGBA")
    im = im.resize((im.width * UP, im.height * UP), Image.LANCZOS)
    base = np.array(im)
    c["size"] = (base.shape[1], base.shape[0])
    c["var"] = {(s, b): make_variant(base, c, s, b) for s in SHAPES for b in (0, 1)}
    if PREVIEW:
        sheet = np.concatenate([c["var"][(s, 0)][: int(c["mouth"][1] * UP * 1.6), :, :3] for s in "XBCDEFGH"] + [c["var"][("X", 1)][: int(c["mouth"][1] * UP * 1.6), :, :3]], axis=1)
        Image.fromarray(sheet).save(f"{S}/sheet_{c['file']}")
if PREVIEW:
    sys.exit()

# ---------- backgrounds ----------
def background(kind):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == 0:  # cool modern studio
        top, bot, floor = np.array([28, 44, 72]), np.array([86, 120, 160]), np.array([52, 62, 80])
        bokeh_cols = [(150, 200, 255), (255, 255, 255), (120, 170, 240)]
    else:  # warm workshop
        top, bot, floor = np.array([30, 18, 12]), np.array([120, 70, 36]), np.array([62, 40, 26])
        bokeh_cols = [(255, 190, 110), (255, 150, 60), (255, 220, 160)]
    t = (yy / H)[..., None]
    img = top * (1 - t) + bot * t
    horizon = 1380
    fm = smooth(horizon - 10, horizon + 10, yy)[..., None]
    fl = floor * (1 - 0.35 * ((yy - horizon) / (H - horizon)).clip(0, 1))[..., None]
    img = img * (1 - fm) + fl * fm
    rng = np.random.default_rng(kind + 3)
    glow = np.zeros((H, W, 3), np.float32)
    for _ in range(38):
        x, y = rng.uniform(0, W), rng.uniform(0, horizon * 0.95)
        r = rng.uniform(18, 70)
        col = np.array(bokeh_cols[rng.integers(len(bokeh_cols))], np.float32)
        m = np.zeros((H, W), np.float32)
        cv2.circle(m, (int(x), int(y)), int(r), rng.uniform(0.08, 0.28), -1, cv2.LINE_AA)
        glow += m[..., None] * col
    glow = cv2.GaussianBlur(glow, (0, 0), 6)
    img = img + glow
    # spotlight behind character
    spot = np.exp(-(((xx - W / 2) / 420) ** 2 + ((yy - 900) / 700) ** 2))[..., None]
    img = img * (0.75 + 0.5 * spot)
    vig = 1 - 0.55 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    img = img * vig[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)

BG = [background(0), background(1)]

# ---------- timing ----------
mouth = [(float(t), s) for t, s in (l.split("\t") for l in open(f"{S}/mouth.tsv").read().strip().splitlines())]
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", AUDIO]))
NF = int(dur * FPS) + 1

def shape_at(t):
    s = "X"
    for tt, ss in mouth:
        if tt <= t:
            s = ss
        else:
            break
    return s

# audio loudness per frame (for emphasis bob)
pcm = np.frombuffer(subprocess.check_output(["ffmpeg", "-v", "error", "-i", f"{S}/voice.wav", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]), np.int16).astype(np.float32)
hop = 16000 // FPS
rms = np.array([np.sqrt(np.mean(pcm[i * hop:(i + 1) * hop] ** 2) + 1e-6) for i in range(NF)])
rms = np.convolve(rms / (rms.max() + 1e-6), np.ones(5) / 5, mode="same")

# scene cuts at speech pauses (silence midpoints), min ~5s apart
pauses = [4.07, 5.21, 12.29, 15.13, 18.76, 23.11, 30.18, 34.56, 37.68, 41.03, 48.95, 52.53, 58.73, 62.02]
cuts, last = [], 0.0
for p in pauses:
    if p - last >= 5.0:
        cuts.append(p - 0.15); last = p
scenes = []  # (start, end, char)
bounds = [0.0] + cuts + [dur]
for i in range(len(bounds) - 1):
    scenes.append((bounds[i], bounds[i + 1], i % 2))

# per-scene shot plan: camera framings (zoom, focus y fraction of face->feet)
FRAMINGS = {"wide": 1.0, "med": 1.9, "close": 3.0}
plans = []
for i, (a, b, ch) in enumerate(scenes):
    seq = [["wide", "med"], ["close", "med"], ["med", "close"], ["wide", "close"]][i % 4]
    mid = (a + b) / 2
    plans.append([(a, mid, seq[0]), (mid, b, seq[1])] if b - a > 6 else [(a, b, seq[0])])

# blinks
blinks = set()
t = 1.2
while t < dur:
    f0 = int(t * FPS)
    blinks.update(range(f0, f0 + 4))
    t += random.uniform(2.4, 5.0)

CHAR_H = 1450  # world px height of character in wide shot
FLOOR = 1760

def render_char_frame(bg, c, varimg, zoom, focus, t, f, push):
    cw, chh = c["size"]
    s0 = CHAR_H / chh
    breath = 1 + 0.006 * math.sin(2 * math.pi * t / 3.6)
    sway = 0.5 * math.sin(2 * math.pi * t / 5.3) + 0.6 * rms[min(f, NF - 1)] * math.sin(2 * math.pi * t / 0.9)
    bob = -6 * rms[min(f, NF - 1)]
    z = zoom * push
    # char local -> world: anchor feet center at (W/2, FLOOR)
    ang = math.radians(sway)
    sx, sy = s0, s0 * breath
    # build matrix: translate(-cw/2,-chh) -> scale -> rotate -> translate(W/2, FLOOR+bob)
    M = np.array([[sx * math.cos(ang), -sy * math.sin(ang), 0], [sx * math.sin(ang), sy * math.cos(ang), 0], [0, 0, 1]])
    T0 = np.array([[1, 0, -cw / 2], [0, 1, -chh], [0, 0, 1]])
    T1 = np.array([[1, 0, W / 2], [0, 1, FLOOR + bob], [0, 0, 1]])
    world = T1 @ M @ T0
    # camera: world -> screen
    fx, fy = focus
    cam = np.array([[z, 0, W / 2 - z * fx], [0, z, H * 0.40 - z * fy], [0, 0, 1]]) if zoom > 1.01 else np.array([[z, 0, W / 2 - z * W / 2], [0, z, H / 2 - z * H / 2], [0, 0, 1]])
    full = (cam @ world)[:2]
    bgw = cv2.warpAffine(bg, cam[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if zoom > 1.01:
        bgw = cv2.GaussianBlur(bgw, (0, 0), 2 + zoom)
    out = bgw.astype(np.float32)
    # floor shadow
    sh = np.zeros((H, W), np.float32)
    p = cam @ np.array([W / 2, FLOOR + 8, 1])
    cv2.ellipse(sh, (int(p[0]), int(p[1])), (int(190 * z), int(26 * z)), 0, 0, 360, 0.55, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 14 * z)
    out *= (1 - sh[..., None])
    ch = cv2.warpAffine(varimg, full, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = ch[..., 3:4].astype(np.float32) / 255
    out = out * (1 - a) + ch[..., :3].astype(np.float32) * a
    return out, world

def face_world(c, world):
    p = world @ np.array([c["face"][0] * UP, c["face"][1] * UP, 1])
    return p[:2]

def frame_at(f):
    t = f / FPS
    si = next(i for i, (a, b, _) in enumerate(scenes) if a <= t < b or i == len(scenes) - 1)
    a, b, chn = scenes[si]
    c = CHARS[chn]
    shot = next(s for s in plans[si] if s[0] <= t < s[1] or s is plans[si][-1])
    zoom = FRAMINGS[shot[2]]
    prog = (t - shot[0]) / max(0.1, shot[1] - shot[0])
    push = 1 + 0.06 * prog  # slow push-in
    var = c["var"][(shape_at(t), 1 if f in blinks else 0)]
    # compute face focus with neutral world (approx)
    cw, chh = c["size"]
    s0 = CHAR_H / chh
    fx = W / 2 + (c["face"][0] * UP - cw / 2) * s0
    fy = FLOOR + (c["face"][1] * UP - chh) * s0
    if shot[2] == "med":
        fy = fy + 230  # frame head + torso
    out, _ = render_char_frame(BG[chn], c, var, zoom, (fx, fy), t, f, push)
    # scene transition: flash + zoom blur in first 0.25s of each scene (except first)
    since = t - a
    if si > 0 and since < 0.25:
        k = 1 - since / 0.25
        out = out * (1 - 0.6 * k) + 255 * 0.6 * k
    return out

# grain/vignette overlay
grain_rng = np.random.default_rng(1)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
prev = None
for f in range(NF):
    fr = frame_at(f)
    # whip-crossfade from previous scene's last frame
    t = f / FPS
    fr += grain_rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)
    ff.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
    if f % 300 == 0:
        print(f, "/", NF, flush=True)
ff.stdin.close(); ff.wait()
print("done", OUT)
