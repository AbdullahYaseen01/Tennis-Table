from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision.ball_profiles import get_profile

OUT = ROOT / "data" / "reports" / "client-screenshots"
DL = Path.home() / "Downloads" / "pickleball-client-screenshots"

N_ANG = 360

def _masks(frame: np.ndarray):
    
    profile = get_profile("pickleball")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = profile.hsv_lower.copy(), profile.hsv_upper.copy()
    
    lo[1] = max(int(lo[1]) - 25, 70)
    lo[2] = max(int(lo[2]) - 40, 80)
    yellow = cv2.inRange(hsv, lo, hi)
    
    yellow |= cv2.inRange(hsv, (18, 60, 70), (45, 255, 255))

    skin = cv2.inRange(hsv, (0, 40, 50), (25, 180, 255))
    skin |= cv2.inRange(hsv, (160, 40, 50), (180, 180, 255))
    
    skin = cv2.bitwise_and(skin, cv2.bitwise_not(yellow))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    
    cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        filled = np.zeros_like(yellow)
        big = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(big) > 5000:
            cv2.drawContours(filled, [big], -1, 255, -1)
            yellow = filled

    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k)
    skin = cv2.dilate(skin, k, iterations=2)
    return yellow, skin

def find_ball(frame: np.ndarray):
    h, w = frame.shape[:2]
    yellow, skin = _masks(frame)
    
    mask = cv2.bitwise_and(yellow, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(mask, k, iterations=1)
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 5000:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.45:
            continue
        (x, y), rad = cv2.minEnclosingCircle(c)
        if y > h * 0.92 or rad < 55:
            continue
        score = area * circ
        if score > best_score:
            best_score, best = score, (float(x), float(y), float(rad), yellow, skin)
    return best

def polar_radii(mask: np.ndarray, cx: float, cy: float, r_max: float) -> np.ndarray:
    h, w = mask.shape[:2]
    radii = np.zeros(N_ANG, dtype=np.float64)
    steps = max(int(r_max * 1.4), 80)
    for i in range(N_ANG):
        ang = 2.0 * np.pi * i / N_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        last = 0.0
        for t in np.linspace(0.20 * r_max, 1.30 * r_max, steps):
            x = int(round(cx + t * ca))
            y = int(round(cy + t * sa))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            if mask[y, x] > 0:
                last = t
            elif last > 0 and t > last + 4:
                break
        radii[i] = last
    for i in range(N_ANG):
        if radii[i] <= 1:
            prev, nxt = radii[(i - 1) % N_ANG], radii[(i + 1) % N_ANG]
            if prev > 1 and nxt > 1:
                radii[i] = 0.5 * (prev + nxt)
    
    pad = np.r_[radii[-9:], radii, radii[:9]]
    return np.convolve(pad, np.ones(19) / 19.0, mode="valid")

def skin_press_peak(cx, cy, r_new, skin) -> tuple[int | None, np.ndarray]:
    
    h, w = skin.shape[:2]
    strength = np.zeros(N_ANG, dtype=np.float64)
    ys, xs = np.where(skin > 0)
    if len(xs) < 30:
        return None, strength
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    dist = np.sqrt(dx * dx + dy * dy)
    
    ring = (dist >= r_new * 0.88) & (dist <= r_new * 1.35)
    if not np.any(ring):
        return None, strength
    ang = (np.arctan2(dy[ring], dx[ring]) + 2 * np.pi) % (2 * np.pi)
    bins = (ang / (2 * np.pi) * N_ANG).astype(np.int32) % N_ANG
    for b in bins:
        strength[b] += 1.0
    
    pad = np.r_[strength[-12:], strength, strength[:12]]
    strength = np.convolve(pad, np.ones(25) / 25.0, mode="valid")
    peak = int(np.argmax(strength))
    if strength[peak] < 3.0:
        return None, strength
    return peak, strength

def analyze(frame: np.ndarray):
    
    hit = find_ball(frame)
    if hit is None:
        return None
    cx, cy, r_enc, yellow, skin = hit
    radii_raw = polar_radii(yellow, cx, cy, r_enc)

    valid = radii_raw > 0.25 * r_enc
    if valid.sum() < 40:
        return None

    r_new = float(np.percentile(radii_raw[valid], 90))
    r_new = float(np.clip(r_new, 0.85 * r_enc, 1.05 * r_enc))

    deficit = np.maximum(0.0, r_new - radii_raw)
    deficit[~valid] = 0.0

    skin_peak, skin_str = skin_press_peak(cx, cy, r_new, skin)
    zone = np.zeros(N_ANG, dtype=bool)
    peak = 0
    press_pct = 0.0

    if skin_peak is not None:
        
        skin_n = skin_str / max(float(skin_str.max()), 1e-6)
        def_n = deficit / max(float(r_new), 1.0)
        joint = skin_n * def_n
        
        pad = np.r_[joint[-8:], joint, joint[:8]]
        joint = np.convolve(pad, np.ones(17) / 17.0, mode="valid")

        deep_thr = max(0.08 * r_new, 10.0)
        
        cand_mask = (skin_str >= max(0.20 * skin_str[skin_peak], 1.2)) & (deficit >= deep_thr) & valid
        if np.any(cand_mask):
            peak = int(np.argmax(np.where(cand_mask, joint, -1.0)))
            dang2 = np.minimum(
                np.abs(np.arange(N_ANG) - peak),
                N_ANG - np.abs(np.arange(N_ANG) - peak),
            )
            
            zone = (
                (dang2 <= 18)
                & (deficit >= max(0.55 * deficit[peak], deep_thr * 0.65))
                & (skin_str >= max(0.18 * skin_str[peak], 1.0))
                & valid
            )
            zone[peak] = True

            if zone.sum() > 42:
                zone = (dang2 <= 15) & zone

            zdef = deficit[zone]
            press_pct = float(np.percentile(zdef, 65) / r_new * 100.0) if np.any(zone) else 0.0
            press_pct = float(np.clip(press_pct, 0.0, 40.0))

            
            if press_pct < 7.0 or zone.sum() < 10:
                zone[:] = False
                press_pct = 0.0
        else:
            zone[:] = False
            press_pct = 0.0

    
    if np.any(zone):
        zone = _largest_arc(zone)
        if zone.sum() < 8:
            zone[:] = False
            press_pct = 0.0
        else:
            peak = int(np.argmax(np.where(zone, deficit, -1.0)))
            zdef = deficit[zone]
            press_pct = float(np.percentile(zdef, 65) / r_new * 100.0)
            press_pct = float(np.clip(press_pct, 0.0, 40.0))

    
    
    
    radii_disp = np.full(N_ANG, r_new, dtype=np.float64)
    if np.any(zone):
        radii_disp[zone] = np.minimum(radii_raw[zone], r_new)
        
        edge = zone.copy()
        for j in np.where(zone)[0]:
            for d in (1, 2):
                for s in (-1, 1):
                    k = (j + s * d) % N_ANG
                    if zone[k]:
                        continue
                    a = 1.0 - d / 3.0
                    radii_disp[k] = a * min(radii_raw[k], r_new) + (1 - a) * r_new
                    edge[k] = True

    outline = _outline_from_radii(cx, cy, radii_disp)
    return cx, cy, r_new, outline, zone, press_pct, peak, skin

def _largest_arc(zone: np.ndarray) -> np.ndarray:
    
    n = len(zone)
    if not np.any(zone):
        return zone
    ext = np.r_[zone, zone]
    best_len, best_start = 0, 0
    i = 0
    while i < len(ext):
        if not ext[i]:
            i += 1
            continue
        j = i
        while j < len(ext) and ext[j]:
            j += 1
        length = j - i
        if length > best_len:
            best_len, best_start = length, i
        i = j
    out = np.zeros(n, dtype=bool)
    for t in range(best_start, best_start + best_len):
        out[t % n] = True
    
    if out.sum() > n // 3:
        
        mid = (best_start + best_len // 2) % n
        dang = np.minimum(np.abs(np.arange(n) - mid), n - np.abs(np.arange(n) - mid))
        out = dang <= 25
    return out

def _outline_from_radii(cx, cy, radii):
    ang = 2.0 * np.pi * np.arange(N_ANG) / N_ANG
    return np.column_stack([cx + radii * np.cos(ang), cy + radii * np.sin(ang)])

def grade(pct: float):
    if pct < 8:
        return "A", "GOOD / NEW", (40, 180, 70), "No meaningful hand press"
    if pct < 18:
        return "B", "USED / MONITOR", (0, 165, 255), "Press depth 8-18%"
    return "C", "REPLACE", (40, 50, 230), "Press depth > 18%"

def _put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def render(frame: np.ndarray, caption: str) -> tuple[np.ndarray, float, str] | None:
    res = analyze(frame)
    if res is None:
        return None
    cx, cy, r_new, outline, zone, pct, peak, _skin = res
    gcode, glabel, gcolor, grule = grade(pct)

    
    if pct < 0.5:
        zone = np.zeros(N_ANG, dtype=bool)
        pct = 0.0
        gcode, glabel, gcolor, grule = grade(0.0)

    h, w = frame.shape[:2]
    scale = 2 if max(h, w) < 1400 else 1
    if scale != 1:
        frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        cx, cy, r_new = cx * scale, cy * scale, r_new * scale
        outline = outline * scale
        h, w = frame.shape[:2]

    out = frame.copy()
    pts = outline

    
    if np.any(zone):
        layer = out.copy()
        z_idx = np.where(zone)[0]
        order = sorted(z_idx, key=lambda i: (i - int(peak) + N_ANG // 2) % N_ANG)
        dent = pts[order]
        angs = 2.0 * np.pi * np.asarray(order) / N_ANG
        arc = np.column_stack([cx + r_new * np.cos(angs), cy + r_new * np.sin(angs)])
        poly = np.vstack([dent, arc[::-1]]).astype(np.int32)
        cv2.fillPoly(layer, [poly], (35, 35, 230))
        cv2.addWeighted(layer, 0.28, out, 0.72, 0, out)

    
    cv2.circle(out, (int(cx), int(cy)), int(round(r_new)), (255, 210, 40), 4, cv2.LINE_AA)
    
    if np.any(zone):
        cv2.polylines(out, [pts.astype(np.int32)], True, (40, 230, 80), 2, cv2.LINE_AA)
    else:
        
        cv2.circle(out, (int(cx), int(cy)), int(round(r_new)), (40, 230, 80), 2, cv2.LINE_AA)

    if np.any(zone):
        for a in range(N_ANG):
            b = (a + 1) % N_ANG
            if zone[a] and zone[b]:
                p1 = (int(pts[a, 0]), int(pts[a, 1]))
                p2 = (int(pts[b, 0]), int(pts[b, 1]))
                cv2.line(out, p1, p2, (30, 30, 255), 7, cv2.LINE_AA)

        ang = 2.0 * np.pi * int(peak) / N_ANG
        ux, uy = float(np.cos(ang)), float(np.sin(ang))
        px, py = float(pts[int(peak), 0]), float(pts[int(peak), 1])
        tip = (int(px + ux * 14), int(py + uy * 14))
        label_pt = (
            int(np.clip(px + ux * 120, 200, w - 40)),
            int(np.clip(py + uy * 120, 270, h - 60)),
        )
        
        if label_pt[1] < 220:
            label_pt = (label_pt[0], 240)
        cv2.arrowedLine(out, label_pt, tip, (30, 30, 255), 3, cv2.LINE_AA, tipLength=0.2)
        text = "HAND PRESS = deformed zone"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        lx = label_pt[0] - tw - 12 if label_pt[0] > w * 0.55 else label_pt[0] + 12
        ly = label_pt[1]
        lx = int(np.clip(lx, 16, w - tw - 20))
        cv2.rectangle(out, (lx - 10, ly - th - 12), (lx + tw + 10, ly + 12), (18, 18, 18), -1)
        cv2.rectangle(out, (lx - 10, ly - th - 12), (lx + tw + 10, ly + 12), (30, 30, 255), 2)
        _put(out, text, (lx, ly), 0.62, (90, 90, 255), 2)

        outer = (int(cx + r_new * ux), int(cy + r_new * uy))
        cv2.line(out, outer, (int(px), int(py)), (0, 220, 255), 2, cv2.LINE_AA)
    else:
        
        note = "No hand-press deformation detected"
        (tw, th), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        nx, ny = 24, h - 36
        cv2.rectangle(out, (nx - 8, ny - th - 10), (nx + tw + 8, ny + 10), (20, 40, 20), -1)
        cv2.rectangle(out, (nx - 8, ny - th - 10), (nx + tw + 8, ny + 10), (40, 180, 70), 2)
        _put(out, note, (nx, ny), 0.60, (80, 220, 100), 2)

    cv2.drawMarker(out, (int(cx), int(cy)), (210, 210, 210), cv2.MARKER_CROSS, 18, 2)

    bar_h = 168
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)

    _put(out, "NEW vs USED  |  RED = real hand-press only", (24, 36), 0.72, (245, 245, 245), 2)
    _put(out, caption, (24, 66), 0.48, (170, 170, 170), 1)
    cv2.circle(out, (36, 100), 8, (255, 210, 40), -1, cv2.LINE_AA)
    _put(out, "CYAN  = brand-new ball circumference (reference)", (54, 105), 0.48, (255, 210, 40), 1)
    cv2.circle(out, (36, 128), 8, (40, 230, 80), -1, cv2.LINE_AA)
    _put(out, "GREEN = actual ball outline", (54, 133), 0.48, (40, 230, 80), 1)
    cv2.circle(out, (36, 156), 8, (30, 30, 255), -1, cv2.LINE_AA)
    _put(out, "RED   = hand-pressed deformed zone ONLY (requires finger contact)", (54, 161), 0.46, (80, 80, 255), 1)

    bx0, bx1 = w - 310, w - 18
    by0, by1 = 14, 154
    cv2.rectangle(out, (bx0, by0), (bx1, by1), (28, 28, 28), -1)
    cv2.rectangle(out, (bx0, by0), (bx1, by1), gcolor, 3)
    _put(out, f"Press depth  {pct:.1f}%", (bx0 + 14, by0 + 38), 0.70, (255, 255, 255), 2)
    _put(out, f"GRADE {gcode}", (bx0 + 14, by0 + 78), 0.95, gcolor, 2)
    _put(out, glabel, (bx0 + 14, by0 + 108), 0.55, gcolor, 2)
    _put(out, grule, (bx0 + 14, by0 + 132), 0.42, (190, 190, 190), 1)

    return out, pct, f"GRADE {gcode}"

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DL.mkdir(parents=True, exist_ok=True)
    for old in list(DL.glob("*.jpg")) + list(OUT.glob("*.jpg")):
        old.unlink()

    video = ROOT / "data" / "samples" / "testing ball.mp4"
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scored = []
    for fi in range(0, n, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        res = analyze(frame)
        if res is None:
            continue
        pct, zone_n = res[5], int(res[4].sum())
        scored.append((pct, fi, zone_n, grade(pct)[0]))
    scored.sort()
    print("grade counts", {g: sum(1 for s in scored if s[3] == g) for g in "ABC"})

    by = {fi: s for s in scored for fi in [s[1]]}

    def grade_of(fi):
        return by[fi][3] if fi in by else None

    def zone_of(fi):
        return by[fi][2] if fi in by else 0

    a_zero = [s[1] for s in scored if s[3] == "A" and s[2] == 0]
    a1 = a_zero[0] if a_zero else scored[0][1]
    a2 = a_zero[1] if len(a_zero) > 1 else a1

    
    curated_b = [435, 417, 70, 100, 444, 50]
    curated_c = [138, 174, 168, 180, 160, 150]
    b_auto = [s[1] for s in scored if s[3] == "B" and s[2] >= 12]
    c_auto = [s[1] for s in scored if s[3] == "C" and 18 <= s[0] < 35 and s[2] >= 12]

    def pick_from(curated, auto, want, exclude=()):
        for fi in curated:
            if fi in exclude:
                continue
            if grade_of(fi) == want and zone_of(fi) >= 10:
                return fi
        for fi in auto:
            if fi not in exclude:
                return fi
        return auto[0] if auto else None

    b1 = pick_from(curated_b, b_auto, "B")
    b2 = pick_from(curated_b, b_auto, "B", exclude={b1})
    c1 = pick_from(curated_c, c_auto, "C")
    c2 = pick_from(curated_c, c_auto, "C", exclude={c1})

    used = set()
    def uniq(fi, fallbacks):
        if fi is not None and fi not in used:
            used.add(fi)
            return fi
        for f in fallbacks:
            if f is not None and f not in used:
                used.add(f)
                return f
        return fi

    a1 = uniq(a1, a_zero)
    a2 = uniq(a2, a_zero)
    b1 = uniq(b1, b_auto)
    b2 = uniq(b2, b_auto)
    c1 = uniq(c1, c_auto)
    c2 = uniq(c2, c_auto)

    print("picks", dict(a1=a1, a2=a2, b1=b1, b2=b2, c1=c1, c2=c2))

    picks = [
        (a1, "01-GRADE-A-good-match", "Grade A: no hand-press dent - outline matches NEW circle"),
        (a2, "02-GRADE-A-near-round", "Grade A: round match - RED only if finger actually presses"),
        (b1, "03-GRADE-B-press-marked", "Grade B: RED only where finger presses the ball"),
        (b2, "04-GRADE-B-indent", "Grade B: thumb contact indent marked in red"),
        (c1, "05-GRADE-C-deep-press", "Grade C: deep hand press - red = press zone only"),
        (c2, "06-GRADE-C-replace", "Grade C: deep press - replace recommended"),
        (b1, "07-METHOD-explained", "Method: RED requires real finger contact (not shadow)"),
    ]

    for fi, name, caption in picks:
        if fi is None:
            print("skip", name)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print("missing", fi)
            continue
        rendered = render(frame, caption)
        if rendered is None:
            print("no detect", fi)
            continue
        img, pct, g = rendered
        path = OUT / f"{name}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        shutil.copy2(path, DL / path.name)
        print(f"{path.name}: {pct:.1f}% {g} (frame {fi})")
    cap.release()

    trio_files = ["01-GRADE-A-good-match.jpg", "03-GRADE-B-press-marked.jpg", "05-GRADE-C-deep-press.jpg"]
    trio = []
    for fn in trio_files:
        p = OUT / fn
        if p.exists():
            trio.append(cv2.resize(cv2.imread(str(p)), (520, 920)))
    if len(trio) == 3:
        strip = np.hstack(trio)
        hdr = np.zeros((110, strip.shape[1], 3), dtype=np.uint8)
        hdr[:] = (18, 18, 18)
        _put(hdr, "RED only where a finger actually presses  |  shadows are NOT marked", (20, 42), 0.78, (0, 255, 180), 2)
        _put(hdr, "Left GRADE A  |  Center GRADE B  |  Right GRADE C", (20, 82), 0.55, (200, 200, 200), 1)
        strip = np.vstack([hdr, strip])
        sp = OUT / "00-FULL-SUMMARY-3-GRADES.jpg"
        cv2.imwrite(str(sp), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        shutil.copy2(sp, DL / sp.name)
        shutil.copy2(sp, DL / "CLIENT-ANSWER-new-vs-old-3grades.jpg")
        overlay = ROOT / "data" / "reports" / "client-overlay"
        overlay.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, overlay / "CLIENT-ANSWER-new-vs-old-3grades.jpg")
        print("summary", sp.name)

    print("\nFOLDER:", DL)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
