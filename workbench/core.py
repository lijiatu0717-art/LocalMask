"""Deterministic, frame-indexed redaction model shared by preview and export.

V3（2026-11-10）：效果扩展为 像素化/磨砂/纯色/云朵/贴纸，支持不透明度、外扩、形状（矩形/椭圆/AI轮廓）。
旧工程字段（mosaic/solid/cover、padding、block）保持兼容。
"""
import json, math, re, hashlib
from pathlib import Path

EFFECTS = ['mosaic', 'blur', 'solid', 'cloud', 'sticker', 'cover']
SHAPES = ['rect', 'ellipse', 'contour']


def write_json(path, data):
    p = Path(path); tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); tmp.replace(p)


def validate_tracks(tracks, frames):
    if not isinstance(tracks, list) or len(tracks) > 2000: raise ValueError('区域数量不合法')
    ids = set()
    for t in tracks:
        if not isinstance(t, dict) or not isinstance(t.get('id'), str) or t['id'] in ids or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', t['id']): raise ValueError('区域ID不合法或重复')
        ids.add(t['id'])
        for k in ['start', 'end', 'padding', 'block']:
            if not isinstance(t.get(k), (int, float)) or not math.isfinite(t[k]): raise ValueError('区域参数必须为有限数字')
        if not 0 <= t['start'] <= t['end'] < frames: raise ValueError('时间范围超出视频')
        if t['start'] != int(t['start']) or t['end'] != int(t['end']): raise ValueError('必须使用整数帧')
        if any(t[k] != int(t[k]) for k in ['padding', 'block']): raise ValueError('外扩与强度必须使用整数')
        if not 0 <= t['padding'] <= 160 or not 2 <= t['block'] <= 100: raise ValueError('遮挡强度不合法')
        if t.get('effect') not in EFFECTS: raise ValueError('未知遮挡方式')
        if t.get('shape', 'rect') not in SHAPES: raise ValueError('未知遮挡形状')
        if t.get('shape') == 'contour' and not t.get('mask'): raise ValueError('只有 AI 抠出的物体可以用轮廓形状')
        if 'strength' in t and (not isinstance(t['strength'], int) or not 0 <= t['strength'] <= 100): raise ValueError('强度需在 0–100 之间')
        op = t.get('opacity', 100)
        if not isinstance(op, int) or not 10 <= op <= 100: raise ValueError('不透明度需在 10–100 之间')
        if t.get('effect') == 'sticker' and not re.fullmatch(r'[a-f0-9]{16}\.png', str(t.get('sticker', ''))): raise ValueError('贴纸文件不合法')
        color = t.get('color', [110, 110, 110])
        if not isinstance(color, list) or len(color) != 3 or any(not isinstance(c, int) or not 0 <= c <= 255 for c in color): raise ValueError('遮挡颜色不合法')
        if not isinstance(t.get('feather', 3), int) or not 0 <= t.get('feather', 3) <= 12: raise ValueError('柔边不合法')
        keys = t.get('keys', [])
        if not keys or len(keys) > frames: raise ValueError('需要位置关键帧')
        seen = set()
        for k in keys:
            f = k.get('frame'); b = k.get('box')
            if not isinstance(f, int) or f in seen or not 0 <= f < frames: raise ValueError('位置帧不合法')
            seen.add(f)
            if not isinstance(b, list) or len(b) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in b): raise ValueError('框坐标不合法')
            x, y, w, h = b
            if min(x, y) < -1e-6 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001: raise ValueError('遮挡框超出画面')
        t['keys'] = sorted(keys, key=lambda k: k['frame'])
    return tracks


def box_at(track, frame):
    if not track.get('enabled', True) or not track['start'] <= frame <= track['end']: return None
    keys = track['keys']; a = keys[0]; b = keys[-1]
    if frame <= a['frame']: return a['box'][:]
    if frame >= b['frame']: return b['box'][:]
    import bisect
    pos = bisect.bisect_right([k['frame'] for k in keys], frame); a, b = keys[pos - 1:pos + 1]
    if track.get('motion') == 'envelope':
        x = min(a['box'][0], b['box'][0]); y = min(a['box'][1], b['box'][1]); r = max(a['box'][0] + a['box'][2], b['box'][0] + b['box'][2]); bottom = max(a['box'][1] + a['box'][3], b['box'][1] + b['box'][3])
        return [x, y, r - x, bottom - y]
    ratio = (frame - a['frame']) / (b['frame'] - a['frame'])
    return [x + (y - x) * ratio for x, y in zip(a['box'], b['box'])]


HOLD = 6  # AI 轮廓短暂丢失时沿用上一形状的帧数（约0.2秒），超过则不画并记入待检查


def polys_at(mask_frames, frame):
    """mask_frames: {'123':[[...]]}；返回该帧（或 HOLD 帧内最近一帧）的轮廓。"""
    for d in range(HOLD + 1):
        v = mask_frames.get(str(frame - d))
        if v:
            return v
    return None


def pixel_rect(b, w, h, pad):
    x, y, bw, bh = b; l = max(0, math.floor(x * w) - pad); t = max(0, math.floor(y * h) - pad)
    r = min(w, math.ceil((x + bw) * w) + pad); bottom = min(h, math.ceil((y + bh) * h) + pad)
    return [int(l), int(t), int(r - l), int(bottom - t)]


def masks_at(project, frame, layers, mask_loader=None):
    w, h = project['width'], project['height']; result = []
    for layer in project.get('layers', []):
        if layer.get('enabled', True):
            result.extend({'box': b, 'effect': 'mosaic', 'block': 22, 'id': layer['id'], 'shape': 'rect', 'opacity': 100} for b in layers.get(layer['id'], {}).get(str(frame), []))
    for t in project['tracks']:
        if not t.get('enabled', True) or not t['start'] <= frame <= t['end']: continue
        polys = None
        if t.get('mask') and mask_loader:
            polys = polys_at(mask_loader(t['id']), frame)
            if not polys: continue  # 目标不在画面（或丢失，已记入待检查）
            xs = [p[i] for p in polys for i in range(0, len(p), 2)]; ys = [p[i] for p in polys for i in range(1, len(p), 2)]
            pad = t['padding']
            box = [max(0, min(xs) - pad), max(0, min(ys) - pad)]
            box += [min(w, max(xs) + 1 + pad) - box[0], min(h, max(ys) + 1 + pad) - box[1]]
        else:
            b = box_at(t, frame)
            if b is None: continue
            box = pixel_rect(b, w, h, t['padding'])
        shape = t.get('shape', 'rect')
        result.append({'box': box, 'effect': t['effect'], 'block': t['block'], 'id': t['id'], 'color': t.get('color', [20, 20, 20] if t['effect'] == 'solid' else [110, 110, 110]), 'feather': t.get('feather', 3),
                       'shape': shape if (shape != 'contour' or polys) else 'rect', 'polys': polys if shape == 'contour' else None, 'pad': t['padding'],
                       'opacity': t.get('opacity', 100), 'sticker': t.get('sticker'), 'strength': t.get('strength')})
    return result


def contiguous_ranges(frames):
    out = []
    for f in sorted(set(frames)):
        if out and f == out[-1][1] + 1: out[-1][1] = f
        else: out.append([f, f])
    return out


def apply_cover(image, box, color=(110, 110, 110), feather=3):
    """Opaque target rectangle, feather strictly outside it; fixed RGB color."""
    import numpy as np
    x, y, w, h = box; H, W = image.shape[:2]; bgr = np.array(color[::-1], dtype=np.float32)
    for d in range(feather, 0, -1):
        l = max(0, x - d); t = max(0, y - d); r = min(W, x + w + d); b = min(H, y + h + d)
        alpha = (feather - d + 1) / (feather + 1)
        image[t:b, l:r] = np.rint(image[t:b, l:r].astype(np.float32) * (1 - alpha) + bgr * alpha).astype(np.uint8)
    image[y:y + h, x:x + w] = bgr.astype(np.uint8)


# ---------------- V3 渲染 ----------------

_NOISE = {}


def _noise(seed, size=256):
    """稳定的云朵噪声纹理（同一物体每帧相同，不闪烁）。"""
    import numpy as np, cv2
    key = (seed, size)
    if key not in _NOISE:
        rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
        acc = np.zeros((size, size), np.float32)
        for octave, weight in [(8, .5), (16, .3), (32, .2)]:
            small = rng.random((octave, octave)).astype(np.float32)
            acc += cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC) * weight
        acc = (acc - acc.min()) / max(1e-6, acc.max() - acc.min())
        _NOISE[key] = acc
    return _NOISE[key]


def _shape_alpha(m, x0, y0, w, h):
    """ROI 内的形状覆盖（0..1 float32）。"""
    import numpy as np, cv2
    a = np.zeros((h, w), np.uint8)
    shape = m.get('shape', 'rect')
    if shape == 'contour' and m.get('polys'):
        pts = [np.array(p, np.int32).reshape(-1, 2) - [x0, y0] for p in m['polys']]
        cv2.fillPoly(a, pts, 255)
        pad = int(m.get('pad', 0))
        if pad > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            a = cv2.dilate(a, k)
    elif shape == 'ellipse':
        bx, by, bw, bh = m['box']
        cv2.ellipse(a, ((bx - x0 + bw / 2, by - y0 + bh / 2), (bw, bh), 0), 255, -1)
    else:
        bx, by, bw, bh = m['box']
        a[max(0, by - y0):by - y0 + bh, max(0, bx - x0):bx - x0 + bw] = 255
    return a.astype(np.float32) / 255


def render_masks(image, masks, sticker_loader=None):
    """就地把所有遮挡画到 BGR 图像上。预览（服务端截图）与导出共用。"""
    import numpy as np, cv2
    H, W = image.shape[:2]
    source = image.copy()  # 效果始终取自原画面，重叠区域不会互相"糊两遍"
    for m in masks:
        bx, by, bw, bh = [int(v) for v in m['box']]
        if bw <= 0 or bh <= 0: continue
        effect = m['effect']
        if effect == 'cover' and m.get('shape', 'rect') == 'rect':
            apply_cover(image, [bx, by, bw, bh], m.get('color', [110, 110, 110]), m.get('feather', 3)); continue
        margin = int(max(bw, bh) * 0.22) + 2 if effect == 'cloud' else (3 if effect in ['blur', 'cover'] else 0)
        x0, y0 = max(0, bx - margin), max(0, by - margin); x1, y1 = min(W, bx + bw + margin), min(H, by + bh + margin)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0: continue
        roi = source[y0:y1, x0:x1].astype(np.float32)
        alpha = _shape_alpha(m, x0, y0, w, h)
        block = max(2, int(m.get('block', 32)))
        s = m.get('strength')
        s = (max(0, min(100, int(s))) / 100.0) if s is not None else max(0.0, min(1.0, (block - 6) / 54))  # 旧工程按原颗粒折算
        side = max(4, min(bw, bh))
        if effect == 'mosaic':
            # 专业像素化：方形大格子、不透明、硬边；脸宽约 12 格（轻）到 3 格（重）
            cell = max(3, int(round(side / (12 - 9 * s))))
            sx, sy = -(-w // cell), -(-h // cell)
            small = cv2.resize(source[y0:y1, x0:x1], (sx, sy), interpolation=cv2.INTER_AREA)
            fx = cv2.resize(small, (sx * cell, sy * cell), interpolation=cv2.INTER_NEAREST)[:h, :w].astype(np.float32)
        elif effect == 'blur':
            # 专业隐私模糊：先压成少量色块彻底去掉细节，再大半径平滑；不提亮、不加噪点
            ext = int(side * 0.6)
            ex0, ey0, ex1, ey1 = max(0, x0 - ext), max(0, y0 - ext), min(W, x1 + ext), min(H, y1 + ext)
            big = source[ey0:ey1, ex0:ex1]
            cell = max(3, int(round(side / (9 - 5 * s))))
            small = cv2.resize(big, (max(1, (ex1 - ex0) // cell), max(1, (ey1 - ey0) // cell)), interpolation=cv2.INTER_AREA)
            up = cv2.resize(small, (ex1 - ex0, ey1 - ey0), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(up, (0, 0), max(3.0, side * (0.10 + 0.22 * s)))
            fx = blurred[y0 - ey0:y0 - ey0 + h, x0 - ex0:x0 - ex0 + w].astype(np.float32)
            alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5) if alpha.max() > 0 else alpha
            alpha = np.maximum(alpha, _shape_alpha(m, x0, y0, w, h))  # 柔边只往外，目标内部始终完全覆盖
        elif effect in ['solid', 'cover']:
            fx = np.empty_like(roi); fx[:] = np.array(m.get('color', [24, 24, 24])[::-1], np.float32)
            if effect == 'cover': alpha = np.maximum(alpha, cv2.GaussianBlur(alpha, (0, 0), max(1, m.get('feather', 3))))
        elif effect == 'cloud':
            n = cv2.resize(_noise(m.get('id', 'c')), (w, h), interpolation=cv2.INTER_LINEAR)
            base = cv2.GaussianBlur(alpha, (0, 0), max(2.0, min(bw, bh) * 0.12))
            soft = np.clip((base * (0.7 + 0.8 * n) - 0.18) * 2.2, 0, 1)
            alpha = np.maximum(soft, alpha)  # 目标本身始终完全覆盖
            shade = 236 + 19 * n
            fx = np.repeat(shade[..., None], 3, axis=2).astype(np.float32)
        elif effect == 'sticker':
            st = sticker_loader(m.get('sticker')) if sticker_loader else None
            if st is None:
                fx = np.empty_like(roi); fx[:] = 40
            else:
                sh, sw = st.shape[:2]; scale = max(bw / sw, bh / sh)
                rw, rh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
                rs = cv2.resize(st, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                cx, cy = (rw - bw) // 2, (rh - bh) // 2
                crop = rs[cy:cy + bh, cx:cx + bw]
                fx = roi.copy(); sa = np.zeros((h, w), np.float32)
                ox, oy = bx - x0, by - y0
                ch, cw = min(crop.shape[0], h - oy), min(crop.shape[1], w - ox)
                fx[oy:oy + ch, ox:ox + cw] = crop[:ch, :cw, :3].astype(np.float32)
                sa[oy:oy + ch, ox:ox + cw] = (crop[:ch, :cw, 3].astype(np.float32) / 255) if crop.shape[2] == 4 else 1
                alpha = sa
        else:
            continue
        alpha = alpha * (int(m.get('opacity', 100)) / 100.0)
        # 叠加到当前结果上（效果取自原图），多个物体互不抵消
        dst = image[y0:y1, x0:x1].astype(np.float32)
        a = alpha[..., None]
        image[y0:y1, x0:x1] = np.clip(dst * (1 - a) + fx * a, 0, 255).astype(np.uint8)
    return image
