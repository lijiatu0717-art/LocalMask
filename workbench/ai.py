"""本机 AI 抠图：SAM 2.1（点选/框选抠图 + 视频追踪）与 Grounding DINO（一句话识别）。

模型放在 ROOT/.cache/ai-models，只读本地文件，不联网。全部懒加载：没装 torch 时工作台其他功能照常可用。
"""
import json, math, os, re, threading, time
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # 个别算子 Apple 芯片不支持时退回 CPU

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / '.cache' / 'ai-models'
SAM_DIR = MODELS / 'sam2.1-hiera-tiny'
DINO_DIR = MODELS / 'grounding-dino-tiny'
_LOCK = threading.RLock()
_STATE = {}

# 常用中文说法 → 检测模型能理解的英文。顺序有意义：长词优先。
WORDS = [
    ('人脸', 'face'), ('脸', 'face'), ('头像', 'face'), ('行人', 'person'), ('路人', 'person'), ('人物', 'person'), ('人', 'person'),
    ('车牌', 'license plate'), ('号牌', 'license plate'), ('二维码', 'qr code'), ('条形码', 'barcode'),
    ('手机屏幕', 'phone screen'), ('电脑屏幕', 'computer screen'), ('屏幕', 'screen'), ('显示器', 'monitor'), ('手机', 'cell phone'),
    ('笔记本电脑', 'laptop'), ('电脑', 'laptop'), ('键盘', 'keyboard'),
    ('商标', 'logo'), ('标志', 'logo'), ('图标', 'app icon'), ('logo', 'logo'), ('水印', 'watermark'),
    ('招牌', 'signboard'), ('广告牌', 'billboard'), ('海报', 'poster'), ('文字', 'text'), ('字幕', 'subtitle text'),
    ('证件', 'id card'), ('身份证', 'id card'), ('名片', 'business card'), ('银行卡', 'bank card'), ('卡片', 'card'), ('文件', 'document'), ('纸', 'paper'),
    ('杯子', 'cup'), ('咖啡杯', 'coffee cup'), ('奶茶', 'drink cup'), ('瓶子', 'bottle'), ('饮料', 'drink bottle'),
    ('汽车', 'car'), ('车', 'car'), ('自行车', 'bicycle'), ('电动车', 'scooter'), ('摩托车', 'motorcycle'),
    ('包', 'bag'), ('帽子', 'hat'), ('衣服', 'clothes'), ('鞋', 'shoe'), ('手', 'hand'), ('门牌', 'house number sign'),
    ('猫', 'cat'), ('狗', 'dog'), ('窗户', 'window'), ('门', 'door'), ('电视', 'tv'),
]


def translate(text):
    """把『人脸、车牌』这类描述拆成检测词。返回 ([(显示名, 英文词)], 不认识的片段)。"""
    parts = [x.strip() for x in re.split(r'[,，、;；/\n]|和|以及|还有', text or '') if x.strip()]
    found, unknown = [], []
    for part in parts:
        low = part.lower()
        if re.fullmatch(r'[a-z0-9 \-_.]+', low):
            found.append((part, low.strip(' .')))
            continue
        hit = next((en for zh, en in WORDS if zh in low), None)
        if hit:
            found.append((part, hit))
        else:
            unknown.append(part)
    seen, words = [], []
    for label, w in found:
        if w and w not in words:
            words.append(w); seen.append((label, w))
    return seen, unknown


def status():
    info = {'models': SAM_DIR.joinpath('model.safetensors').is_file() and DINO_DIR.joinpath('model.safetensors').is_file()}
    try:
        import torch, transformers  # noqa: F401
        info['runtime'] = True
        info['device'] = device_name()
    except Exception as e:  # pragma: no cover - depends on install
        info['runtime'] = False
        info['error'] = str(e)
    info['ready'] = bool(info['models'] and info['runtime'])
    info['loaded'] = sorted(_STATE)
    return info


_THREADS = []


def _threads():
    if not _THREADS:
        import torch
        torch.set_num_threads(max(2, (os.cpu_count() or 4) - 2))  # 留出 CPU 给页面和解码，界面不卡
        _THREADS.append(1)


def device_name():
    import torch
    forced = os.environ.get('MOSAIC_AI_DEVICE')
    if not forced and (MODELS / 'device.txt').is_file():  # 安装脚本测出 Apple 芯片加速不可用时写 cpu
        forced = (MODELS / 'device.txt').read_text().strip()
    if forced in ('cpu', 'mps'):
        return forced
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _require():
    s = status()
    if not s['ready']:
        raise ValueError('AI 抠图还没安装好：请重新运行一次安装命令，完成后重新打开工作台')


def _sam_image():
    with _LOCK:
        if 'sam' not in _STATE:
            _require(); _threads()
            import torch
            from transformers import Sam2Model, Sam2Processor
            dev = device_name()
            model = Sam2Model.from_pretrained(SAM_DIR, local_files_only=True).to(dev).eval()
            _STATE['sam'] = (model, Sam2Processor.from_pretrained(SAM_DIR, local_files_only=True), dev)
        return _STATE['sam']


def _dino():
    with _LOCK:
        if 'dino' not in _STATE:
            _require(); _threads()
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            dev = device_name()
            model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_DIR, local_files_only=True).to(dev).eval()
            _STATE['dino'] = (model, AutoProcessor.from_pretrained(DINO_DIR, local_files_only=True), dev)
        return _STATE['dino']


def _sam_video():
    with _LOCK:
        if 'video' not in _STATE:
            _require(); _threads()
            from transformers import Sam2VideoModel, Sam2VideoProcessor
            import torch
            dev = device_name()
            dtype = torch.float16 if dev == 'mps' and os.environ.get('MOSAIC_AI_DTYPE', 'fp16') == 'fp16' else torch.float32
            model = Sam2VideoModel.from_pretrained(SAM_DIR, local_files_only=True).to(dev, dtype=dtype).eval()
            _STATE['video'] = (model, Sam2VideoProcessor.from_pretrained(SAM_DIR, local_files_only=True), dev)
            _STATE['video_dtype'] = dtype
            # 记忆帧 7→3：在 Apple 芯片上"对比记忆"占了大部分时间，减到 3 帧约快一半，准确度略降
            _STATE['video_mem'] = max(1, min(model.config.num_maskmem, int(os.environ.get('MOSAIC_AI_MEM', '3'))))
            model.num_maskmem = _STATE['video_mem']
        return _STATE['video']


# ---------- 掩码 <-> 多边形 ----------

def mask_to_polys(mask, min_area=24, eps=1.2, max_points=160):
    """二值掩码 → 外轮廓多边形（像素坐标扁平列表）。"""
    import cv2, numpy as np
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(c) < min_area:
            continue
        e = eps
        a = cv2.approxPolyDP(c, e, True)
        while len(a) > max_points:
            e *= 1.6
            a = cv2.approxPolyDP(c, e, True)
        if len(a) >= 3:
            polys.append([int(v) for v in a.reshape(-1)])
    return polys


def polys_bbox(polys, w, h):
    xs = [p[i] for p in polys for i in range(0, len(p), 2)]
    ys = [p[i] for p in polys for i in range(1, len(p), 2)]
    if not xs:
        return None
    x0, y0, x1, y1 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs) + 1), min(h, max(ys) + 1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [round(x0 / w, 5), round(y0 / h, 5), round((x1 - x0) / w, 5), round((y1 - y0) / h, 5)]


def polys_to_mask(polys, w, h):
    import cv2, numpy as np
    m = np.zeros((h, w), np.uint8)
    if polys:
        cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 2) for p in polys], 255)
    return m


# ---------- 单帧：点选 / 框选 ----------

_EMBED = {}


def segment_points(image_bgr, points, box=None, cache_key=None):
    """points: [[x,y,label]] 像素坐标，label 1=加入 0=去掉。box: [x0,y0,x1,y1] 像素。返回多边形列表。"""
    import torch, cv2, numpy as np
    model, proc, dev = _sam_image()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    kwargs = {}
    if points:
        kwargs['input_points'] = [[[[float(p[0]), float(p[1])] for p in points]]]
        kwargs['input_labels'] = [[[int(p[2]) for p in points]]]
    if box is not None:
        kwargs['input_boxes'] = [[[float(v) for v in box]]]
    inputs = proc(images=rgb, return_tensors='pt', **kwargs).to(dev)
    with torch.inference_mode():
        emb = _EMBED.get(cache_key) if cache_key else None
        if emb is None:
            emb = model.get_image_embeddings(inputs['pixel_values'])
            if cache_key:
                _EMBED.clear()
                _EMBED[cache_key] = emb
        out = model(image_embeddings=emb, input_points=inputs.get('input_points'), input_labels=inputs.get('input_labels'),
                    input_boxes=inputs.get('input_boxes'), multimask_output=not (box is not None or len(points or []) > 1))
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu())[0][0]  # (n,H,W)
    scores = out.iou_scores.cpu()[0, 0]
    best = int(scores.argmax()) if masks.shape[0] > 1 else 0
    return mask_to_polys(masks[best].numpy().astype(np.uint8)), float(scores[best])


def detect(image_bgr, words, box_threshold=0.33, text_threshold=0.25, limit=40):
    """一句话识别：返回 [{'label','score','box':[x0,y0,x1,y1]}]（像素）。"""
    import torch, cv2
    model, proc, dev = _dino()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    text = '. '.join(words) + '.'
    inputs = proc(images=rgb, text=text, return_tensors='pt').to(dev)
    with torch.inference_mode():
        out = model(**inputs)
    h, w = image_bgr.shape[:2]
    res = proc.post_process_grounded_object_detection(out, inputs['input_ids'], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(h, w)])[0]
    labels = res.get('text_labels', res.get('labels'))
    found = []
    for b, s, l in zip(res['boxes'].tolist(), res['scores'].tolist(), labels):
        x0, y0, x1, y1 = b
        if (x1 - x0) * (y1 - y0) > 0.6 * w * h:  # 整个画面通常是误检
            continue
        found.append({'label': str(l) or words[0], 'score': round(float(s), 3), 'box': [max(0, x0), max(0, y0), min(w, x1), min(h, y1)]})
    found.sort(key=lambda d: -d['score'])
    keep = []
    for d in found:  # 同一位置只留最高分
        if all(_iou(d['box'], k['box']) < 0.6 for k in keep):
            keep.append(d)
    return keep[:limit]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def segment_boxes(image_bgr, boxes, cache_key=None):
    """检测框 → 精细轮廓。逐个框调用（共享图像特征）。"""
    out = []
    for b in boxes:
        polys, score = segment_points(image_bgr, [], box=b, cache_key=cache_key)
        out.append((polys, score))
    return out


# ---------- 视频追踪（流式，内存有上限） ----------

def track(frames, seeds, width, height, on_frame, check=None, keep=20, timings=None):
    """frames: 依次产出 (frame_index, bgr)；seeds: {obj_id: mask(H,W uint8)}，提示放在第一帧。
    on_frame(frame_index, {obj_id: polys})。分段时用上一段最后的掩码续接。"""
    import torch, cv2, numpy as np
    model, proc, dev = _sam_video()
    dtype = _STATE.get('video_dtype', torch.float32)
    session = proc.init_video_session(inference_device=dev, video_storage_device='cpu', dtype=dtype)
    ids = list(seeds)
    step = 0
    T = timings if timings is not None else {}
    hooks = []
    if T.get('profile'):
        sync = (lambda: torch.mps.synchronize()) if dev == 'mps' else (lambda: None)
        for name in ['vision_encoder', 'memory_attention', 'memory_encoder', 'mask_decoder']:
            mod = getattr(model, name, None)
            if mod is None: continue
            def pre(m, a, kw=None, name=name): sync(); T['_t_' + name] = time.perf_counter()
            def post(m, a, o, name=name): sync(); T['sub_' + name] = T.get('sub_' + name, 0) + time.perf_counter() - T['_t_' + name]
            hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    if T.get('mem'):
        model.num_maskmem = int(T['mem'])
    clock = time.perf_counter
    t_last = clock()
    def tick(name):
        nonlocal t_last
        now = clock(); T[name] = T.get(name, 0) + now - t_last; t_last = now
    for idx, bgr in frames:
        tick('read')
        if check:
            check()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = proc(images=rgb, device=dev, return_tensors='pt')
        tick('preprocess')
        if step == 0:
            proc.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=ids,
                                                 input_masks=[torch.from_numpy((seeds[i] > 0).astype(np.float32)).to(dtype) for i in ids],
                                                 original_size=inputs.original_sizes[0])
        with torch.inference_mode():
            out = model(inference_session=session, frame_idx=step, frame=inputs.pixel_values[0].to(dtype))
            pm = out.pred_masks.float().cpu()
        tick('model')
        masks = proc.post_process_masks([pm], original_sizes=inputs.original_sizes.cpu(), binarize=True)[0]
        tick('postprocess')
        result = {}
        for k, oid in enumerate(out.object_ids if hasattr(out, 'object_ids') else ids):
            m = masks[k, 0].numpy().astype(np.uint8)
            result[oid] = mask_to_polys(m) if m.any() else []
        tick('polygons')
        on_frame(idx, result)
        tick('save')
        T['frames'] = T.get('frames', 0) + 1
        # 只保留最近几帧的记忆，长视频不会把内存吃满
        old = step - keep
        if old >= 0:
            if session.processed_frames:
                session.processed_frames.pop(old, None)
            for d in session.output_dict_per_obj.values():
                d['non_cond_frame_outputs'].pop(old, None)
            for d in session.frames_tracked_per_obj.values():
                d.pop(old, None)
        if session.processed_frames:
            session.processed_frames.pop(step - 2, None) if step >= 2 else None
        step += 1
    for h in hooks: h.remove()
    model.num_maskmem = _STATE.get('video_mem', model.config.num_maskmem)
