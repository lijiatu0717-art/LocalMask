#!/usr/bin/env python3
"""Local, non-destructive video redaction. Detection is not a privacy guarantee."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from fractions import Fraction

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def binary(name):
    found = shutil.which(name)
    if found:
        return found
    try:  # 安装脚本自带的 FFmpeg（static-ffmpeg）
        from static_ffmpeg import run
        ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
        return ffprobe if name == 'ffprobe' else ffmpeg
    except Exception:
        pass
    raise RuntimeError(f'找不到 {name}；请安装 FFmpeg 或将其加入 PATH')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def probe(path):
    return json.loads(subprocess.check_output([binary('ffprobe'), '-v', 'error', '-show_streams',
                                               '-show_format', '-of', 'json', str(path)]))


def pixel_box(box, width, height, padding=8):
    x, y, w, h = box
    x0, y0 = max(0, math.floor(x * width) - padding), max(0, math.floor(y * height) - padding)
    x1, y1 = min(width, math.ceil((x + w) * width) + padding), min(height, math.ceil((y + h) * height) + padding)
    return [x0, y0, x1 - x0, y1 - y0]


def pixelate(image, box, block=18):
    x, y, w, h = map(int, box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, roi.shape[1] // block), max(1, roi.shape[0] // block)), interpolation=cv2.INTER_AREA)
    image[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)


def youtube_candidates(image):
    """Only red rounded play-button candidates, not a general commercial logo classifier."""
    height, width = image.shape[:2]
    scale = min(1., 1280 / width)
    small = cv2.resize(image, (round(width * scale), round(height * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 100), (12, 255, 255)) | cv2.inRange(hsv, (170, 120, 100), (179, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 4 or not .75 < w / h < 2.4:
            continue
        if not .55 < cv2.contourArea(contour) / (w * h) < 1.01:
            continue
        roi = small[y:y+h, x:x+w]
        white = cv2.inRange(roi, (160, 160, 160), (255, 255, 255))
        triangles, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for tri in triangles:
            area = cv2.contourArea(tri)
            if not .035 < area / (w*h) < .30:
                continue
            approx = cv2.approxPolyDP(tri, .10 * cv2.arcLength(tri, True), True)
            if len(approx) != 3:
                continue
            pts = approx[:, 0, :]
            center = pts.mean(axis=0)
            if not (.3*w < center[0] < .7*w and .25*h < center[1] < .75*h):
                continue
            # A right-facing triangle: two left vertices, one right vertex.
            ordered = pts[np.argsort(pts[:, 0])]
            if abs(int(ordered[0, 0])-int(ordered[1, 0])) > w * .15:
                continue
            if ordered[2, 0] - ordered[1, 0] < w * .12:
                continue
            results.append({'kind': 'youtube_icon_candidate', 'box': [x / scale / width, y / scale / height, w / scale / width, h / scale / height],
                            'confidence': None, 'review_required': True,
                            'reason': '红色播放按钮形状；可能误报，不代表已确认品牌身份'})
            break
    return results


def wordmark_icon_neighbors(image, hits):
    """Bind tiny red icons to an actually recognized YouTube wordmark, not arbitrary red UI."""
    height,width=image.shape[:2]
    extra=[]
    for hit in hits:
        if hit['kind']!='text' or hit.get('term','').lower().replace(' ','')!='youtube':
            continue
        x,y,w,h=hit['box'];px,py,ph=round(x*width),round(y*height),max(4,round(h*height))
        x0=max(0,px-round(ph*2.5));y0=max(0,py-round(ph*.4));y1=min(height,py+round(ph*1.4))
        if px<=x0 or y1<=y0:continue
        roi=image[y0:y1,x0:px]
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        red=cv2.inRange(hsv,(0,90,90),(12,255,255))|cv2.inRange(hsv,(165,90,90),(179,255,255))
        cs,_=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            rx,ry,rw,rh=cv2.boundingRect(c)
            if rw>=4 and rh>=3 and .8<rw/rh<2.8 and cv2.contourArea(c)>=rw*rh*.45:
                extra.append({'kind':'youtube_icon_wordmark_neighbor','box':[(x0+rx)/width,(y0+ry)/height,rw/width,rh/height],
                    'confidence':None,'review_required':True,'reason':'已识别YouTube文字左侧的红色图标，需要视觉复核'})
    return extra


def template_hits(image, templates):
    hits=[]
    height,width=image.shape[:2]
    def red_mask(im):
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv,(0,80,70),(12,255,255))|cv2.inRange(hsv,(165,80,70),(179,255,255))
    image_red=None
    for path,template in templates:
        # A distinctive red template can use color components to limit search to
        # actual candidate locations; generic templates retain multiscale search.
        tm=red_mask(template)
        if cv2.countNonZero(tm)>template.shape[0]*template.shape[1]*.08:
            tx,ty,tw,th=cv2.boundingRect(tm)
            if image_red is None:image_red=red_mask(image)
            contours,_=cv2.findContours(image_red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,cw,ch=cv2.boundingRect(contour)
                if min(cw,ch)<4 or max(cw,ch)>150 or not .65<(cw/ch)/(tw/th)<1.5:continue
                scale=math.sqrt(cw*ch/(tw*th))
                for factor in [.85,1.,1.15]:
                    s=scale*factor;w,h=round(template.shape[1]*s),round(template.shape[0]*s)
                    if min(w,h)<7:continue
                    x0=max(0,round(x-tx*s)-4);y0=max(0,round(y-ty*s)-4)
                    roi=image[y0:min(height,y0+h+8),x0:min(width,x0+w+8)]
                    if roi.shape[0]<h or roi.shape[1]<w:continue
                    resized=cv2.resize(template,(w,h),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
                    score=cv2.matchTemplate(roi,resized,cv2.TM_CCOEFF_NORMED)
                    _,confidence,_,location=cv2.minMaxLoc(score)
                    if confidence>=.92:
                        rx,ry=location
                        hits.append({'kind':'reference_logo_candidate','box':[(x0+rx)/width,(y0+ry)/height,w/width,h/height],
                            'confidence':confidence,'template':str(path),'review_required':True})
            continue
        for scale in [.4,.5,.6,.75,.9,1.,1.15,1.3,1.5,1.8,2.]:
            w,h=round(template.shape[1]*scale),round(template.shape[0]*scale)
            if min(w,h)<7 or w>=width or h>=height:continue
            resized=cv2.resize(template,(w,h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            score=cv2.matchTemplate(image,resized,cv2.TM_CCOEFF_NORMED)
            for _ in range(20):
                _,confidence,_,location=cv2.minMaxLoc(score)
                if confidence<.92:break
                x,y=location
                hits.append({'kind':'reference_logo_candidate','box':[x/width,y/height,w/width,h/height],
                    'confidence':confidence,'template':str(path),'review_required':True})
                score[max(0,y-h):min(score.shape[0],y+h),max(0,x-w):min(score.shape[1],x+w)]=-1
    return hits


class OCR:
    def __init__(self):
        source = ROOT / 'vision_ocr.swift'
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        executable = cache / ('vision-' + sha256(source)[:16])
        if not executable.exists():
            subprocess.run(['swiftc', str(source), '-O', '-o', str(executable)], check=True)
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def read(self, image, terms, mode="text"):
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('OCR图像编码失败')
        self.process.stdin.write(json.dumps({'image': base64.b64encode(encoded).decode(), 'terms': terms, 'mode': mode}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('OCR进程意外终止')
        data = json.loads(line)
        if 'error' in data:
            raise RuntimeError(data['error'])
        return data

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=15)
        self.process.stdout.close()


def read_manual(path):
    if not path:
        return []
    rows = json.loads(Path(path).read_text())
    for row in rows:
        if not (0 <= row['start'] < row['end'] and len(row['box']) == 4):
            raise ValueError('手动框需要 start < end，box为归一化[x,y,w,h]')
        x, y, w, h = row['box']
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x+w <= 1.00001 and y+h <= 1.00001):
            raise ValueError('手动框超出画面')
    return rows


class FrameReader:
    """Sample source presentation timestamps onto an explicit CFR grid, never speed audio."""
    def __init__(self, source, pts, fps, duration):
        self.cap = cv2.VideoCapture(str(source))
        self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        self.count = round(duration * float(fps))
        target = np.arange(self.count) / float(fps)
        right = np.searchsorted(pts, target).clip(0, len(pts)-1)
        left = (right-1).clip(0, len(pts)-1)
        self.mapping = np.where(abs(np.array(pts)[left]-target) <= abs(np.array(pts)[right]-target), left, right)
        self.index = 0
        self.source_index = -1
        self.frame = None

    def read(self):
        if self.index >= self.count:
            return False, None
        wanted = self.mapping[self.index]
        while self.source_index < wanted:
            ok, self.frame = self.cap.read()
            if not ok:
                return False, None
            self.source_index += 1
        self.index += 1
        return True, self.frame.copy()

    def release(self):
        self.cap.release()


def execute(args):
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'视频不存在：{source}')
    output = Path(args.output).expanduser().resolve()
    # A run must be independent and new, including cached scan evidence.
    output.mkdir(parents=True, exist_ok=args.resume)
    cached_records = {}
    if args.scan_cache:
        cache=Path(args.scan_cache)
        previous=json.loads((cache.parent/'source_manifest.json').read_text())
        if previous['sha256']!=sha256(source) or previous['terms']!=args.text:
            raise ValueError('扫描缓存不属于同一源文件和文字目标')
        for line in cache.read_text().splitlines():
            record=json.loads(line);cached_records[record['frame']]=record
    if args.resume:
        previous = json.loads((output/'source_manifest.json').read_text())
        if previous['sha256'] != sha256(source) or previous['terms'] != args.text or previous['youtube_icon'] != args.youtube_icon:
            raise ValueError('续跑参数与原始源文件不一致')
        if list(output.glob('*.mp4')):
            raise ValueError('已有视频输出，不覆盖；请使用新目录')
        if (output/'scan.jsonl').exists():
            for line in (output/'scan.jsonl').read_text().splitlines():
                record = json.loads(line)
                cached_records[record['frame']] = record
    started = time.monotonic()
    templates=[]
    for item in args.logo_template:
        path=Path(item).expanduser().resolve();im=cv2.imread(str(path))
        if im is None or float(im.std())<5:raise ValueError('Logo参考图不可读或过于单一')
        templates.append((path,im))
    meta = probe(source)
    stream = next(s for s in meta['streams'] if s['codec_type'] == 'video')
    fps = Fraction(stream['r_frame_rate'])
    if fps <= 0:
        raise ValueError('无法识别帧率')
    # Check actual timestamps; silently flattening VFR can desynchronize audio.
    times = json.loads(subprocess.check_output([binary('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(source)]))['frames']
    pts = [float(f['best_effort_timestamp_time']) for f in times if 'best_effort_timestamp_time' in f]
    if len(pts) < 2 or any(b <= a for a,b in zip(pts,pts[1:])):
        raise ValueError('源视频时间戳无效')
    duration = float(stream.get('duration',meta['format']['duration']))
    if abs(pts[0]) > .002:
        raise ValueError('视频起始时间非零；第一版停止以避免声画偏移')
    for s in meta['streams']:
        if s['codec_type'] == 'audio' and abs(float(s.get('start_time', '0'))) > .04:
            raise ValueError('音频起始时间有偏移，第一版不自动改写')
    if stream.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('第一版不处理HDR，避免意外改变颜色')
    cap = FrameReader(source,pts,fps,duration)
    output_frame_count = cap.count
    success, first = cap.read()
    if not success:
        raise ValueError('无法解码视频')
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError('H.264首版要求偶数宽高')
    save(output/'source_manifest.json', {'path': str(source), 'sha256': sha256(source), 'metadata': meta,
         'output_size': [width, height], 'terms': args.text, 'youtube_icon': args.youtube_icon,
         'original_untouched': True, 'mode': 'independent_postprocess',
         'logo_templates':[{'path':str(p),'sha256':sha256(p)} for p,_ in templates],
         'manual_regions':({'path':str(Path(args.regions).resolve()),'sha256':sha256(Path(args.regions)),
                            'regions':read_manual(args.regions)} if args.regions else None),
         'timing': {'source_frames':len(pts),'output_frames':output_frame_count,'fps':str(fps),
                    'method':'nearest source timestamp on CFR grid; no audio time scaling'}})
    step = max(1, round(float(fps)/args.scan_fps)) if args.scan_fps else 1
    ocr = OCR() if args.text else None
    records, frame_index = [], 0
    frame = first
    try:
        with (output/'scan.jsonl').open('a' if args.resume else 'w') as log:
            while success:
                if frame_index % step == 0:
                    if frame_index in cached_records:
                        record=cached_records[frame_index]
                        if args.youtube_icon:
                            record['hits']=[h for h in record['hits'] if h['kind']=='text']
                            record['hits'].extend(youtube_candidates(frame))
                            record['hits'].extend(wordmark_icon_neighbors(frame,record['hits']))
                        record['hits'].extend(template_hits(frame,templates))
                        records.append(record)
                        if not args.resume:
                            log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                        success, frame = cap.read()
                        frame_index += 1
                        continue
                    hits = []
                    if ocr:
                        data = ocr.read(frame, args.text)
                        for hit in data['hits']:
                            hits.append(dict(hit, kind='text', review_required=hit['confidence'] < .8))
                    else:
                        data = {'texts': []}
                    if args.youtube_icon:
                        hits.extend(youtube_candidates(frame))
                        hits.extend(wordmark_icon_neighbors(frame,hits))
                    hits.extend(template_hits(frame,templates))
                    row = {'frame': frame_index, 'time': frame_index/float(fps), 'hits': hits, 'recognized_text': data['texts']}
                    log.write(json.dumps(row, ensure_ascii=False)+'\n');log.flush()
                    records.append(row)
                    if frame_index % max(step, round(float(fps)*5)) < step:
                        print(f'识别 {frame_index/float(fps):.1f}s，命中区域 {sum(len(r["hits"]) for r in records)}', flush=True)
                success, frame = cap.read()
                frame_index += 1
    finally:
        cap.release()
        if ocr:
            ocr.close()
    if frame_index != output_frame_count:
        raise RuntimeError(f'解码帧数异常：{frame_index} / {output_frame_count}')
    manual = read_manual(args.regions)
    # Conservative time padding covers one missed sampled interval, with explicit review.
    radius = max(step, round(float(fps)*.10))
    boxes_by_frame = [[] for _ in range(frame_index)]
    for record in records:
        for hit in record['hits']:
            box = pixel_box(hit['box'], width, height, args.padding)
            for n in range(max(0,record['frame']-radius),min(frame_index,record['frame']+radius+1)):
                if box not in boxes_by_frame[n]:
                    boxes_by_frame[n].append(box)
    for row in manual:
        for n in range(max(0,math.floor(row['start']*float(fps))),min(frame_index,math.ceil(row['end']*float(fps)))):
            boxes_by_frame[n].append(pixel_box(row['box'],width,height,args.padding))
    save(output/'mask_timeline.json', {'fps':str(fps),'frames':frame_index,'padding_px':args.padding,
          'temporal_radius_frames':radius,'boxes': {str(i):b for i,b in enumerate(boxes_by_frame) if b}})
    final = output / (source.stem + '_马赛克测试版.mp4')
    command = [binary('ffmpeg'), '-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','bgr24',
      '-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0','-i',str(source),
      '-map','0:v:0','-map','1:a?','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709',
      '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
      '-c:a','copy','-map_metadata','-1','-movflags','+faststart',str(final)]
    save(output/'render_command.json',command)
    cap = FrameReader(source,pts,fps,duration)
    comparisons = output/'comparisons';comparisons.mkdir()
    selected = {r['frame'] for r in records if r['hits']}
    # Bound proof images to 12, evenly distributed over the detected interval.
    selected = sorted(selected)
    selected = set(selected[i] for i in np.linspace(0,len(selected)-1,min(12,len(selected)),dtype=int)) if selected else set()
    with (output/'render.stderr.log').open('w') as err:
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
        try:
            for i, boxes in enumerate(boxes_by_frame):
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError('渲染阶段源视频提前结束')
                before = img.copy() if i in selected else None
                # Small boxes first; a larger overlapping mask must not weaken them.
                for box in sorted(boxes,key=lambda b:b[2]*b[3]):
                    pixelate(img,box,args.block)
                if before is not None:
                    cv2.imwrite(str(comparisons/f'{i:06d}_before.jpg'), before)
                    cv2.imwrite(str(comparisons/f'{i:06d}_after.jpg'), img)
                process.stdin.write(img.tobytes())
        finally:
            cap.release();process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('编码失败，见render.stderr.log')
    decoded = subprocess.run([binary('ffmpeg'),'-v','error','-i',str(final),'-f','null','-'],capture_output=True,text=True)
    final_meta = probe(final)
    video = next(s for s in final_meta['streams'] if s['codec_type']=='video')
    original_audio = [s for s in meta['streams'] if s['codec_type']=='audio']
    # Hash decoded samples of EVERY audio track, not just codec metadata.
    audio_checks=[]
    for i,_ in enumerate(original_audio):
        hashes=[]
        for path in [source,final]:
            hashes.append(subprocess.check_output([binary('ffmpeg'),'-v','error','-i',str(path),'-map',f'0:a:{i}',
                       '-c:a','pcm_s16le','-f','hash','-hash','sha256','-']).decode().strip())
        audio_checks.append({'track':i,'status':'PASS' if hashes[0]==hashes[1] else 'FAIL','hashes':hashes})
    hits=sum(len(r['hits']) for r in records)
    qc={'status':'REVIEW_REQUIRED', 'output':str(final),'source_unchanged':'PASS' if sha256(source)==json.loads((output/'source_manifest.json').read_text())['sha256'] else 'FAIL',
        'full_decode':'PASS' if decoded.returncode==0 and not decoded.stderr.strip() else 'FAIL',
        'frame_count':'PASS' if int(video.get('nb_frames',0))==frame_index else 'FAIL',
        'fps':'PASS' if Fraction(video['avg_frame_rate'])==fps else 'FAIL',
        'dimensions':'PASS' if [video['width'],video['height']]==[width,height] else 'FAIL',
        'audio_tracks':audio_checks,'source_frames':len(pts),'output_frames':frame_index,'scan_frames':len(records),'hit_regions':hits,
        'frames_masked':sum(bool(b) for b in boxes_by_frame),'elapsed_seconds':round(time.monotonic()-started,2),
        'recognition_completeness':'UNVERIFIED','human_review':'UNVERIFIED','scan_step_frames':step,
        'limitations':['OCR可能漏检小字、旋转字、动画字；逐帧扫描也不保证全召回',
        'YouTube红色播放图标检测为几何候选，可能将其他播放按钮误判；不支持保证所有商业Logo',
        '时间扩展约0.1秒或一个采样间隔，切镜附近可能短暂多遮挡；动态目标需复核',
        '单独软字幕轨道不写入输出，已烧入画面的字幕由OCR处理',
        '马赛克是视觉遮挡，不保证不可恢复；高度敏感信息建议实心遮盖']}
    save(output/'qc_report.json',qc)
    (output/'review_needed.md').write_text('# 第一轮打码检查\n\n状态：REVIEW_REQUIRED，不是保证无漏检的最终隐私版本。\n\n'
       +f'扫描 {len(records)} 帧；命中 {hits} 个帧内区域；打码 {qc["frames_masked"]} 帧。\n\n'
       +'\n'.join('- '+x for x in qc['limitations'])+'\n\n请检查 comparisons 中的前后图，尤其是图标候选、镜头切换及字幕。未命中不能解释为视频不存在目标。\n')
    technical=[qc[k] for k in ['source_unchanged','full_decode','frame_count','fps','dimensions']]+[x['status'] for x in audio_checks]
    print(json.dumps(qc,ensure_ascii=False,indent=2))
    if 'FAIL' in technical:
        raise RuntimeError('实际输出技术检查失败，不能交付为通过')


def main():
    p=argparse.ArgumentParser(description='本地视频打码：OCR、图标候选及手动补框')
    p.add_argument('input');p.add_argument('--output',required=True,help='必须是尚不存在的新目录')
    p.add_argument('--text',action='append',default=[],help='要遮挡的文字；可重复，大小写不敏感')
    p.add_argument('--youtube-icon',action='store_true',help='遮挡红色播放图标候选，需人工复核')
    p.add_argument('--logo-template',action='append',default=[],help='真实Logo裁图模板，多尺度匹配；可重复指定，候选需复核')
    p.add_argument('--regions',help='手动区域JSON：[{start,end,box:[x,y,w,h]}]，坐标0–1')
    p.add_argument('--resume',action='store_true',help='仅续跑同源、同目标且尚未生成视频的中断任务')
    p.add_argument('--scan-cache',help='复用同源同目标的scan.jsonl，不重做已完成的OCR')
    p.add_argument('--scan-fps',type=float,default=0,help='0=逐帧OCR；抽样可加速但提高漏检风险')
    p.add_argument('--padding',type=int,default=8);p.add_argument('--block',type=int,default=18)
    args=p.parse_args()
    if not (args.text or args.youtube_icon or args.regions or args.logo_template):p.error('至少指定文字、Logo或手动区域')
    if args.scan_fps<0 or args.padding<0 or args.block<2:p.error('非法扫描率、边距或马赛克尺寸')
    execute(args)


if __name__=='__main__':
    main()
