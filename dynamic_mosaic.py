#!/usr/bin/env python3
"""Incremental full-frame OCR plus bounded bidirectional optical-flow masks.
Independent postprocess; never changes source or a desktop editing project.
"""
import argparse, concurrent.futures as cf, json, math, subprocess, threading, time
from collections import deque
from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np
from mosaic import OCR, FrameReader, binary, probe, sha256, save, pixel_box, pixelate, template_hits, youtube_candidates, wordmark_icon_neighbors


def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);d=min(a[1]+a[3],b[1]+b[3])
    q=max(0,r-x)*max(0,d-y)
    return q/max(1e-9,a[2]*a[3]+b[2]*b[3]-q)


def flow_box(before,after,box):
    """Local LK with forward/backward consistency + RANSAC similarity transform."""
    h,w=before.shape;x,y,bw,bh=box
    mask=np.zeros_like(before)
    x0=max(0,int(x*w)-10);y0=max(0,int(y*h)-8)
    x1=min(w,math.ceil((x+bw)*w)+10);y1=min(h,math.ceil((y+bh)*h)+8)
    mask[y0:y1,x0:x1]=255
    points=cv2.goodFeaturesToTrack(before,40,.015,3,mask=mask,blockSize=3)
    if points is None or len(points)<4:return None
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(before,after,points,None,winSize=(25,25),maxLevel=3)
    if nxt is None:return None
    back,ok2,_=cv2.calcOpticalFlowPyrLK(after,before,nxt,None,winSize=(25,25),maxLevel=3)
    if back is None:return None
    valid=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(points[:,0]-back[:,0],axis=1)<1.2)
    if valid.sum()<4:return None
    a=points[valid,0];b=nxt[valid,0]
    transform,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.8)
    if transform is None or inliers.sum()<4:return None
    scale=np.linalg.norm(transform[0,:2]);angle=abs(math.atan2(transform[1,0],transform[0,0]))
    if not .80<scale<1.25 or angle>.18:return None
    corners=np.array([[x*w,y*h],[(x+bw)*w,y*h],[x*w,(y+bh)*h],[(x+bw)*w,(y+bh)*h]],np.float32)
    moved=cv2.transform(corners[None],transform)[0]
    lo=moved.min(axis=0)/[w,h];hi=moved.max(axis=0)/[w,h]
    if np.max(abs((lo+hi)/2-np.array([x+bw/2,y+bh/2])))>.20:return None
    lo=np.maximum(lo,0);hi=np.minimum(hi,1)
    if np.any(hi<=lo):return None
    return [*lo,*(hi-lo)]


def appearance_patch(gray,box):
    h,w=gray.shape;x,y,bw,bh=box
    x0=max(0,int(x*w));y0=max(0,int(y*h));x1=min(w,math.ceil((x+bw)*w));y1=min(h,math.ceil((y+bh)*h))
    if x1-x0<3 or y1-y0<3:return None
    return cv2.resize(gray[y0:y1,x0:x1],(80,24))


def same_appearance(reference,current):
    if reference is None or current is None or reference.std()<5 or current.std()<5:return False
    return float(cv2.matchTemplate(current,reference,cv2.TM_CCOEFF_NORMED)[0,0])>.65


def cut_candidate(a,b):
    # Conservative break, not a claim of perfect semantic scene segmentation.
    x=cv2.resize(a,(160,90));y=cv2.resize(b,(160,90));d=cv2.absdiff(x,y)
    return float(np.mean(d>35))>.42 and float(d.mean())>30


class PoolOCR:
    def __init__(self,workers,terms,templates,mode="text"):
        self.pool=cf.ThreadPoolExecutor(workers);self.local=threading.local();self.all=[];self.lock=threading.Lock()
        self.terms=terms;self.templates=templates;self.mode=mode
    def job(self,i,frame,source=False):
        if not hasattr(self.local,'ocr'):
            self.local.ocr=OCR()
            with self.lock:self.all.append(self.local.ocr)
        data=self.local.ocr.read(frame,self.terms,self.mode)
        hits=[dict(h,kind='text',review_required=True) for h in data['hits']]
        if any(t.lower().replace(' ','')=='youtube' for t in self.terms):
            hits+=youtube_candidates(frame);hits+=wordmark_icon_neighbors(frame,hits)
        hits+=template_hits(frame,self.templates)
        return {'frame':i,'hits':hits,'recognized_text':data['texts'] if source else []}
    def close(self):
        self.pool.shutdown()
        for o in self.all:o.close()


def frame_stream(source,pts,fps,duration,reverse=False):
    reader=FrameReader(source,pts,fps,duration)
    if not reverse:
        try:
            for i in range(reader.count):
                ok,im=reader.read()
                if not ok:raise RuntimeError(f'Failed decoding frame {i}')
                yield i,im
        finally:reader.release()
    else:
        mapping=reader.mapping;reader.release();cap=cv2.VideoCapture(str(source))
        try:
            for end in range(len(mapping),0,-90):
                start=max(0,end-90);first=int(mapping[start]);last=int(mapping[end-1]);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
                buf=[]
                for j in range(first,last+1):
                    ok,im=cap.read()
                    if not ok:raise RuntimeError(f'Failed reverse decoding {j}')
                    buf.append(im)
                for i in range(end-1,start-1,-1):yield i,buf[int(mapping[i])-first]
        finally:cap.release()


def scan(source,pts,fps,duration,out,baseline,terms,templates,workers,audit=False):
    path=out/('output_scan.jsonl' if audit else 'source_scan.jsonl')
    records={}
    if not audit:
        for line in (baseline/'scan.jsonl').read_text().splitlines():
            r=json.loads(line);records[r['frame']]=r
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);records[r['frame']]=r
    reused=len(records);pending=deque();pool=PoolOCR(workers,terms,templates);t=time.monotonic();last=t
    with path.open('a') as log:
        def collect():
            r=pending.popleft().result();records[r['frame']]=r
            log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
        try:
            for i,frame in frame_stream(source,pts,fps,duration):
                if i not in records:
                    pending.append(pool.pool.submit(pool.job,i,frame,not audit))
                    if len(pending)>=workers*2:collect()
                if time.monotonic()-last>15:
                    print(f'{"AUDIT" if audit else "SCAN"} {i+1}/{round(duration*float(fps))}, cache={reused}, elapsed={time.monotonic()-t:.1f}s',flush=True);last=time.monotonic()
            while pending:collect()
        finally:pool.close()
    return records,{'seconds':round(time.monotonic()-t,2),'cache_frames':reused,'total_frames':len(records)}


def track(source,pts,fps,duration,out,records,baseline,width,height):
    n=round(duration*float(fps));base=json.loads((baseline/'mask_timeline.json').read_text())
    boxes={i:[list(b) for b in base['boxes'].get(str(i),[])] for i in range(n)}
    for i,r in records.items():
        boxes[i].extend(pixel_box(h['box'],width,height,12) for h in r['hits'])
    breaks=set();failures=[];moved_count=0
    for reverse in [False,True]:
        tracks=[];prev=None
        for i,frame in frame_stream(source,pts,fps,duration,reverse):
            gray=cv2.cvtColor(cv2.resize(frame,(640,360)),cv2.COLOR_BGR2GRAY)
            predicted=[]
            if prev is not None:
                if cut_candidate(prev,gray):
                    breaks.add(i+1 if reverse else i);tracks=[]
                else:
                    for b,age,reference in tracks:
                        if age>=300:continue
                        new=flow_box(prev,gray,b)
                        if new is not None and same_appearance(reference,appearance_patch(gray,new)):
                            boxes[i].append(pixel_box(new,width,height,14+min(age//3,6)));predicted.append((new,age+1,reference));moved_count+=1
                        elif age<2:failures.append({'frame':i,'box':b,'direction':'backward' if reverse else 'forward'})
            anchors=[]
            for h in records[i]['hits']:
                if not any(iou(h['box'],b)>.55 for b in anchors):anchors.append(h['box'])
            tracks=[(b,0,appearance_patch(gray,b)) for b in anchors]+[(b,age,reference) for b,age,reference in predicted if not any(iou(b,a)>.25 for a in anchors)]
            tracks=tracks[:60];prev=gray
            if i%900==0:print(f'TRACK {"backward" if reverse else "forward"} {i}/{n}',flush=True)
    # Rejected flow stays visible in evidence; do not pretend extrapolation is tracking.
    save(out/'tracking_evidence.json',{'method':'LK forward-backward + RANSAC similarity; appearance-validated max300-frame propagation; full-frame OCR + output-feedback anchors',
        'candidate_boundaries':sorted(breaks),'flow_masks':moved_count,'rejected_near_anchor':failures})
    boxes={str(i):[list(b) for b in set(tuple(b) for b in rows)] for i,rows in boxes.items()}
    save(out/'mask_timeline.json',{'fps':str(fps),'frames':n,'boxes':boxes,'baseline':str(baseline),'mode':'full-frame OCR plus bidirectional optical flow'})
    return boxes


def render(source,pts,fps,duration,out,boxes,width,height):
    final=out/(source.stem+'_动态打码V2.mp4')
    command=[binary('ffmpeg'),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{width}x{height}',
        '-framerate',str(fps),'-i','pipe:0','-i',str(source),'-map','0:v:0','-map','1:a?','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
        '-c:a','copy','-map_metadata','-1','-movflags','+faststart',str(final)]
    save(out/'render_command.json',command);t=time.monotonic()
    with (out/'render.stderr.log').open('w') as err:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
        try:
            for i,frame in frame_stream(source,pts,fps,duration):
                for box in sorted(boxes[str(i)],key=lambda b:b[2]*b[3]):pixelate(frame,box,22)
                proc.stdin.write(frame.tobytes())
        finally:proc.stdin.close()
        if proc.wait():raise RuntimeError('Render failed')
    return final,round(time.monotonic()-t,2)


def validate_resume(previous,current):
    for key in ('sha256','terms','logo_templates','baseline'):
        if previous.get(key)!=current.get(key):
            raise ValueError('续跑来源或目标改变：'+key)


def main():
    p=argparse.ArgumentParser(description='逐帧检测与双向运动跟踪；独立输出，不覆盖原片')
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--baseline');group.add_argument('--input')
    p.add_argument('--text',action='append',default=[]);p.add_argument('--logo-template',action='append',default=[])
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=8:p.error('workers must be 1–8')
    if a.input and not (a.text or a.logo_template):p.error('新视频必须指定目标文字或真实Logo模板')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=a.resume)
    if a.input:
        baseline=out/'_input';baseline.mkdir(exist_ok=a.resume);src=Path(a.input).expanduser().resolve()
        if (baseline/'source_manifest.json').exists():
            existing=json.loads((baseline/'source_manifest.json').read_text())
            if existing['path']!=str(src) or existing['terms']!=a.text:raise ValueError('续跑输入或文字目标不同')
            if [r['path'] for r in existing['logo_templates']]!=[str(Path(t).resolve()) for t in a.logo_template]:raise ValueError('续跑模板不同')
        else:
            save(baseline/'source_manifest.json',{'path':str(src),'sha256':sha256(src),'terms':a.text,
                'logo_templates':[{'path':str(Path(t).resolve()),'sha256':sha256(Path(t))} for t in a.logo_template]})
            (baseline/'scan.jsonl').write_text('');save(baseline/'mask_timeline.json',{'boxes':{}})
    else:baseline=Path(a.baseline).resolve()
    manifest=json.loads((baseline/'source_manifest.json').read_text());source=Path(manifest['path']);assert sha256(source)==manifest['sha256'],'Source changed'
    meta=probe(source);v=next(s for s in meta['streams'] if s['codec_type']=='video');fps=Fraction(v['r_frame_rate']);duration=float(v['duration']);w,h=v['width'],v['height']
    raw=json.loads(subprocess.check_output([binary('ffprobe'),'-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(source)]))
    pts=[float(f['best_effort_timestamp_time']) for f in raw['frames']];n=round(duration*float(fps))
    templates=[(Path(r['path']),cv2.imread(r['path'])) for r in manifest['logo_templates']]
    if any(im is None for _,im in templates):raise ValueError('Logo模板无法读取')
    if abs(pts[0])>.002 or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('源时间戳异常')
    if v.get('color_transfer') in ('smpte2084','arib-std-b67'):raise ValueError('暂不处理HDR')
    if w%2 or h%2:raise ValueError('编码要求偶数尺寸')
    for s in meta['streams']:
        if s['codec_type']=='audio' and abs(float(s.get('start_time','0')))>.04:raise ValueError('音频偏移需人工处理')
    manifest.update({'baseline':str(baseline),'algorithm':'dynamic_v2','source_scan':{'source_frames':len(pts),'output_frames':n,'mode':'all frames; baseline records reused'},'code_sha256':sha256(Path(__file__))})
    if (out/'source_manifest.json').exists():validate_resume(json.loads((out/'source_manifest.json').read_text()),manifest)
    save(out/'source_manifest.json',manifest)
    records,scantime=scan(source,pts,fps,duration,out,baseline,manifest['terms'],templates,a.workers)
    assert len(records)==n
    # Consolidated immutable input for subsequent incremental refinement.
    with (out/'scan.jsonl').open('w') as f:
        for i in range(n):f.write(json.dumps(records[i],ensure_ascii=False)+'\n')
    if (out/'mask_timeline.json').exists():boxes=json.loads((out/'mask_timeline.json').read_text())['boxes']
    else:boxes=track(source,pts,fps,duration,out,records,baseline,w,h)
    final=out/(source.stem+'_动态打码V2.mp4');render_seconds=None
    if not final.exists():final,render_seconds=render(source,pts,fps,duration,out,boxes,w,h)
    check=subprocess.run([binary('ffmpeg'),'-v','error','-i',str(final),'-f','null','-'],capture_output=True,text=True)
    assert check.returncode==0 and not check.stderr.strip(),check.stderr
    audio=[]
    for i,s in enumerate([s for s in meta['streams'] if s['codec_type']=='audio']):
        hashes=[subprocess.check_output([binary('ffmpeg'),'-v','error','-i',str(f),'-map',f'0:a:{i}','-c:a','pcm_s16le','-f','hash','-hash','sha256','-']).decode().strip() for f in [source,final]]
        audio.append({'track':i,'status':'PASS' if hashes[0]==hashes[1] else 'FAIL','hashes':hashes})
    audit,audittime=scan(final,[i/float(fps) for i in range(n)],fps,duration,out,baseline,manifest['terms'],templates,a.workers,True)
    residual=[{'frame':i,'time':i/float(fps),'hits':r['hits']} for i,r in sorted(audit.items()) if r['hits']]
    save(out/'remaining_targets.json',{'status':'FAIL' if residual else 'PASS','scope':'all output frames OCR/icon candidates; not proof of zero visual leakage','frames_checked':len(audit),'rows':residual})
    save(out/'qc_report.json',{'status':'REVIEW_REQUIRED','source_unchanged':'PASS' if sha256(source)==manifest['sha256'] else 'FAIL','full_decode':'PASS','audio_tracks':audio,
        'output':str(final),'metadata':probe(final),'source_scan':scantime,'output_scan':audittime,'render_seconds':render_seconds,'residual_frames':len(residual),
        'human_continuous_review':'UNVERIFIED','recognition_completeness':'UNVERIFIED'})
    print('DONE',final,'residual_frames',len(residual),flush=True)

if __name__=='__main__':main()
