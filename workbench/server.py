#!/usr/bin/env python3
"""Loopback-only video masking workbench. No cloud, shell eval, or editor integration."""
import sys,os,json,time,uuid,threading,subprocess,math,re,secrets,shutil,mimetypes,hashlib
from pathlib import Path
from fractions import Fraction
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs,unquote
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from mosaic import probe,binary,sha256,FrameReader,pixelate,OCR
from dynamic_mosaic import flow_box,same_appearance,appearance_patch,cut_candidate
from core import write_json,validate_tracks,box_at,masks_at,contiguous_ranges,pixel_rect,apply_cover,render_masks,polys_at,EFFECTS,HOLD
import ai
APP=Path(__file__).resolve().parent;DATA=APP/'data';DATA.mkdir(exist_ok=True)
LOCK=threading.RLock();JOBS={};CANCEL={};TOKEN=secrets.token_urlsafe(32)

def folder(pid):
 if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',pid):raise ValueError('工程ID不合法')
 p=DATA/pid
 if not (p/'project.json').is_file():raise FileNotFoundError('工程不存在')
 return p

def load(pid):return json.loads((folder(pid)/'project.json').read_text())
def persist(p):write_json(folder(p['id'])/'project.json',p)
def pts_for(p):return json.loads((folder(p['id'])/'pts.json').read_text())
def layers_for(p):return {l['id']:json.loads((folder(p['id'])/l['file']).read_text())['boxes'] for l in p.get('layers',[])}
MASKS={};STICKERS={}
DEFAULT_STYLE={'effect':'mosaic','block':20,'strength':50,'opacity':100,'padding':4,'color':[28,28,30],'feather':3}
PALETTE=['#8b7cff','#f5b83d','#2fd3c8','#ff6f91','#5aa9ff','#9be15d','#ff9b4a','#c77dff']
def mask_file(pid,tid):
 if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',str(tid)):raise ValueError('物体ID不合法')
 return folder(pid)/'masks'/f'{tid}.json'
def load_masks(pid,tid):
 f=mask_file(pid,tid)
 if not f.is_file():return {}
 m=f.stat().st_mtime_ns;c=MASKS.get((pid,tid))
 if c and c[0]==m:return c[1]
 d=json.loads(f.read_text()).get('frames',{});MASKS[(pid,tid)]=(m,d);return d
def save_masks(pid,tid,frames):
 f=mask_file(pid,tid);f.parent.mkdir(exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps({'frames':frames},separators=(',',':')));tmp.replace(f);MASKS.pop((pid,tid),None)
def sticker_file(pid,name):
 if not re.fullmatch(r'[a-f0-9]{16}\.png',str(name or '')):raise ValueError('贴纸不合法')
 return folder(pid)/'stickers'/name
def sticker_loader(pid):
 def get(name):
  try:f=sticker_file(pid,name)
  except ValueError:return None
  if not f.is_file():return None
  if (pid,name) not in STICKERS:
   im=cv2.imread(str(f),cv2.IMREAD_UNCHANGED)
   if im is not None and im.ndim==2:im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGRA)
   elif im is not None and im.shape[2]==3:im=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
   STICKERS[(pid,name)]=im
  return STICKERS[(pid,name)]
 return get
def mask_loader(pid):return lambda tid:load_masks(pid,tid)
_CORE_MTIME=[Path(__file__).with_name('core.py').stat().st_mtime]
def core_fresh():
 """core.py（效果算法）更新后自动重新载入，不用重启工作台。"""
 global render_masks,masks_at
 m=Path(__file__).with_name('core.py').stat().st_mtime
 if m!=_CORE_MTIME[0]:
  import importlib,core as _c
  importlib.reload(_c);render_masks=_c.render_masks;masks_at=_c.masks_at;_CORE_MTIME[0]=m
def render_frame(p,im,i,layers=None):
 core_fresh()
 return render_masks(im,masks_at(p,i,layers if layers is not None else layers_for(p),mask_loader(p['id'])),sticker_loader(p['id']))
def snapshot(p):
 hist=folder(p['id'])/'history';hist.mkdir(exist_ok=True);write_json(hist/f'rev_{p["revision"]}.json',p)
def style_for(p):
 st=dict(DEFAULT_STYLE);st.update({k:v for k,v in (p.get('default_style') or {}).items() if k in DEFAULT_STYLE})
 if st['effect'] not in EFFECTS or st['effect']=='sticker' and not st.get('sticker'):st['effect']='mosaic'
 return st

def iter_frames(p,start,end,maxside=1280):
 """按输出帧号产出 (i, 图像)。end<start 时倒序（分块读取，内存有上限）。图像长边不超过 maxside。"""
 pts=pts_for(p);r=FrameReader(p['source'],pts,Fraction(p['fps_ratio']),p['duration']);mapping=r.mapping;r.release()
 scale=min(1.0,maxside/max(p['width'],p['height']));size=(round(p['width']*scale),round(p['height']*scale))
 cap=cv2.VideoCapture(p['source'])
 def run(lo,hi):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(mapping[lo]));src=int(mapping[lo])-1;frame=None
  for i in range(lo,hi+1):
   want=int(mapping[i])
   while src<want:
    ok,frame=cap.read()
    if not ok:raise ValueError(f'无法读取第 {i} 帧')
    src+=1
   yield i,(frame if scale==1 else cv2.resize(frame,size,interpolation=cv2.INTER_AREA))
 try:
  if end>=start:yield from run(start,end)
  else:
   hi=start
   while hi>=end:
    lo=max(end,hi-89);chunk=list(run(lo,hi))
    for item in reversed(chunk):yield item
    hi=lo-1
 finally:cap.release()

def frame_read(p,i):
 if not 0<=i<p['frames']:raise ValueError('帧超出范围')
 pts=pts_for(p);target=i/p['fps'];idx=min(range(len(pts)),key=lambda j:abs(pts[j]-target))
 cap=cv2.VideoCapture(p['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read();cap.release()
 if not ok:raise ValueError('无法解码此帧')
 return im

_ENC=[]
def fast_encoder():
 """优先用苹果芯片硬件编码（快很多倍），没有就用 CPU 快速档。"""
 if not _ENC:
  try:out=subprocess.run([binary('ffmpeg'),'-hide_banner','-encoders'],capture_output=True,text=True,timeout=20).stdout
  except Exception:out=''
  _ENC.append(['-c:v','h264_videotoolbox','-b:v','14M','-maxrate','20M','-bufsize','28M','-profile:v','high','-allow_sw','1'] if 'h264_videotoolbox' in out else ['-c:v','libx264','-preset','veryfast','-crf','18'])
 return _ENC[0]

def bake(source,v,hdr,on_progress=None,duration=None):
 """手机视频（竖拍旋转标记 / HDR / 可变帧率）自动转成 1080p 标准副本；原文件保留。"""
 fps=v.get('avg_frame_rate') or v.get('r_frame_rate') or '30/1'
 try:
  f=float(Fraction(fps))
  if f<=0 or f>240:fps='30/1'
  elif f>50:fps='60/1' if f>55 else fps
 except Exception:fps='30/1'
 out=source.with_name(source.stem+'_标准副本.mp4')
 cap="scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))'"
 chains=[]
 if hdr:chains.append(cap+',zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
 chains.append(cap+',format=yuv420p')
 last=''
 for vf in chains:
  for enc in ([fast_encoder()]+([['-c:v','libx264','-preset','veryfast','-crf','18']] if 'videotoolbox' in fast_encoder()[1] else [])):
   for audio in (['-c:a','copy'],['-c:a','aac','-b:a','256k']):
    cmd=[binary('ffmpeg'),'-hide_banner','-loglevel','error','-y','-i',str(source),'-map','0:v:0','-map','0:a?','-vf',vf,'-r',str(fps),'-vsync','cfr']+enc+['-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709']+audio+['-movflags','+faststart','-progress','pipe:1','-nostats',str(out)]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    for line in proc.stdout:
     if on_progress and duration and line.startswith('out_time_us='):
      try:on_progress(min(99,int(line.split('=')[1])/1e6/duration*100))
      except ValueError:pass
    err=proc.stderr.read();proc.wait()
    if proc.returncode==0 and out.is_file() and out.stat().st_size>0:return out
    last=err[-400:]
 raise ValueError('这个视频转换失败：'+last)

def video_pts(source):
 """只读时间戳、不解码画面（4K 视频也很快）。"""
 raw=subprocess.check_output([binary('ffprobe'),'-v','error','-select_streams','v:0','-show_entries','packet=pts_time','-of','csv=p=0',str(source)],text=True)
 return sorted({float(x.split(',')[0]) for x in raw.split() if x and x.split(',')[0] not in ('N/A','')})

def create_project(source,name=None,pid=None,_baked=False,on_progress=None):
 source=Path(source).resolve();meta=probe(source);v=next(s for s in meta['streams'] if s['codec_type']=='video')
 hdr=v.get('color_transfer') in ['smpte2084','arib-std-b67']
 rotated=any(s.get('rotation',0) for s in v.get('side_data_list',[])) or str(v.get('tags',{}).get('rotate','0')) not in ['0','']
 pts=video_pts(source)
 bad_ts=not pts or abs(pts[0])>.002 or any(b<=a for a,b in zip(pts,pts[1:]))
 if pts and len(pts)>2 and not bad_ts:
  step=(pts[-1]-pts[0])/(len(pts)-1);bad_ts=step<=0 or max(abs(b-a-step) for a,b in zip(pts,pts[1:]))>step*0.5  # 可变帧率：转成固定帧率，预览和码才对得齐
 big=max(int(v.get('width',0)),int(v.get('height',0)))>1920  # 4K 等大视频转 1080p，播放、拖动、识别都快很多
 if hdr or rotated or bad_ts or big:
  if _baked:raise ValueError('视频转换后仍不符合要求，请把这个视频发给 Claude 看看')
  dur=float(v.get('duration') or meta['format'].get('duration') or 0) or None
  return create_project(bake(source,v,hdr,on_progress,dur),name,pid,_baked=True)
 fps=Fraction(v.get('avg_frame_rate') or v['r_frame_rate']);duration=float(v.get('duration') or meta['format']['duration']);pid=pid or uuid.uuid4().hex[:12]
 d=DATA/pid;d.mkdir(exist_ok=False)
 p={'id':pid,'name':name or source.name,'source':str(source),'source_hash':sha256(source),'width':v['width'],'height':v['height'],'fps':float(fps),'fps_ratio':str(fps),'duration':duration,'frames':round(duration*float(fps)), 'terms':'','target_mode':'faces','mask_style':'cover','tracks':[],'layers':[],'issues':[],'revision':1,'exports':[],'created':time.time(),'status':'NEEDS_REVIEW'}
 write_json(d/'project.json',p);write_json(d/'pts.json',pts);write_json(d/'source_manifest.json',{'path':str(source),'sha256':p['source_hash'],'metadata':meta})
 return p

class Cancelled(Exception):pass
def check_cancel(pid):
 if CANCEL.get(pid) and CANCEL[pid].is_set():raise Cancelled('已停止准备；识别缓存已保留，下次可继续')

def progress(pid,percent,message):
 with LOCK:JOBS[pid].update(progress=round(percent,1),message=message,updated=time.time())

def launch_job(pid,kind,fn):
 with LOCK:
  if JOBS.get(pid,{}).get('state')=='running':raise ValueError('当前任务仍在运行')
  CANCEL[pid]=threading.Event()
  JOBS[pid]={'kind':kind,'state':'running','progress':0,'message':'准备中','started':time.time()}
 def worker():
  try:
   fn();JOBS[pid].update(state='done',progress=100,message='完成，请检查结果',finished=time.time())
  except Cancelled as e:JOBS[pid].update(state='cancelled',message=str(e),finished=time.time())
  except Exception as e:
   import traceback
   (folder(pid)/f'error_{int(time.time())}.log').write_text(traceback.format_exc());JOBS[pid].update(state='error',message=str(e))
 threading.Thread(target=worker,daemon=True).start()

def detection_pool(mode,terms):
 if mode=='faces':
  from identity import FacePool
  return FacePool()
 from dynamic_mosaic import PoolOCR
 return PoolOCR(4,terms,[],mode)

def detect(pid,mode_override=None):
 p=load(pid);terms=[x.strip() for x in re.split('[,，\n]',p['terms']) if x.strip()]
 mode=mode_override or p.get('target_mode','text')
 if mode=='text' and not terms:raise ValueError('先填写目标文字；纯图标请使用框选')
 if mode=='faces':terms=[]
 from dynamic_mosaic import PoolOCR
 from collections import deque
 pool=detection_pool(mode,terms);pending=deque();rows=[];reader=FrameReader(p['source'],pts_for(p),Fraction(p['fps_ratio']),p['duration'])
 cache=folder(pid)/('ocr_'+sha256_text(json.dumps([mode,terms,'yunet-cpu-v1' if mode=='faces' else 'vision'],ensure_ascii=False))+'.jsonl')
 old={};cuts=[];previous=None
 if cache.exists():
  for line in cache.read_text().splitlines():
   try:r=json.loads(line);old[r['frame']]=r
   except json.JSONDecodeError:pass
 with cache.open('a') as log:
  def collect():
   r=pending.popleft().result();rows.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
  try:
   for i in range(p['frames']):
    check_cancel(pid)
    ok,im=reader.read()
    if not ok:raise ValueError(f'无法读取帧{i}')
    gray=cv2.cvtColor(cv2.resize(im,(160,90)),cv2.COLOR_BGR2GRAY)
    if previous is not None and cut_candidate(previous,gray):cuts.append(i)
    previous=gray
    if i in old:rows.append(old[i])
    else:
     pending.append(pool.pool.submit(pool.job,i,im))
     if len(pending)>=8:collect()
    if i%30==0:progress(pid,i/p['frames']*95,f'全片逐帧初筛 {i}/{p["frames"]} 帧；结果需复核')
   while pending:collect()
  finally:reader.release();pool.close()
 boxes={str(i):[] for i in range(p['frames'])}
 for r in rows:
  for h in r['hits']:boxes[str(r['frame'])].append(pixel_rect(h['box'],p['width'],p['height'],12))
 from stabilize import stable_tracks
 timeline=[boxes[str(i)] for i in range(p['frames'])]
 if mode=='faces':
  from faces import face_tracks
  tracks=face_tracks(timeline,p['width'],p['height'],cuts)
 else:tracks=stable_tracks(timeline,p['width'],p['height'])
 for t in tracks:
  t.update(id=mode+'_'+uuid.uuid4().hex[:12],auto_mode=mode,block=p.get('mask_block',32),effect=p.get('mask_style','cover'),name=('人脸' if mode=='faces' else '文字')+'候选 '+t['name'].split()[-1])
 validate_tracks(tracks,p['frames'])
 with LOCK:
  p=load(pid);hist=folder(pid)/'history';hist.mkdir(exist_ok=True);write_json(hist/f'rev_{p["revision"]}.json',p)
  p['tracks']=[t for t in p['tracks'] if t.get('auto_mode')!=mode]+tracks
  for l in p['layers']:
   if mode=='text' and l['id']=='detection':l['enabled']=False
  p['revision']+=1;p['status']='NEEDS_REVIEW'
  if mode=='faces':p['face_scan_complete']=True;p.pop('face_catalog',None)
  persist(p)

def prepare_faces(pid):
 p=load(pid)
 if p.get('face_catalog',{}).get('version')==1:return
 if not p.get('face_scan_complete') and not any(t.get('auto_mode')=='faces' for t in p['tracks']):detect(pid,'faces')
 check_cancel(pid);p=load(pid);progress(pid,96,'正在把同一人的多段出现整理到一个头像…')
 from identity import build_catalog
 def checked_frame(project,i):check_cancel(pid);return frame_read(project,i)
 catalog=build_catalog(p,checked_frame)
 check_cancel(pid)
 with LOCK:
  p=load(pid);hist=folder(pid)/'history';hist.mkdir(exist_ok=True);write_json(hist/f'rev_{p["revision"]}.json',p)
  p['face_catalog']=catalog;p['face_scan_complete']=True;p['revision']+=1;persist(p)
 finalize_faces(pid)

def finalize_faces(pid,enable=True):
 """人脸码的安全余量：框外扩到盖住整个头；前后几帧取并集防抖动/转头；同一个人的分段之间补上空隙；默认全部遮住。"""
 with LOCK:
  p=load(pid);snapshot(p);W,H=p['width'],p['height'];F=p['frames'];win=max(2,round(p['fps']*0.08))
  faces=[t for t in p['tracks'] if t.get('auto_mode')=='faces' or t.get('tracking_kind')=='face']
  for t in faces:
   ks=sorted(t['keys'],key=lambda k:k['frame'])
   if not ks:continue
   frames=[k['frame'] for k in ks];boxes=[k['box'] for k in ks]
   import bisect
   newk=[]
   for i,f in enumerate(frames):
    lo=bisect.bisect_left(frames,f-win);hi=bisect.bisect_right(frames,f+win)
    x0=min(b[0] for b in boxes[lo:hi]);y0=min(b[1] for b in boxes[lo:hi]);x1=max(b[0]+b[2] for b in boxes[lo:hi]);y1=max(b[1]+b[3] for b in boxes[lo:hi])
    newk.append({'frame':f,'box':[x0,y0,x1-x0,y1-y0]})
   t['keys']=newk
   size=sorted(max(k['box'][2]*W,k['box'][3]*H) for k in newk)[len(newk)//2]
   t['padding']=int(min(160,max(12,round(size*0.32))))
   t['shape']='rect';t['opacity']=100
   if t.get('effect') in (None,'cover'):t['effect']='mosaic';t['strength']=60
   t.setdefault('strength',60)
   if enable:t['enabled']=True
   t['start']=max(0,t['start']-win);t['end']=min(F-1,t['end']+win)
  # 同一个人的相邻片段：间隔 2 秒内补上，用前后两个框的并集遮住中间
  gapmax=round(2*p['fps'])
  for g in p.get('face_catalog',{}).get('groups',[]):
   ts=sorted([t for t in p['tracks'] if t['id'] in g['track_ids']],key=lambda t:t['start'])
   for a,b in zip(ts,ts[1:]):
    gap=b['start']-a['end']-1
    if 0<gap<=gapmax:
     ba=a['keys'][-1]['box'];bb=b['keys'][0]['box'];x0=min(ba[0],bb[0]);y0=min(ba[1],bb[1]);x1=max(ba[0]+ba[2],bb[0]+bb[2]);y1=max(ba[1]+ba[3],bb[1]+bb[3])
     a['keys'].append({'frame':b['start']-1,'box':[x0,y0,x1-x0,y1-y0]});a['end']=b['start']-1
  p['revision']+=1;persist(p)


def sha256_text(s):
 import hashlib
 return hashlib.sha256(s.encode()).hexdigest()[:16]

def track(pid,tid,anchor):
 p=load(pid);t=next(t for t in p['tracks'] if t['id']==tid);seed=box_at(t,anchor)
 if seed is None:raise ValueError('当前帧不在此区域的时间范围内')
 if t.get('tracking_kind')=='face' or t.get('auto_mode')=='faces':
  from face_follow import follow
  pts=pts_for(p);reader=FrameReader(p['source'],pts,Fraction(p['fps_ratio']),p['duration'])
  def frames(start,end):
   if start>end:return
   wanted=int(reader.mapping[start]);reader.cap.set(cv2.CAP_PROP_POS_FRAMES,wanted);reader.source_index=wanted-1;reader.index=start
   for i in range(start,end+1):
    ok,im=reader.read()
    if not ok:raise ValueError(f'无法读取跟随帧 {i}')
    yield i,im
  try:keys,gaps,evidence=follow(p,t,anchor,folder(pid),lambda i:frame_read(p,i),frames,lambda percent,msg:progress(pid,percent,msg),lambda:check_cancel(pid))
  finally:reader.release()
  check_cancel(pid)
  with LOCK:
   p=load(pid);hist=folder(pid)/'history';hist.mkdir(exist_ok=True);write_json(hist/f'rev_{p["revision"]}.json',p)
   t=next(t for t in p['tracks'] if t['id']==tid)
   prefix=[k for k in t['keys'] if k['frame']<anchor]
   if anchor>t['start']:
    previous=box_at(t,anchor-1);prefix=[k for k in prefix if k['frame']!=anchor-1]+[{'frame':anchor-1,'box':previous}]
   t.update(keys=prefix+keys,tracking_gaps=gaps,tracking_kind='face',tracking_evidence=evidence,reviewed=False,tracking_note='未确认区间仅保留或短时推测位置，不能保证遮住；请定位检查')
   # A followed face replaces its group's future segments, avoiding duplicate masks.
   for group in p.get('face_catalog',{}).get('groups',[]):
    if tid not in group['track_ids']:continue
    retained=[]
    for other in p['tracks']:
     if other['id'] not in group['track_ids'] or other['id']==tid:continue
     if other['start']>=anchor:other['enabled']=False
     else:
      other['end']=min(other['end'],anchor-1);retained.append(other['id'])
    group['track_ids']=retained+[tid];group['representative']=tid
    group['review_gaps']=[g for g in group.get('review_gaps',[]) if g[1]<anchor]+gaps
   p['revision']+=1;persist(p)
  return
 # Identity is supplied by the user's box, not guessed by a universal classifier.
 results={anchor:seed};lost=[];cap=cv2.VideoCapture(p['source']);pts=pts_for(p);read=FrameReader(p['source'],pts,Fraction(p['fps_ratio']),p['duration']);mapping=read.mapping;read.release()
 def get(i):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(mapping[i]));ok,f=cap.read()
  if not ok:raise ValueError('跟踪源帧读取失败')
  return cv2.cvtColor(cv2.resize(f,(640,360)),cv2.COLOR_BGR2GRAY)
 for direction,end in [(1,t['end']),(-1,t['start'])]:
  last=get(anchor);box=seed;reference=appearance_patch(last,seed);failed=False
  for i in range(anchor+direction,end+direction,direction):
   gray=get(i);candidate=None if failed or cut_candidate(last,gray) else flow_box(last,gray,box)
   if candidate is not None and same_appearance(reference,appearance_patch(gray,candidate)):box=candidate
   else:failed=True;lost.append(i) # hold last box, visibly flag failure; never silently remove it
   results[i]=box;last=gray
   if i%15==0:progress(pid,len(results)/max(1,t['end']-t['start']+1)*95,f'跟踪 {len(results)} 帧；需检查 {len(lost)} 帧')
 cap.release()
 with LOCK:
  p=load(pid);t=next(t for t in p['tracks'] if t['id']==tid);t['keys']=[{'frame':i,'box':results[i]} for i in sorted(results)];t['reviewed']=False;t['tracking_gaps']=contiguous_ranges(lost);t['tracking_note']='失效后仅保留上一位置，不代表仍跟上目标；请在红色区间补位置关键帧';p['revision']+=1;persist(p)

def new_mask_track(p,polys,frame,name,group=None,extra=None):
 W,H=p['width'],p['height'];bb=ai.polys_bbox(polys,W,H);st=style_for(p);n=len(p['tracks'])
 t={'id':'obj_'+uuid.uuid4().hex[:12],'name':name,'kind':'object','mask':True,'shape':'contour','start':frame,'end':frame,'padding':st['padding'],'block':st['block'],
    'effect':st['effect'],'strength':int(st['strength']),'opacity':st['opacity'],'color':st['color'],'feather':st['feather'],'enabled':True,'reviewed':False,'tracked':False,
    'keys':[{'frame':frame,'box':bb}],'tag':PALETTE[n%len(PALETTE)]}
 if st['effect']=='sticker':t['sticker']=st['sticker']
 if group:t['group']=group
 if extra:t.update(extra)
 return t

def ai_click(pid,body):
 p=load(pid);i=int(body['frame']);W,H=p['width'],p['height']
 if not 0<=i<p['frames']:raise ValueError('帧超出范围')
 pts=[[float(a)*W,float(b)*H,1 if int(l) else 0] for a,b,l in body.get('points',[])][:40]
 box=body.get('box');box=[box[0]*W,box[1]*H,box[2]*W,box[3]*H] if box else None
 if not pts and not box:raise ValueError('请在画面上点一下要遮的东西')
 im=frame_read(p,i);polys,score=ai.segment_points(im,pts,box=box,cache_key=f'{pid}:{i}')
 if not polys:raise ValueError('这里没抠出东西，换个位置点一下试试')
 with LOCK:
  p=load(pid);snapshot(p);tid=body.get('track')
  if tid:
   t=next(t for t in p['tracks'] if t['id']==tid)
   frames=dict(load_masks(pid,tid)) if t.get('mask') else {}
   frames[str(i)]=polys;bb=ai.polys_bbox(polys,W,H)
   if not t.get('mask'):t.update(mask=True,shape='rect' if t.get('shape','rect')!='ellipse' else 'ellipse',keys=[])
   t['keys']=sorted([k for k in t['keys'] if k['frame']!=i]+[{'frame':i,'box':bb}],key=lambda k:k['frame'])
   t['start']=min(t['start'],i);t['end']=max(t['end'],i);t['reviewed']=False
   t.setdefault('prompts',{})[str(i)]={'points':body.get('points',[]),'box':body.get('box')}
  else:
   n=sum(1 for t in p['tracks'] if t.get('kind')=='object')+1
   t=new_mask_track(p,polys,i,body.get('name') or f'物体 {n}',extra={'prompts':{str(i):{'points':body.get('points',[]),'box':body.get('box')}}})
   if body.get('shape') in ['rect','ellipse']:t['shape']=body['shape']
   p['tracks'].append(t);tid=t['id'];frames={str(i):polys}
  save_masks(pid,tid,frames);p['revision']+=1;persist(p)
 return {'project':p,'track':tid,'score':round(score,3)}

def ai_words(prompt):
 words,unknown=ai.translate(prompt)
 if not words:raise ValueError(('暂时不认识「'+'、'.join(unknown)+'」。' if unknown else '先写要遮什么。')+'可以写：人脸、车牌、logo、手机屏幕、二维码、杯子…也可以直接写英文')
 return words,unknown

def ai_detect_frame(pid,body):
 p=load(pid);i=int(body['frame']);words,unknown=ai_words(str(body.get('prompt',''))[:200]);im=frame_read(p,i)
 found=[]
 for label,w in words:
  for d in ai.detect(im,[w],box_threshold=float(body.get('threshold',0.3))):found.append((label,d))
 made=[]
 run=uuid.uuid4().hex[:6]
 for label,d in found:
  polys,score=ai.segment_points(im,[],box=d['box'],cache_key=f'{pid}:{i}')
  if polys:made.append((label,polys,d['score']))
 with LOCK:
  p=load(pid);snapshot(p);ids=[]
  counts={}
  for label,polys,score in made:
   counts[label]=counts.get(label,0)+1
   t=new_mask_track(p,polys,i,f'{label} {counts[label]}',group=f'{label}·{run}',extra={'kind':'object','detect_score':score,'prompt':label})
   p['tracks'].append(t);save_masks(pid,t['id'],{str(i):polys});ids.append(t['id'])
  if made:p['revision']+=1;persist(p)
 return {'project':p,'tracks':ids,'count':len(ids),'unknown':unknown}

def ai_track(pid,tids,frame,direction='both',seed_polys=None,stop_after=None,on_done=None,stride=2):
 """AI 全程追踪：从当前帧往后（和往前）逐帧跟随，轮廓存入 masks/。"""
 p=load(pid);W,H=p['width'],p['height'];F=p['frames'];scale=min(1.0,1280/max(W,H))
 tracks=[next(t for t in p['tracks'] if t['id']==tid) for tid in tids]
 seeds={}
 im=None
 for k,t in enumerate(tracks):
  polys=(seed_polys or {}).get(t['id']) or (polys_at(load_masks(pid,t['id']),frame) if t.get('mask') else None)
  if not polys:
   b=box_at(t,frame)
   if b is None:continue
   if im is None:im=frame_read(p,frame)
   polys,_=ai.segment_points(im,[],box=[b[0]*W,b[1]*H,(b[0]+b[2])*W,(b[1]+b[3])*H])
   if not polys:continue
  sp=[[round(v*scale) for v in q] for q in polys]
  seeds[k+1]=ai.polys_to_mask(sp,round(W*scale),round(H*scale))
 if not seeds:raise ValueError('当前帧没有找到要追踪的物体，请先在这一帧点选它')
 results={k:{} for k in seeds};stop_after=stop_after or round(4*p['fps'])
 legs=[(frame,F-1)]+([(frame,0)] if direction=='both' and frame>0 else [])
 total=sum(abs(b-a)//stride+1 for a,b in legs);done=[0];began=time.time();keyframes={k:set() for k in seeds}
 for a,b in legs:
  absent=[0]
  class Stop(Exception):pass
  def on_frame(i,res):
   done[0]+=1
   anyhere=False
   for oid,polys in res.items():
    full=[[round(v/scale) for v in q] for q in polys]
    results[oid][str(i)]=full;keyframes[oid].add(i)
    if full:anyhere=True
   absent[0]=0 if anyhere else absent[0]+stride
   if done[0]%10==0:
    rate=done[0]*stride/max(.1,time.time()-began);left=(total-done[0])*stride/max(.01,rate)
    eta=f'，约剩 {math.ceil(left/60)} 分钟' if left>=60 else f'，约剩 {int(left)+1} 秒'
    progress(pid,min(97,done[0]/total*97),f'AI 全程追踪 {done[0]*stride}/{total*stride} 帧（{rate:.1f} 帧/秒{eta}）'+('，正在往前补' if b<a else ''))
   if absent[0]>stop_after:raise Stop()
  def sampled(gen,a=a,b=b):
   for i,im in gen:
    if abs(i-a)%stride==0 or i==b:yield i,im
  try:ai.track(sampled(iter_frames(p,a,b)),seeds,W,H,on_frame,check=lambda:check_cancel(pid))
  except Stop:pass
 # 跳过的帧：用前后两个关键帧的轮廓合在一起遮，覆盖中间的移动路径
 for oid,ks in keyframes.items():
  ks=sorted(ks)
  for k0,k1 in zip(ks,ks[1:]):
   if 1<k1-k0<=stride:
    both=results[oid].get(str(k0),[])+results[oid].get(str(k1),[])
    for f in range(k0+1,k1):results[oid][str(f)]=both
 progress(pid,98,'整理追踪结果…')
 with LOCK:
  p=load(pid);snapshot(p);changed=[]
  for k,t0 in enumerate(tracks):
   if k+1 not in results:continue
   t=next(x for x in p['tracks'] if x['id']==t0['id']);frames=dict(load_masks(pid,t['id'])) if t.get('mask') else {}
   frames.update(results[k+1])
   if not t.get('mask'):t.update(mask=True,shape='rect' if t.get('shape','rect')!='ellipse' else 'ellipse')
   present=sorted(int(f) for f,v in frames.items() if v)
   if not present:continue
   t['keys']=[{'frame':f,'box':ai.polys_bbox(frames[str(f)],W,H)} for f in present]
   t['keys']=[k2 for k2 in t['keys'] if k2['box']]
   t['start'],t['end']=present[0],present[-1]
   gaps=[];prev=present[0]
   for f in present[1:]:
    if f-prev-1>HOLD:gaps.append([prev+1,f-1])
    prev=f
   t.update(tracked=True,tracking_kind='ai',tracking_gaps=gaps,reviewed=False,tracking_note='AI 认为这些区间目标不在画面里；如果其实还在，请点「细化」在那一帧补点后重新追踪')
   save_masks(pid,t['id'],frames);changed.append(t['id'])
  p['revision']+=1;persist(p)
 return changed

def ai_scan(pid,prompt):
 """全片一句话识别：每秒抽一帧找目标，新出现的就抠出来并追踪。"""
 p=load(pid);words,_=ai_words(prompt);W,H=p['width'],p['height'];step=max(1,round(p['fps']));F=p['frames']
 run=uuid.uuid4().hex[:6];counts={};covered={}  # frame -> list of boxes already handled
 samples=list(range(0,F,step))
 for n,i in enumerate(samples):
  check_cancel(pid);progress(pid,n/len(samples)*90,f'全片识别「{prompt}」：第 {i//step} 秒，已找到 {sum(counts.values())} 个')
  p=load(pid);im=frame_read(p,i);new=[]
  for label,w in words:
   for d in ai.detect(im,[w]):
    b=d['box'];mine=[]
    for t in p['tracks']:
     if t.get('scan_run')!=run:continue
     bb=box_at(t,i)
     if bb and t['start']<=i<=t['end']:mine.append([bb[0]*W,bb[1]*H,(bb[0]+bb[2])*W,(bb[1]+bb[3])*H])
    if any(ai._iou(b,m)>0.25 for m in mine):continue
    new.append((label,d))
  if not new:continue
  made={}
  with LOCK:
   p=load(pid);snapshot(p)
   for label,d in new[:20]:
    polys,_=ai.segment_points(im,[],box=d['box'],cache_key=f'{pid}:{i}')
    if not polys:continue
    counts[label]=counts.get(label,0)+1
    t=new_mask_track(p,polys,i,f'{label} {counts[label]}',group=f'{label}·{run}',extra={'scan_run':run,'prompt':label})
    p['tracks'].append(t);save_masks(pid,t['id'],{str(i):polys});made[t['id']]=polys
   p['revision']+=1;persist(p)
  if made:ai_track(pid,list(made),i,'both',seed_polys=made,stop_after=round(2*p['fps']))
  if sum(counts.values())>=80:break
 progress(pid,99,'完成')

def export(pid):
 p=load(pid);validate_tracks(p['tracks'],p['frames']);assert sha256(p['source'])==p['source_hash'],'源文件已变化，请重新导入'
 layers=layers_for(p);version=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4];out=folder(pid)/'exports'/version;out.mkdir(parents=True)
 write_json(out/'project_snapshot.json',p);write_json(out/'layers_snapshot.json',layers)
 final=out/'打码版.mp4';cmd=[binary('ffmpeg'),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{p["width"]}x{p["height"]}','-framerate',p['fps_ratio'],'-i','pipe:0','-i',p['source'],'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-c:a','copy','-map_metadata','-1','-movflags','+faststart',str(final)]
 core_fresh();write_json(out/'render_command.json',cmd);ml=mask_loader(pid);sl=sticker_loader(pid)
 if (folder(pid)/'masks').is_dir():shutil.copytree(folder(pid)/'masks',out/'masks')
 reader=FrameReader(p['source'],pts_for(p),Fraction(p['fps_ratio']),p['duration']);timeline={}
 with (out/'render.log').open('w') as err:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
  try:
   for i in range(p['frames']):
    ok,im=reader.read()
    if not ok:raise ValueError('源帧读取失败')
    ms=masks_at(p,i,layers,ml);timeline[str(i)]=[{k:m[k] for k in ('id','box','effect','shape')} for m in ms]
    render_masks(im,ms,sl)
    proc.stdin.write(im.tobytes())
    if i%30==0:progress(pid,i/p['frames']*80,f'导出 {i}/{p["frames"]} 帧')
  finally:reader.release();proc.stdin.close()
  if proc.wait():raise ValueError('编码失败，详见渲染日志')
 write_json(out/'mask_timeline.json',{'frames':p['frames'],'fps':p['fps_ratio'],'masks':timeline})
 progress(pid,82,'检查实际导出：完整解码、规格与音频一致性')
 dec=subprocess.run([binary('ffmpeg'),'-v','error','-i',str(final),'-f','null','-'],capture_output=True,text=True);(out/'decode.log').write_text(dec.stderr)
 meta=probe(final);v=next(s for s in meta['streams'] if s['codec_type']=='video');audio=[]
 for i,_ in enumerate(s for s in meta['streams'] if s['codec_type']=='audio'):
  hs=[subprocess.check_output([binary('ffmpeg'),'-v','error','-i',str(f),'-map',f'0:a:{i}','-c:a','pcm_s16le','-f','hash','-hash','sha256','-']).decode().strip() for f in [p['source'],final]];audio.append({'equal':hs[0]==hs[1],'hashes':hs})
 specs=v['width']==p['width'] and v['height']==p['height'] and int(v['nb_frames'])==p['frames'] and abs(float(Fraction(v['avg_frame_rate']))-p['fps'])<.001
 passed=dec.returncode==0 and not dec.stderr.strip() and specs and all(a['equal'] for a in audio)
 qc={'technical':'PASS' if passed else 'FAIL','status':'REVIEW_REQUIRED','decode':dec.returncode==0 and not dec.stderr.strip(),'specs':specs,'audio':audio,'source_unchanged':sha256(p['source'])==p['source_hash'],'content_review':'UNVERIFIED','note':'技术通过不等于零漏码；没有用同一识别器无命中来宣称完整覆盖','metadata':meta}
 write_json(out/'qc_report.json',qc)
 if not passed:raise ValueError('导出技术检查失败；请查看QC报告')
 with LOCK:
  current=load(pid);current['exports'].append({'version':version,'file':str(final.relative_to(folder(pid))),'revision':p['revision'],'status':'REVIEW_REQUIRED','time':time.time()});persist(current)

class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def send_json(self,obj,status=200):
  data=json.dumps(obj,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',len(data));self.end_headers();self.wfile.write(data)
 def check_host(self):
  host=self.headers.get('Host','');return host in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
 def do_GET(self):
  try:
   if not self.check_host():return self.send_json({'error':'不接受此Host'},403)
   u=urlparse(self.path);parts=u.path.strip('/').split('/');q=parse_qs(u.query)
   if u.path.startswith('/api/import/'):return self.send_json(JOBS.get(u.path.rsplit('/',1)[1],{'state':'missing','message':'找不到这个导入任务'}))
   if u.path=='/api/ai/status':return self.send_json(ai.status())
   if u.path=='/api/bootstrap':return self.send_json({'token':TOKEN,'projects':[json.loads(p.read_text()) for p in DATA.glob('*/project.json')],'jobs':JOBS})
   if len(parts)>=3 and parts[:2]==['api','project']:
    p=load(parts[2]);action=parts[3] if len(parts)>3 else ''
    if not action:return self.send_json({'project':p,'job':JOBS.get(p['id']),'layers':layers_for(p)})
    if action=='job':return self.send_json(JOBS.get(p['id'],{'state':'idle'}))
    if action=='source':return self.serve_file(Path(p['source']))
    if action=='face-thumb':
     t=next(t for t in p['tracks'] if t['id']==q.get('id',[''])[0] and t.get('auto_mode')=='faces')
     k=t['keys'][len(t['keys'])//2];im=frame_read(p,k['frame']);x,y,w,h=pixel_rect(k['box'],p['width'],p['height'],8);im=im[y:y+h,x:x+w]
     ok,b=cv2.imencode('.jpg',cv2.resize(im,(96,96)));self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',len(b));self.end_headers();return self.wfile.write(b.tobytes())
    if action=='ai-bench':
     n=max(3,min(60,int(q.get('n',['15'])[0])));dev=q.get('dev',[''])[0];start=int(q.get('f',['0'])[0]);dt=q.get('dtype',[''])[0]
     import torch
     old=os.environ.get('MOSAIC_AI_DEVICE');oldt=os.environ.get('MOSAIC_AI_DTYPE')
     if dt in ('fp16','fp32'):os.environ['MOSAIC_AI_DTYPE']=dt;ai._STATE.pop('video',None)
     if dev in ('cpu','mps'):os.environ['MOSAIC_AI_DEVICE']=dev;ai._STATE.pop('video',None)
     try:
      im=frame_read(p,start);W,H=p['width'],p['height'];scale=min(1.0,1280/max(W,H))
      polys,_=ai.segment_points(im,[[W/2,H/2,1]])
      seed=ai.polys_to_mask([[round(v*scale) for v in q2] for q2 in polys],round(W*scale),round(H*scale))
      T={'profile':q.get('profile',[''])[0]=='1','mem':int(q.get('mem',['0'])[0])};t0=time.perf_counter();ai.track(iter_frames(p,start,min(p['frames']-1,start+n-1)),{1:seed},W,H,lambda i,r:None,timings=T);total=time.perf_counter()-t0
      info={'device':ai.device_name(),'dtype':str(ai._STATE.get('video_dtype')),'torch':torch.__version__,'mps':torch.backends.mps.is_available(),'threads':torch.get_num_threads(),'frames':T.get('frames'),'seconds':round(total,2),'fps':round(T.get('frames',0)/total,2),'stages_ms_per_frame':{k:round(v*1000/max(1,T.get('frames',1)),1) for k,v in T.items() if k not in ('frames','profile','mem') and not k.startswith('_t_')}}
     finally:
      if dt in ('fp16','fp32'):
       ai._STATE.pop('video',None)
       if oldt is None:os.environ.pop('MOSAIC_AI_DTYPE',None)
       else:os.environ['MOSAIC_AI_DTYPE']=oldt
      if dev in ('cpu','mps'):
       ai._STATE.pop('video',None)
       if old is None:os.environ.pop('MOSAIC_AI_DEVICE',None)
       else:os.environ['MOSAIC_AI_DEVICE']=old
     return self.send_json(info)
    if action=='mask':return self.send_json({'frames':load_masks(p['id'],q.get('t',[''])[0])})
    if action=='sticker':return self.serve_file(sticker_file(p['id'],q.get('n',[''])[0]))
    if action=='thumb':
     t=next(t for t in p['tracks'] if t['id']==q.get('id',[''])[0])
     k=t['keys'][len(t['keys'])//2];im=frame_read(p,k['frame']);x,y,w,h=pixel_rect(k['box'],p['width'],p['height'],max(6,int(0.15*max(k['box'][2]*p['width'],k['box'][3]*p['height']))));im=im[y:y+h,x:x+w]
     s2=96/max(w,h);ok,b=cv2.imencode('.jpg',cv2.resize(im,(max(1,round(w*s2)),max(1,round(h*s2)))));self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Cache-Control','max-age=600');self.send_header('Content-Length',len(b));self.end_headers();return self.wfile.write(b.tobytes())
    if action=='frame':
     i=int(q.get('i',['0'])[0]);im=frame_read(p,i)
     if q.get('masked')==['1']:render_frame(p,im,i)
     ok,b=cv2.imencode('.png',im);self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',len(b));self.end_headers();return self.wfile.write(b.tobytes())
    if action=='output':
     version=q.get('v',[''])[0];e=next(e for e in p['exports'] if e['version']==version);return self.serve_file(folder(p['id'])/e['file'])
    if action=='qc':
     e=next(e for e in p['exports'] if e['version']==q.get('v',[''])[0]);return self.send_json(json.loads((folder(p['id'])/e['file']).with_name('qc_report.json').read_text()))
   path=APP/'static'/('index.html' if u.path=='/' else u.path.lstrip('/'))
   if path.resolve().parent!=(APP/'static').resolve():return self.send_json({'error':'禁止访问'},403)
   return self.serve_file(path)
  except (ValueError,FileNotFoundError,StopIteration) as e:self.send_json({'error':str(e)},404)
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:self.send_json({'error':str(e)},500)
 def serve_file(self,path):
  if not path.is_file():raise FileNotFoundError('文件不存在')
  size=path.stat().st_size;start,end=0,size-1;partial=False;r=self.headers.get('Range')
  if r:
   m=re.fullmatch(r'bytes=(\d*)-(\d*)',r)
   if not m:raise ValueError('无效Range')
   if m[1]:start=int(m[1]);end=min(end,int(m[2])) if m[2] else end
   elif m[2]:start=max(0,size-int(m[2]))
   if start>=size or end<start:
    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
   partial=True
  self.send_response(206 if partial else 200);self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',end-start+1);self.send_header('Cache-Control','no-cache')
  if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  with path.open('rb') as f:
   f.seek(start);left=end-start+1
   while left:
    b=f.read(min(left,1024*1024))
    if not b:break
    self.wfile.write(b);left-=len(b)
 def do_POST(self):
  try:
   if not self.check_host() or self.headers.get('X-Workbench-Token')!=TOKEN:return self.send_json({'error':'请从本机工作台发起操作'},403)
   origin=self.headers.get('Origin')
   if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:return self.send_json({'error':'不接受外站请求'},403)
   parts=urlparse(self.path).path.strip('/').split('/');length=int(self.headers.get('Content-Length','0'))
   if parts==['api','upload']:
    if not 0<length<=4*1024**3:raise ValueError('视频需在4GB以内')
    name=unquote(self.headers.get('X-Filename','video.mp4'));ext=Path(name).suffix.lower()
    if ext not in ['.mp4','.mov','.m4v','.mkv','.webm']:raise ValueError('请选择视频文件')
    uploads=DATA/'uploads';uploads.mkdir(exist_ok=True);dest=uploads/(uuid.uuid4().hex+ext)
    with dest.open('xb') as f:
     remaining=length
     while remaining:
      chunk=self.rfile.read(min(1024*1024,remaining))
      if not chunk:raise ValueError('上传中断')
      f.write(chunk);remaining-=len(chunk)
    jid='imp_'+uuid.uuid4().hex[:10];quick=self.headers.get('X-Quick','')
    JOBS[jid]={'kind':'import','state':'running','progress':0,'message':'检查视频'}
    def run_import():
     try:
      p=create_project(dest,Path(name).name,on_progress=lambda pc:JOBS[jid].update(progress=round(pc),message='转换手机视频'))
      JOBS[jid].update(state='done',progress=100,project=p['id'],message='完成')
      if quick=='face':launch_job(p['id'],'prepare-faces',lambda:prepare_faces(p['id']))
     except Exception as e:JOBS[jid].update(state='error',message=str(e))
    threading.Thread(target=run_import,daemon=True).start()
    return self.send_json({'import':jid})
   if parts==['api','restart']:
    self.send_json({'ok':True});srv=self.server
    def again():
     time.sleep(.4);srv.server_close()
     log=(APP/'server.log').open('a')
     subprocess.Popen([sys.executable,str(Path(__file__).resolve())]+sys.argv[1:],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=str(APP))
     os._exit(0)
    threading.Thread(target=again,daemon=True).start();return
   if len(parts)==4 and parts[:2]==['api','project'] and parts[3]=='sticker':
    if not 0<length<=8*1024**2:raise ValueError('贴纸图片需小于 8MB')
    raw=self.rfile.read(length);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    if im is None:raise ValueError('无法识别这张图片，请用 PNG / JPG / WebP')
    if im.ndim==2:im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGRA)
    elif im.shape[2]==3:im=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
    s3=min(1,512/max(im.shape[:2]));im=cv2.resize(im,(max(1,round(im.shape[1]*s3)),max(1,round(im.shape[0]*s3))),interpolation=cv2.INTER_AREA) if s3<1 else im
    ok,png=cv2.imencode('.png',im);name=hashlib.sha256(png.tobytes()).hexdigest()[:16]+'.png';f=sticker_file(parts[2],name);f.parent.mkdir(exist_ok=True)
    if not f.exists():f.write_bytes(png.tobytes())
    return self.send_json({'name':name})
   if length>10*1024**2:raise ValueError('请求过大')
   body=json.loads(self.rfile.read(length) or b'{}')
   if len(parts)<3 or parts[:2]!=['api','project']:raise ValueError('未知操作')
   pid=parts[2];p=load(pid);action='/'.join(parts[3:])
   if not action:
    with LOCK:
     if JOBS.get(pid,{}).get('state')=='running':return self.send_json({'error':'任务运行中，请等待结束后保存'},409)
     p=load(pid)
     if body.get('revision')!=p['revision']:return self.send_json({'error':'工程已更新，请重新载入'},409)
     mode=body.get('target_mode',p.get('target_mode','text'));style=body.get('mask_style',p.get('mask_style','cover'))
     if mode not in ['text','faces'] or style not in EFFECTS:raise ValueError('未知目标或遮挡方式')
     block=body.get('mask_block',p.get('mask_block',32))
     if not isinstance(block,int) or not 2<=block<=100:raise ValueError('请选择有效遮挡强度')
     ds=body.get('default_style',p.get('default_style'))
     if ds is not None:
      if not isinstance(ds,dict) or ds.get('effect',ds.get('effect','mosaic')) not in EFFECTS:raise ValueError('默认样式不合法')
      p['default_style']={k:ds[k] for k in ['effect','block','strength','opacity','padding','color','feather','sticker'] if k in ds}
     tracks=validate_tracks(body['tracks'],p['frames']);hist=folder(pid)/'history';hist.mkdir(exist_ok=True);write_json(hist/f'rev_{p["revision"]}.json',p)
     p.update(target_mode=mode,mask_style=style,mask_block=block,tracks=tracks,terms=str(body.get('terms',p['terms']))[:500],revision=p['revision']+1)
     for l in p['layers']:l['enabled']=bool(body.get('layer_enabled',{}).get(l['id'],l.get('enabled',True)))
     persist(p)
    return self.send_json(p)
   busy=JOBS.get(pid,{}).get('state')=='running'
   if action in ['ai/click','ai/detect','revert'] and busy:return self.send_json({'error':'后台任务运行中，请等它完成或先停止'},409)
   if action=='delete':
    if busy:return self.send_json({'error':'这个视频正在处理，等处理完再删'},409)
    with LOCK:
     trash=DATA/'.回收站';trash.mkdir(exist_ok=True);src=Path(p['source']);dest=trash/f"{time.strftime('%Y%m%d_%H%M%S')}_{pid}"
     shutil.move(str(folder(pid)),str(dest))
     # 导入时复制进来的视频，若没有其他草稿在用，也一起放进回收站
     up=(DATA/'uploads').resolve()
     if src.is_file() and src.resolve().parent==up and not any(json.loads(f.read_text()).get('source')==p['source'] for f in DATA.glob('*/project.json')):
      shutil.move(str(src),str(dest/src.name))
     JOBS.pop(pid,None)
    return self.send_json({'ok':True})
   if action=='rename':
    name=str(body.get('name','')).strip()[:80]
    if not name:raise ValueError('名字不能为空')
    with LOCK:
     p=load(pid);p['name']=name;persist(p)
    return self.send_json(p)
   if action=='ai/click':return self.send_json(ai_click(pid,body))
   if action=='ai/detect':
    if body.get('scope')=='video':launch_job(pid,'ai-scan',lambda:ai_scan(pid,str(body.get('prompt',''))[:200]));return self.send_json(JOBS[pid])
    return self.send_json(ai_detect_frame(pid,body))
   if action=='ai/track':
    tids=[str(x) for x in body.get('tracks',[])][:60]
    if not tids:raise ValueError('先选择要追踪的物体')
    stride={'fine':1,'standard':2,'fast':3}.get(body.get('quality'),2)
    launch_job(pid,'ai-track',lambda:ai_track(pid,tids,int(body['frame']),'forward' if body.get('direction')=='forward' else 'both',stride=stride));return self.send_json(JOBS[pid])
   if action=='revert':
    r=int(body['revision']);f=folder(pid)/'history'/f'rev_{r}.json'
    if not f.is_file():raise ValueError('找不到这一步的历史记录')
    with LOCK:
     cur=load(pid);snapshot(cur);old=json.loads(f.read_text());old['revision']=cur['revision']+1;old['exports']=cur['exports'];persist(old)
    return self.send_json(old)
   if action=='cancel':
    if JOBS.get(pid,{}).get('kind') not in ['prepare-faces','detect','track','ai-track','ai-scan']:raise ValueError('此任务不能在这里停止')
    if pid in CANCEL:CANCEL[pid].set()
   elif action=='prepare-faces':launch_job(pid,'prepare-faces',lambda:prepare_faces(pid))
   elif action=='detect':launch_job(pid,'detect',lambda:detect(pid))
   elif action=='track':launch_job(pid,'track',lambda:track(pid,str(body['track']),int(body['frame'])))
   elif action=='export':launch_job(pid,'export',lambda:export(pid))
   else:raise ValueError('未知操作')
   return self.send_json(JOBS[pid])
  except (ValueError,KeyError,StopIteration) as e:self.send_json({'error':str(e)},400)
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:self.send_json({'error':str(e)},500)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--port',type=int,default=8765);args=a.parse_args();server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
 print(f'LocalMask http://127.0.0.1:{args.port}',flush=True);server.serve_forever()
