"""Identity-checked face follow with bounded optical fallback and explicit uncertainty."""
import json,hashlib
import cv2,numpy as np
from identity import FacePool,FaceEmbedder
from dynamic_mosaic import flow_box,cut_candidate,iou
from core import box_at,contiguous_ranges

def match_identity(candidates,reference,feature,threshold=.55,margin=.10):
 ranked=[]
 for b in candidates:
  f=feature(b)
  if f is not None:ranked.append((float(reference@f),b))
 ranked.sort(key=lambda a:a[0],reverse=True)
 if not ranked or ranked[0][0]<threshold:return None
 if len(ranked)>1 and ranked[0][0]-ranked[1][0]<margin:return None
 return ranked[0][1]

def cover_face(face,seed,initial):
 x,y,w,h=face;ix,iy,iw,ih=initial
 # Preserve the user's larger cover, but never trim away the detected face.
 l=min(x-.16*w,x+(seed[0]-ix)/iw*w);top=min(y-.20*h,y+(seed[1]-iy)/ih*h)
 r=max(x+1.16*w,x+(seed[0]+seed[2]-ix)/iw*w);bottom=max(y+1.18*h,y+(seed[1]+seed[3]-iy)/ih*h)
 l=max(0,l);top=max(0,top);r=min(1,r);bottom=min(1,bottom)
 return [l,top,r-l,bottom-top]

def follow(p,t,anchor,folder,read_anchor,frame_iterator,progress,check_cancel):
 seed=box_at(t,anchor);detector=FacePool();embedder=FaceEmbedder();initial_image=read_anchor(anchor)
 digest=hashlib.sha256(json.dumps(['faces',[],'yunet-cpu-v1'],ensure_ascii=False).encode()).hexdigest()[:16]
 cache=folder/('ocr_'+digest+'.jsonl');known={}
 if cache.exists():
  for line in cache.read_text().splitlines():
   try:r=json.loads(line);known[r['frame']]=r
   except (json.JSONDecodeError,KeyError):pass
 log=cache.open('a');confirmed=0;reacquired=0;lost=[];was_lost=False;records=[]
 def detections(i,image):
  if i not in known:
   r=detector.job(i,image);known[i]=r;log.write(json.dumps(r)+'\n');log.flush()
  return [h['box'] for h in known[i]['hits']]
 try:
  candidates=detections(anchor,initial_image)
  inside=[b for b in candidates if seed[0]<=b[0]+b[2]/2<=seed[0]+seed[2] and seed[1]<=b[1]+b[3]/2<=seed[1]+seed[3]]
  if len(inside)!=1:raise ValueError('请框住一张清晰的人脸再点跟随；当前框未找到唯一的人脸')
  initial=inside[0];reference=embedder.feature(initial_image,initial)
  if reference is None:raise ValueError('这帧人脸不够清晰，请暂停到正脸更清楚的位置再跟随')
  last_face=initial;last_gray=cv2.cvtColor(cv2.resize(initial_image,(640,360)),cv2.COLOR_BGR2GRAY);hold=cover_face(initial,seed,initial);records.append({'frame':anchor,'box':hold});missing=0
  for i,image in frame_iterator(anchor+1,t['end']):
   check_cancel();gray=cv2.cvtColor(cv2.resize(image,(640,360)),cv2.COLOR_BGR2GRAY);cut=cut_candidate(last_gray,gray)
   found=match_identity(detections(i,image),reference,lambda b:embedder.feature(image,b))
   if found is not None:
    if was_lost:reacquired+=1
    was_lost=False;missing=0;last_face=found;candidate=cover_face(found,seed,initial);confirmed+=1
    # Small detector jitter is covered by a bounded union, never averaged inward.
    if not cut and max(abs(candidate[k]-hold[k]) for k in range(4))<.008:
     l=min(candidate[0],hold[0]);top=min(candidate[1],hold[1]);r=max(candidate[0]+candidate[2],hold[0]+hold[2]);bottom=max(candidate[1]+candidate[3],hold[1]+hold[3]);candidate=[l,top,r-l,bottom-top]
    hold=candidate
   else:
    missing+=1;lost.append(i);was_lost=True
    # Optical propagation is only a short conservative bridge; always flagged.
    moved=flow_box(last_gray,gray,last_face) if not cut and missing<=max(1,round(p['fps']*.25)) else None
    if moved is not None:last_face=moved;hold=cover_face(moved,seed,initial)
   records.append({'frame':i,'box':hold[:]});last_gray=gray
   if i%15==0:progress((i-anchor)/max(1,t['end']-anchor)*95,f'人脸跟随 {i-anchor}/{t["end"]-anchor} 帧；待检查 {len(lost)} 帧')
  return records,contiguous_ranges(lost),{'confirmed_frames':confirmed+1,'unverified_frames':len(lost),'reacquisitions':reacquired,'method':'YuNet + fixed SFace reference; short optical bridge; uncertain intervals flagged'}
 finally:log.close();detector.close()
