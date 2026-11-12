"""Convert frame detections into persistent, fixed-style region tracks.
No detector call. Inputs are existing evidence; no claim of new target recall.
"""
import numpy as np

def union(a,b):
 x=min(a[0],b[0]);y=min(a[1],b[1]);r=max(a[0]+a[2],b[0]+b[2]);d=max(a[1]+a[3],b[1]+b[3]);return [x,y,r-x,d-y]
def overlap(a,b):
 x=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));y=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]));return x*y/max(1,min(a[2]*a[3],b[2]*b[3]))
def merge_boxes(boxes):
 result=[]
 for box in boxes:
  b=list(box);changed=True
  while changed:
   changed=False
   for i,a in enumerate(result):
    if overlap(a,b)>.12:b=union(a,b);result.pop(i);changed=True;break
  result.append(b)
 return result

def stable_tracks(timeline,width,height,max_gap=12):
 tracks=[];active=[]
 for f in range(len(timeline)):
  boxes=merge_boxes(timeline[f]);active=[t for t in active if f-t['last']<=max_gap+1];pairs=[]
  for ti,t in enumerate(active):
   for bi,b in enumerate(boxes):
    score=overlap(t['samples'][-1][1],b)
    if score>.2:pairs.append((score,ti,bi))
  used_t=set();used_b=set()
  for score,ti,bi in sorted(pairs,reverse=True):
   if ti in used_t or bi in used_b:continue
   t=active[ti];t['samples'].append((f,boxes[bi]));t['last']=f;used_t.add(ti);used_b.add(bi)
  for bi,b in enumerate(boxes):
   if bi not in used_b:
    t={'samples':[(f,b)],'last':f};tracks.append(t);active.append(t)
 result=[]
 for num,t in enumerate(tracks):
  obs=t['samples'];start,end=obs[0][0],obs[-1][0];frames=np.arange(start,end+1);raw=np.array([np.interp(frames,[s[0] for s in obs],[s[1][k] for s in obs]) for k in range(4)]).T
  edges=np.column_stack((raw[:,0],raw[:,1],raw[:,0]+raw[:,2],raw[:,1]+raw[:,3]))
  # Lock static geometry, rather than following every noisy detection.
  stable=np.empty_like(edges);anchor=0
  while anchor<len(edges):
   stop=anchor+1
   while stop<len(edges) and np.max(np.ptp(edges[anchor:stop+1],axis=0))<=10:stop+=1
   if stop-anchor>=6:
    stable[anchor:stop,:2]=edges[anchor:stop,:2].min(axis=0);stable[anchor:stop,2:]=edges[anchor:stop,2:].max(axis=0)
   else:stable[anchor:stop]=edges[anchor:stop]
   anchor=stop
  # Conservative 5-frame envelope in moving spans: no averaging inward.
  for i in range(len(edges)):
   window=stable[max(0,i-2):min(len(edges),i+3)];l,top=window[:,:2].min(axis=0);r,bottom=window[:,2:].max(axis=0)
   raw[i]=[max(0,l-2),max(0,top-2),min(width,r+2)-max(0,l-2),min(height,bottom+2)-max(0,top-2)]
  keys=[{'frame':int(f),'box':[float(b[0]/width),float(b[1]/height),float(b[2]/width),float(b[3]/height)]} for f,b in zip(frames,raw)]
  result.append({'id':f'stable_{num:04d}','name':f'连续遮挡 {num+1:03d}','start':int(start),'end':int(end),'enabled':True,'reviewed':False,'effect':'cover','color':[210,210,210],'feather':3,'padding':0,'block':22,'motion':'linear','keys':keys,'origin':'existing masks merged, static geometry locked; gaps bridged <=12 frames'})
 return result
