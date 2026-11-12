"""Conservative face segments, not cross-shot biometric identity recognition."""
from stabilize import stable_tracks

def score(a,b):
 x=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));y=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]));i=x*y
 return i/max(1,a[2]*a[3]+b[2]*b[3]-i)

def face_tracks(timeline,width,height,cuts=()):
 segments=[];active=[]
 for f,boxes in enumerate(timeline):
  if f in cuts:active=[]
  ambiguous={j for j,b in enumerate(boxes) if any(k!=j and score(b,c)>.25 for k,c in enumerate(boxes))}
  pairs=[(i,j) for i,t in enumerate(active) for j,b in enumerate(boxes) if j not in ambiguous and score(t[-1][1],b)>.25]
  used=set();next_active=[]
  for i,t in enumerate(active):
   matches=[j for a,j in pairs if a==i]
   if len(matches)==1 and sum(j==matches[0] for _,j in pairs)==1:
    j=matches[0];t.append((f,boxes[j]));used.add(j);next_active.append(t)
  for j,b in enumerate(boxes):
   if j not in used:
    t=[(f,b)];segments.append(t);next_active.append(t)
  active=next_active
 out=[]
 for segment in segments:
  start=segment[0][0];local=[[b] for _,b in segment]
  t=stable_tracks(local,width,height,max_gap=0)[0]
  t['start']+=start;t['end']+=start
  for k in t['keys']:k['frame']+=start
  t.update(enabled=False,face_segment=True,identity_scope='continuous unambiguous segment only; reselect after gaps/cuts',name=f'人脸片段 {len(out)+1}')
  out.append(t)
 return out
