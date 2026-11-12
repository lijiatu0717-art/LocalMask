"""Local face embeddings for conservative within-video grouping; no identity names."""
from pathlib import Path
import cv2,numpy as np
MODEL_DIR=Path(__file__).resolve().parents[1]/'.cache/face-models'

class FaceEmbedder:
 def __init__(self):
  self.detector=cv2.FaceDetectorYN_create(str(MODEL_DIR/'yunet.onnx'),'',(320,320),.7,.3,5000)
  self.recognizer=cv2.FaceRecognizerSF_create(str(MODEL_DIR/'sface.onnx'),'')
 def feature(self,image,box):
  h,w=image.shape[:2];x,y,bw,bh=box
  x0=max(0,int((x-bw*.35)*w));y0=max(0,int((y-bh*.35)*h));x1=min(w,int((x+bw*1.35)*w));y1=min(h,int((y+bh*1.35)*h))
  crop=image[y0:y1,x0:x1]
  if min(crop.shape[:2])<16:return None
  crop=cv2.resize(crop,(320,320));self.detector.setInputSize((320,320));_,faces=self.detector.detect(crop)
  if faces is None or len(faces)!=1:return None
  aligned=self.recognizer.alignCrop(crop,faces[0]);v=self.recognizer.feature(aligned).flatten();return v/max(1e-9,np.linalg.norm(v))

def group_faces(tracks,features,threshold=.60):
 groups=[]
 for t in tracks:
  vectors=features.get(t['id'],[]);candidates=[]
  for g in groups:
   # Never group simultaneous tracks, nor trust a single lucky pose match.
   if any(not(t['end']<a['start'] or t['start']>a['end']) for a in g['_tracks']):continue
   if len(vectors)<2 or len(g['_vectors'])<2:continue
   scores=[max(float(v@r) for r in g['_vectors']) for v in vectors]
   score=min(scores)
   if score>=threshold:candidates.append((score,g))
  candidates.sort(key=lambda a:a[0],reverse=True)
  if candidates and (len(candidates)==1 or candidates[0][0]-candidates[1][0]>=.10):g=candidates[0][1];g['matches'].append(round(candidates[0][0],4))
  else:
   g={'id':'person_'+t['id'],'track_ids':[],'representative':t['id'],'matches':[],'_tracks':[],'_vectors':[]};groups.append(g)
  g['track_ids'].append(t['id']);g['_tracks'].append(t);g['_vectors'].extend(vectors)
 return [{k:v for k,v in g.items() if not k.startswith('_')} for g in groups]

def build_catalog(p,read_frame):
 tracks=[t for t in p['tracks'] if t.get('auto_mode')=='faces'];embedder=FaceEmbedder();features={}
 for t in tracks:
  samples=sorted(set([len(t['keys'])//4,len(t['keys'])//2,3*len(t['keys'])//4]));features[t['id']]=[]
  for index in samples:
   k=t['keys'][index];v=embedder.feature(read_frame(p,k['frame']),k['box'])
   if v is not None:features[t['id']].append(v)
 groups=group_faces(tracks,features)
 for g in groups:
  spans=sorted([t for t in tracks if t['id'] in g['track_ids']],key=lambda t:t['start'])
  g['review_gaps']=[[a['end']+1,b['start']-1] for a,b in zip(spans,spans[1:]) if 0<b['start']-a['end']-1<=p['fps']]
 return {'version':1,'method':'OpenCV SFace + YuNet, local; conservative grouping, verify each appearance','groups':groups,'tracks':len(tracks),'sample_counts':{k:len(v) for k,v in features.items()}}

class FacePool:
 """CPU face detection avoids the macOS Vision/CoreImage GPU wait path."""
 def __init__(self):
  from concurrent.futures import ThreadPoolExecutor
  self.pool=ThreadPoolExecutor(1)
  self.detector=cv2.FaceDetectorYN_create(str(MODEL_DIR/'yunet.onnx'),'',(640,360),.7,.3,5000)
 def job(self,i,image):
  h,w=image.shape[:2];scale=min(1.,960/max(h,w));small=cv2.resize(image,(round(w*scale),round(h*scale)))
  sh,sw=small.shape[:2];self.detector.setInputSize((sw,sh));_,faces=self.detector.detect(small);hits=[]
  if faces is not None:
   for face in faces:
    x=max(0,float(face[0])/sw);y=max(0,float(face[1])/sh);bw=min(1-x,float(face[2])/sw);bh=min(1-y,float(face[3])/sh)
    if bw>0 and bh>0:hits.append({'box':[x,y,bw,bh],'confidence':float(face[-1]),'kind':'face','review_required':True})
  return {'frame':i,'hits':hits,'recognized_text':[]}
 def close(self):self.pool.shutdown()
