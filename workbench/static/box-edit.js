'use strict';
// Geometry shared by browser gestures and focused unit tests.
const BoxEdit={
 clamp(b,minW=.002,minH=.002){let w=Math.max(minW,Math.min(1,b[2])),h=Math.max(minH,Math.min(1,b[3]));return[Math.max(0,Math.min(1-w,b[0])),Math.max(0,Math.min(1-h,b[1])),w,h];},
 move(b,dx,dy){return this.clamp([b[0]+dx,b[1]+dy,b[2],b[3]]);},
 resize(b,corner,dx,dy,minW,minH){let [x,y,w,h]=b,r=x+w,d=y+h;if(corner.includes('w'))x=Math.max(0,Math.min(r-minW,x+dx));else r=Math.min(1,Math.max(x+minW,r+dx));if(corner.includes('n'))y=Math.max(0,Math.min(d-minH,y+dy));else d=Math.min(1,Math.max(y+minH,d+dy));return[x,y,r-x,d-y];},
 applyFuture(track,frame,before,after,previous){
  const sx=after[2]/before[2],sy=after[3]/before[3];
  const transform=b=>this.clamp([after[0]+(b[0]-before[0])*sx,after[1]+(b[1]-before[1])*sy,b[2]*sx,b[3]*sy]);
  let keys=track.keys.filter(k=>k.frame<frame).map(k=>({frame:k.frame,box:[...k.box]}));
  if(frame>track.start&&previous){keys=keys.filter(k=>k.frame!==frame-1);keys.push({frame:frame-1,box:previous});}
  keys.push({frame,box:[...after]});
  keys.push(...track.keys.filter(k=>k.frame>frame).map(k=>({frame:k.frame,box:transform(k.box)})));
  track.keys=keys.sort((a,b)=>a.frame-b.frame);track.reviewed=false;
 }
};
if(typeof module!=='undefined')module.exports=BoxEdit;
