'use strict';
// 浏览器端预览渲染：与 core.render_masks 保持同样的效果语义（导出以服务端为准）。
const FX=(()=>{
 const HOLD=6;
 const noiseCache={};
 function noise(seed){
  if(noiseCache[seed])return noiseCache[seed];
  let h=2166136261;for(const c of seed)h=Math.imul(h^c.charCodeAt(0),16777619);
  const rnd=()=>{h^=h<<13;h^=h>>>17;h^=h<<5;return((h>>>0)%10000)/10000;};
  const c=document.createElement('canvas');c.width=c.height=128;const x=c.getContext('2d');
  for(const [n,a] of [[8,.5],[16,.3],[32,.2]]){const s=document.createElement('canvas');s.width=s.height=n;const sx=s.getContext('2d');const d=sx.createImageData(n,n);for(let i=0;i<n*n;i++){const v=rnd()*255;d.data.set([v,v,v,255],i*4);}sx.putImageData(d,0,0);x.globalAlpha=a;x.imageSmoothingEnabled=true;x.drawImage(s,0,0,128,128);}
  return noiseCache[seed]=c;
 }
 function polysAt(frames,f){if(!frames)return null;for(let d=0;d<=HOLD;d++){const v=frames[f-d];if(v&&v.length)return v;}return null;}
 function shapePath(m){
  const p=new Path2D();const [x,y,w,h]=m.box;
  if(m.shape==='contour'&&m.polys){for(const poly of m.polys){p.moveTo(poly[0],poly[1]);for(let i=2;i<poly.length;i+=2)p.lineTo(poly[i],poly[i+1]);p.closePath();}}
  else if(m.shape==='ellipse')p.ellipse(x+w/2,y+h/2,w/2,h/2,0,0,Math.PI*2);
  else p.rect(x,y,w,h);
  return p;
 }
 const fxc=document.createElement('canvas'),fx=fxc.getContext('2d'),mc=document.createElement('canvas'),mk=mc.getContext('2d'),tiny=document.createElement('canvas'),tc=tiny.getContext('2d');
 // ctx: 目标画布；src: 原始画面（canvas/图片），坐标与 ctx 相同；m: 遮挡描述
 function draw(ctx,src,m,stickers){
  let [bx,by,bw,bh]=m.box;if(bw<=0||bh<=0)return;
  const W=ctx.canvas.width,H=ctx.canvas.height;
  const margin=m.effect==='cloud'?Math.round(Math.max(bw,bh)*.22)+2:(m.effect==='blur'?3:0);
  const x0=Math.max(0,Math.floor(bx-margin)),y0=Math.max(0,Math.floor(by-margin)),x1=Math.min(W,Math.ceil(bx+bw+margin)),y1=Math.min(H,Math.ceil(by+bh+margin));
  const w=x1-x0,h=y1-y0;if(w<=0||h<=0)return;
  fxc.width=w;fxc.height=h;fx.save();fx.clearRect(0,0,w,h);
  const block=Math.max(2,m.block||20),s=m.strength!=null?Math.max(0,Math.min(100,m.strength))/100:Math.max(0,Math.min(1,(block-6)/54)),side=Math.max(4,Math.min(bw,bh));
  if(m.effect==='mosaic'){const cell=Math.max(3,Math.round(side/(12-9*s)));const sx=Math.ceil(w/cell),sy=Math.ceil(h/cell);tiny.width=sx;tiny.height=sy;tc.imageSmoothingEnabled=true;tc.imageSmoothingQuality='high';tc.drawImage(src,x0,y0,Math.min(w,sx*cell),Math.min(h,sy*cell),0,0,sx,sy);fx.imageSmoothingEnabled=false;fx.drawImage(tiny,0,0,sx,sy,0,0,sx*cell,sy*cell);}
  else if(m.effect==='blur'){const ext=Math.round(side*.6),ex0=Math.max(0,x0-ext),ey0=Math.max(0,y0-ext),ex1=Math.min(W,x1+ext),ey1=Math.min(H,y1+ext),cell=Math.max(3,Math.round(side/(9-5*s)));tiny.width=Math.max(1,Math.floor((ex1-ex0)/cell));tiny.height=Math.max(1,Math.floor((ey1-ey0)/cell));tc.imageSmoothingEnabled=true;tc.imageSmoothingQuality='high';tc.drawImage(src,ex0,ey0,ex1-ex0,ey1-ey0,0,0,tiny.width,tiny.height);fx.imageSmoothingEnabled=true;fx.filter=`blur(${Math.max(3,side*(.10+.22*s))}px)`;fx.drawImage(tiny,0,0,tiny.width,tiny.height,ex0-x0,ey0-y0,ex1-ex0,ey1-ey0);fx.filter='none';}
  else if(m.effect==='solid'||m.effect==='cover'){fx.fillStyle=`rgb(${(m.color||[28,28,30]).join(',')})`;fx.fillRect(0,0,w,h);}
  else if(m.effect==='cloud'){fx.fillStyle='#f4f4f2';fx.fillRect(0,0,w,h);fx.globalAlpha=.25;fx.drawImage(noise(m.id||'c'),0,0,w,h);fx.globalAlpha=1;}
  else if(m.effect==='sticker'){const img=stickers&&stickers(m.sticker);if(img&&img.complete&&img.naturalWidth){const s=Math.max(bw/img.naturalWidth,bh/img.naturalHeight),rw=img.naturalWidth*s,rh=img.naturalHeight*s;fx.drawImage(img,bx-x0+(bw-rw)/2,by-y0+(bh-rh)/2,rw,rh);}else{fx.fillStyle='#333';fx.fillRect(0,0,w,h);}}
  fx.restore();
  if(m.effect!=='sticker'){
   // 形状遮罩
   mc.width=w;mc.height=h;mk.save();mk.translate(-x0,-y0);const path=shapePath(m);
   if(m.effect==='cloud'){mk.filter=`blur(${Math.max(2,Math.min(bw,bh)*.12)}px)`;mk.fillStyle='#000';mk.fill(path);mk.lineJoin='round';mk.lineWidth=Math.max(bw,bh)*.18+2*(m.pad||0);mk.stroke(path);mk.filter='none';}
   mk.fillStyle='#000';mk.strokeStyle='#000';mk.fill(path);
   if(m.shape==='contour'&&m.pad>0){mk.lineJoin='round';mk.lineWidth=2*m.pad;mk.stroke(path);}
   mk.restore();
   fx.globalCompositeOperation='destination-in';fx.drawImage(mc,0,0);fx.globalCompositeOperation='source-over';
  }else{
   // 贴纸只在框内
   mc.width=w;mc.height=h;mk.fillStyle='#000';mk.fillRect(bx-x0,by-y0,bw,bh);fx.globalCompositeOperation='destination-in';fx.drawImage(mc,0,0);fx.globalCompositeOperation='source-over';
  }
  ctx.save();ctx.globalAlpha=(m.opacity??100)/100;ctx.drawImage(fxc,x0,y0);ctx.restore();
 }
 return {draw,polysAt,shapePath,HOLD};
})();
