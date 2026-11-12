'use strict';
const $=id=>document.getElementById(id);
const video=$('video'),canvas=$('canvas'),ctx=canvas.getContext('2d');
const base=document.createElement('canvas'),bctx=base.getContext('2d');
const clone=x=>JSON.parse(JSON.stringify(x));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const EMOJI=['🙂','😎','🙈','🐱','🐶','⭐','❤️','🌸','🔥','👻'];
const COLORS=[[20,20,22],[245,245,245],[242,178,58],[255,111,145],[90,169,255]];
const JOB_TEXT={'prepare-faces':'正在找画面里的人',detect:'正在逐帧找文字',track:'正在跟随',export:'正在导出','ai-track':'正在跟随','ai-scan':'正在整段查找'};

let token='',projects=[],p=null,layers={},ai={ready:false},current=0,masked=true,tool='select',selKey=null,refine=false,still=null;
let dirty=false,busy=false,undoStack=[],saveTimer=null,saving=Promise.resolve(),seekSeq=0,loadSeq=0,poll=null,drag=null,gesture=null;
const wide=()=>matchMedia('(min-width:900px), (orientation:landscape) and (max-height:600px) and (min-width:600px)').matches;
let refineSub=false;
let quality='fast',masks={},maskLoading={},stickerImgs={},uploadedStickers=[],renderQueued=false,tilesTimer=null,lastJob=null;

let toastTimer;
function toast(msg,err=false,ms=3800){const t=$('toast');t.textContent=msg;t.className='toast'+(err?' err':'');t.hidden=!msg;clearTimeout(toastTimer);if(msg)toastTimer=setTimeout(()=>t.hidden=true,ms);}
async function api(url,body,timeout=30000){const ctl=new AbortController(),tm=setTimeout(()=>ctl.abort(),timeout);let r;try{r=await fetch(url,body===undefined?{signal:ctl.signal}:{signal:ctl.signal,method:'POST',headers:{'Content-Type':'application/json','X-Workbench-Token':token},body:JSON.stringify(body)});}catch(e){throw Error(e.name==='AbortError'?'等待太久没有响应，请再试一次':'连接断了，请刷新页面');}finally{clearTimeout(tm);}const d=await r.json().catch(()=>({error:r.statusText}));if(!r.ok)throw Error(d.error||r.statusText);return d;}
const path=a=>`/api/project/${p.id}${a?'/'+a:''}`;
function scheduleRender(){if(renderQueued)return;renderQueued=true;requestAnimationFrame(()=>{renderQueued=false;render();});}

/* ---------- 物体 ---------- */
function single(t){return{key:'t:'+t.id,name:t.name,kind:t.mask?'ai':'box',tracks:[t],rep:t.id};}
function entities(){
 if(!p)return[];const out=[],used=new Set();
 (p.face_catalog?.groups||[]).forEach((g,i)=>{const ts=p.tracks.filter(t=>g.track_ids.includes(t.id));if(!ts.length)return;ts.forEach(t=>used.add(t.id));out.push({key:'face:'+g.id,name:`人物 ${i+1}`,kind:'person',tracks:ts,rep:g.representative,faceGaps:g.review_gaps||[]});});
 const groups={},order=[];
 for(const t of p.tracks){if(used.has(t.id))continue;let g=t.group;if(!g){const m=t.name.match(/^(.*?)[\s_-]*\d+$/);if(m&&m[1].trim())g='auto:'+m[1].trim();}
  if(g){if(!groups[g]){groups[g]=[];order.push({g});}groups[g].push(t);}else order.push({t});}
 for(const o of order){if(o.t){out.push(single(o.t));continue;}const ts=groups[o.g];
  if(ts.length===1||(o.g.startsWith('auto:')&&ts.length<3))ts.forEach(t=>out.push(single(t)));
  else out.push({key:'grp:'+o.g,name:o.g.startsWith('auto:')?o.g.slice(5):o.g.split('·')[0],kind:'group',tracks:ts,rep:ts[Math.floor(ts.length/2)].id});}
 return out;
}
const entOf=t=>entities().find(e=>e.tracks.includes(t));
function selEntity(){return selKey?entities().find(e=>e.key===selKey):null;}
function selTracks(){return selEntity()?.tracks||[];}
function entGaps(e){const g=[];for(const t of e.tracks)if(t.enabled)for(const x of t.tracking_gaps||[])g.push(x);for(const x of e.faceGaps||[])g.push(x);return g;}

/* ---------- 几何 ---------- */
function boxAt(t,f){if(!t.enabled||f<t.start||f>t.end)return null;const k=t.keys;if(!k.length)return null;if(f<=k[0].frame)return k[0].box;if(f>=k.at(-1).frame)return k.at(-1).box;let lo=0,hi=k.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(k[m].frame<=f)lo=m;else hi=m;}const a=k[lo],b=k[hi],r=(f-a.frame)/(b.frame-a.frame);if(t.motion==='envelope'){const x=Math.min(a.box[0],b.box[0]),y=Math.min(a.box[1],b.box[1]);return[x,y,Math.max(a.box[0]+a.box[2],b.box[0]+b.box[2])-x,Math.max(a.box[1]+a.box[3],b.box[1]+b.box[3])-y];}return a.box.map((v,i)=>v+(b.box[i]-v)*r);}
function pixelRect(b,pad){const[x,y,w,h]=b,l=Math.max(0,Math.floor(x*p.width)-pad),t=Math.max(0,Math.floor(y*p.height)-pad),r=Math.min(p.width,Math.ceil((x+w)*p.width)+pad),bo=Math.min(p.height,Math.ceil((y+h)*p.height)+pad);return[l,t,r-l,bo-t];}
async function loadMask(id){if(maskLoading[id]||!p)return;maskLoading[id]=true;const pid=p.id;try{const d=await api(path('mask')+'?t='+encodeURIComponent(id),undefined,60000);if(p?.id===pid)masks[id]=d.frames;}catch(e){}finally{maskLoading[id]=false;}scheduleRender();}
function trackMask(t,f){
 if(!t.enabled||f<t.start||f>t.end)return null;let box,polys=null;
 if(t.mask){const fr=masks[t.id];if(!fr){loadMask(t.id);return null;}polys=FX.polysAt(fr,f);if(!polys)return null;let x0=1e9,y0=1e9,x1=-1,y1=-1;for(const q of polys)for(let i=0;i<q.length;i+=2){x0=Math.min(x0,q[i]);x1=Math.max(x1,q[i]);y0=Math.min(y0,q[i+1]);y1=Math.max(y1,q[i+1]);}const pd=t.padding;const l=Math.max(0,x0-pd),tp=Math.max(0,y0-pd);box=[l,tp,Math.min(p.width,x1+1+pd)-l,Math.min(p.height,y1+1+pd)-tp];}
 else{const b=(gesture?.id===t.id&&f===current)?gesture.preview:boxAt(t,f);if(!b)return null;box=pixelRect(b,t.padding);}
 const shape=t.shape||'rect';
 return{id:t.id,box,effect:t.effect,block:t.block,strength:t.strength,opacity:t.opacity??100,color:t.color,shape:shape==='contour'&&!polys?'rect':shape,polys:shape==='contour'?polys:null,pad:t.padding,sticker:t.sticker,track:t};
}
function frameMasks(f){const ms=[];for(const l of p.layers||[])if(l.enabled)for(const b of layers[l.id]?.[f]||[])ms.push({box:b,effect:'mosaic',block:22,shape:'rect',opacity:100,id:l.id});for(const t of p.tracks){const m=trackMask(t,f);if(m)ms.push(m);}return ms;}
function stickerImg(name){if(!name)return null;if(!stickerImgs[name]){const im=new Image();im.onload=()=>{scheduleRender();queueTiles();};im.src=path('sticker')+'?n='+encodeURIComponent(name);stickerImgs[name]=im;}return stickerImgs[name];}

/* ---------- 画面 ---------- */
function render(image=still||video){
 if(!p||!image||(image===video&&video.readyState<2))return;
 bctx.drawImage(image,0,0,base.width,base.height);ctx.drawImage(base,0,0);
 const ms=frameMasks(current);
 if(masked)for(const m of ms)FX.draw(ctx,base,m,stickerImg);
 overlays(ms);updateClock();
}
function overlays(ms){
 if(!masked)return;const sel=new Set(selTracks().map(t=>t.id)),scale=p.width/(canvas.getBoundingClientRect().width||p.width);
 ctx.save();ctx.lineJoin='round';
 for(const m of ms){if(!m.track||!sel.has(m.id))continue;ctx.lineWidth=2.5*scale;ctx.strokeStyle='#fff';ctx.setLineDash(m.track.mask?[]:[8*scale,5*scale]);ctx.stroke(FX.shapePath(m));
  if(!m.track.mask&&video.paused){const[x,y,w,h]=m.box,r=6*scale;ctx.setLineDash([]);ctx.fillStyle='#fff';for(const[hx,hy]of[[x,y],[x+w,y],[x,y+h],[x+w,y+h]]){ctx.beginPath();ctx.arc(hx,hy,r,0,7);ctx.fill();}}}
 const t=refine&&selTracks().length===1?selTracks()[0]:null;
 for(const pt of t?.prompts?.[current]?.points||[]){const x=pt[0]*p.width,y=pt[1]*p.height,r=10*scale;ctx.setLineDash([]);ctx.fillStyle=pt[2]?'#fff':'#ff6b6b';ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();ctx.strokeStyle=pt[2]?'#111':'#fff';ctx.lineWidth=2*scale;ctx.beginPath();ctx.moveTo(x-r*.5,y);ctx.lineTo(x+r*.5,y);if(pt[2]){ctx.moveTo(x,y-r*.5);ctx.lineTo(x,y+r*.5);}ctx.stroke();}
 if(drag){ctx.setLineDash([]);ctx.fillStyle='rgba(255,255,255,.15)';ctx.strokeStyle='#fff';ctx.lineWidth=2*scale;const r=[drag.x*p.width,drag.y*p.height,(drag.ex-drag.x)*p.width,(drag.ey-drag.y)*p.height];ctx.fillRect(...r);ctx.strokeRect(...r);}
 ctx.restore();
}
function updateClock(){if(!p)return;$('clock').textContent=fmt(current/p.fps);$('dur').textContent=fmt(p.duration);$('playIcon').setAttribute('d',video.paused?'M6 4l10 6-10 6z':'M5 4h3.5v12H5zM11.5 4H15v12h-3.5z');const x=(current/Math.max(1,p.frames-1)*100)+'%';$('scrubFill').style.width=x;$('scrubKnob').style.left=x;}

/* ---------- 播放 ---------- */
function videoTick(now,meta){if(p&&!video.paused){still=null;current=Math.min(p.frames-1,Math.floor(meta.mediaTime*p.fps+.5));render(video);}video.requestVideoFrameCallback(videoTick);}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(videoTick);else(function f(){if(p&&!video.paused){still=null;current=Math.floor(video.currentTime*p.fps+.5);render(video);}requestAnimationFrame(f);})();
video.addEventListener('ended',()=>{if(p){current=p.frames-1;updateClock();}});
video.addEventListener('loadeddata',()=>{if(!still)render(video);});video.addEventListener('pause',()=>{if(p){updateClock();seek(current);}});
async function seek(f){if(!p)return;if(!video.paused)video.pause();current=Math.max(0,Math.min(p.frames-1,Math.round(f)));const id=++seekSeq;video.currentTime=current/p.fps;updateClock();try{const r=await fetch(path('frame')+'?i='+current);if(!r.ok)return;const bmp=await createImageBitmap(await r.blob());if(id!==seekSeq){bmp.close();return;}if(still)still.close();still=bmp;render();queueTiles();}catch(e){}}
function togglePlay(){if(!p)return;if(video.paused){if(current>=p.frames-2)current=0;seekSeq++;if(still){still.close();still=null;}video.currentTime=current/p.fps;video.play().catch(e=>toast(e.message,true));}else video.pause();updateClock();}
$('play').onclick=togglePlay;
function scrubAt(e){const r=$('scrub').getBoundingClientRect();return Math.round(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*(p.frames-1));}
$('scrub').onpointerdown=e=>{if(!p)return;seek(scrubAt(e));const mv=ev=>seek(scrubAt(ev));const up=()=>{removeEventListener('pointermove',mv);removeEventListener('pointerup',up);};addEventListener('pointermove',mv);addEventListener('pointerup',up);};
function renderScrub(){if(!p)return;const F=Math.max(1,p.frames-1),ts=selTracks();
 if(ts.length){const s=Math.min(...ts.map(t=>t.start)),e=Math.max(...ts.map(t=>t.end));$('scrubRange').style.left=s/F*100+'%';$('scrubRange').style.width=Math.max(.5,(e-s)/F*100)+'%';$('scrubRange').hidden=false;}else $('scrubRange').hidden=true;
 const gaps=[];for(const en of entities())for(const g of entGaps(en))gaps.push(g);
 $('scrubMarks').innerHTML=gaps.map(g=>`<span class="mark" title="这里可能漏了" style="left:${g[0]/F*100}%;width:${Math.max(.3,(g[1]-g[0])/F*100)}%"></span>`).join('');}
const cmp=$('compare');cmp.onpointerdown=()=>{masked=false;scheduleRender();};for(const ev of['pointerup','pointerleave','pointercancel'])cmp.addEventListener(ev,()=>{if(!masked){masked=true;scheduleRender();}});

/* ---------- 首页 / 工程 ---------- */
function showHome(){video.pause();$('home').hidden=false;$('editor').hidden=true;
 $('recentWrap').hidden=!projects.length;
 $('recent').innerHTML=projects.map(x=>`<div class="card" data-id="${esc(x.id)}"><button class="open"><span class="cover" style="background-image:url('/api/project/${encodeURIComponent(x.id)}/frame?i=${Math.min(30,Math.max(0,x.frames-1))}')"><em>${fmt(x.duration)}</em></span><b>${esc(x.name.replace(/\?.*$/,'').replace(/\.(mp4|mov|m4v|mkv|webm)$/i,''))}</b></button><button class="dots" title="更多" aria-label="更多">⋯</button></div>`).join('');
 $('recent').querySelectorAll('.card').forEach(c=>{c.querySelector('.open').onclick=()=>loadProject(c.dataset.id);c.querySelector('.dots').onclick=e=>{e.stopPropagation();openCardMenu(c.dataset.id,e.currentTarget);};});}
let menuFor=null;
function openCardMenu(id,btn){menuFor=id;const m=$('cardMenu'),r=btn.getBoundingClientRect();m.hidden=false;m.style.left=Math.min(innerWidth-150,r.right-140)+'px';m.style.top=(r.bottom+6+scrollY)+'px';}
document.addEventListener('click',e=>{if(!e.target.closest('#cardMenu,.dots'))$('cardMenu').hidden=true;});
function ask(title,text,{input=null,yes='确定',danger=false}={}){return new Promise(res=>{$('confirmTitle').textContent=title;$('confirmText').textContent=text;$('confirmText').hidden=!text;const inp=$('confirmInput');inp.hidden=input===null;if(input!==null)inp.value=input;$('confirmYes').textContent=yes;$('confirmYes').classList.toggle('danger-fill',danger);const d=$('confirmDialog');d.showModal();if(input!==null)setTimeout(()=>{inp.focus();inp.select();},0);const done=v=>{d.close();$('confirmYes').onclick=$('confirmNo').onclick=null;res(v);};$('confirmYes').onclick=()=>done(input!==null?inp.value.trim():true);$('confirmNo').onclick=()=>done(null);inp.onkeydown=e=>{if(e.key==='Enter')done(inp.value.trim());};});}
$('menuDelete').onclick=async()=>{$('cardMenu').hidden=true;const x=projects.find(q=>q.id===menuFor);if(!x)return;const ok=await ask('删除这个草稿？','会放进「马赛克AI工具/workbench/data/.回收站」，需要时还能找回。',{yes:'删除',danger:true});if(!ok)return;try{await api('/api/project/'+x.id+'/delete',{});projects=projects.filter(q=>q.id!==x.id);showHome();toast('已删除');}catch(e){toast(e.message,true);}};
$('menuRename').onclick=async()=>{$('cardMenu').hidden=true;const x=projects.find(q=>q.id===menuFor);if(!x)return;const name=await ask('改名','',{input:x.name.replace(/\?.*$/,'').replace(/\.(mp4|mov|m4v|mkv|webm)$/i,''),yes:'保存'});if(!name)return;try{const np=await api('/api/project/'+x.id+'/rename',{name});Object.assign(x,{name:np.name});showHome();}catch(e){toast(e.message,true);}};
async function goHome(){await flushSave();try{projects=(await api('/api/bootstrap')).projects.sort((a,b)=>b.created-a.created);}catch(e){}showHome();}
$('back').onclick=goHome;
async function loadProject(id,keep=false){
 try{
  if(!keep)await flushSave();
  const gen=++loadSeq;const d=await api('/api/project/'+id);if(gen!==loadSeq)return;
  const switching=!p||p.id!==id;p=d.project;layers=d.layers;masks={};if(switching){undoStack=[];selKey=null;refine=false;stickerImgs={};uploadedStickers=[];tool='select';}
  if(selKey&&!selEntity())selKey=null;dirty=false;
  $('home').hidden=true;$('editor').hidden=false;
  canvas.width=base.width=p.width;canvas.height=base.height=p.height;
  if(switching){video.src=path('source');current=0;}
  const low=p.tracks.filter(t=>(t.opacity??100)<100);if(low.length&&!(d.job?.state==='running')&&change())low.forEach(t=>t.opacity=100);
  await seek(current);refreshAll();
  const job=d.job;if(job?.state==='running')watchJob();
 }catch(e){toast(e.message,true);}
}
function refreshAll(){renderChips();renderPanes();renderScrub();scheduleRender();queueTiles();}

/* ---------- 保存 / 撤销 ---------- */
function change(){if(!p||busy)return false;undoStack.push({local:clone({tracks:p.tracks,default_style:p.default_style||null})});if(undoStack.length>60)undoStack.shift();dirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(flushSave,600);return true;}
function flushSave(){clearTimeout(saveTimer);saving=saving.catch(()=>{}).then(async()=>{
 if(!p||!dirty||busy)return;const pid=p.id;dirty=false;
 try{const saved=await api(path(),{revision:p.revision,tracks:p.tracks,terms:p.terms||'',target_mode:p.target_mode||'faces',mask_style:p.mask_style||'cover',mask_block:p.mask_block||32,default_style:p.default_style||undefined,layer_enabled:Object.fromEntries((p.layers||[]).map(l=>[l.id,l.enabled]))});if(p.id===pid){p.revision=saved.revision;p.exports=saved.exports;}}
 catch(e){dirty=true;toast('保存失败：'+e.message,true);if(/重新载入/.test(e.message))loadProject(pid,true);}
});return saving;}
async function undo(){if(busy)return;const u=undoStack.pop();if(!u)return toast('没有可以撤销的了');
 if(u.local){Object.assign(p,clone(u.local));if(!p.default_style)delete p.default_style;dirty=true;refreshAll();clearTimeout(saveTimer);saveTimer=setTimeout(flushSave,400);}
 else{try{await flushSave();applyProject(await api(path('revert'),{revision:u.server}),true);}catch(e){toast(e.message,true);}}}
$('undo').onclick=undo;
function applyProject(np,clearMasks=false,changed=[]){p=np;if(clearMasks)masks={};for(const id of changed)delete masks[id];dirty=false;refreshAll();}

/* ---------- 物体条 ---------- */
function renderChips(){
 const es=entities(),people=es.filter(e=>e.kind==='person');
 const allOn=people.length&&people.every(e=>e.tracks.every(t=>t.enabled));
 const head=people.length>1?`<button class="chip all ${allOn?'done':''}" id="allFaces"><i>${allOn?'✓':'全'}</i><em>${allOn?'已全遮':'全部遮住'}</em></button>`:'';
 $('chips').innerHTML=head+es.map(e=>{const on=e.tracks.some(t=>t.enabled),gaps=entGaps(e).length;return`<button class="chip ${selKey===e.key?'on':''} ${on?'masked':'off'}" data-key="${esc(e.key)}" title="${esc(e.name)}"><i style="background-image:url('${path('thumb')}?id=${encodeURIComponent(e.rep)}&v=${e.tracks[0].keys.length}-${e.tracks[0].start}')"><span class="del" data-del="${esc(e.key)}" role="button" title="删除">×</span><span class="ok ${on?'':'no'}" data-toggle="${esc(e.key)}" role="button" title="${on?'点一下取消遮挡':'点一下遮住'}">${on?'✓':''}</span>${gaps?'<span class="dot"></span>':''}</i><em>${esc(e.name)}</em></button>`;}).join('');
 $('chips').querySelectorAll('.chip[data-key]').forEach(c=>{let timer=null,long=false;
  c.onpointerdown=ev=>{if(ev.button>0)return;long=false;clearTimeout(timer);timer=setTimeout(()=>{long=true;deleteEntity(c.dataset.key);},550);};
  for(const ev of['pointerup','pointerleave','pointercancel'])c.addEventListener(ev,()=>clearTimeout(timer));
  c.oncontextmenu=ev=>{ev.preventDefault();clearTimeout(timer);deleteEntity(c.dataset.key);};
  c.onclick=ev=>{if(long){long=false;return;}const d=ev.target.closest('[data-del]');if(d){deleteEntity(d.dataset.del);return;}const tg=ev.target.closest('[data-toggle]');if(tg){toggleEntity(tg.dataset.toggle);return;}selectEntity(c.dataset.key,true);};});
 const all=$('allFaces');if(all)all.onclick=()=>{if(!change())return;const turnOn=!allOn;for(const e of people)for(const t of e.tracks){t.enabled=turnOn;t.opacity=100;}refreshAll();toast(turnOn?`${people.length} 个人都遮住了`:'已全部取消');};
}
async function deleteEntity(key){const e=entities().find(x=>x.key===key);if(!e||busy)return;const ok=await ask(`删除「${e.name}」？`,e.kind==='person'?'这个人的遮挡会被移除。只是暂时不想遮的话，点头像右下角的圆圈就行。':'这个遮挡会被移除，删错了可以点撤销恢复。',{yes:'删除',danger:true});if(!ok||!change())return;const ids=new Set(e.tracks.map(t=>t.id));p.tracks=p.tracks.filter(t=>!ids.has(t.id));if(p.face_catalog)for(const g of p.face_catalog.groups)g.track_ids=g.track_ids.filter(id=>!ids.has(id));if(selKey===key){selKey=null;refine=false;tool='select';}refreshAll();toast(`已删除「${e.name}」，可以撤销`,false,2500);}
function toggleEntity(key){const e=entities().find(x=>x.key===key);if(!e||!change())return;const on=!e.tracks.some(t=>t.enabled);for(const t of e.tracks){t.enabled=on;t.opacity=100;}refreshAll();toast(on?`遮住了「${e.name}」`:`「${e.name}」不遮了`,false,2200);}
function selectEntity(key,jump=false){selKey=key;refine=false;tool='select';canvas.style.cursor='default';updateHint();const e=selEntity();if(jump&&e){const vis=e.tracks.some(t=>trackMask(t,current)||boxAt(t,current));if(!vis)seek(Math.min(...e.tracks.filter(t=>t.enabled).map(t=>t.start).concat([e.tracks[0].start])));}refreshAll();}
function deselect(){selKey=null;refine=false;tool='select';canvas.style.cursor='default';updateHint();refreshAll();}
$('doneBtn').onclick=deselect;
$('faceToggle').onclick=()=>{if(selKey)toggleEntity(selKey);};

/* ---------- 面板 ---------- */
function renderPanes(){
 const e=selEntity();
 document.querySelectorAll('.tool').forEach(b=>b.classList.toggle('on',b.dataset.tool===tool||(b.dataset.tool==='point'&&refine)));
 const showStyle=!!e||tool==='fx';
 const active=showStyle||['text','point','box'].includes(tool);$('sheet').hidden=!active&&!wide();$('sideIntro').hidden=active;
 $('describe').hidden=tool!=='text'||!!e;
 const tip=!e&&tool==='point'?'在画面上点一下要遮的人或物':!e&&tool==='box'?'在画面上拖一个框，框住要遮的地方':'';
 $('toolTip').textContent=tip;$('toolTip').hidden=!tip;
 $('stylePane').hidden=!showStyle;$('objActions').hidden=!e;$('moreBox').hidden=!e;
 if(!showStyle)return;renderStyle();
 if(!e){$('selNote').textContent='这里选的是之后新遮挡的默认效果。点下面的头像，可以单独改某一个。';$('selNote').hidden=false;$('selNote').classList.remove('alert');return;}
 const ts=selTracks(),one=ts.length===1?ts[0]:null;
 const person=e.kind==='person',personOn=ts.some(t=>t.enabled);
 $('trackBtn').hidden=person;$('trackBtn').textContent=ts.some(t=>t.tracked)?'重新跟随':'跟随整段';
 $('faceToggle').hidden=false;$('faceToggle').textContent=personOn?(person?'不遮这个人':'不遮'):(person?'遮住这个人':'遮住');$('faceToggle').className=personOn?'pill':'pill-primary';
 $('refineBtn').hidden=!(one&&one.mask);$('refineSeg').hidden=!refine;$('refineBtn').classList.toggle('on',refine);$('refineBtn').textContent=refine?'修好了':'修轮廓';
 $('removeBtn').hidden=person;$('removeBtn').textContent='删除';
 const gaps=entGaps(e);let note='',alert=false;
 if(refine)note='选「加一块」或「去掉一块」，再点画面';
 else if(person)note=personOn?'这个人已经逐帧找好位置，整段都会遮住。':'这个人现在没有遮。';
 else if(gaps.length){note=`进度条上有 ${gaps.length} 处红色，可能漏了，点过去看看`;alert=true;}
 else if(one&&one.mask&&!one.tracked)note='现在只遮住了这一帧，点「跟随整段」';
 else if(one&&!one.mask&&one.keys.length===1&&!one.tracked)note='会动的东西点「跟随整段」，不动的就这样即可';
 $('selNote').textContent=note;$('selNote').hidden=!note;$('selNote').classList.toggle('alert',alert);
}
function styleSource(){const ts=selTracks();return ts[0]||Object.assign({effect:'mosaic',strength:50,opacity:100,padding:4,color:[20,20,22],shape:'rect'},p?.default_style||{});}
function renderStyle(){
 const s=styleSource(),ts=selTracks();const eff=s.effect==='cover'?'solid':s.effect;
 document.querySelectorAll('#fxRow button').forEach(b=>b.classList.toggle('on',b.dataset.fx===eff));
 $('swatches').hidden=eff!=='solid';
 $('swatches').innerHTML=COLORS.map(c=>`<button data-c="${c}" style="background:rgb(${c})" class="${(s.color||[]).join()===c.join()?'on':''}"></button>`).join('');
 $('swatches').querySelectorAll('button').forEach(b=>b.onclick=()=>setStyle({effect:'solid',color:b.dataset.c.split(',').map(Number)}));
 $('strengthRow').hidden=!['mosaic','blur'].includes(eff);$('strength').value=s.strength??50;
 const ts0=selTracks();$('sizeRow').hidden=!ts0.length;if(ts0.length){const v=sizeOf(ts0[0]);$('sizeRange').value=v;$('sizeOut').textContent=sizeLabel(v);}
 const allMask=ts.length&&ts.every(t=>t.mask),shape=s.shape||'rect';
 document.querySelectorAll('#shapeSeg button').forEach(b=>{b.classList.toggle('on',b.dataset.shape===shape);b.disabled=b.dataset.shape==='contour'&&!allMask;});
 const used=new Set(uploadedStickers);for(const t of p.tracks)if(t.sticker)used.add(t.sticker);
 $('stickerRow').innerHTML=EMOJI.map(em=>`<button data-emoji="${em}">${em}</button>`).join('')+[...used].map(n=>`<button data-sticker="${n}" class="${s.sticker===n?'on':''}"><img src="${path('sticker')}?n=${n}" alt=""></button>`).join('')+'<button id="stickerUpload" title="上传图片">＋</button>';
 $('stickerRow').querySelectorAll('[data-emoji]').forEach(b=>b.onclick=()=>emojiSticker(b.dataset.emoji));
 $('stickerRow').querySelectorAll('[data-sticker]').forEach(b=>b.onclick=()=>setStyle({effect:'sticker',sticker:b.dataset.sticker}));
 $('stickerUpload').onclick=()=>$('stickerFile').click();
 if(eff!=='sticker'&&!$('stickerRow').dataset.open)$('stickerRow').hidden=true;
}
function setStyle(v){
 const ts=selTracks();if(!change())return;
 for(const t of ts){Object.assign(t,v);t.opacity=100;if(v.effect&&v.effect!=='sticker')delete t.sticker;if(t.shape==='contour'&&!t.mask)t.shape='rect';}
 p.default_style=Object.assign({},p.default_style||{},{opacity:100},Object.fromEntries(Object.entries(v).filter(([k])=>['effect','strength','padding','color','sticker'].includes(k))));
 if(v.effect&&v.effect!=='sticker')delete p.default_style.sticker;
 renderStyle();scheduleRender();queueTiles();
}
document.querySelectorAll('#fxRow button').forEach(b=>b.onclick=()=>{const fx=b.dataset.fx;
 if(fx==='sticker'){$('stickerRow').hidden=false;$('stickerRow').dataset.open='1';const s=styleSource();if(s.sticker)setStyle({effect:'sticker',sticker:s.sticker});return;}
 $('stickerRow').hidden=true;delete $('stickerRow').dataset.open;const s=styleSource();setStyle(fx==='solid'?{effect:'solid',color:s.color||[20,20,22]}:{effect:fx});});
document.querySelectorAll('#shapeSeg button').forEach(b=>b.onclick=()=>setStyle({shape:b.dataset.shape}));
document.querySelectorAll('#qualitySeg button').forEach(b=>b.onclick=()=>{quality=b.dataset.q;document.querySelectorAll('#qualitySeg button').forEach(x=>x.classList.toggle('on',x===b));});
for(const[id,key]of[['strength','strength']]){
 const el=$(id);let pending=false;
 el.oninput=()=>{const ts=selTracks();if(!pending){pending=true;if(!change()){pending=false;return;}}for(const t of ts){t[key]=Number(el.value);t.opacity=100;}p.default_style=Object.assign({},p.default_style||{},{[key]:Number(el.value)});scheduleRender();};
 el.onchange=()=>{pending=false;queueTiles();};
}
$('applyAll').onclick=()=>{const s=styleSource();if(!change())return;for(const t of p.tracks){t.effect=s.effect;if(s.strength!=null)t.strength=s.strength;t.opacity=100;t.padding=s.padding;t.color=s.color||t.color;if(s.sticker)t.sticker=s.sticker;else delete t.sticker;}refreshAll();toast('全部都用上这个效果了');};
function editSel(fn){const ts=selTracks();if(!ts.length||!change())return;ts.forEach(fn);refreshAll();}
$('startHere').onclick=()=>editSel(t=>{t.start=Math.min(current,t.end);});
$('endHere').onclick=()=>editSel(t=>{t.end=Math.max(current,t.start);});
$('toEnd').onclick=()=>editSel(t=>{t.end=p.frames-1;});
$('removeBtn').onclick=()=>{if(selKey&&selEntity()?.kind!=='person'){deleteEntity(selKey);return;}const e=selEntity();if(!e||!change())return;
 if(e.kind==='person'){for(const t of e.tracks)t.enabled=false;deselect();return;}
 const ids=new Set(e.tracks.map(t=>t.id));p.tracks=p.tracks.filter(t=>!ids.has(t.id));deselect();toast('已删除，⌘Z 可以撤销');};
document.querySelectorAll('#refineSeg button').forEach(b=>b.onclick=()=>{refineSub=b.dataset.r==='sub';document.querySelectorAll('#refineSeg button').forEach(x=>x.classList.toggle('on',x===b));});
$('refineBtn').onclick=()=>{refineSub=false;document.querySelectorAll('#refineSeg button').forEach(x=>x.classList.toggle('on',x.dataset.r==='add'));refine=!refine;tool=refine?'point':'select';canvas.style.cursor=refine?'crosshair':'default';updateHint();renderPanes();scheduleRender();};
$('trackBtn').onclick=()=>{
 const ts=selTracks().filter(t=>t.enabled);if(!ts.length)return;
 const here=ts.filter(t=>trackMask(t,current)||boxAt(t,current));
 if(!here.length)return toast('这一帧里没有它，先拖到它出现的地方',true);
 refine=false;
 if(!ai.ready){if(here.length===1&&!here[0].mask){if(change())here[0].end=p.frames-1;return runJob('track',{track:here[0].id,frame:current});}return $('aiDialog').showModal();}
 runJob('ai/track',{tracks:here.map(t=>t.id),frame:current,direction:here.every(t=>t.tracked)?'forward':'both',quality});
};
async function uploadSticker(blob){const r=await fetch(path('sticker'),{method:'POST',headers:{'X-Workbench-Token':token,'Content-Type':blob.type||'image/png'},body:blob});const d=await r.json();if(!r.ok)throw Error(d.error);if(!uploadedStickers.includes(d.name))uploadedStickers.push(d.name);return d.name;}
async function emojiSticker(em){try{const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d');x.font='210px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(em,128,140);const blob=await new Promise(r=>c.toBlob(r,'image/png'));setStyle({effect:'sticker',sticker:await uploadSticker(blob)});}catch(e){toast(e.message,true);}}
$('stickerFile').onchange=async()=>{const f=$('stickerFile').files[0];$('stickerFile').value='';if(!f)return;try{setStyle({effect:'sticker',sticker:await uploadSticker(f)});}catch(e){toast(e.message,true);}};

/* 大小：按物体本身尺寸换算外扩；最小=正好盖住识别到的范围，核心部位始终遮住 */
function baseSize(t){const ks=t.keys;if(!ks?.length)return 60;const v=ks.map(k=>Math.max(k.box[2]*p.width,k.box[3]*p.height)).sort((a,b)=>a-b);return Math.max(8,v[v.length>>1]);}
function sizeOf(t){if(t.size!=null)return t.size;return Math.max(0,Math.min(100,Math.round((t.padding/baseSize(t)-0.1)/0.8*100)));}
function sizeLabel(v){return v<=2?'最小':`+${Math.round(10+v*0.8)}%`;}
{const el=$('sizeRange');let pending=false;
 el.oninput=()=>{const ts=selTracks();if(!ts.length)return;if(!pending){pending=true;if(!change()){pending=false;return;}}const v=Number(el.value);for(const t of ts){t.size=v;t.padding=Math.min(160,Math.round(baseSize(t)*(0.1+0.8*v/100)));t.opacity=100;}$('sizeOut').textContent=sizeLabel(v);scheduleRender();};
 el.onchange=()=>{pending=false;queueTiles();};}
/* 效果缩略图：用你自己的画面 */
function queueTiles(){clearTimeout(tilesTimer);tilesTimer=setTimeout(renderTiles,120);}
function renderTiles(){
 if(!p||!video.paused||$('stylePane').hidden||$('sheet').hidden)return;const s=styleSource();let m=null;
 for(const t of selTracks()){m=trackMask(t,current);if(m)break;}
 let bx,by,bw,bh;if(m)[bx,by,bw,bh]=m.box;else{bw=p.width*.25;bh=p.height*.3;bx=(p.width-bw)/2;by=(p.height-bh)/2;}
 const side=Math.min(p.width,p.height,Math.max(bw,bh)*1.25),cx=Math.max(0,Math.min(p.width-side,bx+bw/2-side/2)),cy=Math.max(0,Math.min(p.height-side,by+bh/2-side/2));
 document.querySelectorAll('#fxRow canvas').forEach(c=>{
  const fxName=c.parentElement.dataset.fx,x=c.getContext('2d'),k=c.width/side;
  const src=document.createElement('canvas');src.width=c.width;src.height=c.height;src.getContext('2d').drawImage(base,cx,cy,side,side,0,0,c.width,c.height);
  x.drawImage(src,0,0);
  const mm={id:'tile'+fxName,box:[(bx-cx)*k,(by-cy)*k,bw*k,bh*k],effect:fxName,strength:s.strength??50,opacity:100,color:fxName==='solid'?(s.color||[20,20,22]):null,shape:m?.shape==='contour'?'contour':(m?.shape||'rect'),polys:m?.polys?m.polys.map(q=>q.map((v,i)=>i%2?(v-cy)*k:(v-cx)*k)):null,pad:(s.padding||0)*k,sticker:s.sticker};
  if(fxName==='sticker'&&!s.sticker){x.fillStyle='rgba(20,21,24,.55)';x.fillRect(0,0,c.width,c.height);x.font='34px "Apple Color Emoji",sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText('🙂',c.width/2,c.height/2+2);return;}
  FX.draw(x,src,mm,stickerImg);
 });
}

/* ---------- 工具 ---------- */
function setTool(t){
 if(t==='point'&&!ai.ready){$('aiDialog').showModal();return;}
 if(t==='text'&&!ai.ready){$('aiDialog').showModal();return;}
 tool=t;if(t!=='point')refine=false;if(['text','point','box'].includes(t)&&selKey)selKey=null;
 canvas.style.cursor=t==='point'||t==='box'?'crosshair':'default';
 if(t==='text')setTimeout(()=>$('prompt').focus(),0);updateHint();renderPanes();renderChips();renderScrub();scheduleRender();queueTiles();
}
document.querySelectorAll('.tool').forEach(b=>b.onclick=()=>{if(selKey&&b.dataset.tool==='fx'){deselect();return;}setTool(tool===b.dataset.tool?'select':b.dataset.tool);});
function updateHint(){const h=$('hint');const s=refine?'点 = 加一块，Alt+点 = 去掉一块':'';h.textContent=s;h.hidden=!s;}
function pos(e){const r=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))};}
function hitTest(a){const ms=frameMasks(current).filter(m=>m.track).reverse();const x=a.x*p.width,y=a.y*p.height,r=canvas.getBoundingClientRect(),tol=14*p.width/r.width;
 const st=selTracks();const one=st.length===1&&!st[0].mask?trackMask(st[0],current):null;
 if(one){const[bx,by,bw,bh]=one.box;for(const[n,cx,cy]of[['nw',bx,by],['ne',bx+bw,by],['sw',bx,by+bh],['se',bx+bw,by+bh]])if(Math.abs(x-cx)<=tol&&Math.abs(y-cy)<=tol)return{m:one,corner:n};}
 for(const m of ms){const[bx,by,bw,bh]=m.box;if(x<bx||y<by||x>bx+bw||y>by+bh)continue;if(m.shape==='contour'&&m.polys&&!ctx.isPointInPath(FX.shapePath(m),x,y))continue;return{m,corner:null};}return null;}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.onpointerdown=async e=>{
 if(!p||busy)return;if(!video.paused)video.pause();const a=pos(e);
 if(tool==='point'){const remove=e.altKey||e.button===2||(refine&&refineSub);const ts=selTracks();const target=(refine||e.shiftKey||remove)&&ts.length===1&&ts[0].mask?ts[0]:null;
  if(remove&&!target)return;
  const pts=(target?.prompts?.[current]?.points||[]).concat([[+a.x.toFixed(4),+a.y.toFixed(4),remove?0:1]]);
  const r=await aiCall(()=>api(path('ai/click'),{frame:current,points:pts,track:target?.id},180000),target?[target.id]:[]);
  if(r){const t=p.tracks.find(x=>x.id===r.track);if(t&&!target){t.opacity=100;}selKey=entOf(t)?.key||'t:'+r.track;if(!refine)tool='select';canvas.style.cursor=refine?'crosshair':'default';updateHint();refreshAll();}
  return;}
 if(tool==='box'){drag={...a,ex:a.x,ey:a.y};canvas.setPointerCapture(e.pointerId);return;}
 const h=hitTest(a);if(!h){if(selKey)deselect();return;}
 const ent=entOf(h.m.track);if(ent&&ent.key!==selKey){selKey=ent.key;refreshAll();}
 if(!h.m.track.mask){const b=boxAt(h.m.track,current);if(b){gesture={id:h.m.track.id,origin:a,before:[...b],preview:[...b],corner:h.corner};canvas.setPointerCapture(e.pointerId);}}
};
canvas.onpointermove=e=>{if(!p)return;const a=pos(e);
 if(drag){drag.ex=a.x;drag.ey=a.y;scheduleRender();return;}
 if(gesture){const g=gesture,dx=a.x-g.origin.x,dy=a.y-g.origin.y;g.preview=g.corner?BoxEdit.resize(g.before,g.corner,dx,dy,8/p.width,8/p.height):BoxEdit.move(g.before,dx,dy);scheduleRender();return;}
 if(tool==='select'&&video.paused){const h=hitTest(a);canvas.style.cursor=h?(h.corner?(h.corner==='nw'||h.corner==='se'?'nwse-resize':'nesw-resize'):'pointer'):'default';}
};
canvas.onpointerup=()=>{
 if(gesture){const g=gesture;gesture=null;const t=p.tracks.find(x=>x.id===g.id);if(t&&g.preview.some((v,i)=>Math.abs(v-g.before[i])>1e-6)&&change()){const prev=current>t.start?boxAt(t,current-1):null;BoxEdit.applyFuture(t,current,g.before,g.preview,prev);}refreshAll();return;}
 if(!drag)return;const b=[Math.min(drag.x,drag.ex),Math.min(drag.y,drag.ey),Math.abs(drag.x-drag.ex),Math.abs(drag.y-drag.ey)];drag=null;
 if(b[2]*p.width>=6&&b[3]*p.height>=6&&change()){const s=styleSource(),n=p.tracks.filter(t=>t.kind==='box').length+1;const t={id:'box_'+Math.random().toString(36).slice(2,12),name:`框 ${n}`,kind:'box',start:current,end:p.frames-1,padding:s.padding??4,block:20,strength:s.strength??50,effect:s.effect==='sticker'&&!s.sticker?'mosaic':(s.effect||'mosaic'),opacity:100,color:s.color||[20,20,22],feather:3,shape:'rect',enabled:true,reviewed:false,keys:[{frame:current,box:b}]};if(t.effect==='sticker')t.sticker=s.sticker;p.tracks.push(t);selKey='t:'+t.id;tool='select';canvas.style.cursor='default';updateHint();}
 refreshAll();
};
async function aiCall(fn,changed=[]){
 if(!ai.ready){$('aiDialog').showModal();return null;}if(busy)return null;
 await flushSave();const rev=p.revision;const h=$('hint');h.textContent='正在识别…';h.hidden=false;busy=true;
 try{const r=await fn();undoStack.push({server:rev});applyProject(r.project,false,changed);return r;}
 catch(e){toast(e.message,true);return null;}
 finally{busy=false;updateHint();}
}
$('describe').onsubmit=async e=>{e.preventDefault();const q=$('prompt').value.trim();if(!q)return;
 const r=await aiCall(()=>api(path('ai/detect'),{frame:current,prompt:q,scope:'frame'},180000));
 if(r){if(!r.count)toast('这一帧没找到，换个说法或拖到它出现的地方再找',true);else{const t=p.tracks.find(x=>x.id===r.tracks[0]);tool='select';selKey=entOf(t)?.key;refreshAll();toast(`找到 ${r.count} 个，点「跟随整段」让它们跟着走`);}}};

/* ---------- 后台任务 ---------- */
async function runJob(kind,body={}){try{await flushSave();await api('/api/project/'+p.id+'/'+kind,body);watchJob();}catch(e){toast(e.message,true);}}
function lock(on){busy=on;$('sheet').inert=on;document.querySelector('.tabbar').inert=on;$('chips').inert=on;$('exportOpen').disabled=on;}
function watchJob(){
 clearTimeout(poll);lock(true);$('job').hidden=false;const id=p.id;
 const tick=async()=>{try{const j=await api('/api/project/'+id+'/job');lastJob=j;if(p.id!==id)return;
  const m=(j.message||'').match(/（([^）]*约剩[^）]*)）/);$('jobText').textContent=(JOB_TEXT[j.kind]||'处理中')+(m?' · '+m[1].split('，').pop():'');$('progress').style.width=(j.progress||0)+'%';$('jobCancel').hidden=j.kind==='export';if($('exportDialog').open)renderExport();
  if(j.state==='running'){poll=setTimeout(tick,1000);return;}
  lock(false);$('job').hidden=true;await loadProject(id,true);if(j.kind==='prepare-faces'&&j.state==='done')afterFaces();
  if(j.state==='error'||j.state==='cancelled')toast(j.message,j.state==='error',7000);
  else toast({export:'导出好了','ai-track':'跟好了，播放看看','ai-scan':'找完了，播放看看','prepare-faces':'画面里的人找好了，点下面的头像就能遮'}[j.kind]||'好了',false,5000);
  if(j.kind==='export'&&$('exportDialog').open)renderExport();
 }catch(e){lock(false);$('job').hidden=true;toast('连接断了，刷新页面即可，结果都保留着',true,8000);}};
 tick();
}
$('jobCancel').onclick=async()=>{try{await api(path('cancel'),{});}catch(e){toast(e.message,true);}};

/* ---------- 导出 ---------- */
function renderExport(){
 const b=$('exportBody');if(!p)return;const e=p.exports.at(-1);const running=lastJob?.kind==='export'&&lastJob?.state==='running';
 let h='';
 if(running)h+=`<p>正在导出… ${Math.round(lastJob.progress||0)}%</p>`;
 else if(!e||e.revision!==p.revision)h+=`<p class="note">导出一份新的视频，原视频不会被改动。</p><p><button id="doExport" class="pill-primary">开始导出</button></p>`;
 if(e&&!running){const u=path('output')+'?v='+encodeURIComponent(e.version);h+=`${e.revision!==p.revision?'<p class="note">下面是改动之前导出的版本。</p>':''}<video controls playsinline preload="metadata" src="${u}"></video><div class="export-links"><a href="${u}" download="打码版.mp4" class="pill-primary" style="text-decoration:none;color:#111">下载视频</a><span class="note" style="margin:0">导出后请完整看一遍，确认没有漏掉。</span></div>`;}
 b.innerHTML=h;if($('doExport'))$('doExport').onclick=()=>runJob('export');
}
$('exportOpen').onclick=async()=>{await flushSave();renderExport();$('exportDialog').showModal();};

/* ---------- 导入 ---------- */
let pendingQuick=null;
$('uploadHome').onclick=()=>{pendingQuick=null;if(!busy)$('file').click();};
document.querySelectorAll('[data-quick]').forEach(b=>b.onclick=()=>{pendingQuick=b.dataset.quick;if(!busy)$('file').click();});
$('file').onchange=()=>{const f=$('file').files[0];$('file').value='';if(f)upload(f);};
async function upload(f){if(busy)return;if(!/\.(mp4|mov|m4v|mkv|webm)$/i.test(f.name))return toast('请选择视频文件（MP4、MOV 等）',true);toast('正在导入…',false,600000);
 try{await flushSave();const r=await fetch('/api/upload',{method:'POST',headers:{'X-Workbench-Token':token,'X-Filename':encodeURIComponent(f.name),'X-Quick':pendingQuick||''},body:f});const d=await r.json();if(!r.ok)throw Error(d.error);
  let j;for(;;){await new Promise(s=>setTimeout(s,700));j=await api('/api/import/'+d.import);if(j.state!=='running')break;toast(j.message==='转换手机视频'?`正在转换成标准格式 ${j.progress||0}%`:'正在检查视频…',false,600000);}
  if(j.state!=='done')throw Error(j.message||'导入失败');
  projects=(await api('/api/bootstrap')).projects.sort((a,b)=>b.created-a.created);p=null;toast('');await loadProject(j.project);applyQuick();}catch(e){toast(e.message,true,8000);}}
let dragDepth=0;addEventListener('dragenter',e=>{if(e.dataTransfer?.types?.includes('Files')){dragDepth++;$('drop').hidden=false;}});addEventListener('dragleave',()=>{if(--dragDepth<=0){dragDepth=0;$('drop').hidden=true;}});addEventListener('dragover',e=>e.preventDefault());addEventListener('drop',e=>{e.preventDefault();dragDepth=0;$('drop').hidden=true;const f=e.dataTransfer.files[0];if(f)upload(f);});

function applyQuick(){const q=pendingQuick;if(!q||!p)return;
 if(q==='text'||q==='plate'){pendingQuick=null;setTool('text');$('prompt').value=q==='plate'?'车牌':'logo、文字';}
 else if(q==='sticker'){pendingQuick=null;p.default_style=Object.assign({},p.default_style||{},{effect:'sticker'});emojiStickerDefault('🙂');setTool('point');}
 else if(q==='face')toast('正在找画面里的人，找到后会自动遮住',false,6000);}
async function emojiStickerDefault(em){try{const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d');x.font='210px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(em,128,140);const blob=await new Promise(r=>c.toBlob(r,'image/png'));const n=await uploadSticker(blob);if(change())p.default_style=Object.assign({},p.default_style||{},{effect:'sticker',sticker:n,opacity:100});}catch(e){}}
function afterFaces(){if(pendingQuick!=='face'||!p)return;pendingQuick=null;const fs=p.tracks.filter(t=>t.auto_mode==='faces');if(!fs.length)return toast('没有找到人脸，可以用「点选」手动点',true);if(change())for(const t of fs){t.enabled=true;t.effect='mosaic';t.strength=55;t.opacity=100;}refreshAll();toast(`遮住了 ${entities().filter(e=>e.kind==='person').length} 个人，不想遮谁就点头像再点「不遮」`,false,6000);}
/* ---------- 键盘 ---------- */
document.addEventListener('keydown',e=>{
 if(e.target.closest('input,textarea,select,dialog'))return;
 if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo();return;}
 if(e.metaKey||e.ctrlKey||e.altKey||!p||$('editor').hidden)return;
 if(e.code==='Space'){e.preventDefault();togglePlay();}
 else if(e.key==='ArrowLeft'){e.preventDefault();seek(current-(e.shiftKey?10:1));}
 else if(e.key==='ArrowRight'){e.preventDefault();seek(current+(e.shiftKey?10:1));}
 else if(e.key==='Escape'){drag=null;gesture=null;if(tool!=='select'||refine){refine=false;setTool('select');}else deselect();}
 else if((e.key==='Delete'||e.key==='Backspace')&&selEntity()){e.preventDefault();$('removeBtn').click();}
});
addEventListener('beforeunload',e=>{if(dirty){flushSave();e.preventDefault();e.returnValue='';}});
addEventListener('resize',()=>{if(p){renderPanes();scheduleRender();}});

/* ---------- 启动 ---------- */
(async()=>{try{const b=await api('/api/bootstrap');token=b.token;projects=b.projects.sort((a,b)=>b.created-a.created);try{ai=await api('/api/ai/status');}catch(e){}$('aiError').textContent=ai.ready?'':(ai.error||'');showHome();}catch(e){toast('连不上，请双击「重启工作台.command」',true,20000);}})();
