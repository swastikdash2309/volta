"""
build_dashboard.py
------------------
Reads volta_run.json and writes a single self-contained, professional dashboard
(volta_dashboard.html) you can open in any browser. No server, no install.

If a VOLTA backend is serving the page, a "New day" control appears that runs
any day on demand via /api/simulate. Hosted statically, it replays one prebuilt
day. Same file, both modes.
"""
import os, json

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "volta_run.json")) as f:
    DATA = json.load(f)

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VOLTA | Fleet charging orchestration</title>
<style>
  :root{--bg:#0b0e13;--card:#0e1217;--card2:#121822;--line:#1f2733;--line2:#2b3542;
        --text:#e6edf3;--muted:#8b97a4;--green:#3fb984;--solar:#e3b341;--carbon:#58a6ff;--grey:#566270;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:radial-gradient(1200px 600px at 50% -10%,#121a24,var(--bg));color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    min-height:100vh;display:flex;align-items:flex-start;justify-content:center;padding:34px 18px;}
  .wrap{width:100%;max-width:880px;}
  .vcard{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px 24px 20px;
    box-shadow:0 24px 70px rgba(0,0,0,.35);animation:vIn .6s cubic-bezier(.2,.7,.2,1);}
  @keyframes vIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
  .vhead{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:18px;flex-wrap:wrap;}
  .vbrand{display:flex;align-items:center;gap:13px;}
  .vlogo{width:38px;height:38px;border-radius:10px;background:conic-gradient(from 140deg,#3fb984,#2f8f66 55%,#2bd49a);box-shadow:0 0 0 1px #ffffff14 inset;}
  .vbrand h1{font-size:20px;letter-spacing:.5px;font-weight:700;}
  .vbrand p{font-size:12px;color:var(--muted);margin-top:1px;}
  .vseg{position:relative;display:flex;background:#0a0e12;border:1px solid var(--line);border-radius:11px;padding:3px;width:208px;}
  .vseg button{position:relative;z-index:2;flex:1;background:none;border:none;color:var(--muted);
    font-size:13px;font-weight:600;padding:8px 0;cursor:pointer;transition:color .25s;letter-spacing:.3px;}
  .vseg button.on{color:#06210f;}
  .vpill{position:absolute;top:3px;left:3px;width:calc(50% - 3px);height:calc(100% - 6px);border-radius:9px;
    background:linear-gradient(180deg,#46cf90,#34b27c);transition:transform .45s cubic-bezier(.2,.8,.2,1);}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin-bottom:16px;}
  .kpi{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:12px 13px;}
  .kpi .kl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);}
  .kpi .kv{font-size:23px;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums;letter-spacing:.3px;}
  .kpi .kd{font-size:10.5px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums;height:13px;transition:opacity .3s;}
  .kpi .kd.good{color:var(--green);} .kpi .kd.warn{color:var(--solar);} .kpi .kd.flat{color:var(--muted);}
  .chartwrap{position:relative;}
  .chartwrap canvas{width:100%;display:block;border-radius:11px;}
  .legend{display:flex;gap:18px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin:10px 2px 0;}
  .legend i{display:inline-block;width:20px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px;}
  .tip{position:absolute;pointer-events:none;background:#0a0e12;border:1px solid var(--line2);border-radius:9px;
    padding:9px 11px;font-size:11.5px;opacity:0;transition:opacity .15s;transform:translateX(-50%);white-space:nowrap;z-index:5;box-shadow:0 8px 24px rgba(0,0,0,.45);}
  .tip b{color:#fff;} .tip .r{display:flex;justify-content:space-between;gap:16px;margin-top:3px;color:var(--muted);}
  .tip .r span:last-child{color:var(--text);}
  .vctrls{display:flex;align-items:center;gap:14px;margin:17px 0 15px;flex-wrap:wrap;}
  .vbtn{background:#1b2531;border:1px solid var(--line2);color:var(--text);font-weight:600;font-size:13px;
    border-radius:10px;padding:8px 17px;cursor:pointer;transition:background .2s,border-color .2s;}
  .vbtn:hover{background:#243043;border-color:#3a4757;}
  .vbtn.accent{background:#13344a;border-color:#1f5478;color:#cfe8ff;}
  .vctrls input[type=range]{flex:1;min-width:160px;accent-color:var(--green);height:4px;}
  .clk{font-variant-numeric:tabular-nums;font-weight:700;font-size:14px;min-width:56px;text-align:right;letter-spacing:.5px;}
  .seed{width:78px;background:#0a0e12;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:7px 9px;font-size:12.5px;}
  .live{display:none;align-items:center;gap:9px;}
  .lstat{font-size:11.5px;color:var(--muted);min-width:64px;}
  .psw{display:flex;align-items:center;gap:8px;}
  .plab{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .ptog{position:relative;display:flex;background:#0a0e12;border:1px solid var(--line);border-radius:9px;padding:3px;}
  .ptog button{position:relative;z-index:2;background:none;border:none;color:var(--muted);font-size:11.5px;font-weight:600;padding:5px 13px;cursor:pointer;transition:color .25s;}
  .ptog button.on{color:#cfe8ff;}
  .ppill{position:absolute;top:3px;left:3px;width:calc(50% - 3px);height:calc(100% - 6px);border-radius:7px;
    background:#13344a;border:1px solid #1f5478;transition:transform .4s cubic-bezier(.2,.8,.2,1);}
  #pbadge{color:var(--carbon);font-weight:600;}
  .sech{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:4px 2px 8px;}
  .cars{display:grid;grid-template-columns:repeat(12,1fr);gap:6px;height:70px;align-items:end;}
  .vbar{position:relative;height:100%;background:#0a0e12;border:1px solid var(--line);border-radius:5px;overflow:hidden;}
  .vfill{position:absolute;left:0;right:0;bottom:0;background:var(--grey);}
  .vtgt{position:absolute;left:0;right:0;height:1.5px;background:#ffffff44;}
  .caxis{display:flex;justify-content:space-between;font-size:9.5px;color:#6b7681;margin-top:6px;letter-spacing:.3px;}
  .foot{color:#6b7681;font-size:11px;margin-top:16px;text-align:center;letter-spacing:.2px;}
  /* hero / landing */
  .hero{text-align:center;padding:30px 10px 36px;animation:vIn .6s cubic-bezier(.2,.7,.2,1);}
  .eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.16em;color:var(--green);font-weight:700;}
  .hero h2{font-size:clamp(26px,4.4vw,40px);line-height:1.12;font-weight:800;margin:14px auto 0;max-width:660px;letter-spacing:-.2px;}
  .hero h2 .hl{color:var(--green);}
  .hero .lead{color:var(--muted);font-size:15px;line-height:1.6;max-width:560px;margin:16px auto 0;}
  .stats{display:flex;justify-content:center;gap:34px;flex-wrap:wrap;margin:30px auto 6px;}
  .stat .n{font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:.3px;}
  .stat .n.g{color:var(--green);} .stat .c{font-size:12px;color:var(--muted);margin-top:3px;max-width:170px;}
  .steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:30px auto 26px;max-width:720px;text-align:left;}
  .step{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:15px 16px;}
  .step .si{width:26px;height:26px;border-radius:7px;background:#16202c;border:1px solid var(--line2);
    display:flex;align-items:center;justify-content:center;font-size:12.5px;font-weight:700;color:var(--green);}
  .step h4{font-size:13.5px;margin:11px 0 4px;font-weight:700;}
  .step p{font-size:12px;color:var(--muted);line-height:1.5;}
  .cta{display:inline-flex;align-items:center;gap:9px;background:linear-gradient(180deg,#46cf90,#34b27c);
    color:#06210f;font-weight:700;font-size:14px;border:none;border-radius:11px;padding:13px 26px;cursor:pointer;
    text-decoration:none;transition:transform .18s,box-shadow .18s;box-shadow:0 8px 26px rgba(63,185,132,.25);}
  .cta:hover{transform:translateY(-2px);box-shadow:0 12px 34px rgba(63,185,132,.34);}
  .cta .arr{transition:transform .18s;} .cta:hover .arr{transform:translateY(3px);}
  @media(max-width:620px){.steps{grid-template-columns:1fr;}.stats{gap:22px;}}
</style>
</head>
<body>
<div class="wrap">
 <section class="hero">
   <div class="eyebrow">Vehicle-to-grid intelligence · software only</div>
   <h2>Charge the fleet with <span class="hl">the sun</span>, not against the grid.</h2>
   <p class="lead">VOLTA is an open-source controller that teaches a fleet of electric vehicles to
     time their charging around clean, low-carbon power. No hardware. Just software and a policy
     that learns to coordinate.</p>
   <div class="stats">
     <div class="stat"><div class="n g">~40%</div><div class="c">lower charging emissions vs uncontrolled charging</div></div>
     <div class="stat"><div class="n">12</div><div class="c">autonomous agents coordinating, no central controller</div></div>
     <div class="stat"><div class="n">100%</div><div class="c">software, runs in any browser</div></div>
   </div>
   <div class="steps">
     <div class="step"><div class="si">1</div><h4>Sense the grid</h4><p>Read solar supply, demand, prices, and real-time carbon intensity.</p></div>
     <div class="step"><div class="si">2</div><h4>Learn the timing</h4><p>Each vehicle is a reinforcement-learning agent that learns when to charge.</p></div>
     <div class="step"><div class="si">3</div><h4>Charge clean</h4><p>The fleet shifts its load into the low-carbon window, on its own.</p></div>
   </div>
   <a class="cta" href="#demo">Explore the live demo <span class="arr">&#8595;</span></a>
 </section>
 <div class="vcard" id="demo">
  <div class="vhead">
    <div class="vbrand"><div class="vlogo"></div><div><h1>VOLTA</h1><p>Fleet charging orchestration<span id="pbadge"></span></p></div></div>
    <div class="vseg" id="seg"><div class="vpill" id="pill"></div>
      <button data-m="0">Baseline</button><button data-m="1" class="on">VOLTA</button></div>
  </div>
  <div class="kpis" id="kpis"></div>
  <div class="chartwrap">
    <canvas id="cv"></canvas><div class="tip" id="tip"></div>
    <div class="legend">
      <span><i style="background:var(--solar)"></i>Solar available</span>
      <span><i style="background:var(--carbon)"></i>Grid carbon intensity</span>
      <span><i style="background:var(--green)"></i>Fleet charging load</span>
    </div>
  </div>
  <div class="vctrls">
    <button class="vbtn" id="play">Play</button>
    <input type="range" id="sl" min="0" max="47" value="0"/>
    <span class="clk" id="clk">00:00</span>
    <span class="psw"><span class="plab">Training</span>
      <div class="ptog" id="ptog"><div class="ppill" id="ppill"></div>
        <button data-p="0" class="on">Standard</button><button data-p="1">Private</button></div></span>
    <span class="live" id="live">
      <span style="color:var(--muted);font-size:12px">seed</span>
      <input class="seed" id="seedIn" type="number" value="303" min="0"/>
      <button class="vbtn accent" id="newDay">New day</button>
      <span class="lstat" id="lstat"></span>
    </span>
  </div>
  <div class="sech">Per-vehicle charge (green while charging, tick marks the driver target)</div>
  <div class="cars" id="cars"></div>
  <div class="caxis"><span>Vehicle 1</span><span>Vehicle 12</span></div>
  <div class="foot">Self-contained demo. Synthetic offline data. Seed __SEED__. Built by Swastik Dash.</div>
</div></div>
<script>
let DATA = __DATA__;
__CORE__
</script>
</body>
</html>
"""

CORE = r"""
const $=function(i){return document.getElementById(i);};
const cv=$("cv"), ctx=cv.getContext("2d"), wrap=cv.parentElement, tip=$("tip");
let N=DATA.steps_per_day, LIM=DATA.limit, NC=DATA.cars.length;
let aT=1, a=1, t=0, playing=false, hover=null, dragging=false, carD=new Array(NC).fill(0);
let STD=DATA.runs.volta, PRIV=DATA.runs.volta_private||DATA.runs.volta, priv=false;

const KP=[{k:"carbon",l:"CO2 emissions",u:" kg",d:1,low:true,pre:""},
          {k:"cost",l:"Charging cost",u:"",d:1,low:true,pre:"$"},
          {k:"peak",l:"Peak grid load",u:" kW",d:0,low:true,pre:""},
          {k:"worst",l:"Worst-off driver",u:"",d:2,low:false,pre:""}];
$("kpis").innerHTML=KP.map(function(c,i){return '<div class="kpi"><div class="kl">'+c.l+
  '</div><div class="kv" id="kv'+i+'">-</div><div class="kd" id="kd'+i+'"></div></div>';}).join("");

function buildCars(){
  $("cars").innerHTML="";
  for(let c=0;c<NC;c++){$("cars").insertAdjacentHTML("beforeend",
    '<div class="vbar"><div class="vtgt" style="bottom:'+(DATA.cars[c].target*100)+'%"></div><div class="vfill" id="fl'+c+'"></div></div>');}
  carD=new Array(NC).fill(0);
}
buildCars();

let W=760,H=240,dpr=1;
function fit(){dpr=Math.min(window.devicePixelRatio||1,2);W=wrap.clientWidth;H=240;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.height=H+"px";ctx.setTransform(dpr,0,0,dpr,0,0);}
const ML=36,MR=36,MT=14,MB=24;
const X=function(i){return ML+(W-ML-MR)*i/(N-1);};
const YP=function(v){return H-MB-(H-MT-MB)*(v/LIM);};
const cN=function(v){return (v-120)/(600-120);};
const YC=function(v){return H-MB-(H-MT-MB)*Math.max(0,Math.min(1,cN(v)));};
const lerp=function(x,y,f){return x+(y-x)*f;};
function pAt(i){return lerp(DATA.runs.naive.power[i],DATA.runs.volta.power[i],a);}
function socAt(s,c){return lerp(DATA.runs.naive.soc[s][c],DATA.runs.volta.soc[s][c],a);}

function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle="rgba(255,255,255,0.05)";ctx.lineWidth=1;
  for(let h=0;h<=24;h+=6){const x=ML+(W-ML-MR)*(h/24);ctx.beginPath();ctx.moveTo(x,MT);ctx.lineTo(x,H-MB);ctx.stroke();}
  ctx.beginPath();ctx.moveTo(X(0),H-MB);
  for(let i=0;i<N;i++)ctx.lineTo(X(i),H-MB-(H-MT-MB)*DATA.solar[i]);
  ctx.lineTo(X(N-1),H-MB);ctx.closePath();
  let g=ctx.createLinearGradient(0,MT,0,H-MB);g.addColorStop(0,"rgba(227,179,65,0.22)");g.addColorStop(1,"rgba(227,179,65,0.02)");
  ctx.fillStyle=g;ctx.fill();
  ctx.beginPath();for(let i=0;i<N;i++){const y=YC(DATA.carbon[i]);i?ctx.lineTo(X(i),y):ctx.moveTo(X(i),y);}
  ctx.strokeStyle="rgba(88,166,255,0.85)";ctx.lineWidth=1.6;ctx.stroke();
  ctx.beginPath();for(let i=0;i<N;i++){const y=YP(DATA.runs.naive.power[i]);i?ctx.lineTo(X(i),y):ctx.moveTo(X(i),y);}
  ctx.strokeStyle="rgba(120,131,144,"+(0.18+0.30*(1-a))+")";ctx.lineWidth=1.4;ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([]);
  ctx.beginPath();ctx.moveTo(X(0),H-MB);
  for(let i=0;i<N;i++)ctx.lineTo(X(i),YP(pAt(i)));
  ctx.lineTo(X(N-1),H-MB);ctx.closePath();
  const col=[Math.round(lerp(107,63,a)),Math.round(lerp(118,185,a)),132];
  const cc="rgba("+col[0]+","+col[1]+","+col[2]+",";
  g=ctx.createLinearGradient(0,MT,0,H-MB);g.addColorStop(0,cc+"0.30)");g.addColorStop(1,cc+"0.02)");
  ctx.fillStyle=g;ctx.fill();
  ctx.beginPath();for(let i=0;i<N;i++){const y=YP(pAt(i));i?ctx.lineTo(X(i),y):ctx.moveTo(X(i),y);}
  ctx.strokeStyle=cc+"1)";ctx.lineWidth=2.4;ctx.lineJoin="round";ctx.stroke();
  const ph=hover!=null?hover:t;
  ctx.beginPath();ctx.moveTo(X(ph),MT-2);ctx.lineTo(X(ph),H-MB);ctx.strokeStyle="rgba(230,237,243,0.55)";ctx.lineWidth=1;ctx.stroke();
  const py=YP(pAt(Math.round(ph)));
  ctx.beginPath();ctx.arc(X(ph),py,3.6,0,7);ctx.fillStyle=cc+"1)";ctx.fill();ctx.strokeStyle="#0e1217";ctx.lineWidth=1.6;ctx.stroke();
  ctx.fillStyle="#6b7681";ctx.font="10px -apple-system,sans-serif";ctx.textAlign="center";
  ["00:00","06:00","12:00","18:00","24:00"].forEach(function(s,k){ctx.fillText(s,ML+(W-ML-MR)*(k/4),H-8);});
  ctx.textAlign="left";ctx.fillText(LIM+" kW",4,MT+8);ctx.fillText("0",4,H-MB);
}
function clock(i){const h=Math.floor(DATA.hours[i]),m=Math.round((DATA.hours[i]-h)*60);
  return (h<10?"0":"")+h+":"+(m<10?"0":"")+m;}
function updKPI(){
  KP.forEach(function(c,i){
    const nv=DATA.runs.naive.metrics[c.k], vv=DATA.runs.volta.metrics[c.k];
    $("kv"+i).textContent=c.pre+lerp(nv,vv,a).toFixed(c.d)+c.u;
    const kd=$("kd"+i);
    if(c.k==="worst"){const diff=vv-nv;kd.textContent=(diff>=0?"+":"")+diff.toFixed(2)+" vs baseline";
      kd.className="kd "+(Math.abs(diff)<0.005?"flat":(diff<0?"warn":"good"));}
    else{const pc=nv===0?0:100*(vv-nv)/nv;kd.textContent=(pc>0?"+":"")+pc.toFixed(1)+"% vs baseline";
      kd.className="kd "+(Math.abs(pc)<0.05?"flat":((c.low?pc<0:pc>0)?"good":"warn"));}
    kd.style.opacity=a.toFixed(2);
  });
}
function updCars(){
  const ds=hover!=null?hover:Math.round(t);
  for(let c=0;c<NC;c++){
    const tg=socAt(ds,c);carD[c]+=(tg-carD[c])*0.25;
    const f=$("fl"+c);if(!f)continue;f.style.height=(carD[c]*100)+"%";
    const prev=ds>0?socAt(ds-1,c):tg;const charging=(tg-prev)>0.004;
    const plugged=ds>=DATA.cars[c].arrival && ds<DATA.cars[c].departure;
    f.style.background=charging?"#3fb984":(plugged?"#566270":"#2c333d");
    f.style.opacity=plugged?"1":"0.55";
  }
}
function frame(){
  a+=(aT-a)*0.12;if(Math.abs(a-aT)<0.002)a=aT;
  if(playing){t+=0.16;if(t>=N-1)t=0;}
  $("pill").style.transform="translateX("+(a*100)+"%)";
  const ds=hover!=null?hover:Math.round(t);
  $("clk").textContent=clock(ds);
  if(!dragging)$("sl").value=Math.round(t);
  draw();updKPI();updCars();requestAnimationFrame(frame);
}
function setMode(m){aT=m;document.querySelectorAll(".vseg button").forEach(function(b){b.classList.toggle("on",(+b.dataset.m)===m);});}
document.querySelectorAll(".vseg button").forEach(function(b){b.onclick=function(){setMode(+b.dataset.m);};});
function setPriv(p){priv=p;DATA.runs.volta=p?PRIV:STD;
  document.querySelectorAll(".ptog button").forEach(function(b){b.classList.toggle("on",(+b.dataset.p)===(p?1:0));});
  $("ppill").style.transform="translateX("+(p?100:0)+"%)";
  $("pbadge").textContent=p?"  /  private mode, federated":"";}
document.querySelectorAll(".ptog button").forEach(function(b){b.onclick=function(){setPriv((+b.dataset.p)===1);};});
$("play").onclick=function(){playing=!playing;this.textContent=playing?"Pause":"Play";hover=null;};
$("sl").addEventListener("input",function(e){t=+e.target.value;playing=false;$("play").textContent="Play";hover=null;});
$("sl").addEventListener("pointerdown",function(){dragging=true;});
window.addEventListener("pointerup",function(){dragging=false;});
cv.addEventListener("mousemove",function(e){
  const r=cv.getBoundingClientRect();let i=Math.round((e.clientX-r.left-ML)/(W-ML-MR)*(N-1));
  i=Math.max(0,Math.min(N-1,i));hover=i;
  tip.style.left=Math.max(64,Math.min(W-64,X(i)))+"px";tip.style.top=MT+"px";tip.style.opacity="1";
  tip.innerHTML="<b>"+clock(i)+"</b><div class='r'><span>Solar</span><span>"+Math.round(DATA.solar[i]*100)+"%</span></div>"+
    "<div class='r'><span>Carbon</span><span>"+Math.round(DATA.carbon[i])+" g</span></div>"+
    "<div class='r'><span>Load</span><span>"+pAt(i).toFixed(1)+" kW</span></div>";
});
cv.addEventListener("mouseleave",function(){hover=null;tip.style.opacity="0";});
window.addEventListener("resize",fit);

function applyData(d){DATA=d;N=d.steps_per_day;LIM=d.limit;NC=d.cars.length;
  STD=d.runs.volta;PRIV=d.runs.volta_private||d.runs.volta;DATA.runs.volta=priv?PRIV:STD;
  $("sl").max=N-1;t=0;buildCars();}

// live backend mode
fetch("api/health").then(function(r){return r.ok?r.json():Promise.reject();}).then(function(){
  $("live").style.display="inline-flex";
  $("newDay").onclick=function(){
    const seed=Math.max(0,parseInt($("seedIn").value||"0",10));
    $("lstat").textContent="running";if(playing)$("play").click();
    fetch("api/simulate?seed="+seed+"&cars=12").then(function(r){return r.json();}).then(function(d){
      applyData(d);$("lstat").textContent="day "+seed+" ready";
    }).catch(function(){$("lstat").textContent="error";});
  };
}).catch(function(){});

fit();frame();
"""

html = (HEAD.replace("__DATA__", json.dumps(DATA))
            .replace("__CORE__", CORE)
            .replace("__SEED__", str(DATA["seed"])))
out = os.path.join(HERE, "volta_dashboard.html")
with open(out, "w") as f:
    f.write(html)
print("Wrote", out, "(", len(html), "bytes )")
