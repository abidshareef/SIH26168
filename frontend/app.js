const API = window.location.origin;
let busy = false;
let token = sessionStorage.getItem('navAionApiKey') || '';
const $ = id => document.getElementById(id);
const modes = {GNSS:'GNSS NAVIGATION',GNSS_DEGRADED:'GNSS DEGRADED',DEAD_RECKONING:'DEAD RECKONING',RECOVERY:'GNSS RECOVERY'};
const labels = {gnss_confidence:'GNSS',ai_confidence:'AI',imu_confidence:'IMU',nhc_confidence:'NHC',map_confidence:'MAP',mount_confidence:'MOUNT'};

function valid(s){return s&&Number.isFinite(s.latitude)&&Math.abs(s.latitude)<=90&&Number.isFinite(s.longitude)&&Math.abs(s.longitude)<=180&&Number.isFinite(s.velocity)&&Number.isFinite(s.heading)&&s.heading>=0&&s.heading<360&&Number.isFinite(s.position_uncertainty)&&s.position_uncertainty>=0&&s.trust&&Object.values(s.trust).every(v=>Number.isFinite(v)&&v>=0&&v<=1)}
function pct(v){return `${Math.round(Math.max(0,Math.min(1,v))*100)}%`}
function setConnected(ok){$('connection').textContent=ok?'CONNECTED':'BACKEND DISCONNECTED';$('connection').className=`connection ${ok?'ok':'error'}`}
function showApp(){ $('auth-gate').style.display='none'; $('app-shell').classList.remove('hidden-app') }
function showAuth(message=''){ $('auth-gate').style.display='grid'; $('app-shell').classList.add('hidden-app'); $('auth-error').textContent=message }

function render(s){
  if(!valid(s)){ $('notice').textContent='Backend returned invalid state; waiting for a valid navigation state.';return }
  const trust=s.trust;
  $('mode').textContent=modes[s.mode]||s.mode;$('mode-chip').textContent=modes[s.mode]||s.mode;
  $('speed').textContent=(s.velocity*3.6).toFixed(1);$('heading').textContent=`${Math.round(s.heading).toString().padStart(3,'0')}°`;
  $('uncertainty').textContent=`±${s.position_uncertainty.toFixed(1)} m`;$('variance').textContent=Number(s.velocity_variance).toFixed(3);$('timestamp').textContent=Number(s.timestamp).toFixed(1)+' s';
  $('coordinates').textContent=`${s.latitude.toFixed(5)}, ${s.longitude.toFixed(5)}`;$('gnss-sensor').textContent=trust.gnss_confidence===0?'LOST':'ACTIVE';$('vehicle').style.transform=`rotate(${s.heading}deg)`;
  $('metric-uncertainty').textContent=`±${s.position_uncertainty.toFixed(1)} m`;$('metric-gnss').textContent=pct(trust.gnss_confidence);$('metric-ai').textContent=pct(trust.ai_confidence);
  const vals=[trust.imu_confidence,trust.nhc_confidence,trust.map_confidence,trust.mount_confidence];$('metric-fusion').textContent=pct(vals.reduce((a,b)=>a+b,0)/vals.length);
  $('trust').innerHTML=Object.entries(labels).map(([k,label])=>{const v=trust[k]??0;return `<div class="trust-row ${v<.4?'low':''}"><span>${label}</span><div class="bar"><i style="width:${v*100}%"></i></div><b>${pct(v)}</b></div>`}).join('');
  const outage=s.mode==='DEAD_RECKONING';$('warning').textContent=outage?'GNSS SIGNAL LOST — DEAD RECKONING ACTIVE':s.mode==='RECOVERY'?'GNSS RETURNED — INNOVATION CHECK IN PROGRESS':'';$('warning').classList.toggle('hidden',!outage&&s.mode!=='RECOVERY');$('notice').textContent=s.simulation_note||'Backend state synchronized.';
}

async function request(path,method='GET',renderResponse=true){
  if(busy)return null;busy=true;
  try{
    const r=await fetch(API+path,{method,headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},signal:AbortSignal.timeout(5000)});
    if(r.status===401){sessionStorage.removeItem('navAionApiKey');token='';showAuth('Authentication failed. Check the API key configured on the backend.');setConnected(false);return null}
    if(!r.ok)throw Error(`HTTP ${r.status}`);
    const data=await r.json();setConnected(true);if(renderResponse)render(data.state||data);return data;
  }catch(e){setConnected(false);$('notice').textContent=`${e.message}. Verify the NAV-AION backend and API key.`;return null}
  finally{busy=false}
}

$('auth-form').addEventListener('submit',async event=>{
  event.preventDefault();
  const candidate=$('api-key').value.trim();
  if(candidate.length<16){$('auth-error').textContent='API key is too short.';return}
  token=candidate;
  const result=await request('/health','GET',false);
  if(result){sessionStorage.setItem('navAionApiKey',token);showApp();await request('/navigation/reset','POST');}
  else{token='';$('api-key').value='';}
});

document.querySelectorAll('button[data-action]').forEach(b=>b.onclick=()=>request(b.dataset.action,'POST'));

async function boot(){
  if(!token){showAuth();return}
  const result=await request('/health','GET',false);
  if(!result){showAuth('Your saved session is no longer authorized.');return}
  showApp();await request('/navigation/reset','POST');setInterval(()=>request('/navigation/state'),500);
}
boot();
