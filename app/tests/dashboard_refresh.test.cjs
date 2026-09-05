const assert = require('node:assert/strict');
const {createResource, createLifecycle} = require('../../static/dashboard-refresh.js');
const flush = async () => { for(let i=0;i<12;i++) await Promise.resolve(); };
function clock(){
  let time=0, seq=0; const tasks=new Map();
  return {now:()=>time, setTimeout:(fn,ms)=>{tasks.set(++seq,{fn,at:time+ms});return seq;},
    clearTimeout:id=>tasks.delete(id), async advance(ms=0){time+=ms;
      for(let i=0;i<30;i++){const due=[...tasks].filter(([,t])=>t.at<=time);if(!due.length)break;
        for(const [id,t] of due){tasks.delete(id);t.fn();} await flush();}}, tasks};
}
(async()=>{
  const c=clock();let resolve, count=0, active=true, applied=[];
  const r=createResource({...c,active:()=>active,minInterval:100,interval:1000,
    load:()=>{count++;return new Promise(r=>resolve=r);},apply:d=>applied.push(d)});
  r.request();await c.advance();assert.equal(count,1);
  r.request();r.request();resolve(1);await flush();await c.advance(100);
  assert.equal(count,2,'in-flight signals coalesce into one follow-up');
  resolve(2);await flush();assert.deepEqual(applied,[1,2]);
  await c.advance(1000);assert.equal(count,3,'periodic refresh even without invalidation');
  active=false;r.suspend();resolve(3);await flush();assert.deepEqual(applied,[1,2],'late hidden response discarded');
  await c.advance(10000);assert.equal(count,3,'no hidden traffic');
  active=true;r.request();await c.advance();assert.equal(count,4,'resume immediately');
  resolve(4);await flush();r.suspend();

  const c2=clock();let attempts=0, rendered=0;const states=[];
  const fail=createResource({...c2,active:()=>true,minInterval:0,interval:10000,
    load:async()=>++attempts,apply:()=>{if(++rendered===1)throw Error('render');},onState:s=>states.push(s)});
  fail.request();await c2.advance();assert.equal(states.at(-1),'stale');
  await c2.advance(2000);assert.equal(attempts,2,'render failures retry without new SSE');
  assert.equal(states.at(-1),'ready');fail.suspend();

  const c3=clock();let aborted=false;
  const timed=createResource({...c3,active:()=>true,timeout:50,
    load:signal=>new Promise((_,reject)=>signal.addEventListener('abort',()=>{aborted=true;reject(Error('aborted'));})),
    apply:()=>assert.fail('timed out response applied'),onState:s=>states.push(s)});
  timed.request();await c3.advance();await c3.advance(50);assert.ok(aborted);
  assert.equal(states.at(-1),'stale','timeout is a visible retryable failure');timed.suspend();

  function target(){const listeners={};return {addEventListener:(k,fn)=>listeners[k]=fn,emit:k=>listeners[k]?.(),listeners};}
  const doc={...target(),hidden:false};const win={...target(),navigator:{onLine:true}};
  let streamCount=0, closed=0;
  win.EventSource=function(url){assert.equal(url,'/api/stats/stream');streamCount++;this.close=()=>closed++;};
  const life=createLifecycle({document:doc,window:win,...clock()});
  life.start();assert.equal(streamCount,1);doc.hidden=true;doc.emit('visibilitychange');assert.equal(closed,1);
  win.emit('offline');doc.hidden=false;win.navigator.onLine=false;doc.emit('visibilitychange');assert.equal(streamCount,1);
  win.navigator.onLine=true;win.emit('online');assert.equal(streamCount,2);
  win.emit('pagehide');assert.equal(closed,2);win.emit('pageshow');assert.equal(streamCount,3);life.stop();
  console.log('resource coalescing, periodic freshness, suspension, late responses, retry, timeout, lifecycle: passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
