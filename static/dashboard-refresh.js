/* Shared dashboard freshness lifecycle. Each feed has one bounded request and
 * one dirty bit: bursts coalesce, but a change during a fetch is never lost. */
(function(root, factory){
  if(typeof module === 'object' && module.exports) module.exports = factory();
  else root.TokenfoldRefresh = factory();
})(typeof window === 'undefined' ? globalThis : window, function(){
  'use strict';
  function createResource(options){
    var clock = options.now || Date.now;
    var later = options.setTimeout || setTimeout, cancel = options.clearTimeout || clearTimeout;
    var timer = null, running = null, pending = false, generation = 0;
    var lastStart = -Infinity, failures = 0;
    var minimum = options.minInterval == null ? 1000 : options.minInterval;
    function state(value){ if(options.onState) options.onState(value); }
    function schedule(delay){
      if(!options.active() || running) return;
      if(timer !== null) cancel(timer);
      timer = later(run, Math.max(delay || 0, minimum - (clock() - lastStart)));
    }
    function run(){
      timer = null;
      if(!options.active() || running) return;
      pending = false;
      var token = {generation:++generation, controller:new AbortController()};
      running = token; lastStart = clock();
      // The timeout also bounds helpers which forget to honour AbortSignal.
      var timeout;
      var deadline = new Promise(function(_, reject){
        timeout = later(function(){ token.controller.abort(); reject(new Error('request timed out')); }, options.timeout || 30000);
      });
      Promise.race([Promise.resolve().then(function(){
        if(running !== token || !options.active()) return;
        return options.load(token.controller.signal);
      }), deadline])
        .then(function(value){
          if(running !== token || !options.active()) return;
          return Promise.resolve(options.apply(value)).then(function(){
            if(running !== token || !options.active()) return;
            failures = 0; state('ready');
          });
        })
        .catch(function(){
          if(running !== token || !options.active()) return;
          failures++; pending = true; state('stale');
        })
        .finally(function(){
          cancel(timeout);
          if(running !== token) return;
          running = null;
          if(pending) schedule(failures ? Math.min(30000, 1000 * Math.pow(2, failures)) : 0);
          else if(options.interval) schedule(options.interval);
        });
    }
    return {
      request:function(immediate){
        if(immediate) lastStart = -Infinity;
        pending = true; schedule(0);
      },
      suspend:function(){
        if(timer !== null) cancel(timer); timer = null;
        if(running) running.controller.abort();
        running = null; generation++; pending = true; lastStart = -Infinity;
      }
    };
  }

  function createLifecycle(options){
    var doc = options.document, win = options.window;
    var resources = {}, errors = {}, started = false, paused = false;
    var stream = null, retry = null;
    var later = options.setTimeout || setTimeout, cancel = options.clearTimeout || clearTimeout;
    function active(){ return started && !paused && !doc.hidden && win.navigator.onLine !== false; }
    function report(){ if(options.onState) options.onState(win.navigator.onLine === false ? 'offline' : Object.keys(errors).length ? 'stale' : 'ready', Object.keys(errors)); }
    function request(name, immediate){ if(resources[name]) resources[name].request(immediate); }
    function stopStream(){
      if(stream){ stream.close(); stream = null; }
      if(retry !== null){ cancel(retry); retry = null; }
    }
    function startStream(){
      stopStream();
      if(!active() || typeof win.EventSource === 'undefined') return;
      var current = new win.EventSource('/api/stats/stream'); stream = current;
      current.onmessage = function(event){
        if(stream !== current || !active()) return;
        try { if(options.onVersion) options.onVersion(JSON.parse(event.data).version); } catch(_) {}
      };
      current.onerror = function(){
        if(stream !== current) return;
        stopStream(); request('version');
        // A persistent 30s cheap version watchdog also covers a silently stuck
        // proxy stream, not just EventSource's explicit error callback.
        if(active()) retry = later(startStream, 60000);
      };
    }
    function resume(){
      if(!active()){ stopStream(); Object.values(resources).forEach(function(r){r.suspend();}); report(); return; }
      Object.keys(resources).forEach(function(name){ request(name); });
      if(!stream) startStream(); report();
    }
    doc.addEventListener('visibilitychange', resume);
    win.addEventListener('online', resume);
    win.addEventListener('offline', resume);
    win.addEventListener('pagehide', function(){ paused = true; resume(); });
    win.addEventListener('pageshow', function(){ paused = false; resume(); });
    return {
      active:active,
      add:function(name, config){
        resources[name] = createResource(Object.assign({}, options, config, {active:active, onState:function(value){
          if(value === 'stale') errors[name] = true; else delete errors[name];
          if(config.onState) config.onState(value);
          report();
        }}));
        if(active()) request(name);
        return resources[name];
      },
      request:request,
      start:function(){ started = true; resume(); },
      stop:function(){ started = false; resume(); }
    };
  }
  function fetchJSON(url, signal, options){
    return fetch(url, Object.assign({cache:'no-store'}, options, {signal:signal})).then(function(response){
      if(!response.ok) throw new Error('http ' + response.status);
      return response.json();
    });
  }
  return {createResource:createResource, createLifecycle:createLifecycle, fetchJSON:fetchJSON};
});
