// Enhjørningen min: keeps a copy of the game on the device, so it also starts without internet.
// The newest version is always fetched first; the saved copy is only used when there is no connection.
var CACHE = "enhjorning-app-1";
var FILES = ["./", "./index.html", "./three.min.js", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", function(e){
  e.waitUntil(caches.open(CACHE).then(function(c){ return c.addAll(FILES); }).then(function(){ return self.skipWaiting(); }));
});
self.addEventListener("activate", function(e){
  e.waitUntil(caches.keys().then(function(keys){
    return Promise.all(keys.filter(function(k){ return k !== CACHE; }).map(function(k){ return caches.delete(k); }));
  }).then(function(){ return self.clients.claim(); }));
});
self.addEventListener("fetch", function(e){
  if (e.request.method !== "GET") return;
  e.respondWith(
    fetch(e.request).then(function(res){
      // save what was fetched (the game's own files and the font), for next time without a connection
      if (res && (res.ok || res.type === "opaque")) {
        var copy = res.clone();
        caches.open(CACHE).then(function(c){ c.put(e.request, copy); });
      }
      return res;
    }).catch(function(){
      return caches.match(e.request, { ignoreSearch: true }).then(function(hit){ return hit || caches.match("./index.html"); });
    })
  );
});
