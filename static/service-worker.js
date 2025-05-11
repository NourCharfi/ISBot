const CACHE_NAME = 'isbot-cache-v1';
const urlsToCache = [
  '/',
  '/static/style.css',
  '/static/logoX.PNG',
  '/static/assistant-avatar.png',
  '/static/manifest.json',
  // Add more static assets if needed
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
}); 