/* ===========================================================================
   Cinematic hero for index_black.html
   ---------------------------------------------------------------------------
   Two bands of this project's own generated clips scroll across the full width
   behind the title, looping left to right, with a particle field behind them
   and an ambient bed the top bar can switch on and off.

   The clip list is a fixed array, so the wall is one small request and never
   waits on the manifest.

   Three rules keep it cheap and safe:

     * A tile shows its poster until the clip is actually playable. The video
       element is created in the same pass, so nothing waits on the network
       before the wall appears.
     * At most MAX_PLAY clips decode at once. They are handed out in turn, so
       the wall keeps moving rather than one band hogging the decoder.
     * Everything stops when the hero scrolls out of view or the tab is hidden.

   With motion reduced, the bands stand still, no clip is ever decoded, the
   field is not drawn, and the title is untouched.
   =========================================================================== */
(function () {
  "use strict";

  /* Copies of each row's run. Three is the minimum that stays seamless on a
     very wide screen: the shift is 100/COPIES %, and the track has to remain
     wider than the viewport at every offset. */
  var COPIES = 3;

  var stage  = document.getElementById("cine-stage");
  var wall   = document.getElementById("cine-wall");
  var canvas = document.getElementById("cine-canvas");
  if (!stage || !wall) return;

  var reduce = window.matchMedia &&
               window.matchMedia("(prefers-reduced-motion: reduce)").matches;


  /* The wall's own clips, beside them; see gen_assets.py. */
  var CLIPS = [
    { src: "static/videos/hero/hero_00.mp4?v=4dcd4ff6", poster: "static/images/hero/hero_00.jpg?v=7f0d48a7" },
    { src: "static/videos/hero/hero_01.mp4?v=0cd30fc0", poster: "static/images/hero/hero_01.jpg?v=1eabe959" },
    { src: "static/videos/hero/hero_02.mp4?v=2aa4789f", poster: "static/images/hero/hero_02.jpg?v=569e0378" },
    { src: "static/videos/hero/hero_03.mp4?v=0ee6bd00", poster: "static/images/hero/hero_03.jpg?v=232fcc3b" },
    { src: "static/videos/hero/hero_04.mp4?v=b1e27515", poster: "static/images/hero/hero_04.jpg?v=a7dc1593" },
    { src: "static/videos/hero/hero_05.mp4?v=1f6181f8", poster: "static/images/hero/hero_05.jpg?v=ba5bf420" },
    { src: "static/videos/hero/hero_06.mp4?v=f3906e5a", poster: "static/images/hero/hero_06.jpg?v=ab7f985e" },
    { src: "static/videos/hero/hero_07.mp4?v=414deee1", poster: "static/images/hero/hero_07.jpg?v=9141d186" },
    { src: "static/videos/hero/hero_08.mp4?v=c3c6d72a", poster: "static/images/hero/hero_08.jpg?v=9aa7d907" },
    { src: "static/videos/hero/hero_09.mp4?v=0316b2e8", poster: "static/images/hero/hero_09.jpg?v=bcbd1da0" },
    { src: "static/videos/hero/hero_10.mp4?v=3c3c0e25", poster: "static/images/hero/hero_10.jpg?v=ced20816" },
    { src: "static/videos/hero/hero_11.mp4?v=fc2f8d33", poster: "static/images/hero/hero_11.jpg?v=129374dd" },
    { src: "static/videos/hero/hero_12.mp4?v=9ef93efe", poster: "static/images/hero/hero_12.jpg?v=156c719a" },
    { src: "static/videos/hero/hero_13.mp4?v=aaa46589", poster: "static/images/hero/hero_13.jpg?v=9768daf1" },
    { src: "static/videos/hero/hero_14.mp4?v=040aec30", poster: "static/images/hero/hero_14.jpg?v=801359bf" },
  ];

  /* -------------------------------------------------------------------------
     1. Build the two bands

     A band holds three identical copies of its run. One copy plus its trailing
     gap is exactly a third of the track, so a shift of -33.3333% lands on an
     identical frame: the loop has no seam at any width.
     ---------------------------------------------------------------------- */
  function tile(c, i) {
    var t = document.createElement("div");
    t.className = "cine-tile";
    t.style.setProperty("--d", (0.06 + i * 0.05).toFixed(3) + "s");

    var img = document.createElement("img");
    img.src = c.poster;
    img.alt = "";
    img.decoding = "async";
    img.loading = "eager";
    t.appendChild(img);

    /* the clip is created now but stays behind the poster until it can play,
       so a slow one never shows an empty frame */
    var v = document.createElement("video");
    v.src = c.src;
    v.poster = c.poster;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.setAttribute("playsinline", "");
    v.setAttribute("muted", "");
    v.preload = "auto";
    v.addEventListener("loadeddata", function () { t.classList.add("is-live"); });
    v.addEventListener("playing", function () { t.classList.add("is-live"); });
    t.appendChild(v);

    t._video = v;
    return t;
  }

  function build() {
    if (!CLIPS.length) return [];
    var made = [];
    /* Three rows dividing the header height, five clips each. ROWS carries the
       direction and speed of each: the row the eye lands on first runs the
       fastest, and no two rows share a speed. */
    var rows = [
      { cls: "is-row-a", run: CLIPS.slice(0, 5) },
      { cls: "is-row-b", run: CLIPS.slice(5, 10) },
      { cls: "is-row-c", run: CLIPS.slice(10, 15) }
    ].filter(function (r) { return r.run.length; });

    rows.forEach(function (r, b) {
      var run = r.run;
      var band = document.createElement("div");
      band.className = "cine-band " + r.cls;
      var track = document.createElement("div");
      track.className = "cine-track";
      for (var copy = 0; copy < COPIES; copy++) {
        var c = document.createElement("div");
        c.className = "cine-copy";
        run.forEach(function (clip, i) {
          var t = tile(clip, b * 3 + i);
          c.appendChild(t);
          made.push(t);
        });
        track.appendChild(c);
      }
      band.appendChild(track);
      wall.appendChild(band);
    });
    return made;
  }

  var tiles = build();

  /* -------------------------------------------------------------------------
     2. Decoding budget

     Round-robin over the tiles, topping up to MAX_PLAY alive at once.
     ---------------------------------------------------------------------- */
  /* Start every tile, in waves rather than all at once, so the browser is not
     asked to begin 27 decodes on a single frame. */
  var paused = false, order = null, rampT = null, watchT = null;

  /* Which tile to start next. Centre-out, not DOM order: at load only about a
     third of each row is on screen, and those are the tiles the eye is on, so
     they must not queue behind the ones off the edge. */
  function startOrder() {
    if (order) return order;
    var vw = window.innerWidth || 1440;
    function mid(t) {
      var r = t.getBoundingClientRect ? t.getBoundingClientRect() : null;
      return r ? Math.abs((r.left + r.right) / 2 - vw / 2) : 0;
    }
    order = tiles.slice().sort(function (a, b) { return mid(a) - mid(b); });
    return order;
  }

  /* Start up to n tiles that are not already running. Returns how many went. */
  function startNext(n) {
    var ord = startOrder(), started = 0;
    for (var i = 0; i < ord.length && started < n; i++) {
      var t = ord[i];
      if (!t._video || !t._video.paused) continue;
      var p = t._video.play();
      if (p && p.catch) p.catch(function () {});
      started++;
    }
    return started;
  }

  /* Bring the wall up a batch at a time.
     Asking all forty-five tiles to decode on one frame is what makes the first
     second stutter: the clips in view wait on tiles that are off the edge. The
     batch is a sixth of the wall, so the eye's tiles get the decoder to
     themselves and the rest fill in behind them. */
  function rampUp() {
    if (reduce || !tiles.length) return;
    if (rampT) clearInterval(rampT);
    rampT = setInterval(function () {
      var batch = Math.max(3, Math.round(tiles.length / 6));
      if (startNext(batch) === 0) {
        clearInterval(rampT); rampT = null;
        /* the wall is up; keep watch from here, so a clip the browser parks is
           restarted rather than left frozen */
        if (watchT === null) watchT = setInterval(tick, 2500);
      }
    }, 160);
  }

  /* Anything that has stalled or been paused by the browser is picked up here,
     so no tile is left sitting on a still frame. */
  function tick() {
    if (paused || !tiles.length || reduce) return;
    var stalled = 0, i;
    for (i = 0; i < tiles.length; i++) {
      if (tiles[i]._video && tiles[i]._video.paused) stalled++;
    }
    if (stalled) startNext(stalled);
  }

  function pauseAll() {
    paused = true;
    tiles.forEach(function (t) { if (t._video) t._video.pause(); });
  }
  function resumeAll() { paused = false; rampUp(); }

  rampUp();

  /* -------------------------------------------------------------------------
     3. Particle field, behind the wall
     ---------------------------------------------------------------------- */
  function field() {
    if (!canvas || reduce) return null;
    var ctx = canvas.getContext && canvas.getContext("2d");
    if (!ctx) return null;

    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = 0, H = 0, dots = [], raf = null, alive = true;
    var N = 110;

    function size() {
      W = canvas.clientWidth || window.innerWidth;
      H = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      dots = [];
      for (var i = 0; i < N; i++) {
        dots.push({ x: Math.random() * W, y: Math.random() * H,
                    z: 0.22 + Math.random() * 0.78,
                    vx: (Math.random() - 0.5) * 0.14,
                    vy: -0.04 - Math.random() * 0.12 });
      }
    }

    function draw() {
      raf = null;
      if (!alive) return;
      ctx.clearRect(0, 0, W, H);
      for (var i = 0; i < dots.length; i++) {
        var p = dots[i];
        p.x += p.vx * p.z;
        p.y += p.vy * p.z;
        if (p.y < -4) { p.y = H + 4; p.x = Math.random() * W; }
        if (p.x < -4) p.x = W + 4; else if (p.x > W + 4) p.x = -4;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.z * 1.5, 0, 6.2832);
        ctx.fillStyle = "rgba(214,228,255," + (0.05 + p.z * 0.30).toFixed(3) + ")";
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    }

    size();
    raf = requestAnimationFrame(draw);

    var rt = null;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(size, 180);
    });
    return {
      stop: function () { alive = false; if (raf) cancelAnimationFrame(raf); },
      go: function () { if (!alive) { alive = true; raf = requestAnimationFrame(draw); } }
    };
  }

  var f = field();

  /* reveal the wall on the frame after layout, so the transition runs */
  requestAnimationFrame(function () {
    requestAnimationFrame(function () { stage.classList.add("is-lit"); });
  });

  /* -------------------------------------------------------------------------
     4. Stop when the hero is off screen
     ---------------------------------------------------------------------- */
  function setActive(on) {
    if (on) { resumeAll(); if (f) f.go(); }
    else { pauseAll(); if (f) f.stop(); }
  }

  if (!reduce) {
    if ("IntersectionObserver" in window) {
      var onScreen = true;
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (e.isIntersecting === onScreen) return;
          onScreen = e.isIntersecting;
          setActive(onScreen);
        });
      }, { threshold: 0.02 }).observe(stage);
    }
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) { pauseAll(); if (f) f.stop(); }
      else { setActive(true); }
    });
  }

  /* the scroll cue fades as soon as the page moves */
  var cue = document.getElementById("cine-scroll");
  if (cue) {
    window.addEventListener("scroll", function () {
      cue.classList.toggle("is-gone", window.scrollY > 40);
    }, { passive: true });
  }

  /* -------------------------------------------------------------------------
     5. Ambient bed
     ------------------------------------------------------------------------
     Browsers refuse unmuted playback without a gesture, so this never tries to
     play on load: the button is the gesture. The choice is remembered, and a
     page opened with music already enabled starts it on the first interaction
     anywhere, which is the earliest moment the browser will allow.
     ---------------------------------------------------------------------- */
  var toggle = document.getElementById("sound-toggle");
  /* Look the element up when the document is ready, not on the line the script
     happens to run: an <audio> placed after this script is not in the DOM yet,
     and a null lookup here is what left the music button doing nothing. */
  var audio = (function () {
    var a = document.getElementById("page-audio");
    if (!a) {
      document.addEventListener("DOMContentLoaded", function () {
        a = document.getElementById("page-audio");
      });
    }
    return a;
  })();
  if (toggle && audio) {
    var KEY = "uniavc-music";
    audio.volume = 0;
    audio.loop = true;

    function paint(on) {
      toggle.classList.toggle("is-on", on);
      toggle.setAttribute("aria-pressed", on ? "true" : "false");
      var label = toggle.querySelector(".sound-label");
      if (label) label.textContent = on ? "Music on" : "Music off";
      toggle.setAttribute("aria-label",
        on ? "Turn background music off" : "Turn background music on");
    }

    /* the bed fades rather than cutting, so toggling never clicks */
    var fade = null;
    function ramp(to, then) {
      if (fade) clearInterval(fade);
      var from = audio.volume, steps = 24, n = 0;
      fade = setInterval(function () {
        n++;
        audio.volume = Math.max(0, Math.min(1, from + (to - from) * (n / steps)));
        if (n >= steps) { clearInterval(fade); fade = null; if (then) then(); }
      }, 24);
    }

    function start() {
      var p = audio.play();
      if (p && p.catch) p.catch(function () {});
      ramp(0.55);
      paint(true);
      try { localStorage.setItem(KEY, "1"); } catch (e) {}
    }
    function stop() {
      ramp(0, function () { audio.pause(); });
      paint(false);
      try { localStorage.setItem(KEY, "0"); } catch (e) {}
    }

    var on = false;
    try { on = localStorage.getItem(KEY) === "1"; } catch (e) {}
    paint(on);

    toggle.addEventListener("click", function (e) {
      e.preventDefault();
      if (toggle.classList.contains("is-on")) stop(); else start();
    });

    /* if the page was opened with music remembered as on, the first gesture
       anywhere is the earliest the browser will let it start */
    if (on) {
      var kick = function () {
        document.removeEventListener("pointerdown", kick, true);
        document.removeEventListener("keydown", kick, true);
        if (toggle.classList.contains("is-on")) return;
        start();
      };
      document.addEventListener("pointerdown", kick, true);
      document.addEventListener("keydown", kick, true);
    }
  }
})();
