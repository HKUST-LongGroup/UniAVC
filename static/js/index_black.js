/* ===========================================================================
   UniAVC project page — comparison, demo and generation galleries
   ---------------------------------------------------------------------------
   Data comes from window.UNIAVC_DATA (static/js/manifest.js).

   Four widgets; every clip is a sounding video (one .mp4 carries both the
   picture and the sound):

   1. Comparison — four method panels per sample, held FRAME-LOCKED to a shared
                   transport so a difference is attributable to the method and
                   not to playback drift. Focus reflows them to 2x2 and widens
                   the column; arrow keys change sample.

   2. Demos      — one card per source clip: the source beside the outputs
                   answering it, each labelled with its verbatim instruction.
                   Its own Play / Loop transport runs a whole family at once.

   3. Generation — a tile grid with click-to-play and a Focus button.

   4. Inference  — the same request under each operating mode.

   Three page-wide rules:

     * PLAYBACK IS CLICK-ONLY. Hovering never starts a clip, so sweeping the
       mouse across a gallery cannot set a row of them going by accident.
       Only a click on a panel, or the transport's Play button, starts a clip.
     * At most one clip is ever UNMUTED.
     * The first click or keypress unlocks audio for the rest of the session,
       because browsers refuse unmuted playback without a user gesture.
   =========================================================================== */
(function () {
  "use strict";

  var DATA = window.UNIAVC_DATA || {};

  /* Audio is locked until the user interacts; see safePlay(). */
  var audioUnlocked = false;
  /* The single unmuted cell/tile, page-wide. */
  var audibleCell = null;

  function unlock() {
    if (audioUnlocked) return;
    audioUnlocked = true;
    // The current audio target can now actually be heard; update its icon.
    if (audibleCell) {
      var ic = iconOf(audibleCell);
      if (ic) ic.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
  }
  document.addEventListener("click", unlock, true);
  document.addEventListener("keydown", unlock, true);
  document.addEventListener("touchstart", unlock, true);

  /* Play a clip, honouring the audio-unlock rule.
     Returns whether it ended up audible. */
  function safePlay(v, wantSound) {
    var sound = !!wantSound && audioUnlocked;
    v.muted = !sound;
    var p = v.play();
    if (p && p.catch) {
      p.catch(function () {
        // Autoplay with sound was refused: fall back to a muted start so the
        // clip still moves rather than sitting frozen.
        v.muted = true;
        var q = v.play();
        if (q && q.catch) q.catch(function () {});
      });
    }
    return sound;
  }

  /* Every clip must decode a first frame without playing, or a panel shows an
     empty box until it is first clicked.

     The frame it parks on is the POSTER's own timestamp, not zero. The
     filmstrip tiles are the posters, so if the panel sat at 0.001s while the
     tile showed the 0.5s frame, the selected sample and the sample on screen
     were visibly different moments of the same clip.

     Parking happens once, on the first decode. `loadeddata` fires again on
     every load, and pausing there would stop the clip the moment Play started
     it. */
  var POSTER_T = 0.5;

  function prime(video) {
    var parked = false;
    video.addEventListener("loadedmetadata", function () {
      video.muted = true;
      try { video.currentTime = POSTER_T; } catch (e) {}
    });
    video.addEventListener("loadeddata", function () {
      if (parked) return;                       // park on first decode only
      parked = true;
      try { video.currentTime = POSTER_T; } catch (e) {}
      video.pause();
    });
  }

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  /* Focus widens the page column so the reflowed grid is actually larger.
     Every media section shares the column, so this only widens while at least
     one of them is focused. */
  function widenColumn(node, on) {
    var col = node && node.closest ? node.closest(".container") : null;
    if (col) col.classList.toggle("is-focus-wide", on);
  }

  function iconOf(node) {
    return node ? node.querySelector(".cmp-audio, .cell-audio, .sound-icon") : null;
  }

  /* Mute everything inside `node` and clear its audible styling. */
  function hush(node) {
    if (!node) return;
    node.querySelectorAll("video").forEach(function (v) { v.muted = true; });
    node.classList.remove("is-audible", "is-audio");
    var ic = iconOf(node);
    if (ic) ic.innerHTML = '<i class="fas fa-volume-mute"></i>';
  }

  /* Make `cell` the page's audio target.
     The icon shows what will actually be heard: while audio is still locked
     (no click yet) a clip plays muted, so claiming otherwise would be a lie. */
  function beAudible(cell) {
    if (audibleCell && audibleCell !== cell) hush(audibleCell);
    cell.querySelectorAll("video").forEach(function (v) {
      v.muted = !audioUnlocked;                 // resolved properly on play
    });
    cell.classList.add("is-audible");
    var ic = iconOf(cell);
    if (ic) {
      ic.innerHTML = audioUnlocked
        ? '<i class="fas fa-volume-up"></i>'
        : '<i class="fas fa-volume-mute"></i>';
    }
    audibleCell = cell;
  }

  /* Clicking a panel that is already the audio target turns the sound off
     rather than leaving a lit panel that is silent. */
  function toggleAudible(cell) {
    if (audibleCell === cell) {
      hush(cell);
      audibleCell = null;
      return false;
    }
    beAudible(cell);
    return true;
  }

  /* ======================================================================
     1. COMPARISON
     ====================================================================== */
  var METHODS = [
    { key: "input",   label: "Input",         baseline: false },
    { key: "ours",    label: "UniAVC (Ours)", baseline: false, ours: true },
    { key: "ia2av",   label: "InstructAV2AV", baseline: true },
    { key: "minimax", label: "MiniMax-H3",    baseline: true }
  ];

  var CMP = {
    items: DATA.comparison || [],
    byFamily: {},
    fams: [],
    tab: 0,
    index: 0,
    videos: [],
    focus: false,
    loop: true,
    dragging: false,
    raf: null
  };

  function buildComparison() {
    if (!CMP.items.length) return;
    CMP.items.forEach(function (it) {
      if (!CMP.byFamily[it.family]) {
        CMP.byFamily[it.family] = [];
        CMP.fams.push(it.family);
      }
      CMP.byFamily[it.family].push(it);
    });

    var ul = document.querySelector("#cmp-tabs ul");
    CMP.fams.forEach(function (fam, i) {
      var li = el("li");
      li.appendChild(el("a", null, "<span>" + fam + "</span>"));
      li.addEventListener("click", function () { showFamily(i); });
      ul.appendChild(li);
    });

    CMP.stage = document.getElementById("cmp-stage");
    CMP.strip = document.getElementById("cmp-strip");
    // the bordered block that gives this section the same boundary a demo
    // card has; the prompt and the transport go inside it too
    var block = CMP.stage.parentNode;

    // The instruction is the point of a sample, so it is stated once, plainly,
    // above the panels rather than repeated under every clip, and above the
    // controls so a reader sees what is being asked before touching anything.
    CMP.prompt = el("div", "cmp-prompt");
    CMP.prompt.setAttribute("id", "cmp-prompt");

    // Same organisation as the demo and inference sections: the instruction,
    // then a centred row of Play / Loop / Focus, then the time progress bar,
    // then the clips. The sample counter moves into the instruction line
    // rather than trailing the buttons.
    var transport = el("div", "cmp-transport aligned-center");
    transport.innerHTML =
      '<button class="btn" id="cmp-play"><i class="fas fa-play"></i> Play</button>' +
      '<button class="btn is-on" id="cmp-loop"><i class="fas fa-repeat"></i> Loop</button>' +
      '<button class="btn" id="cmp-focus"><i class="fas fa-expand"></i> Focus</button>';

    var prog = el("div", "cmp-progress");
    prog.innerHTML =
      '<input class="scrub" id="cmp-scrub" type="range" min="0" max="1000" value="0"' +
      ' aria-label="Scrub all panels">' +
      '<span class="timecode" id="cmp-time">0.00s</span>';

    block.insertBefore(prog, CMP.stage);
    block.insertBefore(transport, prog);
    block.insertBefore(CMP.prompt, transport);

    CMP.scrub = document.getElementById("cmp-scrub");
    CMP.timeEl = document.getElementById("cmp-time");
    CMP.playBtn = document.getElementById("cmp-play");

    CMP.playBtn.addEventListener("click", function () { togglePlay(); });

    // Focus keeps all four panels and reflows them to 2x2. The column is also
    // widened, so each panel genuinely renders larger instead of being
    // rearranged into the same width.
    document.getElementById("cmp-focus").addEventListener("click", function () {
      CMP.focus = !CMP.focus;
      CMP.stage.classList.toggle("is-focus", CMP.focus);
      this.classList.toggle("is-on", CMP.focus);
      this.innerHTML = CMP.focus
        ? '<i class="fas fa-compress"></i> Unfocus'
        : '<i class="fas fa-expand"></i> Focus';
      widenColumn(CMP.stage, CMP.focus);
    });

    var loopBtn = document.getElementById("cmp-loop");
    loopBtn.addEventListener("click", function () {
      CMP.loop = !CMP.loop;
      loopBtn.classList.toggle("is-on", CMP.loop);
      CMP.videos.forEach(function (v) { v.loop = CMP.loop; });
    });

    CMP.scrub.addEventListener("input", function () {
      CMP.dragging = true;
      var v = CMP.videos[0];
      if (!v || !v.duration) return;
      var t = (CMP.scrub.value / 1000) * v.duration;
      CMP.videos.forEach(function (x) { try { x.currentTime = t; } catch (e) {} });
      CMP.timeEl.textContent = t.toFixed(2) + "s";
    });
    CMP.scrub.addEventListener("change", function () { CMP.dragging = false; });

    document.addEventListener("keydown", function (e) {
      var ae = document.activeElement;
      if (ae && /^(INPUT|TEXTAREA)$/.test(ae.tagName)) return;
      if (!isFocusedSection(CMP.stage)) return;
      if (e.key === "ArrowRight") { showIndex(CMP.index + 1); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { showIndex(CMP.index - 1); e.preventDefault(); }
      else if (e.key === " " || e.key === "k") { togglePlay(); e.preventDefault(); }
    });

    showFamily(0);
  }

  function showFamily(i) {
    CMP.tab = i;
    document.querySelectorAll("#cmp-tabs li").forEach(function (li, k) {
      li.classList.toggle("is-active", k === i);
    });
    var list = CMP.byFamily[CMP.fams[i]];
    CMP.strip.innerHTML = "";
    list.forEach(function (it, k) {
      var b = el("button");
      b.title = it.family + " sample " + it.index;
      var img = el("img");
      img.src = it.input.poster;
      img.alt = it.family + " sample " + it.index + " input frame";
      b.appendChild(img);
      b.appendChild(el("span", "strip-num", String(it.index)));
      b.addEventListener("click", function () { showIndex(k); });
      CMP.strip.appendChild(b);
    });
    showIndex(0);
  }

  function showIndex(i) {
    var list = CMP.byFamily[CMP.fams[CMP.tab]];
    if (!list || !list.length) return;
    i = ((i % list.length) + list.length) % list.length;
    CMP.index = i;
    var item = list[i];

    stopTransport();
    CMP.stage.innerHTML = "";
    CMP.videos = [];
    audibleCell = null;

    METHODS.forEach(function (m) {
      var d = item[m.key];
      if (!d) return;
      var cell = el("div", "cmp-cell" + (m.baseline ? " is-baseline" : "") +
                          (m.ours ? " is-ours" : ""));
      var v = el("video");
      v.src = d.src;
      v.poster = d.poster;
      v.muted = true;
      v.playsInline = true;
      v.preload = "metadata";
      v.loop = CMP.loop;
      cell.appendChild(v);
      cell.appendChild(el("span", "cmp-tag", m.label));
      cell.appendChild(el("span", "cmp-audio",
                          '<i class="fas fa-volume-mute"></i>'));
      prime(v);
      CMP.videos.push(v);

      cell.addEventListener("click", function (e) {
        e.stopPropagation();
        toggleAudible(cell);
        // clicking a panel also starts the transport, so the routed audio is
        // actually heard rather than just selected
        if (CMP.videos[0] && CMP.videos[0].paused) playAll();
      });
      CMP.stage.appendChild(cell);
    });

    // The paper colours each prompt by role; use that markup when the sample
    // has it, and fall back to the plain sentence otherwise so a sample added
    // later still reads correctly rather than losing its instruction.
    CMP.prompt.innerHTML =
      '<div class="who">' + item.family + " &middot; sample " + (i + 1) + " / " +
      list.length + "</div>" +
      '<div class="what">' + (item.prompt_html || item.prompt) + "</div>";

    // the panel under test starts holding the audio
    var ours = CMP.stage.querySelector(".cmp-cell.is-ours");
    if (ours) beAudible(ours);

    CMP.strip.querySelectorAll("button").forEach(function (b, k) {
      b.classList.toggle("is-current", k === i);
    });
    syncTime();
  }

  function syncTime() {
    var v = CMP.videos[0];
    if (!v) return;
    if (!CMP.dragging && v.duration) {
      CMP.scrub.value = (v.currentTime / v.duration) * 1000;
      CMP.timeEl.textContent = v.currentTime.toFixed(2) + "s";
    }
    CMP.raf = requestAnimationFrame(syncTime);
  }

  function playAll() {
    var voice = CMP.stage.querySelector(".cmp-cell.is-audible") ||
                CMP.stage.querySelector(".cmp-cell.is-ours");
    CMP.videos.forEach(function (v) {
      safePlay(v, voice && voice.contains(v));
    });
    CMP.playBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
    if (!CMP.raf) syncTime();
  }

  function stopTransport() {
    if (CMP.raf) { cancelAnimationFrame(CMP.raf); CMP.raf = null; }
    CMP.videos.forEach(function (v) {
      v.pause();
      try { v.currentTime = 0; } catch (e) {}
    });
    if (CMP.playBtn) CMP.playBtn.innerHTML = '<i class="fas fa-play"></i> Play';
  }

  function togglePlay() {
    var v = CMP.videos[0];
    if (!v) return;
    if (v.paused) playAll();
    else stopTransport();
  }

  /* ======================================================================
     2. VERSATILITY DEMOS
     ----------------------------------------------------------------------
     One sample at a time, stepped with a filmstrip exactly as the comparison
     section is, so the two read the same way. The transport runs the clips of
     the sample on screen together.
     ====================================================================== */
  var DEMO = {
    decks: [], cards: [], deck: 0, index: 0,
    focus: false, loop: true, dragging: false, raf: null, videos: []
  };

  function buildDemos() {
    var raw = DATA.demo || {};
    Object.keys(raw).forEach(function (k) {
      var d = raw[k];
      DEMO.decks.push({ key: k, label: d.label, cards: d.cards || [] });
    });
    if (!DEMO.decks.length) return;

    var ul = document.querySelector("#demo-tabs ul");
    DEMO.decks.forEach(function (d, i) {
      var li = el("li");
      li.appendChild(el("a", null, "<span>" + d.label + "</span>"));
      li.addEventListener("click", function () { showDeck(i); });
      ul.appendChild(li);
    });

    DEMO.panel = document.getElementById("demo-panel");
    DEMO.strip = document.getElementById("demo-strip");

    // the instruction, the transport and the clips share one bordered block,
    // and the filmstrip that steps between samples sits just below it
    var parent = DEMO.panel.parentNode;
    var block = el("div", "sample-block");
    parent.insertBefore(block, DEMO.panel);

    DEMO.prompt = el("div", "cmp-prompt");
    DEMO.prompt.setAttribute("id", "demo-prompt");

    var bar = el("div", "cmp-transport aligned-center");
    bar.innerHTML =
      '<button class="btn" id="demo-play"><i class="fas fa-play"></i> Play</button>' +
      '<button class="btn is-on" id="demo-loop"><i class="fas fa-repeat"></i> Loop</button>' +
      '<button class="btn" id="demo-focus"><i class="fas fa-expand"></i> Focus</button>';

    var prog = el("div", "cmp-progress");
    prog.innerHTML =
      '<input class="scrub" id="demo-scrub" type="range" min="0" max="1000" value="0"' +
      ' aria-label="Scrub this sample">' +
      '<span class="timecode" id="demo-time">0.00s</span>';

    block.appendChild(DEMO.prompt);
    block.appendChild(bar);
    block.appendChild(prog);
    block.appendChild(DEMO.panel);
    parent.insertBefore(DEMO.strip, block.nextSibling);

    DEMO.playBtn = document.getElementById("demo-play");
    DEMO.scrub = document.getElementById("demo-scrub");
    DEMO.timeEl = document.getElementById("demo-time");

    DEMO.playBtn.addEventListener("click", function () { toggleDemo(); });

    DEMO.scrub.addEventListener("input", function () {
      DEMO.dragging = true;
      var v = DEMO.videos[0];
      if (!v || !v.duration) return;
      var t = (DEMO.scrub.value / 1000) * v.duration;
      DEMO.videos.forEach(function (x) { try { x.currentTime = t; } catch (e) {} });
      DEMO.timeEl.textContent = t.toFixed(2) + "s";
    });
    DEMO.scrub.addEventListener("change", function () { DEMO.dragging = false; });

    var loopBtn = document.getElementById("demo-loop");
    loopBtn.addEventListener("click", function () {
      DEMO.loop = !DEMO.loop;
      loopBtn.classList.toggle("is-on", DEMO.loop);
      DEMO.videos.forEach(function (v) { v.loop = DEMO.loop; });
    });

    document.getElementById("demo-focus").addEventListener("click", function () {
      DEMO.focus = !DEMO.focus;
      DEMO.panel.querySelectorAll(".demo-row").forEach(function (r) {
        r.classList.toggle("is-focus", DEMO.focus);
      });
      this.classList.toggle("is-on", DEMO.focus);
      this.innerHTML = DEMO.focus
        ? '<i class="fas fa-compress"></i> Unfocus'
        : '<i class="fas fa-expand"></i> Focus';
      widenColumn(DEMO.panel, DEMO.focus);
    });

    // left / right step between samples, as they do in the comparison section
    document.addEventListener("keydown", function (e) {
      var ae = document.activeElement;
      if (ae && /^(INPUT|TEXTAREA)$/.test(ae.tagName)) return;
      if (!isFocusedSection(DEMO.panel)) return;
      if (e.key === "ArrowRight") showCard(DEMO.index + 1);
      else if (e.key === "ArrowLeft") showCard(DEMO.index - 1);
    });

    showDeck(0);
  }

  /* The Play button reports what the panel is actually doing, so a single
     panel click cannot leave the label claiming something untrue. */
  function syncPlayBtn(btn, panel) {
    if (!btn || !panel) return;
    var running = false;
    panel.querySelectorAll("video").forEach(function (v) {
      if (!v.paused) running = true;
    });
    btn.innerHTML = running
      ? '<i class="fas fa-pause"></i> Pause'
      : '<i class="fas fa-play"></i> Play';
  }

  /* Three sections now answer to the arrow keys, so a bare keypress would step
     all of them at once. Only the section whose block is closest to the middle
     of the viewport responds; the rest ignore the key. */
  function isFocusedSection(node) {
    if (!node) return false;
    var block = node.closest ? (node.closest(".sample-block") || node) : node;
    var r = block.getBoundingClientRect ? block.getBoundingClientRect() : null;
    if (!r) return true;                        // no layout engine (tests)
    var mid = window.innerHeight / 2;
    if (r.top > mid || r.bottom < mid) return false;
    // visible AND nearer the middle of the screen than the other sections
    var best = null, bestD = Infinity;
    document.querySelectorAll(".sample-block").forEach(function (b) {
      var br = b.getBoundingClientRect();
      var d = Math.abs((br.top + br.bottom) / 2 - mid);
      if (d < bestD) { bestD = d; best = b; }
    });
    return best === block;
  }

  function toggleDemo() {
    if (!DEMO.videos.length) return;
    if (DEMO.videos[0].paused) {
      var voice = DEMO.panel.querySelector(".cell.is-audible") ||
                  DEMO.panel.querySelector(".cell.is-output") ||
                  DEMO.panel.querySelector(".cell");
      if (voice) beAudible(voice);
      DEMO.videos.forEach(function (v) {
        try { v.currentTime = 0; } catch (e) {}
        safePlay(v, voice && voice.contains(v));
      });
      if (!DEMO.raf) syncDemoTime();
    } else {
      stopDemo();
    }
    syncPlayBtn(DEMO.playBtn, DEMO.panel);
  }

  function stopDemo() {
    if (DEMO.raf) { cancelAnimationFrame(DEMO.raf); DEMO.raf = null; }
    DEMO.videos.forEach(function (v) {
      v.pause();
      try { v.currentTime = 0; } catch (e) {}
    });
    if (DEMO.playBtn) DEMO.playBtn.innerHTML = '<i class="fas fa-play"></i> Play';
    if (DEMO.timeEl) DEMO.timeEl.textContent = "0.00s";
    if (DEMO.scrub) DEMO.scrub.value = 0;
  }

  function syncDemoTime() {
    var v = DEMO.videos[0];
    if (!v) { DEMO.raf = null; return; }
    if (!DEMO.dragging && v.duration) {
      DEMO.scrub.value = (v.currentTime / v.duration) * 1000;
      DEMO.timeEl.textContent = v.currentTime.toFixed(2) + "s";
    }
    DEMO.raf = requestAnimationFrame(syncDemoTime);
  }

  function showDeck(i) {
    DEMO.deck = i;
    document.querySelectorAll("#demo-tabs li").forEach(function (li, k) {
      li.classList.toggle("is-active", k === i);
    });
    var deck = DEMO.decks[i];
    DEMO.cards = deck.cards;

    DEMO.strip.innerHTML = "";
    deck.cards.forEach(function (card, k) {
      var b = el("button");
      b.title = deck.label + " sample " + (k + 1);
      var img = el("img");
      img.src = card.input_poster;
      img.alt = deck.label + " sample " + (k + 1) + " source frame";
      b.appendChild(img);
      b.appendChild(el("span", "strip-num", String(k + 1)));
      b.addEventListener("click", function () { showCard(k); });
      DEMO.strip.appendChild(b);
    });

    showCard(0);
  }

  function showCard(k) {
    stopDemo();
    audibleCell = null;
    var deck = DEMO.decks[DEMO.deck];
    DEMO.panel.innerHTML = "";
    DEMO.videos = [];

    if (!DEMO.cards.length) {
      DEMO.prompt.innerHTML = '<div class="what">No clips found for this family.</div>';
      DEMO.strip.querySelectorAll("button").forEach(function (b) {
        b.classList.remove("is-current");
      });
      return;
    }

    k = ((k % DEMO.cards.length) + DEMO.cards.length) % DEMO.cards.length;
    DEMO.index = k;
    var card = DEMO.cards[k];

    var wrap = el("div", "demo-card");
    var nCells = 1 + card.outputs.length;
    var row = el("div", "demo-row" + (nCells < 4 ? " is-narrow" : "") +
                          (DEMO.focus ? " is-focus" : ""));

    var srcCell = el("div", "cell is-source");
    srcCell.appendChild(makeVideo(card.input_src, card.input_poster));
    srcCell.appendChild(el("span", "cell-tag", "Source"));
    srcCell.appendChild(el("span", "cell-audio",
                           '<i class="fas fa-volume-mute"></i>'));
    wireCell(srcCell, row);
    row.appendChild(srcCell);

    card.outputs.forEach(function (o, oi) {
      var c = el("div", "cell is-output");
      c.appendChild(makeVideo(o.src, o.poster));
      // numbered by position, so a card always reads 1, 2, 3 left to right
      c.appendChild(el("span", "cell-tag", "Instruction " + (oi + 1)));
      c.appendChild(el("span", "cell-audio",
                       '<i class="fas fa-volume-mute"></i>'));
      wireCell(c, row);
      row.appendChild(c);
    });
    wrap.appendChild(row);

    // instructions, numbered to match the panel labels
    card.outputs.forEach(function (o, oi) {
      var note = el("div", "demo-note");
      note.appendChild(el("span", "instr-num", String(oi + 1)));
      note.appendChild(el("span", null, o.caption_html || o.caption));
      wrap.appendChild(note);
    });
    DEMO.panel.appendChild(wrap);

    row.querySelectorAll("video").forEach(function (v) { DEMO.videos.push(v); });

    DEMO.prompt.innerHTML =
      '<div class="who">' + deck.label + " &middot; sample " +
      (k + 1) + " / " + DEMO.cards.length + "</div>";

    DEMO.strip.querySelectorAll("button").forEach(function (b, j) {
      b.classList.toggle("is-current", j === k);
    });
    syncPlayBtn(DEMO.playBtn, DEMO.panel);
  }

  function makeVideo(src, poster) {
    var v = el("video");
    v.src = src;
    if (poster) v.poster = poster;
    v.muted = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.loop = DEMO.loop;
    prime(v);
    return v;
  }

  /* A click on any panel plays the whole row together, so the instruction's
     effect reads as a comparison rather than as loose clips; the panel under
     the cursor carries the sound. Nothing starts on hover. */
  function wireCell(cell, row) {
    cell.addEventListener("click", function (e) {
      e.stopPropagation();
      toggleAudible(cell);
      playRow(row, cell);
    });
  }

  function playRow(row, voiceCell) {
    var started = false;
    row.querySelectorAll("video").forEach(function (v) {
      if (v.paused) started = true;
    });
    row.querySelectorAll("video").forEach(function (v) {
      if (started) { try { v.currentTime = 0; } catch (e) {} }
      safePlay(v, voiceCell.contains(v));
    });
    beAudible(voiceCell);
    // a row elsewhere on the page must not keep playing
    document.querySelectorAll(".demo-row").forEach(function (r) {
      if (r === row) return;
      var any = false;
      r.querySelectorAll("video").forEach(function (v) {
        if (!v.paused) { v.pause(); any = true; }
      });
      if (any) r.querySelectorAll(".cell").forEach(hush);
    });
    if (!DEMO.raf) syncDemoTime();
    syncPlayBtn(DEMO.playBtn, DEMO.panel);
  }

  /* ======================================================================
     3. JOINT GENERATION GALLERY
     ====================================================================== */
  var pinnedTile = null;
  var jointFocus = false;

  function buildJoint() {
    var grid = document.getElementById("joint-grid");
    var items = DATA.joint || [];
    if (!grid) return;
    if (!items.length) {
      grid.appendChild(el("div", "gallery-empty", "No clips found."));
      return;
    }

    var bar = el("div", "cmp-transport joint-toolbar");
    bar.innerHTML =
      '<button class="btn" id="joint-focus"><i class="fas fa-expand"></i> Focus</button>';
    grid.parentNode.insertBefore(bar, grid);

    document.getElementById("joint-focus").addEventListener("click", function () {
      jointFocus = !jointFocus;
      grid.classList.toggle("is-focus", jointFocus);
      this.classList.toggle("is-on", jointFocus);
      this.innerHTML = jointFocus
        ? '<i class="fas fa-compress"></i> Unfocus'
        : '<i class="fas fa-expand"></i> Focus';
      widenColumn(grid, jointFocus);
    });

    items.forEach(function (item) {
      var card = el("div", "video-card");
      card.appendChild(makeVideo(item.src, item.poster));
      card.appendChild(el("span", "sound-icon",
                          '<i class="fas fa-volume-mute"></i>'));

      card.addEventListener("click", function (e) {
        e.stopPropagation();
        if (pinnedTile === card) {
          pinnedTile = null;
          stopTile(card);
          return;
        }
        if (pinnedTile) { var prev = pinnedTile; pinnedTile = null; stopTile(prev); }
        pinnedTile = card;
        playTile(card);
      });
      grid.appendChild(card);
    });
  }

  function playTile(card) {
    if (audibleCell && audibleCell !== card) {
      var prev = audibleCell;
      audibleCell = null;
      if (prev.classList.contains("video-card")) {
        var pv = prev.querySelector("video");
        if (pv) { pv.pause(); pv.muted = true; }
        prev.classList.remove("is-audio");
        var pi = prev.querySelector(".sound-icon");
        if (pi) pi.innerHTML = '<i class="fas fa-volume-mute"></i>';
      } else {
        hush(prev);
      }
    }
    var v = card.querySelector("video");
    if (!v) return;
    try { v.currentTime = 0; } catch (e) {}
    safePlay(v, true);
    card.classList.add("is-audio");
    var ic = card.querySelector(".sound-icon");
    if (ic) ic.innerHTML = '<i class="fas fa-volume-up"></i>';
    audibleCell = card;
  }

  function stopTile(card) {
    var v = card.querySelector("video");
    if (v) {
      v.pause();
      v.muted = true;
      try { v.currentTime = 0; } catch (e) {}
    }
    card.classList.remove("is-audio");
    var ic = card.querySelector(".sound-icon");
    if (ic) ic.innerHTML = '<i class="fas fa-volume-mute"></i>';
    if (audibleCell === card) audibleCell = null;
  }

  /* ======================================================================
     4. INFERENCE MODES
     ----------------------------------------------------------------------
     Same shape as the demos: one instruction at a time, stepped with a
     filmstrip, with its own transport. The four modes of a sample stay in
     step, which is what makes the comparison legible -- what differs is the
     mode and not the moment.
     ====================================================================== */
  var MODE_ORDER = ["input", "base", "intent", "refine"];
  var INFER = {
    items: [], index: 0, focus: false, loop: true,
    dragging: false, raf: null, videos: []
  };

  function buildInference() {
    var host = document.getElementById("infer-panel");
    INFER.items = DATA.inference || [];
    if (!host) return;
    if (!INFER.items.length) {
      host.appendChild(el("div", "gallery-empty", "No clips found."));
      return;
    }
    INFER.host = host;
    INFER.strip = document.getElementById("infer-strip");

    var parent = host.parentNode;
    var block = el("div", "sample-block");
    parent.insertBefore(block, host);

    // the instruction for the sample on screen
    INFER.prompt = el("div", "cmp-prompt infer-prompt");
    INFER.prompt.setAttribute("id", "infer-prompt");

    var bar = el("div", "cmp-transport aligned-center");
    bar.innerHTML =
      '<button class="btn" id="infer-play"><i class="fas fa-play"></i> Play</button>' +
      '<button class="btn is-on" id="infer-loop"><i class="fas fa-repeat"></i> Loop</button>' +
      '<button class="btn" id="infer-focus"><i class="fas fa-expand"></i> Focus</button>';

    var prog = el("div", "cmp-progress");
    prog.innerHTML =
      '<input class="scrub" id="infer-scrub" type="range" min="0" max="1000" value="0"' +
      ' aria-label="Scrub this sample">' +
      '<span class="timecode" id="infer-time">0.00s</span>';

    block.appendChild(INFER.prompt);
    block.appendChild(bar);
    block.appendChild(prog);
    block.appendChild(host);
    parent.insertBefore(INFER.strip, block.nextSibling);

    INFER.playBtn = document.getElementById("infer-play");
    INFER.scrub = document.getElementById("infer-scrub");
    INFER.timeEl = document.getElementById("infer-time");

    INFER.playBtn.addEventListener("click", function () { toggleInfer(); });

    INFER.scrub.addEventListener("input", function () {
      INFER.dragging = true;
      var v = INFER.videos[0];
      if (!v || !v.duration) return;
      var t = (INFER.scrub.value / 1000) * v.duration;
      INFER.videos.forEach(function (x) { try { x.currentTime = t; } catch (e) {} });
      INFER.timeEl.textContent = t.toFixed(2) + "s";
    });
    INFER.scrub.addEventListener("change", function () { INFER.dragging = false; });

    var loopBtn = document.getElementById("infer-loop");
    loopBtn.addEventListener("click", function () {
      INFER.loop = !INFER.loop;
      loopBtn.classList.toggle("is-on", INFER.loop);
      INFER.videos.forEach(function (v) { v.loop = INFER.loop; });
    });

    document.getElementById("infer-focus").addEventListener("click", function () {
      INFER.focus = !INFER.focus;
      host.querySelectorAll(".infer-grid").forEach(function (g) {
        g.classList.toggle("is-focus", INFER.focus);
      });
      this.classList.toggle("is-on", INFER.focus);
      this.innerHTML = INFER.focus
        ? '<i class="fas fa-compress"></i> Unfocus'
        : '<i class="fas fa-expand"></i> Focus';
      widenColumn(host, INFER.focus);
    });

    // a filmstrip of the sample's own input frame, as in the other sections
    INFER.items.forEach(function (item, k) {
      var b = el("button");
      b.title = "Inference sample " + (k + 1) + " (" + item.task + ")";
      var img = el("img");
      img.src = item.input.poster;
      img.alt = "Inference sample " + (k + 1) + " source frame";
      b.appendChild(img);
      b.appendChild(el("span", "strip-num", String(k + 1)));
      b.addEventListener("click", function () { showInfer(k); });
      INFER.strip.appendChild(b);
    });

    document.addEventListener("keydown", function (e) {
      var ae = document.activeElement;
      if (ae && /^(INPUT|TEXTAREA)$/.test(ae.tagName)) return;
      if (!isFocusedSection(host)) return;
      if (e.key === "ArrowRight") showInfer(INFER.index + 1);
      else if (e.key === "ArrowLeft") showInfer(INFER.index - 1);
    });

    showInfer(0);
  }

  function toggleInfer() {
    if (!INFER.videos.length) return;
    if (INFER.videos[0].paused) {
      var voice = INFER.host.querySelector(".infer-cell.is-audible") ||
                  INFER.host.querySelector(".infer-cell");
      if (voice) beAudible(voice);
      INFER.videos.forEach(function (v) {
        try { v.currentTime = 0; } catch (e) {}
        safePlay(v, voice && voice.contains(v));
      });
      if (!INFER.raf) syncInferTime();
    } else {
      stopInfer();
    }
    syncPlayBtn(INFER.playBtn, INFER.host);
  }

  function stopInfer() {
    if (INFER.raf) { cancelAnimationFrame(INFER.raf); INFER.raf = null; }
    INFER.videos.forEach(function (v) {
      v.pause();
      try { v.currentTime = 0; } catch (e) {}
    });
    if (INFER.playBtn) INFER.playBtn.innerHTML = '<i class="fas fa-play"></i> Play';
    if (INFER.timeEl) INFER.timeEl.textContent = "0.00s";
    if (INFER.scrub) INFER.scrub.value = 0;
  }

  function syncInferTime() {
    var v = INFER.videos[0];
    if (!v) { INFER.raf = null; return; }
    if (!INFER.dragging && v.duration) {
      INFER.scrub.value = (v.currentTime / v.duration) * 1000;
      INFER.timeEl.textContent = v.currentTime.toFixed(2) + "s";
    }
    INFER.raf = requestAnimationFrame(syncInferTime);
  }

  function showInfer(k) {
    stopInfer();
    audibleCell = null;
    INFER.host.innerHTML = "";
    INFER.videos = [];

    k = ((k % INFER.items.length) + INFER.items.length) % INFER.items.length;
    INFER.index = k;
    var item = INFER.items[k];

    INFER.prompt.innerHTML =
      '<div class="who">Inference mode &middot; sample ' + (k + 1) + " / " +
      INFER.items.length + "</div>" +
      '<div class="what">' + (item.prompt_html || item.prompt) + "</div>";

    var row = el("div", "infer-grid" + (INFER.focus ? " is-focus" : ""));
    MODE_ORDER.forEach(function (key) {
      var d = item[key];
      if (!d) return;
      var cell = el("div", "infer-cell" + (key === "input" ? " is-source" : ""));
      var v = el("video");
      v.src = d.src;
      v.poster = d.poster;
      v.muted = true;
      v.playsInline = true;
      v.preload = "metadata";
      v.loop = INFER.loop;
      cell.appendChild(v);
      // the mode name rides on the clip, as every other tag on the page does
      cell.appendChild(el("span", "cell-tag", d.label));
      cell.appendChild(el("span", "cell-audio",
                          '<i class="fas fa-volume-mute"></i>'));
      prime(v);
      INFER.videos.push(v);

      cell.addEventListener("click", function (e) {
        e.stopPropagation();
        if (toggleAudible(cell)) {
          if (v.paused) safePlay(v, true);
        } else {
          v.pause();
        }
        syncPlayBtn(INFER.playBtn, INFER.host);
      });
      row.appendChild(cell);
    });
    INFER.host.appendChild(row);

    INFER.strip.querySelectorAll("button").forEach(function (b, j) {
      b.classList.toggle("is-current", j === k);
    });
    syncPlayBtn(INFER.playBtn, INFER.host);
  }

  /* ====================================================================== */
  /* Page chrome: the top bar and the scroll reveals. Both are additive -- with
     no IntersectionObserver, or with motion reduced, every section is simply
     already visible and the bar stays put. */
  function wireChrome() {
    var bar = document.getElementById("topbar");
    var links = document.getElementById("topbar-links");

    /* Sections the bar highlights, in page order. */
    var marks = [];
    Array.prototype.forEach.call(
      document.querySelectorAll("section[id]"), function (sec) {
        var a = links && links.querySelector('a[href="#' + sec.id + '"]');
        if (a) marks.push({ sec: sec, a: a });
      });

    function onScroll() {
      if (bar) bar.classList.toggle("is-stuck", window.scrollY > 8);
      if (!marks.length) return;
      /* the last section whose top has passed the bar wins */
      var best = null;
      marks.forEach(function (m) {
        if (m.sec.getBoundingClientRect().top <= 120) best = m;
      });
      marks.forEach(function (m) { m.a.classList.toggle("is-active", m === best); });
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    var reveals = document.querySelectorAll(".reveal");
    var reduce = window.matchMedia &&
                 window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!("IntersectionObserver" in window) || reduce) {
      Array.prototype.forEach.call(reveals, function (n) { n.classList.add("is-in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add("is-in");
        io.unobserve(e.target);
      });
    }, { rootMargin: "0px 0px -12% 0px", threshold: 0.04 });
    Array.prototype.forEach.call(reveals, function (n) { io.observe(n); });
  }

  document.addEventListener("DOMContentLoaded", function () {
    buildComparison();
    buildDemos();
    buildInference();
    buildJoint();
    wireChrome();
  });
})();
