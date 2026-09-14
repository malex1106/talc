// Shared A/B comparison player.
//
// Replaces every <audio> in the comparison tables with a compact "listen to
// this system" button, and adds one transport docked at the bottom of the
// viewport. Each example is a comparison group — a row of the main table, or a
// truncation block together with its Original — and all files of a group share
// one timeline: switching system keeps the playhead exactly where it is.
//
// Why Web Audio rather than <audio> elements: independent media elements each
// run on their own clock and seek imprecisely inside MP3s, so "switch at the
// same instant" lands tens to hundreds of milliseconds off. Here every file of
// the active group is decoded to an AudioBuffer and restarted from the same
// sample offset with a short crossfade, which makes the switch sample-aligned
// and gapless.
//
// The tables are enhanced in place, so the generating notebook needs no
// changes.
//
// Web Audio needs the files' bytes, which fetch() cannot read when the page is
// opened as a local file (file://). There, and in browsers without Web Audio,
// the same player runs on <audio> elements instead: everything works except the
// waveform, and a switch lands within a few tens of milliseconds rather than on
// the exact sample. Served over HTTP — GitHub Pages, or python -m http.server —
// the full engine is used.

(function () {
    'use strict';

    var SKIP_SEC = 5;          // dock buttons and arrow keys
    var FINE_SEC = 1;          // shift + arrow keys
    var XFADE_SEC = 0.012;     // switch/seek crossfade: long enough to avoid clicks
    var MIN_LOOP_SEC = 0.25;
    var WAVE_COLOR = '#94a3b8';

    // Shortcut per system, in group order: the digit row, then the row below
    // it. A truncation block has 13 files, so digits alone are not enough.
    // None of these collide with the player's own keys (space, arrows, A, B,
    // Esc, Home, ?), and each chip shows the key it answers to.
    var KEYS = '1234567890qwertyuiop';

    var AudioCtx = window.AudioContext || window.webkitAudioContext;
    var MODE = (AudioCtx && window.fetch && window.AbortController &&
                location.protocol !== 'file:') ? 'webaudio' : 'media';

    var ICONS = {
        play: '<svg class="ab-i-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>',
        pause: '<svg class="ab-i-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>',
        restart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6h2v12H6zM9.5 12 18 18V6z"/></svg>',
        close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>'
    };

    function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }

    // m:ss.t — built from whole tenths so 59.96 s reads 1:00.0, not 0:60.0.
    function fmt(t) {
        if (!isFinite(t)) return '–:––.–';
        var d = Math.floor(Math.max(0, t) * 10);
        var m = Math.floor(d / 600);
        var s = (d % 600) / 10;
        return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
    }

    // ------------------------------------------------------------------ groups

    // Text of an element without its <small> annotations, so a header like
    // "25% masked<small>48 of 64 ch · 85×</small>" yields "25% masked".
    function ownText(el) {
        if (!el) return '';
        var clone = el.cloneNode(true);
        clone.querySelectorAll('small').forEach(function (s) { s.remove(); });
        return clone.textContent.replace(/\s+/g, ' ').trim();
    }

    function headerFor(cell) {
        var table = cell.closest('table');
        var head = table && table.tHead && table.tHead.rows[0];
        return head ? ownText(head.cells[cell.cellIndex]) : '';
    }

    function source(audio, label, row, col) {
        return { el: audio, url: audio.getAttribute('src'), label: label,
                 row: row, col: col, cell: null, chip: null };
    }

    function collectGroups() {
        var groups = [];

        document.querySelectorAll('.audio-table tbody tr').forEach(function (tr) {
            var sources = [];
            tr.querySelectorAll('td audio').forEach(function (audio) {
                var label = headerFor(audio.closest('td'));
                sources.push(source(audio, label, '', label));
            });
            if (sources.length) {
                groups.push({ root: tr, sources: sources,
                              title: ownText(tr.querySelector('.example-name')) || 'Example' });
            }
        });

        document.querySelectorAll('.trunc-example').forEach(function (block) {
            var sources = [];
            var orig = block.querySelector('.trunc-original audio');
            if (orig) sources.push(source(orig, 'Original', '', 'Original'));

            block.querySelectorAll('.trunc-table tbody tr').forEach(function (tr) {
                var row = ownText(tr.cells[0]);
                tr.querySelectorAll('td audio').forEach(function (audio) {
                    var col = headerFor(audio.closest('td'));
                    sources.push(source(audio, row + ' · ' + col, row, col));
                });
            });
            if (sources.length) {
                var name = ownText(block.querySelector('summary')) || 'Example';
                groups.push({ root: block, sources: sources, title: name + ' (truncation)' });
            }
        });

        return groups;
    }

    // "25% masked" -> "25%", "no masking" -> "0%": inside a model's chip group
    // the model is already named, so a chip only needs the level.
    function shortLevel(col) {
        var m = col.match(/(\d+(?:\.\d+)?)\s*%/);
        if (m) return m[1] + '%';
        if (/no mask/i.test(col)) return '0%';
        return col;
    }

    // ------------------------------------------------------------------ engine

    var webAudioEngine = {
        ctx: null,
        out: null,
        buffer: null,
        node: null,
        gain: null,
        playing: false,
        startedAt: 0,      // ctx time at which the current node started
        offset: 0,         // buffer position the current node started from
        loopA: null,
        loopB: null,
        onEnded: null,

        context: function () {
            if (!this.ctx) {
                try {
                    // The files are 44.1 kHz; decoding at the same rate avoids
                    // resampling the content itself.
                    this.ctx = new AudioCtx({ sampleRate: 44100 });
                } catch (e) {
                    this.ctx = new AudioCtx();
                }
                this.out = this.ctx.createGain();
                this.out.connect(this.ctx.destination);
            }
            return this.ctx;
        },

        // Must run inside a user gesture, or the browser keeps audio suspended.
        unlock: function () {
            var ctx = this.context();
            if (ctx.state === 'suspended') ctx.resume().catch(function () {});
        },

        duration: function () { return this.buffer ? this.buffer.duration : 0; },

        looping: function () {
            return this.loopA !== null && this.loopB !== null &&
                   this.loopB - this.loopA >= MIN_LOOP_SEC;
        },

        position: function () {
            if (!this.playing || !this.ctx) return this.offset;
            var raw = this.offset + (this.ctx.currentTime - this.startedAt);
            // A looping node plays on from its start offset to loopEnd, then
            // wraps to loopStart — mirror that to report the true position.
            if (this.looping() && raw >= this.loopB) {
                return this.loopA + ((raw - this.loopB) % (this.loopB - this.loopA));
            }
            return Math.min(raw, this.duration());
        },

        _start: function (offset) {
            var self = this;
            var ctx = this.ctx;
            var node = ctx.createBufferSource();
            var gain = ctx.createGain();
            node.buffer = this.buffer;
            node.connect(gain);
            gain.connect(this.out);
            if (this.looping()) {
                node.loop = true;
                node.loopStart = this.loopA;
                node.loopEnd = this.loopB;
            }
            var t = ctx.currentTime;
            gain.gain.setValueAtTime(0, t);
            gain.gain.linearRampToValueAtTime(1, t + XFADE_SEC);
            node.onended = function () {
                if (node !== self.node) return;       // stopped on purpose
                self.node = self.gain = null;
                self.playing = false;
                self.offset = 0;
                if (self.onEnded) self.onEnded();
            };
            node.start(t, offset);
            this.node = node;
            this.gain = gain;
            this.startedAt = t;
            this.offset = offset;
        },

        _stop: function () {
            var node = this.node;
            var gain = this.gain;
            if (!node) return;
            this.node = this.gain = null;             // turns its onended into a no-op
            var t = this.ctx.currentTime;
            gain.gain.cancelScheduledValues(t);
            gain.gain.setValueAtTime(gain.gain.value, t);
            gain.gain.linearRampToValueAtTime(0, t + XFADE_SEC);
            node.stop(t + XFADE_SEC + 0.01);
        },

        // Crossfades from the running node to a fresh one at `at`. Inaudible
        // when `at` is the current position, which is what makes switching work.
        _restartAt: function (at) {
            this._stop();
            this._start(at);
        },

        play: function () {
            if (!this.buffer || this.playing) return;
            this.context();
            var at = this.offset;
            if (at >= this.duration() - 0.05) at = 0;
            if (this.looping() && (at < this.loopA || at >= this.loopB)) at = this.loopA;
            this._start(at);
            this.playing = true;
        },

        pause: function () {
            if (!this.playing) return;
            var at = this.position();
            this._stop();
            this.offset = at;
            this.playing = false;
        },

        seek: function (at) {
            at = clamp(at, 0, Math.max(0, this.duration() - 0.01));
            if (this.playing) this._restartAt(at);
            else this.offset = at;
        },

        // Swap the audible file while keeping the playhead.
        setBuffer: function (buffer) {
            var at = Math.min(this.position(), buffer.duration);
            this.buffer = buffer;          // before _start, which reads it
            if (this.playing) this._restartAt(at);
            else this.offset = at;
        },

        setLoop: function (a, b) {
            var at = this.position();      // computed under the old loop settings
            this.loopA = a;
            this.loopB = b;
            if (this.looping() && (at < a || at >= b)) at = a;
            if (this.playing) this._restartAt(at);
            else this.offset = at;
        },

        reset: function () {
            this._stop();
            this.playing = false;
            this.buffer = null;
            this.offset = 0;
            this.loopA = this.loopB = null;
        }
    };

    // Fallback on plain <audio> elements, one per file; "buffer" is the element
    // itself. A switch hands the playhead from one element to the next, and a
    // loop is enforced by poll() on every animation frame.
    var mediaEngine = {
        buffer: null,
        playing: false,
        loopA: null,
        loopB: null,
        onEnded: null,

        unlock: function () {},

        duration: function () {
            var d = this.buffer ? this.buffer.duration : 0;
            return isFinite(d) ? d : 0;
        },

        looping: webAudioEngine.looping,

        position: function () { return this.buffer ? this.buffer.currentTime : 0; },

        play: function () {
            var el = this.buffer;
            if (!el || this.playing) return;
            if (el.currentTime >= this.duration() - 0.05) el.currentTime = 0;
            if (this.looping() && (el.currentTime < this.loopA || el.currentTime >= this.loopB)) {
                el.currentTime = this.loopA;
            }
            el.play().catch(function () {});
            this.playing = true;
        },

        pause: function () {
            if (!this.playing) return;
            this.buffer.pause();
            this.playing = false;
        },

        seek: function (at) {
            if (this.buffer) {
                this.buffer.currentTime = clamp(at, 0, Math.max(0, this.duration() - 0.01));
            }
        },

        setBuffer: function (el) {
            var at = this.position();
            var old = this.buffer;
            if (old && old !== el) old.pause();
            this.buffer = el;
            el.currentTime = Math.min(at, this.duration() || at);
            if (this.playing) el.play().catch(function () {});
        },

        setLoop: function (a, b) {
            this.loopA = a;
            this.loopB = b;
            var el = this.buffer;
            if (el && this.looping() && (el.currentTime < a || el.currentTime >= b)) {
                el.currentTime = a;
            }
        },

        poll: function () {
            var el = this.buffer;
            if (this.playing && el && this.looping() && el.currentTime >= this.loopB) {
                el.currentTime = this.loopA;
            }
        },

        ended: function (el) {
            if (el !== this.buffer) return;
            this.playing = false;
            el.currentTime = 0;
            if (this.onEnded) this.onEnded();
        },

        reset: function () {
            if (this.buffer) this.buffer.pause();
            this.buffer = null;
            this.playing = false;
            this.loopA = this.loopB = null;
        }
    };

    var engine = MODE === 'webaudio' ? webAudioEngine : mediaEngine;

    // ----------------------------------------------------------------- loading

    // Only the active group's files stay decoded: a 60 s mono buffer is ~10 MB,
    // and a truncation block has 13 of them.
    var cache = new Map();       // url -> { promise, cancel }
    var urlState = new Map();    // url -> 'loading' | 'ready' | 'error'
    var cellsByUrl = new Map();  // url -> [table buttons]

    function decode(bytes) {
        var ctx = webAudioEngine.context();
        // Callback form: older Safari has no promise-returning decodeAudioData.
        return new Promise(function (resolve, reject) {
            ctx.decodeAudioData(bytes, resolve, reject);
        });
    }

    function fetchAndDecode(url) {
        var controller = new AbortController();
        var promise = fetch(url, { signal: controller.signal })
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
                return r.arrayBuffer();
            })
            .then(decode);
        return { promise: promise, cancel: function () { controller.abort(); } };
    }

    function mediaElement(url) {
        var el = new Audio();
        el.preload = 'auto';
        var promise = new Promise(function (resolve, reject) {
            el.addEventListener('loadedmetadata', function () { resolve(el); }, { once: true });
            el.addEventListener('error', function () {
                reject(new Error('Could not load ' + url));
            }, { once: true });
        });
        el.addEventListener('ended', function () { mediaEngine.ended(el); });
        el.src = url;
        return {
            promise: promise,
            cancel: function () {
                el.pause();
                el.removeAttribute('src');
                el.load();              // stops the download
            }
        };
    }

    function load(url) {
        var entry = cache.get(url);
        if (!entry) {
            entry = MODE === 'webaudio' ? fetchAndDecode(url) : mediaElement(url);
            cache.set(url, entry);
            entry.promise.catch(function () {
                if (cache.get(url) === entry) cache.delete(url);
            });
        }
        return entry.promise;
    }

    // Drop everything the next group does not use. Files shared between groups
    // (the Original and unmasked reconstructions appear in both sections) are
    // kept, and their in-flight downloads are not aborted.
    function evictExcept(keep) {
        cache.forEach(function (entry, url) {
            if (!keep.has(url)) {
                entry.cancel();
                cache.delete(url);
                urlState.delete(url);
                paintUrl(url);
            }
        });
    }

    function setUrlState(url, st) {
        urlState.set(url, st);
        paintUrl(url);
    }

    function paintUrl(url) {
        var st = urlState.get(url);
        var els = (cellsByUrl.get(url) || []).slice();
        if (state.group) {
            state.group.sources.forEach(function (s) {
                if (s.url === url && s.chip) els.push(s.chip);
            });
        }
        els.forEach(function (el) {
            el.classList.toggle('is-loading', st === 'loading');
            el.classList.toggle('is-error', st === 'error');
        });
    }

    // Decode the rest of the group one file at a time, so every later switch is
    // instant. Skipped where the browser reports little memory or data saver.
    function preload(group, token) {
        var lowMemory = (navigator.deviceMemory && navigator.deviceMemory <= 2) ||
                        (navigator.connection && navigator.connection.saveData);
        if (lowMemory) return;

        var queue = group.sources.slice();
        (function next() {
            if (token !== state.groupToken || !queue.length) return;
            var src = queue.shift();
            if (urlState.get(src.url) === 'ready') { next(); return; }
            setUrlState(src.url, 'loading');
            load(src.url).then(function () {
                if (token === state.groupToken) setUrlState(src.url, 'ready');
                next();
            }, function (err) {
                if (token === state.groupToken && err.name !== 'AbortError') {
                    setUrlState(src.url, 'error');
                }
                next();
            });
        })();
    }

    // -------------------------------------------------------------- controller

    var state = {
        groups: [],
        group: null,
        index: -1,
        groupToken: 0,
        selectToken: 0,
        preloaded: false,
        markA: null,
        markB: null,
        scrub: null,          // seek-bar drag position, null when not dragging
        raf: 0,
        waveBuffer: null,
        active: []            // elements currently marked active
    };

    var dom = null;

    function current() {
        return state.group && state.index >= 0 ? state.group.sources[state.index] : null;
    }

    function activateGroup(group) {
        if (state.group) state.group.root.classList.remove('ab-active-group');
        engine.reset();
        state.group = group;
        state.index = -1;
        state.groupToken++;
        state.preloaded = false;
        state.markA = state.markB = null;
        state.waveBuffer = null;
        evictExcept(new Set(group.sources.map(function (s) { return s.url; })));
        group.root.classList.add('ab-active-group');
        showDock();
        buildSwitcher(group);
        drawWave(null);
        hideMessage();
    }

    // autoplay: clicks mean "listen to this"; keyboard switches keep the
    // current play/pause state.
    function select(group, index, autoplay) {
        engine.unlock();
        if (group === state.group && index === state.index) {
            togglePlay();
            return;
        }
        if (group !== state.group) activateGroup(group);

        state.index = index;
        var src = group.sources[index];
        var token = ++state.selectToken;
        if (urlState.get(src.url) !== 'ready') setUrlState(src.url, 'loading');
        render();

        load(src.url).then(function (buffer) {
            if (token !== state.selectToken) return;     // superseded by a later pick
            setUrlState(src.url, 'ready');
            engine.setBuffer(buffer);
            if (autoplay) engine.play();
            drawWave(buffer);
            render();
            kick();
            if (!state.preloaded) {
                state.preloaded = true;
                preload(group, state.groupToken);
            }
        }, function (err) {
            if (token !== state.selectToken || err.name === 'AbortError') return;
            setUrlState(src.url, 'error');
            showMessage('Could not load “' + src.label + '”.');
            render();
        });
    }

    function step(delta) {
        if (!state.group || state.index < 0) return;
        var n = state.group.sources.length;
        select(state.group, (state.index + delta + n) % n, false);
    }

    function togglePlay() {
        if (!engine.buffer) return;
        engine.unlock();
        if (engine.playing) engine.pause();
        else engine.play();
        render();
        kick();
    }

    // Seeking or skipping outside an active loop means "go somewhere else", so
    // the loop is dropped rather than snapping back into it.
    function seekTo(t) {
        if (!engine.buffer) return;
        if (engine.looping() && (t < engine.loopA || t >= engine.loopB)) {
            state.markA = state.markB = null;
            engine.loopA = engine.loopB = null;
        }
        engine.seek(t);
        render();
        kick();
    }

    function skip(delta) {
        if (engine.buffer) seekTo(engine.position() + delta);
    }

    function setMark(which) {
        if (!engine.buffer) return;
        var t = engine.position();
        if (which === 'A') state.markA = t;
        else state.markB = t;

        if (state.markA !== null && state.markB !== null) {
            var a = Math.min(state.markA, state.markB);
            var b = Math.max(state.markA, state.markB);
            if (b - a < MIN_LOOP_SEC) {
                // Too short to loop: keep only the mark just set.
                if (which === 'A') state.markB = null;
                else state.markA = null;
                if (engine.loopA !== null) engine.setLoop(null, null);
            } else {
                state.markA = a;
                state.markB = b;
                engine.setLoop(a, b);
            }
        }
        render();
        kick();
    }

    function clearLoop() {
        state.markA = state.markB = null;
        if (engine.loopA !== null || engine.loopB !== null) engine.setLoop(null, null);
        render();
        kick();
    }

    engine.onEnded = function () { render(); };

    // --------------------------------------------------------------- the table

    function enhanceCell(group, src, index) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'ab-cell';
        btn.setAttribute('aria-pressed', 'false');
        btn.setAttribute('aria-label', 'Play ' + src.label + ' — ' + group.title);
        btn.title = src.label;
        btn.innerHTML = ICONS.play + ICONS.pause;
        btn.addEventListener('click', function () { select(group, index, true); });
        src.el.replaceWith(btn);
        src.cell = btn;
        if (!cellsByUrl.has(src.url)) cellsByUrl.set(src.url, []);
        cellsByUrl.get(src.url).push(btn);
    }

    // ---------------------------------------------------------------- the dock

    function button(cls, html, title, onClick) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = cls;
        b.innerHTML = html;
        if (title) {
            b.title = title;
            b.setAttribute('aria-label', title);
        }
        b.addEventListener('click', onClick);
        return b;
    }

    function buildDock() {
        var dock = document.createElement('div');
        dock.className = 'ab-dock';
        dock.setAttribute('role', 'region');
        dock.setAttribute('aria-label', 'Audio comparison player');
        dock.hidden = true;

        var inner = document.createElement('div');
        inner.className = 'ab-dock-inner';
        dock.appendChild(inner);

        // -- top row: what is playing, transport, time, loop, help, close
        var top = document.createElement('div');
        top.className = 'ab-top';

        var now = button('ab-now', '', 'Scroll to this example', function () {
            if (state.group) state.group.root.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
        now.innerHTML = '<span class="ab-now-title"></span><span class="ab-now-sep">·</span>' +
                        '<span class="ab-now-label"></span>';

        var transport = document.createElement('div');
        transport.className = 'ab-transport';
        var play = button('ab-btn ab-play', ICONS.play + ICONS.pause, 'Play / pause (Space)', togglePlay);
        transport.append(
            button('ab-btn ab-restart', ICONS.restart, 'Back to start (Home)', function () { seekTo(0); }),
            button('ab-btn', '−' + SKIP_SEC + 's', 'Back ' + SKIP_SEC + ' s (←)', function () { skip(-SKIP_SEC); }),
            play,
            button('ab-btn', '+' + SKIP_SEC + 's', 'Forward ' + SKIP_SEC + ' s (→)', function () { skip(SKIP_SEC); })
        );

        var time = document.createElement('div');
        time.className = 'ab-time';
        time.innerHTML = '<span class="ab-cur">0:00.0</span> / <span class="ab-dur">–:––.–</span>';

        var loop = document.createElement('div');
        loop.className = 'ab-loop';
        var markA = button('ab-btn', 'A', 'Set loop start (A)', function () { setMark('A'); });
        var markB = button('ab-btn', 'B', 'Set loop end (B)', function () { setMark('B'); });
        var unloop = button('ab-btn', '✕ loop', 'Clear loop (Esc)', clearLoop);
        loop.append(markA, markB, unloop);

        var extra = document.createElement('div');
        extra.className = 'ab-extra';
        extra.append(
            button('ab-btn ab-help-btn', '?', 'Keyboard shortcuts (?)', function () { toggleHelp(); }),
            button('ab-btn', ICONS.close, 'Close player', closeDock)
        );

        top.append(now, transport, time, loop, extra);

        // -- seek bar with the active file's waveform
        var seek = document.createElement('div');
        seek.className = 'ab-seek';
        seek.tabIndex = 0;
        seek.setAttribute('role', 'slider');
        seek.setAttribute('aria-label', 'Seek');
        seek.setAttribute('aria-valuemin', '0');
        seek.innerHTML = '<canvas class="ab-wave"></canvas><div class="ab-region" hidden></div>' +
                         '<div class="ab-mark ab-mark-a" hidden></div><div class="ab-mark ab-mark-b" hidden></div>' +
                         '<div class="ab-progress"></div><div class="ab-head"></div>';

        // -- switcher: every file of the active group
        var switcher = document.createElement('div');
        switcher.className = 'ab-switch';
        switcher.setAttribute('role', 'group');
        switcher.setAttribute('aria-label', 'Systems in this example');

        var help = document.createElement('div');
        help.className = 'ab-help';
        help.hidden = true;
        help.innerHTML =
            '<span><kbd>Space</kbd> play / pause</span>' +
            '<span><kbd>←</kbd> <kbd>→</kbd> ∓' + SKIP_SEC + ' s (<kbd>Shift</kbd> ∓' + FINE_SEC + ' s)</span>' +
            '<span><kbd>↑</kbd> <kbd>↓</kbd> previous / next system</span>' +
            '<span><kbd>1</kbd>…<kbd>0</kbd> <kbd>Q</kbd>…<kbd>P</kbd> jump to system</span>' +
            '<span><kbd>A</kbd> <kbd>B</kbd> loop start / end</span>' +
            '<span><kbd>Esc</kbd> clear loop</span>' +
            '<span><kbd>Home</kbd> back to start</span>';

        var msg = document.createElement('div');
        msg.className = 'ab-msg';
        msg.setAttribute('role', 'status');
        msg.hidden = true;

        inner.append(top, seek, switcher, help, msg);
        document.body.appendChild(dock);

        dom = {
            dock: dock, now: now, play: play, seek: seek, switcher: switcher,
            help: help, msg: msg, markA: markA, markB: markB, unloop: unloop,
            title: now.querySelector('.ab-now-title'),
            label: now.querySelector('.ab-now-label'),
            cur: time.querySelector('.ab-cur'),
            dur: time.querySelector('.ab-dur'),
            wave: seek.querySelector('.ab-wave'),
            region: seek.querySelector('.ab-region'),
            mA: seek.querySelector('.ab-mark-a'),
            mB: seek.querySelector('.ab-mark-b'),
            progress: seek.querySelector('.ab-progress'),
            head: seek.querySelector('.ab-head')
        };

        bindSeek(seek);

        if (window.ResizeObserver) {
            new ResizeObserver(function () {
                syncDockSpace();
                drawWave(state.waveBuffer);
            }).observe(dock);
        } else {
            window.addEventListener('resize', function () {
                syncDockSpace();
                drawWave(state.waveBuffer);
            });
        }
    }

    // Drag to scrub: the position only previews while dragging and is applied on
    // release, so a drag does not stutter through dozens of restarts.
    function bindSeek(seek) {
        function timeAt(clientX) {
            var r = seek.getBoundingClientRect();
            return clamp((clientX - r.left) / r.width, 0, 1) * engine.duration();
        }
        seek.addEventListener('pointerdown', function (e) {
            if (!engine.buffer || e.button > 0) return;
            e.preventDefault();
            seek.setPointerCapture(e.pointerId);
            state.scrub = timeAt(e.clientX);
            renderTime();
            kick();
        });
        seek.addEventListener('pointermove', function (e) {
            if (state.scrub === null) return;
            state.scrub = timeAt(e.clientX);
        });
        seek.addEventListener('pointerup', function (e) {
            if (state.scrub === null) return;
            var t = timeAt(e.clientX);
            state.scrub = null;
            seekTo(t);
        });
        seek.addEventListener('pointercancel', function () {
            state.scrub = null;
            renderTime();
        });
    }

    function buildSwitcher(group) {
        var sw = dom.switcher;
        sw.textContent = '';
        var currentRow = null;
        var rowBox = null;

        group.sources.forEach(function (src, i) {
            var chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'ab-chip';
            chip.title = src.label;
            chip.setAttribute('aria-label', src.label);
            chip.setAttribute('aria-pressed', 'false');
            if (i < KEYS.length) {
                var key = document.createElement('span');
                key.className = 'ab-key';
                key.textContent = KEYS[i].toUpperCase();
                chip.appendChild(key);
            }
            chip.appendChild(document.createTextNode(src.row ? shortLevel(src.col) : src.label));
            chip.addEventListener('click', function () { select(group, i, true); });
            src.chip = chip;

            if (!src.row) {
                currentRow = null;
                sw.appendChild(chip);
                return;
            }
            // Sources come in row order, so a new row label opens a new box.
            if (src.row !== currentRow) {
                currentRow = src.row;
                rowBox = document.createElement('span');
                rowBox.className = 'ab-chip-group';
                var lab = document.createElement('span');
                lab.className = 'ab-chip-group-label';
                lab.textContent = src.row;
                rowBox.appendChild(lab);
                sw.appendChild(rowBox);
            }
            rowBox.appendChild(chip);
        });

        group.sources.forEach(function (s) { paintUrl(s.url); });
    }

    function showDock() {
        if (!dom) buildDock();
        dom.dock.hidden = false;
        syncDockSpace();
    }

    function closeDock() {
        engine.pause();
        if (state.group) state.group.root.classList.remove('ab-active-group');
        state.group = null;
        state.index = -1;
        state.groupToken++;
        state.selectToken++;
        clearActive();
        dom.dock.hidden = true;
        syncDockSpace();
    }

    // Reserve room at the bottom of the page so the dock never hides content,
    // and lift the scroll-to-top button above it.
    function syncDockSpace() {
        var h = dom && !dom.dock.hidden ? dom.dock.offsetHeight : 0;
        document.documentElement.style.setProperty('--ab-dock-h', h + 'px');
        document.body.classList.toggle('ab-dock-open', h > 0);
    }

    function toggleHelp(force) {
        if (!dom) return;
        dom.help.hidden = force === undefined ? !dom.help.hidden : !force;
        syncDockSpace();
    }

    function showMessage(text) {
        dom.msg.textContent = text;
        dom.msg.hidden = false;
        syncDockSpace();
    }

    function hideMessage() {
        if (dom && !dom.msg.hidden) {
            dom.msg.hidden = true;
            syncDockSpace();
        }
    }

    // ------------------------------------------------------------------ render

    function clearActive() {
        state.active.forEach(function (el) {
            el.classList.remove('is-active', 'is-playing');
            el.setAttribute('aria-pressed', 'false');
        });
        state.active = [];
    }

    function render() {
        if (!dom) return;
        clearActive();
        var src = current();
        if (src) {
            [src.cell, src.chip].forEach(function (el) {
                if (!el) return;
                el.classList.add('is-active');
                el.classList.toggle('is-playing', engine.playing);
                el.setAttribute('aria-pressed', 'true');
                state.active.push(el);
            });
            dom.title.textContent = state.group.title;
            dom.label.textContent = src.label;
        }
        dom.play.classList.toggle('is-playing', engine.playing);

        var looping = engine.looping();
        dom.markA.setAttribute('aria-pressed', String(state.markA !== null));
        dom.markB.setAttribute('aria-pressed', String(state.markB !== null));
        dom.unloop.hidden = state.markA === null && state.markB === null;

        var dur = engine.duration();
        dom.seek.setAttribute('aria-valuemax', dur.toFixed(1));
        function place(el, t) {
            el.hidden = t === null || !dur;
            if (!el.hidden) el.style.left = (100 * t / dur) + '%';
        }
        if (looping && dur) {
            dom.region.hidden = false;
            dom.region.style.left = (100 * engine.loopA / dur) + '%';
            dom.region.style.width = (100 * (engine.loopB - engine.loopA) / dur) + '%';
            dom.mA.hidden = dom.mB.hidden = true;
        } else {
            dom.region.hidden = true;
            place(dom.mA, state.markA);
            place(dom.mB, state.markB);
        }
        renderTime();
    }

    function renderTime() {
        if (!dom) return;
        var dur = engine.duration();
        var t = state.scrub !== null ? state.scrub : engine.position();
        dom.cur.textContent = fmt(t);
        dom.dur.textContent = dur ? fmt(dur) : '–:––.–';
        var f = dur ? clamp(t / dur, 0, 1) : 0;
        dom.progress.style.width = (100 * f) + '%';
        dom.head.style.left = (100 * f) + '%';
        dom.seek.setAttribute('aria-valuenow', t.toFixed(1));
        dom.seek.setAttribute('aria-valuetext', fmt(t) + ' of ' + fmt(dur));
    }

    function tick() {
        state.raf = 0;
        if (engine.poll) engine.poll();
        renderTime();
        if (engine.playing || state.scrub !== null) state.raf = requestAnimationFrame(tick);
    }

    function kick() {
        if (!state.raf) state.raf = requestAnimationFrame(tick);
    }

    // The waveform of whatever is audible — switching to a broken
    // reconstruction is visible as well as audible.
    var peakCache = new WeakMap();   // AudioBuffer -> { width, peaks }

    function peaks(buffer, width) {
        var data = buffer.getChannelData(0);
        var n = data.length;
        var out = new Float32Array(2 * width);
        var step = n / width;
        for (var x = 0; x < width; x++) {
            var s0 = Math.floor(x * step);
            var s1 = Math.min(n, Math.floor((x + 1) * step));
            var lo = 0, hi = 0;
            for (var i = s0; i < s1; i++) {
                var v = data[i];
                if (v < lo) lo = v;
                else if (v > hi) hi = v;
            }
            out[2 * x] = lo;
            out[2 * x + 1] = hi;
        }
        return out;
    }

    function drawWave(buffer) {
        if (!dom) return;
        state.waveBuffer = buffer;
        var canvas = dom.wave;
        var dpr = window.devicePixelRatio || 1;
        var w = Math.max(1, Math.round(canvas.clientWidth * dpr));
        var h = Math.max(1, Math.round(canvas.clientHeight * dpr));
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        var g = canvas.getContext('2d');
        g.clearRect(0, 0, w, h);
        if (!buffer || !buffer.getChannelData) return;   // media fallback: no samples

        var pk = peakCache.get(buffer);
        if (!pk || pk.width !== w) {
            pk = { width: w, peaks: peaks(buffer, w) };
            peakCache.set(buffer, pk);
        }
        var mid = h / 2;
        var amp = mid * 0.9;
        g.fillStyle = WAVE_COLOR;
        for (var x = 0; x < w; x++) {
            var y0 = mid - pk.peaks[2 * x + 1] * amp;
            var y1 = mid - pk.peaks[2 * x] * amp;
            g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
        }
    }

    // ---------------------------------------------------------------- keyboard

    function typing(target) {
        return target && (target.isContentEditable ||
                          /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
    }

    // Shortcuts are live only while the dock is open, so the page scrolls
    // normally otherwise.
    function onKeyDown(e) {
        if (!state.group || !dom || dom.dock.hidden) return;
        if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;

        var handled = true;
        switch (e.key) {
            case ' ': togglePlay(); break;
            case 'ArrowLeft': skip(-(e.shiftKey ? FINE_SEC : SKIP_SEC)); break;
            case 'ArrowRight': skip(e.shiftKey ? FINE_SEC : SKIP_SEC); break;
            case 'ArrowUp': step(-1); break;
            case 'ArrowDown': step(1); break;
            case 'Home': seekTo(0); break;
            case 'a': case 'A': setMark('A'); break;
            case 'b': case 'B': setMark('B'); break;
            case 'Escape':
                if (!dom.help.hidden) toggleHelp(false);
                else clearLoop();
                break;
            case '?': toggleHelp(); break;
            default:
                var i = e.key.length === 1 ? KEYS.indexOf(e.key.toLowerCase()) : -1;
                if (i >= 0 && i < state.group.sources.length) {
                    if (i !== state.index) select(state.group, i, false);
                } else {
                    handled = false;
                }
        }
        if (handled) e.preventDefault();
    }

    // A focused button fires its click on Space keyup; swallow it so Space
    // does not both toggle playback and press the button.
    function onKeyUp(e) {
        if (e.key === ' ' && state.group && dom && !dom.dock.hidden && !typing(e.target)) {
            e.preventDefault();
        }
    }

    // ----------------------------------------------------------------- startup

    function init() {
        // iOS mutes Web Audio under the silent switch unless the session is
        // declared as playback (Safari 17+; ignored elsewhere).
        if (navigator.audioSession) {
            try { navigator.audioSession.type = 'playback'; } catch (e) { /* unsupported */ }
        }

        state.groups = collectGroups();
        if (!state.groups.length) return;

        state.groups.forEach(function (group) {
            group.sources.forEach(function (src, i) { enhanceCell(group, src, i); });
        });
        document.querySelectorAll('.audio-table, .trunc-table').forEach(function (t) {
            t.classList.add('ab-enhanced');
        });

        document.addEventListener('keydown', onKeyDown);
        document.addEventListener('keyup', onKeyUp);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
