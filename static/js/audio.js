// Audio comparison behaviour.
//
// The page can hold dozens of <audio> elements. Two things make that usable:
//
//   1. Only one plays at a time — starting a player pauses every other one.
//   2. Switching system within a row keeps your position, so you can A/B the
//      same moment across systems instead of restarting from zero each time.

(function () {
    'use strict';

    function setup() {
        const table = document.querySelector('.audio-table');
        if (!table) return;

        const players = Array.from(table.querySelectorAll('audio'));
        if (players.length === 0) return;

        players.forEach(function (audio) {
            audio.addEventListener('play', function () {
                const row = audio.closest('tr');

                players.forEach(function (other) {
                    if (other === audio || other.paused) return;

                    // Same row: carry the playhead over so the comparison is
                    // aligned. Different row: just stop it.
                    if (row && other.closest('tr') === row) {
                        const t = other.currentTime;
                        other.pause();
                        if (isFinite(t) && (!audio.duration || t < audio.duration)) {
                            audio.currentTime = t;
                        }
                    } else {
                        other.pause();
                    }
                });

                // Highlight the row being auditioned.
                table.querySelectorAll('tr.is-playing')
                     .forEach(function (r) { r.classList.remove('is-playing'); });
                if (row) row.classList.add('is-playing');
            });

            audio.addEventListener('ended', function () {
                const row = audio.closest('tr');
                if (row) row.classList.remove('is-playing');
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup);
    } else {
        setup();
    }
})();
