// LaTeX rendering via KaTeX's auto-render extension.
//
// Runs over the whole body once the DOM is ready, so math works anywhere —
// the abstract, figure captions, table headers, and text injected by
// prepare_project_page_audio.ipynb alike.
//
// Loaded with `defer`, so it executes after katex.min.js and auto-render.min.js
// (also deferred, and earlier in the document) have run. Deferred scripts
// execute in document order, so renderMathInElement is defined by this point.

(function () {
    'use strict';

    function render() {
        if (typeof renderMathInElement !== 'function') {
            // KaTeX blocked or failed to load: leave the raw source visible
            // rather than blanking the text.
            console.warn('KaTeX not available — math left unrendered');
            return;
        }

        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false },
                { left: '$', right: '$', display: false },
            ],

            // $...$ is enabled above, which is convenient for captions pasted
            // out of the paper but would wreck any literal text containing a
            // dollar sign or stray braces. These tags are skipped, which is
            // what keeps the BibTeX block (inside <pre><code>) intact.
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre',
                          'code', 'option'],

            // Malformed math renders in red instead of throwing and aborting
            // the rest of the pass.
            throwOnError: false,
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', render);
    } else {
        render();
    }
})();
