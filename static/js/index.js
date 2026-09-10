// Page behaviour for the project page.
//
// Trimmed from the Academic Project Page Template: the "More Works" dropdown,
// the image/video carousels and the jQuery + bulmaCarousel + bulmaSlider
// bootstrap were removed along with their markup, so nothing here depends on
// jQuery any more.

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    if (!bibtexElement || !button) return;

    const copyText = button.querySelector('.copy-text');

    function flashCopied() {
        button.classList.add('copied');
        copyText.textContent = 'Copied!';
        setTimeout(function () {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    }

    navigator.clipboard.writeText(bibtexElement.textContent)
        .then(flashCopied)
        .catch(function (err) {
            console.error('Failed to copy: ', err);
            // Fallback for browsers without the async clipboard API
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            flashCopied();
        });
}

// Scroll to top
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (!scrollButton) return;
    scrollButton.classList.toggle('visible', window.pageYOffset > 300);
});
