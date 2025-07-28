document.addEventListener('DOMContentLoaded', function() {
    const newsForm = document.getElementById('newsForm');
    const newsArticle = document.getElementById('newsArticle');
    const detectBtn = document.getElementById('detectBtn');

    newsForm.addEventListener('submit', function() {
        // Disable button and show loading text when submitting
        detectBtn.disabled = true;
        detectBtn.textContent = 'Detecting...';
    });

    // Optional: Auto-resize textarea (more complex if full auto-resize is needed)
    // For now, let's keep it simple with fixed rows and vertical resize.
    // If you need more advanced auto-resize:
    // newsArticle.addEventListener('input', function() {
    //     this.style.height = 'auto';
    //     this.style.height = (this.scrollHeight) + 'px';
    // });
});
