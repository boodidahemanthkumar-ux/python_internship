/**
 * VisionSense AI - Interactive Frontend Logic
 * Handles drag-and-drop, client-side validation, instant image preview,
 * confidence slider, and loading states.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Element selectors
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const dropzoneEmpty = document.getElementById('dropzoneEmpty');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const previewImage = document.getElementById('previewImage');
    const previewName = document.getElementById('previewName');
    const previewSize = document.getElementById('previewSize');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const submitBtn = document.getElementById('submitBtn');
    const detectionForm = document.getElementById('detectionForm');
    const confRange = document.getElementById('confRange');
    const confDisplay = document.getElementById('confDisplay');
    const loadingOverlay = document.getElementById('loadingOverlay');

    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB in bytes
    const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'];

    // 1. Confidence Slider Listener
    if (confRange && confDisplay) {
        confRange.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            const pct = Math.round(val * 100);
            confDisplay.textContent = `${pct}% (${val.toFixed(2)})`;
        });
    }

    // 2. Format Bytes Helper
    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 3. File Validation
    function validateFile(file) {
        if (!file) return false;

        const ext = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            alert(`Unsupported file format (.${ext}). Please select a JPG, JPEG, PNG, or WEBP image.`);
            return false;
        }

        if (file.size > MAX_FILE_SIZE) {
            alert(`File size exceeds the 10 MB limit (${formatBytes(file.size)}). Please choose a smaller image.`);
            return false;
        }

        return true;
    }

    // 4. Show Image Preview
    function displayFilePreview(file) {
        if (!validateFile(file)) {
            resetDropzone();
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        previewImage.src = objectUrl;
        previewName.textContent = file.name;
        previewSize.textContent = formatBytes(file.size);

        dropzoneEmpty.style.display = 'none';
        dropzonePreview.style.display = 'flex';
        submitBtn.disabled = false;
    }

    // 5. Reset Dropzone
    function resetDropzone() {
        fileInput.value = '';
        previewImage.src = '';
        dropzoneEmpty.style.display = 'block';
        dropzonePreview.style.display = 'none';
        submitBtn.disabled = true;
    }

    // 6. Click to browse
    if (dropzone) {
        dropzone.addEventListener('click', (e) => {
            // Avoid triggering file picker if user clicks remove button
            if (e.target.closest('#removeFileBtn')) return;
            fileInput.click();
        });

        // Accessibility: allow keyboard space/enter
        dropzone.setAttribute('tabindex', '0');
        dropzone.setAttribute('role', 'button');
        dropzone.setAttribute('aria-label', 'File upload dropzone');
        dropzone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });
    }

    // 7. File Input Change
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                displayFilePreview(e.target.files[0]);
            }
        });
    }

    // 8. Remove File Button
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetDropzone();
        });
    }

    // 9. Drag and Drop Events
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                const droppedFile = dt.files[0];
                fileInput.files = dt.files;
                displayFilePreview(droppedFile);
            }
        });
    }

    // 10. Form Submission & Loading State
    if (detectionForm) {
        detectionForm.addEventListener('submit', (e) => {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select an image before detecting.');
                return;
            }

            // Show loading animation
            if (loadingOverlay) {
                loadingOverlay.style.display = 'flex';
            }

            // Disable submit button to prevent double submission
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-icon">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M12 2a10 10 0 0 1 10 10"></path>
                    </svg>
                    <span>Processing Image...</span>
                `;
            }
        });
    }
});
