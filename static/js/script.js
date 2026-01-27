document.addEventListener('DOMContentLoaded', () => {
    // Elements - Upload
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');
    const analyzeBtn = document.getElementById('analyze-btn');
    const confRange = document.getElementById('conf-range');
    const confVal = document.getElementById('conf-val');
    const loader = document.getElementById('loader');
    const resultsArea = document.getElementById('results-area');
    const detectionsList = document.getElementById('detections-list');

    // Elements - Webcam
    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const webcamOverlay = document.getElementById('webcam-placeholder');
    const webcamControls = document.getElementById('webcam-controls');
    const webcamStats = document.getElementById('webcam-stats');
    const fpsCounter = document.getElementById('fps-counter');
    const detCounter = document.getElementById('detection-count');

    // Elements - Modal
    const modal = document.getElementById('modal');
    const modalImage = document.getElementById('modal-image');
    const modalDetails = document.getElementById('modal-details');
    const closeModal = document.querySelector('.close-modal');

    let stream = null;
    let detectionInterval = null;
    let currentFile = null;

    // --- Upload Logic ---

    confRange.addEventListener('input', (e) => {
        confVal.textContent = e.target.value;
    });

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
        };
        reader.readAsDataURL(file);
        resultsArea.classList.add('hidden');
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) {
            alert('Please select an image first');
            return;
        }

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('conf', confRange.value);

        loader.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                showResults(data);
            } else {
                alert('Analysis failed: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error(err);
            alert('Error connecting to API');
        } finally {
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function showResults(data) {
        // Show modal with marked image
        modalImage.src = `data:image/jpeg;base64,${data.image_base64}`;
        modal.classList.remove('hidden');

        // Show simplified results in the card
        resultsArea.classList.remove('hidden');
        detectionsList.innerHTML = '';

        if (data.num_detections === 0) {
            detectionsList.innerHTML = '<p>No detections found.</p>';
        } else {
            data.detections.forEach(det => {
                const tag = document.createElement('div');
                tag.className = 'detection-tag';
                tag.innerHTML = `
                    <span>${det.class}</span>
                    <span class="conf-pill">${(det.confidence * 100).toFixed(1)}%</span>
                `;
                detectionsList.appendChild(tag);
            });
        }
    }

    // --- Webcam Logic ---

    startWebcamBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 }
            });
            webcamVideo.srcObject = stream;
            webcamOverlay.classList.add('hidden');
            webcamControls.classList.remove('hidden');
            webcamStats.classList.remove('hidden');

            // Start detection loop
            startDetectionLoop();
        } catch (err) {
            console.error(err);
            alert('Could not access webcam. Please check permissions.');
        }
    });

    stopWebcamBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamVideo.srcObject = null;
            webcamOverlay.classList.remove('hidden');
            webcamControls.classList.add('hidden');
            webcamStats.classList.add('hidden');
            clearInterval(detectionInterval);
        }
    });

    function startDetectionLoop() {
        const ctx = webcamCanvas.getContext('2d');
        const hiddenCanvas = document.createElement('canvas'); // For capturing frames
        const hCtx = hiddenCanvas.getContext('2d');

        let lastTime = Date.now();

        detectionInterval = setInterval(async () => {
            // Check if hidden
            if (document.hidden) return;

            // Capture frame from video
            hiddenCanvas.width = webcamVideo.videoWidth;
            hiddenCanvas.height = webcamVideo.videoHeight;
            hCtx.drawImage(webcamVideo, 0, 0);

            // Convert to base64
            const base64Img = hiddenCanvas.toDataURL('image/jpeg', 0.6).split(',')[1];

            try {
                const response = await fetch('/predict/base64', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: base64Img,
                        conf: confRange.value
                    })
                });

                const data = await response.json();

                // Update UI stats
                const now = Date.now();
                const fps = Math.round(1000 / (now - lastTime));
                lastTime = now;

                fpsCounter.textContent = `FPS: ${fps}`;
                detCounter.textContent = `Detections: ${data.num_detections || 0}`;

                // Drawing boxes on client-side is faster than sending back base64 image
                // for real-time webcam
                drawDetections(data.detections);

            } catch (err) {
                console.warn('Webcam detection error:', err);
            }
        }, 300); // 3-4 fps for detection is plenty for web
    }

    function drawDetections(detections) {
        if (!detections) return;

        webcamCanvas.classList.remove('hidden');
        webcamCanvas.width = webcamVideo.offsetWidth;
        webcamCanvas.height = webcamVideo.offsetHeight;

        const ctx = webcamCanvas.getContext('2d');
        ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

        // Map YOLO coords to canvas coords
        // Note: YOLO coords from predictor are already in absolute pixels 
        // relative to original image. We need to scale them to the video display size.
        const scaleX = webcamCanvas.width / webcamVideo.videoWidth;
        const scaleY = webcamCanvas.height / webcamVideo.videoHeight;

        detections.forEach(det => {
            const { x1, y1, x2, y2 } = det.bbox;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;
            const x = x1 * scaleX;
            const y = y1 * scaleY;

            // Color based on class
            const color = det.class === 'fire' ? '#ff4d00' : '#00f2ff';

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = color;
            ctx.font = 'bold 16px Inter';
            const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
            const labelWidth = ctx.measureText(label).width;

            ctx.fillRect(x, y - 25, labelWidth + 10, 25);
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 5, y - 7);
        });
    }

    // --- Modal Logic ---
    closeModal.addEventListener('click', () => modal.classList.add('hidden'));
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.add('hidden');
    });
});
