// Constants
const EYE_CLOSED_THRESH = 0.18;
const DROWSY_THRESH = 0.25;
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const LEFT_IRIS = [474, 475, 476, 477];
const RIGHT_IRIS = [469, 470, 471, 472];

// State
let isRunning = false;
let drowsyFrames = 0;
let distractionCount = 0;
let drowsyCount = 0;
let startTime = null;
let alertAudio = null;
let lastAlertTime = 0;

// Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusCard = document.getElementById('statusCard');
const statusText = document.getElementById('status');
const earValue = document.getElementById('earValue');
const distractionCountEl = document.getElementById('distractionCount');
const drowsyCountEl = document.getElementById('drowsyCount');
const sessionTimeEl = document.getElementById('sessionTime');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const alertEl = document.getElementById('alert');

// Initialize audio
function initAudio() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    alertAudio = {
        context: audioContext,
        play: function() {
            const oscillator = this.context.createOscillator();
            const gainNode = this.context.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(this.context.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.3, this.context.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, this.context.currentTime + 0.5);
            
            oscillator.start(this.context.currentTime);
            oscillator.stop(this.context.currentTime + 0.5);
        }
    };
}

// Calculate Eye Aspect Ratio
function calculateEAR(landmarks, eyeIndices) {
    const points = eyeIndices.map(i => landmarks[i]);
    
    let vertical = 0;
    let count = 0;
    for (let i = 0; i < points.length / 2; i++) {
        const p1 = points[i];
        const p2 = points[points.length - 1 - i];
        const dist = Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
        vertical += dist;
        count++;
    }
    vertical /= count;
    
    const horizontal = Math.sqrt(
        Math.pow(points[0].x - points[Math.floor(points.length / 2)].x, 2) +
        Math.pow(points[0].y - points[Math.floor(points.length / 2)].y, 2)
    );
    
    return vertical / (horizontal + 1e-6);
}

// Get iris center
function getIrisCenter(landmarks, indices) {
    let x = 0, y = 0;
    indices.forEach(i => {
        x += landmarks[i].x;
        y += landmarks[i].y;
    });
    return { x: x / indices.length, y: y / indices.length };
}

// Get eye box
function getEyeBox(landmarks, indices) {
    const xs = indices.map(i => landmarks[i].x);
    const ys = indices.map(i => landmarks[i].y);
    return {
        min: { x: Math.min(...xs), y: Math.min(...ys) },
        max: { x: Math.max(...xs), y: Math.max(...ys) }
    };
}

// Check if iris is centered
function isCentered(iris, box, threshold = 0.25) {
    const cx = (box.min.x + box.max.x) / 2;
    const cy = (box.min.y + box.max.y) / 2;
    const ox = (box.max.x - box.min.x) * threshold;
    const oy = (box.max.y - box.min.y) * threshold;
    
    return Math.abs(iris.x - cx) < ox && Math.abs(iris.y - cy) < oy;
}

// Show alert
function showAlert() {
    const now = Date.now();
    if (now - lastAlertTime > 3000) {
        alertEl.classList.add('show');
        if (alertAudio) alertAudio.play();
        
        setTimeout(() => {
            alertEl.classList.remove('show');
        }, 2000);
        
        lastAlertTime = now;
        drowsyCount++;
        drowsyCountEl.textContent = drowsyCount;
    }
}

// Update session time
function updateSessionTime() {
    if (startTime && isRunning) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        sessionTimeEl.textContent = `${elapsed}s`;
    }
}

// MediaPipe Face Mesh
const faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

faceMesh.onResults((results) => {
    if (!isRunning) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    let status = 'NO FACE';
    let distracted = true;
    let avgEAR = 0;
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        
        // Calculate EAR
        const leftEAR = calculateEAR(landmarks, LEFT_EYE);
        const rightEAR = calculateEAR(landmarks, RIGHT_EYE);
        avgEAR = (leftEAR + rightEAR) / 2.0;
        
        // Check drowsiness
        if (avgEAR < DROWSY_THRESH) {
            drowsyFrames++;
            if (drowsyFrames >= 10) {
                status = 'DROWSY';
                showAlert();
            }
        } else {
            drowsyFrames = 0;
        }
        
        // Check eye closure
        const eyesClosed = avgEAR < EYE_CLOSED_THRESH;
        
        // Get iris positions
        const leftIris = getIrisCenter(landmarks, LEFT_IRIS);
        const rightIris = getIrisCenter(landmarks, RIGHT_IRIS);
        const leftBox = getEyeBox(landmarks, LEFT_EYE);
        const rightBox = getEyeBox(landmarks, RIGHT_EYE);
        
        // Draw eyes
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        LEFT_EYE.forEach((idx, i) => {
            const point = landmarks[idx];
            const nextIdx = LEFT_EYE[(i + 1) % LEFT_EYE.length];
            const nextPoint = landmarks[nextIdx];
            ctx.beginPath();
            ctx.moveTo(point.x * canvas.width, point.y * canvas.height);
            ctx.lineTo(nextPoint.x * canvas.width, nextPoint.y * canvas.height);
            ctx.stroke();
        });
        
        ctx.strokeStyle = '#FFFF00';
        RIGHT_EYE.forEach((idx, i) => {
            const point = landmarks[idx];
            const nextIdx = RIGHT_EYE[(i + 1) % RIGHT_EYE.length];
            const nextPoint = landmarks[nextIdx];
            ctx.beginPath();
            ctx.moveTo(point.x * canvas.width, point.y * canvas.height);
            ctx.lineTo(nextPoint.x * canvas.width, nextPoint.y * canvas.height);
            ctx.stroke();
        });
        
        // Draw iris
        if (!eyesClosed) {
            ctx.fillStyle = '#00FF00';
            ctx.beginPath();
            ctx.arc(leftIris.x * canvas.width, leftIris.y * canvas.height, 3, 0, 2 * Math.PI);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(rightIris.x * canvas.width, rightIris.y * canvas.height, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Check focus
        if (eyesClosed) {
            status = 'EYES CLOSED';
        } else if (isCentered(leftIris, leftBox) && isCentered(rightIris, rightBox)) {
            status = 'FOCUSED';
            distracted = false;
        } else {
            status = 'DISTRACTED';
        }
        
        if (distracted) {
            distractionCount++;
            distractionCountEl.textContent = distractionCount;
        }
    }
    
    // Update UI
    statusText.textContent = status;
    earValue.textContent = avgEAR.toFixed(2);
    
    statusCard.className = 'status-card';
    if (status === 'FOCUSED') statusCard.classList.add('focused');
    else if (status === 'DROWSY') statusCard.classList.add('drowsy');
    else if (status === 'DISTRACTED' || status === 'EYES CLOSED') statusCard.classList.add('distracted');
});

// Camera
const camera = new Camera(video, {
    onFrame: async () => {
        if (isRunning) {
            await faceMesh.send({ image: video });
        }
    },
    width: 1280,
    height: 720
});

// Start button
startBtn.addEventListener('click', async () => {
    initAudio();
    await camera.start();
    isRunning = true;
    startTime = Date.now();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    setInterval(updateSessionTime, 1000);
});

// Stop button
stopBtn.addEventListener('click', () => {
    isRunning = false;
    camera.stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});