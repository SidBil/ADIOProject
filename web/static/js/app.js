/* ================================================================
   ADI/O — Frontend Controller
   ================================================================ */

const state = {
    sessionId: null,
    imageUrl: null,
    currentQuestion: null,
    totalQuestions: 0,
    progress: null,
    lastTranscription: null,
    isRecording: false,
    isProcessing: false,
    mediaRecorder: null,
    audioChunks: [],
};

const $ = (sel) => document.querySelector(sel);
const views = {
    welcome: $('#welcome-view'),
    session: $('#session-view'),
    summary: $('#summary-view'),
};

/* ---- Helpers ---- */
function showView(name) {
    Object.values(views).forEach(v => v.classList.remove('active'));
    views[name].classList.add('active');
}

function scoreEmoji(score) {
    if (score >= 4) return '\u{1F60A}';
    if (score >= 3) return '\u{1F610}';
    return '\u{1F641}';
}

function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.9;
    utt.pitch = 1.1;
    window.speechSynthesis.speak(utt);
}

/* ================================================================
   START SESSION
   ================================================================ */
async function startSession() {
    const btn = $('#start-btn');
    btn.textContent = 'Starting\u2026';
    btn.disabled = true;

    try {
        const res = await fetch('/api/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        state.sessionId = data.session_id;
        state.imageUrl = data.image_url;
        state.currentQuestion = data.question;
        state.totalQuestions = data.total_questions;
        state.progress = data.progress;

        renderSession();
        showView('session');
    } catch (err) {
        alert('Failed to start session. ' + err.message);
    } finally {
        btn.textContent = 'Begin a Session';
        btn.disabled = false;
    }
}

/* ================================================================
   RENDER SESSION
   ================================================================ */
function renderSession() {
    $('#session-image').src = state.imageUrl;
    renderQuestion();
    updateProgress();
    resetInteraction();
}

function renderQuestion() {
    const q = state.currentQuestion;
    if (!q) return;
    $('#question-text').textContent = q.text;
}

function updateProgress() {
    const p = state.progress;
    if (!p) return;
    const current = p.answered + 1;
    const pct = p.total ? Math.round((p.answered / p.total) * 100) : 0;
    $('#progress-text').textContent = `${current}/${p.total}`;
    $('#progress-fill').style.width = pct + '%';
}

function resetInteraction() {
    $('#transcription-area').classList.add('hidden');
    $('#feedback-area').classList.add('hidden');
    $('#feedback-area').innerHTML = '';
    $('#next-btn').classList.add('hidden');
    $('#mic-section').classList.remove('hidden');
    $('#mic-label').textContent = 'Tap to speak';
}

/* ================================================================
   RECORDING
   ================================================================ */
async function toggleRecording() {
    if (state.isProcessing) return;
    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        state.audioChunks = [];
        state.mediaRecorder.ondataavailable = (e) => state.audioChunks.push(e.data);
        state.mediaRecorder.onstop = handleRecordingComplete;
        state.mediaRecorder.start();
        state.isRecording = true;
        $('#mic-btn').classList.add('recording');
        $('#waveform').classList.remove('hidden');
        $('#mic-label').textContent = 'Listening\u2026 tap to stop';
    } catch {
        alert('Microphone access is required for this activity.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
        state.isRecording = false;
        $('#mic-btn').classList.remove('recording');
        $('#mic-btn').classList.add('processing');
        $('#waveform').classList.add('hidden');
        $('#mic-label').textContent = 'Processing\u2026';
    }
}

async function handleRecordingComplete() {
    state.isProcessing = true;

    const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
    const form = new FormData();
    form.append('audio', blob, 'recording.webm');

    try {
        const res = await fetch(`/api/transcribe?session_id=${state.sessionId}`, {
            method: 'POST',
            body: form,
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        state.lastTranscription = data.transcription || data.text;
        showTranscription();
        await evaluateResponse();
    } catch (err) {
        alert('Transcription failed. ' + err.message);
        resetInteraction();
    } finally {
        state.isProcessing = false;
        $('#mic-btn').classList.remove('processing');
    }
}

function showTranscription() {
    $('#transcription-text').textContent = state.lastTranscription;
    $('#transcription-area').classList.remove('hidden');
    $('#mic-section').classList.add('hidden');
}

/* ================================================================
   EVALUATE
   ================================================================ */
async function evaluateResponse() {
    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                transcription: state.lastTranscription,
            }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        state.currentQuestion = data.next_question;
        state.progress = data.progress;
        updateProgress();
        renderFeedback(data);

        if (data.completed) {
            $('#next-btn').textContent = 'See Summary';
        } else {
            $('#next-btn').textContent = 'Next';
        }
        $('#next-btn').classList.remove('hidden');
    } catch (err) {
        alert('Evaluation failed. ' + err.message);
    }
}

function renderFeedback(result) {
    const ev = result.evaluation || {};
    const overall = ev.overall_score || 3;
    const feedback = ev.feedback || '';
    const comment = result.followup || '';
    const display = comment || feedback;

    const emoji = scoreEmoji(overall);

    const headingMap = {
        5: 'Excellent work!',
        4: 'Great job!',
        3: 'Good effort!',
        2: 'Nice try!',
        1: 'Keep going!',
    };
    const heading = headingMap[Math.round(overall)] || 'Good effort!';

    const area = $('#feedback-area');
    area.innerHTML = `
        <div class="feedback-card">
            <span class="feedback-emoji">${emoji}</span>
            <p class="feedback-heading">${heading}</p>
            <p class="feedback-text">${display}</p>
            <div class="feedback-actions">
                <button class="tts-btn" id="tts-play-btn" title="Read aloud">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </button>
            </div>
        </div>
    `;
    area.classList.remove('hidden');

    const ttsBtn = $('#tts-play-btn');
    ttsBtn.addEventListener('click', () => {
        speakText(display);
        ttsBtn.classList.add('speaking');
        const checkDone = setInterval(() => {
            if (!window.speechSynthesis.speaking) {
                ttsBtn.classList.remove('speaking');
                clearInterval(checkDone);
            }
        }, 200);
    });
}

/* ================================================================
   NEXT / CLOSE / SUMMARY
   ================================================================ */
function handleNext() {
    window.speechSynthesis.cancel();
    if (state.progress && state.progress.completed) {
        loadSummary();
        return;
    }
    if (state.currentQuestion) {
        renderQuestion();
        resetInteraction();
    }
}

async function handleClose() {
    window.speechSynthesis.cancel();
    if (!state.sessionId) return;
    try {
        await fetch(`/api/session/${state.sessionId}/end`, { method: 'POST' });
    } catch { /* best-effort */ }
    loadSummary();
}

async function loadSummary() {
    try {
        const res = await fetch(`/api/session/${state.sessionId}/summary`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        renderSummary(data);
        showView('summary');
    } catch (err) {
        alert('Could not load summary. ' + err.message);
    }
}

function renderSummary(data) {
    const p = data.progress || {};
    const total = p.total || 0;
    const answered = p.answered || 0;
    $('#summary-stats').textContent = `You answered ${answered} question${answered !== 1 ? 's' : ''}. Great effort!`;
    $('#summary-image').src = `/images/${data.image_filename}`;

    const history = data.qa_history || [];
    const container = $('#summary-history');
    container.innerHTML = history.map(item => {
        const overall = item.evaluation?.overall_score || 3;
        const emoji = scoreEmoji(overall);
        const fb = item.followup || item.evaluation?.feedback || '';
        const feedbackHtml = fb
            ? `<div class="history-feedback"><span class="history-emoji">${emoji}</span>"${fb}"</div>`
            : '';

        return `
            <div class="history-item">
                <p class="history-q">
                    <span class="history-badge">${item.structure_word}</span>
                    ${item.question}
                </p>
                <div class="history-meta">
                    <strong>Expected:</strong> ${item.expected_answer || '\u2014'}<br>
                    <strong>You said:</strong> ${item.transcription || '\u2014'}
                </div>
                ${feedbackHtml}
            </div>
        `;
    }).join('');
}

function resetForNewSession() {
    state.sessionId = null;
    state.imageUrl = null;
    state.currentQuestion = null;
    state.totalQuestions = 0;
    state.progress = null;
    state.lastTranscription = null;
    showView('welcome');
}

/* ================================================================
   INIT
   ================================================================ */
document.addEventListener('DOMContentLoaded', () => {
    showView('welcome');
    $('#start-btn').addEventListener('click', startSession);
    $('#mic-btn').addEventListener('click', toggleRecording);
    $('#next-btn').addEventListener('click', handleNext);
    $('#new-session-btn').addEventListener('click', resetForNewSession);
    $('#close-btn').addEventListener('click', handleClose);
});
