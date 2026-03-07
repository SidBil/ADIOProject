/* ADI/O Therapy — Frontend */

const API = '';

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

/* ── Helpers ──────────────────────────────────────────────────────── */

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showView(id) {
    $$('.view').forEach(v => v.classList.remove('active'));
    const el = $(`#${id}`);
    if (el) el.classList.add('active');
}

function showToast(msg, type = 'info') {
    const div = document.createElement('div');
    div.className = `toast toast-${type}`;
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 3500);
}

async function api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return res.json();
}

function scoreClass(score) {
    return `sc-${Math.min(5, Math.max(0, Math.round(score)))}`;
}

function fillClass(score) {
    return `fill-${Math.min(5, Math.max(0, Math.round(score)))}`;
}

function textColorClass(score) {
    return `tc-${Math.min(5, Math.max(0, Math.round(score)))}`;
}

function badgeClass(word) {
    const map = {
        who: 'badge-who', what: 'badge-what', where: 'badge-where',
        color: 'badge-color', size: 'badge-size', shape: 'badge-shape',
        mood: 'badge-mood', movement: 'badge-movement',
    };
    return map[word] || 'badge-default';
}

/* ── Welcome ─────────────────────────────────────────────────────── */

async function startSession() {
    const btn = $('#start-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Preparing\u2026';

    try {
        const data = await api('/api/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });

        state.sessionId = data.session_id;
        state.imageUrl = data.image_url;
        state.currentQuestion = data.question;
        state.totalQuestions = data.total_questions;
        state.progress = data.progress;

        renderSession();
        showView('session-view');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Begin a Session';
    }
}

/* ── Session ─────────────────────────────────────────────────────── */

function renderSession() {
    $('#session-image').src = state.imageUrl;
    updateProgress();
    renderQuestion();
    clearFeedback();
}

function updateProgress() {
    const p = state.progress;
    if (!p) return;
    const pct = p.total > 0 ? (p.answered / p.total) * 100 : 0;
    $('#progress-fill').style.width = `${pct}%`;
    $('#progress-text').textContent = `Question ${p.answered + 1} of ${p.total}`;
}

function renderQuestion() {
    const q = state.currentQuestion;
    if (!q) return;
    $('#question-text').textContent = q.text;
    const badge = $('#structure-badge');
    badge.textContent = q.structure_word;
    badge.className = `structure-badge ${badgeClass(q.structure_word)}`;
}

function clearFeedback() {
    $('#transcription-area').classList.add('hidden');
    $('#feedback-area').classList.add('hidden');
    $('#feedback-area').innerHTML = '';
    $('#next-btn').classList.add('hidden');
    $('#mic-section').classList.remove('hidden');
}

/* ── Recording ───────────────────────────────────────────────────── */

async function toggleRecording() {
    if (state.isProcessing) return;
    if (state.isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.audioChunks = [];
        state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

        state.mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) state.audioChunks.push(e.data);
        };

        state.mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
            processAudio();
        };

        state.mediaRecorder.start();
        state.isRecording = true;
        updateMicUI();
    } catch (e) {
        showToast('Microphone access denied. Please allow microphone access.', 'error');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    state.isRecording = false;
    updateMicUI();
}

function updateMicUI() {
    const btn = $('#mic-btn');
    const label = $('#mic-label');
    const wave = $('#waveform');

    if (state.isRecording) {
        btn.classList.add('recording');
        label.textContent = 'Tap to stop';
        wave.classList.remove('hidden');
    } else {
        btn.classList.remove('recording');
        label.textContent = state.isProcessing ? 'Processing\u2026' : 'Tap to speak';
        wave.classList.add('hidden');
    }
}

async function processAudio() {
    state.isProcessing = true;
    updateMicUI();
    $('#mic-btn').disabled = true;
    $('#mic-label').textContent = 'Transcribing\u2026';

    const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
    const form = new FormData();
    form.append('audio', blob, 'recording.webm');

    try {
        const result = await api(`/api/transcribe?session_id=${state.sessionId}`, {
            method: 'POST',
            body: form,
        });

        state.lastTranscription = result.transcription;

        $('#transcription-text').textContent = result.transcription || '(no speech detected)';
        $('#transcription-area').classList.remove('hidden');

        await evaluateResponse(result.transcription);
    } catch (e) {
        showToast('Transcription failed: ' + e.message, 'error');
    } finally {
        state.isProcessing = false;
        $('#mic-btn').disabled = false;
        updateMicUI();
    }
}

/* ── Evaluation ──────────────────────────────────────────────────── */

async function evaluateResponse(transcription) {
    $('#mic-label').textContent = 'Evaluating\u2026';

    try {
        const result = await api('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                transcription: transcription,
            }),
        });

        renderFeedback(result);
        state.progress = result.progress;
        updateProgress();

        if (result.next_question) {
            state.currentQuestion = result.next_question;
        }

        if (result.completed) {
            $('#next-btn').textContent = 'View Summary \u2192';
        }
    } catch (e) {
        showToast('Evaluation failed: ' + e.message, 'error');
    } finally {
        updateMicUI();
    }
}

function renderFeedback(result) {
    const ev = result.evaluation;
    const area = $('#feedback-area');
    const scores = ev.scores || {};
    const categories = ['accuracy', 'detail', 'clarity', 'relevance'];
    const overall = ev.overall_score || 0;

    let scoresHtml = categories.map(cat => {
        const val = scores[cat] || 0;
        return `
            <div class="score-row">
                <span class="score-row-label">${cat}</span>
                <div class="score-bar">
                    <div class="score-bar-fill ${fillClass(val)}" style="width:${val * 20}%"></div>
                </div>
                <span class="score-row-val ${textColorClass(val)}">${val}/5</span>
            </div>`;
    }).join('');

    area.innerHTML = `
        <div class="feedback-card">
            <div class="feedback-header">
                <span class="feedback-heading">Feedback</span>
                <div class="feedback-score-badge ${scoreClass(overall)}">${overall}</div>
            </div>
            <p class="feedback-text">${ev.feedback || ''}</p>
            <div class="scores-list">${scoresHtml}</div>
            ${result.followup ? `<div class="followup-box">${result.followup}</div>` : ''}
        </div>`;

    area.classList.remove('hidden');
    $('#mic-section').classList.add('hidden');
    $('#next-btn').classList.remove('hidden');
}

function nextQuestion() {
    if (state.progress && state.progress.completed) {
        showSummary();
        return;
    }
    renderQuestion();
    clearFeedback();
}

/* ── Summary ─────────────────────────────────────────────────────── */

async function showSummary() {
    showView('summary-view');

    try {
        const data = await api(`/api/session/${state.sessionId}/summary`);
        renderSummary(data);
    } catch (e) {
        showToast('Failed to load summary', 'error');
    }
}

function renderSummary(data) {
    const avg = data.category_averages || {};
    const categories = ['accuracy', 'detail', 'clarity', 'relevance'];

    $('#summary-image').src = `/images/${data.image_filename}`;

    const overallAvg = categories.reduce((s, c) => s + (avg[c] || 0), 0) / categories.length;
    $('#summary-overall').textContent = overallAvg.toFixed(1);
    $('#summary-overall-wrap').className = `score-circle ${scoreClass(Math.round(overallAvg))}`;

    let scoresHtml = categories.map(cat => {
        const val = avg[cat] || 0;
        return `
            <div class="score-row">
                <span class="score-row-label">${cat}</span>
                <div class="score-bar">
                    <div class="score-bar-fill ${fillClass(Math.round(val))}" style="width:${val * 20}%"></div>
                </div>
                <span class="score-row-val ${textColorClass(Math.round(val))}">${val.toFixed(1)}/5</span>
            </div>`;
    }).join('');
    $('#summary-scores').innerHTML = scoresHtml;

    let histHtml = (data.qa_history || []).map((item, i) => {
        const ev = item.evaluation || {};
        const overall = ev.overall_score || 0;
        return `
            <div class="history-item">
                <div class="history-item-header">
                    <div>
                        <span class="history-q-num">Q${i + 1}</span>
                        <span class="structure-badge ${badgeClass(item.structure_word)}" style="font-size:0.7rem;padding:2px 10px;margin-left:8px;">${item.structure_word}</span>
                    </div>
                    <span class="score-row-val ${textColorClass(overall)}">${overall}/5</span>
                </div>
                <p class="history-question">${item.question}</p>
                <p class="history-detail"><em>Expected:</em> ${item.expected_answer}</p>
                <p class="history-detail"><em>You said:</em> ${item.transcription || '\u2014'}</p>
                ${ev.feedback ? `<p class="history-feedback">\u201c${ev.feedback}\u201d</p>` : ''}
            </div>`;
    }).join('');
    $('#summary-history').innerHTML = histHtml;

    const p = data.progress || {};
    $('#summary-stats').textContent =
        `You answered ${p.answered || 0} questions out of ${p.total || 0}.`;
}

function newSession() {
    state.sessionId = null;
    showView('welcome-view');
}

/* ── Init ─────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    showView('welcome-view');

    $('#start-btn').addEventListener('click', startSession);
    $('#mic-btn').addEventListener('click', toggleRecording);
    $('#next-btn').addEventListener('click', nextQuestion);
    $('#new-session-btn').addEventListener('click', newSession);
});
