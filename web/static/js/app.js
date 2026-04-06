/* ================================================================
   ADI/O — Controller
   ================================================================ */

const MIC_IMG = '<img src="/static/images/micV3.png" alt="mic" style="width:100%;height:100%;object-fit:contain;">';

const STAR_PATH = 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z';

const HEADING_MAP = { 5: 'Excellent!', 4: 'Great Job!', 3: 'Good Effort!', 2: 'Nice Try!', 1: 'Keep Going!' };

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

const $ = (s) => document.querySelector(s);

function showView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    $(`#${name}-view`).classList.add('active');
}

/* ---- Stars (rounded) ---- */
function starsHTML(score) {
    let html = '<div style="display:flex;justify-content:center;gap:10px;margin-bottom:0;">';
    for (let i = 1; i <= 5; i++) {
        const fill = i <= score ? '#F71D73' : '#001543';
        html += `<svg viewBox="0 0 24 24" width="44" height="44" xmlns="http://www.w3.org/2000/svg"><path d="${STAR_PATH}" fill="${fill}" stroke="${fill}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
    }
    html += '</div>';
    return html;
}

/* ---- Progress ---- */
function updateProgress() {
    const p = state.progress;
    if (!p) return;
    const current = Math.min(p.answered + 1, p.total);
    const pct = p.total ? Math.round((p.answered / p.total) * 100) : 0;
    $('#progress-text').textContent = `${current}/${p.total}`;
    $('#progress-fill').style.width = pct + '%';
}

/* ================================================================
   SIDEBAR CARD — STATE 1: Question + Mic
   ================================================================ */
function renderQuestionCard() {
    const q = state.currentQuestion;
    if (!q) return;
    const card = $('#sidebar-card');
    card.innerHTML = `
        <div style="background:#FFDEE9;border:5px solid #EF1A6A;border-radius:30px;padding:16px 14px;box-shadow:0 10px 0 #EF1A6A;">
            <p style="color:#002248;font-family:'League Spartan',sans-serif;font-weight:800;font-size:32px;text-align:center;line-height:1.35;margin:0 0 10px 0;">
                ${q.text}
            </p>
            <div id="yellow-card" style="background:#FFF6C1;border:5px solid #fbde28;border-radius:26px;padding:24px 20px 20px 20px;display:flex;flex-direction:column;align-items:center;gap:10px;box-shadow:0 10px 0 #fbde28;transition:box-shadow 0.08s,transform 0.08s;">
                <button id="mic-btn-inner"
                        style="width:160px;height:160px;border-radius:50%;background:transparent;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;padding:0;overflow:hidden;">
                    ${MIC_IMG}
                </button>
                <span id="mic-label" style="color:#002248;font-family:'Inter',sans-serif;font-weight:600;font-size:24px;text-align:center;">
                    Tap to Answer
                </span>
            </div>
        </div>
    `;
    $('#mic-btn-inner').addEventListener('click', toggleRecording);
}

/* ================================================================
   SIDEBAR CARD — STATE 2: Feedback
   ================================================================ */
function renderFeedbackCard(result) {
    const ev = result.evaluation || {};
    const score = Math.round(ev.overall_score || 3);
    const heading = HEADING_MAP[score] || 'Good Effort!';
    const comment = result.followup || ev.feedback || '';

    const card = $('#sidebar-card');
    card.innerHTML = `
        <div style="background:#C8EFFF;border:5px solid #29A5E1;border-radius:30px;padding:28px 24px;text-align:center;box-shadow:0 10px 0 #29A5E1;">
            <p style="color:#002252;font-family:'League Spartan',sans-serif;font-weight:800;font-size:44px;margin:0 0 12px 0;">
                ${heading}
            </p>
            ${starsHTML(score)}
            <p style="color:#002252;font-family:'Inter',sans-serif;font-weight:400;font-size:24px;text-align:center;line-height:1.55;margin:16px 0 24px 0;">
                ${comment}
            </p>
            <button id="next-btn-inner" class="bevel-next"
                    style="background:#F3FFAD;border:5px solid #BCD533;border-radius:999px;padding:16px 0;width:80%;margin:0 auto;display:block;cursor:pointer;">
                <span style="color:#101101;font-family:'League Spartan',sans-serif;font-weight:800;font-size:30px;">Next</span>
            </button>
        </div>
    `;
    $('#next-btn-inner').addEventListener('click', handleNext);
}

/* ================================================================
   SESSION
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
        $('#session-image').src = state.imageUrl;
        updateProgress();
        renderQuestionCard();
        showView('session');
    } catch (err) {
        alert('Failed to start session. ' + err.message);
    } finally {
        btn.textContent = 'Begin a Session';
        btn.disabled = false;
    }
}

/* ================================================================
   RECORDING
   ================================================================ */
async function toggleRecording() {
    if (state.isProcessing) return;
    state.isRecording ? stopRecording() : await startRecording();
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
        const btn = $('#mic-btn-inner');
        if (btn) btn.classList.add('mic-recording');
        const yc = $('#yellow-card');
        if (yc) { yc.style.boxShadow = '0 2px 0 #fbde28'; yc.style.transform = 'translateY(8px)'; }
        const lbl = $('#mic-label');
        if (lbl) lbl.textContent = 'Listening\u2026 tap to stop';
    } catch {
        alert('Microphone access is required.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
        state.isRecording = false;
        const btn = $('#mic-btn-inner');
        if (btn) { btn.classList.remove('mic-recording'); btn.style.opacity = '0.5'; btn.style.cursor = 'wait'; }
        const yc = $('#yellow-card');
        if (yc) { yc.style.boxShadow = '0 10px 0 #fbde28'; yc.style.transform = 'translateY(0)'; }
        const lbl = $('#mic-label');
        if (lbl) lbl.textContent = 'Processing\u2026';
    }
}

async function handleRecordingComplete() {
    state.isProcessing = true;
    const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
    const form = new FormData();
    form.append('audio', blob, 'recording.webm');

    try {
        const tres = await fetch(`/api/transcribe?session_id=${state.sessionId}`, { method: 'POST', body: form });
        if (!tres.ok) throw new Error(await tres.text());
        const tdata = await tres.json();
        state.lastTranscription = tdata.transcription || tdata.text;

        const eres = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.sessionId, transcription: state.lastTranscription }),
        });
        if (!eres.ok) throw new Error(await eres.text());
        const edata = await eres.json();

        state.currentQuestion = edata.next_question;
        state.progress = edata.progress;
        updateProgress();
        renderFeedbackCard(edata);
    } catch (err) {
        alert('Something went wrong. ' + err.message);
        renderQuestionCard();
    } finally {
        state.isProcessing = false;
    }
}

/* ================================================================
   NEXT / CLOSE / SUMMARY
   ================================================================ */
function handleNext() {
    if (state.progress && state.progress.completed) {
        loadSummary();
    } else if (state.currentQuestion) {
        renderQuestionCard();
    }
}

async function handleClose() {
    if (!state.sessionId) return;
    try { await fetch(`/api/session/${state.sessionId}/end`, { method: 'POST' }); } catch {}
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
    $('#summary-stats').textContent = `You answered ${p.answered || 0} question${(p.answered||0) !== 1 ? 's' : ''}. Great effort!`;
    $('#summary-image').src = `/images/${data.image_filename}`;

    const history = data.qa_history || [];
    const container = $('#summary-history');
    container.innerHTML = history.map(item => {
        const fb = item.followup || item.evaluation?.feedback || '';
        const fbHTML = fb ? `<p class="h-fb">"${fb}"</p>` : '';
        return `
            <div class="history-item">
                <p class="h-q">${item.question}</p>
                <p><strong>Expected:</strong> ${item.expected_answer || '\u2014'}</p>
                <p><strong>You said:</strong> ${item.transcription || '\u2014'}</p>
                ${fbHTML}
            </div>`;
    }).join('');
}

function resetForNewSession() {
    Object.assign(state, { sessionId: null, imageUrl: null, currentQuestion: null, totalQuestions: 0, progress: null, lastTranscription: null });
    showView('welcome');
}

/* ================================================================
   INIT
   ================================================================ */
document.addEventListener('DOMContentLoaded', () => {
    showView('welcome');
    $('#start-btn').addEventListener('click', startSession);
    $('#close-btn').addEventListener('click', handleClose);
    $('#new-session-btn').addEventListener('click', resetForNewSession);
});
