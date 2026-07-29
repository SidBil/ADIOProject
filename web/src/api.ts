import { Platform } from "react-native";
import { Audio as ExpoAudio } from "expo-av";
import { API_BASE } from "./config";
import { supabase } from "./lib/supabase";

async function getAuthHeaders(extraHeaders: Record<string, string> = {}) {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return {
    ...extraHeaders,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

export async function startSession() {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/session/start`, {
    method: "POST",
    headers,
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function transcribeAudio(
  sessionId: string,
  audioUri: string,
  mimeType: string = "audio/webm"
) {
  const form = new FormData();

  if (Platform.OS === "web") {
    const blob = await fetch(audioUri).then((r) => r.blob());
    const ext = mimeType.split("/")[1] || "webm";
    const file = new File([blob], `recording.${ext}`, { type: mimeType });
    form.append("audio", file);
  } else {
    form.append("audio", {
      uri: audioUri,
      name: `recording.${mimeType.split("/")[1] || "m4a"}`,
      type: mimeType,
    } as any);
  }

  const headers = await getAuthHeaders();
  
  const res = await fetch(
    `${API_BASE}/api/transcribe?session_id=${sessionId}`,
    { method: "POST", headers, body: form }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function evaluate(
  sessionId: string,
  transcription: string,
  initiationLatencyMs?: number,
  answerDurationMs?: number
) {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      session_id: sessionId,
      transcription,
      initiation_latency_ms: initiationLatencyMs ?? null,
      answer_duration_ms: answerDurationMs ?? null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function endSession(sessionId: string) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/session/${sessionId}/end`, { method: "POST", headers });
  if (!res.ok) console.warn(`endSession failed: ${res.status} ${await res.text().catch(() => "")}`);
}

// Child celebration: only a completion signal + a single flourish boolean.
// No accuracy number crosses this boundary (PRD 01_session_summary §3.4).
export async function getCelebration(sessionId: string) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/session/${sessionId}/celebration`, { headers });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Home path state: the durable completed-session count (+ streak) the trail
// derives from. Read from Supabase, not in-memory session state, so it renders
// correctly on a cold serverless instance (03_navigation_shell.md §6).
export async function getHomeState() {
  const headers = await getAuthHeaders();
  // Time-box the request: if the backend is unreachable (e.g. wrong dev host),
  // reject after 8s so Home can render the trail instead of spinning forever.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/home/state`, { headers, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    completed_count: number;
    streak_days: number;
    last_session_at: string | null;
    // Image filename per completed session, oldest-first (node N → index N-1).
    completed_images?: string[];
    child_name?: string | null;
  }>;
}

// Parent/SLP dashboard: recent sessions + per-structure-word sparklines.
export async function getDashboardOverview() {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/dashboard/overview`, { headers });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Parent/SLP clinical detail for one session (per-structure-word breakdown).
export async function getSessionDetail(sessionId: string) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/session/${sessionId}/detail`, { headers });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function speakTTS(text: string): Promise<void> {
  const url = `${API_BASE}/api/tts?text=${encodeURIComponent(text)}`;

  // Web: the browser Audio element. `play()` resolves when playback starts.
  if (Platform.OS === "web") {
    try {
      const audio = new Audio(url);
      await audio.play();
    } catch (err) {
      console.error("TTS playback failed:", err);
    }
    return;
  }

  // Native (iOS/Android): `new Audio()` doesn't exist here — play via expo-av.
  // Set the audio mode so it plays through the speaker even with the ringer on
  // silent, and isn't stuck in the recording session's config.
  try {
    await ExpoAudio.setAudioModeAsync({
      allowsRecordingIOS: false,
      playsInSilentModeIOS: true,
    });
    const { sound } = await ExpoAudio.Sound.createAsync({ uri: url }, { shouldPlay: true });
    // Free the sound once it finishes so repeated feedback doesn't leak players.
    sound.setOnPlaybackStatusUpdate((status) => {
      if (status.isLoaded && status.didJustFinish) {
        sound.unloadAsync().catch(() => {});
      }
    });
  } catch (err) {
    console.error("TTS playback failed:", err);
  }
}

export function imageUrl(path: string) {
  return `${API_BASE}${path}`;
}
