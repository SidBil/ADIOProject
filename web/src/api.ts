import { Platform } from "react-native";
import { API_BASE } from "./config";

export async function startSession() {
  const res = await fetch(`${API_BASE}/api/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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

  const res = await fetch(
    `${API_BASE}/api/transcribe?session_id=${sessionId}`,
    { method: "POST", body: form }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function evaluate(sessionId: string, transcription: string) {
  const res = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, transcription }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function endSession(sessionId: string) {
  await fetch(`${API_BASE}/api/session/${sessionId}/end`, { method: "POST" });
}

export async function getSummary(sessionId: string) {
  const res = await fetch(`${API_BASE}/api/session/${sessionId}/summary`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function speakTTS(text: string): Promise<void> {
  const url = `${API_BASE}/api/tts?text=${encodeURIComponent(text)}`;
  const audio = new Audio(url);
  // Re-use the onended clean-up if needed, though with GET it's less critical
  // but good for consistency if we ever switch back to object URLs.
  
  try {
    // audio.play() returns a promise that resolves when playback starts
    await audio.play();
  } catch (err) {
    console.error("TTS Playback failed:", err);
  }
}

export function imageUrl(path: string) {
  return `${API_BASE}${path}`;
}
