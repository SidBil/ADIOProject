import { useRef, useState, useCallback } from "react";
import { Platform } from "react-native";

/**
 * Real-time volume metering (0–1).
 *
 * Web:    Web Audio API AnalyserNode on the mic MediaStream.
 * Native: expo-av Recording.getStatusAsync().metering (dB → 0-1).
 */
export default function useVolumeMeter() {
  const [volume, setVolume] = useState(0);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const rafRef = useRef<number>(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordingRef = useRef<any>(null);
  const activeRef = useRef(false);

  const startMetering = useCallback(
    (streamOrRecording?: MediaStream | any) => {
      activeRef.current = true;

      if (Platform.OS === "web" && streamOrRecording instanceof MediaStream) {
        startWeb(streamOrRecording);
      } else if (Platform.OS !== "web" && streamOrRecording) {
        startNative(streamOrRecording);
      }
    },
    []
  );

  function startWeb(stream: MediaStream) {
    try {
      const ctx = new AudioContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.5;

      const source = ctx.createMediaStreamSource(stream);
      source.connect(analyser);

      ctxRef.current = ctx;
      analyserRef.current = analyser;
      sourceRef.current = source;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      function poll() {
        if (!activeRef.current) return;
        analyser.getByteFrequencyData(dataArray);

        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        const avg = sum / dataArray.length / 255;

        setVolume(Math.min(1, avg * 3.2));
        rafRef.current = requestAnimationFrame(poll);
      }
      poll();
    } catch {
      // Web Audio not available
    }
  }

  function startNative(recording: any) {
    recordingRef.current = recording;
    intervalRef.current = setInterval(async () => {
      if (!activeRef.current || !recordingRef.current) return;
      try {
        const status = await recordingRef.current.getStatusAsync();
        if (status.metering != null) {
          const db = status.metering;
          const normalized = Math.max(0, Math.min(1, (db + 60) / 60));
          setVolume(normalized);
        }
      } catch {}
    }, 80);
  }

  const stopMetering = useCallback(() => {
    activeRef.current = false;
    setVolume(0);

    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (ctxRef.current) {
      ctxRef.current.close().catch(() => {});
      ctxRef.current = null;
    }
    analyserRef.current = null;

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    recordingRef.current = null;
  }, []);

  return { volume, startMetering, stopMetering };
}
