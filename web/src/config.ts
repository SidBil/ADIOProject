import { NativeModules, Platform } from "react-native";
import Constants from "expo-constants";

const PORT = 8000;

// Last-resort host for a PHYSICAL device when the dev-server host can't be
// auto-detected. Normally unused (auto-detection below handles it), but it keeps
// the app working if detection ever comes back empty. Update if your Mac's IP
// changes and auto-detection isn't kicking in.
const FALLBACK_LAN_IP = "192.168.86.39";

/**
 * The dev machine's host, derived from whatever is serving the app so the API
 * always points at the same machine (no hardcoded IP to babysit):
 *   1. Expo's dev-server host (`hostUri`, e.g. "192.168.86.39:8081") — reliable
 *      on a dev build connected to Metro; reports "localhost" on the simulator.
 *   2. Metro's bundle URL as a secondary source.
 * Returns null if neither is available.
 */
function detectDevHost(): string | null {
  const hostUri: string =
    (Constants as any)?.expoConfig?.hostUri ||
    (Constants as any)?.expoGoConfig?.debuggerHost ||
    (Constants as any)?.manifest2?.extra?.expoGo?.debuggerHost ||
    "";
  const fromExpo = hostUri.split(":")[0];
  if (fromExpo) return fromExpo;

  const scriptURL: string | undefined = (NativeModules as any)?.SourceCode?.scriptURL;
  const fromScript = scriptURL?.match(/^https?:\/\/([^:/]+)/)?.[1];
  return fromScript || null;
}

function devApiBase(): string {
  if (Platform.OS === "web") return `http://localhost:${PORT}`;
  const host = detectDevHost();
  // A detected host is correct for both simulator ("localhost") and device (LAN
  // IP). Only when detection fails do we fall back to the known LAN IP.
  if (host) return `http://${host}:${PORT}`;
  return `http://${FALLBACK_LAN_IP}:${PORT}`;
}

export const API_BASE = __DEV__ ? devApiBase() : "";
