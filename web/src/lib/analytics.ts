import { initializeApp } from "firebase/app";
import { getAnalytics, logEvent, isSupported } from "firebase/analytics";

const firebaseConfig = {
  apiKey: process.env.EXPO_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID,
  appId: process.env.EXPO_PUBLIC_FIREBASE_APP_ID,
  measurementId: process.env.EXPO_PUBLIC_FIREBASE_MEASUREMENT_ID,
};

let analyticsPromise: Promise<ReturnType<typeof getAnalytics> | null> | null = null;

function getAnalyticsClient() {
  if (!analyticsPromise) {
    // Only initialize if we have config
    if (!firebaseConfig.apiKey) {
      console.warn("Firebase Analytics disabled: Missing EXPO_PUBLIC_FIREBASE_API_KEY");
      analyticsPromise = Promise.resolve(null);
      return analyticsPromise;
    }
    
    analyticsPromise = isSupported().then((supported) => {
      if (!supported) return null;
      const app = initializeApp(firebaseConfig);
      return getAnalytics(app);
    });
  }
  return analyticsPromise;
}

export async function track(eventName: string, params: Record<string, any> = {}) {
  const analytics = await getAnalyticsClient();
  if (!analytics) return;

  // Never send PII or raw therapy content to Firebase Analytics
  const safeParams = Object.fromEntries(
    Object.entries(params).filter(([key]) => ![
      "email",
      "name",
      "transcription",
      "expected_answer",
      "audio_path",
      "llm_feedback",
    ].includes(key))
  );

  logEvent(analytics, eventName, safeParams);
}
