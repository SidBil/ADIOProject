import { createClient } from "@supabase/supabase-js";
import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL ?? "";
const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY ?? "";

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.warn(
    "Supabase credentials not set. Add EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY to .env"
  );
}

const isWeb = Platform.OS === "web";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    // Native has no localStorage, so without an explicit storage adapter the
    // session lives only in memory and is lost on app restart. AsyncStorage
    // persists it to device storage; on web we keep the default (localStorage).
    ...(isWeb ? {} : { storage: AsyncStorage }),
    flowType: "implicit",
    // Only web completes OAuth via tokens in the URL; native uses setSession
    // after the deep-link callback, so URL detection must be off there.
    detectSessionInUrl: isWeb,
    autoRefreshToken: true,
    persistSession: true,
  },
});
