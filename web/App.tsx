import React, { useState, useCallback, useEffect, useRef } from "react";
import { StatusBar } from "expo-status-bar";
import { View, ActivityIndicator, StyleSheet } from "react-native";
import { useFonts } from "expo-font";
import {
  LeagueSpartan_400Regular,
  LeagueSpartan_800ExtraBold,
} from "@expo-google-fonts/league-spartan";
import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
} from "@expo-google-fonts/inter";
import { Session, User } from "@supabase/supabase-js";

import LoginScreen from "./src/screens/LoginScreen";
import LandingScreen from "./src/screens/LandingScreen";
import WelcomeScreen from "./src/screens/WelcomeScreen";
import SessionScreen from "./src/screens/SessionScreen";
import SummaryScreen from "./src/screens/SummaryScreen";
import DashboardScreen from "./src/screens/DashboardScreen";
import OnboardingScreen from "./src/screens/OnboardingScreen";
import { startSession as apiStartSession } from "./src/api";
import { supabase } from "./src/lib/supabase";
import { track } from "./src/lib/analytics";
import { colors } from "./src/theme";

type Screen = "landing" | "welcome" | "session" | "summary" | "dashboard" | "onboarding";

export default function App() {
  const [fontsLoaded] = useFonts({
    LeagueSpartan_400Regular,
    LeagueSpartan_800ExtraBold,
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
  });

  const [authReady, setAuthReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [screen, setScreen] = useState<Screen>("landing");
  const [session, setSession] = useState<any>(null);

  const [hasProfile, setHasProfile] = useState<boolean | null>(null);
  // Track whether the initial auth check has already run so that subsequent
  // token refreshes / user-update events don't navigate away from an active session.
  const initialAuthDone = useRef(false);

  useEffect(() => {
    const checkProfile = async (u: User, isInitial: boolean) => {
      try {
        const { data } = await supabase.from("user_profiles").select("id").eq("id", u.id).maybeSingle();
        if (data) {
          setHasProfile(true);
          // Only navigate on the very first auth check — never interrupt an
          // active session/summary screen due to a token refresh or user-update event.
          if (isInitial) setScreen("welcome");
        } else {
          setHasProfile(false);
          if (isInitial) setScreen("onboarding");
        }
      } catch (e) {
        console.error("Profile check failed", e);
      } finally {
        setAuthReady(true);
      }
    };

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, s: Session | null) => {
        const u = s?.user ?? null;
        setUser(u);
        if (u) {
          const isInitial = !initialAuthDone.current;
          initialAuthDone.current = true;
          checkProfile(u, isInitial);
        } else {
          // Signed out — always navigate back to login
          initialAuthDone.current = false;
          setAuthReady(true);
        }
      }
    );

    // Seed initial session state from storage (runs once on mount)
    supabase.auth.getSession().then(({ data: { session: s } }) => {
      const u = s?.user ?? null;
      setUser(u);
      if (u) {
        initialAuthDone.current = true;
        checkProfile(u, true);
      } else {
        setAuthReady(true);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleStart = useCallback(async () => {
    try {
      const data = await apiStartSession();
      setSession(data);
      setScreen("session");
      track("session_started", { total_questions: data.total_questions });
    } catch (error: any) {
      track("app_error", { area: "session_start", error_message: error.message });
      throw error;
    }
  }, []);

  const handleEnd = useCallback(() => {
    setScreen("summary");
  }, []);

  const handleNewSession = useCallback(() => {
    setSession(null);
    setScreen("welcome");
  }, []);

  const handleUpdateSession = useCallback(
    (patch: any) => {
      setSession((prev: any) => (prev ? { ...prev, ...patch } : prev));
    },
    []
  );

  const handleSignOut = useCallback(async () => {
    await supabase.auth.signOut();
    setSession(null);
    setScreen("welcome");
  }, []);

  if (!fontsLoaded || !authReady) {
    return (
      <View style={styles.loading}>
        <ActivityIndicator size="large" color={colors.darkBlue} />
      </View>
    );
  }

  if (!user) {
    if (screen === "landing") {
      return (
        <>
          <StatusBar style="dark" />
          <LandingScreen
            onStartSession={() => setScreen("session")}
            onSignUp={() => setScreen("session")}
            onLogIn={() => setScreen("session")}
          />
        </>
      );
    }
    return (
      <>
        <StatusBar style="dark" />
        <LoginScreen onAuth={() => {}} />
      </>
    );
  }

  return (
    <>
      <StatusBar style="dark" />
      {screen === "onboarding" && user && (
        <OnboardingScreen 
          userId={user.id} 
          onComplete={() => {
             setHasProfile(true);
             setScreen("welcome");
          }} 
        />
      )}
      {screen === "welcome" && (
        <WelcomeScreen
          onStart={handleStart}
          onHistory={() => setScreen("dashboard")}
          onSignOut={handleSignOut}
        />
      )}
      {screen === "session" && session && (
        <SessionScreen
          session={session}
          onEnd={handleEnd}
          onUpdateSession={handleUpdateSession}
        />
      )}
      {screen === "summary" && session && (
        <SummaryScreen
          sessionId={session.session_id}
          imageId={session.image_id}
          userId={user.id}
          onNewSession={handleNewSession}
        />
      )}
      {screen === "dashboard" && (
        <DashboardScreen
          onBack={() => setScreen("welcome")}
          onStartSession={handleStart}
        />
      )}
    </>
  );
}

const styles = StyleSheet.create({
  loading: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
  },
});
