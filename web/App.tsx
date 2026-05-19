import React, { useState, useCallback, useEffect } from "react";
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
import WelcomeScreen from "./src/screens/WelcomeScreen";
import SessionScreen from "./src/screens/SessionScreen";
import SummaryScreen from "./src/screens/SummaryScreen";
import DashboardScreen from "./src/screens/DashboardScreen";
import OnboardingScreen from "./src/screens/OnboardingScreen";
import { startSession as apiStartSession } from "./src/api";
import { supabase } from "./src/lib/supabase";
import { track } from "./src/lib/analytics";
import { colors } from "./src/theme";

type Screen = "welcome" | "session" | "summary" | "dashboard" | "onboarding";

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
  const [screen, setScreen] = useState<Screen>("welcome");
  const [session, setSession] = useState<any>(null);

  const [hasProfile, setHasProfile] = useState<boolean | null>(null);

  useEffect(() => {
    const checkProfile = async (u: User) => {
      try {
        const { data } = await supabase.from("user_profiles").select("id").eq("id", u.id).maybeSingle();
        if (data) {
          setHasProfile(true);
          setScreen("welcome");
        } else {
          setHasProfile(false);
          setScreen("onboarding");
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
        if (u) checkProfile(u);
        else setAuthReady(true);
      }
    );

    supabase.auth.getSession().then(({ data: { session: s } }) => {
      const u = s?.user ?? null;
      setUser(u);
      if (u) checkProfile(u);
      else setAuthReady(true);
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
