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
import { startSession as apiStartSession } from "./src/api";
import { supabase } from "./src/lib/supabase";
import { colors } from "./src/theme";

type Screen = "welcome" | "session" | "summary" | "dashboard";

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

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session: s } }) => {
      setUser(s?.user ?? null);
      setAuthReady(true);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, s: Session | null) => {
        setUser(s?.user ?? null);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const handleStart = useCallback(async () => {
    const data = await apiStartSession();
    setSession(data);
    setScreen("session");
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
