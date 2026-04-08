import React, { useState, useCallback } from "react";
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

import WelcomeScreen from "./src/screens/WelcomeScreen";
import SessionScreen from "./src/screens/SessionScreen";
import SummaryScreen from "./src/screens/SummaryScreen";
import { startSession as apiStartSession } from "./src/api";
import { colors } from "./src/theme";

type Screen = "welcome" | "session" | "summary";

export default function App() {
  const [fontsLoaded] = useFonts({
    LeagueSpartan_400Regular,
    LeagueSpartan_800ExtraBold,
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
  });

  const [screen, setScreen] = useState<Screen>("welcome");
  const [session, setSession] = useState<any>(null);

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

  if (!fontsLoaded) {
    return (
      <View style={styles.loading}>
        <ActivityIndicator size="large" color={colors.darkBlue} />
      </View>
    );
  }

  return (
    <>
      <StatusBar style="dark" />
      {screen === "welcome" && <WelcomeScreen onStart={handleStart} />}
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
          onNewSession={handleNewSession}
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
