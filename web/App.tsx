import React, { useState, useCallback, useEffect, useRef } from "react";
import { StatusBar } from "expo-status-bar";
import { View, StyleSheet, Platform } from "react-native";
import { useFonts } from "expo-font";
import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
} from "@expo-google-fonts/inter";
import {
  Fredoka_400Regular,
  Fredoka_600SemiBold,
  Fredoka_700Bold,
} from "@expo-google-fonts/fredoka";
import { Session, User } from "@supabase/supabase-js";

import LoginScreen from "./src/screens/LoginScreen";
import LandingScreen from "./src/screens/LandingScreen";
import HomeScreen from "./src/screens/HomeScreen";
import CollectionScreen from "./src/screens/CollectionScreen";
import AdultScreen from "./src/screens/AdultScreen";
import SessionScreen from "./src/screens/SessionScreen";
import CelebrationScreen from "./src/screens/CelebrationScreen";
import DashboardScreen from "./src/screens/DashboardScreen";
import OnboardingScreen from "./src/screens/OnboardingScreen";
import AboutScreen from "./src/screens/AboutScreen";
import TabBar, { TabKey } from "./src/components/TabBar";
import { startSession as apiStartSession } from "./src/api";
import { supabase } from "./src/lib/supabase";
import { track } from "./src/lib/analytics";
import { colors } from "./src/theme";
import Spinner from "./src/components/Spinner";

// Top-level surfaces. The four tab keys are the persistent child-facing shell
// (03_navigation_shell.md); `session`, `summary`, `onboarding`, `about` are
// full-screen states that take over (tab bar hidden). `landing` is logged-out.
type Screen =
  | "landing"
  | "home"
  | "collection"
  | "progress"
  | "adult"
  | "session"
  | "summary"
  | "onboarding"
  | "about";

const TAB_SCREENS: TabKey[] = ["home", "collection", "progress", "adult"];
const isTabScreen = (s: Screen): s is TabKey => (TAB_SCREENS as string[]).includes(s);

export default function App() {
  const [fontsLoaded] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Fredoka_400Regular,
    Fredoka_600SemiBold,
    Fredoka_700Bold,
  });

  const [authReady, setAuthReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [screen, setScreen] = useState<Screen>(Platform.OS === "web" ? "landing" : "home");
  const [session, setSession] = useState<any>(null);
  // Bumped whenever the child returns to Home from a completed session so the
  // trail refetches its position (the completed-count advanced).
  const [homeRefresh, setHomeRefresh] = useState(0);

  const [hasProfile, setHasProfile] = useState<boolean | null>(null);
  // Track whether the initial auth check has already run so that subsequent
  // token refreshes / user-update events don't navigate away from an active session.
  const initialAuthDone = useRef(false);

  // Auto-sign-in with dev account on localhost so Google OAuth redirect isn't needed
  useEffect(() => {
    if (typeof window !== "undefined" && window.location?.hostname === "localhost") {
      supabase.auth.signInWithPassword({
        email: "dev@adio.local",
        password: "devlocal123",
      });
    }
  }, []);

  useEffect(() => {
    const checkProfile = async (u: User, isInitial: boolean) => {
      try {
        const { data } = await supabase.from("user_profiles").select("id").eq("id", u.id).maybeSingle();
        if (data) {
          setHasProfile(true);
          // Only navigate on the very first auth check — never interrupt an
          // active session/summary screen due to a token refresh or user-update event.
          // After auth the child lands on the Home tab, not a Begin button.
          if (isInitial) setScreen("home");
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
      track("session_started", { total_questions: data.total_questions, mode: data.mode });
    } catch (error: any) {
      track("app_error", { area: "session_start", error_message: error.message });
      throw error;
    }
  }, []);

  const returnHome = useCallback(() => {
    setSession(null);
    setScreen("home");
    // Advance the trail: the completed-count moved, so refetch Home position.
    setHomeRefresh((n) => n + 1);
  }, []);

  const handleEnd = useCallback((completed: boolean) => {
    // Only a fully completed session reaches the child celebration. An early
    // quit exits straight home with no celebration (PRD 01_session_summary §3.2).
    if (completed) {
      setScreen("summary");
    } else {
      returnHome();
    }
  }, [returnHome]);

  const handleUpdateSession = useCallback(
    (patch: any) => {
      setSession((prev: any) => (prev ? { ...prev, ...patch } : prev));
    },
    []
  );

  const handleSignOut = useCallback(async () => {
    await supabase.auth.signOut();
    setSession(null);
    setScreen("home");
  }, []);

  if (!fontsLoaded || !authReady) {
    return (
      <View style={styles.loading}>
        <Spinner size="large" />
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
            onAbout={() => setScreen("about")}
          />
        </>
      );
    }
    if (screen === "about") {
      return (
        <>
          <StatusBar style="dark" />
          <AboutScreen onBack={() => setScreen("landing")} />
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

  // ── Full-screen states (tab bar hidden) ──
  if (screen === "onboarding") {
    return (
      <>
        <StatusBar style="dark" />
        <OnboardingScreen
          userId={user.id}
          onComplete={() => {
            setHasProfile(true);
            setScreen("home");
          }}
        />
      </>
    );
  }

  if (screen === "session" && session) {
    return (
      <>
        <StatusBar style="dark" />
        <SessionScreen
          session={session}
          onEnd={handleEnd}
          onUpdateSession={handleUpdateSession}
        />
      </>
    );
  }

  if (screen === "summary" && session) {
    // Celebration plays first, then drops the child back onto the Home path
    // with the just-finished node completed (01_home_screen.md §6).
    return (
      <>
        <StatusBar style="dark" />
        <CelebrationScreen sessionId={session.session_id} onDone={returnHome} />
      </>
    );
  }

  if (screen === "about") {
    return (
      <>
        <StatusBar style="dark" />
        <AboutScreen onBack={() => setScreen("adult")} />
      </>
    );
  }

  // ── Tab shell: persistent bottom bar wrapping the child-facing surfaces ──
  const activeTab: TabKey = isTabScreen(screen) ? screen : "home";
  return (
    <>
      <StatusBar style="dark" />
      <View style={styles.shell}>
        <View style={styles.tabContent}>
          {activeTab === "home" && (
            <HomeScreen onStart={handleStart} refreshKey={homeRefresh} />
          )}
          {activeTab === "collection" && <CollectionScreen />}
          {activeTab === "progress" && (
            <DashboardScreen onBack={() => setScreen("home")} />
          )}
          {activeTab === "adult" && (
            <AdultScreen userId={user.id} onAbout={() => setScreen("about")} onSignOut={handleSignOut} />
          )}
        </View>
        <TabBar active={activeTab} onChange={(t) => setScreen(t)} />
      </View>
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
  shell: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  tabContent: {
    flex: 1,
  },
});
