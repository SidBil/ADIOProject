import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Image,
  Platform,
  KeyboardAvoidingView,
  ScrollView,
  Pressable,
} from "react-native";
import * as WebBrowser from "expo-web-browser";
import * as Linking from "expo-linking";
import { colors, fonts } from "../theme";
import Spinner from "../components/Spinner";
import { supabase } from "../lib/supabase";
import { track } from "../lib/analytics";

/* eslint-disable @typescript-eslint/no-require-imports */
const privacyPdf = require("../constants/legal/Adio_Privacy_Policy.docx.pdf");
const termsPdf = require("../constants/legal/Adio_Terms_and_Conditions.docx.pdf");

const getAssetUri = (source: any) => {
  if (typeof source === "string") return source;
  return Image.resolveAssetSource(source)?.uri || source;
};

interface Props {
  onAuth: () => void;
}

export default function LoginScreen({ onAuth }: Props) {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [burstCount, setBurstCount] = useState(0);
  const [clickCenter, setClickCenter] = useState<{ x: number; y: number } | undefined>();

  async function handleEmailAuth(e: any) {
    if (e?.nativeEvent) {
      setClickCenter({ x: e.nativeEvent.pageX, y: e.nativeEvent.pageY });
      setBurstCount((c) => c + 1);
    }
    if (!email.trim() || !password.trim()) {
      setError("Please enter email and password.");
      return;
    }
    setLoading(true);
    setError(null);

    track("auth_started", { method: "email", is_signup: isSignUp });

    const { error: authError } = isSignUp
      ? await supabase.auth.signUp({ email: email.trim(), password })
      : await supabase.auth.signInWithPassword({ email: email.trim(), password });

    setLoading(false);

    if (authError) {
      setError(authError.message);
      track("app_error", { area: "auth", error_code: authError.code });
    } else {
      track("auth_completed", { method: "email", success: true });
      if (isSignUp) {
      setError(null);
      setIsSignUp(false);
      if (Platform.OS === "web") window.alert("Check your email to confirm your account.");
      else setError("Check your email to confirm your account.");
      }
    }
  }

  async function handleGoogleAuth(e: any) {
    if (e?.nativeEvent) {
      setClickCenter({ x: e.nativeEvent.pageX, y: e.nativeEvent.pageY });
      setBurstCount((c) => c + 1);
    }
    setLoading(true);
    setError(null);
    track("auth_started", { method: "google" });

    if (Platform.OS === "web") {
      const { error: oauthError } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: window.location.origin },
      });
      setLoading(false);
      if (oauthError) {
        setError(oauthError.message);
        track("app_error", { area: "auth", error_code: oauthError.code });
      }
      return;
    }

    // iOS/Android: open Google OAuth in an in-app browser and handle deep link callback
    const redirectTo = Linking.createURL("/");
    const { data, error: oauthError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo, skipBrowserRedirect: true },
    });

    if (oauthError || !data?.url) {
      setLoading(false);
      setError(oauthError?.message ?? "Could not start Google sign-in.");
      track("app_error", { area: "auth", error_code: oauthError?.code });
      return;
    }

    const result = await WebBrowser.openAuthSessionAsync(data.url, redirectTo);
    setLoading(false);

    if (result.type === "success") {
      // Implicit flow returns tokens in the hash fragment: adio-therapy:///#access_token=...
      const hash = result.url.split("#")[1] ?? "";
      const params = Object.fromEntries(new URLSearchParams(hash));
      const accessToken = params.access_token;
      const refreshToken = params.refresh_token;
      if (!accessToken || !refreshToken) {
        setError("Sign-in failed: no tokens returned.");
        return;
      }
      const { error: sessionError } = await supabase.auth.setSession({ access_token: accessToken, refresh_token: refreshToken });
      if (sessionError) {
        setError(sessionError.message);
        track("app_error", { area: "auth", error_code: sessionError.code });
        return;
      }
      track("auth_completed", { method: "google", success: true });
    }
  }

  function handleSwitch(e: any) {
    if (e?.nativeEvent) {
      setClickCenter({ x: e.nativeEvent.pageX, y: e.nativeEvent.pageY });
      setBurstCount((c) => c + 1);
    }
    setIsSignUp(!isSignUp);
    setError(null);
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.content}>
          <Image
            source={require("../../assets/adiologo2.png")}
            style={styles.logo}
            resizeMode="contain"
          />
          <Text style={styles.title}>
            {isSignUp ? "Create Account" : "Welcome Back"}
          </Text>

          <TextInput
            style={styles.input}
            placeholder="Email"
            placeholderTextColor={colors.textMuted}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
            value={email}
            onChangeText={setEmail}
          />
          <TextInput
            style={styles.input}
            placeholder="Password"
            placeholderTextColor={colors.textMuted}
            secureTextEntry
            value={password}
            onChangeText={setPassword}
          />

          {error && <Text style={styles.errorText}>{error}</Text>}

          <ClayButton
            title={isSignUp ? "Sign Up" : "Sign In"}
            onPress={handleEmailAuth}
            loading={loading}
            topColor={colors.blueCard}
            accent="41,165,225"
            textColor={colors.darkBlue}
          />

          {isSignUp && (
            <Text style={styles.legalText}>
              By signing up, you agree to our{" "}
              <Text
                style={styles.legalLink}
                onPress={() => Platform.OS === "web" && window.open(getAssetUri(termsPdf), "_blank")}
              >
                Terms & Conditions
              </Text>
              {" and "}
              <Text
                style={styles.legalLink}
                onPress={() => Platform.OS === "web" && window.open(getAssetUri(privacyPdf), "_blank")}
              >
                Privacy Policy
              </Text>
              .
            </Text>
          )}

          <View style={styles.dividerRow}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>or</Text>
            <View style={styles.dividerLine} />
          </View>

          <ClayButton
            title="Sign in with Google"
            onPress={handleGoogleAuth}
            loading={loading}
            topColor={colors.greenBtn}
            accent="188,213,51"
            textColor={colors.darkBlue}
          />

          <Pressable
            onPress={handleSwitch}
            style={styles.switchRow}
            hitSlop={10}
          >
            <Text style={styles.switchText}>
              {isSignUp
                ? "Already have an account? Sign In"
                : "Don't have an account? Sign Up"}
            </Text>
          </Pressable>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  content: {
    width: "100%",
    maxWidth: 420,
    alignItems: "center",
    paddingHorizontal: 8,
  },
  logo: { 
    height: 180, 
    width: 320, 
    marginTop: -40, 
    marginBottom: -25 
  },
  title: {
    fontFamily: fonts.heading,
    fontSize: 28,
    color: colors.darkBlue,
    textAlign: "center",
    marginBottom: 24,
  },
  input: {
    width: "100%",
    borderWidth: 3,
    borderColor: "#e0e0e8",
    borderRadius: 999,
    paddingVertical: 14,
    paddingHorizontal: 22,
    fontFamily: fonts.body,
    fontSize: 17,
    color: colors.darkBlueText,
    marginBottom: 14,
    backgroundColor: "#ffffff",
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.pinkBorder,
    textAlign: "center",
    marginBottom: 10,
  },
  primaryBtn: {
    backgroundColor: colors.darkBlueBtnBg,
    borderRadius: 14,
    paddingVertical: 16,
    width: "100%",
    alignItems: "center",
    marginTop: 4,
  },
  primaryBtnText: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.white,
  },
  dividerRow: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    marginVertical: 18,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#e0e0e8",
  },
  dividerText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlue,
    marginHorizontal: 14,
  },
  switchRow: {
    marginTop: 22,
  },
  switchText: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.darkBlue,
    textAlign: "center",
  },
  legalText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.textMuted || "#666680",
    textAlign: "center",
    marginTop: 16,
    lineHeight: 20,
    paddingHorizontal: 10,
  },
  legalLink: {
    color: colors.darkBlue,
    textDecorationLine: "underline",
    fontFamily: fonts.bodySemiBold,
  },
});

/* ═══════════════════════════════════════════════════════════════
   Claymorphic button — outer drop shadow + a brighter inset sculpt in the
   button's own accent colour, presses in on tap. `accent` is an "r,g,b" string.
   ═══════════════════════════════════════════════════════════════ */

function ClayButton({
  title,
  onPress,
  topColor,
  accent,
  textColor,
  loading,
}: {
  title: string;
  onPress: (e: any) => void;
  topColor: string;
  accent: string;
  textColor: string;
  loading: boolean;
}) {
  const [pressed, setPressed] = useState(false);
  const clayShadow = pressed
    ? `4px 4px 14px rgba(0,0,0,0.12), inset -5px -5px 14px rgba(${accent},0.85)`
    : `9px 9px 28px rgba(0,0,0,0.15), inset -8px -8px 24px rgba(${accent},0.7)`;
  const clayTransition =
    Platform.OS === "web"
      ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
      : undefined;

  return (
    <Pressable
      onPress={(e) => {
        if (!loading) onPress(e);
      }}
      onPressIn={() => !loading && setPressed(true)}
      onPressOut={() => setPressed(false)}
      style={{ width: "100%", marginTop: 6, marginBottom: 6 }}
    >
      <View
        style={[
          {
            backgroundColor: topColor,
            borderRadius: 999,
            paddingVertical: 16,
            alignItems: "center",
            boxShadow: clayShadow,
            transform: [{ translateY: pressed ? 3 : 0 }],
          } as any,
          clayTransition,
        ]}
      >
        {loading ? (
          <Spinner size="small" />
        ) : (
          <Text style={{ fontFamily: fonts.heading, fontSize: 20, color: textColor }}>
            {title}
          </Text>
        )}
      </View>
    </Pressable>
  );
}

