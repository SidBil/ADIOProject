import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ActivityIndicator,
  Image,
  Platform,
  KeyboardAvoidingView,
  ScrollView,
  Animated,
  Easing,
  Pressable,
} from "react-native";
import { colors, fonts } from "../theme";
import { supabase } from "../lib/supabase";
import { track } from "../lib/analytics";
import ShapePattern from "../components/ShapePattern";

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
    const { error: oauthError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: Platform.OS === "web" ? window.location.origin : undefined,
      },
    });
    setLoading(false);
    if (oauthError) {
      setError(oauthError.message);
      track("app_error", { area: "auth", error_code: oauthError.code });
    } else {
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
      <ShapePattern burst={burstCount} cardCenter={clickCenter} />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.card}>
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

          <Button3D
            title={isSignUp ? "Sign Up" : "Sign In"}
            onPress={handleEmailAuth}
            loading={loading}
            topColor={colors.blueCard}
            bottomColor={colors.blueBorder}
            textColor={colors.darkBlue}
          />

          {isSignUp && (
            <Text style={styles.legalText}>
              By signing up, you agree to our{" "}
              <Text
                style={styles.legalLink}
                onPress={() => window.open(getAssetUri(termsPdf), "_blank")}
              >
                Terms & Conditions
              </Text>
              {" and "}
              <Text
                style={styles.legalLink}
                onPress={() => window.open(getAssetUri(privacyPdf), "_blank")}
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

          <Button3D
            title="Sign in with Google"
            onPress={handleGoogleAuth}
            loading={loading}
            topColor={colors.greenBtn}
            bottomColor={colors.greenBorder}
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
  card: {
    backgroundColor: colors.cardWhite,
    borderRadius: 28,
    padding: 36,
    width: "100%",
    maxWidth: 420,
    alignItems: "center",
    borderWidth: 5,
    borderColor: "#d8d8d8",
    shadowColor: "#d8d8d8",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 1,
    shadowRadius: 0,
    elevation: 4,
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
   Button3D component to mimic the 3D card layout from SessionScreen
   ═══════════════════════════════════════════════════════════════ */

function Button3D({ 
  title, 
  onPress, 
  topColor, 
  bottomColor, 
  textColor, 
  loading,
}: {
  title: string;
  onPress: (e: any) => void;
  topColor: string;
  bottomColor: string;
  textColor: string;
  loading: boolean;
}) {
  const [pressed, setPressed] = useState(false);
  const pressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(pressAnim, {
      toValue: pressed ? 1 : 0,
      duration: 150,
      easing: Easing.inOut(Easing.sin),
      useNativeDriver: false,
    }).start();
  }, [pressed]);

  const webTransitionStyle = Platform.OS === "web" ? {
    transition: "transform 150ms cubic-bezier(0.445, 0.05, 0.55, 0.95), box-shadow 150ms cubic-bezier(0.445, 0.05, 0.55, 0.95)",
    boxShadow: pressed
      ? `0px 0px 0px ${bottomColor}`
      : `0px 8px 0px ${bottomColor}`,
    transform: pressed ? "translateY(8px)" : "translateY(0px)",
  } as any : undefined;

  const nativeAnimStyle = Platform.OS !== "web" ? {
    transform: [{ translateY: pressAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 8] }) }],
    shadowOffset: { width: 0, height: pressAnim.interpolate({ inputRange: [0, 1], outputRange: [8, 0] }) as unknown as number },
    shadowOpacity: pressAnim.interpolate({ inputRange: [0, 1], outputRange: [1, 0] }),
    elevation: pressAnim.interpolate({ inputRange: [0, 1], outputRange: [6, 0] }),
  } : undefined;

  return (
    <Pressable
      onPress={(e) => {
        if (!loading) onPress(e);
      }}
      onPressIn={() => !loading && setPressed(true)}
      onPressOut={() => setPressed(false)}
      style={{ width: "100%", marginTop: 4, marginBottom: 4 }}
    >
      <Animated.View
        style={[
          {
            backgroundColor: topColor,
            borderWidth: 4,
            borderColor: bottomColor,
            borderRadius: 999,
            paddingVertical: 16,
            alignItems: "center",
            shadowColor: bottomColor,
            shadowOffset: { width: 0, height: 8 },
            shadowOpacity: 1,
            shadowRadius: 0,
          },
          Platform.OS === "web" 
            ? { shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0, elevation: 0 } 
            : undefined,
          nativeAnimStyle,
          webTransitionStyle,
        ]}
      >
        {loading ? (
          <ActivityIndicator color={textColor} />
        ) : (
          <Text style={{ fontFamily: fonts.heading, fontSize: 20, color: textColor }}>
            {title}
          </Text>
        )}
      </Animated.View>
    </Pressable>
  );
}

