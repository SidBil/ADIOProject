import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ActivityIndicator,
  Platform,
  KeyboardAvoidingView,
  ScrollView,
  Pressable,
} from "react-native";
import { colors, fonts } from "../theme";
import { supabase } from "../lib/supabase";
import ShapePattern from "../components/ShapePattern";


interface Props {
  onComplete: () => void;
  userId: string;
}

export default function OnboardingScreen({ onComplete, userId }: Props) {
  const [guardianFirstName, setGuardianFirstName] = useState("");
  const [guardianLastName, setGuardianLastName] = useState("");
  const [childNickname, setChildNickname] = useState("");
  const [gradeLevel, setGradeLevel] = useState("");
  const [optIn, setOptIn] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSave() {
    if (!guardianFirstName.trim() || !guardianLastName.trim() || !childNickname.trim() || !gradeLevel.trim()) {
      setError("Please fill out all fields.");
      return;
    }

    setLoading(true);
    setError(null);

    const { error: insertError } = await supabase
      .from("user_profiles")
      .insert({
        id: userId,
        guardian_first_name: guardianFirstName.trim(),
        guardian_last_name: guardianLastName.trim(),
        child_nickname: childNickname.trim(),
        grade_level: gradeLevel.trim(),
        speech_data_opt_in: optIn,
      });

    setLoading(false);

    if (insertError) {
      // If it's a unique violation, maybe they already have a profile? 
      // We can just proceed if they already exist, but normally this screen wouldn't show.
      if (insertError.code === "23505") {
        onComplete();
      } else {
        setError(insertError.message);
      }
    } else {
      onComplete();
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ShapePattern burst={0} />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.card}>
          <Text style={styles.title}>Welcome to Adio</Text>
          <Text style={styles.subtitle}>Before we begin, please tell us a little bit about yourself and your child.</Text>

          <TextInput
            style={styles.input}
            placeholder="Guardian First Name"
            placeholderTextColor={colors.textMuted}
            value={guardianFirstName}
            onChangeText={setGuardianFirstName}
          />
          <TextInput
            style={styles.input}
            placeholder="Guardian Last Name"
            placeholderTextColor={colors.textMuted}
            value={guardianLastName}
            onChangeText={setGuardianLastName}
          />
          <TextInput
            style={styles.input}
            placeholder="Child First Name / Nickname"
            placeholderTextColor={colors.textMuted}
            value={childNickname}
            onChangeText={setChildNickname}
          />
          <TextInput
            style={styles.input}
            placeholder="Child Grade Level (e.g., Pre-K, 1st Grade)"
            placeholderTextColor={colors.textMuted}
            value={gradeLevel}
            onChangeText={setGradeLevel}
          />

          <Pressable 
            style={styles.checkboxRow} 
            onPress={() => setOptIn(!optIn)}
          >
            <View style={[styles.checkbox, optIn && styles.checkboxActive]}>
              {optIn && <Text style={styles.checkmark}>✓</Text>}
            </View>
            <Text style={styles.checkboxLabel}>
              I consent to the collection of anonymized speech data for product improvement and research purposes.
            </Text>
          </Pressable>

          {error && <Text style={styles.errorText}>{error}</Text>}

          <View style={{ marginTop: 24, width: "100%" }}>
              <Button3DInline
                title="Continue"
                onPress={handleSave}
                loading={loading}
                topColor={colors.blueCard}
                bottomColor={colors.blueBorder}
                textColor={colors.darkBlue}
              />
          </View>
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
    maxWidth: 480,
    alignItems: "center",
    borderWidth: 5,
    borderColor: "#d8d8d8",
    shadowColor: "#d8d8d8",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 1,
    shadowRadius: 0,
    elevation: 4,
  },
  title: {
    fontFamily: fonts.heading,
    fontSize: 28,
    color: colors.darkBlue,
    textAlign: "center",
    marginBottom: 8,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.textMuted,
    textAlign: "center",
    marginBottom: 24,
    lineHeight: 22,
  },
  input: {
    width: "100%",
    borderWidth: 3,
    borderColor: "#e0e0e8",
    borderRadius: 16, // slightly less rounded than login to accommodate more inputs
    paddingVertical: 14,
    paddingHorizontal: 22,
    fontFamily: fonts.body,
    fontSize: 17,
    color: colors.darkBlueText,
    marginBottom: 14,
    backgroundColor: "#ffffff",
  },
  checkboxRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginTop: 8,
    marginBottom: 16,
    width: "100%",
    paddingHorizontal: 4,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderWidth: 3,
    borderColor: "#e0e0e8",
    borderRadius: 6,
    marginRight: 12,
    marginTop: 2,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
  },
  checkboxActive: {
    borderColor: colors.blueBorder,
    backgroundColor: colors.blueCard,
  },
  checkmark: {
    color: colors.darkBlue,
    fontSize: 16,
    fontWeight: "bold",
    marginTop: -2, // optical alignment
  },
  checkboxLabel: {
    flex: 1,
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.textMuted,
    lineHeight: 20,
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.pinkBorder,
    textAlign: "center",
    marginTop: 10,
  },
});

/* ═══════════════════════════════════════════════════════════════
   Button3D component to mimic the 3D card layout 
   ═══════════════════════════════════════════════════════════════ */

import { Animated, Easing } from "react-native";
import { useEffect, useRef } from "react";

function Button3DInline({ 
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
