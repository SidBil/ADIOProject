import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Platform,
  KeyboardAvoidingView,
  ScrollView,
  Pressable,
} from "react-native";
import { colors, fonts } from "../theme";
import Spinner from "../components/Spinner";
import { supabase } from "../lib/supabase";


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
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.content}>
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
            <ClayButton title="Continue" onPress={handleSave} loading={loading} />
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
  content: {
    width: "100%",
    maxWidth: 480,
    alignItems: "center",
    paddingHorizontal: 8,
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
  clayBtn: {
    backgroundColor: colors.blueCard,
    borderRadius: 22,
    paddingVertical: 16,
    paddingHorizontal: 40,
    width: "100%",
    alignItems: "center",
  },
  clayBtnText: {
    fontFamily: fonts.fredoka,
    fontSize: 22,
    color: colors.darkBlue,
  },
});

/* ═══════════════════════════════════════════════════════════════
   Claymorphic button — matches the app's clay design language
   (outer drop shadow + a brighter inset sculpt, presses in on tap).
   ═══════════════════════════════════════════════════════════════ */

function ClayButton({
  title,
  onPress,
  loading,
}: {
  title: string;
  onPress: (e: any) => void;
  loading: boolean;
}) {
  const [pressed, setPressed] = useState(false);
  const clayShadow = pressed
    ? "5px 5px 18px rgba(10,70,130,0.22), inset -5px -5px 16px rgba(50,180,240,0.8)"
    : "10px 10px 34px rgba(10,70,130,0.27), inset -9px -9px 28px rgba(50,180,240,0.72)";
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
      style={[
        styles.clayBtn,
        { boxShadow: clayShadow, transform: [{ translateY: pressed ? 3 : 0 }] } as any,
        clayTransition,
      ]}
    >
      {loading ? (
        <Spinner size="small" />
      ) : (
        <Text style={styles.clayBtnText}>{title}</Text>
      )}
    </Pressable>
  );
}
