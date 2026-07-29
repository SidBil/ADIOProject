import React, { useEffect, useState } from "react";
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
  userId: string;
  onBack: () => void;
}

/**
 * Edit the guardian/child details captured at onboarding, after the fact
 * (Adult tab). Reads and writes the same `user_profiles` row onboarding creates,
 * directly via Supabase — RLS scopes every row to the caller's auth id.
 */
export default function ProfileEditScreen({ userId, onBack }: Props) {
  const [loading, setLoading] = useState(true); // initial fetch
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const [guardianFirstName, setGuardianFirstName] = useState("");
  const [guardianLastName, setGuardianLastName] = useState("");
  const [childNickname, setChildNickname] = useState("");
  const [gradeLevel, setGradeLevel] = useState("");
  const [optIn, setOptIn] = useState(false);

  useEffect(() => {
    let alive = true;
    supabase
      .from("user_profiles")
      .select("guardian_first_name, guardian_last_name, child_nickname, grade_level, speech_data_opt_in")
      .eq("id", userId)
      .maybeSingle()
      .then(({ data, error: fetchError }) => {
        if (!alive) return;
        if (fetchError) {
          setError(fetchError.message);
        } else if (data) {
          setGuardianFirstName(data.guardian_first_name ?? "");
          setGuardianLastName(data.guardian_last_name ?? "");
          setChildNickname(data.child_nickname ?? "");
          setGradeLevel(data.grade_level ?? "");
          setOptIn(!!data.speech_data_opt_in);
        }
        setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [userId]);

  async function handleSave() {
    if (
      !guardianFirstName.trim() ||
      !guardianLastName.trim() ||
      !childNickname.trim() ||
      !gradeLevel.trim()
    ) {
      setError("Please fill out all fields.");
      return;
    }
    setSaving(true);
    setError(null);
    setSaved(false);

    const { error: updateError } = await supabase
      .from("user_profiles")
      .update({
        guardian_first_name: guardianFirstName.trim(),
        guardian_last_name: guardianLastName.trim(),
        child_nickname: childNickname.trim(),
        grade_level: gradeLevel.trim(),
        speech_data_opt_in: optIn,
      })
      .eq("id", userId);

    setSaving(false);
    if (updateError) setError(updateError.message);
    else setSaved(true);
  }

  // Any edit clears the "Saved" confirmation so it never lingers as stale.
  function edited<T>(setter: (v: T) => void) {
    return (v: T) => {
      if (saved) setSaved(false);
      setter(v);
    };
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <Pressable onPress={onBack} style={styles.backRow} hitSlop={10}>
        <Text style={styles.backText}>‹ For Grown-Ups</Text>
      </Pressable>

      {loading ? (
        <View style={styles.loading}>
          <Spinner size="large" />
        </View>
      ) : (
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.content}>
            <Text style={styles.title}>Profile & Child Info</Text>
            <Text style={styles.subtitle}>
              Update the details you gave when you first set up Adio.
            </Text>

            <Field label="Guardian First Name">
              <TextInput
                style={styles.input}
                placeholderTextColor={colors.textMuted}
                value={guardianFirstName}
                onChangeText={edited(setGuardianFirstName)}
              />
            </Field>
            <Field label="Guardian Last Name">
              <TextInput
                style={styles.input}
                placeholderTextColor={colors.textMuted}
                value={guardianLastName}
                onChangeText={edited(setGuardianLastName)}
              />
            </Field>
            <Field label="Child First Name / Nickname">
              <TextInput
                style={styles.input}
                placeholderTextColor={colors.textMuted}
                value={childNickname}
                onChangeText={edited(setChildNickname)}
              />
            </Field>
            <Field label="Child Grade Level">
              <TextInput
                style={styles.input}
                placeholder="e.g., Pre-K, 1st Grade"
                placeholderTextColor={colors.textMuted}
                value={gradeLevel}
                onChangeText={edited(setGradeLevel)}
              />
            </Field>

            <Pressable
              style={styles.checkboxRow}
              onPress={() => edited(setOptIn)(!optIn)}
            >
              <View style={[styles.checkbox, optIn && styles.checkboxActive]}>
                {optIn && <Text style={styles.checkmark}>✓</Text>}
              </View>
              <Text style={styles.checkboxLabel}>
                I consent to the collection of anonymized speech data for product
                improvement and research purposes.
              </Text>
            </Pressable>

            {error && <Text style={styles.errorText}>{error}</Text>}
            {saved && <Text style={styles.savedText}>Saved ✓</Text>}

            <View style={{ marginTop: 20, width: "100%" }}>
              <ClayButton title="Save Changes" onPress={handleSave} loading={saving} />
            </View>
          </View>
        </ScrollView>
      )}
    </KeyboardAvoidingView>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      {children}
    </View>
  );
}

/* Claymorphic button — matches the app's clay design language. */
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

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  backRow: {
    paddingHorizontal: 20,
    paddingTop: Platform.OS === "ios" ? 60 : 36,
    paddingBottom: 4,
  },
  backText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 17,
    color: colors.darkBlue,
  },
  loading: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: "center",
    padding: 20,
  },
  content: {
    width: "100%",
    maxWidth: 480,
    paddingHorizontal: 8,
  },
  title: {
    fontFamily: fonts.heading,
    fontSize: 28,
    color: colors.darkBlue,
    marginBottom: 6,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.textMuted,
    marginBottom: 22,
    lineHeight: 22,
  },
  field: {
    width: "100%",
    marginBottom: 14,
  },
  fieldLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.darkBlue,
    marginBottom: 6,
    marginLeft: 4,
  },
  input: {
    width: "100%",
    borderWidth: 3,
    borderColor: "#e0e0e8",
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 22,
    fontFamily: fonts.body,
    fontSize: 17,
    color: colors.darkBlueText,
    backgroundColor: "#ffffff",
  },
  checkboxRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginTop: 8,
    marginBottom: 6,
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
    marginTop: -2,
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
  savedText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 15,
    color: colors.blueBorder,
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
