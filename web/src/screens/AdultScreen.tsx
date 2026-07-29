import React from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ScrollView,
  Linking,
  Platform,
} from "react-native";
import { colors, fonts } from "../theme";
import ProfileEditScreen from "./ProfileEditScreen";

interface Props {
  userId: string;
  onAbout: () => void;
  onSignOut: () => void;
}

/**
 * Adult / Settings tab (03_navigation_shell.md §3). Holds the parent/SLP-facing
 * information and controls that used to live scattered on the old WelcomeScreen —
 * About, feedback/bug report, and sign-out. Child-first-safe: a child landing
 * here finds nothing alarming or breakable. Detailed progress lives on the
 * Progress tab; this surface is settings + info.
 */
export default function AdultScreen({ userId, onAbout, onSignOut }: Props) {
  const [editingProfile, setEditingProfile] = React.useState(false);

  // Drill-in: edit the onboarding details in place, with a back control.
  if (editingProfile) {
    return <ProfileEditScreen userId={userId} onBack={() => setEditingProfile(false)} />;
  }

  return (
    <ScrollView
      style={styles.root}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      <Text style={styles.title}>For Grown-Ups</Text>
      <Text style={styles.subtitle}>
        Settings and information for parents, carers, and therapists.
      </Text>

      <Row label="Profile & Child Info" onPress={() => setEditingProfile(true)} />
      <Row label="About Adio" onPress={onAbout} />
      <Row
        label="Send Feedback / Report a Bug"
        onPress={() =>
          Linking.openURL(
            "mailto:sidharthbildikar@gmail.com?subject=Adio%20Feedback%20%2F%20Bug%20Report"
          )
        }
      />
      <Row label="Sign Out" onPress={onSignOut} destructive chevron={false} />
    </ScrollView>
  );
}

function Row({
  label,
  onPress,
  destructive,
  chevron = true,
}: {
  label: string;
  onPress: () => void;
  destructive?: boolean;
  // Whether to show the trailing "›" — true for rows that navigate/open
  // something, false for terminal actions like Sign Out.
  chevron?: boolean;
}) {
  const [pressed, setPressed] = React.useState(false);
  const clayTransition =
    Platform.OS === "web"
      ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
      : undefined;
  return (
    <Pressable
      onPress={onPress}
      onPressIn={() => setPressed(true)}
      onPressOut={() => setPressed(false)}
    >
      <View
        style={[
          styles.row,
          {
            boxShadow: pressed
              ? "6px 6px 14px rgba(0,0,0,0.08), inset -3px -3px 8px rgba(41,165,225,0.25)"
              : "14px 14px 30px rgba(0,0,0,0.08), inset -6px -6px 14px rgba(41,165,225,0.2)",
            transform: [{ translateY: pressed ? 2 : 0 }],
          } as any,
          clayTransition,
        ]}
      >
        <Text style={[styles.rowLabel, destructive && { color: colors.pinkBorder }]}>
          {label}
        </Text>
        {chevron && <Text style={styles.chevron}>›</Text>}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  content: {
    padding: 24,
    paddingTop: Platform.OS === "ios" ? 64 : 40,
    gap: 14,
  },
  title: {
    fontFamily: fonts.heading,
    fontSize: 34,
    color: colors.darkBlue,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    lineHeight: 23,
    color: colors.textMuted,
    marginBottom: 12,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: colors.white,
    borderRadius: 20,
    paddingVertical: 20,
    paddingHorizontal: 22,
  },
  rowLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 17,
    color: colors.darkBlueText,
  },
  chevron: {
    fontFamily: fonts.heading,
    fontSize: 24,
    color: colors.textMuted,
  },
});
