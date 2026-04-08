import React, { useState } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { colors, fonts } from "../theme";

interface Props {
  onStart: () => Promise<void>;
  onHistory?: () => void;
  onSignOut?: () => void;
}

export default function WelcomeScreen({ onStart, onHistory, onSignOut }: Props) {
  const [loading, setLoading] = useState(false);

  const handlePress = async () => {
    setLoading(true);
    try {
      await onStart();
    } catch (e: any) {
      alert("Failed to start session. " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {onSignOut && (
        <TouchableOpacity style={styles.signOutBtn} onPress={onSignOut}>
          <Text style={styles.signOutText}>Sign Out</Text>
        </TouchableOpacity>
      )}
      <View style={styles.card}>
        <Image
          source={require("../../assets/adio_logo2.png")}
          style={styles.logo}
          resizeMode="contain"
        />
        <Text style={styles.title}>Reading Comprehension Therapy</Text>
        <Text style={styles.description}>
          You will be shown a picture and asked to describe what you see. Speak
          your answers out loud — a friendly guide will listen and help you
          notice more details.
        </Text>
        <TouchableOpacity
          style={styles.btn}
          onPress={handlePress}
          disabled={loading}
          activeOpacity={0.8}
        >
          {loading ? (
            <ActivityIndicator color={colors.white} />
          ) : (
            <Text style={styles.btnText}>Begin a Session</Text>
          )}
        </TouchableOpacity>

        {onHistory && (
          <TouchableOpacity
            style={styles.historyBtn}
            onPress={onHistory}
            activeOpacity={0.8}
          >
            <Text style={styles.historyBtnText}>View Past Sessions</Text>
          </TouchableOpacity>
        )}

        <Text style={styles.hint}>An image will be chosen for you.</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  signOutBtn: {
    position: "absolute",
    top: 40,
    right: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  signOutText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.textMuted,
  },
  card: {
    backgroundColor: colors.cardWhite,
    borderRadius: 28,
    padding: 40,
    width: "100%",
    maxWidth: 480,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 10,
    elevation: 3,
  },
  logo: { height: 80, width: 200, marginBottom: 12 },
  title: {
    fontFamily: fonts.heading,
    fontSize: 24,
    color: colors.darkBlue,
    textAlign: "center",
    marginBottom: 20,
  },
  description: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.textMuted,
    textAlign: "center",
    lineHeight: 26,
    marginBottom: 28,
  },
  btn: {
    backgroundColor: colors.darkBlueBtnBg,
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 48,
    marginBottom: 12,
  },
  btnText: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.white,
  },
  historyBtn: {
    borderWidth: 2,
    borderColor: colors.blueBorder,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 36,
    marginBottom: 12,
  },
  historyBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 16,
    color: colors.blueBorder,
  },
  hint: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.textMuted,
  },
});
