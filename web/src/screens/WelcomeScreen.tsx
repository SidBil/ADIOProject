import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Linking,
  Platform,
  Pressable,
  useWindowDimensions,
} from "react-native";
import { colors, fonts } from "../theme";
import ShapePattern from "../components/ShapePattern";

interface Props {
  onStart: () => Promise<void>;
  onHistory?: () => void;
  onSignOut?: () => void;
}

export default function WelcomeScreen({ onStart, onHistory, onSignOut }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();
  const [loading, setLoading] = useState(false);
  const [beginPressed, setBeginPressed] = useState(false);
  const [historyPressed, setHistoryPressed] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const msgIndexRef = useRef(0);

  const LOADING_MESSAGES = [
    "Getting everything ready for you!",
    "Warming up Adio…",
    "Almost there!",
    "Just a few more seconds…",
  ];

  useEffect(() => {
    if (!loading) return;
    setLoadingMsg(LOADING_MESSAGES[0]);
    msgIndexRef.current = 0;
    const interval = setInterval(() => {
      msgIndexRef.current = Math.min(msgIndexRef.current + 1, LOADING_MESSAGES.length - 1);
      setLoadingMsg(LOADING_MESSAGES[msgIndexRef.current]);
    }, 4000);
    return () => clearInterval(interval);
  }, [loading]);

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

  // Responsive scaling — same system as SummaryScreen
  const pad           = Math.max(14, winW * 0.018);
  const cardBorder    = Math.max(4, Math.min(6, winH * 0.007));
  const cardRadius    = Math.max(20, winH * 0.03);

  const logoH         = Math.max(120, Math.min(210, winH * 0.24));
  const logoW         = logoH * 2.4;


  const descSz        = Math.max(16, Math.min(22, winH * 0.026));
  const btnFontSz     = Math.max(20, Math.min(30, winH * 0.036));
  const hintSz        = Math.max(13, Math.min(16, winH * 0.02));

  const beginWebStyle = Platform.OS === "web" ? ({
    transition: "transform 150ms ease, box-shadow 150ms ease",
    boxShadow: beginPressed ? `0px 0px 0px ${colors.yellowBorder}` : `0px ${cardBorder}px 0px ${colors.yellowBorder}`,
    transform: beginPressed ? `translateY(${cardBorder}px)` : "translateY(0px)",
  } as any) : undefined;

  const historyWebStyle = Platform.OS === "web" ? ({
    transition: "transform 150ms ease, box-shadow 150ms ease",
    boxShadow: historyPressed ? `0px 0px 0px ${colors.blueBorder}` : `0px ${cardBorder}px 0px ${colors.blueBorder}`,
    transform: historyPressed ? `translateY(${cardBorder}px)` : "translateY(0px)",
  } as any) : undefined;

  return (
    <View style={[styles.container, { width: winW, height: winH }]}>
      <ShapePattern />

      {onSignOut && (
        <TouchableOpacity style={styles.signOutBtn} onPress={onSignOut}>
          <Text style={[styles.signOutText, { fontSize: hintSz }]}>Sign Out</Text>
        </TouchableOpacity>
      )}

      <View style={[styles.contentWrap, { padding: pad }]}>
        <Image
          source={require("../../assets/adiologowithtext-03.png")}
          style={{ width: logoW, height: logoH, marginBottom: pad * 1.5 }}
          resizeMode="contain"
        />

        <Text style={[styles.description, {
          fontSize: descSz,
          lineHeight: descSz * 1.5,
          marginBottom: pad * 2,
          maxWidth: 640,
        }]}>
          You will be shown a picture and asked to describe what you see. Speak
          your answers out loud. A friendly guide will listen and help you
          notice more details.
        </Text>

        {/* ── Primary: Begin a Session (yellow chunky button) ── */}
        <Pressable
          onPress={handlePress}
          disabled={loading}
          onPressIn={() => setBeginPressed(true)}
          onPressOut={() => setBeginPressed(false)}
          style={{ marginBottom: pad }}
        >
          <View
            style={[
              styles.beginBtn,
              {
                borderWidth: cardBorder,
                borderRadius: 999,
                paddingVertical: pad,
                paddingHorizontal: pad * 3,
              },
              Platform.OS === "web"
                ? { shadowOpacity: 0, elevation: 0 }
                : {
                    shadowColor: colors.yellowBorder,
                    shadowOffset: { width: 0, height: cardBorder },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                    elevation: 4,
                  },
              beginWebStyle,
            ]}
          >
            {loading ? (
              <Text style={[styles.beginBtnText, { fontSize: btnFontSz * 0.7 }]}>{loadingMsg}</Text>
            ) : (
              <Text style={[styles.beginBtnText, { fontSize: btnFontSz }]}>Begin a Session</Text>
            )}
          </View>
        </Pressable>

        {/* ── Secondary: View Past Sessions (blue chunky button) ── */}
        {onHistory && (
          <Pressable
            onPress={onHistory}
            onPressIn={() => setHistoryPressed(true)}
            onPressOut={() => setHistoryPressed(false)}
            style={{ marginBottom: pad }}
          >
            <View
              style={[
                styles.historyBtn,
                {
                  borderWidth: cardBorder,
                  borderRadius: 999,
                  paddingVertical: pad * 0.8,
                  paddingHorizontal: pad * 2.4,
                },
                Platform.OS === "web"
                  ? { shadowOpacity: 0, elevation: 0 }
                  : {
                      shadowColor: colors.blueBorder,
                      shadowOffset: { width: 0, height: cardBorder },
                      shadowOpacity: 1,
                      shadowRadius: 0,
                      elevation: 4,
                    },
                historyWebStyle,
              ]}
            >
              <Text style={[styles.historyBtnText, { fontSize: btnFontSz * 0.75 }]}>
                View Past Sessions
              </Text>
            </View>
          </Pressable>
        )}

      </View>

      <TouchableOpacity
        style={styles.feedbackBtn}
        onPress={() =>
          Linking.openURL(
            "mailto:sidharthbildikar@gmail.com?subject=Adio%20Feedback%20%2F%20Bug%20Report"
          )
        }
        activeOpacity={0.7}
      >
        <Text style={[styles.feedbackText, { fontSize: hintSz }]}>
          Send Feedback / Report a Bug
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
  },
  signOutBtn: {
    position: "absolute",
    top: 40,
    right: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
    zIndex: 10,
    elevation: 10,
  },
  signOutText: {
    fontFamily: fonts.bodySemiBold,
    color: colors.textMuted,
  },
  contentWrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
  },
  description: {
    fontFamily: fonts.body,
    color: colors.darkBlueText,
    textAlign: "center",
  },

  /* ── Primary yellow chunky button ── */
  beginBtn: {
    backgroundColor: colors.yellowCard,
    borderColor: colors.yellowBorder,
    alignItems: "center",
    justifyContent: "center",
  },
  beginBtnText: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
  },

  /* ── Secondary blue chunky button ── */
  historyBtn: {
    backgroundColor: colors.blueCard,
    borderColor: colors.blueBorder,
    alignItems: "center",
    justifyContent: "center",
  },
  historyBtnText: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
  },

  hint: {
    fontFamily: fonts.body,
    color: colors.textMuted,
  },
  feedbackBtn: {
    position: "absolute",
    bottom: 20,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  feedbackText: {
    fontFamily: fonts.body,
    color: colors.textMuted,
    textDecorationLine: "underline",
  },
});
