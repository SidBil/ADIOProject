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

  /* ── Claymorphism shadow recipes ──
     Layer stack (matches the CSS example the user shared):
       1. outer drop  — soft black, gives the puffy floating feel
       2. inset bottom-right — an "adjacent" warm color for the yellow / blue for the blue,
          which sculpts the 3D rounded-clay look
       3. inset top highlight — pure white, mimics a glossy top edge catching light  */
  // Inset colors are only ~1 shade deeper than each button's base card color,
  // so the sculpt reads as "same clay, slightly in shadow" instead of a contrasting outline.
  const beginClayShadow = beginPressed
    ? "10px 10px 26px rgba(0,0,0,0.14), inset -4px -4px 14px rgba(245,158,11,0.55)"
    : "26px 26px 50px rgba(0,0,0,0.2), inset -12px -12px 22px rgba(245,158,11,0.6)";

  const historyClayShadow = historyPressed
    ? "10px 10px 26px rgba(0,0,0,0.14), inset -4px -4px 14px rgba(41,165,225,0.5)"
    : "26px 26px 50px rgba(0,0,0,0.2), inset -12px -12px 22px rgba(41,165,225,0.55)";

  const clayTransition = Platform.OS === "web"
    ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
    : undefined;

  return (
    <View style={[styles.container, { width: winW, height: winH }]}>
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
                borderRadius: Math.max(22, Math.min(34, winH * 0.038)),
                paddingVertical: pad * 1.15,
                paddingHorizontal: pad * 3.2,
                boxShadow: beginClayShadow,
                transform: [{ translateY: beginPressed ? 3 : 0 }],
              },
              clayTransition,
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
                  borderRadius: Math.max(18, Math.min(28, winH * 0.032)),
                  paddingVertical: pad * 0.95,
                  paddingHorizontal: pad * 2.6,
                  boxShadow: historyClayShadow,
                  transform: [{ translateY: historyPressed ? 3 : 0 }],
                },
                clayTransition,
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
