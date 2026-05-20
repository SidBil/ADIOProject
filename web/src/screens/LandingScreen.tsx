import React from "react";
import {
  View,
  Text,
  Image,
  StyleSheet,
  Pressable,
  ScrollView,
  useWindowDimensions,
} from "react-native";
import { colors, fonts } from "../theme";

/* eslint-disable @typescript-eslint/no-require-imports */
const adioLogo = require("../../assets/adiologo.png");
const heroImg  = require("../../assets/hero_child.png");
const privacyPdf = require("../constants/legal/Adio_Privacy_Policy.docx.pdf");
const termsPdf   = require("../constants/legal/Adio_Terms_and_Conditions.docx.pdf");

const getAssetUri = (source: any) => {
  if (typeof source === "string") return source;
  return Image.resolveAssetSource(source)?.uri || source;
};

interface Props {
  onStartSession: () => void;
  onSignUp: () => void;
  onLogIn: () => void;
}

// Width breakpoints
const BP_SM = 700;
const BP_MD = 1000;

export default function LandingScreen({ onStartSession, onSignUp, onLogIn }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();

  const isLarge  = winW >  BP_MD;
  const isMedium = winW >  BP_SM && winW <= BP_MD;
  const isSmall  = winW <= BP_SM;

  // ── Scaled values ─────────────────────────────────────────────
  const navFontSize  = isLarge ? 30  : isMedium ? 22  : 17;
  const navGap       = isLarge ? 56  : isMedium ? 32  : 20;
  const navPadH      = isLarge ? 64  : isMedium ? 40  : 20;
  const navPadV      = isLarge ? 32  : isMedium ? 20  : 14;

  const logoW        = isLarge ? 360 : isMedium ? 240 : 180;
  const logoH        = isLarge ? 150 : isMedium ? 100 : 76;

  const taglineSize  = isLarge ? 56  : isMedium ? 40  : 30;
  const taglineLineH = isLarge ? 68  : isMedium ? 50  : 38;
  const taglineMB    = isLarge ? 32  : isMedium ? 24  : 16;

  const sectionSize  = isLarge ? 26  : isMedium ? 20  : 17;
  const bulletSize   = isLarge ? 19  : isMedium ? 15  : 13;
  const bulletLineH  = isLarge ? 30  : isMedium ? 24  : 20;
  const bulletGap    = isLarge ? 12  : isMedium ? 9   : 7;
  const bulletDotSz  = isLarge ? 20  : isMedium ? 16  : 13;

  const colsGap      = isLarge ? 56  : isMedium ? 32  : 20;
  const mainPadL     = isLarge ? 64  : isMedium ? 40  : 20;
  const leftPadR     = isLarge ? 40  : isMedium ? 20  : 16;
  const contentPadT  = isLarge ? 32  : isMedium ? 24  : 20;
  const contentPadB  = isLarge ? 40  : isMedium ? 32  : 28;

  const footerSize   = isLarge ? 15  : isMedium ? 13  : 11;
  const footerMT     = isLarge ? 36  : isMedium ? 24  : 16;

  const showHero     = !isSmall;
  const heroFlex     = isMedium ? 0.55 : 0.65;

  // Hero needs an explicit height so it doesn't collapse in the scrollable container.
  // Target ~85% of viewport height, clamped so it always looks good.
  const heroHeight   = Math.max(400, winH * 0.85);

  return (
    // Outer shell — fills the viewport, clips nothing upward
    <View style={[styles.shell, { width: winW, height: winH }]}>

      {/* ── NAV — always on top, never scrolls away ── */}
      <View style={[styles.nav, { paddingHorizontal: navPadH, paddingVertical: navPadV }]}>
        <View style={[styles.navRow, { gap: navGap }]}>
          <Pressable>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Home</Text>
          </Pressable>
          <Pressable onPress={onStartSession}>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Start a Session</Text>
          </Pressable>
        </View>
        <View style={[styles.navRow, { gap: navGap }]}>
          <Pressable onPress={onSignUp}>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Sign Up</Text>
          </Pressable>
          <Pressable onPress={onLogIn}>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Log In</Text>
          </Pressable>
        </View>
      </View>

      {/* ── BODY — always scrollable so tall content never overflows into nav ── */}
      <ScrollView
        style={styles.scrollArea}
        contentContainerStyle={{ flexGrow: 1 }}
        showsVerticalScrollIndicator={false}
      >
        {/* Inner row — flex: 1 so it fills the ScrollView height, enabling safe vertical centering */}
        <View
          style={[
            styles.scrollContent,
            {
              flexDirection: showHero ? "row" : "column",
              paddingLeft: mainPadL,
            },
          ]}
        >
        {/* Left / full-width column */}
        <View
          style={[
            styles.left,
            {
              paddingTop:    contentPadT,
              paddingBottom: contentPadB,
              paddingRight:  showHero ? leftPadR : mainPadL,
            },
          ]}
        >
          <Image
            source={adioLogo}
            style={{ width: logoW, height: logoH, marginBottom: 14 }}
            resizeMode="contain"
          />

          <Text style={[styles.tagline, { fontSize: taglineSize, lineHeight: taglineLineH, marginBottom: taglineMB }]}>
            See it. Say it. Understand it.
          </Text>

          {/* Two content columns — stack on small */}
          <View style={[styles.contentCols, { flexDirection: isSmall ? "column" : "row", gap: colsGap }]}>
            <View style={styles.contentCol}>
              <Text style={[styles.sectionTitle, { fontSize: sectionSize }]}>What We Do</Text>
              <View style={[styles.bulletList, { gap: bulletGap }]}>
                <Bullet text="We help children turn words into clear mental images"                    dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="Students describe and refine what they see to strengthen understanding"  dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="Our method boosts comprehension, memory, and language skills"            dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="Lessons are calm, structured, and supportive"                           dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="Light gamification motivates without distracting"                       dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="We build confidence, independent thinking, and clear communication"     dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
              </View>
            </View>

            <View style={styles.contentCol}>
              <Text style={[styles.sectionTitle, { fontSize: sectionSize }]}>How It Works</Text>
              <View style={[styles.bulletList, { gap: bulletGap }]}>
                <Bullet text="Your child signs in and starts a session"                                       dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="They describe what they see using their voice"                                   dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="Our AI listens, evaluates, and asks follow-up questions"                        dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="A summary shows progress in observation, understanding, and engagement"         dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
                <Bullet text="Every session is unique — no two are the same"                                  dot={bulletDotSz} size={bulletSize} lineH={bulletLineH} />
              </View>
            </View>
          </View>

          <View style={{ marginTop: footerMT }}>
            <Text style={[styles.footerText, { fontSize: footerSize }]}>
              © 2026 Adio. All rights reserved.{"  "}•{"  "}
              <Text style={styles.footerLink} onPress={() => Platform.OS === "web" && window.open(getAssetUri(termsPdf), "_blank")}>
                Terms & Conditions
              </Text>
              {"  "}•{"  "}
              <Text style={styles.footerLink} onPress={() => Platform.OS === "web" && window.open(getAssetUri(privacyPdf), "_blank")}>
                Privacy Policy
              </Text>
            </Text>
          </View>
        </View>

        {/* Right — hero image, hidden on small screens */}
        {showHero && (
          <View style={[styles.heroCol, { flex: heroFlex }]}>
            <View style={[styles.heroImageWrap, { height: heroHeight }]}>
              <Image source={heroImg} style={styles.heroImage} resizeMode="cover" />
            </View>
          </View>
        )}
        </View>{/* end inner row */}
      </ScrollView>
    </View>
  );
}

/* ── Bullet ──────────────────────────────────────────────────── */

function Bullet({ text, dot, size, lineH }: { text: string; dot: number; size: number; lineH: number }) {
  return (
    <View style={styles.bulletRow}>
      <Text style={[styles.bulletDot, { fontSize: dot, lineHeight: lineH }]}>•</Text>
      <Text style={[styles.bulletText, { fontSize: size, lineHeight: lineH }]}>{text}</Text>
    </View>
  );
}

/* ── Styles ──────────────────────────────────────────────────── */

const styles = StyleSheet.create({
  shell: {
    backgroundColor: colors.bg,
    // No overflow:hidden — we want the ScrollView to own clipping
  },

  // Nav sits outside the ScrollView so it never scrolls away
  nav: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: colors.bg,
    zIndex: 10,
  },
  navRow: { flexDirection: "row" },
  navLink: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
  },

  scrollArea: { flex: 1 },
  // flex: 1 lets this row fill the ScrollView height → safe to center within it
  scrollContent: {
    flex: 1,
    alignItems: "stretch",
  },

  // Now safely centered — bounded by the ScrollView which sits below the nav
  left: {
    flex: 1,
    justifyContent: "center",
  },

  tagline: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
  },

  contentCols: {},
  contentCol: { flex: 1 },

  sectionTitle: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
    marginBottom: 16,
  },

  bulletList: {},
  bulletRow: {
    flexDirection: "row",
    alignItems: "flex-start",
  },
  bulletDot: {
    fontFamily: fonts.body,
    color: colors.darkBlue,
    marginRight: 12,
  },
  bulletText: {
    fontFamily: fonts.body,
    color: colors.darkBlueText,
    flex: 1,
  },

  heroCol: {
    justifyContent: "flex-end",
  },
  heroImageWrap: {
    borderTopLeftRadius: 999,
    borderTopRightRadius: 999,
    overflow: "hidden",
    backgroundColor: "#E0EAF5",
  },
  heroImage: {
    width: "100%",
    height: "100%",
  },

  footerText: {
    fontFamily: fonts.body,
    color: colors.textMuted,
    lineHeight: 22,
  },
  footerLink: {
    fontFamily: fonts.bodySemiBold,
    color: colors.darkBlue,
    textDecorationLine: "underline",
  },
});
