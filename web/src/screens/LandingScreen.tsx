import React from "react";
import {
  View,
  Text,
  Image,
  StyleSheet,
  Pressable,
  useWindowDimensions,
} from "react-native";
import { colors, fonts } from "../theme";

/* eslint-disable @typescript-eslint/no-require-imports */
const adioLogo = require("../../assets/adiologo.png");
const heroImg = require("../../assets/hero_child.png");

interface Props {
  onStartSession: () => void;
  onSignUp: () => void;
  onLogIn: () => void;
}

export default function LandingScreen({
  onStartSession,
  onSignUp,
  onLogIn,
}: Props) {
  const { width: winW, height: winH } = useWindowDimensions();

  return (
    <View style={[s.root, { width: winW, height: winH }]}>
      {/* ═══════  TOP NAV  ═══════ */}
      <View style={s.nav}>
        <View style={s.navLeft}>
          <Pressable>
            <Text style={s.navLink}>Home</Text>
          </Pressable>
          <Pressable onPress={onStartSession}>
            <Text style={s.navLink}>Start a Session</Text>
          </Pressable>
        </View>
        <View style={s.navRight}>
          <Pressable onPress={onSignUp}>
            <Text style={s.navLink}>Sign Up</Text>
          </Pressable>
          <Pressable onPress={onLogIn}>
            <Text style={s.navLink}>Log In</Text>
          </Pressable>
        </View>
      </View>

      {/* ═══════  MAIN CONTENT — fills remaining height  ═══════ */}
      <View style={s.main}>
        {/* Left side */}
        <View style={s.left}>
          {/* Logo — cropped version, large */}
          <Image
            source={adioLogo}
            style={s.logo}
            resizeMode="contain"
          />

          {/* Tagline */}
          <Text style={s.tagline}>See it. Say it. Understand it.</Text>

          {/* Two content columns */}
          <View style={s.contentCols}>
            {/* What We Do */}
            <View style={s.contentCol}>
              <Text style={s.sectionTitle}>What We Do</Text>
              <View style={s.bulletList}>
                <Bullet text="We help children turn words into clear mental images" />
                <Bullet text="Students describe and refine what they see to strengthen understanding" />
                <Bullet text="Our method boosts comprehension, memory, and language skills" />
                <Bullet text="Lessons are calm, structured, and supportive" />
                <Bullet text="Light gamification motivates without distracting" />
                <Bullet text="We build confidence, independent thinking, and clear communication" />
              </View>
            </View>

            {/* How It Works */}
            <View style={s.contentCol}>
              <Text style={s.sectionTitle}>How It Works</Text>
              <View style={s.bulletList}>
                <Bullet text="Your child signs in and starts a session" />
                <Bullet text="They describe what they see using their voice" />
                <Bullet text="Our AI listens, evaluates, and asks follow-up questions" />
                <Bullet text="A summary shows progress in observation, understanding, and engagement" />
                <Bullet text="Every session is unique — no two are the same" />
              </View>
            </View>
          </View>

          {/* Footer legal links */}
          <View style={s.footer}>
            <Text style={s.footerText}>
              © 2026 Adio. All rights reserved.{"  "}•{"  "}
              <Text
                style={s.footerLink}
                onPress={() => window.open("/Adio_Terms_and_Conditions.docx.pdf", "_blank")}
              >
                Terms & Conditions
              </Text>
              {"  "}•{"  "}
              <Text
                style={s.footerLink}
                onPress={() => window.open("/Adio_Privacy_Policy.docx.pdf", "_blank")}
              >
                Privacy Policy
              </Text>
            </Text>
          </View>
        </View>

        {/* Right side — hero image, bottom-anchored, sharp bottom */}
        <View style={s.right}>
          <View style={s.heroImageWrap}>
            <Image
              source={heroImg}
              style={s.heroImage}
              resizeMode="cover"
            />
          </View>
        </View>
      </View>
    </View>
  );
}

/* ── Bullet component ── */

function Bullet({ text }: { text: string }) {
  return (
    <View style={s.bulletRow}>
      <Text style={s.bulletDot}>•</Text>
      <Text style={s.bulletText}>{text}</Text>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Styles
   ═══════════════════════════════════════════════════════════════ */

const s = StyleSheet.create({
  root: {
    backgroundColor: colors.bg,
    overflow: "hidden",
  },

  /* ── Nav ── */
  nav: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingTop: 48,
    paddingBottom: 24,
    paddingHorizontal: 64,
  },
  navLeft: {
    flexDirection: "row",
    gap: 56,
  },
  navRight: {
    flexDirection: "row",
    gap: 56,
  },
  navLink: {
    fontFamily: fonts.heading,
    fontSize: 30,
    color: colors.darkBlue,
  },

  /* ── Main content area — fills remaining height ── */
  main: {
    flex: 1,
    flexDirection: "row",
    paddingLeft: 64,
  },

  /* ── Left column ── */
  left: {
    flex: 1,
    justifyContent: "center",
    paddingRight: 48,
    paddingBottom: 32,
  },

  /* ── Logo ── */
  logo: {
    width: 400,
    height: 168,
    marginBottom: 16,
  },

  /* ── Tagline ── */
  tagline: {
    fontFamily: fonts.heading,
    fontSize: 64,
    color: colors.darkBlue,
    lineHeight: 76,
    marginBottom: 40,
  },

  /* ── Content columns ── */
  contentCols: {
    flexDirection: "row",
    gap: 64,
  },
  contentCol: {
    flex: 1,
  },
  sectionTitle: {
    fontFamily: fonts.heading,
    fontSize: 30,
    color: colors.darkBlue,
    marginBottom: 20,
  },

  /* ── Bullets ── */
  bulletList: {
    gap: 14,
  },
  bulletRow: {
    flexDirection: "row",
    alignItems: "flex-start",
  },
  bulletDot: {
    fontFamily: fonts.body,
    fontSize: 24,
    color: colors.darkBlue,
    marginRight: 14,
    lineHeight: 34,
  },
  bulletText: {
    fontFamily: fonts.body,
    fontSize: 22,
    color: colors.darkBlueText,
    lineHeight: 34,
    flex: 1,
  },

  /* ── Right column — hero image ── */
  right: {
    flex: 0.65,
    justifyContent: "flex-end",
  },

  heroImageWrap: {
    flex: 1,
    borderTopLeftRadius: 999,
    borderTopRightRadius: 999,
    borderBottomLeftRadius: 0,
    borderBottomRightRadius: 0,
    overflow: "hidden",
    backgroundColor: "#E0EAF5",
  },
  heroImage: {
    width: "100%",
    height: "100%",
  },
  footer: {
    marginTop: 48,
  },
  footerText: {
    fontFamily: fonts.body,
    fontSize: 18,
    color: colors.textMuted || "#666680",
    lineHeight: 26,
  },
  footerLink: {
    fontFamily: fonts.bodySemiBold || fonts.heading,
    color: colors.darkBlue,
    textDecorationLine: "underline",
  },
});
