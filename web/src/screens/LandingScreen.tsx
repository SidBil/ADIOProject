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
const heroImg = require("../../assets/hero_child.png");
const privacyPdf = require("../constants/legal/Adio_Privacy_Policy.docx.pdf");
const termsPdf = require("../constants/legal/Adio_Terms_and_Conditions.docx.pdf");

const getAssetUri = (source: any) => {
  if (typeof source === "string") return source;
  return Image.resolveAssetSource(source)?.uri || source;
};

interface Props {
  onStartSession: () => void;
  onSignUp: () => void;
  onLogIn: () => void;
}

// Breakpoints
const BP_SM = 700;   // phone / very small tablet
const BP_MD = 1000;  // mid-size laptop
// > BP_MD → large desktop (original design)

export default function LandingScreen({
  onStartSession,
  onSignUp,
  onLogIn,
}: Props) {
  const { width: winW, height: winH } = useWindowDimensions();

  const isLarge  = winW > BP_MD;
  const isMedium = winW > BP_SM && winW <= BP_MD;
  const isSmall  = winW <= BP_SM;

  // Scale helpers
  const navFontSize     = isLarge ? 30   : isMedium ? 22  : 17;
  const navGap          = isLarge ? 56   : isMedium ? 32  : 20;
  const navPadH         = isLarge ? 64   : isMedium ? 40  : 20;
  const navPadTop       = isLarge ? 48   : isMedium ? 32  : 20;
  const navPadBottom    = isLarge ? 24   : isMedium ? 16  : 12;

  const logoW           = isLarge ? 400  : isMedium ? 280 : 200;
  const logoH           = isLarge ? 168  : isMedium ? 118 : 84;

  const taglineFontSize = isLarge ? 64   : isMedium ? 44  : 32;
  const taglineLineH    = isLarge ? 76   : isMedium ? 54  : 40;
  const taglineMB       = isLarge ? 40   : isMedium ? 28  : 20;

  const sectionFontSize = isLarge ? 30   : isMedium ? 22  : 18;
  const bulletFontSize  = isLarge ? 22   : isMedium ? 16  : 14;
  const bulletLineH     = isLarge ? 34   : isMedium ? 26  : 22;
  const bulletGap       = isLarge ? 14   : isMedium ? 10  : 8;
  const bulletDotSize   = isLarge ? 24   : isMedium ? 18  : 15;

  const colsGap         = isLarge ? 64   : isMedium ? 36  : 24;
  const mainPadLeft     = isLarge ? 64   : isMedium ? 40  : 20;
  const leftPadRight    = isLarge ? 48   : isMedium ? 24  : 16;

  const footerFontSize  = isLarge ? 18   : isMedium ? 14  : 12;
  const footerMT        = isLarge ? 48   : isMedium ? 28  : 20;

  // On small screens: stack vertically, hide hero; on medium+: side-by-side
  const mainIsRow = !isSmall;

  // On small screens wrap in ScrollView so content isn't clipped
  const ContentWrapper = isSmall ? ScrollView : View;
  const contentWrapperProps = isSmall
    ? { style: { flex: 1 }, contentContainerStyle: { flexGrow: 1 } }
    : { style: { flex: 1, flexDirection: "row" as const, paddingLeft: mainPadLeft } };

  return (
    <View style={[styles.root, { width: winW, height: winH }]}>
      {/* ═══════  TOP NAV  ═══════ */}
      <View
        style={[
          styles.nav,
          {
            paddingTop: navPadTop,
            paddingBottom: navPadBottom,
            paddingHorizontal: navPadH,
          },
        ]}
      >
        <View style={[styles.navLeft, { gap: navGap }]}>
          <Pressable>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Home</Text>
          </Pressable>
          <Pressable onPress={onStartSession}>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>
              Start a Session
            </Text>
          </Pressable>
        </View>
        <View style={[styles.navRight, { gap: navGap }]}>
          <Pressable onPress={onSignUp}>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Sign Up</Text>
          </Pressable>
          <Pressable onPress={onLogIn}>
            <Text style={[styles.navLink, { fontSize: navFontSize }]}>Log In</Text>
          </Pressable>
        </View>
      </View>

      {/* ═══════  MAIN CONTENT  ═══════ */}
      <ContentWrapper {...contentWrapperProps}>
        {/* Left / only column */}
        <View
          style={[
            styles.left,
            {
              paddingRight: mainIsRow ? leftPadRight : mainPadLeft,
              paddingLeft: isSmall ? mainPadLeft : 0,
              paddingBottom: isSmall ? 32 : 32,
            },
          ]}
        >
          <Image
            source={adioLogo}
            style={{ width: logoW, height: logoH, marginBottom: 16 }}
            resizeMode="contain"
          />

          <Text
            style={[
              styles.tagline,
              {
                fontSize: taglineFontSize,
                lineHeight: taglineLineH,
                marginBottom: taglineMB,
              },
            ]}
          >
            See it. Say it. Understand it.
          </Text>

          {/* Two content columns — stack on small */}
          <View
            style={[
              styles.contentCols,
              {
                flexDirection: isSmall ? "column" : "row",
                gap: colsGap,
              },
            ]}
          >
            <View style={styles.contentCol}>
              <Text style={[styles.sectionTitle, { fontSize: sectionFontSize }]}>
                What We Do
              </Text>
              <View style={[styles.bulletList, { gap: bulletGap }]}>
                <Bullet text="We help children turn words into clear mental images"          dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="Students describe and refine what they see to strengthen understanding" dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="Our method boosts comprehension, memory, and language skills"  dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="Lessons are calm, structured, and supportive"                  dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="Light gamification motivates without distracting"              dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="We build confidence, independent thinking, and clear communication" dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
              </View>
            </View>

            <View style={styles.contentCol}>
              <Text style={[styles.sectionTitle, { fontSize: sectionFontSize }]}>
                How It Works
              </Text>
              <View style={[styles.bulletList, { gap: bulletGap }]}>
                <Bullet text="Your child signs in and starts a session"                     dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="They describe what they see using their voice"                 dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="Our AI listens, evaluates, and asks follow-up questions"       dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="A summary shows progress in observation, understanding, and engagement" dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
                <Bullet text="Every session is unique — no two are the same"                dotSize={bulletDotSize} textSize={bulletFontSize} lineH={bulletLineH} />
              </View>
            </View>
          </View>

          <View style={[styles.footer, { marginTop: footerMT }]}>
            <Text style={[styles.footerText, { fontSize: footerFontSize }]}>
              © 2026 Adio. All rights reserved.{"  "}•{"  "}
              <Text
                style={styles.footerLink}
                onPress={() => window.open(getAssetUri(termsPdf), "_blank")}
              >
                Terms & Conditions
              </Text>
              {"  "}•{"  "}
              <Text
                style={styles.footerLink}
                onPress={() => window.open(getAssetUri(privacyPdf), "_blank")}
              >
                Privacy Policy
              </Text>
            </Text>
          </View>
        </View>

        {/* Right — hero image; hidden on small screens */}
        {mainIsRow && (
          <View style={[styles.right, { flex: isMedium ? 0.5 : 0.65 }]}>
            <View style={styles.heroImageWrap}>
              <Image source={heroImg} style={styles.heroImage} resizeMode="cover" />
            </View>
          </View>
        )}
      </ContentWrapper>
    </View>
  );
}

/* ── Bullet component ── */

function Bullet({
  text,
  dotSize,
  textSize,
  lineH,
}: {
  text: string;
  dotSize: number;
  textSize: number;
  lineH: number;
}) {
  return (
    <View style={styles.bulletRow}>
      <Text style={[styles.bulletDot, { fontSize: dotSize, lineHeight: lineH }]}>•</Text>
      <Text style={[styles.bulletText, { fontSize: textSize, lineHeight: lineH }]}>
        {text}
      </Text>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Base styles (size-independent)
   ═══════════════════════════════════════════════════════════════ */

const styles = StyleSheet.create({
  root: {
    backgroundColor: colors.bg,
    overflow: "hidden",
  },

  nav: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  navLeft: { flexDirection: "row" },
  navRight: { flexDirection: "row" },
  navLink: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
  },

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
    marginBottom: 20,
  },

  bulletList: {},
  bulletRow: {
    flexDirection: "row",
    alignItems: "flex-start",
  },
  bulletDot: {
    fontFamily: fonts.body,
    color: colors.darkBlue,
    marginRight: 14,
  },
  bulletText: {
    fontFamily: fonts.body,
    color: colors.darkBlueText,
    flex: 1,
  },

  right: {
    justifyContent: "flex-end",
  },
  heroImageWrap: {
    flex: 1,
    borderTopLeftRadius: 999,
    borderTopRightRadius: 999,
    overflow: "hidden",
    backgroundColor: "#E0EAF5",
  },
  heroImage: {
    width: "100%",
    height: "100%",
  },

  footer: {},
  footerText: {
    fontFamily: fonts.body,
    color: colors.textMuted,
    lineHeight: 26,
  },
  footerLink: {
    fontFamily: fonts.bodySemiBold,
    color: colors.darkBlue,
    textDecorationLine: "underline",
  },
});
