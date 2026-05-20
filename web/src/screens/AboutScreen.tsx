import React from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  useWindowDimensions,
  Platform,
} from "react-native";
import { colors, fonts } from "../theme";

interface Props {
  onBack: () => void;
}

export default function AboutScreen({ onBack }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();

  const pad      = Math.max(14, winW * 0.018);
  const titleSz  = Math.max(32, Math.min(56, winH * 0.07));
  const bodySz   = Math.max(16, Math.min(22, winH * 0.026));
  const hintSz   = Math.max(13, Math.min(16, winH * 0.02));
  const maxBody  = 720;

  return (
    <View style={[styles.container, { width: winW, height: winH }]}>
      <Pressable style={styles.backBtn} onPress={onBack}>
        <Text style={[styles.backText, { fontSize: hintSz }]}>← Back</Text>
      </Pressable>

      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={[styles.scrollContent, { padding: pad * 2, paddingTop: pad * 5 }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={{ maxWidth: maxBody, width: "100%" }}>
          <Text style={[styles.title, { fontSize: titleSz, marginBottom: pad * 1.5 }]}>
            About Me
          </Text>

          <Text style={[styles.body, { fontSize: bodySz, lineHeight: bodySz * 1.6, marginBottom: pad }]}>
            Hi, I'm Sidharth: a high schooler from Redmond, WA who builds things
            at the intersection of language and technology.
          </Text>

          <Text style={[styles.body, { fontSize: bodySz, lineHeight: bodySz * 1.6, marginBottom: pad }]}>
            I started Adio because I have a younger brother with autism, and I
            saw firsthand how hard it was for kids like him to get tools that
            actually worked with their voice, not against it. So I built one.
          </Text>

          <Text style={[styles.body, { fontSize: bodySz, lineHeight: bodySz * 1.6, marginBottom: pad }]}>
            Adio won 1st Place at the Washington State Science and Engineering
            Fair in 2026 and is live at adioreading.com. I'm still actively
            building it.
          </Text>

          <Text style={[styles.body, { fontSize: bodySz, lineHeight: bodySz * 1.6, marginBottom: pad }]}>
            When I'm not working on Adio, I'm into linguistics, making up
            languages, and competing in math and CS competitions.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "flex-start",
  },
  backBtn: {
    position: "absolute",
    top: 40,
    left: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
    zIndex: 10,
    elevation: 10,
  },
  backText: {
    fontFamily: fonts.bodySemiBold,
    color: colors.darkBlue,
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: "center",
  },
  title: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
    textAlign: "left",
  },
  body: {
    fontFamily: fonts.body,
    color: colors.darkBlueText,
    textAlign: "left",
  },
});
