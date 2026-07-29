import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Svg, { Path } from "react-native-svg";
import { colors, fonts } from "../theme";

/**
 * Collection tab (03_navigation_shell.md §2). The creature collection wall —
 * per-session surprise drops and archived milestone creatures
 * (`PRDs/gamification/01_collection.md`). The wall itself is built by the
 * gamification feature; this is the calm placeholder surface it slots into so
 * the tab shell is complete and navigable today.
 */
export default function CollectionScreen() {
  return (
    <View style={styles.root}>
      <View style={styles.iconBadge}>
        <Svg width={54} height={54} viewBox="0 0 24 24">
          <Path
            d="M4 5 h7 v7 h-7 Z M13 5 h7 v7 h-7 Z M4 14 h7 v5 h-7 Z M13 14 h7 v5 h-7 Z"
            stroke={colors.darkBlue}
            strokeWidth={1.6}
            fill={colors.yellowCard}
            strokeLinejoin="round"
          />
        </Svg>
      </View>
      <Text style={styles.title}>Your Collection</Text>
      <Text style={styles.body}>
        Every session brings a new friend to your wall. They'll all gather here as
        you go.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
  },
  iconBadge: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: colors.white,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 24,
    boxShadow: "16px 16px 34px rgba(0,0,0,0.1), inset -8px -8px 16px rgba(251,222,40,0.4)" as any,
  },
  title: {
    fontFamily: fonts.fredoka,
    fontSize: 30,
    color: colors.darkBlue,
    textAlign: "center",
  },
  body: {
    fontFamily: fonts.body,
    fontSize: 17,
    lineHeight: 25,
    color: colors.darkBlueText,
    textAlign: "center",
    marginTop: 12,
    maxWidth: 340,
  },
});
