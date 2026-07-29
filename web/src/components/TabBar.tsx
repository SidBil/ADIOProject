import React from "react";
import { View, Text, Pressable, StyleSheet, Platform } from "react-native";
import Svg, { Path, Circle } from "react-native-svg";
import { colors, fonts } from "../theme";

export type TabKey = "home" | "collection" | "progress" | "adult";

interface Props {
  active: TabKey;
  onChange: (tab: TabKey) => void;
}

/**
 * Persistent bottom tab bar — the app's top-level navigation shell
 * (03_navigation_shell.md §2). Four peer surfaces the child moves between
 * freely: Home / Collection / Progress / Adult. Hidden during an active session
 * (the parent decides when to mount it). Home is the default tab.
 *
 * The gamification mechanics are distributed across the shell — streak on Home,
 * collection on Collection, milestones on Progress — rather than crammed onto
 * one wall.
 */
export default function TabBar({ active, onChange }: Props) {
  return (
    <View style={styles.bar}>
      <Tab k="home" label="Home" active={active} onChange={onChange} icon={HomeIcon} />
      <Tab k="collection" label="Collection" active={active} onChange={onChange} icon={CollectionIcon} />
      <Tab k="progress" label="Progress" active={active} onChange={onChange} icon={ProgressIcon} />
      <Tab k="adult" label="Settings" active={active} onChange={onChange} icon={SettingsIcon} />
    </View>
  );
}

function Tab({
  k,
  label,
  active,
  onChange,
  icon: Icon,
}: {
  k: TabKey;
  label: string;
  active: TabKey;
  onChange: (t: TabKey) => void;
  icon: (p: { color: string }) => React.ReactElement;
}) {
  const isActive = active === k;
  const color = isActive ? colors.darkBlue : colors.textMuted;
  return (
    <Pressable
      style={styles.tab}
      onPress={() => onChange(k)}
      accessibilityRole="button"
      accessibilityState={{ selected: isActive }}
      accessibilityLabel={label}
    >
      <View style={[styles.iconWrap, isActive && styles.iconWrapActive]}>
        <Icon color={color} />
      </View>
      <Text style={[styles.label, { color }]}>{label}</Text>
    </Pressable>
  );
}

const ICON = 26;

function HomeIcon({ color }: { color: string }) {
  return (
    <Svg width={ICON} height={ICON} viewBox="0 0 24 24">
      <Path d="M3 11 L12 3 L21 11 M5 9.5 V20 h5 v-6 h4 v6 h5 V9.5" stroke={color} strokeWidth={2} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}
function CollectionIcon({ color }: { color: string }) {
  return (
    <Svg width={ICON} height={ICON} viewBox="0 0 24 24">
      <Path d="M4 5 h7 v7 h-7 Z M13 5 h7 v7 h-7 Z M4 14 h7 v5 h-7 Z M13 14 h7 v5 h-7 Z" stroke={color} strokeWidth={2} fill="none" strokeLinejoin="round" />
    </Svg>
  );
}
function ProgressIcon({ color }: { color: string }) {
  // Upward trending line ("trending-up") — growth over time, the point of the
  // Progress tab. Distinct from the Collection grid and the bar-chart it replaces.
  return (
    <Svg width={ICON} height={ICON} viewBox="0 0 24 24">
      <Path
        d="M23 6 L13.5 15.5 L8.5 10.5 L1 18 M17 6 H23 V12"
        stroke={color}
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}
function SettingsIcon({ color }: { color: string }) {
  // Gear/cog (Feather "settings") — the Settings tab.
  return (
    <Svg width={ICON} height={ICON} viewBox="0 0 24 24">
      <Circle cx={12} cy={12} r={3} stroke={color} strokeWidth={2} fill="none" />
      <Path
        d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
        stroke={color}
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    backgroundColor: colors.white,
    // Floating claymorphic card: detached from the screen edges, rounded, with
    // the app's puffy layered clay shadow (dark drop bottom-right + light glow
    // top-left + soft inset highlight).
    marginHorizontal: 16,
    marginBottom: Platform.OS === "ios" ? 28 : 16,
    borderRadius: 30,
    paddingVertical: 12,
    paddingHorizontal: 8,
    // Claymorphic: soft dark drop (floating) + inset highlight for the puffy
    // clay sculpt. No bright white OUTER glow (that was the "glow").
    boxShadow:
      "12px 14px 32px rgba(45,30,10,0.16), inset 6px 6px 14px rgba(255,255,255,0.6), inset -6px -6px 14px rgba(120,90,50,0.12)" as any,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 2,
  },
  iconWrap: {
    paddingHorizontal: 16,
    paddingVertical: 3,
    borderRadius: 14,
  },
  iconWrapActive: {
    backgroundColor: "#E7E7EC",
  },
  label: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 11,
  },
});
