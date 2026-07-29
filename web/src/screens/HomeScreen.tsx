import React, { useEffect, useState, useCallback } from "react";
import {
  View,
  Text,
  Image,
  Pressable,
  StyleSheet,
  Platform,
  useWindowDimensions,
} from "react-native";
import Svg, { Path, Defs, LinearGradient, Stop, Rect } from "react-native-svg";
import { colors, fonts } from "../theme";
import Spinner from "../components/Spinner";
import { getHomeState } from "../api";
import SessionPath from "../components/SessionPath";

interface Props {
  /** Starts the current node's session. Throws on failure. */
  onStart: () => Promise<void>;
  /** Bumped by the parent after a session completes so Home refetches position. */
  refreshKey?: number;
}

/**
 * The Home tab surface (01_home_screen.md). Hosts the winding session path (the
 * dominant element) and a compact streak badge in the top corner. Deliberately
 * uncluttered so the trail is the unmistakable focus — the milestone bar,
 * collection wall, and adult controls live on their own tabs.
 */
export default function HomeScreen({ onStart, refreshKey }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();
  const [completedCount, setCompletedCount] = useState<number | null>(null);
  const [streakDays, setStreakDays] = useState(0);
  const [completedImages, setCompletedImages] = useState<string[]>([]);
  const [childName, setChildName] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    let alive = true;
    getHomeState()
      .then((s) => {
        if (!alive) return;
        setCompletedCount(s.completed_count ?? 0);
        setStreakDays(s.streak_days ?? 0);
        setCompletedImages(s.completed_images ?? []);
        setChildName(s.child_name ?? null);
      })
      .catch(() => {
        // A read failure shouldn't wall the child out of starting — assume a
        // fresh trail so node 1 is still tappable.
        if (alive) setCompletedCount((c) => (c == null ? 0 : c));
      });
    return () => {
      alive = false;
    };
  }, [refreshKey]);

  const handleStart = useCallback(async () => {
    setStarting(true);
    try {
      await onStart();
    } catch (e: any) {
      alert("Failed to start session. " + (e?.message ?? ""));
    } finally {
      setStarting(false);
    }
  }, [onStart]);

  const colW = Math.min(440, winW);
  const topPad = Platform.OS === "ios" ? 52 : 20;

  return (
    <View style={styles.root}>
      {/* ─── Warm header ─── */}
      <View style={[styles.header, { paddingTop: topPad }]}>
        <View style={styles.headerTop}>
          <Image
            source={require("../../assets/adiologo.png")}
            style={styles.logo}
            resizeMode="contain"
          />
          <StreakBadge days={streakDays} />
        </View>
        <Text style={styles.greeting}>
          {childName ? `Hi, ${childName}!` : "Hi there!"}
        </Text>
        <Text style={styles.subtitle}>Ready to learn?</Text>
      </View>

      {completedCount == null ? (
        <View style={styles.loading}>
          <Spinner size="large" />
        </View>
      ) : (
        <View style={{ flex: 1, width: colW, alignSelf: "center" }}>
          <SessionPath
            completedCount={completedCount}
            completedImages={completedImages}
            onStartNode={handleStart}
            busy={starting}
          />
          {/* Soft fade so trail content dissolves into the background as it meets
              the header, instead of a hard edge under the greeting text. */}
          <TopFade width={colW} />
        </View>
      )}
    </View>
  );
}

/**
 * A vertical gradient overlay pinned to the top of the trail, fading from the
 * page background (opaque) down to transparent. Non-interactive — taps fall
 * through to the trail beneath it.
 */
function TopFade({ width }: { width: number }) {
  const H = 56;
  return (
    <View
      pointerEvents="none"
      style={{ position: "absolute", top: 0, left: 0, right: 0, height: H, zIndex: 5 }}
    >
      <Svg width={width} height={H}>
        <Defs>
          <LinearGradient id="homeTopFade" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor={colors.bg} stopOpacity={1} />
            <Stop offset="1" stopColor={colors.bg} stopOpacity={0} />
          </LinearGradient>
        </Defs>
        <Rect x="0" y="0" width={width} height={H} fill="url(#homeTopFade)" />
      </Svg>
    </View>
  );
}

/**
 * The streak badge — present but secondary (01_home_screen.md §3). Shows a flame
 * plus the day count with a small "day streak" label; before any streak it warmly
 * nudges rather than showing a bare "0".
 */
function StreakBadge({ days }: { days: number }) {
  return (
    <View style={styles.badge}>
      {/* Flame artwork from assets/flame-icon.svg, inlined so it renders through
          react-native-svg on web + native. Two-tone (outer ember + inner glow). */}
      <Svg width={20} height={26} viewBox="0 0 92.27 122.88">
        <Path
          d="M18.61,54.89C15.7,28.8,30.94,10.45,59.52,0C42.02,22.71,74.44,47.31,76.23,70.89 c4.19-7.15,6.57-16.69,7.04-29.45c21.43,33.62,3.66,88.57-43.5,80.67c-4.33-0.72-8.5-2.09-12.3-4.13C10.27,108.8,0,88.79,0,69.68 C0,57.5,5.21,46.63,11.95,37.99C12.85,46.45,14.77,52.76,18.61,54.89L18.61,54.89z"
          fill="#EC6F59"
          fillRule="evenodd"
          clipRule="evenodd"
        />
        <Path
          d="M33.87,92.58c-4.86-12.55-4.19-32.82,9.42-39.93c0.1,23.3,23.05,26.27,18.8,51.14 c3.92-4.44,5.9-11.54,6.25-17.15c6.22,14.24,1.34,25.63-7.53,31.43c-26.97,17.64-50.19-18.12-34.75-37.72 C26.53,84.73,31.89,91.49,33.87,92.58L33.87,92.58z"
          fill="#FAD15C"
          fillRule="evenodd"
          clipRule="evenodd"
        />
      </Svg>
      {days > 0 ? (
        <View style={styles.badgeTextCol}>
          <Text style={styles.badgeNum}>{days}</Text>
          <Text style={styles.badgeLabel}>day{days === 1 ? "" : "s"}</Text>
        </View>
      ) : (
        <Text style={styles.badgeLabel}>Let's go!</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  loading: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  header: {
    paddingHorizontal: 22,
    paddingBottom: 8,
    zIndex: 20,
  },
  headerTop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  logo: {
    width: 84,
    height: 36,
  },
  greeting: {
    fontFamily: fonts.fredoka,
    fontSize: 30,
    color: colors.darkBlue,
    marginTop: 14,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.darkBlue,
    opacity: 0.6,
    marginTop: 2,
  },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: colors.white,
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 18,
  },
  badgeTextCol: {
    flexDirection: "row",
    alignItems: "baseline",
    gap: 4,
  },
  badgeNum: {
    fontFamily: fonts.heading,
    fontSize: 17,
    color: colors.darkBlue,
  },
  badgeLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13,
    color: colors.darkBlue,
    opacity: 0.7,
  },
});
