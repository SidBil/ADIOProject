import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Platform,
  Pressable,
  useWindowDimensions,
} from "react-native";
import Svg, { Path } from "react-native-svg";
import { colors, fonts } from "../theme";
import { getCelebration } from "../api";

interface Props {
  sessionId: string;
  onDone: () => void;
}

// Single fixed base message — no pool, no randomization, same every session.
const BASE_TITLE = "You did it!";
const BASE_SUBTITLE = "Great work finishing your session.";

function fmtTime(ms: number | null): string {
  if (ms == null) return "—";
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function CelebrationScreen({ sessionId, onDone }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();
  const [stats, setStats] = useState<{ accuracy: number | null; total_time_ms: number | null }>({
    accuracy: null,
    total_time_ms: null,
  });
  const [donePressed, setDonePressed] = useState(false);

  // The base celebration renders immediately; the stat cards fill in when the
  // fetch resolves. A failure just leaves them blank ("—").
  useEffect(() => {
    let alive = true;
    getCelebration(sessionId)
      .then((d) => {
        if (alive) setStats({ accuracy: d?.accuracy ?? null, total_time_ms: d?.total_time_ms ?? null });
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [sessionId]);

  const badge = Math.min(150, Math.max(112, winW * 0.34));
  const titleSz = Math.min(52, winW * 0.13);

  // Claymorphic recipes — same layered outer-drop + warm inset language used
  // across the app (WelcomeScreen / SessionScreen cards).
  const badgeClay =
    "18px 18px 40px rgba(120,60,0,0.18), inset -10px -10px 22px rgba(245,158,11,0.55)";
  const blueClay =
    "12px 12px 28px rgba(10,70,130,0.16), inset -8px -8px 18px rgba(50,180,240,0.4)";
  const greenClay =
    "12px 12px 28px rgba(90,120,0,0.16), inset -8px -8px 18px rgba(170,210,20,0.45)";
  const doneClay = donePressed
    ? "8px 8px 20px rgba(0,0,0,0.14), inset -4px -4px 12px rgba(245,158,11,0.55)"
    : "18px 18px 38px rgba(0,0,0,0.18), inset -10px -10px 18px rgba(245,158,11,0.6)";
  const clayTransition =
    Platform.OS === "web"
      ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
      : undefined;

  return (
    <View style={[styles.root, { width: winW, minHeight: winH }]}>
      <View style={styles.content}>
        {/* ── Completion badge: a clay checkmark ── */}
        <View
          style={[
            styles.badge,
            { width: badge, height: badge, borderRadius: badge / 2, boxShadow: badgeClay } as any,
          ]}
        >
          <Svg width={badge * 0.56} height={badge * 0.56} viewBox="0 0 100 100">
            <Path
              d="M22 52 L42 72 L78 30"
              stroke={colors.darkBlue}
              strokeWidth={12}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          </Svg>
        </View>

        {/* ── Fixed warm message (Fredoka header) ── */}
        <Text
          style={[styles.title, { fontSize: titleSz }]}
          numberOfLines={1}
          adjustsFontSizeToFit
        >
          {BASE_TITLE}
        </Text>
        <Text style={styles.subtitle}>{BASE_SUBTITLE}</Text>

        {/* ── Accuracy + time cards ── */}
        <View style={styles.cardRow}>
          <StatCard
            label="Accuracy"
            value={stats.accuracy != null ? `${stats.accuracy}` : "—"}
            unit={stats.accuracy != null ? "/ 5" : undefined}
            bg={colors.blueCard}
            clay={blueClay}
            transition={clayTransition}
          />
          <StatCard
            label="Time"
            value={fmtTime(stats.total_time_ms)}
            bg={colors.greenBtn}
            clay={greenClay}
            transition={clayTransition}
          />
        </View>

        {/* ── Single manual dismiss ── */}
        <Pressable
          onPress={onDone}
          onPressIn={() => setDonePressed(true)}
          onPressOut={() => setDonePressed(false)}
          style={styles.donePressable}
        >
          <View
            style={[
              styles.doneBtn,
              { boxShadow: doneClay, transform: [{ translateY: donePressed ? 3 : 0 }] } as any,
              clayTransition,
            ]}
          >
            <Text style={styles.doneText}>Done</Text>
          </View>
        </Pressable>
      </View>
    </View>
  );
}

function StatCard({
  label,
  value,
  unit,
  bg,
  clay,
  transition,
}: {
  label: string;
  value: string;
  unit?: string;
  bg: string;
  clay: string;
  transition: any;
}) {
  return (
    <View style={[styles.statCard, { backgroundColor: bg, boxShadow: clay } as any, transition]}>
      <Text style={styles.statLabel}>{label}</Text>
      <View style={styles.statValueRow}>
        <Text style={styles.statValue}>{value}</Text>
        {unit ? <Text style={styles.statUnit}>{unit}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 28,
  },
  content: {
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    maxWidth: 360,
  },
  badge: {
    backgroundColor: colors.yellowCard,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 28,
  },
  title: {
    fontFamily: fonts.fredoka,
    color: colors.darkBlue,
    textAlign: "center",
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 18,
    lineHeight: 25,
    color: colors.darkBlueText,
    textAlign: "center",
    marginTop: 10,
    paddingHorizontal: 8,
  },

  cardRow: {
    flexDirection: "row",
    gap: 14,
    marginTop: 28,
    alignSelf: "stretch",
  },
  statCard: {
    flex: 1,
    borderRadius: 22,
    paddingVertical: 18,
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  statLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.darkBlueText,
    marginBottom: 6,
  },
  statValueRow: {
    flexDirection: "row",
    alignItems: "flex-end",
  },
  statValue: {
    fontFamily: fonts.fredoka,
    fontSize: 32,
    color: colors.darkBlue,
  },
  statUnit: {
    fontFamily: fonts.fredoka,
    fontSize: 16,
    color: colors.darkBlueText,
    marginLeft: 4,
    marginBottom: 5,
  },

  donePressable: {
    marginTop: 36,
  },
  doneBtn: {
    backgroundColor: colors.yellowCard,
    borderRadius: 28,
    paddingVertical: 16,
    paddingHorizontal: 56,
    alignItems: "center",
    justifyContent: "center",
  },
  doneText: {
    fontFamily: fonts.fredoka,
    fontSize: 26,
    color: colors.darkBlue,
  },
});
