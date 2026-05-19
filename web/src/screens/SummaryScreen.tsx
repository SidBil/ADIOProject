import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Platform,
  Pressable,
} from "react-native";
import Svg, { Path, Circle, Line } from "react-native-svg";
import { colors, fonts } from "../theme";
import { getSummary, imageUrl } from "../api";
import { supabase } from "../lib/supabase";
import ShapePattern from "../components/ShapePattern";

interface Props {
  sessionId: string;
  imageId?: string;
  userId: string;
  onNewSession: () => void;
}

/* ═══════════════════════════════════════════════════════════════
   Gauge Meter — semicircular speedometer with needle
   ═══════════════════════════════════════════════════════════════ */

function GaugeMeter({ value, size = 140 }: { value: number; size?: number }) {
  const cx = size / 2;
  const cy = size * 0.65;
  const r = size * 0.38;
  const strokeW = size * 0.13;
  const v = Math.max(0, Math.min(1, value));

  const pt = (angle: number) => ({
    x: cx + r * Math.cos(angle),
    y: cy - r * Math.sin(angle),
  });

  const arcPath = (a1: number, a2: number) => {
    const p1 = pt(a1);
    const p2 = pt(a2);
    const largeArc = Math.abs(a1 - a2) > Math.PI ? 1 : 0;
    return `M ${p1.x} ${p1.y} A ${r} ${r} 0 ${largeArc} 1 ${p2.x} ${p2.y}`;
  };

  const segments = [
    { from: Math.PI, to: (2 * Math.PI) / 3, color: "#F87171" },
    { from: (2 * Math.PI) / 3, to: Math.PI / 3, color: "#FBBF24" },
    { from: Math.PI / 3, to: 0.01, color: "#4ADE80" },
  ];

  const needleAngle = Math.PI * (1 - v);
  const needleLen = r * 0.82;
  const tip = {
    x: cx + needleLen * Math.cos(needleAngle),
    y: cy - needleLen * Math.sin(needleAngle),
  };

  return (
    <Svg
      width={size}
      height={size * 0.55}
      viewBox={`0 0 ${size} ${size * 0.7}`}
    >
      {segments.map((seg, i) => (
        <Path
          key={i}
          d={arcPath(seg.from, seg.to)}
          stroke={seg.color}
          strokeWidth={strokeW}
          fill="none"
          strokeLinecap="round"
        />
      ))}
      <Line
        x1={cx}
        y1={cy}
        x2={tip.x}
        y2={tip.y}
        stroke={colors.darkBlue}
        strokeWidth={3}
        strokeLinecap="round"
      />
      <Circle cx={cx} cy={cy} r={5} fill={colors.darkBlue} />
      <Circle cx={cx} cy={cy} r={2.5} fill="#fff" />
    </Svg>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Star SVG
   ═══════════════════════════════════════════════════════════════ */

function StarIcon({ size = 50, color = "#FBBF24" }: { size?: number; color?: string }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path
        d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
        fill={color}
        stroke="#E5A300"
        strokeWidth={0.5}
      />
    </Svg>
  );
}

/* ═══════════════════════════════════════════════════════════════
   3D Button
   ═══════════════════════════════════════════════════════════════ */

function Button3D({
  title,
  onPress,
  topColor,
  bottomColor,
  textColor,
}: {
  title: string;
  onPress: () => void;
  topColor: string;
  bottomColor: string;
  textColor: string;
}) {
  const [pressed, setPressed] = useState(false);
  const webStyle =
    Platform.OS === "web"
      ? ({
          transition:
            "transform 150ms cubic-bezier(0.445,0.05,0.55,0.95), box-shadow 150ms cubic-bezier(0.445,0.05,0.55,0.95)",
          boxShadow: pressed
            ? `0px 0px 0px ${bottomColor}`
            : `0px 8px 0px ${bottomColor}`,
          transform: pressed ? "translateY(8px)" : "translateY(0px)",
        } as any)
      : undefined;

  return (
    <Pressable
      onPress={onPress}
      onPressIn={() => setPressed(true)}
      onPressOut={() => setPressed(false)}
    >
      <View
        style={[
          {
            backgroundColor: topColor,
            borderWidth: 4,
            borderColor: bottomColor,
            borderRadius: 999,
            paddingVertical: 14,
            paddingHorizontal: 40,
            alignItems: "center",
            flexDirection: "row",
            gap: 8,
          },
          Platform.OS === "web"
            ? { shadowOpacity: 0, elevation: 0 }
            : {
                shadowColor: bottomColor,
                shadowOffset: { width: 0, height: 8 },
                shadowOpacity: 1,
                shadowRadius: 0,
                elevation: 4,
              },
          webStyle,
        ]}
      >
        <Text style={{ fontFamily: fonts.heading, fontSize: 22, color: textColor }}>
          {title}
        </Text>
        <Text style={{ fontSize: 20, color: textColor }}>›</Text>
      </View>
    </Pressable>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Main Summary Screen
   ═══════════════════════════════════════════════════════════════ */

export default function SummaryScreen({
  sessionId,
  imageId,
  userId,
  onNewSession,
}: Props) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const saved = useRef(false);
  const [burstCount, setBurstCount] = useState(0);

  useEffect(() => {
    getSummary(sessionId, userId)
      .then((d) => {
        setData(d);
        if (!saved.current) {
          saved.current = true;
          const history = (d.qa_history || []).map((item: any) => ({
            question: item.question,
            structure_word: item.structure_word,
            expected_answer: item.expected_answer,
            transcription: item.transcription,
            evaluation: item.evaluation,
            followup: item.followup,
            initiation_latency_ms: item.initiation_latency_ms,
          }));
          const scores = d.scores || {};
          supabase
            .from("sessions")
            .insert({
              user_id: userId,
              session_id: sessionId,
              image_id: imageId || d.image_id || null,
              questions_answered: d.progress?.answered ?? 0,
              total_questions: d.progress?.total ?? 0,
              qa_history: history,
              avg_latency_ms: scores.avg_latency_ms ?? null,
              observation_score: scores.observation ?? null,
              understanding_score: scores.understanding ?? null,
              engagement_score: scores.engagement ?? null,
            })
            .then(({ error: insertErr }) => {
              if (insertErr)
                console.warn("Failed to save session:", insertErr.message);
            });
        }
      })
      .catch((e) => setError(e.message));
  }, [sessionId]);

  if (error) {
    return (
      <View style={styles.container}>
        <ShapePattern />
        <Text style={styles.errorText}>Could not load summary: {error}</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={onNewSession}>
          <Text style={styles.retryBtnText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!data) {
    return (
      <View style={styles.container}>
        <ShapePattern />
        <ActivityIndicator size="large" color={colors.darkBlue} />
      </View>
    );
  }

  const progress = data.progress || {};
  const history: any[] = data.qa_history || [];
  const scores = data.scores || {};
  const answered = progress.answered || 0;
  const total = progress.total || 0;

  // Compute stars: count questions with overall_score >= 3
  let starsEarned = 0;
  for (const item of history) {
    if (item.evaluation?.overall_score >= 3) starsEarned++;
  }

  const engagementBuilding =
    scores.engagement === null || scores.engagement === undefined;
  const sessionsToward = scores.sessions_toward_baseline ?? 0;
  const baselineMin = scores.baseline_min_sessions ?? 3;

  // Dynamic encouragement
  const avgScore =
    scores.observation != null && scores.understanding != null
      ? (scores.observation + scores.understanding) / 2
      : null;
  const encourageTitle =
    avgScore != null && avgScore >= 0.8
      ? "Amazing work!"
      : avgScore != null && avgScore >= 0.5
      ? "Keep it up!"
      : "Great effort!";
  const encourageMsg =
    avgScore != null && avgScore >= 0.8
      ? "You're really getting the hang of describing what you see!"
      : avgScore != null && avgScore >= 0.5
      ? "The more you practice, the better you'll become at spotting details!"
      : "Every session helps you get better. Keep practicing!";

  return (
    <View style={styles.container}>
      <ShapePattern burst={burstCount} />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Great Job Card ── */}
        <View
          style={[
            styles.card3d,
            {
              backgroundColor: colors.greenBtn,
              borderColor: colors.greenBorder,
            },
            Platform.OS === "web"
              ? ({ boxShadow: `0px 8px 0px ${colors.greenBorder}` } as any)
              : {
                  shadowColor: colors.greenBorder,
                  shadowOffset: { width: 0, height: 8 },
                  shadowOpacity: 1,
                  shadowRadius: 0,
                },
          ]}
        >
          <View style={styles.greatJobRow}>
            <StarIcon size={44} />
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.greatJobTitle}>Great job!</Text>
              <Text style={styles.greatJobSub}>
                You explored the scene and answered {answered} question
                {answered !== 1 ? "s" : ""}.
              </Text>
            </View>
          </View>
        </View>

        {/* ── Score Cards Row ── */}
        <View style={styles.scoreRow}>
          {/* Understanding */}
          <View
            style={[
              styles.gaugeCard,
              {
                backgroundColor: colors.greenBtn,
                borderColor: colors.greenBorder,
              },
              Platform.OS === "web"
                ? ({ boxShadow: `0px 6px 0px ${colors.greenBorder}` } as any)
                : {
                    shadowColor: colors.greenBorder,
                    shadowOffset: { width: 0, height: 6 },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                  },
            ]}
          >
            <Text style={styles.gaugeTitle}>Understanding</Text>
            {scores.understanding != null ? (
              <>
                <GaugeMeter value={scores.understanding} />
                <Text style={styles.gaugePct}>
                  {Math.round(scores.understanding * 100)}%
                </Text>
                <Text style={styles.gaugeDesc}>
                  You understood most of what you saw!
                </Text>
              </>
            ) : (
              <Text style={styles.gaugeNa}>—</Text>
            )}
          </View>

          {/* Observation */}
          <View
            style={[
              styles.gaugeCard,
              {
                backgroundColor: colors.pinkCard,
                borderColor: colors.pinkBorder,
              },
              Platform.OS === "web"
                ? ({ boxShadow: `0px 6px 0px ${colors.pinkBorder}` } as any)
                : {
                    shadowColor: colors.pinkBorder,
                    shadowOffset: { width: 0, height: 6 },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                  },
            ]}
          >
            <Text style={styles.gaugeTitle}>Observation</Text>
            {scores.observation != null ? (
              <>
                <GaugeMeter value={scores.observation} />
                <Text style={styles.gaugePct}>
                  {Math.round(scores.observation * 100)}%
                </Text>
                <Text style={styles.gaugeDesc}>
                  You noticed lots of great details!
                </Text>
              </>
            ) : (
              <Text style={styles.gaugeNa}>—</Text>
            )}
          </View>

          {/* Engagement */}
          <View
            style={[
              styles.gaugeCard,
              {
                backgroundColor: colors.blueCard,
                borderColor: colors.blueBorder,
              },
              Platform.OS === "web"
                ? ({ boxShadow: `0px 6px 0px ${colors.blueBorder}` } as any)
                : {
                    shadowColor: colors.blueBorder,
                    shadowOffset: { width: 0, height: 6 },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                  },
            ]}
          >
            <Text style={styles.gaugeTitle}>Engagement</Text>
            {engagementBuilding ? (
              <View style={styles.buildingWrap}>
                <Text style={styles.buildingText}>Building baseline…</Text>
                <Text style={styles.buildingProg}>
                  {sessionsToward} / {baselineMin} sessions
                </Text>
              </View>
            ) : scores.engagement != null ? (
              <>
                <GaugeMeter value={scores.engagement} />
                <Text style={styles.gaugePct}>
                  {Math.round(scores.engagement * 100)}%
                </Text>
                <Text style={styles.gaugeDesc}>
                  You stayed focused and did an awesome job!
                </Text>
              </>
            ) : (
              <Text style={styles.gaugeNa}>—</Text>
            )}
          </View>

          {/* Stars Earned */}
          <View
            style={[
              styles.gaugeCard,
              {
                backgroundColor: colors.yellowCard,
                borderColor: colors.yellowBorder,
              },
              Platform.OS === "web"
                ? ({ boxShadow: `0px 6px 0px ${colors.yellowBorder}` } as any)
                : {
                    shadowColor: colors.yellowBorder,
                    shadowOffset: { width: 0, height: 6 },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                  },
            ]}
          >
            <Text style={styles.gaugeTitle}>You earned</Text>
            <View style={styles.starWrap}>
              <StarIcon size={64} />
            </View>
            <Text style={styles.gaugePct}>
              {starsEarned} / {total}
            </Text>
            <Text style={styles.gaugeDesc}>stars</Text>
          </View>
        </View>

        {/* ── Encouragement Card ── */}
        <View
          style={[
            styles.card3d,
            {
              backgroundColor: "#F3EEFF",
              borderColor: "#B89AFF",
            },
            Platform.OS === "web"
              ? ({ boxShadow: `0px 6px 0px #B89AFF` } as any)
              : {
                  shadowColor: "#B89AFF",
                  shadowOffset: { width: 0, height: 6 },
                  shadowOpacity: 1,
                  shadowRadius: 0,
                },
          ]}
        >
          <View style={styles.greatJobRow}>
            <Text style={{ fontSize: 28 }}>💡</Text>
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.encourageTitle}>{encourageTitle}</Text>
              <Text style={styles.encourageMsg}>{encourageMsg}</Text>
            </View>
          </View>
        </View>

        {/* ── Q&A History ── */}
        <Text style={styles.historyTitle}>Question History</Text>
        {history.map((item: any, idx: number) => {
          const fb = item.followup || item.evaluation?.feedback || "";
          const latency = item.initiation_latency_ms;
          return (
            <View key={idx} style={styles.historyItem}>
              <View style={styles.hHeader}>
                <Text style={styles.hQuestion}>{item.question}</Text>
                {latency != null && (
                  <Text style={styles.hLatency}>
                    {(latency / 1000).toFixed(1)}s
                  </Text>
                )}
              </View>
              <Text style={styles.hDetail}>
                <Text style={styles.bold}>Expected: </Text>
                {item.expected_answer || "—"}
              </Text>
              <Text style={styles.hDetail}>
                <Text style={styles.bold}>You said: </Text>
                {item.transcription || "—"}
              </Text>
              {fb ? <Text style={styles.hFeedback}>"{fb}"</Text> : null}
            </View>
          );
        })}

        {/* ── Continue Button ── */}
        <View style={{ marginTop: 16, marginBottom: 40, alignItems: "center" }}>
          <Button3D
            title="Continue"
            onPress={() => {
              setBurstCount((n) => n + 1);
              onNewSession();
            }}
            topColor={colors.yellowCard}
            bottomColor={colors.yellowBorder}
            textColor={colors.darkBlue}
          />
        </View>
      </ScrollView>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Styles
   ═══════════════════════════════════════════════════════════════ */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    padding: 16,
  },
  scroll: { flex: 1, width: "100%" },
  scrollContent: {
    alignItems: "center",
    paddingVertical: 24,
    paddingHorizontal: 4,
    maxWidth: 700,
    alignSelf: "center",
    width: "100%",
  },

  /* 3D Cards */
  card3d: {
    borderWidth: 4,
    borderRadius: 22,
    padding: 18,
    width: "100%",
    marginBottom: 16,
  },

  /* Great Job */
  greatJobRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  greatJobTitle: {
    fontFamily: fonts.heading,
    fontSize: 26,
    color: colors.darkBlue,
  },
  greatJobSub: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.darkBlueText,
    marginTop: 2,
  },

  /* Score Cards */
  scoreRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    width: "100%",
    marginBottom: 16,
    justifyContent: "center",
  },
  gaugeCard: {
    borderWidth: 4,
    borderRadius: 20,
    padding: 14,
    alignItems: "center",
    minWidth: 145,
    flex: 1,
    maxWidth: 180,
  },
  gaugeTitle: {
    fontFamily: fonts.heading,
    fontSize: 15,
    color: colors.darkBlue,
    marginBottom: 6,
    textAlign: "center",
  },
  gaugePct: {
    fontFamily: fonts.heading,
    fontSize: 24,
    color: colors.darkBlue,
    marginTop: 2,
  },
  gaugeDesc: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.darkBlueText,
    textAlign: "center",
    marginTop: 4,
    lineHeight: 16,
  },
  gaugeNa: {
    fontFamily: fonts.heading,
    fontSize: 28,
    color: colors.textMuted,
    height: 90,
    lineHeight: 90,
  },
  starWrap: {
    height: 72,
    alignItems: "center",
    justifyContent: "center",
  },
  buildingWrap: {
    height: 90,
    alignItems: "center",
    justifyContent: "center",
  },
  buildingText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.darkBlue,
    textAlign: "center",
  },
  buildingProg: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 4,
  },

  /* Encouragement */
  encourageTitle: {
    fontFamily: fonts.heading,
    fontSize: 18,
    color: colors.darkBlue,
  },
  encourageMsg: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlueText,
    marginTop: 2,
    lineHeight: 20,
  },

  /* History */
  historyTitle: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.darkBlue,
    marginBottom: 12,
    alignSelf: "flex-start",
    width: "100%",
  },
  historyItem: {
    backgroundColor: colors.cardWhite,
    borderRadius: 16,
    borderWidth: 3,
    borderColor: "#e0e0e8",
    padding: 16,
    marginBottom: 10,
    width: "100%",
  },
  hHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 6,
  },
  hQuestion: {
    fontFamily: fonts.heading,
    fontSize: 17,
    color: colors.darkBlue,
    flex: 1,
  },
  hLatency: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13,
    color: colors.textMuted,
    backgroundColor: "#f0f0f6",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
    marginLeft: 8,
    overflow: "hidden",
  },
  hDetail: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlueText,
    lineHeight: 22,
  },
  bold: { fontWeight: "700" },
  hFeedback: {
    fontFamily: fonts.body,
    fontStyle: "italic",
    fontSize: 14,
    color: colors.blueBorder,
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: "#e8e8f0",
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: "#cc0000",
    textAlign: "center",
    marginBottom: 20,
  },
  retryBtn: {
    backgroundColor: colors.darkBlueBtnBg,
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 36,
  },
  retryBtnText: {
    fontFamily: fonts.heading,
    fontSize: 18,
    color: colors.white,
  },
});
