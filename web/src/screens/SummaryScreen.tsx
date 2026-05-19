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
  useWindowDimensions,
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

function GaugeMeter({ value }: { value: number | null }) {
  const W = 200;
  const H = 120;
  const cx = W / 2;
  const cy = H - 10;
  const r = 75;
  const sw = 22;
  const v = value != null ? Math.max(0, Math.min(1, value)) : 0;

  const pt = (a: number) => ({
    x: cx + r * Math.cos(a),
    y: cy - r * Math.sin(a),
  });

  const arc = (a1: number, a2: number) => {
    const p1 = pt(a1);
    const p2 = pt(a2);
    return `M ${p1.x} ${p1.y} A ${r} ${r} 0 0 1 ${p2.x} ${p2.y}`;
  };

  const segs = [
    { from: Math.PI, to: (2 * Math.PI) / 3, color: "#F87171" },
    { from: (2 * Math.PI) / 3, to: Math.PI / 3, color: "#FBBF24" },
    { from: Math.PI / 3, to: 0.02, color: "#4ADE80" },
  ];

  const na = Math.PI * (1 - v);
  const nl = r * 0.75;
  const tip = { x: cx + nl * Math.cos(na), y: cy - nl * Math.sin(na) };

  return (
    <Svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}>
      {segs.map((s, i) => (
        <Path
          key={i}
          d={arc(s.from, s.to)}
          stroke={s.color}
          strokeWidth={sw}
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
        strokeWidth={4}
        strokeLinecap="round"
      />
      <Circle cx={cx} cy={cy} r={7} fill={colors.darkBlue} />
      <Circle cx={cx} cy={cy} r={3} fill="#fff" />
    </Svg>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Star SVG
   ═══════════════════════════════════════════════════════════════ */

function StarIcon({ size = 50 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path
        d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
        fill="#FBBF24"
        stroke="#E5A300"
        strokeWidth={0.5}
      />
    </Svg>
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
  const { width: winW, height: winH } = useWindowDimensions();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const saved = useRef(false);
  const [burstCount, setBurstCount] = useState(0);
  const [contPressed, setContPressed] = useState(false);

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
          const sc = d.scores || {};
          supabase
            .from("sessions")
            .insert({
              user_id: userId,
              session_id: sessionId,
              image_id: imageId || d.image_id || null,
              questions_answered: d.progress?.answered ?? 0,
              total_questions: d.progress?.total ?? 0,
              qa_history: history,
              avg_latency_ms: sc.avg_latency_ms ?? null,
              observation_score: sc.observation ?? null,
              understanding_score: sc.understanding ?? null,
              engagement_score: sc.engagement ?? null,
            })
            .then(({ error: err }) => {
              if (err) console.warn("Save failed:", err.message);
            });
        }
      })
      .catch((e) => setError(e.message));
  }, [sessionId]);

  if (error) {
    return (
      <View style={s.container}>
        <ShapePattern />
        <Text style={s.errorText}>Could not load summary: {error}</Text>
        <TouchableOpacity style={s.retryBtn} onPress={onNewSession}>
          <Text style={s.retryBtnText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }
  if (!data) {
    return (
      <View style={s.container}>
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

  let starsEarned = 0;
  for (const item of history) {
    if (item.evaluation?.overall_score >= 3) starsEarned++;
  }

  const engBuilding = scores.engagement == null;
  const sessToward = scores.sessions_toward_baseline ?? 0;
  const baseMin = scores.baseline_min_sessions ?? 3;

  const avg =
    scores.observation != null && scores.understanding != null
      ? (scores.observation + scores.understanding) / 2
      : null;
  const encTitle =
    avg != null && avg >= 0.8
      ? "Amazing work!"
      : avg != null && avg >= 0.5
      ? "Keep it up!"
      : "Great effort!";
  const encMsg =
    avg != null && avg >= 0.8
      ? "You're really getting the hang of describing what you see!"
      : avg != null && avg >= 0.5
      ? "The more you practice, the better you'll become at spotting details!"
      : "Every session helps you get better. Keep practicing!";

  const contWebStyle =
    Platform.OS === "web"
      ? ({
          transition:
            "transform 150ms cubic-bezier(0.445,0.05,0.55,0.95), box-shadow 150ms cubic-bezier(0.445,0.05,0.55,0.95)",
          boxShadow: contPressed
            ? "0px 0px 0px #d4a017"
            : "0px 6px 0px #d4a017",
          transform: contPressed ? "translateY(6px)" : "translateY(0px)",
        } as any)
      : undefined;

  return (
    <View style={s.container}>
      <ShapePattern burst={burstCount} />
      <ScrollView
        style={s.scroll}
        contentContainerStyle={s.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ════════════  OUTER PINK CARD  ════════════ */}
        <View
          style={[
            s.outerCard,
            Platform.OS === "web"
              ? ({ boxShadow: `0px 10px 0px ${colors.pinkBorder}` } as any)
              : {
                  shadowColor: colors.pinkBorder,
                  shadowOffset: { width: 0, height: 10 },
                  shadowOpacity: 1,
                  shadowRadius: 0,
                },
          ]}
        >
          {/* ── Great Job Banner ── */}
          <View
            style={[
              s.bannerCard,
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
            <StarIcon size={40} />
            <View style={{ flex: 1, marginLeft: 14 }}>
              <Text style={s.bannerTitle}>Great job!</Text>
              <Text style={s.bannerSub}>
                You explored the scene and answered {answered} question
                {answered !== 1 ? "s" : ""}.
              </Text>
            </View>
          </View>

          {/* ── Scores Grid (3 gauges + stars/continue) ── */}
          <View style={s.gridRow}>
            {/* Left: 3 gauge cards */}
            <View style={s.gaugesCol}>
              {/* Understanding */}
              <View
                style={[
                  s.gaugeCard,
                  {
                    backgroundColor: colors.greenBtn,
                    borderColor: colors.greenBorder,
                  },
                  Platform.OS === "web"
                    ? ({
                        boxShadow: `0px 5px 0px ${colors.greenBorder}`,
                      } as any)
                    : {
                        shadowColor: colors.greenBorder,
                        shadowOffset: { width: 0, height: 5 },
                        shadowOpacity: 1,
                        shadowRadius: 0,
                      },
                ]}
              >
                <Text style={s.gaugeLabel}>Understanding</Text>
                <View style={s.gaugeWrap}>
                  <GaugeMeter value={scores.understanding} />
                </View>
                <Text style={s.gaugePct}>
                  {scores.understanding != null
                    ? `${Math.round(scores.understanding * 100)}%`
                    : "—"}
                </Text>
                <Text style={s.gaugeDesc}>
                  You understood most of what you saw!
                </Text>
              </View>

              {/* Observation */}
              <View
                style={[
                  s.gaugeCard,
                  {
                    backgroundColor: colors.pinkCard,
                    borderColor: colors.pinkBorder,
                  },
                  Platform.OS === "web"
                    ? ({
                        boxShadow: `0px 5px 0px ${colors.pinkBorder}`,
                      } as any)
                    : {
                        shadowColor: colors.pinkBorder,
                        shadowOffset: { width: 0, height: 5 },
                        shadowOpacity: 1,
                        shadowRadius: 0,
                      },
                ]}
              >
                <Text style={s.gaugeLabel}>Observation</Text>
                <View style={s.gaugeWrap}>
                  <GaugeMeter value={scores.observation} />
                </View>
                <Text style={s.gaugePct}>
                  {scores.observation != null
                    ? `${Math.round(scores.observation * 100)}%`
                    : "—"}
                </Text>
                <Text style={s.gaugeDesc}>
                  You noticed lots of great details!
                </Text>
              </View>

              {/* Engagement */}
              <View
                style={[
                  s.gaugeCard,
                  {
                    backgroundColor: colors.blueCard,
                    borderColor: colors.blueBorder,
                  },
                  Platform.OS === "web"
                    ? ({
                        boxShadow: `0px 5px 0px ${colors.blueBorder}`,
                      } as any)
                    : {
                        shadowColor: colors.blueBorder,
                        shadowOffset: { width: 0, height: 5 },
                        shadowOpacity: 1,
                        shadowRadius: 0,
                      },
                ]}
              >
                <Text style={s.gaugeLabel}>Engagement</Text>
                {engBuilding ? (
                  <View style={s.buildWrap}>
                    <Text style={s.buildText}>Building baseline…</Text>
                    <Text style={s.buildProg}>
                      {sessToward} / {baseMin} sessions
                    </Text>
                  </View>
                ) : (
                  <>
                    <View style={s.gaugeWrap}>
                      <GaugeMeter value={scores.engagement} />
                    </View>
                    <Text style={s.gaugePct}>
                      {Math.round((scores.engagement ?? 0) * 100)}%
                    </Text>
                    <Text style={s.gaugeDesc}>
                      You stayed focused and did an awesome job!
                    </Text>
                  </>
                )}
              </View>
            </View>

            {/* Right column: Stars + Continue */}
            <View style={s.rightCol}>
              <Text style={s.earnedLabel}>You earned</Text>
              <StarIcon size={80} />
              <Text style={s.starCount}>
                {starsEarned} / {total}
              </Text>
              <Text style={s.starWord}>stars</Text>

              <Pressable
                onPress={() => {
                  setBurstCount((n) => n + 1);
                  onNewSession();
                }}
                onPressIn={() => setContPressed(true)}
                onPressOut={() => setContPressed(false)}
                style={{ marginTop: 16, width: "100%" }}
              >
                <View
                  style={[
                    s.contBtn,
                    Platform.OS === "web"
                      ? { shadowOpacity: 0, elevation: 0 }
                      : {
                          shadowColor: "#d4a017",
                          shadowOffset: { width: 0, height: 6 },
                          shadowOpacity: 1,
                          shadowRadius: 0,
                          elevation: 4,
                        },
                    contWebStyle,
                  ]}
                >
                  <Text style={s.contText}>Continue</Text>
                  <Text style={s.contArrow}>›</Text>
                </View>
              </Pressable>
            </View>
          </View>

          {/* ── Encouragement Card ── */}
          <View
            style={[
              s.encCard,
              Platform.OS === "web"
                ? ({ boxShadow: "0px 5px 0px #B89AFF" } as any)
                : {
                    shadowColor: "#B89AFF",
                    shadowOffset: { width: 0, height: 5 },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                  },
            ]}
          >
            <Text style={{ fontSize: 24 }}>💡</Text>
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={s.encTitle}>{encTitle}</Text>
              <Text style={s.encMsg}>{encMsg}</Text>
            </View>
          </View>
        </View>

        {/* ════════════  QUESTION HISTORY (below outer card)  ════════════ */}
        <Text style={s.histTitle}>Question History</Text>
        {history.map((item: any, idx: number) => {
          const fb = item.followup || item.evaluation?.feedback || "";
          const lat = item.initiation_latency_ms;
          return (
            <View key={idx} style={s.histItem}>
              <View style={s.histHead}>
                <Text style={s.histQ}>{item.question}</Text>
                {lat != null && (
                  <Text style={s.histLat}>{(lat / 1000).toFixed(1)}s</Text>
                )}
              </View>
              <Text style={s.histDet}>
                <Text style={{ fontWeight: "700" }}>Expected: </Text>
                {item.expected_answer || "—"}
              </Text>
              <Text style={s.histDet}>
                <Text style={{ fontWeight: "700" }}>You said: </Text>
                {item.transcription || "—"}
              </Text>
              {fb ? <Text style={s.histFb}>"{fb}"</Text> : null}
            </View>
          );
        })}
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Styles
   ═══════════════════════════════════════════════════════════════ */

const s = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  scroll: { flex: 1 },
  scrollContent: {
    padding: 16,
    paddingTop: 24,
  },

  /* Outer pink card */
  outerCard: {
    backgroundColor: "#FFF0F5",
    borderWidth: 5,
    borderColor: colors.pinkBorder,
    borderRadius: 28,
    padding: 16,
    marginBottom: 24,
  },

  /* Great Job Banner */
  bannerCard: {
    backgroundColor: colors.greenBtn,
    borderWidth: 4,
    borderColor: colors.greenBorder,
    borderRadius: 20,
    padding: 16,
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 14,
  },
  bannerTitle: {
    fontFamily: fonts.heading,
    fontSize: 26,
    color: colors.darkBlue,
  },
  bannerSub: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.darkBlueText,
    marginTop: 2,
  },

  /* Grid: gauges left, stars/continue right */
  gridRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 14,
  },
  gaugesCol: {
    flex: 3,
    flexDirection: "row",
    gap: 10,
  },
  gaugeCard: {
    flex: 1,
    borderWidth: 4,
    borderRadius: 18,
    padding: 10,
    paddingTop: 12,
    alignItems: "center",
  },
  gaugeLabel: {
    fontFamily: fonts.heading,
    fontSize: 14,
    color: colors.darkBlue,
    textAlign: "center",
    marginBottom: 4,
  },
  gaugeWrap: {
    width: "100%",
    aspectRatio: 1.7,
  },
  gaugePct: {
    fontFamily: fonts.heading,
    fontSize: 26,
    color: colors.darkBlue,
    marginTop: -2,
  },
  gaugeDesc: {
    fontFamily: fonts.body,
    fontSize: 11,
    color: colors.darkBlueText,
    textAlign: "center",
    lineHeight: 15,
    marginTop: 2,
  },
  buildWrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
  },
  buildText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.darkBlue,
    textAlign: "center",
  },
  buildProg: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 4,
  },

  /* Right column */
  rightCol: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
  },
  earnedLabel: {
    fontFamily: fonts.heading,
    fontSize: 16,
    color: colors.darkBlue,
    marginBottom: 6,
  },
  starCount: {
    fontFamily: fonts.heading,
    fontSize: 30,
    color: colors.darkBlue,
    marginTop: 4,
  },
  starWord: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.darkBlueText,
  },
  contBtn: {
    backgroundColor: colors.yellowCard,
    borderWidth: 4,
    borderColor: "#d4a017",
    borderRadius: 999,
    paddingVertical: 12,
    paddingHorizontal: 20,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  contText: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.darkBlue,
  },
  contArrow: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.darkBlue,
  },

  /* Encouragement */
  encCard: {
    backgroundColor: "#F3EEFF",
    borderWidth: 4,
    borderColor: "#B89AFF",
    borderRadius: 18,
    padding: 14,
    flexDirection: "row",
    alignItems: "center",
  },
  encTitle: {
    fontFamily: fonts.heading,
    fontSize: 17,
    color: colors.darkBlue,
  },
  encMsg: {
    fontFamily: fonts.body,
    fontSize: 13,
    color: colors.darkBlueText,
    marginTop: 2,
    lineHeight: 18,
  },

  /* Question History */
  histTitle: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.darkBlue,
    marginBottom: 12,
  },
  histItem: {
    backgroundColor: colors.cardWhite,
    borderRadius: 16,
    borderWidth: 3,
    borderColor: "#e0e0e8",
    padding: 16,
    marginBottom: 10,
  },
  histHead: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 6,
  },
  histQ: {
    fontFamily: fonts.heading,
    fontSize: 16,
    color: colors.darkBlue,
    flex: 1,
  },
  histLat: {
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
  histDet: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlueText,
    lineHeight: 22,
  },
  histFb: {
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
