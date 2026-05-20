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
import Svg, { Path } from "react-native-svg";
import { colors, fonts } from "../theme";
import { getSummary, imageUrl } from "../api";
import { supabase } from "../lib/supabase";
import ShapePattern from "../components/ShapePattern";

/* eslint-disable @typescript-eslint/no-require-imports */
const gaugeArcImg = require("../../assets/Untitled-6-01.png");
const gaugeNeedleImg = require("../../assets/spinner.png");

interface Props {
  sessionId: string;
  imageId?: string;
  userId: string;
  onNewSession: () => void;
}

/* ═══════════════════════════════════════════════════════════════
   Gauge Meter — PNG arc image + rotated PNG needle
   Arc image: 300×150 (2:1).  Needle image: 172×46.
   The needle's circle hub is at roughly (23, 23) from its top-left.
   ═══════════════════════════════════════════════════════════════ */

function GaugeMeter({ value }: { value: number | null }) {
  const v = value != null ? Math.max(0, Math.min(1, value)) : 0;
  // 0% → 180° (points left), 50% → 90° (points up), 100% → 0° (points right)
  // CSS rotate: positive = clockwise. Our needle PNG points right at 0°.
  // So at v=0: rotate 180°. At v=1: rotate 0°.
  const angleDeg = 180 - v * 180;

  // The needle image is 172×46. The circle hub center is at ~(23, 23).
  // We render the needle at a size proportional to the gauge.
  // Needle display width = 55% of gauge width gives a good visual.
  // Needle display height = (46/172) * needleWidth.
  // The hub center as fraction: x=23/172 ≈ 13.4%, y=23/46 = 50%.

  return (
    <View style={gS.wrap}>
      {/* Arc background */}
      <Image source={gaugeArcImg} style={gS.arc} resizeMode="contain" />

      {/* Needle — absolutely positioned so hub sits at arc center-bottom */}
      <View
        style={[
          gS.needleContainer,
          {
            transform: [{ rotate: `${angleDeg}deg` }],
          } as any,
          Platform.OS === "web"
            ? ({ transformOrigin: "13.4% 50%" } as any)
            : {},
        ]}
      >
        <Image
          source={gaugeNeedleImg}
          style={gS.needleImg}
          resizeMode="contain"
        />
      </View>
    </View>
  );
}

const NEEDLE_W_PCT = 55; // % of gauge width
const NEEDLE_ASPECT = 46 / 172; // height/width of the needle image
const HUB_X_FRAC = 23 / 172; // hub center x as fraction of needle width

const gS = StyleSheet.create({
  wrap: {
    width: "100%",
    aspectRatio: 2,
    position: "relative",
    alignItems: "center",
    justifyContent: "flex-end",
  },
  arc: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
  },
  needleContainer: {
    position: "absolute",
    // Position so the hub center aligns with arc center-bottom.
    // The hub is at 13.4% of needle width from left.
    // So we shift left by (50% - 13.4% * NEEDLE_W_PCT%).
    // needleContainer.left = 50% - hubOffsetPx. We approximate:
    bottom: -2,
    left: `${50 - HUB_X_FRAC * NEEDLE_W_PCT}%` as any,
    width: `${NEEDLE_W_PCT}%` as any,
    aspectRatio: 172 / 46,
  },
  needleImg: {
    width: "100%",
    height: "100%",
  },
});

/* ═══════════════════════════════════════════════════════════════
   Star SVG
   ═══════════════════════════════════════════════════════════════ */

function StarIcon({ size = 50 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path
        d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
        fill="#FED330"
        stroke="#D48C00"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
    </Svg>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Dynamic message helpers
   ═══════════════════════════════════════════════════════════════ */

function gaugeMessage(
  kind: "understanding" | "observation" | "engagement",
  v: number | null
): string {
  if (v == null) return "";
  const pct = Math.round(v * 100);
  if (kind === "understanding") {
    if (pct >= 80) return "You understood most of what you saw!";
    if (pct >= 50) return "You're getting there — keep describing!";
    return "Let's keep practicing together!";
  }
  if (kind === "observation") {
    if (pct >= 80) return "You noticed lots of great details!";
    if (pct >= 50) return "Good eye! Try spotting even more next time.";
    return "Look closely — there's so much to find!";
  }
  // engagement
  if (pct >= 80) return "You stayed focused and did an awesome job!";
  if (pct >= 50) return "Nice focus! Let's keep it going.";
  return "Try to stay focused a little longer next time!";
}

function bannerMessage(answered: number, total: number): string {
  if (answered >= total && total > 0)
    return "You explored the scene and answered all the questions.";
  if (answered > 0)
    return `You explored the scene and answered ${answered} question${answered !== 1 ? "s" : ""}.`;
  return "You explored the scene and answered 0 questions.";
}

function encouragement(avg: number | null): {
  title: string;
  message: string;
} {
  if (avg != null && avg >= 0.8)
    return {
      title: "Amazing work!",
      message:
        "You're really getting the hang of describing what you see!",
    };
  if (avg != null && avg >= 0.5)
    return {
      title: "Keep it up!",
      message:
        "The more you practice, the better you'll become at spotting details!",
    };
  return {
    title: "Great effort!",
    message: "Every session helps you get better. Keep practicing!",
  };
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
  const { width: winW } = useWindowDimensions();
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

  /* ── Loading / Error states ── */

  if (error) {
    return (
      <View style={s.center}>
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
      <View style={s.center}>
        <ShapePattern />
        <ActivityIndicator size="large" color={colors.darkBlue} />
      </View>
    );
  }

  /* ── Derived data ── */

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
  const enc = encouragement(avg);

  const contWebStyle =
    Platform.OS === "web"
      ? ({
          transition:
            "transform 150ms ease, box-shadow 150ms ease",
          boxShadow: contPressed
            ? "0px 0px 0px #D4A017"
            : "0px 5px 0px #D4A017",
          transform: contPressed ? "translateY(5px)" : "translateY(0px)",
        } as any)
      : undefined;

  const maxW = Math.min(winW - 32, 750);

  /* ── Render ── */

  return (
    <View style={s.root}>
      <ShapePattern burst={burstCount} />
      <ScrollView
        style={s.scroll}
        contentContainerStyle={s.scrollInner}
        showsVerticalScrollIndicator={false}
      >
        <View style={{ width: maxW, alignSelf: "center" }}>

          {/* ═══════  OUTER PINK CARD  ═══════ */}
          <View
            style={[
              s.outerCard,
              Platform.OS === "web"
                ? ({ boxShadow: "0px 8px 0px #FFA8BE" } as any)
                : {
                    shadowColor: "#FFA8BE",
                    shadowOffset: { width: 0, height: 8 },
                    shadowOpacity: 1,
                    shadowRadius: 0,
                  },
            ]}
          >
            {/* ── Great job banner ── */}
            <View
              style={[
                s.banner,
                Platform.OS === "web"
                  ? ({ boxShadow: "0px 5px 0px #B5CC26" } as any)
                  : {
                      shadowColor: "#B5CC26",
                      shadowOffset: { width: 0, height: 5 },
                      shadowOpacity: 1,
                      shadowRadius: 0,
                    },
              ]}
            >
              <StarIcon size={42} />
              <View style={{ flex: 1, marginLeft: 14 }}>
                <Text style={s.bannerTitle}>Great job!</Text>
                <Text style={s.bannerSub}>
                  {bannerMessage(answered, total)}
                </Text>
              </View>
            </View>

            {/* ── Scores row: 3 gauges + stars column ── */}
            <View style={s.scoresRow}>
              {/* Gauge cards (3 side by side) */}
              <View style={s.gaugesRow}>
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
                  <Text style={s.gaugeDesc}>
                    {gaugeMessage("understanding", scores.understanding)}
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
                  <Text style={s.gaugeDesc}>
                    {gaugeMessage("observation", scores.observation)}
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
                      <Text style={s.gaugeDesc}>
                        {gaugeMessage("engagement", scores.engagement)}
                      </Text>
                    </>
                  )}
                </View>
              </View>

              {/* Stars + Continue column */}
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
                  style={{ marginTop: 14, width: "100%" }}
                >
                  <View
                    style={[
                      s.contBtn,
                      Platform.OS === "web"
                        ? { shadowOpacity: 0, elevation: 0 }
                        : {
                            shadowColor: "#D4A017",
                            shadowOffset: { width: 0, height: 5 },
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

            {/* ── Encouragement strip ── */}
            <View
              style={[
                s.encCard,
                Platform.OS === "web"
                  ? ({ boxShadow: "0px 5px 0px #C5B2FF" } as any)
                  : {
                      shadowColor: "#C5B2FF",
                      shadowOffset: { width: 0, height: 5 },
                      shadowOpacity: 1,
                      shadowRadius: 0,
                    },
              ]}
            >
              <Text style={{ fontSize: 28 }}>💡</Text>
              <View style={{ flex: 1, marginLeft: 14 }}>
                <Text style={s.encTitle}>{enc.title}</Text>
                <Text style={s.encMsg}>{enc.message}</Text>
              </View>
            </View>
          </View>

          {/* ═══════  QUESTION HISTORY  ═══════ */}
          <Text style={s.histTitle}>Question History</Text>
          {history.map((item: any, idx: number) => {
            const fb = item.followup || item.evaluation?.feedback || "";
            const lat = item.initiation_latency_ms;
            return (
              <View key={idx} style={s.histItem}>
                <View style={s.histHead}>
                  <Text style={s.histQ}>{item.question}</Text>
                  {lat != null && (
                    <Text style={s.histLat}>
                      {(lat / 1000).toFixed(1)}s
                    </Text>
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
        </View>
      </ScrollView>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Styles  —  ALL text scaled up for readability
   ═══════════════════════════════════════════════════════════════ */

const s = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  center: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
  },
  scroll: { flex: 1 },
  scrollInner: {
    padding: 20,
    paddingTop: 28,
  },

  /* ── Outer pink card ── */
  outerCard: {
    backgroundColor: "#FFF5F7",
    borderWidth: 5,
    borderColor: "#FFA8BE",
    borderRadius: 28,
    padding: 18,
    marginBottom: 28,
  },

  /* ── Great Job banner ── */
  banner: {
    backgroundColor: colors.greenBtn,
    borderWidth: 4,
    borderColor: colors.greenBorder,
    borderRadius: 22,
    padding: 18,
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  bannerTitle: {
    fontFamily: fonts.heading,
    fontSize: 30,
    color: colors.darkBlue,
  },
  bannerSub: {
    fontFamily: fonts.body,
    fontSize: 18,
    color: colors.darkBlueText,
    marginTop: 3,
  },

  /* ── Scores row ── */
  scoresRow: {
    flexDirection: "row",
    gap: 14,
    marginBottom: 16,
  },
  gaugesRow: {
    flex: 3,
    flexDirection: "row",
    gap: 12,
  },
  gaugeCard: {
    flex: 1,
    borderWidth: 4,
    borderRadius: 20,
    padding: 12,
    paddingTop: 14,
    alignItems: "center",
  },
  gaugeLabel: {
    fontFamily: fonts.heading,
    fontSize: 17,
    color: colors.darkBlue,
    textAlign: "center",
    marginBottom: 6,
  },
  gaugeWrap: {
    width: "100%",
    aspectRatio: 2,
    marginBottom: 8,
  },
  gaugeDesc: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlueText,
    textAlign: "center",
    lineHeight: 19,
    marginTop: 2,
  },
  buildWrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 18,
  },
  buildText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 16,
    color: colors.darkBlue,
    textAlign: "center",
  },
  buildProg: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.textMuted,
    marginTop: 6,
  },

  /* ── Right column (stars + continue) ── */
  rightCol: {
    flex: 1.1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
  earnedLabel: {
    fontFamily: fonts.heading,
    fontSize: 19,
    color: colors.darkBlue,
    marginBottom: 6,
  },
  starCount: {
    fontFamily: fonts.heading,
    fontSize: 34,
    color: colors.darkBlue,
    marginTop: 6,
  },
  starWord: {
    fontFamily: fonts.body,
    fontSize: 18,
    color: colors.darkBlueText,
  },
  contBtn: {
    backgroundColor: colors.yellowCard,
    borderWidth: 4,
    borderColor: "#D4A017",
    borderRadius: 999,
    paddingVertical: 12,
    paddingHorizontal: 22,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  contText: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.darkBlue,
  },
  contArrow: {
    fontFamily: fonts.heading,
    fontSize: 24,
    color: colors.darkBlue,
  },

  /* ── Encouragement strip ── */
  encCard: {
    backgroundColor: "#F3EEFF",
    borderWidth: 4,
    borderColor: "#B89AFF",
    borderRadius: 20,
    padding: 16,
    flexDirection: "row",
    alignItems: "center",
  },
  encTitle: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.darkBlue,
  },
  encMsg: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.darkBlueText,
    marginTop: 3,
    lineHeight: 22,
  },

  /* ── Question History ── */
  histTitle: {
    fontFamily: fonts.heading,
    fontSize: 26,
    color: colors.darkBlue,
    marginBottom: 14,
  },
  histItem: {
    backgroundColor: colors.cardWhite,
    borderRadius: 18,
    borderWidth: 3,
    borderColor: "#e0e0e8",
    padding: 18,
    marginBottom: 12,
  },
  histHead: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 8,
  },
  histQ: {
    fontFamily: fonts.heading,
    fontSize: 19,
    color: colors.darkBlue,
    flex: 1,
  },
  histLat: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 15,
    color: colors.textMuted,
    backgroundColor: "#f0f0f6",
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 10,
    marginLeft: 10,
    overflow: "hidden",
  },
  histDet: {
    fontFamily: fonts.body,
    fontSize: 17,
    color: colors.darkBlueText,
    lineHeight: 26,
  },
  histFb: {
    fontFamily: fonts.body,
    fontStyle: "italic",
    fontSize: 16,
    color: colors.blueBorder,
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "#e8e8f0",
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 18,
    color: "#cc0000",
    textAlign: "center",
    marginBottom: 20,
  },
  retryBtn: {
    backgroundColor: colors.darkBlueBtnBg,
    borderRadius: 18,
    paddingVertical: 16,
    paddingHorizontal: 40,
  },
  retryBtnText: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.white,
  },
});
