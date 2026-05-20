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
import { getSummary } from "../api";
import { supabase } from "../lib/supabase";
import ShapePattern from "../components/ShapePattern";

/* eslint-disable @typescript-eslint/no-require-imports */
const gaugeArcImg    = require("../../assets/Untitled-6-01.png");
const gaugeNeedleImg = require("../../assets/spinner.png");
const adioLogo       = require("../../assets/adiologo.png");

interface Props {
  sessionId: string;
  imageId?: string;
  userId: string;
  onNewSession: () => void;
}

/* ═══════════════════════════════════════════════════════════════
   Gauge Meter
   ═══════════════════════════════════════════════════════════════ */

function GaugeMeter({ value }: { value: number | null }) {
  const v = value != null ? Math.max(0, Math.min(1, value)) : 0;
  const angleDeg = 180 - v * 180;
  return (
    <View style={gS.wrap}>
      <Image source={gaugeArcImg} style={gS.arc} resizeMode="contain" />
      <View
        style={[
          gS.needleContainer,
          { transform: [{ rotate: `${angleDeg}deg` }] } as any,
          Platform.OS === "web" ? ({ transformOrigin: "13.4% 50%" } as any) : {},
        ]}
      >
        <Image source={gaugeNeedleImg} style={gS.needleImg} resizeMode="contain" />
      </View>
    </View>
  );
}

const NEEDLE_W_PCT = 55;
const HUB_X_FRAC = 23 / 172;
const gS = StyleSheet.create({
  wrap: { width: "100%", aspectRatio: 2, position: "relative", alignItems: "center", justifyContent: "flex-end" },
  arc: { position: "absolute", top: 0, left: 0, width: "100%", height: "100%" },
  needleContainer: {
    position: "absolute", bottom: -2,
    left: `${50 - HUB_X_FRAC * NEEDLE_W_PCT}%` as any,
    width: `${NEEDLE_W_PCT}%` as any, aspectRatio: 172 / 46,
  },
  needleImg: { width: "100%", height: "100%" },
});

/* ═══════════════════════════════════════════════════════════════
   Star SVG
   ═══════════════════════════════════════════════════════════════ */

function StarIcon({ size = 50 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path
        d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
        fill={colors.yellowCard} stroke={colors.yellowBorder} strokeWidth={1.5} strokeLinejoin="round"
      />
    </Svg>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════ */

function gaugeMessage(kind: "understanding" | "observation" | "engagement", v: number | null): string {
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
  if (pct >= 80) return "You stayed focused and did an awesome job!";
  if (pct >= 50) return "Nice focus! Let's keep it going.";
  return "Try to stay focused a little longer next time!";
}

function bannerMessage(answered: number, total: number): string {
  if (answered >= total && total > 0) return "You explored the scene and answered all the questions.";
  if (answered > 0) return `You explored the scene and answered ${answered} question${answered !== 1 ? "s" : ""}.`;
  return "You explored the scene and answered 0 questions.";
}

function encouragement(avg: number | null): { title: string; message: string } {
  if (avg != null && avg >= 0.8) return { title: "Amazing work!", message: "You're really getting the hang of describing what you see!" };
  if (avg != null && avg >= 0.5) return { title: "Keep it up!", message: "The more you practice, the better you'll become at spotting details!" };
  return { title: "Great effort!", message: "Every session helps you get better. Keep practicing!" };
}

/* ═══════════════════════════════════════════════════════════════
   Main Summary Screen
   ═══════════════════════════════════════════════════════════════ */

export default function SummaryScreen({ sessionId, imageId, userId, onNewSession }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const saved = useRef(false);
  const [contPressed, setContPressed] = useState(false);

  useEffect(() => {
    getSummary(sessionId, userId)
      .then((d) => {
        setData(d);
        if (!saved.current) {
          saved.current = true;
          const history = (d.qa_history || []).map((item: any) => ({
            question: item.question, structure_word: item.structure_word,
            expected_answer: item.expected_answer, transcription: item.transcription,
            evaluation: item.evaluation, followup: item.followup,
            initiation_latency_ms: item.initiation_latency_ms,
          }));
          const sc = d.scores || {};
          supabase.from("sessions").insert({
            user_id: userId, session_id: sessionId,
            image_id: imageId || d.image_id || null,
            questions_answered: d.progress?.answered ?? 0,
            total_questions: d.progress?.total ?? 0,
            qa_history: history,
            avg_latency_ms: sc.avg_latency_ms ?? null,
            observation_score: sc.observation ?? null,
            understanding_score: sc.understanding ?? null,
            engagement_score: sc.engagement ?? null,
          }).then(({ error: err }) => { if (err) console.warn("Save failed:", err.message); });
        }
      })
      .catch((e) => setError(e.message));
  }, [sessionId]);

  if (error) {
    return (
      <View style={s.center}>
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

  const contWebStyle = Platform.OS === "web" ? ({
    transition: "transform 150ms ease, box-shadow 150ms ease",
    boxShadow: contPressed ? `0px 0px 0px ${colors.yellowBorder}` : `0px 6px 0px ${colors.yellowBorder}`,
    transform: contPressed ? "translateY(6px)" : "translateY(0px)",
  } as any) : undefined;

  // Generous responsive scaling — heavily biased toward LARGE text
  const pad           = Math.max(14, winW * 0.018);
  const titleBarH     = Math.max(36, winH * 0.05);
  const titleFontSz   = Math.max(32, Math.min(56, winH * 0.07));
  const logoH         = titleBarH;
  const logoW         = logoH * 2.4;

  const bannerTitleSz = Math.max(38, Math.min(64, winH * 0.08));
  const bannerSubSz   = Math.max(20, Math.min(30, winH * 0.038));

  const gaugeLabelSz  = Math.max(36, Math.min(56, winH * 0.07));
  const gaugePctSz    = Math.max(40, Math.min(72, winH * 0.09));
  const gaugeDescSz   = Math.max(15, Math.min(22, winH * 0.026));

  const starIconSz    = Math.max(80, Math.min(140, winH * 0.18));
  const earnedSz      = Math.max(36, Math.min(56, winH * 0.07));
  const starCountSz   = Math.max(34, Math.min(54, winH * 0.066));
  const starWordSz    = Math.max(20, Math.min(30, winH * 0.036));
  const contFontSz    = Math.max(24, Math.min(36, winH * 0.045));

  const cardGap       = Math.max(10, winW * 0.012);
  const cardBorder    = Math.max(4, Math.min(6, winH * 0.007));
  const cardRadius    = Math.max(20, winH * 0.03);

  return (
    <View style={[s.root, { width: winW, height: winH }]}>
      <ShapePattern />

      {/* ════════  TOP BAR: logo + plain title text ════════ */}
      <View style={[s.topBar, { paddingHorizontal: pad, paddingTop: pad, paddingBottom: pad * 0.3 }]}>
        <Image source={adioLogo} style={{ width: logoW, height: logoH }} resizeMode="contain" />
        <Text style={[s.titleText, { fontSize: titleFontSz, marginLeft: pad * 1.2 }]}>Session Summary</Text>
      </View>

      {/* ════════  MAIN CONTENT ════════ */}
      <View style={[s.main, { paddingHorizontal: pad, paddingTop: pad * 1.5, gap: pad }]}>

        {/* ── Great Job — plain left-aligned text, no card ── */}
        <View>
          <Text style={[s.bannerTitle, { fontSize: bannerTitleSz }]}>Great job!</Text>
          <Text style={[s.bannerSub, { fontSize: bannerSubSz, marginTop: 4 }]}>
            {bannerMessage(answered, total)}
          </Text>
        </View>

        {/* ── Scores row: 3 colored gauges + pink stars card (all flex: 1) ── */}
        <View style={[s.scoresRow, { gap: cardGap }]}>

          {/* Understanding — YELLOW */}
          <GaugeCard
            label="Understanding"
            bg={colors.yellowCard}
            border={colors.yellowBorder}
            scores={scores}
            scoreKey="understanding"
            engBuilding={false}
            sessToward={0}
            baseMin={0}
            cardBorder={cardBorder}
            cardRadius={cardRadius}
            labelSz={gaugeLabelSz}
            pctSz={gaugePctSz}
            descSz={gaugeDescSz}
            pad={pad}
          />

          {/* Observation — GREEN */}
          <GaugeCard
            label="Observation"
            bg={colors.greenBtn}
            border={colors.greenBorder}
            scores={scores}
            scoreKey="observation"
            engBuilding={false}
            sessToward={0}
            baseMin={0}
            cardBorder={cardBorder}
            cardRadius={cardRadius}
            labelSz={gaugeLabelSz}
            pctSz={gaugePctSz}
            descSz={gaugeDescSz}
            pad={pad}
          />

          {/* Engagement — BLUE */}
          <GaugeCard
            label="Engagement"
            bg={colors.blueCard}
            border={colors.blueBorder}
            scores={scores}
            scoreKey="engagement"
            engBuilding={engBuilding}
            sessToward={sessToward}
            baseMin={baseMin}
            cardBorder={cardBorder}
            cardRadius={cardRadius}
            labelSz={gaugeLabelSz}
            pctSz={gaugePctSz}
            descSz={gaugeDescSz}
            pad={pad}
          />

          {/* Stars + Continue — PINK */}
          <View style={[s.starsCard, {
            backgroundColor: colors.pinkCard,
            borderColor: colors.pinkBorder,
            borderWidth: cardBorder,
            borderRadius: cardRadius,
            paddingTop: pad * 0.9, paddingHorizontal: pad * 0.6, paddingBottom: pad * 0.6,
          },
            Platform.OS === "web"
              ? ({ boxShadow: `0px ${cardBorder}px 0px ${colors.pinkBorder}` } as any)
              : { shadowColor: colors.pinkBorder, shadowOffset: { width: 0, height: cardBorder }, shadowOpacity: 1, shadowRadius: 0 },
          ]}>
            <Text style={[s.earnedLabel, { fontSize: earnedSz, marginBottom: pad * 0.4 }]}>You earned</Text>
            <View style={{ flex: 1, alignItems: "center", justifyContent: "center", width: "100%" }}>
              <StarIcon size={starIconSz} />
              <Text style={[s.starCount, { fontSize: starCountSz, marginTop: pad * 0.4 }]}>{starsEarned} / {total}</Text>
              <Text style={[s.starWord, { fontSize: starWordSz }]}>stars</Text>
            </View>

            <Pressable
              onPress={onNewSession}
              onPressIn={() => setContPressed(true)}
              onPressOut={() => setContPressed(false)}
              style={{ marginTop: pad * 1.4, width: "100%" }}
            >
              <View style={[s.contBtn, {
                borderWidth: cardBorder,
                borderRadius: cardRadius * 0.6,
                paddingVertical: pad * 0.7,
                paddingHorizontal: pad,
              },
                Platform.OS === "web" ? { shadowOpacity: 0, elevation: 0 } : {
                  shadowColor: colors.yellowBorder, shadowOffset: { width: 0, height: 6 }, shadowOpacity: 1, shadowRadius: 0, elevation: 4,
                },
                contWebStyle,
              ]}>
                <Text style={[s.contText, { fontSize: contFontSz }]}>Continue</Text>
              </View>
            </Pressable>
          </View>
        </View>
      </View>

      {/* ════════  QUESTION HISTORY — scrollable, always below summary ════════ */}
      <ScrollView
        style={[s.histScroll, { paddingHorizontal: pad, marginTop: pad }]}
        contentContainerStyle={{ paddingBottom: pad * 2 }}
        showsVerticalScrollIndicator={false}
      >
        <Text style={[s.histTitle, { fontSize: bannerTitleSz * 0.7, marginBottom: pad }]}>Question History</Text>
        {history.map((item: any, idx: number) => {
          const fb = item.followup || item.evaluation?.feedback || "";
          const lat = item.initiation_latency_ms;
          return (
            <View key={idx} style={[s.histItem, { borderWidth: cardBorder - 1, padding: pad, borderRadius: cardRadius * 0.7, marginBottom: pad * 0.7 }]}>
              <View style={s.histHead}>
                <Text style={[s.histQ, { fontSize: gaugeDescSz + 6 }]}>{item.question}</Text>
                {lat != null && <Text style={[s.histLat, { fontSize: gaugeDescSz + 2 }]}>{(lat / 1000).toFixed(1)}s</Text>}
              </View>
              <Text style={[s.histDet, { fontSize: gaugeDescSz + 3 }]}>
                <Text style={{ fontWeight: "700" }}>Expected: </Text>{item.expected_answer || "—"}
              </Text>
              <Text style={[s.histDet, { fontSize: gaugeDescSz + 3 }]}>
                <Text style={{ fontWeight: "700" }}>You said: </Text>{item.transcription || "—"}
              </Text>
              {fb ? <Text style={[s.histFb, { fontSize: gaugeDescSz + 2 }]}>"{fb}"</Text> : null}
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
}

/* ── GaugeCard subcomponent ─────────────────────────────────── */

function GaugeCard(props: {
  label: string;
  bg: string;
  border: string;
  scores: any;
  scoreKey: "understanding" | "observation" | "engagement";
  engBuilding: boolean;
  sessToward: number;
  baseMin: number;
  cardBorder: number;
  cardRadius: number;
  labelSz: number;
  pctSz: number;
  descSz: number;
  pad: number;
}) {
  const { label, bg, border, scores, scoreKey, engBuilding, sessToward, baseMin,
          cardBorder, cardRadius, labelSz, pctSz, descSz, pad } = props;
  const value = scores[scoreKey] as number | null | undefined;

  return (
    <View style={[s.gaugeCard, {
      backgroundColor: bg, borderColor: border, borderWidth: cardBorder, borderRadius: cardRadius,
      paddingTop: pad * 0.9, paddingHorizontal: pad * 0.6, paddingBottom: pad * 0.6,
    },
      Platform.OS === "web"
        ? ({ boxShadow: `0px ${cardBorder}px 0px ${border}` } as any)
        : { shadowColor: border, shadowOffset: { width: 0, height: cardBorder }, shadowOpacity: 1, shadowRadius: 0 },
    ]}>
      <Text style={[s.gaugeLabel, { fontSize: labelSz, marginBottom: pad * 0.4 }]}>{label}</Text>

      {engBuilding && scoreKey === "engagement" ? (
        <View style={s.buildWrap}>
          <Text style={[s.buildText, { fontSize: labelSz * 0.7 }]}>Building baseline…</Text>
          <Text style={[s.buildProg, { fontSize: descSz, marginTop: 8 }]}>{sessToward} / {baseMin} sessions</Text>
        </View>
      ) : (
        <View style={{ flex: 1, width: "100%", alignItems: "center", justifyContent: "center" }}>
          <View style={s.gaugeWrap}>
            <GaugeMeter value={value ?? null} />
          </View>
          <Text style={[s.gaugePct, { fontSize: pctSz, marginTop: pad * 0.2 }]}>
            {value != null ? `${Math.round(value * 100)}%` : "—"}
          </Text>
          <Text style={[s.gaugeDesc, { fontSize: descSz, marginTop: pad * 0.3, paddingHorizontal: pad * 0.3 }]}>
            {gaugeMessage(scoreKey, value ?? null)}
          </Text>
        </View>
      )}
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Styles
   ═══════════════════════════════════════════════════════════════ */

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, backgroundColor: colors.bg, alignItems: "center", justifyContent: "center" },

  /* ── Top bar ── */
  topBar: {
    flexDirection: "row",
    alignItems: "center",
  },
  titleText: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
  },

  /* ── Main content (no outer card) ── */
  main: {
    // No flex:1 — let content size naturally so question history below isn't pushed off-screen
  },

  /* ── Great Job — plain text ── */
  bannerTitle: { fontFamily: fonts.heading, color: colors.darkBlue },
  bannerSub: { fontFamily: fonts.body, color: colors.darkBlueText },

  /* ── Scores row ── */
  scoresRow: {
    flexDirection: "row",
  },
  gaugeCard: {
    flex: 1,
    aspectRatio: 1,
    alignItems: "center",
    justifyContent: "flex-start",
  },
  gaugeLabel: { fontFamily: fonts.heading, color: colors.darkBlue, textAlign: "center" },
  gaugeWrap: { width: "92%", aspectRatio: 2 },
  gaugePct: { fontFamily: fonts.heading, color: colors.darkBlue, textAlign: "center", marginTop: 4 },
  gaugeDesc: { fontFamily: fonts.body, color: colors.darkBlueText, textAlign: "center", lineHeight: 22 },
  buildWrap: { flex: 1, alignItems: "center", justifyContent: "center" },
  buildText: { fontFamily: fonts.bodySemiBold, color: colors.darkBlue, textAlign: "center" },
  buildProg: { fontFamily: fonts.body, color: colors.textMuted },

  /* ── Stars card (pink) ── */
  starsCard: {
    flex: 1,
    aspectRatio: 1,
    alignItems: "center",
    justifyContent: "flex-start",
  },
  earnedLabel: { fontFamily: fonts.heading, color: colors.darkBlue, marginBottom: 8 },
  starCount: { fontFamily: fonts.heading, color: colors.darkBlue },
  starWord: { fontFamily: fonts.body, color: colors.darkBlueText },
  contBtn: {
    backgroundColor: colors.yellowCard,
    borderColor: colors.yellowBorder,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  },
  contText: { fontFamily: fonts.heading, color: colors.darkBlue },
  contArrow: { fontFamily: fonts.heading, color: colors.darkBlue },

  /* ── Encouragement (purple) ── */
  encCard: {
    backgroundColor: "#F3EEFF",
    borderColor: "#B89AFF",
    flexDirection: "row",
    alignItems: "center",
  },
  encBulb: {
    backgroundColor: "#B89AFF",
    alignItems: "center",
    justifyContent: "center",
  },
  encTitle: { fontFamily: fonts.heading, color: colors.darkBlue },
  encMsg: { fontFamily: fonts.body, color: colors.darkBlueText, lineHeight: 26 },

  /* ── Question History ── */
  histScroll: { flex: 1, marginTop: 8 },
  histTitle: { fontFamily: fonts.heading, color: colors.darkBlue },
  histItem: {
    backgroundColor: colors.cardWhite,
    borderColor: "#e0e0e8",
  },
  histHead: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 },
  histQ: { fontFamily: fonts.heading, color: colors.darkBlue, flex: 1 },
  histLat: {
    fontFamily: fonts.bodySemiBold, color: colors.textMuted,
    backgroundColor: "#f0f0f6", paddingHorizontal: 10, paddingVertical: 3,
    borderRadius: 10, marginLeft: 10, overflow: "hidden",
  },
  histDet: { fontFamily: fonts.body, color: colors.darkBlueText, lineHeight: 28 },
  histFb: {
    fontFamily: fonts.body, fontStyle: "italic", color: colors.blueBorder,
    marginTop: 10, paddingTop: 10, borderTopWidth: 1, borderTopColor: "#e8e8f0",
  },
  errorText: { fontFamily: fonts.body, fontSize: 20, color: "#cc0000", textAlign: "center", marginBottom: 20 },
  retryBtn: { backgroundColor: colors.darkBlueBtnBg, borderRadius: 18, paddingVertical: 16, paddingHorizontal: 40 },
  retryBtnText: { fontFamily: fonts.heading, fontSize: 22, color: colors.white },
});
