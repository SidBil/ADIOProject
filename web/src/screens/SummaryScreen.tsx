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
import Svg, { Circle } from "react-native-svg";
import { colors, fonts } from "../theme";
import { getSummary } from "../api";

/* eslint-disable @typescript-eslint/no-require-imports */
const adioLogo = require("../../assets/adiologo2.png");
const brainIcon = require("../../assets/brain.png");
const eyeIcon = require("../../assets/eye.png");
const targetIcon = require("../../assets/target.png");
const starIcon = require("../../assets/star-04.png");

interface Props {
  sessionId: string;
  imageId?: string;
  userId: string;
  onNewSession: () => void;
}

/* ═══════════════════════════════════════════════════════════════
   Per-metric color palette (matches mockup)
   ═══════════════════════════════════════════════════════════════ */
type MetricKey = "understanding" | "observation" | "engagement";

const PALETTE: Record<MetricKey, {
  cardBg: string;
  iconBg: string;
  ringTrack: string;
  accent: string;
  scoreMuted: string;
}> = {
  understanding: {
    cardBg: "#FDF1F5",
    iconBg: "#F9D9E5",
    ringTrack: "#EFD7E0",
    accent: "#EB008C",
    scoreMuted: "#C75A8F",
  },
  observation: {
    cardBg: "#F3F8EA",
    iconBg: "#DDEBC0",
    ringTrack: "#DCE6CB",
    accent: "#6FB400",
    scoreMuted: "#7E9450",
  },
  engagement: {
    cardBg: "#FFFAEA",
    iconBg: "#FBEFC2",
    ringTrack: "#EFE6CB",
    accent: "#F5B400",
    scoreMuted: "#B8862E",
  },
};

/* ═══════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════ */
function gaugeMessage(kind: MetricKey, v: number | null): string {
  if (v == null) return "";
  const pct = Math.round(v * 100);
  if (kind === "understanding") {
    if (pct >= 80) return "Excellent comprehension!";
    if (pct >= 50) return "Good comprehension!";
    return "Keep practicing!";
  }
  if (kind === "observation") {
    if (pct >= 80) return "Good attention to detail!";
    if (pct >= 50) return "Nice eye for detail!";
    return "Look a little closer next time!";
  }
  if (pct >= 80) return "Great focus and effort!";
  if (pct >= 50) return "Good focus and effort!";
  return "Stay focused a bit longer!";
}

function encouragement(scores: any): string {
  const vals: number[] = [];
  if (scores.understanding != null) vals.push(scores.understanding);
  if (scores.observation != null) vals.push(scores.observation);
  if (scores.engagement != null) vals.push(scores.engagement);
  if (!vals.length) return "Awesome work! You're building great skills.";
  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  if (avg >= 0.8) return "Awesome work! You're building great skills.";
  if (avg >= 0.5) return "Nice job! Keep practicing to build your skills.";
  return "Great effort! Every session helps you improve.";
}

/* ═══════════════════════════════════════════════════════════════
   Main Summary Screen
   ═══════════════════════════════════════════════════════════════ */
export default function SummaryScreen({ sessionId, imageId, userId, onNewSession }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [contPressed, setContPressed] = useState(false);

  useEffect(() => {
    getSummary(sessionId, userId)
      .then(setData)
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
  const total = progress.total || 0;

  let starsEarned = 0;
  for (const item of history) {
    if (item.evaluation?.overall_score >= 3) starsEarned++;
  }

  const isMobile = winW < 700;

  // Responsive scaling
  const pad         = Math.max(14, winW * 0.018);
  const titleSz     = Math.max(60, Math.min(120, winH * 0.14));
  const subSz       = Math.max(18, Math.min(28, winH * 0.032));
  // Fixed logo size + position to match SessionScreen exactly.
  const logoH       = 150;
  const logoW       = 170;

  const cardGap     = Math.max(16, winW * 0.018);
  const cardsAvail  = winW - pad * 2;
  const cardW       = isMobile
    ? winW - pad * 2
    : Math.max(300, Math.min((cardsAvail - cardGap * 2) / 3, 620));
  const cardH       = cardW * 0.95;
  const cardPad     = Math.max(24, cardW * 0.07);

  // Top-row sizing: icon (left) and progress ring (right) share the row.
  const iconCircle  = Math.min(cardW * 0.42, 200);
  const iconImg     = iconCircle;
  const ringSize    = Math.min(cardW * 0.44, 220);

  const labelSz     = Math.min(cardW * 0.13, 60);
  const scoreSz     = Math.min(ringSize * 0.46, 64);
  const descSz      = Math.min(cardW * 0.065, 28);

  const starIconSz  = Math.max(60, Math.min(100, winH * 0.12));
  const starCountSz = Math.max(28, Math.min(48, winH * 0.055));
  const starWordSz  = Math.max(13, Math.min(18, winH * 0.022));

  const contFontSz  = Math.max(20, Math.min(34, winH * 0.04));
  const cardBorder  = Math.max(4, Math.min(6, winH * 0.007));
  const cardRadius  = Math.max(22, winH * 0.032);

  const contWebStyle = Platform.OS === "web" ? ({
    transition: "transform 150ms ease, box-shadow 150ms ease",
    boxShadow: contPressed ? `0px 0px 0px ${colors.yellowBorder}` : `0px ${cardBorder}px 0px ${colors.yellowBorder}`,
    transform: contPressed ? `translateY(${cardBorder}px)` : "translateY(0px)",
  } as any) : undefined;

  return (
    <View style={[s.root, { width: winW, minHeight: winH }]}>
      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={{ paddingBottom: pad * 2 }}
        showsVerticalScrollIndicator={false}
      >
        {/* ════════  SUMMARY PAGE — title at top, cards + continue vertically centered ════════ */}
        <View style={{
          width: winW,
          minHeight: winH,
          paddingHorizontal: pad,
          paddingTop: pad,
        }}>

          {/* ── Logo (matches SessionScreen exactly: 170×150 at left:20, vertically centered on a 52px topbar at top:20) ── */}
          <Image
            source={adioLogo}
            style={{
              width: logoW,
              height: logoH,
              position: "absolute",
              top: -29,
              left: 20,
              zIndex: 10,
            }}
            resizeMode="contain"
          />

          {/* ── Title + Subtitle (centered) ── */}
          <View style={{ alignItems: "center", marginTop: pad * 0.4 }}>
            <Text style={[s.title, { fontSize: titleSz }]}>Session Summary</Text>
            <Text style={[s.subtitle, { fontSize: subSz, marginTop: pad * 0.3 }]}>
              {encouragement(scores)}
            </Text>
          </View>

          {/* ── Cards + Continue group, vertically centered in remaining space ── */}
          <View style={{ flex: 1, justifyContent: "center", minHeight: 0 }}>

          {/* ── 3 Score Cards ── */}
          <View style={{
            flexDirection: isMobile ? "column" : "row",
            gap: cardGap,
            justifyContent: "center",
          }}>
            <ScoreCard
              metric="understanding"
              label="Understanding"
              icon={brainIcon}
              value={scores.understanding ?? null}
              cardW={cardW}
              cardH={cardH}
              cardPad={cardPad}
              cardBorder={cardBorder}
              cardRadius={cardRadius}
              iconCircle={iconCircle}
              iconImg={iconImg}
              ringSize={ringSize}
              labelSz={labelSz}
              scoreSz={scoreSz}
              descSz={descSz}
              pad={pad}
            />
            <ScoreCard
              metric="observation"
              label="Observation"
              icon={eyeIcon}
              value={scores.observation ?? null}
              cardW={cardW}
              cardH={cardH}
              cardPad={cardPad}
              cardBorder={cardBorder}
              cardRadius={cardRadius}
              iconCircle={iconCircle}
              iconImg={iconImg}
              ringSize={ringSize}
              labelSz={labelSz}
              scoreSz={scoreSz}
              descSz={descSz}
              pad={pad}
            />
            <ScoreCard
              metric="engagement"
              label="Engagement"
              icon={targetIcon}
              value={scores.engagement ?? null}
              cardW={cardW}
              cardH={cardH}
              cardPad={cardPad}
              cardBorder={cardBorder}
              cardRadius={cardRadius}
              iconCircle={iconCircle}
              iconImg={iconImg}
              ringSize={ringSize}
              labelSz={labelSz}
              scoreSz={scoreSz}
              descSz={descSz}
              pad={pad}
            />
          </View>

          {/* ── Stars + Continue (bottom row) ── */}
          <View style={{
            flexDirection: isMobile ? "column" : "row",
            alignItems: "center",
            justifyContent: "center",
            gap: pad * 2.5,
            marginTop: pad * 1.2,
          }}>
            {/* Stars */}
            <View style={{ flexDirection: "row", alignItems: "center", gap: pad * 0.4 }}>
              <Image
                source={starIcon}
                style={{ width: starIconSz, height: starIconSz }}
                resizeMode="contain"
              />
              <View>
                <Text style={[s.starCount, { fontSize: starCountSz }]}>
                  {starsEarned} / {total}
                </Text>
                <Text style={[s.starWord, { fontSize: starWordSz }]}>Stars Earned</Text>
              </View>
            </View>

            {/* Continue */}
            <Pressable
              onPress={onNewSession}
              onPressIn={() => setContPressed(true)}
              onPressOut={() => setContPressed(false)}
            >
              <View style={[s.contBtn, {
                borderWidth: cardBorder,
                borderColor: colors.yellowBorder,
                borderRadius: cardRadius * 0.9,
                paddingVertical: pad * 0.9,
                paddingHorizontal: pad * 2.6,
              },
                Platform.OS === "web" ? contWebStyle : {
                  shadowColor: colors.yellowBorder,
                  shadowOffset: { width: 0, height: cardBorder },
                  shadowOpacity: 1,
                  shadowRadius: 0,
                  elevation: 4,
                },
              ]}>
                <Text style={[s.contText, { fontSize: contFontSz }]}>Continue</Text>
                <Text style={[s.contArrow, { fontSize: contFontSz }]}>→</Text>
              </View>
            </Pressable>
          </View>

          </View>{/* end cards+continue centered group */}
        </View>

        {/* ════════  QUESTION HISTORY ════════ */}
        <View style={{ paddingHorizontal: pad, marginTop: pad }}>
          <Text style={[s.histTitle, { fontSize: titleSz * 0.55, marginBottom: pad }]}>
            Question History
          </Text>
          {history.map((item: any, idx: number) => {
            const fb = item.followup || item.evaluation?.feedback || "";
            const lat = item.initiation_latency_ms;
            return (
              <View key={idx} style={[s.histItem, {
                borderWidth: cardBorder - 1,
                padding: pad,
                borderRadius: cardRadius * 0.7,
                marginBottom: pad * 0.7,
              }]}>
                <View style={s.histHead}>
                  <Text style={[s.histQ, { fontSize: descSz + 6 }]}>{item.question}</Text>
                  {lat != null && (
                    <Text style={[s.histLat, { fontSize: descSz + 2 }]}>
                      {(lat / 1000).toFixed(1)}s
                    </Text>
                  )}
                </View>
                <Text style={[s.histDet, { fontSize: descSz + 3 }]}>
                  <Text style={{ fontWeight: "700" }}>Expected: </Text>
                  {item.expected_answer || ""}
                </Text>
                <Text style={[s.histDet, { fontSize: descSz + 3 }]}>
                  <Text style={{ fontWeight: "700" }}>You said: </Text>
                  {item.transcription || ""}
                </Text>
                {fb ? <Text style={[s.histFb, { fontSize: descSz + 2 }]}>"{fb}"</Text> : null}
              </View>
            );
          })}
        </View>
      </ScrollView>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   ScoreCard
   ═══════════════════════════════════════════════════════════════ */
function ProgressRing({
  size, stroke, progress, trackColor, fillColor,
}: {
  size: number;
  stroke: number;
  progress: number; // 0..1
  trackColor: string;
  fillColor: string;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dashOffset = c * (1 - Math.max(0, Math.min(1, progress)));
  return (
    <Svg width={size} height={size}>
      <Circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        stroke={trackColor}
        strokeWidth={stroke}
        fill="none"
      />
      <Circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        stroke={fillColor}
        strokeWidth={stroke}
        fill="none"
        strokeLinecap="round"
        strokeDasharray={`${c} ${c}`}
        strokeDashoffset={dashOffset}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
    </Svg>
  );
}

function ScoreCard({
  metric, label, icon, value,
  cardW, cardH, cardPad, cardBorder, cardRadius,
  iconCircle, iconImg, ringSize,
  labelSz, scoreSz, descSz, pad,
}: {
  metric: MetricKey;
  label: string;
  icon: any;
  value: number | null;
  cardW: number;
  cardH: number;
  cardPad: number;
  cardBorder: number;
  cardRadius: number;
  iconCircle: number;
  iconImg: number;
  ringSize: number;
  labelSz: number;
  scoreSz: number;
  descSz: number;
  pad: number;
}) {
  const p = PALETTE[metric];
  const tenth = value != null ? Math.round(value * 10) : null;
  const progress = value != null ? Math.max(0, Math.min(1, value)) : 0;
  const ringStroke = Math.max(8, ringSize * 0.1);

  const cardWebShadow = Platform.OS === "web"
    ? ({ boxShadow: `0px ${cardBorder + 2}px 0px ${colors.darkBlueText}` } as any)
    : undefined;

  return (
    <View style={[{
      width: cardW,
      minHeight: cardH,
      backgroundColor: p.cardBg,
      borderWidth: cardBorder,
      borderColor: colors.darkBlueText,
      borderRadius: cardRadius,
      padding: cardPad,
      marginBottom: cardBorder + 2,
    },
      Platform.OS === "web"
        ? cardWebShadow
        : {
            shadowColor: colors.darkBlueText,
            shadowOffset: { width: 0, height: cardBorder + 2 },
            shadowOpacity: 1,
            shadowRadius: 0,
            elevation: 4,
          },
    ]}>
      {/* Top row: icon circle + progress ring with score */}
      <View style={{
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-around",
        marginBottom: pad * 0.6,
      }}>
        <Image
          source={icon}
          style={{ width: iconCircle, height: iconCircle }}
          resizeMode="contain"
        />

        <View style={{ width: ringSize, height: ringSize, alignItems: "center", justifyContent: "center" }}>
          <ProgressRing
            size={ringSize}
            stroke={ringStroke}
            progress={progress}
            trackColor={p.ringTrack}
            fillColor={p.accent}
          />
          <View style={{
            position: "absolute",
            width: ringSize,
            height: ringSize,
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "row",
          }}>
            <Text style={[s.cardScore, { fontSize: scoreSz, color: p.accent }]}>
              {tenth != null ? tenth : "—"}
            </Text>
            <Text style={[s.cardScore, { fontSize: scoreSz * 0.45, color: p.scoreMuted, marginLeft: 2, marginTop: scoreSz * 0.25 }]}>
              {" /10"}
            </Text>
          </View>
        </View>
      </View>

      {/* Label */}
      <Text style={[s.cardLabel, { fontSize: labelSz, marginTop: pad * 0.2, textAlign: "center" }]}>
        {label}
      </Text>

      {/* Description */}
      <Text style={[s.cardDesc, { fontSize: descSz, lineHeight: descSz * 1.35, marginTop: pad * 0.3 }]}>
        {gaugeMessage(metric, value)}
      </Text>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Styles
   ═══════════════════════════════════════════════════════════════ */
const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, backgroundColor: colors.bg, alignItems: "center", justifyContent: "center" },

  title: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
    textAlign: "center",
  },
  subtitle: {
    fontFamily: fonts.body,
    color: colors.darkBlueText,
    textAlign: "center",
  },

  cardLabel: {
    fontFamily: fonts.heading,
    color: colors.darkBlue,
    textAlign: "center",
  },
  cardScore: {
    fontFamily: fonts.heading,
  },
  cardDesc: {
    fontFamily: fonts.body,
    color: colors.darkBlueText,
    textAlign: "center",
  },

  starCount: { fontFamily: fonts.heading, color: colors.darkBlue },
  starWord: { fontFamily: fonts.body, color: colors.darkBlueText },

  contBtn: {
    backgroundColor: colors.yellowCard,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
  },
  contText: { fontFamily: fonts.heading, color: colors.darkBlue },
  contArrow: { fontFamily: fonts.heading, color: colors.darkBlue },

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
