import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Animated,
  Easing,
  Platform,
  Pressable,
} from "react-native";
import Svg, { Circle } from "react-native-svg";
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
   Score Ring — circular progress indicator
   ═══════════════════════════════════════════════════════════════ */

function ScoreRing({
  value,
  size = 90,
  strokeWidth = 8,
  color,
  bgColor = "#e0e0e8",
}: {
  value: number; // 0..1
  size?: number;
  strokeWidth?: number;
  color: string;
  bgColor?: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.max(0, Math.min(1, value));
  const strokeDashoffset = circumference * (1 - progress);

  return (
    <View style={{ width: size, height: size, alignItems: "center", justifyContent: "center" }}>
      <Svg width={size} height={size} style={{ transform: [{ rotate: "-90deg" }] }}>
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={bgColor}
          strokeWidth={strokeWidth}
          fill="none"
        />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
        />
      </Svg>
      <Text
        style={{
          position: "absolute",
          fontFamily: fonts.heading,
          fontSize: 22,
          color: colors.darkBlue,
        }}
      >
        {Math.round(progress * 100)}%
      </Text>
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   3D Score Card
   ═══════════════════════════════════════════════════════════════ */

function ScoreCard({
  title,
  value,
  ringColor,
  borderColor,
  bgColor,
  subtitle,
  building,
  buildingProgress,
}: {
  title: string;
  value: number | null;
  ringColor: string;
  borderColor: string;
  bgColor: string;
  subtitle?: string;
  building?: boolean;
  buildingProgress?: string;
}) {
  return (
    <View
      style={[
        scoreStyles.card,
        {
          backgroundColor: bgColor,
          borderColor: borderColor,
        },
        Platform.OS === "web"
          ? ({
              boxShadow: `0px 8px 0px ${borderColor}`,
            } as any)
          : {
              shadowColor: borderColor,
              shadowOffset: { width: 0, height: 8 },
              shadowOpacity: 1,
              shadowRadius: 0,
              elevation: 4,
            },
      ]}
    >
      <Text style={scoreStyles.cardTitle}>{title}</Text>
      {building ? (
        <View style={scoreStyles.buildingWrap}>
          <Text style={scoreStyles.buildingText}>Building baseline…</Text>
          {buildingProgress && (
            <Text style={scoreStyles.buildingProgress}>{buildingProgress}</Text>
          )}
        </View>
      ) : value != null ? (
        <ScoreRing value={value} color={ringColor} />
      ) : (
        <Text style={scoreStyles.naText}>—</Text>
      )}
      {subtitle && !building && (
        <Text style={scoreStyles.subtitle}>{subtitle}</Text>
      )}
    </View>
  );
}

const scoreStyles = StyleSheet.create({
  card: {
    borderWidth: 4,
    borderRadius: 22,
    padding: 18,
    alignItems: "center",
    flex: 1,
    minWidth: 140,
  },
  cardTitle: {
    fontFamily: fonts.heading,
    fontSize: 16,
    color: colors.darkBlue,
    marginBottom: 10,
    textAlign: "center",
  },
  buildingWrap: {
    alignItems: "center",
    justifyContent: "center",
    height: 90,
  },
  buildingText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.darkBlue,
    textAlign: "center",
  },
  buildingProgress: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 4,
    textAlign: "center",
  },
  naText: {
    fontFamily: fonts.heading,
    fontSize: 28,
    color: colors.textMuted,
    height: 90,
    lineHeight: 90,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 8,
    textAlign: "center",
  },
});

/* ═══════════════════════════════════════════════════════════════
   3D Button (consistent with LoginScreen)
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
      style={{ width: "100%", maxWidth: 360, marginTop: 20 }}
    >
      <View
        style={[
          {
            backgroundColor: topColor,
            borderWidth: 4,
            borderColor: bottomColor,
            borderRadius: 999,
            paddingVertical: 16,
            alignItems: "center",
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
        <Text
          style={{
            fontFamily: fonts.heading,
            fontSize: 20,
            color: textColor,
          }}
        >
          {title}
        </Text>
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
        <Text style={styles.errorText}>
          Could not load summary: {error}
        </Text>
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

  const engagementBuilding =
    scores.engagement === null || scores.engagement === undefined;
  const sessionsToward = scores.sessions_toward_baseline ?? 0;
  const baselineMin = scores.baseline_min_sessions ?? 3;

  return (
    <View style={styles.container}>
      <ShapePattern burst={burstCount} />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <Image
          source={require("../../assets/adiologo2.png")}
          style={styles.logo}
          resizeMode="contain"
        />
        <Text style={styles.title}>Session Complete!</Text>
        <Text style={styles.subtitle}>
          You answered {progress.answered || 0} question
          {(progress.answered || 0) !== 1 ? "s" : ""}. Great effort!
        </Text>

        {/* Score Cards */}
        <View style={styles.scoreRow}>
          <ScoreCard
            title="Observation"
            value={scores.observation}
            ringColor={colors.pinkBorder}
            borderColor={colors.pinkBorder}
            bgColor={colors.pinkCard}
            subtitle="Noticing details"
          />
          <ScoreCard
            title="Understanding"
            value={scores.understanding}
            ringColor={colors.blueBorder}
            borderColor={colors.blueBorder}
            bgColor={colors.blueCard}
            subtitle="Comprehension"
          />
          <ScoreCard
            title="Engagement"
            value={scores.engagement}
            ringColor={colors.greenBorder}
            borderColor={colors.greenBorder}
            bgColor={colors.greenBtn}
            building={engagementBuilding}
            buildingProgress={
              engagementBuilding
                ? `${sessionsToward} / ${baselineMin} sessions`
                : undefined
            }
            subtitle={!engagementBuilding ? "Response speed" : undefined}
          />
        </View>

        {/* Session Image */}
        {data.image_filename && (
          <Image
            source={{ uri: imageUrl(`/images/${data.image_filename}`) }}
            style={styles.summaryImage}
            resizeMode="cover"
          />
        )}

        {/* Q&A History */}
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

        {/* New Session Button */}
        <Button3D
          title="Start a New Session"
          onPress={() => {
            setBurstCount((n) => n + 1);
            onNewSession();
          }}
          topColor={colors.greenBtn}
          bottomColor={colors.greenBorder}
          textColor={colors.darkBlue}
        />

        <View style={{ height: 40 }} />
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
    paddingVertical: 30,
    paddingHorizontal: 8,
  },
  logo: {
    height: 100,
    width: 200,
    marginTop: -15,
    marginBottom: -10,
  },
  title: {
    fontFamily: fonts.heading,
    fontSize: 36,
    color: colors.darkBlueText,
    marginBottom: 6,
    textAlign: "center",
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.textMuted,
    marginBottom: 24,
    textAlign: "center",
  },
  scoreRow: {
    flexDirection: "row",
    gap: 12,
    width: "100%",
    maxWidth: 520,
    marginBottom: 24,
  },
  summaryImage: {
    width: "100%",
    maxWidth: 400,
    height: 220,
    borderRadius: 20,
    marginBottom: 24,
  },
  historyTitle: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.darkBlue,
    marginBottom: 12,
    alignSelf: "flex-start",
    maxWidth: 520,
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
    maxWidth: 520,
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
