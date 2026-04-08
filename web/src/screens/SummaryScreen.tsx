import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { colors, fonts } from "../theme";
import { getSummary, imageUrl } from "../api";
import { supabase } from "../lib/supabase";

interface Props {
  sessionId: string;
  imageId?: string;
  userId: string;
  onNewSession: () => void;
}

export default function SummaryScreen({
  sessionId,
  imageId,
  userId,
  onNewSession,
}: Props) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const saved = useRef(false);

  useEffect(() => {
    getSummary(sessionId)
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
          }));

          supabase
            .from("sessions")
            .insert({
              user_id: userId,
              session_id: sessionId,
              image_id: imageId || d.image_id || null,
              questions_answered: d.progress?.answered ?? 0,
              total_questions: d.progress?.total ?? 0,
              qa_history: history,
            })
            .then(({ error: insertErr }) => {
              if (insertErr) console.warn("Failed to save session:", insertErr.message);
            });
        }
      })
      .catch((e) => setError(e.message));
  }, [sessionId]);

  if (error) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Could not load summary: {error}</Text>
        <TouchableOpacity style={styles.btn} onPress={onNewSession}>
          <Text style={styles.btnText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!data) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color={colors.darkBlue} />
      </View>
    );
  }

  const progress = data.progress || {};
  const history: any[] = data.qa_history || [];

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.card}>
          <Image
            source={require("../../assets/adio_logo2.png")}
            style={styles.logo}
            resizeMode="contain"
          />
          <Text style={styles.title}>Well Done!</Text>
          <Text style={styles.subtitle}>
            You answered {progress.answered || 0} question
            {(progress.answered || 0) !== 1 ? "s" : ""}. Great effort!
          </Text>

          {data.image_filename && (
            <Image
              source={{ uri: imageUrl(`/images/${data.image_filename}`) }}
              style={styles.summaryImage}
              resizeMode="cover"
            />
          )}

          {history.map((item: any, idx: number) => {
            const fb = item.followup || item.evaluation?.feedback || "";
            return (
              <View key={idx} style={styles.historyItem}>
                <Text style={styles.hQuestion}>{item.question}</Text>
                <Text style={styles.hDetail}>
                  <Text style={styles.bold}>Expected:</Text>{" "}
                  {item.expected_answer || "—"}
                </Text>
                <Text style={styles.hDetail}>
                  <Text style={styles.bold}>You said:</Text>{" "}
                  {item.transcription || "—"}
                </Text>
                {fb ? <Text style={styles.hFeedback}>"{fb}"</Text> : null}
              </View>
            );
          })}

          <TouchableOpacity
            style={styles.btn}
            onPress={onNewSession}
            activeOpacity={0.8}
          >
            <Text style={styles.btnText}>Start a New Session</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
  },
  scroll: { flex: 1, width: "100%" },
  scrollContent: {
    alignItems: "center",
    paddingVertical: 40,
  },
  card: {
    backgroundColor: colors.cardWhite,
    borderRadius: 28,
    padding: 32,
    width: "100%",
    maxWidth: 520,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 10,
    elevation: 3,
  },
  logo: { height: 56, width: 140, marginBottom: 10 },
  title: {
    fontFamily: fonts.heading,
    fontSize: 36,
    color: colors.darkBlueText,
    marginBottom: 6,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.textMuted,
    marginBottom: 20,
    textAlign: "center",
  },
  summaryImage: {
    width: "100%",
    maxWidth: 260,
    height: 180,
    borderRadius: 14,
    marginBottom: 24,
  },
  historyItem: {
    backgroundColor: colors.cardWhite,
    borderRadius: 14,
    padding: 14,
    marginBottom: 10,
    width: "100%",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.04,
    shadowRadius: 4,
    elevation: 1,
  },
  hQuestion: {
    fontFamily: fonts.heading,
    fontSize: 17,
    color: colors.darkBlue,
    marginBottom: 4,
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
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: 1,
    borderTopColor: "#e8e8f0",
  },
  btn: {
    backgroundColor: colors.darkBlueBtnBg,
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 40,
    marginTop: 20,
  },
  btnText: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.white,
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: "#cc0000",
    textAlign: "center",
    marginBottom: 20,
  },
});
