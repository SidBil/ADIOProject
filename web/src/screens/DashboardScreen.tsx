import React, { useEffect, useState } from "react";
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
import { supabase } from "../lib/supabase";
import { imageUrl } from "../api";
import ShapePattern from "../components/ShapePattern";

interface SessionRow {
  id: string;
  session_id: string;
  image_id: string | null;
  questions_answered: number;
  total_questions: number;
  created_at: string;
}

interface Props {
  onBack: () => void;
  onStartSession: () => Promise<void>;
}

export default function DashboardScreen({ onBack, onStartSession }: Props) {
  const [rows, setRows] = useState<SessionRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    supabase
      .from("sessions")
      .select("*")
      .order("created_at", { ascending: false })
      .then(({ data, error: fetchErr }) => {
        setLoading(false);
        if (fetchErr) {
          setError(fetchErr.message);
        } else {
          setRows((data as SessionRow[]) || []);
        }
      });
  }, []);

  function formatDate(iso: string) {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  const handleStart = async () => {
    setStarting(true);
    try {
      await onStartSession();
    } catch {
      setStarting(false);
    }
  };

  return (
    <View style={styles.container}>
      <ShapePattern />
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Past Sessions</Text>
        <View style={styles.backBtn} />
      </View>

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.darkBlue} />
        </View>
      ) : error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : rows.length === 0 ? (
        <View style={styles.center}>
          <Text style={styles.emptyTitle}>No Sessions Yet</Text>
          <Text style={styles.emptySubtitle}>
            Complete a session to see your history here.
          </Text>
        </View>
      ) : (
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {rows.map((row) => (
            <View key={row.id} style={styles.card}>
              {row.image_id && (
                <Image
                  source={{
                    uri: imageUrl(`/images/${row.image_id}`),
                  }}
                  style={styles.thumb}
                  resizeMode="cover"
                />
              )}
              <View style={styles.cardBody}>
                <Text style={styles.cardDate}>{formatDate(row.created_at)}</Text>
                <Text style={styles.cardScore}>
                  {row.questions_answered} / {row.total_questions} questions
                </Text>
              </View>
            </View>
          ))}
        </ScrollView>
      )}

      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.newBtn}
          onPress={handleStart}
          disabled={starting}
          activeOpacity={0.8}
        >
          {starting ? (
            <ActivityIndicator color={colors.white} />
          ) : (
            <Text style={styles.newBtnText}>Start New Session</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 50,
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  backBtn: {
    width: 60,
  },
  backText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 16,
    color: colors.blueBorder,
  },
  headerTitle: {
    fontFamily: fonts.heading,
    fontSize: 24,
    color: colors.darkBlue,
    textAlign: "center",
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 40,
  },
  emptyTitle: {
    fontFamily: fonts.heading,
    fontSize: 22,
    color: colors.darkBlue,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: colors.textMuted,
    textAlign: "center",
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 16,
    color: "#cc0000",
    textAlign: "center",
  },
  scroll: { flex: 1 },
  scrollContent: {
    padding: 20,
    paddingBottom: 100,
  },
  card: {
    flexDirection: "row",
    backgroundColor: colors.cardWhite,
    borderRadius: 18,
    overflow: "hidden",
    marginBottom: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 6,
    elevation: 2,
  },
  thumb: {
    width: 90,
    height: 72,
  },
  cardBody: {
    flex: 1,
    padding: 14,
    justifyContent: "center",
  },
  cardDate: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.darkBlueText,
    marginBottom: 4,
  },
  cardScore: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.textMuted,
  },
  footer: {
    padding: 20,
    paddingBottom: 36,
    alignItems: "center",
  },
  newBtn: {
    backgroundColor: colors.darkBlueBtnBg,
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 48,
  },
  newBtnText: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.white,
  },
});
