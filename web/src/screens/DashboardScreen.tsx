import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Platform,
  useWindowDimensions,
} from "react-native";
import { colors, fonts } from "../theme";
import Spinner from "../components/Spinner";
import { getDashboardOverview, imageUrl } from "../api";
import AccuracyChart from "../components/AccuracyChart";
import SessionDetailScreen from "./SessionDetailScreen";

interface SessionRow {
  session_id: string;
  image_id: string | null;
  image_filename: string | null;
  started_at: string | null;
  ended_at: string | null;
  completed: boolean;
  questions_answered: number;
  total_questions: number | null;
  duration_ms: number | null;
  followup_count: number;
}

interface Overview {
  sessions: SessionRow[];
  sparklines: Record<string, (number | null)[]>;
  structure_words: string[];
  window: number;
}

interface Props {
  onBack: () => void;
}

function formatDate(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(ms: number | null): string {
  if (ms == null) return "—";
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const softClay =
  "10px 10px 26px rgba(45,30,10,0.10), inset -6px -6px 14px rgba(255,255,255,0.4)";

export default function DashboardScreen({ onBack }: Props) {
  const { width: winW } = useWindowDimensions();
  const [overview, setOverview] = useState<Overview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // Single-open accordion: which structure word's trend is expanded.
  const [expandedWord, setExpandedWord] = useState<string | null>(null);

  useEffect(() => {
    getDashboardOverview()
      .then((data) => {
        setOverview(data);
        // Open the first word by default so the section isn't an empty list.
        setExpandedWord(data.structure_words?.[0] ?? null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Most recent non-null accuracy for a word, shown as a badge on its row.
  const latestValue = (sw: string): number | null => {
    const arr = overview?.sparklines[sw] || [];
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i] != null) return arr[i];
    }
    return null;
  };

  // Drill-in: render the clinical detail for one session in place.
  if (selected) {
    return <SessionDetailScreen sessionId={selected} onBack={() => setSelected(null)} />;
  }

  const sessions = overview?.sessions || [];
  const words = overview?.structure_words || [];

  // Full-width focused chart sized to the viewport (accordion, one open at a time).
  const contentW = Math.min(winW, 720) - 40; // scrollContent horizontal padding
  const chartW = contentW - 40; // minus trendCard padding

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Past Sessions</Text>
        <View style={styles.backBtn} />
      </View>

      {loading ? (
        <View style={styles.center}>
          <Spinner size="large" />
        </View>
      ) : error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : sessions.length === 0 ? (
        <View style={styles.center}>
          <Text style={styles.emptyTitle}>No Sessions Yet</Text>
          <Text style={styles.emptySubtitle}>
            Complete a session to see trends and history here.
          </Text>
        </View>
      ) : (
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* ── Per-structure-word trends ── */}
          <Text style={styles.sectionTitle}>Structure-word trends</Text>
          <Text style={styles.sectionSub}>
            Accuracy (0–5) across the last {overview?.window ?? 10} sessions
          </Text>
          <View style={[styles.trendCard, { boxShadow: softClay } as any]}>
            {words.map((sw, idx) => {
              const open = expandedWord === sw;
              const lv = latestValue(sw);
              return (
                <View key={sw} style={[styles.accItem, idx > 0 && styles.accDivider]}>
                  <TouchableOpacity
                    style={styles.accHeader}
                    activeOpacity={0.6}
                    onPress={() => setExpandedWord(open ? null : sw)}
                  >
                    <Text style={styles.accWord}>{sw}</Text>
                    <View style={styles.accRight}>
                      {lv != null ? (
                        <View style={styles.accBadge}>
                          <Text style={styles.accBadgeText}>{lv}/5</Text>
                        </View>
                      ) : (
                        <Text style={styles.accEmpty}>—</Text>
                      )}
                      <Text style={[styles.accChevron, open && styles.accChevronOpen]}>›</Text>
                    </View>
                  </TouchableOpacity>
                  {open && (
                    <View style={styles.accBody}>
                      <AccuracyChart values={overview?.sparklines[sw] || []} width={chartW} />
                    </View>
                  )}
                </View>
              );
            })}
          </View>

          {/* ── Session history (tap to open clinical detail) ── */}
          <Text style={[styles.sectionTitle, { marginTop: 24 }]}>History</Text>
          {sessions.map((row) => (
            <TouchableOpacity
              key={row.session_id}
              style={[styles.card, { boxShadow: softClay } as any]}
              activeOpacity={0.7}
              onPress={() => setSelected(row.session_id)}
            >
              {row.image_filename && (
                <Image
                  source={{ uri: imageUrl(`/images/${row.image_filename}`) }}
                  style={styles.thumb}
                  resizeMode="cover"
                />
              )}
              <View style={styles.cardBody}>
                <Text style={styles.cardDate}>{formatDate(row.started_at)}</Text>
                <View style={styles.cardMetaRow}>
                  <View
                    style={[
                      styles.statusPill,
                      row.completed ? styles.statusDone : styles.statusAbandoned,
                    ]}
                  >
                    <Text
                      style={[
                        styles.statusText,
                        { color: row.completed ? "#5C8A00" : "#B36A00" },
                      ]}
                    >
                      {row.completed ? "Completed" : "Abandoned"}
                    </Text>
                  </View>
                  <Text style={styles.cardMeta}>{formatDuration(row.duration_ms)}</Text>
                </View>
              </View>
              <Text style={styles.cardChevron}>›</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}

    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: Platform.OS === "ios" ? 58 : 50,
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  backBtn: { width: 60 },
  backText: { fontFamily: fonts.bodySemiBold, fontSize: 16, color: colors.blueBorder },
  headerTitle: {
    fontFamily: fonts.heading,
    fontSize: 24,
    color: colors.darkBlue,
    textAlign: "center",
  },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 40 },
  emptyTitle: { fontFamily: fonts.heading, fontSize: 22, color: colors.darkBlue, marginBottom: 8 },
  emptySubtitle: { fontFamily: fonts.body, fontSize: 16, color: colors.textMuted, textAlign: "center" },
  errorText: { fontFamily: fonts.body, fontSize: 16, color: "#cc0000", textAlign: "center" },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 100, maxWidth: 720, width: "100%", alignSelf: "center" },

  sectionTitle: { fontFamily: fonts.heading, fontSize: 20, color: colors.darkBlue, marginBottom: 2 },
  sectionSub: { fontFamily: fonts.body, fontSize: 14, color: colors.textMuted, marginBottom: 12 },

  trendCard: { backgroundColor: colors.cardWhite, borderRadius: 22, paddingHorizontal: 20, paddingVertical: 6 },
  accItem: {},
  accDivider: { borderTopWidth: 1, borderTopColor: "#F0EEF4" },
  accHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 16,
  },
  accWord: {
    fontFamily: fonts.heading,
    fontSize: 18,
    color: colors.darkBlue,
    textTransform: "capitalize",
  },
  accRight: { flexDirection: "row", alignItems: "center", gap: 12 },
  accBadge: {
    backgroundColor: colors.blueCard,
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  accBadgeText: { fontFamily: fonts.bodySemiBold, fontSize: 13, color: colors.darkBlue },
  accEmpty: { fontFamily: fonts.body, fontSize: 15, color: colors.textMuted },
  accChevron: {
    fontFamily: fonts.heading,
    fontSize: 26,
    color: colors.textMuted,
    width: 18,
    textAlign: "center",
    transform: [{ rotate: "0deg" }],
  },
  accChevronOpen: { transform: [{ rotate: "90deg" }] },
  accBody: { paddingBottom: 18, paddingTop: 2 },

  card: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.cardWhite,
    borderRadius: 18,
    overflow: "hidden",
    marginBottom: 12,
  },
  thumb: { width: 90, height: 78 },
  cardBody: { flex: 1, padding: 14, justifyContent: "center", gap: 8 },
  cardDate: { fontFamily: fonts.bodySemiBold, fontSize: 15, color: colors.darkBlueText },
  cardMetaRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  statusPill: { borderRadius: 10, paddingHorizontal: 10, paddingVertical: 3 },
  statusDone: { backgroundColor: "#EAF4D3" },
  statusAbandoned: { backgroundColor: "#FBEBCE" },
  statusText: { fontFamily: fonts.bodySemiBold, fontSize: 12 },
  cardMeta: { fontFamily: fonts.body, fontSize: 14, color: colors.textMuted },
  cardChevron: { fontFamily: fonts.heading, fontSize: 28, color: colors.textMuted, paddingRight: 16 },

});
