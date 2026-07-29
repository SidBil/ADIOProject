import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Platform,
} from "react-native";
import { colors, fonts } from "../theme";
import Spinner from "../components/Spinner";
import { getSessionDetail, imageUrl } from "../api";

interface Props {
  sessionId: string;
  onBack: () => void;
}

interface TurnDetail {
  question: string;
  expected_answer: string;
  transcription: string;
  accuracy: number | null;
  feedback: string;
  identified_elements: string[];
  missed_elements: string[];
  answer_duration_ms: number | null;
}

interface Entry {
  structure_word: string;
  original: TurnDetail | null;
  followup: TurnDetail | null;
}

interface Detail {
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
  entries: Entry[];
}

/* Accuracy (0-5) -> a soft clay card color + a bolder accent. Only accuracy is
   surfaced here; the evaluation's detail/clarity/relevance/overall are not. */
function accuracyPalette(a: number | null): { bg: string; accent: string } {
  if (a == null) return { bg: "#F1F1F6", accent: colors.textMuted };
  if (a >= 4) return { bg: "#F3F8EA", accent: "#6FB400" };
  if (a >= 3) return { bg: "#FFFAEA", accent: "#F5B400" };
  return { bg: "#FDF1F5", accent: "#EB008C" };
}

function fmtDateTime(iso: string | null): string {
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

function fmtDuration(ms: number | null): string {
  if (ms == null) return "—";
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const softClay =
  "10px 10px 26px rgba(45,30,10,0.10), inset -6px -6px 14px rgba(255,255,255,0.4)";

export default function SessionDetailScreen({ sessionId, onBack }: Props) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    getSessionDetail(sessionId)
      .then(setDetail)
      .catch((e) => setError(e.message));
  }, [sessionId]);

  const toggle = (sw: string) =>
    setExpanded((prev) => ({ ...prev, [sw]: !prev[sw] }));

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Session Detail</Text>
        <View style={styles.backBtn} />
      </View>

      {error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>Could not load session: {error}</Text>
        </View>
      ) : !detail ? (
        <View style={styles.center}>
          <Spinner size="large" />
        </View>
      ) : (
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* ── Session-level metadata ── */}
          <View style={[styles.metaCard, { boxShadow: softClay } as any]}>
            {detail.image_filename && (
              <Image
                source={{ uri: imageUrl(`/images/${detail.image_filename}`) }}
                style={styles.metaThumb}
                resizeMode="cover"
              />
            )}
            <View style={styles.metaBody}>
              <Text style={styles.metaDate}>{fmtDateTime(detail.started_at)}</Text>
              <View style={styles.metaRow}>
                <MetaChip label="Duration" value={fmtDuration(detail.duration_ms)} />
                <MetaChip
                  label="Status"
                  value={detail.completed ? "Completed" : "Abandoned"}
                />
              </View>
              <View style={styles.metaRow}>
                <MetaChip
                  label="Answered"
                  value={`${detail.questions_answered}${
                    detail.total_questions ? ` / ${detail.total_questions}` : ""
                  }`}
                />
                <MetaChip label="Follow-ups" value={String(detail.followup_count)} />
              </View>
            </View>
          </View>

          {/* ── Per-structure-word breakdown (all 10, always) ── */}
          {detail.entries.map((entry) => {
            const pal = accuracyPalette(entry.original?.accuracy ?? null);
            const isOpen = !!expanded[entry.structure_word];
            const answered = !!entry.original;
            return (
              <View
                key={entry.structure_word}
                style={[styles.entryCard, { backgroundColor: pal.bg, boxShadow: softClay } as any]}
              >
                <TouchableOpacity
                  style={styles.entryHead}
                  onPress={() => toggle(entry.structure_word)}
                  activeOpacity={0.7}
                >
                  <Text style={styles.entryWord}>{entry.structure_word}</Text>
                  <View style={styles.entryHeadRight}>
                    {answered ? (
                      <View style={[styles.scoreBadge, { borderColor: pal.accent }]}>
                        <Text style={[styles.scoreBadgeText, { color: pal.accent }]}>
                          {entry.original!.accuracy != null
                            ? `${entry.original!.accuracy}/5`
                            : "—"}
                        </Text>
                      </View>
                    ) : (
                      <Text style={styles.notAnswered}>Not answered</Text>
                    )}
                    {answered && (
                      <Text style={styles.chevron}>{isOpen ? "▾" : "▸"}</Text>
                    )}
                  </View>
                </TouchableOpacity>

                {isOpen && entry.original && (
                  <View style={styles.entryBody}>
                    <TurnBlock turn={entry.original} accent={pal.accent} />
                    {entry.followup && (
                      <View style={styles.followupWrap}>
                        <Text style={styles.followupLabel}>Follow-up</Text>
                        <TurnBlock
                          turn={entry.followup}
                          accent={accuracyPalette(entry.followup.accuracy).accent}
                        />
                      </View>
                    )}
                  </View>
                )}
              </View>
            );
          })}
        </ScrollView>
      )}
    </View>
  );
}

function MetaChip({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.metaChip}>
      <Text style={styles.metaChipLabel}>{label}</Text>
      <Text style={styles.metaChipValue}>{value}</Text>
    </View>
  );
}

/* One transcript + reasoning block, reused for the original answer and any
   nested follow-up answer. */
function TurnBlock({ turn, accent }: { turn: TurnDetail; accent: string }) {
  return (
    <View>
      <Text style={styles.question}>{turn.question}</Text>

      <Text style={styles.fieldLabel}>Expected</Text>
      <Text style={styles.fieldValue}>{turn.expected_answer || "—"}</Text>

      <Text style={styles.fieldLabel}>Child said</Text>
      <Text style={styles.fieldValue}>
        {turn.transcription ? `"${turn.transcription}"` : "—"}
      </Text>

      {!!turn.feedback && (
        <>
          <Text style={styles.fieldLabel}>Evaluation</Text>
          <Text style={[styles.feedback, { borderLeftColor: accent }]}>{turn.feedback}</Text>
        </>
      )}

      {(turn.identified_elements.length > 0 || turn.missed_elements.length > 0) && (
        <View style={styles.chipsRow}>
          {turn.identified_elements.map((el, i) => (
            <View key={`i-${i}`} style={[styles.tag, styles.tagIdentified]}>
              <Text style={styles.tagText}>✓ {el}</Text>
            </View>
          ))}
          {turn.missed_elements.map((el, i) => (
            <View key={`m-${i}`} style={[styles.tag, styles.tagMissed]}>
              <Text style={styles.tagText}>✕ {el}</Text>
            </View>
          ))}
        </View>
      )}

      {turn.answer_duration_ms != null && (
        <Text style={styles.latency}>
          Answered in {(turn.answer_duration_ms / 1000).toFixed(1)}s
        </Text>
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
  errorText: { fontFamily: fonts.body, fontSize: 16, color: "#cc0000", textAlign: "center" },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 60 },

  /* Metadata */
  metaCard: {
    flexDirection: "row",
    backgroundColor: colors.cardWhite,
    borderRadius: 22,
    overflow: "hidden",
    marginBottom: 18,
  },
  metaThumb: { width: 104, height: "100%", minHeight: 128 },
  metaBody: { flex: 1, padding: 16, gap: 10 },
  metaDate: { fontFamily: fonts.heading, fontSize: 18, color: colors.darkBlue },
  metaRow: { flexDirection: "row", gap: 10 },
  metaChip: { flex: 1 },
  metaChipLabel: { fontFamily: fonts.body, fontSize: 12, color: colors.textMuted },
  metaChipValue: { fontFamily: fonts.bodySemiBold, fontSize: 16, color: colors.darkBlueText },

  /* Entry */
  entryCard: { borderRadius: 20, marginBottom: 12, overflow: "hidden" },
  entryHead: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 16,
    paddingHorizontal: 18,
  },
  entryWord: {
    fontFamily: fonts.heading,
    fontSize: 20,
    color: colors.darkBlue,
    textTransform: "capitalize",
  },
  entryHeadRight: { flexDirection: "row", alignItems: "center", gap: 12 },
  scoreBadge: {
    borderWidth: 2,
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 4,
    backgroundColor: "rgba(255,255,255,0.7)",
  },
  scoreBadgeText: { fontFamily: fonts.heading, fontSize: 16 },
  notAnswered: { fontFamily: fonts.body, fontSize: 14, color: colors.textMuted, fontStyle: "italic" },
  chevron: { fontFamily: fonts.body, fontSize: 16, color: colors.textMuted, width: 14, textAlign: "center" },

  entryBody: {
    paddingHorizontal: 18,
    paddingBottom: 18,
    paddingTop: 2,
    backgroundColor: "rgba(255,255,255,0.55)",
  },
  question: { fontFamily: fonts.bodySemiBold, fontSize: 16, color: colors.darkBlue, marginBottom: 10 },
  fieldLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 12,
    color: colors.textMuted,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginTop: 10,
  },
  fieldValue: { fontFamily: fonts.body, fontSize: 15, color: colors.darkBlueText, lineHeight: 21, marginTop: 2 },
  feedback: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.darkBlueText,
    lineHeight: 22,
    marginTop: 4,
    paddingLeft: 12,
    borderLeftWidth: 3,
    fontStyle: "italic",
  },
  chipsRow: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 12 },
  tag: { borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  tagIdentified: { backgroundColor: "#E4F2CE" },
  tagMissed: { backgroundColor: "#FBE0EC" },
  tagText: { fontFamily: fonts.body, fontSize: 13, color: colors.darkBlueText },
  latency: { fontFamily: fonts.body, fontSize: 13, color: colors.textMuted, marginTop: 12 },

  /* Follow-up */
  followupWrap: {
    marginTop: 16,
    paddingTop: 14,
    borderTopWidth: 1,
    borderTopColor: "rgba(0,0,0,0.08)",
    paddingLeft: 12,
    borderLeftWidth: 3,
    borderLeftColor: colors.blueBorder,
  },
  followupLabel: {
    fontFamily: fonts.heading,
    fontSize: 13,
    color: colors.blueBorder,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
});
