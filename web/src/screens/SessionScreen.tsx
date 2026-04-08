import React, { useRef, useState, useEffect } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Platform,
  Alert,
  ScrollView,
  useWindowDimensions,
} from "react-native";
import { Audio } from "expo-av";
import * as Speech from "expo-speech";
import { colors, fonts } from "../theme";
import { imageUrl, transcribeAudio, evaluate, endSession } from "../api";
import Stars from "../components/Stars";
import ShapePattern from "../components/ShapePattern";
import useVolumeMeter from "../hooks/useVolumeMeter";

function showAlert(title: string, msg: string) {
  if (Platform.OS === "web") {
    window.alert(`${title}: ${msg}`);
  } else {
    Alert.alert(title, msg);
  }
}

const HEADING_MAP: Record<number, string> = {
  5: "Excellent!",
  4: "Great Job!",
  3: "Good Effort!",
  2: "Nice Try!",
  1: "Keep Going!",
};

interface Question {
  id: string;
  text: string;
  structure_word: string;
  difficulty: number;
}

interface SessionData {
  session_id: string;
  image_url: string;
  question: Question | null;
  total_questions: number;
  progress: { answered: number; total: number; completed: boolean };
}

interface Props {
  session: SessionData;
  onEnd: () => void;
  onUpdateSession: (patch: Partial<SessionData>) => void;
}

type CardState = "question" | "feedback";

export default function SessionScreen({
  session,
  onEnd,
  onUpdateSession,
}: Props) {
  const { width, height } = useWindowDimensions();
  const isPortrait = height > width || width < 700;

  const [cardState, setCardState] = useState<CardState>("question");
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [feedbackData, setFeedbackData] = useState<any>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [yellowPressed, setYellowPressed] = useState(false);
  const [burstCount, setBurstCount] = useState(0);
  const [micCenter, setMicCenter] = useState<{ x: number; y: number } | undefined>();
  const { volume, startMetering, stopMetering } = useVolumeMeter();

  const progress = session.progress;
  const currentNum = Math.min(progress.answered + 1, progress.total);
  const pct = progress.total
    ? Math.round((progress.answered / progress.total) * 100)
    : 0;

  async function startRecording() {
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        showAlert("Permission needed", "Microphone access is required.");
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      if (Platform.OS === "web") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        startMetering(stream);
      }

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;

      if (Platform.OS !== "web") {
        startMetering(recording);
      }

      setIsRecording(true);
      setYellowPressed(true);
    } catch (err: any) {
      showAlert("Recording error", err.message);
    }
  }

  async function stopRecording() {
    if (!recordingRef.current) return;
    setIsRecording(false);
    setYellowPressed(false);
    stopMetering();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    try {
      await recordingRef.current.stopAndUnloadAsync();
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      if (!uri) return;

      setIsProcessing(true);

      const mimeType =
        Platform.OS === "web"
          ? "audio/webm"
          : Platform.OS === "ios"
          ? "audio/m4a"
          : "audio/mp4";
      const tData = await transcribeAudio(session.session_id, uri, mimeType);
      const transcription = tData.transcription || tData.text;

      const eData = await evaluate(session.session_id, transcription);

      setFeedbackData(eData);
      onUpdateSession({
        question: eData.next_question,
        progress: eData.progress,
      });
      setCardState("feedback");

      const comment = eData.followup || eData.evaluation?.feedback || "";
      if (comment) {
        Speech.speak(comment, { language: "en-US", rate: 0.9 });
      }
    } catch (err: any) {
      showAlert("Error", err.message);
    } finally {
      setIsProcessing(false);
    }
  }

  function toggleRecording() {
    if (isProcessing) return;
    isRecording ? stopRecording() : startRecording();
  }

  function handleNext() {
    setBurstCount((n) => n + 1);
    if (session.progress.completed || !session.question) {
      onEnd();
    } else {
      setCardState("question");
    }
  }

  async function handleClose() {
    try {
      await endSession(session.session_id);
    } catch {}
    onEnd();
  }

  const topPad = Platform.OS === "ios" ? 58 : 20;
  const bodyHeight = height - topPad - 20 - 52 - 20;
  const imageWidth = isPortrait ? undefined : bodyHeight * (16 / 10);

  return (
    <View style={styles.container}>
      <ShapePattern volume={volume} burst={burstCount} cardCenter={micCenter} />
      {/* ─── Top Bar ─── */}
      <View style={styles.topBar}>
        <Image
          source={require("../../assets/adio_logo2.png")}
          style={styles.logo}
          resizeMode="contain"
        />
        <View style={styles.logoSpacer} />
        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${pct}%` }]} />
          <View style={styles.progressInner}>
            <View style={{ flex: 1 }} />
            <Text style={styles.progressText}>
              {currentNum}/{progress.total}
            </Text>
          </View>
        </View>
        <TouchableOpacity onPress={handleClose} hitSlop={12}>
          <Text style={styles.closeBtn}>×</Text>
        </TouchableOpacity>
      </View>

      {/* ─── Body ─── */}
      {isPortrait ? (
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={styles.bodyVertical}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.imageWrapPortrait}>
            <Image
              source={{ uri: imageUrl(session.image_url) }}
              style={styles.image}
              resizeMode="cover"
            />
          </View>
          <View style={styles.sidebarWrapPortrait}>
            <View style={styles.sidebarInner}>
              {cardState === "question" ? (
                <QuestionCard
                  question={session.question}
                  isRecording={isRecording}
                  isProcessing={isProcessing}
                  onToggle={toggleRecording}
                  pressed={yellowPressed}
                  onMicLayout={setMicCenter}
                />
              ) : (
                <FeedbackCard data={feedbackData} onNext={handleNext} />
              )}
            </View>
          </View>
        </ScrollView>
      ) : (
        <View style={styles.bodyLandscape}>
          <View style={[styles.imageWrapLandscape, { width: imageWidth }]}>
            <Image
              source={{ uri: imageUrl(session.image_url) }}
              style={styles.imageLandscape}
              resizeMode="cover"
            />
          </View>
          <View style={styles.sidebarWrapLandscape}>
            <View style={styles.sidebarInner}>
              {cardState === "question" ? (
                <QuestionCard
                  question={session.question}
                  isRecording={isRecording}
                  isProcessing={isProcessing}
                  onToggle={toggleRecording}
                  pressed={yellowPressed}
                  onMicLayout={setMicCenter}
                />
              ) : (
                <FeedbackCard data={feedbackData} onNext={handleNext} />
              )}
            </View>
          </View>
        </View>
      )}
    </View>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Question Card
   ═══════════════════════════════════════════════════════════════ */


function QuestionCard({
  question,
  isRecording,
  isProcessing,
  onToggle,
  pressed,
  onMicLayout,
}: {
  question: Question | null;
  isRecording: boolean;
  isProcessing: boolean;
  onToggle: () => void;
  pressed: boolean;
  onMicLayout?: (center: { x: number; y: number }) => void;
}) {
  const micRef = useRef<View>(null);

  if (!question) return null;

  const label = isProcessing
    ? "Processing…"
    : isRecording
    ? "Listening… tap to stop"
    : "Tap to Answer";

  const handleMicLayout = () => {
    if (micRef.current && onMicLayout) {
      (micRef.current as any).measureInWindow?.(
        (x: number, y: number, w: number, h: number) => {
          if (x != null) onMicLayout({ x: x + w / 2, y: y + h / 2 });
        }
      );
    }
  };

  return (
    <View style={qStyles.pinkCard}>
      <Text style={qStyles.questionText}>{question.text}</Text>
      <View
        style={[
          qStyles.yellowCard,
          pressed && qStyles.yellowCardPressed,
        ]}
      >
        <TouchableOpacity
          onPress={onToggle}
          disabled={isProcessing}
          activeOpacity={0.7}
          style={qStyles.micBtn}
        >
          <View
            ref={micRef}
            style={qStyles.micWrap}
            onLayout={handleMicLayout}
          >
            <Image
              source={require("../../assets/micV3.png")}
              style={qStyles.micImage}
              resizeMode="contain"
            />
          </View>
        </TouchableOpacity>
        <Text style={qStyles.micLabel}>{label}</Text>
      </View>
    </View>
  );
}

const qStyles = StyleSheet.create({
  pinkCard: {
    backgroundColor: colors.pinkCard,
    borderWidth: 5,
    borderColor: colors.pinkBorder,
    borderRadius: 30,
    paddingTop: 20,
    paddingBottom: 14,
    paddingHorizontal: 12,
    shadowColor: colors.pinkBorder,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 1,
    shadowRadius: 0,
    elevation: 8,
  },
  questionText: {
    fontFamily: fonts.heading,
    fontSize: 32,
    color: colors.darkBlue,
    textAlign: "center",
    lineHeight: 42,
    marginBottom: 14,
  },
  yellowCard: {
    backgroundColor: colors.yellowCard,
    borderWidth: 5,
    borderColor: colors.yellowBorder,
    borderRadius: 26,
    paddingVertical: 20,
    paddingHorizontal: 20,
    alignItems: "center",
    gap: 8,
    shadowColor: colors.yellowBorder,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 1,
    shadowRadius: 0,
    elevation: 6,
  },
  yellowCardPressed: {
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0,
    transform: [{ translateY: 10 }],
    elevation: 0,
  },
  micBtn: { alignItems: "center" },
  micWrap: {
    width: 150,
    height: 150,
    alignItems: "center",
    justifyContent: "center",
    overflow: "visible",
  },
  micImage: {
    width: 150,
    height: 150,
  },
  micLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 24,
    color: colors.darkBlue,
    textAlign: "center",
  },
});

/* ═══════════════════════════════════════════════════════════════
   Feedback Card
   ═══════════════════════════════════════════════════════════════ */

function FeedbackCard({ data, onNext }: { data: any; onNext: () => void }) {
  const ev = data?.evaluation || {};
  const score = Math.round(ev.overall_score || 3);
  const heading = HEADING_MAP[score] || "Good Effort!";
  const comment = data?.followup || ev.feedback || "";

  return (
    <View style={fStyles.blueCard}>
      <Text style={fStyles.heading}>{heading}</Text>
      <Stars score={score} size={40} />
      <Text style={fStyles.comment}>{comment}</Text>
      <TouchableOpacity
        style={fStyles.nextBtn}
        onPress={onNext}
        activeOpacity={0.7}
      >
        <Text style={fStyles.nextText}>Next</Text>
      </TouchableOpacity>
    </View>
  );
}

const fStyles = StyleSheet.create({
  blueCard: {
    backgroundColor: colors.blueCard,
    borderWidth: 5,
    borderColor: colors.blueBorder,
    borderRadius: 30,
    padding: 24,
    alignItems: "center",
    shadowColor: colors.blueBorder,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 1,
    shadowRadius: 0,
    elevation: 8,
  },
  heading: {
    fontFamily: fonts.heading,
    fontSize: 44,
    color: colors.darkBlueText,
    marginBottom: 10,
  },
  comment: {
    fontFamily: fonts.body,
    fontSize: 22,
    color: colors.darkBlueText,
    textAlign: "center",
    lineHeight: 32,
    marginVertical: 16,
  },
  nextBtn: {
    backgroundColor: colors.greenBtn,
    borderWidth: 5,
    borderColor: colors.greenBorder,
    borderRadius: 999,
    paddingVertical: 14,
    width: "80%",
    alignItems: "center",
    shadowColor: colors.greenBorder,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 1,
    shadowRadius: 0,
    elevation: 6,
  },
  nextText: {
    fontFamily: fonts.heading,
    fontSize: 30,
    color: colors.black,
  },
});

/* ═══════════════════════════════════════════════════════════════
   Main Layout Styles
   ═══════════════════════════════════════════════════════════════ */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
    padding: 20,
    paddingTop: Platform.OS === "ios" ? 58 : 20,
    position: "relative" as const,
  },

  /* Top Bar */
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    height: 52,
    marginBottom: 20,
    position: "relative" as const,
    overflow: "visible" as const,
  },
  logo: {
    height: 150,
    width: 170,
    position: "absolute",
    left: 0,
    top: "50%",
    transform: [{ translateY: -75 }],
    zIndex: 10,
  },
  logoSpacer: { width: 160, flexShrink: 0 },
  progressTrack: {
    flex: 1,
    height: 52,
    backgroundColor: colors.darkBlue,
    borderRadius: 999,
    overflow: "hidden",
    position: "relative" as const,
  },
  progressFill: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    backgroundColor: colors.progressFill,
    borderRadius: 999,
  },
  progressInner: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    paddingRight: 16,
  },
  progressText: {
    fontFamily: fonts.heading,
    fontSize: 26,
    color: colors.white,
  },
  closeBtn: {
    fontFamily: fonts.heading,
    fontSize: 52,
    color: colors.darkBlue,
    lineHeight: 52,
  },

  /* Body — landscape (side by side) */
  bodyLandscape: {
    flex: 1,
    flexDirection: "row",
    gap: 16,
  },
  imageWrapLandscape: {
    borderRadius: 30,
    overflow: "hidden" as const,
    alignSelf: "stretch",
  },
  imageLandscape: {
    width: "100%",
    height: "100%",
  },
  sidebarWrapLandscape: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    alignSelf: "stretch",
    position: "relative" as const,
  },
  sidebarInner: {
    width: "100%",
    maxWidth: 400,
  },

  /* Body — portrait (stacked) */
  bodyVertical: {
    gap: 14,
    paddingBottom: 20,
  },
  imageWrapPortrait: {
    width: "100%",
    aspectRatio: 16 / 10,
    borderRadius: 24,
    overflow: "hidden",
  },
  sidebarWrapPortrait: {
    width: "100%",
    position: "relative" as const,
    alignItems: "center",
    paddingVertical: 20,
  },

  image: {
    width: "100%",
    height: "100%",
  },
});
