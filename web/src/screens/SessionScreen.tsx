import React, { useRef, useState, useEffect } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  Pressable,
  StyleSheet,
  Platform,
  Alert,
  ScrollView,
  useWindowDimensions,
} from "react-native";
import { Audio } from "expo-av";
import { colors, fonts } from "../theme";
import { imageUrl, transcribeAudio, evaluate, endSession, speakTTS } from "../api";
import { track } from "../lib/analytics";
import Stars from "../components/Stars";
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
  const [processingStep, setProcessingStep] = useState<"transcribing" | "evaluating" | "speaking" | null>(null);
  const [heardText, setHeardText] = useState<string>("");
  const [adioComment, setAdioComment] = useState<string>("");
  const recordingRef = useRef<Audio.Recording | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [micCenter, setMicCenter] = useState<{ x: number; y: number } | undefined>();
  const { volume, startMetering, stopMetering } = useVolumeMeter();
  const questionShownAtRef = useRef<number>(Date.now());
  const initiationLatencyMsRef = useRef<number | undefined>(undefined);

  // Same clay recipe as the question cards, but with softer blur/offset and
  // lower opacity so the photo doesn't compete with the cards for depth.
  const imageClayShadow =
    "8px 8px 18px rgba(45,30,10,0.14), inset -4px -4px 10px rgba(255,255,255,0.25)";

  // White pill sitting behind the progress bar — same clay recipe as the
  // pink/yellow cards (outer drop shadow + a bold inset shadow for volume),
  // recolored gray since the pill has no border hue to draw an inset from.
  const progressPillClayShadow =
    "10px 10px 22px rgba(40,40,40,0.15), inset -7px -7px 14px rgba(120,120,120,0.45)";

  // Same recipe applied to the green fill itself — inset only, since the
  // track clips overflow and an outer drop shadow would just be cut off.
  const progressFillClayShadow =
    "inset -6px -6px 12px rgba(140,160,20,0.35)";

  const progress = session.progress;
  const currentNum = Math.min(progress.answered + 1, progress.total);
  const pct = progress.total
    ? Math.round((progress.answered / progress.total) * 100)
    : 0;

  useEffect(() => {
    if (session.question) {
      track("question_viewed", {
        question_index: progress.answered,
        structure_word: session.question.structure_word,
        difficulty: session.question.difficulty,
      });
    }
  }, [session.question?.id]);

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
      // Capture how long the user waited before speaking
      initiationLatencyMsRef.current = Date.now() - questionShownAtRef.current;
      
      track("recording_started", {
        question_index: progress.answered,
        structure_word: session.question?.structure_word,
      });
    } catch (err: any) {
      track("app_error", { area: "recording_start", error_message: err.message });
      showAlert("Recording error", err.message);
    }
  }

  async function stopRecording() {
    if (!recordingRef.current) return;
    
    // Calculate how long they were speaking
    const recordingDurationMs = recordingRef.current._finalDurationMillis || 0; // rough duration
    
    setIsRecording(false);
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

      track("recording_stopped", { duration_ms: recordingDurationMs });

      setIsProcessing(true);
      setProcessingStep("transcribing");
      setHeardText("");
      setAdioComment("");

      const mimeType =
        Platform.OS === "web"
          ? "audio/webm"
          : Platform.OS === "ios"
          ? "audio/m4a"
          : "audio/mp4";
      const tData = await transcribeAudio(session.session_id, uri, mimeType);
      const transcription = (tData.transcription || tData.text || "").trim();

      if (!transcription) {
        showAlert("Couldn't hear that", "We didn't catch anything. Please try speaking again.");
        return;
      }

      setHeardText(transcription);
      setProcessingStep("evaluating");

      const eData = await evaluate(
        session.session_id,
        transcription,
        initiationLatencyMsRef.current,
      );
      track("evaluation_completed", {
        latency_ms: eData.evaluation?.llm_latency_ms || 0,
        followup_created: !!eData.followup
      });

      const comment = eData.followup || eData.evaluation?.feedback || "";
      if (comment) {
        setAdioComment(comment);
        setProcessingStep("speaking");
        await speakTTS(comment).catch(() => {});
      }

      setFeedbackData(eData);
      onUpdateSession({
        question: eData.next_question,
        progress: eData.progress,
      });
      setCardState("feedback");
    } catch (err: any) {
      track("app_error", { area: "processing", error_message: err.message });
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
    questionShownAtRef.current = Date.now();
    initiationLatencyMsRef.current = undefined;
    setHeardText("");
    setAdioComment("");
    setProcessingStep(null);
    if (session.progress.completed || !session.question) {
      track("session_completed", { 
        questions_answered: session.progress.answered,
        total_questions: session.progress.total
      });
      onEnd();
    } else {
      setCardState("question");
    }
  }

  async function handleClose() {
    try {
      track("session_abandoned", {
        questions_answered: session.progress.answered,
        total_questions: session.progress.total
      });
      await endSession(session.session_id);
    } catch {}
    onEnd();
  }

  const topPad = Platform.OS === "ios" ? 58 : 20;
  const bodyHeight = height - topPad - 20 - 52 - 20;
  const imageWidth = isPortrait ? undefined : bodyHeight * (9 / 10);

  return (
    <View style={styles.container}>
      {/* ─── Top Bar ─── */}
      <View style={styles.topBar}>
        <View style={[styles.progressPillWrap, { boxShadow: progressPillClayShadow } as any]}>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${pct}%`, boxShadow: progressFillClayShadow } as any]} />
            <View style={styles.progressInner}>
              <View style={{ flex: 1 }} />
              <Text style={styles.progressText}>
                {currentNum}/{progress.total}
              </Text>
            </View>
          </View>
        </View>
        <TouchableOpacity onPress={handleClose} hitSlop={12} style={styles.closeBtnWrap}>
          <Text style={styles.closeBtn}>×</Text>
        </TouchableOpacity>
      </View>

      {/* ─── Body ─── */}
      {isPortrait ? (
        <ScrollView
          style={{ flex: 1, overflow: "visible" }}
          contentContainerStyle={[styles.bodyVertical, { overflow: "visible" }]}
          showsVerticalScrollIndicator={false}
        >
          <View style={[styles.imageShadowPortrait, { boxShadow: imageClayShadow } as any]}>
            <View style={styles.imageWrapPortrait}>
              <Image
                source={{ uri: imageUrl(session.image_url) }}
                style={styles.image}
                resizeMode="cover"
              />
            </View>
          </View>
          <View style={styles.sidebarWrapPortrait}>
            <View style={styles.sidebarInner}>
              {cardState === "question" ? (
                <QuestionCard
                  question={session.question}
                  isRecording={isRecording}
                  isProcessing={isProcessing}
                  processingStep={processingStep}
                  heardText={heardText}
                  adioComment={adioComment}
                  onToggle={toggleRecording}
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
          <View style={[styles.imageShadowLandscape, { width: imageWidth, boxShadow: imageClayShadow } as any]}>
            <View style={styles.imageWrapLandscape}>
              <Image
                source={{ uri: imageUrl(session.image_url) }}
                style={styles.imageLandscape}
                resizeMode="cover"
              />
            </View>
          </View>
          <View style={styles.sidebarWrapLandscape}>
            <View style={styles.sidebarInner}>
              {cardState === "question" ? (
                <QuestionCard
                  question={session.question}
                  isRecording={isRecording}
                  isProcessing={isProcessing}
                  processingStep={processingStep}
                  heardText={heardText}
                  adioComment={adioComment}
                  onToggle={toggleRecording}
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
  processingStep,
  heardText,
  adioComment,
  onToggle,
  onMicLayout,
}: {
  question: Question | null;
  isRecording: boolean;
  isProcessing: boolean;
  processingStep?: "transcribing" | "evaluating" | "speaking" | null;
  heardText?: string;
  adioComment?: string;
  onToggle: () => void;
  onMicLayout?: (center: { x: number; y: number }) => void;
}) {
  const micRef = useRef<View>(null);
  const [micPressed, setMicPressed] = useState(false);

  if (!question) return null;

  const label = isProcessing
    ? processingStep === "transcribing" ? "Working on it!"
    : processingStep === "evaluating"  ? "Working on it!"
    : processingStep === "speaking"    ? "Adio says:"
    : "Processing…"
    : isRecording
    ? "Tap to stop"
    : "Tap to speak";

  const handleMicLayout = () => {
    if (micRef.current && onMicLayout) {
      (micRef.current as any).measureInWindow?.(
        (x: number, y: number, w: number, h: number) => {
          if (x != null) onMicLayout({ x: x + w / 2, y: y + h / 2 });
        }
      );
    }
  };

  // Outer pink card — claymorphic sculpt using a brighter, more vivid red
  // instead of a near-black maroon, for a punchier, more playful feel.
  const pinkClayShadow =
    "18px 18px 55px rgba(120,10,40,0.29), inset -12px -12px 34px rgba(255,45,110,0.6)";

  // Inner yellow card — same recipe, scaled for the smaller surface.
  // Hue-shifted toward a brighter, more vivid orange instead of a muddy brown.
  const yellowClayShadow =
    "10px 10px 34px rgba(140,60,0,0.27), inset -9px -9px 28px rgba(255,150,20,0.75)";

  const clayTransition = Platform.OS === "web"
    ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
    : undefined;

  return (
    <View style={qStyles.wrapper}>
      <View style={[qStyles.pinkCard, { boxShadow: pinkClayShadow } as any]}>
        <View style={qStyles.questionTextWrap}>
          <Text style={qStyles.questionText}>{question.text}</Text>
        </View>
        <View
          style={[qStyles.yellowInner, { boxShadow: yellowClayShadow } as any, clayTransition]}
        >
          <Pressable
            onPress={onToggle}
            disabled={isProcessing}
            onPressIn={() => setMicPressed(true)}
            onPressOut={() => setMicPressed(false)}
            style={qStyles.micBtn}
          >
            <View
              ref={micRef}
              style={[
                qStyles.micWrap,
                { transform: [{ translateY: micPressed ? 2 : 0 }] },
                clayTransition,
              ]}
              onLayout={handleMicLayout}
            >
              {isProcessing ? (
                <Image
                  source={require("../../assets/spinner.gif")}
                  style={qStyles.micImage}
                  resizeMode="contain"
                />
              ) : (
                <Image
                  source={require("../../assets/micV3.png")}
                  style={qStyles.micImage}
                  resizeMode="contain"
                />
              )}
            </View>
          </Pressable>
          <View style={qStyles.micTextCol}>
            {!!label && <Text style={qStyles.micLabel}>{label}</Text>}
            {!!heardText && processingStep !== "speaking" && (
              <Text style={qStyles.heardText}>"{heardText}"</Text>
            )}
            {!!adioComment && processingStep === "speaking" && (
              <Text style={qStyles.adioText}>{adioComment}</Text>
            )}
          </View>
        </View>
      </View>
    </View>
  );
}

const qStyles = StyleSheet.create({
  wrapper: {
    paddingTop: 0,
    paddingBottom: 20,
    paddingHorizontal: 0,
    alignItems: "center",
  },
  pinkCard: {
    backgroundColor: "#FFDDEB",
    borderRadius: 30,
    padding: 20,
    gap: 20,
    alignItems: "center",
    alignSelf: "stretch",
    width: "100%",
  },
  questionTextWrap: {
    minHeight: 80,
    justifyContent: "center",
    alignItems: "center",
    alignSelf: "stretch",
  },
  questionText: {
    fontFamily: fonts.fredoka,
    fontSize: 26,
    color: colors.darkBlue,
    textAlign: "center",
    lineHeight: 30,
  },
  yellowInner: {
    backgroundColor: "#FFF0A8",
    borderRadius: 22,
    paddingVertical: 16,
    paddingHorizontal: 44,
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
    alignSelf: "stretch",
  },
  micBtn: { alignItems: "center" },
  micWrap: {
    width: 100,
    height: 100,
    alignItems: "center",
    justifyContent: "center",
    overflow: "visible",
  },
  micImage: {
    width: 100,
    height: 100,
  },
  micTextCol: {
    flex: 1,
    alignItems: "flex-end",
    gap: 4,
  },
  micLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 19,
    color: colors.darkBlue,
    textAlign: "right",
  },
  heardText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlue,
    textAlign: "right",
    fontStyle: "italic",
  },
  adioText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.darkBlue,
    textAlign: "right",
    lineHeight: 21,
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

  // Same clay recipe as the question cards — outer drop shadow + a bold
  // inset shadow, brightened toward a vivid sky blue instead of dark navy.
  const blueClayShadow =
    "18px 18px 55px rgba(10,70,130,0.29), inset -12px -12px 34px rgba(50,180,240,0.65)";
  // Brightened toward a vivid lime green instead of a muddy dark olive.
  const greenClayShadow =
    "10px 10px 34px rgba(90,120,0,0.27), inset -9px -9px 28px rgba(170,210,20,0.8)";

  return (
    <View style={[fStyles.blueCard, { boxShadow: blueClayShadow } as any]}>
      <Text style={fStyles.heading}>{heading}</Text>
      <Stars score={score} size={32} />
      <Text style={fStyles.comment}>{comment}</Text>
      <TouchableOpacity
        style={[fStyles.nextBtn, { boxShadow: greenClayShadow } as any]}
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
    borderRadius: 30,
    padding: 24,
    alignItems: "center",
    alignSelf: "stretch",
    width: "100%",
  },
  heading: {
    fontFamily: fonts.fredoka,
    fontSize: 36,
    color: colors.darkBlueText,
    marginBottom: 8,
  },
  comment: {
    fontFamily: fonts.body,
    fontSize: 18,
    color: colors.darkBlueText,
    textAlign: "center",
    lineHeight: 25,
    marginVertical: 14,
    width: "100%",
    alignSelf: "stretch",
  },
  nextBtn: {
    backgroundColor: colors.greenBtn,
    borderRadius: 20,
    paddingVertical: 12,
    width: "80%",
    alignItems: "center",
  },
  nextText: {
    fontFamily: fonts.fredoka,
    fontSize: 26,
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
    gap: 4,
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
  progressPillWrap: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 999,
    padding: 10,
  },
  progressTrack: {
    flex: 1,
    height: 28,
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
    fontFamily: fonts.bodySemiBold,
    fontSize: 15,
    color: colors.white,
  },
  closeBtnWrap: {
    height: 48,
    width: 32,
    alignItems: "flex-end",
    justifyContent: "center",
  },
  closeBtn: {
    fontFamily: fonts.heading,
    fontSize: 40,
    color: colors.darkBlue,
    lineHeight: 40,
  },

  /* Body — landscape (side by side) */
  bodyLandscape: {
    flex: 1,
    flexDirection: "row",
    gap: 20,
  },
  imageShadowLandscape: {
    borderRadius: 30,
    backgroundColor: colors.bg,
    alignSelf: "stretch",
  },
  imageWrapLandscape: {
    flex: 1,
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
    gap: 20,
    paddingBottom: 20,
  },
  imageShadowPortrait: {
    width: "100%",
    aspectRatio: 9 / 10,
    borderRadius: 24,
    backgroundColor: colors.bg,
  },
  imageWrapPortrait: {
    flex: 1,
    borderRadius: 24,
    overflow: "hidden",
  },
  sidebarWrapPortrait: {
    width: "100%",
    position: "relative" as const,
    alignItems: "center",
    paddingVertical: 0,
  },

  image: {
    width: "100%",
    height: "100%",
  },
});
