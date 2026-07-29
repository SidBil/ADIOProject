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
  Animated,
  Easing,
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

// How long a single peek reveals the real image before it auto-re-blurs. A peek
// is a brief glance, not a hold-open toggle (01_recall_session.md §4).
const PEEK_SECONDS = 3;

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
  // V&V condition for this session. "stage2" (recall) hides the image after a
  // viewing phase; "stage1" (or absent) keeps it visible throughout.
  mode?: "stage1" | "stage2";
  question: Question | null;
  total_questions: number;
  progress: { answered: number; total: number; completed: boolean };
}

interface Props {
  session: SessionData;
  // completed=true only when all questions (and follow-ups) resolved; the
  // celebration screen is gated on this. An early quit passes false.
  onEnd: (completed: boolean) => void;
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
  // Total time to answer: question shown -> answer submitted (recording stopped).
  const answerDurationMsRef = useRef<number | undefined>(undefined);

  // ─── Stage 2 (recall) state ───
  // A Stage 2 session shows the image, then hides it (blurred) for the whole
  // question set; the child answers from memory (01_recall_session.md §3).
  const isStage2 = session.mode === "stage2";
  // Flipped once, via the "ready" gate — a single hide event per session, not a
  // per-question toggle. Stage 1 never hides.
  const [imageHidden, setImageHidden] = useState(false);
  // One free peek per session: reveals the real image once, logged but never
  // scored, then re-blurs for the rest of the session (§4).
  const [peekUsed, setPeekUsed] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const [peekLeft, setPeekLeft] = useState(PEEK_SECONDS);

  // The image reads as hidden only in Stage 2, after the ready gate, and while
  // not mid-peek. Blur is applied via CSS filter on web and the Image
  // `blurRadius` prop on native (§8 open question — MVP: both paths covered).
  const blurred = isStage2 && imageHidden && !peeking;
  // Heavy blur so no recallable detail survives (01_recall_session.md §3.4). The
  // scale-up covers the transparent edge bleed the filter leaves at this radius.
  const webBlurStyle =
    Platform.OS === "web" && blurred
      ? ({ filter: "blur(60px)", transform: [{ scale: 1.3 }] } as any)
      : null;
  const nativeBlurRadius = Platform.OS !== "web" && blurred ? 55 : 0;
  // Before the ready gate is tapped, Stage 2 sits in a viewing phase: image at
  // full visibility, no question card yet.
  const showViewingGate = isStage2 && !imageHidden;

  const startPeek = () => {
    if (peekUsed || peeking) return;
    setPeekLeft(PEEK_SECONDS);
    setPeeking(true);
    // Logged for instrumentation (which turn it was used on); does NOT affect
    // the accuracy score (§4).
    track("peek_used", {
      question_index: session.progress.answered,
      mode: session.mode,
    });
  };

  // A peek is a timed glance: once started it counts down, then auto re-blurs
  // and is spent — so it can't be held open as unlimited viewing.
  useEffect(() => {
    if (!peeking) return;
    const end = setTimeout(() => {
      setPeeking(false);
      setPeekUsed(true);
    }, PEEK_SECONDS * 1000);
    const tick = setInterval(
      () => setPeekLeft((s) => Math.max(0, s - 1)),
      1000,
    );
    return () => {
      clearTimeout(end);
      clearInterval(tick);
    };
  }, [peeking]);

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

  // Read the question aloud when its card is shown (supports pre-readers and the
  // reading-comprehension goal). Only while the question card is actually up — not
  // during the feedback card, and not during the Stage 2 viewing gate.
  useEffect(() => {
    if (cardState === "question" && !showViewingGate && session.question?.text) {
      speakTTS(session.question.text).catch(() => {});
    }
  }, [cardState, showViewingGate, session.question?.id]);

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

      // expo-av permits only ONE prepared Recording at a time. Clear any leftover
      // (e.g. from a fast re-tap or a prior cycle that errored before unloading)
      // so createAsync can't throw "only one recording object can be prepared".
      if (recordingRef.current) {
        try {
          await recordingRef.current.stopAndUnloadAsync();
        } catch {}
        recordingRef.current = null;
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
    // Total time to answer this question: question shown -> answer submitted.
    answerDurationMsRef.current = Date.now() - questionShownAtRef.current;

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
        answerDurationMsRef.current,
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
    answerDurationMsRef.current = undefined;
    setHeardText("");
    setAdioComment("");
    setProcessingStep(null);
    if (session.progress.completed || !session.question) {
      track("session_completed", {
        questions_answered: session.progress.answered,
        total_questions: session.progress.total
      });
      // Finalize the DB row (ended_at, completed=true, questions_answered) so the
      // parent/SLP clinical view has accurate timing + status. Fire-and-forget:
      // the celebration doesn't depend on it.
      endSession(session.session_id).catch(() => {});
      onEnd(true);
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
    onEnd(false);
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
      {showViewingGate ? (
        <ViewingGate
          uri={imageUrl(session.image_url)}
          isPortrait={isPortrait}
          imageWidth={imageWidth}
          imageClayShadow={imageClayShadow}
          onReady={() => setImageHidden(true)}
        />
      ) : isPortrait ? (
        <ScrollView
          style={{ flex: 1, overflow: "visible" }}
          contentContainerStyle={[styles.bodyVertical, { overflow: "visible" }]}
          showsVerticalScrollIndicator={false}
        >
          <View style={[styles.imageShadowPortrait, { boxShadow: imageClayShadow } as any]}>
            <View style={styles.imageWrapPortrait}>
              <Image
                source={{ uri: imageUrl(session.image_url) }}
                style={[styles.image, webBlurStyle]}
                resizeMode="cover"
                blurRadius={nativeBlurRadius}
              />
              {isStage2 && imageHidden && (
                <PeekOverlay
                  peeking={peeking}
                  peekUsed={peekUsed}
                  peekLeft={peekLeft}
                  onPeek={startPeek}
                />
              )}
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
                style={[styles.imageLandscape, webBlurStyle]}
                resizeMode="cover"
                blurRadius={nativeBlurRadius}
              />
              {isStage2 && imageHidden && (
                <PeekOverlay
                  peeking={peeking}
                  peekUsed={peekUsed}
                  peekLeft={peekLeft}
                  onPeek={startPeek}
                />
              )}
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
   Stage 2 (recall) — viewing gate + peek overlay
   ═══════════════════════════════════════════════════════════════ */

/**
 * The pre-question viewing phase for a Stage 2 (recall) session. The image is
 * shown at full visibility with no minimum viewing time — the child views for as
 * long as they want, then taps the "ready" control to hide it and start
 * answering from memory (01_recall_session.md §3).
 */
function ViewingGate({
  uri,
  isPortrait,
  imageWidth,
  imageClayShadow,
  onReady,
}: {
  uri: string;
  isPortrait: boolean;
  imageWidth?: number;
  imageClayShadow: string;
  onReady: () => void;
}) {
  const [pressed, setPressed] = useState(false);
  const redClayShadow = pressed
    ? "5px 5px 18px rgba(150,10,60,0.22), inset -5px -5px 16px rgba(247,29,115,0.8)"
    : "10px 10px 34px rgba(150,10,60,0.27), inset -9px -9px 28px rgba(247,29,115,0.72)";
  const clayTransition =
    Platform.OS === "web"
      ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
      : undefined;

  const readyCard = (
    <View style={gateStyles.card}>
      <Text style={gateStyles.title}>Look closely!</Text>
      <Text style={gateStyles.subtitle}>
        Remember the picture. When you're ready, we'll hide it and you'll tell me
        about it from memory.
      </Text>
      <Pressable
        onPress={onReady}
        onPressIn={() => setPressed(true)}
        onPressOut={() => setPressed(false)}
        accessibilityRole="button"
        accessibilityLabel="I'm ready — hide the picture"
        style={[
          gateStyles.readyBtn,
          { boxShadow: redClayShadow, transform: [{ translateY: pressed ? 3 : 0 }] } as any,
          clayTransition,
        ]}
      >
        <Text style={gateStyles.readyText}>I'm ready</Text>
      </Pressable>
    </View>
  );

  if (isPortrait) {
    return (
      <ScrollView
        style={{ flex: 1, overflow: "visible" }}
        contentContainerStyle={[styles.bodyVertical, { overflow: "visible" }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={[styles.imageShadowPortrait, { boxShadow: imageClayShadow } as any]}>
          <View style={styles.imageWrapPortrait}>
            <Image source={{ uri }} style={styles.image} resizeMode="cover" />
          </View>
        </View>
        <View style={styles.sidebarWrapPortrait}>
          <View style={styles.sidebarInner}>{readyCard}</View>
        </View>
      </ScrollView>
    );
  }

  return (
    <View style={styles.bodyLandscape}>
      <View style={[styles.imageShadowLandscape, { width: imageWidth, boxShadow: imageClayShadow } as any]}>
        <View style={styles.imageWrapLandscape}>
          <Image source={{ uri }} style={styles.imageLandscape} resizeMode="cover" />
        </View>
      </View>
      <View style={styles.sidebarWrapLandscape}>
        <View style={styles.sidebarInner}>{readyCard}</View>
      </View>
    </View>
  );
}

/**
 * The peek control, overlaid on the hidden (blurred) image during a Stage 2
 * session. Offers a single free peek; once used it shows a spent-state hint
 * rather than another peek (01_recall_session.md §4). While peeking, the image
 * is un-blurred (handled by the parent) and this shows a countdown — the reveal
 * is timed and auto-ends, so it can't be held open.
 */
function PeekOverlay({
  peeking,
  peekUsed,
  peekLeft,
  onPeek,
}: {
  peeking: boolean;
  peekUsed: boolean;
  peekLeft: number;
  onPeek: () => void;
}) {
  if (peeking) {
    return (
      <View style={peekStyles.pill}>
        <Text style={peekStyles.pillText}>Peeking… {peekLeft}</Text>
      </View>
    );
  }
  if (peekUsed) {
    return (
      <View style={[peekStyles.pill, peekStyles.pillSpent]}>
        <Text style={[peekStyles.pillText, peekStyles.pillTextSpent]}>Peek used</Text>
      </View>
    );
  }
  return (
    <Pressable
      onPress={onPeek}
      accessibilityRole="button"
      accessibilityLabel="Peek at the picture once"
      style={peekStyles.pill}
    >
      <Text style={peekStyles.pillText}>👀 Peek once</Text>
    </Pressable>
  );
}

const gateStyles = StyleSheet.create({
  card: {
    backgroundColor: colors.yellowCard,
    borderRadius: 30,
    padding: 24,
    gap: 16,
    alignItems: "center",
    alignSelf: "stretch",
    width: "100%",
    boxShadow:
      "18px 18px 55px rgba(150,120,0,0.24), inset -12px -12px 34px rgba(251,222,40,0.7)" as any,
  },
  title: {
    fontFamily: fonts.fredoka,
    fontSize: 30,
    color: colors.darkBlue,
    textAlign: "center",
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 16,
    lineHeight: 23,
    color: colors.darkBlue,
    textAlign: "center",
  },
  readyBtn: {
    backgroundColor: colors.pinkCard,
    borderRadius: 22,
    paddingVertical: 14,
    paddingHorizontal: 40,
    width: "80%",
    alignItems: "center",
  },
  readyText: {
    fontFamily: fonts.fredoka,
    fontSize: 26,
    color: colors.darkBlue,
  },
});

const peekStyles = StyleSheet.create({
  pill: {
    position: "absolute",
    bottom: 14,
    alignSelf: "center",
    backgroundColor: colors.white,
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: 999,
    boxShadow: "8px 8px 20px rgba(0,0,0,0.18)" as any,
  },
  pillSpent: {
    backgroundColor: "rgba(255,255,255,0.6)",
  },
  pillText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 16,
    color: colors.darkBlue,
  },
  pillTextSpent: {
    color: "#9297A0",
  },
});

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

  // Pulsing "radar" halo behind the mic while recording, so it's obvious the app
  // is listening. Loops an expanding, fading ring; stops + resets when not recording.
  const pulse = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (!isRecording) {
      pulse.setValue(0);
      return;
    }
    const loop = Animated.loop(
      Animated.timing(pulse, {
        toValue: 1,
        duration: 1200,
        easing: Easing.out(Easing.ease),
        useNativeDriver: false,
      }),
    );
    loop.start();
    return () => loop.stop();
  }, [isRecording]);
  const ringScale = pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.85] });
  const ringOpacity = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.4, 0] });

  if (!question) return null;

  const label = isProcessing
    ? "Working on it!"
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
  // Presses in on tap (same clay-button feel as the Adult-tab rows): the outer
  // drop shrinks and the inset deepens while held.
  const yellowClayShadow = micPressed
    ? "5px 5px 18px rgba(140,60,0,0.25), inset -5px -5px 16px rgba(255,150,20,0.8)"
    : "10px 10px 34px rgba(140,60,0,0.27), inset -9px -9px 28px rgba(255,150,20,0.75)";

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
          style={[
            qStyles.yellowInner,
            {
              boxShadow: yellowClayShadow,
              transform: [{ translateY: micPressed ? 3 : 0 }],
            } as any,
            clayTransition,
          ]}
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
              style={[qStyles.micWrap, clayTransition]}
              onLayout={handleMicLayout}
            >
              {isRecording && (
                <Animated.View
                  pointerEvents="none"
                  style={[
                    qStyles.micRing,
                    { transform: [{ scale: ringScale }], opacity: ringOpacity } as any,
                  ]}
                />
              )}
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
  micRing: {
    position: "absolute",
    width: 92,
    height: 92,
    borderRadius: 46,
    backgroundColor: "rgba(247,29,115,0.45)",
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
    fontSize: 26,
    color: colors.darkBlueText,
    marginBottom: 8,
  },
  comment: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.darkBlueText,
    textAlign: "center",
    lineHeight: 21,
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
    zIndex: 10,
    backgroundColor: colors.bg,
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
