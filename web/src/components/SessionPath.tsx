import React, { useRef, useCallback } from "react";
import {
  View,
  Text,
  Image,
  Pressable,
  StyleSheet,
  ScrollView,
  Platform,
  useWindowDimensions,
} from "react-native";
import Svg, { Path as SvgPath } from "react-native-svg";
import { colors, fonts } from "../theme";
import { imageUrl } from "../api";
import { buildPath, PathNode } from "../lib/path";

interface Props {
  completedCount: number;
  /**
   * Image filename per completed session, oldest-first (node N → index N-1).
   * A completed node shows its session's picture as a thumbnail instead of a
   * generic check.
   */
  completedImages?: string[];
  /** Fired when the single lit (current) node is tapped. */
  onStartNode: () => void;
  busy?: boolean;
}

// Green clay for reached nodes; blue for future (locked) nodes.
const NODE_FILL = "#C4E86A";
const NODE_FILL_LOCKED = "#C8EFFF"; // blue — future/locked nodes
const NODE_ACCENT = "150,190,50"; // green inset sculpt
const NODE_ACCENT_LOCKED = "41,165,225"; // blue inset sculpt
const TRAIL_LINE = "#CFE79A"; // green — reached portion
const TRAIL_LINE_BLUE = "#B4DEF4"; // blue — future portion

/**
 * Smooth Catmull-Rom spline through the node centres (emitted as cubic béziers),
 * drawing nodes [from..to]. Neighbours are read from the FULL array so the tangent
 * at the split node is identical for the green and blue halves — no seam. Because
 * the curve leaves each node along the neighbour direction, the connections bulge
 * (curvy), never straight diagonals, and it stays C1-smooth throughout.
 */
// Longer handles (vs the standard 1/6) → each connection bulges into a real
// curve instead of a near-straight diagonal.
const TENSION = 0.34;
function catmullRom(pts: { x: number; y: number }[], from: number, to: number): string {
  if (to <= from) return "";
  let d = `M ${pts[from].x.toFixed(1)} ${pts[from].y.toFixed(1)}`;
  for (let i = from; i < to; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1x = p1.x + (p2.x - p0.x) * TENSION;
    const c1y = p1.y + (p2.y - p0.y) * TENSION;
    const c2x = p2.x - (p3.x - p1.x) * TENSION;
    const c2y = p2.y - (p3.y - p1.y) * TENSION;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}

/**
 * The winding vertical session trail (Home tab centrepiece).
 *
 * Renders as a pure function of `completedCount` (see `src/lib/path.ts`).
 * Progression goes DOWNWARD: node 1 sits at the top and the child moves down the
 * trail toward the goal (next milestone) at the bottom. Exactly one node — the
 * current node — is lit and tappable (north_star §3 principle 1). Nodes sit on a
 * single smooth curve, uniformly green.
 */
export default function SessionPath({ completedCount, completedImages, onStartNode, busy }: Props) {
  const { width: winW, height: winH } = useWindowDimensions();
  const scrollRef = useRef<ScrollView>(null);
  const currentYRef = useRef<number | null>(null);

  // Progression goes DOWNWARD: node 1 at the top, later nodes (and the goal)
  // below it, so the child moves down the trail as they complete sessions.
  const ordered = buildPath(completedCount);

  // Responsive sizing — larger stones, generous vertical spacing.
  const colW = Math.min(460, winW);
  // Uniform node size for now — milestone/stage differentiation will come from
  // icons later, not size.
  const nodeSize = Math.max(96, Math.min(136, colW * 0.32));
  const rowH = nodeSize * 0.88;
  const cx = colW / 2;
  // Nodes alternate left/right; the spline bulges between them. Amplitude clamped
  // off the edges.
  const amp = Math.min(cx - nodeSize / 2 - 18, rowH * 0.7);
  const totalH = ordered.length * rowH;

  const positioned = ordered.map((node, i) => ({
    node,
    x: cx + amp * (i % 2 === 0 ? -1 : 1),
    y: i * rowH + rowH / 2,
  }));

  const ptsAll = positioned.map((p) => ({ x: p.x, y: p.y }));
  const lastIdx = ptsAll.length - 1;
  const currentIdx = positioned.findIndex((p) => p.node.state === "current");
  const current = currentIdx >= 0 ? positioned[currentIdx] : undefined;
  currentYRef.current = current ? current.y : null;
  // Split at the current node: green above (reached), blue below (future).
  const greenD = catmullRom(ptsAll, 0, currentIdx >= 0 ? currentIdx : lastIdx);
  const blueD = currentIdx >= 0 ? catmullRom(ptsAll, currentIdx, lastIdx) : "";

  // Centre the current node in the viewport once the content is laid out.
  const centreCurrent = useCallback(() => {
    const y = currentYRef.current;
    if (y == null) return;
    const target = Math.max(0, y - winH * 0.45);
    scrollRef.current?.scrollTo({ y: target, animated: false });
  }, [winH]);

  return (
    <ScrollView
      ref={scrollRef}
      style={{ flex: 1, width: "100%" }}
      contentContainerStyle={styles.content}
      onContentSizeChange={centreCurrent}
      showsVerticalScrollIndicator={false}
    >
      <View style={{ width: colW, height: totalH + rowH * 0.4 }}>
        {/* One continuous smooth trail line behind the nodes. */}
        <Svg
          width={colW}
          height={totalH + rowH * 0.4}
          style={[StyleSheet.absoluteFill, { pointerEvents: "none" }] as any}
        >
          {/* future (blue) drawn first, reached (green) on top so the join is clean */}
          <SvgPath
            d={blueD}
            stroke={TRAIL_LINE_BLUE}
            strokeWidth={14}
            strokeLinecap="round"
            fill="none"
          />
          <SvgPath
            d={greenD}
            stroke={TRAIL_LINE}
            strokeWidth={14}
            strokeLinecap="round"
            fill="none"
          />
        </Svg>

        {positioned.map(({ node, x, y }) => {
          const size = nodeSize;
          // Completed (non-milestone) nodes show the picture that session used.
          const filename =
            node.state === "completed" && !node.isMilestone
              ? completedImages?.[node.number - 1]
              : undefined;
          const thumbUri = filename ? imageUrl(`/images/${filename}`) : undefined;
          return (
            <View
              key={node.number}
              style={{
                position: "absolute",
                left: x - size / 2,
                top: y - size / 2,
                width: size,
                height: size,
                alignItems: "center",
                zIndex: 1,
              }}
            >
              <TrailNode
                node={node}
                size={size}
                thumbUri={thumbUri}
                onPress={onStartNode}
                busy={busy}
              />
            </View>
          );
        })}
      </View>
    </ScrollView>
  );
}

function TrailNode({
  node,
  size,
  thumbUri,
  onPress,
  busy,
}: {
  node: PathNode;
  size: number;
  thumbUri?: string;
  onPress: () => void;
  busy?: boolean;
}) {
  const [pressed, setPressed] = React.useState(false);
  const isCurrent = node.state === "current";
  const isLocked = node.state === "locked";
  const isDone = node.state === "completed";

  // Claymorphism: layered outer drop + green inset sculpt. The current node gets
  // the strongest "hero" treatment; locked nodes are flat.
  const depth = isCurrent ? 1 : isDone ? 0.6 : 0.35;
  const clay = isLocked
    ? `6px 6px 14px rgba(0,0,0,0.06), inset -5px -5px 12px rgba(${NODE_ACCENT_LOCKED},0.3)`
    : `${18 * depth}px ${18 * depth}px ${34 * depth}px rgba(0,0,0,${0.14 * depth + 0.04}), ` +
      `inset -${12 * depth}px -${12 * depth}px ${22 * depth}px rgba(${NODE_ACCENT},${0.55 * depth + 0.1})`;

  const clayTransition =
    Platform.OS === "web"
      ? ({ transition: "box-shadow 180ms ease, transform 180ms ease" } as any)
      : undefined;

  const content = (
    <View
      style={[
        styles.node,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: isLocked ? NODE_FILL_LOCKED : NODE_FILL,
          boxShadow: clay,
          transform: [{ translateY: pressed ? 3 : 0 }],
        } as any,
        clayTransition,
      ]}
    >
      {node.isMilestone ? (
        // Milestone: always a star — grey until completed, background colour once
        // done. (Warming-up spinner still takes over on a current milestone tap.)
        isCurrent && busy ? (
          <Image
            source={require("../../assets/spinner.gif")}
            style={{ width: size * 0.58, height: size * 0.58 }}
            resizeMode="contain"
          />
        ) : (
          <Star size={size * 0.58} completed={isDone} />
        )
      ) : (
        <>
          {/* Completed nodes show that session's picture, framed by the clay rim;
              fall back to a check when no image is available. */}
          {isDone && thumbUri ? (
            <Image
              source={{ uri: thumbUri }}
              style={{
                width: size * 0.82,
                height: size * 0.82,
                borderRadius: (size * 0.82) / 2,
              }}
              resizeMode="cover"
            />
          ) : (
            isDone && <Check size={size * 0.5} />
          )}
          {/* Warming up: a clear spinner so the tap obviously registered. */}
          {isCurrent && busy && (
            <Image
              source={require("../../assets/spinner.gif")}
              style={{ width: size * 0.56, height: size * 0.56 }}
              resizeMode="contain"
            />
          )}
          {/* Stage 2 (recall) nodes show the timer glyph in place of the image
              icon, signalling the timed-recall condition (V&V Stage 2 PRD). */}
          {isCurrent && !busy &&
            (node.stage === "stage2" ? <Timer size={size * 0.6} /> : <Play size={size * 0.56} />)}
          {isLocked && <Lock size={size * 0.44} />}
        </>
      )}
    </View>
  );

  if (!isCurrent) {
    // Completed and locked nodes are not tappable in v1 (02_session_path.md §3).
    return content;
  }

  return (
    <Pressable
      disabled={busy}
      onPress={onPress}
      onPressIn={() => setPressed(true)}
      onPressOut={() => setPressed(false)}
      accessibilityRole="button"
      accessibilityLabel={node.isMilestone ? "Start milestone session" : "Start session"}
    >
      {content}
      <Text style={[styles.startLabel, { width: size * 1.8, left: -size * 0.4 }]}>
        {busy ? "Getting ready…" : "Start"}
      </Text>
    </Pressable>
  );
}

function Play({ size }: { size: number }) {
  // Artwork from assets/image-svgrepo-com.svg, inlined so it renders through
  // react-native-svg on web + native without adding an SVG-file loader. Recoloured
  // to the background colour to match the trail's other node glyphs.
  return (
    <Svg width={size} height={size} viewBox="0 0 32 32">
      <SvgPath
        d="M0 26.016q0 2.496 1.76 4.224t4.256 1.76h20q2.464 0 4.224-1.76t1.76-4.224v-20q0-2.496-1.76-4.256t-4.224-1.76h-20q-2.496 0-4.256 1.76t-1.76 4.256v20zM4 26.016v-20q0-0.832 0.576-1.408t1.44-0.608h20q0.8 0 1.408 0.608t0.576 1.408v20q0 0.832-0.576 1.408t-1.408 0.576h-20q-0.832 0-1.44-0.576t-0.576-1.408zM6.016 24q0 0.832 0.576 1.44t1.408 0.576h16q0.832 0 1.408-0.576t0.608-1.44v-0.928q-0.224-0.448-1.12-2.688t-1.6-3.584-1.28-2.112q-0.544-0.576-1.12-0.608t-1.152 0.384-1.152 1.12-1.184 1.568-1.152 1.696-1.152 1.6-1.088 1.184-1.088 0.448q-0.576 0-1.664-1.44-0.16-0.192-0.48-0.608-1.12-1.504-1.6-1.824-0.768-0.512-1.184 0.352-0.224 0.512-0.928 2.24t-1.056 2.56v0.64zM6.016 9.024q0 1.248 0.864 2.112t2.112 0.864 2.144-0.864 0.864-2.112-0.864-2.144-2.144-0.864-2.112 0.864-0.864 2.144z"
        fill={colors.bg}
      />
    </Svg>
  );
}

function Timer({ size }: { size: number }) {
  // Stopwatch artwork from assets/timer-icon.svg, inlined so it renders through
  // react-native-svg on web + native. Recoloured to the background colour to
  // match the trail's other node glyphs. Taller than wide — react-native-svg
  // centres it via the default preserveAspectRatio.
  return (
    <Svg width={size} height={size} viewBox="0 0 101.5 122.88">
      <SvgPath
        d="M56.83,21.74c11.58,1.38,21.96,6.67,29.8,14.5c9.18,9.18,14.86,21.87,14.86,35.88c0,14.01-5.68,26.7-14.86,35.89 s-21.87,14.87-35.89,14.87c-14.01,0-26.7-5.68-35.88-14.87C5.68,98.83,0,86.14,0,72.13c0-14.01,5.68-26.7,14.86-35.88 c8.41-8.41,19.77-13.89,32.38-14.75v-8.51c0-0.13,0.01-0.26,0.02-0.38l-9.51,0c-1.56,0-2.83-1.28-2.83-2.83V2.83 C34.92,1.28,36.2,0,37.76,0h28.57c1.56,0,2.84,1.28,2.84,2.83v6.94c0,1.56-1.28,2.83-2.84,2.83h-9.51 c0.01,0.13,0.02,0.25,0.02,0.38V21.74L56.83,21.74L56.83,21.74z M54.82,64.55c2.7,1.45,4.53,4.3,4.53,7.58 c0,4.75-3.85,8.61-8.61,8.61c-4.75,0-8.61-3.85-8.61-8.61c0-3.28,1.84-6.13,4.53-7.58l0-19.72c0-2.25,1.82-4.08,4.07-4.08 c2.25,0,4.08,1.82,4.08,4.08L54.82,64.55L54.82,64.55L54.82,64.55z M96.33,37.07c1.97-4.7,1.74-9.63-1.08-12.92 c-3.38-3.96-9.5-4.41-15.17-1.63C86.08,26.65,91.54,31.45,96.33,37.07L96.33,37.07L96.33,37.07z M5.17,37.07 c-1.97-4.7-1.74-9.63,1.08-12.92c3.38-3.96,9.5-4.41,15.18-1.63C15.41,26.65,9.95,31.45,5.17,37.07L5.17,37.07L5.17,37.07z M80.87,42.01c-7.71-7.71-18.36-12.48-30.12-12.48c-11.76,0-22.41,4.77-30.12,12.48C12.92,49.72,8.15,60.37,8.15,72.13 c0,11.76,4.77,22.42,12.48,30.12s18.36,12.48,30.12,12.48c11.77,0,22.42-4.77,30.12-12.48s12.48-18.36,12.48-30.13 C93.35,60.37,88.58,49.72,80.87,42.01L80.87,42.01L80.87,42.01z"
        fill={colors.bg}
      />
    </Svg>
  );
}

function Check({ size }: { size: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      <SvgPath
        d="M22 52 L42 72 L78 30"
        stroke={colors.bg}
        strokeWidth={17}
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </Svg>
  );
}

function Star({ size, completed }: { size: number; completed?: boolean }) {
  // Grey until the milestone is completed; the background colour once it is.
  // No contrasting border — the same-colour stroke only rounds the star points.
  const c = completed ? colors.bg : "#9297A0";
  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      <SvgPath
        d="M50 12 L58.8 37.9 L86.1 38.3 L64.3 54.6 L72.3 80.7 L50 65 L27.7 80.7 L35.7 54.6 L13.9 38.3 L41.2 37.9 Z"
        fill={c}
        stroke={c}
        strokeWidth={10}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </Svg>
  );
}

function Lock({ size }: { size: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      <SvgPath
        d="M30 45 V32 a20 20 0 0 1 40 0 V45"
        stroke="#9297A0"
        strokeWidth={9}
        fill="none"
        strokeLinecap="round"
      />
      <SvgPath
        d="M24 45 h52 a6 6 0 0 1 6 6 v30 a6 6 0 0 1 -6 6 h-52 a6 6 0 0 1 -6 -6 v-30 a6 6 0 0 1 6 -6 Z"
        fill="#9297A0"
      />
    </Svg>
  );
}

const styles = StyleSheet.create({
  content: {
    alignItems: "center",
    paddingVertical: 24,
  },
  node: {
    alignItems: "center",
    justifyContent: "center",
  },
  startLabel: {
    position: "absolute",
    bottom: -28,
    textAlign: "center",
    fontFamily: fonts.heading,
    fontSize: 17,
    color: colors.darkBlue,
  },
});
