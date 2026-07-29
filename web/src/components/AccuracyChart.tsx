import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Svg, { Path, Circle, Line } from "react-native-svg";
import { colors, fonts } from "../theme";

interface Props {
  // One point per session, oldest -> newest. null = the structure word was
  // never answered that session -> a gap in the line, not a zero.
  values: (number | null)[];
  width: number;
  height?: number;
  min?: number;
  max?: number;
  color?: string;
}

/**
 * A focused accuracy trend chart with a real 0–5 Y-axis and per-session X-axis.
 *
 * Used one-at-a-time inside the Progress-tab accordion so each structure word
 * gets a full, legible graph instead of a cramped, scale-less sparkline. Nulls
 * break the line into separate segments so an unanswered session reads as
 * missing data rather than a drop to zero.
 */
export default function AccuracyChart({
  values,
  width,
  height = 200,
  min = 0,
  max = 5,
  color = colors.blueBorder,
}: Props) {
  const AXIS_W = 26; // left gutter for Y labels
  const AXIS_H = 20; // bottom gutter for X labels
  const padTop = 14;
  const plotX = AXIS_W;
  const plotW = Math.max(1, width - AXIS_W - 6);
  const plotTop = padTop;
  const plotH = Math.max(1, height - padTop - AXIS_H);

  const n = values.length;
  const stepX = n > 1 ? plotW / (n - 1) : 0;

  const toXY = (v: number, i: number) => {
    const x = n > 1 ? plotX + i * stepX : plotX + plotW / 2;
    const clamped = Math.max(min, Math.min(max, v));
    const y = plotTop + plotH * (1 - (clamped - min) / (max - min || 1));
    return { x, y };
  };

  // Contiguous non-null segments.
  const segments: { x: number; y: number; v: number }[][] = [];
  let cur: { x: number; y: number; v: number }[] = [];
  values.forEach((v, i) => {
    if (v == null) {
      if (cur.length) segments.push(cur);
      cur = [];
    } else {
      cur.push({ ...toXY(v, i), v });
    }
  });
  if (cur.length) segments.push(cur);
  const points = segments.flat();

  // Y gridlines / labels at each integer 0..5.
  const ticks: number[] = [];
  for (let t = min; t <= max; t++) ticks.push(t);

  const hasData = points.length > 0;

  return (
    <View style={{ width }}>
      <Svg width={width} height={height}>
        {/* horizontal gridlines + Y labels */}
        {ticks.map((t) => {
          const y = plotTop + plotH * (1 - (t - min) / (max - min || 1));
          return (
            <React.Fragment key={t}>
              <Line
                x1={plotX}
                y1={y}
                x2={plotX + plotW}
                y2={y}
                stroke={t === min ? "#c9c9d6" : "#ececf2"}
                strokeWidth={1}
              />
            </React.Fragment>
          );
        })}
        {/* line segments */}
        {segments.map((seg, si) => {
          if (seg.length === 1) return null; // lone points shown as dots
          const d = seg
            .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
            .join(" ");
          return (
            <Path
              key={si}
              d={d}
              stroke={color}
              strokeWidth={3}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          );
        })}
        {/* points */}
        {points.map((p, i) => (
          <Circle key={i} cx={p.x} cy={p.y} r={4} fill={color} />
        ))}
      </Svg>

      {/* Y-axis labels (absolute so they sit on the gridlines) */}
      {ticks.map((t) => {
        const y = plotTop + plotH * (1 - (t - min) / (max - min || 1));
        return (
          <Text key={t} style={[styles.yLabel, { top: y - 8 }]}>
            {t}
          </Text>
        );
      })}

      {/* per-point value labels */}
      {points.map((p, i) => (
        <Text
          key={i}
          style={[styles.valLabel, { left: p.x - 12, top: p.y - 22, color }]}
        >
          {p.v}
        </Text>
      ))}

      {/* X-axis caption */}
      <View style={styles.xCaption}>
        <Text style={styles.xText}>{hasData ? "Older" : ""}</Text>
        <Text style={styles.xText}>{hasData ? "Newer →" : "No data yet"}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  yLabel: {
    position: "absolute",
    left: 0,
    width: 20,
    textAlign: "right",
    fontFamily: fonts.body,
    fontSize: 11,
    color: colors.textMuted,
  },
  valLabel: {
    position: "absolute",
    width: 24,
    textAlign: "center",
    fontFamily: fonts.bodySemiBold,
    fontSize: 11,
  },
  xCaption: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingLeft: 26,
    marginTop: 2,
  },
  xText: {
    fontFamily: fonts.body,
    fontSize: 11,
    color: colors.textMuted,
  },
});
