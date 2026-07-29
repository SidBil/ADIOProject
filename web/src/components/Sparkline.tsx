import React from "react";
import { View } from "react-native";
import Svg, { Path, Circle, Line } from "react-native-svg";
import { colors } from "../theme";

interface Props {
  // One point per session, oldest -> newest. null = structure word never
  // answered that session -> rendered as a gap in the line, not a zero.
  values: (number | null)[];
  width: number;
  height: number;
  min?: number;
  max?: number;
  color?: string;
}

/* A tiny accuracy trend line. Nulls break the line into separate segments so
   an unanswered session reads as missing data rather than a drop to zero. */
export default function Sparkline({
  values,
  width,
  height,
  min = 0,
  max = 5,
  color = colors.blueBorder,
}: Props) {
  const padY = 4;
  const usableH = Math.max(1, height - padY * 2);
  const n = values.length;
  const stepX = n > 1 ? width / (n - 1) : 0;

  const xy = (v: number, i: number) => {
    const x = n > 1 ? i * stepX : width / 2;
    const clamped = Math.max(min, Math.min(max, v));
    const y = padY + usableH * (1 - (clamped - min) / (max - min || 1));
    return { x, y };
  };

  // Split into contiguous segments of non-null points.
  const segments: { x: number; y: number }[][] = [];
  let current: { x: number; y: number }[] = [];
  values.forEach((v, i) => {
    if (v == null) {
      if (current.length) segments.push(current);
      current = [];
    } else {
      current.push(xy(v, i));
    }
  });
  if (current.length) segments.push(current);

  const points = segments.flat();

  return (
    <View style={{ width, height }}>
      <Svg width={width} height={height}>
        {/* subtle mid gridline (accuracy 2.5 / 5) */}
        <Line
          x1={0}
          y1={padY + usableH / 2}
          x2={width}
          y2={padY + usableH / 2}
          stroke="#e6e6ef"
          strokeWidth={1}
        />
        {segments.map((seg, si) => {
          if (seg.length === 1) return null; // lone points drawn as dots below
          const d = seg
            .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
            .join(" ");
          return (
            <Path
              key={si}
              d={d}
              stroke={color}
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          );
        })}
        {points.map((p, i) => (
          <Circle key={i} cx={p.x} cy={p.y} r={2.6} fill={color} />
        ))}
      </Svg>
    </View>
  );
}
