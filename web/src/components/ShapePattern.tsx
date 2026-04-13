import React, { useState, useRef, useEffect, useCallback } from "react";
import { View, LayoutChangeEvent } from "react-native";
import Svg, { Circle, Polygon, Path } from "react-native-svg";

const COLORS = ["#EF1A6A", "#fbde28", "#29A5E1", "#BFD62C"];
const TILE = 65;
const SIZE = TILE * 0.5;

const WAVE_SPEED = 250;
const WAVELENGTH = 190;
const MAX_PUSH = 20;
const DECAY_DIST = 520;
const TWO_PI = 2 * Math.PI;

const BURST_SPEED = 680;
const BURST_PUSH = 38;
const BURST_WIDTH = 180;
const BURST_LIFE = 1.3;

const SWIRL_SPEED = 1.2;
const SWIRL_PUSH = 14;
const SWIRL_SCALE_AMP = 0.15;

type ST = "circle" | "triangle" | "star";
const ORDER: ST[] = ["circle", "star", "triangle", "star", "circle", "triangle"];

function st(r: number, c: number): ST { return ORDER[(r * 3 + c) % ORDER.length]; }
function cl(r: number, c: number) { return COLORS[(r * 3 + c) % COLORS.length]; }
function rt(r: number, c: number) { return `${((r + c) * 25) % 360}deg`; }

function S({ type, fill }: { type: ST; fill: string }) {
  const h = SIZE / 2;
  if (type === "circle")
    return <Svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}><Circle cx={h} cy={h} r={h * 0.85} fill={fill} /></Svg>;
  if (type === "triangle")
    return <Svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}><Polygon points={`${h},${SIZE * 0.08} ${SIZE * 0.92},${SIZE * 0.88} ${SIZE * 0.08},${SIZE * 0.88}`} fill={fill} /></Svg>;
  return <Svg width={SIZE} height={SIZE} viewBox="0 0 24 24"><Path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill={fill} /></Svg>;
}

interface BurstRing {
  radius: number;
  age: number;
}

interface Props {
  volume?: number;
  burst?: number;
  cardCenter?: { x: number; y: number };
  swirl?: boolean;
}

export default function ShapePattern({ volume = 0, burst = 0, cardCenter, swirl = false }: Props) {
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const phaseRef = useRef(0);
  const swirlPhaseRef = useRef(0);
  const smoothVolRef = useRef(0);
  const lastFrameRef = useRef(0);
  const animatingRef = useRef(false);
  const volumeRef = useRef(0);
  const swirlRef = useRef(false);
  const burstsRef = useRef<BurstRing[]>([]);
  const lastBurstRef = useRef(0);
  const [, forceRender] = useState(0);

  volumeRef.current = volume;
  swirlRef.current = swirl;

  useEffect(() => {
    if (burst > lastBurstRef.current) {
      lastBurstRef.current = burst;
      burstsRef.current.push({ radius: 0, age: 0 });
      if (!animatingRef.current) {
        animatingRef.current = true;
        lastFrameRef.current = performance.now();
        requestAnimationFrame(loop);
      }
    }
  }, [burst]);

  const loop = useCallback(() => {
    const now = performance.now();
    const dt = (now - lastFrameRef.current) / 1000;
    lastFrameRef.current = now;

    const v = volumeRef.current;
    if (v > smoothVolRef.current) {
      smoothVolRef.current = v;
    } else {
      smoothVolRef.current *= 0.94;
    }

    phaseRef.current += dt * WAVE_SPEED;
    swirlPhaseRef.current += dt * SWIRL_SPEED;

    burstsRef.current = burstsRef.current
      .map((b) => ({ radius: b.radius + dt * BURST_SPEED, age: b.age + dt }))
      .filter((b) => b.age < BURST_LIFE);

    const hasVolume = smoothVolRef.current > 0.003;
    const hasBursts = burstsRef.current.length > 0;
    const hasSwirl = swirlRef.current;

    if (!hasVolume && !hasBursts && !hasSwirl) {
      smoothVolRef.current = 0;
      animatingRef.current = false;
      forceRender((n) => n + 1);
      return;
    }

    forceRender((n) => n + 1);
    requestAnimationFrame(loop);
  }, []);

  useEffect(() => {
    if ((volume > 0.01 || swirl) && !animatingRef.current) {
      animatingRef.current = true;
      lastFrameRef.current = performance.now();
      requestAnimationFrame(loop);
    }
  }, [volume, swirl, loop]);

  useEffect(() => {
    return () => { animatingRef.current = false; };
  }, []);

  const onLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setDims({ w: width, h: height });
  };

  const cols = dims.w ? Math.ceil(dims.w / TILE) : 0;
  const rows = dims.h ? Math.ceil(dims.h / TILE) : 0;
  const center = cardCenter || { x: dims.w * 0.5, y: dims.h * 0.5 };
  const vol = animatingRef.current ? smoothVolRef.current : 0;
  const phase = phaseRef.current;
  const swirlPhase = swirlPhaseRef.current;
  const bursts = burstsRef.current;
  const isSwirling = swirl;

  return (
    <View
      onLayout={onLayout}
      pointerEvents="none"
      style={{
        position: "absolute",
        top: 0, left: 0, right: 0, bottom: 0,
        opacity: 0.25,
        maxWidth: dims.w || "100%",
        maxHeight: dims.h || "100%",
      }}
    >
      {cols > 0 && rows > 0 &&
        Array.from({ length: rows }, (_, r) => {
          const offset = r % 2 === 1 ? TILE / 2 : 0;
          const rowCols = offset > 0 ? cols - 1 : cols;
          return (
            <View
              key={r}
              style={{ flexDirection: "row", height: TILE, paddingLeft: offset }}
            >
              {Array.from({ length: rowCols }, (_, c) => {
                const tileX = offset + c * TILE + TILE / 2;
                const tileY = r * TILE + TILE / 2;
                const dx = tileX - center.x;
                const dy = tileY - center.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const angle = dist > 0.1 ? Math.atan2(dy, dx) : 0;

                const waveArg = (dist / WAVELENGTH) * TWO_PI - (phase / WAVELENGTH) * TWO_PI;
                const wave = Math.sin(waveArg);
                const atten = Math.exp(-dist / DECAY_DIST);
                let pushX = vol * MAX_PUSH * wave * atten * Math.cos(angle);
                let pushY = vol * MAX_PUSH * wave * atten * Math.sin(angle);
                let sc = 1 + vol * atten * Math.max(0, wave) * 0.3;

                for (const b of bursts) {
                  const fade = 1 - b.age / BURST_LIFE;
                  const ringDist = Math.abs(dist - b.radius);
                  const inRing = Math.max(0, 1 - ringDist / BURST_WIDTH);
                  const push = BURST_PUSH * inRing * fade * fade;
                  pushX += Math.cos(angle) * push;
                  pushY += Math.sin(angle) * push;
                  sc += inRing * fade * 0.45;
                }

                // Swirl: orbit shapes around the center
                if (isSwirling && dist > 5) {
                  const swirlAngle = swirlPhase + dist * 0.008;
                  const distFade = Math.exp(-dist / 500);
                  const tangentX = -Math.sin(angle);
                  const tangentY = Math.cos(angle);
                  const swirlAmt = SWIRL_PUSH * Math.sin(swirlAngle) * distFade;
                  pushX += tangentX * swirlAmt;
                  pushY += tangentY * swirlAmt;
                  // Gentle radial breathing
                  const breathe = Math.sin(swirlPhase * 2 + dist * 0.01) * SWIRL_PUSH * 0.3 * distFade;
                  pushX += Math.cos(angle) * breathe;
                  pushY += Math.sin(angle) * breathe;
                  sc += SWIRL_SCALE_AMP * Math.sin(swirlAngle * 1.5) * distFade;
                }

                return (
                  <View
                    key={c}
                    style={{
                      width: TILE,
                      height: TILE,
                      alignItems: "center",
                      justifyContent: "center",
                      transform: [
                        { translateX: pushX },
                        { translateY: pushY },
                        { rotate: rt(r, c) },
                        { scale: sc },
                      ],
                    }}
                  >
                    <S type={st(r, c)} fill={cl(r, c)} />
                  </View>
                );
              })}
            </View>
          );
        })}
    </View>
  );
}
