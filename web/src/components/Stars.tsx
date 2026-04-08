import React from "react";
import { View } from "react-native";
import Svg, { Path } from "react-native-svg";
import { colors } from "../theme";

const STAR_D =
  "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z";

export default function Stars({ score, size = 32 }: { score: number; size?: number }) {
  return (
    <View style={{ flexDirection: "row", justifyContent: "center", gap: 6 }}>
      {[1, 2, 3, 4, 5].map((i) => {
        const fill = i <= score ? colors.starPink : colors.starDark;
        return (
          <Svg key={i} viewBox="0 0 24 24" width={size} height={size}>
            <Path
              d={STAR_D}
              fill={fill}
              stroke={fill}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </Svg>
        );
      })}
    </View>
  );
}
