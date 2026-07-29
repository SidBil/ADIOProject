import React from "react";
import { Image, ImageStyle, StyleProp } from "react-native";

// The app's custom loading animation, used everywhere in place of the default
// React Native ActivityIndicator. Mirrors ActivityIndicator's `size` API
// ("small" | "large" | number) so it's a drop-in replacement. The gif carries
// its own colours, so there is no `color` prop.
const SPINNER = require("../../assets/spinner.gif");

interface Props {
  size?: "small" | "large" | number;
  style?: StyleProp<ImageStyle>;
}

export default function Spinner({ size = "large", style }: Props) {
  const px = typeof size === "number" ? size : size === "small" ? 26 : 48;
  return (
    <Image
      source={SPINNER}
      style={[{ width: px, height: px }, style]}
      resizeMode="contain"
      accessibilityRole="image"
      accessibilityLabel="Loading"
    />
  );
}
