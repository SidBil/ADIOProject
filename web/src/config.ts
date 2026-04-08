import { Platform } from "react-native";

const LOCAL_IP = "localhost";
const PORT = 8000;

const DEV_BASE =
  Platform.OS === "android"
    ? `http://10.0.2.2:${PORT}`
    : `http://${LOCAL_IP}:${PORT}`;

export const API_BASE = __DEV__ ? DEV_BASE : "";
