/**
 * Data of Coding Skills ▸ Chips and ▸ Mixed precision, merged from sourced research by scripts/chips_data.py:
 * chips (specs with a source URL per number), the library × hardware-family support matrix, what runs underneath
 * each family, the timeline, number formats and mixed-precision practice. Facts (names, numbers, device strings,
 * URLs) are language-neutral; the explanatory sentences have translations in i18n/<lang>.json, keyed as listed
 * below, falling back to English.
 */
import { useSyncExternalStore } from "react";
import { useLang } from "../../i18n";
import chipsJson from "./chips.json";
import flagsJson from "./flags.json";
import formatsJson from "./formats.json";
import historyJson from "./history.json";
import librariesJson from "./libraries.json";

export type ChipClass = "igpu" | "dgpu" | "soc" | "accelerator" | "cpu" | "board" | "npu";
export interface Chip {
  id: string;
  vendor: string;
  name: string;
  nameZh?: string | null;
  group: "global" | "china";
  class: ChipClass;
  year: number | null;
  memoryType: string | null;
  memoryGB: number | number[] | null;
  bandwidthGBs: number | null;
  fp64TF: number | null;
  fp64TensorTF?: number | null;
  fp32TF: number | null;
  tf32TF?: number | null;
  bf16TF: number | null;
  fp16TF: number | null;
  fp8TF: number | null;
  fp6TF?: number | null;
  fp4TF?: number | null;
  int8TOPS: number | null;
  npuTOPS: number | null;
  powerW: number | number[] | null;
  matrixUnit: string | null;
  formats: string[];
  pytorchDevice: string | null;
  blas: string;
  kernelLangs: string;
  interconnect?: string | null;
  notes: string;
  confidence: string;
  sources: Record<string, string>;
  /** fields computed from vendor inputs (cores × clock × operations per clock, or a vendor ratio); the notes say how */
  derived?: string[];
}

export type Level = "official" | "preview" | "community" | "none";
export interface Cell {
  level: Level;
  device?: string;
  notes?: string;
  source?: string;
  version?: string;
}
export interface Library {
  id: string;
  name: string;
  cells: Record<string, Cell>;
}
export interface Under {
  family: string;
  blas: string[];
  kernelLangs: string[];
  compilers: string[];
  bestFor: string[];
  avoidFor: string[];
  sources: string[];
}
export interface TimelineEvent {
  id: string;
  year: number;
  month?: number;
  title: string;
  text: string;
  vendor?: string;
  kind: "library" | "chip" | "standard" | "policy";
  source: string;
}
export interface FormatInfo {
  id: string;
  name: string;
  bits: number;
  sign: number;
  exponent: number;
  mantissa: number;
  bias?: number;
  maxNormal: number;
  minNormal: number;
  minSubnormal?: number | null;
  machineEpsilon?: number | null;
  decimalDigits?: number | null;
  notes: string;
  standard: string;
  source: string;
}

export const CHIPS = (chipsJson as unknown as { chips: Chip[] }).chips;
export interface Stack {
  vendor: string;
  sdk: string | null;
  compiler: string | null;
  kernelLanguage: string | null;
  mathLibraries: string | null;
  collectives: string | null;
  pytorch: string | null;
  pytorchRepo: string | null;
  frameworks: string | null;
  opensource: string | null;
  opensourceLink: string | null;
  source: string | null;
}
export const STACKS = (chipsJson as unknown as { stacks: Stack[] }).stacks;
const libs = librariesJson as unknown as { libraries: Library[]; under: Under[]; precisionDefaults: { text: string; source: string }[]; checked: string };
export const LIBRARIES = libs.libraries;
export const UNDER = libs.under;
export const PRECISION_DEFAULTS = libs.precisionDefaults;
export const CHECKED = libs.checked;
const hist = historyJson as unknown as { timeline: TimelineEvent[]; facts: { id: string; text: string; source: string }[] };
export const TIMELINE = [...hist.timeline].sort((a, b) => a.year - b.year || (a.month ?? 0) - (b.month ?? 0));
export const FACTS = hist.facts;
export const FORMAT_INFO = formatsJson as unknown as FormatInfo[];

/** the switches each package uses to choose precision, each checked on its official page (flag.<i> translated) */
export interface PrecisionFlag {
  package: string;
  kind: "python" | "env" | "config" | "c-api";
  flag: string;
  what: string;
  default: string | null;
  scope: string;
  source: string;
  status: "current" | "deprecated" | "preview";
}
export const FLAGS = (flagsJson as unknown as { flags: PrecisionFlag[] }).flags;

/** the hardware families of the library table, in column order */
export const FAMILIES = [
  "nvidia-gpu",
  "amd-gpu",
  "intel-gpu",
  "x86-cpu",
  "arm-cpu",
  "apple-gpu",
  "apple-ane",
  "qualcomm",
  "google-tpu",
  "aws-trainium",
  "huawei-ascend",
  "other-china",
  "edge-npu",
] as const;
export type Family = (typeof FAMILIES)[number];

/** which library-table column a chip belongs to (null: none fits, e.g. Gaudi) */
export function familyOf(c: Chip): Family | null {
  const v = c.vendor.toLowerCase();
  const n = c.name.toLowerCase();
  if (c.group === "china") return v.includes("huawei") ? "huawei-ascend" : "other-china";
  if (n.includes("grace") && c.class === "cpu") return "arm-cpu";
  if (v.includes("nvidia")) return "nvidia-gpu";
  if (v.includes("amd")) return c.class === "cpu" ? "x86-cpu" : "amd-gpu";
  if (v.includes("intel")) return n.includes("gaudi") ? null : c.class === "cpu" ? "x86-cpu" : "intel-gpu";
  if (v.includes("apple")) return "apple-gpu";
  if (v.includes("qualcomm")) return "qualcomm";
  if (v.includes("google")) return n.includes("coral") || n.includes("edge tpu") ? "edge-npu" : "google-tpu";
  if (v.includes("aws") || v.includes("amazon")) return n.includes("graviton") ? "arm-cpu" : "aws-trainium";
  if (n.includes("hailo") || n.includes("ai hat") || n.includes("coral")) return "edge-npu";
  if (v.includes("raspberry") || n.includes("rk3588") || v.includes("rockchip") || v.includes("radxa")) return "arm-cpu";
  return null;
}

/** the tensor rate of a chip in a precision (TFLOP/s; TOPS for INT8), or null when unpublished */
export type Precision = "fp64" | "fp32" | "tf32" | "bf16" | "fp16" | "fp8" | "fp4" | "int8";
export function rate(c: Chip, p: Precision): number | null {
  const v = {
    fp64: c.fp64TensorTF ?? c.fp64TF,
    fp32: c.fp32TF,
    tf32: c.tf32TF ?? null,
    bf16: c.bf16TF ?? c.fp16TF,
    fp16: c.fp16TF ?? c.bf16TF,
    fp8: c.fp8TF,
    fp4: c.fp4TF ?? null,
    int8: c.int8TOPS,
  }[p];
  return typeof v === "number" ? v : null;
}

const overlays = import.meta.glob<Record<string, string>>("./i18n/*.json", { eager: true, import: "default" });

/** "show the sources' English original" switch, shared by every data table and remembered in the browser */
const ORIGINAL_KEY = "tensorizer.chipsOriginal";
let original = (() => {
  try {
    return localStorage.getItem(ORIGINAL_KEY) === "1";
  } catch {
    return false;
  }
})();
const listeners = new Set<() => void>();
export function setShowOriginal(on: boolean) {
  original = on;
  try {
    localStorage.setItem(ORIGINAL_KEY, on ? "1" : "0");
  } catch {
    /* private window: keep it for this visit only */
  }
  listeners.forEach((l) => l());
}
export function useShowOriginal(): boolean {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => original,
  );
}

/** translated explanatory text of the data: t(key, english); English when the switch asks for the original */
export function useDataText(): (key: string, english: string) => string {
  const { lang } = useLang();
  const showOriginal = useShowOriginal();
  const tr = showOriginal ? {} : (overlays[`./i18n/${lang}.json`] ?? {});
  return (key, english) => tr[key] || english;
}

/** whether a chip's figure in a precision is derived rather than printed by the vendor */
export const isDerived = (c: Chip, p: Precision) => (c.derived ?? []).some((k) => k.startsWith(p === "int8" ? "int8" : p));
