import type { Lang } from "@/lib/i18n";

const FORMATTERS: Record<Lang, { integer: Intl.NumberFormat; decimal: Intl.NumberFormat }> = {
  vi: {
    integer: new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 }),
    decimal: new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 }),
  },
  en: {
    integer: new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }),
    decimal: new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }),
  },
};

export function formatNumber(value: number | null | undefined, lang: Lang = "vi"): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const fmt = FORMATTERS[lang];
  return Number.isInteger(value) ? fmt.integer.format(value) : fmt.decimal.format(value);
}

export function formatMoney(value: number | null | undefined, currency?: string | null, lang: Lang = "vi"): string {
  if (value === null || value === undefined) return "—";
  const unit = currency === "VND" || !currency ? "VND" : currency;
  return `${FORMATTERS[lang].integer.format(value)} ${unit}`;
}

export function formatShortMoney(value: number, lang: Lang = "vi"): string {
  const fmt = FORMATTERS[lang];
  if (value >= 1e6) {
    const scaled = fmt.decimal.format(Math.round(value / 1e4) / 100);
    return lang === "en" ? `${scaled}M` : `${scaled} tr`;
  }
  if (value >= 1e3) {
    const scaled = fmt.integer.format(Math.round(value / 1e3));
    return `${scaled}K`;
  }
  return fmt.integer.format(value);
}

export function formatMs(value: number | null | undefined, lang: Lang = "vi"): string {
  if (value === null || value === undefined) return "—";
  const fmt = FORMATTERS[lang];
  return value >= 1000 ? `${fmt.decimal.format(value / 1000)} s` : `${fmt.integer.format(value)} ms`;
}

export function formatSpeedup(value: number | null | undefined, lang: Lang = "vi"): string {
  if (value === null || value === undefined) return "—";
  return `${FORMATTERS[lang].decimal.format(value)}×`;
}

export function formatCell(value: unknown, lang: Lang = "vi"): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return formatNumber(value, lang);
  return String(value);
}

export function localeTag(lang: Lang): string {
  return lang === "en" ? "en-US" : "vi-VN";
}
