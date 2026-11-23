"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { COPY, type Copy, type Lang } from "@/lib/copy";

interface LangValue {
  lang: Lang;
  t: Copy;
  setLang: (lang: Lang) => void;
}

const LangContext = createContext<LangValue>({ lang: "th", t: COPY.th, setLang: () => {} });
const KEY = "rc3d.lang";

export function LangProvider({ children }: { children: React.ReactNode }) {
  // Server-rendered in Thai; a stored English preference is applied after hydration.
  const [lang, setLangState] = useState<Lang>("th");

  useEffect(() => {
    try {
      const stored = localStorage.getItem(KEY);
      if (stored === "en" || stored === "th") setLangState(stored);
    } catch {
      /* storage unavailable: keep Thai */
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo(() => ({ lang, t: COPY[lang], setLang }), [lang, setLang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export const useLang = () => useContext(LangContext);
