"use client";

import { useLang } from "./LangProvider";
import { Segmented } from "./Segmented";

export function LangSwitch({ className }: { className?: string }) {
  const { lang, setLang, t } = useLang();
  return (
    <Segmented
      className={className}
      label={t.nav.language}
      value={lang}
      onChange={setLang}
      options={[
        { value: "th", label: "TH", lang: "th" },
        { value: "en", label: "EN", lang: "en" },
      ]}
    />
  );
}
