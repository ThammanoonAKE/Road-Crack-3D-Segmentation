"use client";

import { useLang } from "./LangProvider";

export function SkipLink({ target = "main" }: { target?: string }) {
  const { t } = useLang();
  return (
    <a className="skip-link" href={`#${target}`}>
      {t.nav.skip}
    </a>
  );
}
