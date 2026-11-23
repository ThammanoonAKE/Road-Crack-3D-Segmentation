"use client";

import Link from "next/link";
import { useLang } from "@/components/LangProvider";
import ui from "@/components/ui.module.css";

export default function NotFound() {
  const { t } = useLang();
  return (
    <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: "var(--gutter)" }}>
      <div className="lg" style={{ display: "grid", gap: "var(--s-4)", maxWidth: 440, padding: "var(--s-6)", borderRadius: "var(--r-xl)" }}>
        <p className={ui.kicker}>404</p>
        <h1 className={ui.title}>{t.notFound.title}</h1>
        <p className={ui.lead}>{t.notFound.body}</p>
        <Link href="/" className={ui.primary} style={{ justifySelf: "start" }}>
          {t.notFound.home}
        </Link>
      </div>
    </main>
  );
}
