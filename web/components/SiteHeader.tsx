"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Cube, Mark } from "./Icons";
import { useLang } from "./LangProvider";
import { LangSwitch } from "./LangSwitch";
import styles from "./SiteHeader.module.css";

export function SiteHeader() {
  const { t } = useLang();
  const [scrolled, setScrolled] = useState(false);

  // The bar tightens once the page scrolls, as iOS bars do.
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 24);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  return (
    <header className={styles.header} data-scrolled={scrolled}>
      <div className={`${styles.bar} lg`}>
        <Link href="/" className={styles.brand} aria-label={t.nav.home}>
          <Mark className={styles.mark} />
          <span className={styles.name}>Road Crack 3D</span>
        </Link>
        <nav className={styles.nav} aria-label={t.nav.primary}>
          <a href="#sections">{t.nav.sections}</a>
          <a href="#method">{t.nav.method}</a>
          <a href="#edge">{t.nav.edge}</a>
          <a href="#model">{t.nav.model}</a>
        </nav>
        <LangSwitch className={styles.lang} />
        <Link href="/viewer/2" className={styles.cta}>
          <Cube />
          <span>{t.nav.viewer}</span>
        </Link>
      </div>
    </header>
  );
}
