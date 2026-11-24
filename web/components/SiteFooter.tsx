"use client";

import { External, Mark } from "./Icons";
import { useLang } from "./LangProvider";
import ui from "./ui.module.css";
import styles from "./SiteFooter.module.css";

const REPO = process.env.NEXT_PUBLIC_REPO_URL;

export function SiteFooter() {
  const { t } = useLang();
  return (
    <footer className={styles.footer}>
      <div className={ui.wrap}>
        <div className={`${styles.inner} lg`} data-reveal="up">
          <div className={styles.brand}>
            <Mark className={styles.mark} />
            <span>Road Crack 3D</span>
          </div>
          <p className={styles.licence}>{t.footer.licence}</p>
          <div className={styles.meta}>
            {REPO && (
              <a href={REPO} className={styles.link} target="_blank" rel="noreferrer">
                {t.footer.source}
                <External />
              </a>
            )}
            <span className={styles.built}>{t.footer.built}</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
