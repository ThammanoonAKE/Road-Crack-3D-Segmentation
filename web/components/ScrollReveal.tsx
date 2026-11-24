"use client";

import { useEffect } from "react";

/**
 * Plays an element's entrance (the `data-reveal` rules in globals.css) when it scrolls into view.
 *   - An element that is below the screen when the page opens (or less than 60 px into it) gets
 *     data-shown="false", which hides it, and data-shown="true" once 60 px of it has entered; that starts its
 *     animation.
 *   - An element already on screen is left alone, so nothing flashes after hydration.
 *   - Without JavaScript, or under reduced motion, nothing is ever hidden.
 * Renders nothing; mounted once on the home page.
 */
export function ScrollReveal() {
  useEffect(() => {
    if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const seen = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.setAttribute("data-shown", "true");
          seen.unobserve(entry.target);
        }
      },
      // a fixed margin, not a share of the screen: the footer never rises far above the bottom edge
      { rootMargin: "0px 0px -60px 0px" },
    );
    for (const el of document.querySelectorAll<HTMLElement>('[data-reveal]:not([data-shown="true"])')) {
      // still waiting from an earlier run (the page was hidden and shown again), or new and not yet 60 px in
      if (el.dataset.shown === "false" || el.getBoundingClientRect().top > window.innerHeight - 60) {
        el.dataset.shown = "false";
        seen.observe(el);
      }
    }
    return () => seen.disconnect();
  }, []);

  return null;
}
