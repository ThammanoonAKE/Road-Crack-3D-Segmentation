"use client";

import { useEffect } from "react";
import { metered, warmDetector } from "@/lib/detector";

/**
 * Fetches the /detect page's model in the background once the home page has loaded, so that a visitor who reaches
 * the last section can try the detector without waiting for 110 MB. Skipped under data saver and on slow or mobile
 * networks; the download goes on if the visitor opens the viewer meanwhile. Renders nothing.
 */
export function DetectorWarmup() {
  useEffect(() => {
    if (metered()) return;
    let timer = 0;
    // after the page's own files: the hero's point cloud and the pictures come first
    const start = () => {
      timer = window.setTimeout(warmDetector, 2500);
    };
    if (document.readyState === "complete") start();
    else window.addEventListener("load", start, { once: true });
    return () => {
      window.removeEventListener("load", start);
      window.clearTimeout(timer);
    };
  }, []);
  return null;
}
