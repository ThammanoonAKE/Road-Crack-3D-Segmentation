import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // The reconstructions are 80–110 MB binary files that barely compress; gzipping them per request would only
  // cost CPU. Put a CDN or reverse proxy in front to compress the text assets instead.
  compress: false,
  async headers() {
    const day = "public, max-age=86400, stale-while-revalidate=604800";
    // Cross-origin isolation lets the detector of /detect use several threads where there is no WebGPU. The page,
    // its worker and the runtime's thread script each need the headers. The page loads nothing from other sites
    // except the model, which is fetched with CORS.
    const isolated = [
      { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
      { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
    ];
    return [
      { source: "/splats/:path*", headers: [{ key: "Cache-Control", value: day }] },
      { source: "/media/:path*", headers: [{ key: "Cache-Control", value: day }] },
      // the detector of /detect: its runtime (30 MB) and, when run locally, the model
      { source: "/ort/:path*", headers: [{ key: "Cache-Control", value: day }, ...isolated] },
      { source: "/models/:path*", headers: [{ key: "Cache-Control", value: day }] },
      { source: "/detect", headers: isolated },
      { source: "/detect/:path*", headers: isolated },
    ];
  },
};

export default nextConfig;
