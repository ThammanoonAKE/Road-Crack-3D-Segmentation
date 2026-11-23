import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // The reconstructions are 80–110 MB binary files that barely compress; gzipping them per request would only
  // cost CPU. Put a CDN or reverse proxy in front to compress the text assets instead.
  compress: false,
  async headers() {
    const day = "public, max-age=86400, stale-while-revalidate=604800";
    return [
      { source: "/splats/:path*", headers: [{ key: "Cache-Control", value: day }] },
      { source: "/media/:path*", headers: [{ key: "Cache-Control", value: day }] },
    ];
  },
};

export default nextConfig;
