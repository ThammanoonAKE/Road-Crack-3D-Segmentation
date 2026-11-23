import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans_Thai } from "next/font/google";
import { Ambient, GlassFilter } from "@/components/GlassFilter";
import { LangProvider } from "@/components/LangProvider";
import "./globals.css";

const plexThai = IBM_Plex_Sans_Thai({
  weight: ["400", "500", "600", "700"],
  subsets: ["thai", "latin"],
  variable: "--font-plex-thai",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  weight: ["400", "500"],
  subsets: ["latin"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  // Absolute URLs for the Open Graph image; set NEXT_PUBLIC_SITE_URL to the deployed address.
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"),
  title: {
    default: "Road Crack 3D — สำรวจรอยแตกผิวถนนในสามมิติ",
    template: "%s · Road Crack 3D",
  },
  description:
    "แบบจำลอง 3D Gaussian Splatting ของผิวถนนคอนกรีต 8 จุดจากวิดีโอโทรศัพท์มือถือ พร้อมรอยแตกที่ระบุรายเกาเซียนด้วยโมเดล DeepCrack และการตรวจจับแบบเรียลไทม์บน TensorRT",
};

export const viewport: Viewport = {
  themeColor: "#000000",
  colorScheme: "dark",
  viewportFit: "cover",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="th" className={`${plexThai.variable} ${plexMono.variable}`}>
      <body>
        <GlassFilter />
        <Ambient />
        <LangProvider>{children}</LangProvider>
      </body>
    </html>
  );
}
