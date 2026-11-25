import type { Metadata } from "next";
import { DetectApp } from "@/components/detect/DetectApp";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { SkipLink } from "@/components/SkipLink";

export const metadata: Metadata = {
  title: "ทดลองตรวจจับรอยแตก",
  description:
    "เลือกภาพผิวถนนแล้วให้โมเดล DeepCrack ตรวจจับรอยแตกในเบราว์เซอร์ของคุณ ภาพไม่ถูกส่งออกจากเครื่องและไม่ถูกเก็บไว้",
};

export default function Page() {
  return (
    <>
      <SkipLink />
      <SiteHeader />
      <main id="main">
        <DetectApp />
      </main>
      <SiteFooter />
    </>
  );
}
