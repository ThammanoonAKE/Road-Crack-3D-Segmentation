import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import { ViewerApp } from "@/components/viewer/ViewerApp";
import { SECTIONS, getSection } from "@/lib/sections";

export function generateStaticParams() {
  return SECTIONS.map((s) => ({ section: String(s.n) }));
}

export const metadata: Metadata = {
  title: "ตัวแสดงแบบจำลอง 3D",
};

export default function Page({ params }: PageProps<"/viewer/[section]">) {
  // The URL-dependent part sits inside Suspense, as cacheComponents with partial prefetching expects.
  return (
    <Suspense fallback={<div style={{ position: "fixed", inset: 0, background: "#050506" }} />}>
      <Section params={params} />
    </Suspense>
  );
}

async function Section({ params }: Pick<PageProps<"/viewer/[section]">, "params">) {
  const { section } = await params;
  const n = Number(section);
  if (!Number.isInteger(n) || !getSection(n)) notFound();
  return <ViewerApp key={n} n={n} />;
}
