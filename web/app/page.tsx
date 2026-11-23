import { EdgeAI } from "@/components/EdgeAI";
import { Hero } from "@/components/Hero";
import { Method } from "@/components/Method";
import { ModelSection } from "@/components/ModelSection";
import { SectionGrid } from "@/components/SectionGrid";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { SkipLink } from "@/components/SkipLink";

export default function Home() {
  return (
    <>
      <SkipLink />
      <SiteHeader />
      <main id="main">
        <Hero />
        <SectionGrid />
        <Method />
        <EdgeAI />
        <ModelSection />
      </main>
      <SiteFooter />
    </>
  );
}
