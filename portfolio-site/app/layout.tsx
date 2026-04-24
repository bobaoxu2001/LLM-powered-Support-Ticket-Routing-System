import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "AI-powered Support Operations | Allen Xu",
  description:
    "LLM-powered Support Ticket Routing System — a gDATA-style case routing system combining rules, calibrated ML, LLM fallback, and human triage to make support operations measurable, cost-aware, and controllable.",
  keywords: [
    "machine learning",
    "NLP",
    "support operations",
    "LLM",
    "routing system",
    "data science",
  ],
  authors: [{ name: "Allen Xu" }],
  openGraph: {
    title: "AI-powered Support Operations Optimization",
    description:
      "A gDATA-style case routing system combining rules, calibrated ML, LLM fallback, and human triage.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} bg-[#080c1a] text-white`}>
        {children}
      </body>
    </html>
  );
}
