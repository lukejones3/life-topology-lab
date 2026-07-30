import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Life Topology Lab — AI Evaluation Demo",
  description:
    "A privacy-preserving demonstration of longitudinal retrieval, evaluation, topology, and forecast validation.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
