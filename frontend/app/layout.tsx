import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hallucination-Resistant NL2SQL",
  description: "Multi-agent natural language analytics with verified SQL",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
