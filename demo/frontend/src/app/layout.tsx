import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";

import "./globals.css";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-wp-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-wp-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Wild Palm — Detection Verification",
  description:
    "Inspection workstation for precomputed wild palm YOLO detections and VLM verification results.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`h-full ${plexSans.variable} ${plexMono.variable}`}>
      <body className={`${plexSans.className} h-full overflow-hidden`}>{children}</body>
    </html>
  );
}
