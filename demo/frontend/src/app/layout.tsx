import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wild Palm Verification Demo",
  description: "Web demo for browsing wild palm LVM verification experiment outputs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full overflow-hidden">{children}</body>
    </html>
  );
}
