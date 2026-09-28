import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "딥페이크 방지 노이즈 실험 데모",
  description: "사진에 여러 노이즈를 적용해보고 결과를 비교하는 데모 페이지",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ko" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
