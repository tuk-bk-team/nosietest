import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "이미지 보호 실험실",
  description: "이미지 인식 회피, 학습 방해, 생성 편집 방해 실험 결과를 비교합니다.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ko" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
