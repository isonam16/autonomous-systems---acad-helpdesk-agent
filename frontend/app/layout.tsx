import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "IIT Jodhpur Academic Helpdesk", description: "AI-powered academic helpdesk for IIT Jodhpur students." };
export default function RootLayout({children}:{children:React.ReactNode}) { return <html lang="en"><body>{children}</body></html>; }