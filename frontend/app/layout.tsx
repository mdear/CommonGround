import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Suspense } from 'react';
import LoadingSpinner from '@/components/layout/LoadingSpinner';
import { AppSidebar } from "@/components/layout/AppSidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { SessionProvider } from "@/components/providers/SessionProvider";
import { AppProvider } from "@/components/providers/AppProvider";
import { Toaster } from "@/components/ui/sonner";

const montserrat = localFont({
  src: [
    {
      path: './fonts/Montserrat-Light.woff2',
      weight: '300',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Light.woff',
      weight: '300',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Regular.woff2',
      weight: '400',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Regular.woff',
      weight: '400',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Medium.woff2',
      weight: '500',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Medium.woff',
      weight: '500',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-SemiBold.woff2',
      weight: '600',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-SemiBold.woff',
      weight: '600',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Bold.woff2',
      weight: '700',
      style: 'normal',
    },
    {
      path: './fonts/Montserrat-Bold.woff',
      weight: '700',
      style: 'normal',
    },
  ],
  variable: '--font-montserrat',
});

export const metadata: Metadata = {
  title: {
    template: '%s | Common Ground',
    default: 'Common Ground',
  },
  description: "Common Ground System",
  keywords: ["common ground", "AI", "machine learning"],
  openGraph: {
    title: 'Common Ground',
    description: 'Common Ground System',
    url: '/',
    siteName: 'Common Ground',
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Common Ground',
    description: 'Common Ground System',
  },
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${montserrat.variable} h-full overflow-hidden`}>
        <Suspense
          fallback={
            <div className="min-h-screen flex items-center justify-center bg-background">
              <LoadingSpinner />
            </div>
          }
        >
          <SessionProvider>
            <AppProvider>
              <SidebarProvider className="h-screen !min-h-0 overflow-hidden">
                <AppSidebar />
                <main className="flex-1 min-w-0 h-screen overflow-hidden">
                  {children}
                </main>
              </SidebarProvider>
            </AppProvider>
          </SessionProvider>
        </Suspense>
        <Toaster position="bottom-center" />
      </body>
    </html>
  );
}
