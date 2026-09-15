"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { getStoredUser, getToken } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace(getToken() && getStoredUser() ? "/chat" : "/login");
  }, [router]);
  return null;
}
