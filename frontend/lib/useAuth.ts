"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { getStoredUser, getToken } from "./api";
import type { Role, User } from "./types";

/** Redirects to /login without a session, or to /chat when the role is not allowed. */
export function useAuth(requiredRole?: Role): User | null {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const stored = getStoredUser();
    if (!getToken() || !stored) {
      router.replace("/login");
      return;
    }
    if (requiredRole && stored.role !== requiredRole) {
      router.replace("/chat");
      return;
    }
    setUser(stored);
  }, [router, requiredRole]);

  return user;
}
