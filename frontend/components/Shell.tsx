"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { clearSession } from "@/lib/api";
import type { User } from "@/lib/types";

interface Props {
  user: User;
  title: string;
  sidebar?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
}

export default function Shell({ user, title, sidebar, actions, children }: Props) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);

  const logout = () => {
    clearSession();
    router.replace("/login");
  };

  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`} onClick={(e) => e.target === e.currentTarget && setOpen(false)}>
        <div className="brand">
          <div className="brand-mark">NL</div>
          <div>
            <div className="brand-title">Verified Analytics</div>
            <div className="brand-sub">Hallucination-resistant NL2SQL</div>
          </div>
        </div>
        <nav className="nav">
          <Link href="/chat" className={pathname === "/chat" ? "active" : ""} onClick={() => setOpen(false)}>
            <span aria-hidden>💬</span> Ask a question
          </Link>
          {user.role === "admin" && (
            <Link href="/admin" className={pathname === "/admin" ? "active" : ""} onClick={() => setOpen(false)}>
              <span aria-hidden>🛠️</span> Administration
            </Link>
          )}
        </nav>
        {sidebar}
        <div className="sidebar-footer">
          <div className="avatar">{user.display_name.slice(0, 1)}</div>
          <div style={{ minWidth: 0 }}>
            <div style={{ color: "#fff", fontWeight: 500 }}>{user.display_name}</div>
            <div style={{ fontSize: 12, color: "#64748b" }}>
              {user.username} · {user.role === "admin" ? "Administrator" : "Insights User"}
            </div>
          </div>
          <div className="spacer" />
          <button className="btn btn-ghost btn-sm" onClick={logout} title="Sign out">
            Sign out
          </button>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <button className="btn btn-ghost btn-sm menu-btn" onClick={() => setOpen(true)} aria-label="Open menu">
            ☰
          </button>
          <h1>{title}</h1>
          <div className="spacer" />
          {actions}
        </header>
        {children}
      </div>
    </div>
  );
}
