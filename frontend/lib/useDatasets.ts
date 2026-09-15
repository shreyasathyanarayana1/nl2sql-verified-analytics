"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { DatasetSummary } from "./types";

const KEY = "nl2sql.dataset";

/** Loads available datasets and remembers the viewer's selection in this browser. */
export function useDatasets(enabled: boolean) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    api
      .datasets()
      .then((list) => {
        setDatasets(list);
        let saved: string | null = null;
        try {
          saved = localStorage.getItem(KEY);
        } catch {
          /* storage unavailable */
        }
        setSelectedId(list.find((d) => d.id === saved)?.id ?? list[0]?.id ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load datasets"));
  }, [enabled]);

  const select = useCallback((id: string) => {
    setSelectedId(id);
    try {
      localStorage.setItem(KEY, id);
    } catch {
      /* storage unavailable */
    }
  }, []);

  return { datasets, selected: datasets.find((d) => d.id === selectedId) ?? null, select, error };
}
