import type {
  AuditResponse,
  DatasetSummary,
  Evidence,
  PipelineResponse,
  SchemaTable,
  SuggestionGroup,
  User,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "nl2sql.token";
const USER_KEY = "nl2sql.user";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getStoredUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  } catch {
    return null;
  }
}

export function clearSession() {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  } catch {
    /* storage unavailable */
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...init.headers,
      },
    });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is the backend running?`);
  }
  if (res.status === 401 && path !== "/api/auth/login") {
    clearSession();
    if (typeof window !== "undefined") window.location.href = "/login";
  }
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async login(username: string, password: string): Promise<User> {
    const data = await request<{ token: string; user: User }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    localStorage.setItem(TOKEN_KEY, data.token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    return data.user;
  },
  me: () => request<User>("/api/auth/me"),
  health: () =>
    request<{ status: string; llm: string; llm_status: { mode: "local"; reason: string | null }; embedder: string; config: Record<string, number> }>("/api/health"),
  datasets: () => request<DatasetSummary[]>("/api/datasets"),
  suggestions: (datasetId: string) => request<SuggestionGroup[]>(`/api/suggestions?dataset_id=${encodeURIComponent(datasetId)}`),
  query: (question: string, datasetId: string, sessionId: string | null) =>
    request<PipelineResponse>("/api/query", {
      method: "POST",
      body: JSON.stringify({ question, dataset_id: datasetId, session_id: sessionId }),
    }),
  evidence: (sessionId: string, responseId: string, key: string | number | null) =>
    request<Evidence>("/api/query/evidence", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, response_id: responseId, key }),
    }),
  clarify: (sessionId: string, optionId: string | null, text: string | null) =>
    request<PipelineResponse>("/api/query/clarify", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, option_id: optionId, text }),
    }),
  schema: (datasetId: string) =>
    request<{ dataset: DatasetSummary; tables: SchemaTable[]; embedder: string }>(`/api/admin/schema?dataset_id=${encodeURIComponent(datasetId)}`),
  annotate: (datasetId: string, table: string, column: string, description: string) =>
    request<{ ok: boolean }>("/api/admin/schema/annotations", {
      method: "PUT",
      body: JSON.stringify({ dataset_id: datasetId, table, column, description }),
    }),
  reindex: (datasetId: string) =>
    request<{ chunks: number; embedder: string }>(`/api/admin/reindex?dataset_id=${encodeURIComponent(datasetId)}`, { method: "POST" }),
  policies: (datasetId: string) =>
    request<{ roles: Record<string, { table: string; column: string }[]> }>(`/api/admin/policies?dataset_id=${encodeURIComponent(datasetId)}`),
  setPolicies: (datasetId: string, role: string, denied: { table: string; column: string }[]) =>
    request<{ ok: boolean }>("/api/admin/policies", { method: "PUT", body: JSON.stringify({ dataset_id: datasetId, role, denied }) }),
  audit: (datasetId?: string) =>
    request<AuditResponse>(`/api/admin/audit${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
};
