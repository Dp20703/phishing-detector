// Set VITE_API_URL in frontend/.env (local) or in Vercel's environment
// variables (production). Falls back to the local FastAPI server.
const API_URL = (
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export type RiskLevel = "low" | "uncertain" | "high";

export interface PredictionResponse {
  url: string;
  label: "phishing" | "legitimate";
  risk_level: RiskLevel;
  probability: number;
  reasons: string[];
  model: string;
}

// FastAPI returns either a string or a list of validation errors in `detail`.
function extractError(data: unknown): string {
  const detail = (data as { detail?: unknown })?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) {
    return String(detail[0].msg).replace(/^Value error,\s*/, "");
  }
  return "Unable to analyze this URL.";
}

export async function checkUrl(url: string): Promise<PredictionResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  } catch {
    throw new Error(
      "Could not reach the server. If it was idle, it may take up to a minute to wake up — please try again.",
    );
  }

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(extractError(data));
  }

  return data as PredictionResponse;
}
