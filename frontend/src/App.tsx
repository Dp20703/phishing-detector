import { type SubmitEvent, useState } from "react";
import { checkUrl, type PredictionResponse } from "./services/api";

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedUrl = url.trim();

    if (!trimmedUrl) {
      setError("Please enter a URL.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await checkUrl(trimmedUrl);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const riskScore = result ? Math.round(result.probability * 100) : 0;

  // Three-level result: the model is only trusted for confident predictions.
  const tones = {
    low: {
      card: "border-emerald-900/60 bg-emerald-950/20",
      iconBg: "bg-emerald-500/10",
      text: "text-emerald-400",
      bar: "bg-emerald-500",
      bullet: "bg-slate-800 text-slate-400",
      title: "Likely Safe",
      icon: "✓",
      bulletIcon: "•",
    },
    uncertain: {
      card: "border-amber-900/60 bg-amber-950/20",
      iconBg: "bg-amber-500/10",
      text: "text-amber-400",
      bar: "bg-amber-500",
      bullet: "bg-amber-500/10 text-amber-400",
      title: "Uncertain — check manually",
      icon: "?",
      bulletIcon: "•",
    },
    high: {
      card: "border-red-900/60 bg-red-950/20",
      iconBg: "bg-red-500/10",
      text: "text-red-400",
      bar: "bg-red-500",
      bullet: "bg-red-500/10 text-red-400",
      title: "Suspicious URL",
      icon: "!",
      bulletIcon: "!",
    },
  } as const;
  const tone = tones[result?.risk_level ?? "low"];

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-5 py-10 sm:px-8">
        {/* Header */}
        <header className="py-10 text-center">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            AI-Powered URL Analysis
          </div>

          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Phishing Detector
          </h1>

          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-slate-400 sm:text-lg">
            Check a URL for suspicious patterns using a machine learning model
            trained to detect phishing signals.
          </p>
        </header>

        {/* URL Form */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-2xl sm:p-7">
          <form onSubmit={handleSubmit}>
            <label
              htmlFor="url"
              className="mb-3 block text-sm font-medium text-slate-300"
            >
              URL to analyze
            </label>

            <div className="flex flex-col gap-3 sm:flex-row">
              <input
                id="url"
                type="text"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com"
                disabled={loading}
                className="min-w-0 flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3.5 text-white outline-none transition placeholder:text-slate-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 disabled:cursor-not-allowed disabled:opacity-60"
              />

              <button
                type="submit"
                disabled={loading}
                className="rounded-xl bg-blue-600 px-7 py-3.5 font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Checking..." : "Check URL"}
              </button>
            </div>
          </form>

          {/* Error */}
          {error && (
            <div className="mt-5 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </section>

        {/* Loading */}
        {loading && (
          <div className="mt-6 flex items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 p-8">
            <div className="flex items-center gap-3 text-slate-400">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-600 border-t-blue-500" />
              <span>Analyzing URL...</span>
            </div>
          </div>
        )}

        {/* Result */}
        {result && !loading && (
          <section
            className={`mt-6 overflow-hidden rounded-2xl border ${tone.card}`}
          >
            {/* Result Header */}
            <div className="p-6 sm:p-8">
              <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-4">
                  <div
                    className={`flex h-14 w-14 items-center justify-center rounded-2xl text-2xl ${tone.iconBg} ${tone.text}`}
                  >
                    {tone.icon}
                  </div>

                  <div>
                    <p className="text-sm text-slate-400">Analysis result</p>

                    <h2 className={`text-2xl font-bold ${tone.text}`}>
                      {tone.title}
                    </h2>
                  </div>
                </div>

                {/* Risk score */}
                <div className="text-left sm:text-right">
                  <p className="text-sm text-slate-400">Phishing risk</p>

                  <p className={`text-4xl font-bold ${tone.text}`}>
                    {riskScore}%
                  </p>
                </div>
              </div>

              {/* URL */}
              <div className="mt-6 rounded-xl bg-slate-950/70 p-4">
                <p className="mb-1 text-xs uppercase tracking-wide text-slate-500">
                  Analyzed URL
                </p>

                <p className="break-all text-sm text-slate-300">{result.url}</p>
              </div>

              {/* Risk bar */}
              <div className="mt-6">
                <div className="mb-2 flex justify-between text-xs text-slate-500">
                  <span>Low risk</span>
                  <span>High risk</span>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${tone.bar}`}
                    style={{
                      width: `${riskScore}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Signals */}
            <div className="border-t border-slate-800/80 p-6 sm:p-8">
              <h3 className="font-semibold text-white">Signals in the URL text</h3>

              <p className="mt-1 text-sm text-slate-500">
                Simple rule-based checks. These are not the model's internal
                reasoning.
              </p>

              <div className="mt-4 space-y-3">
                {result.reasons.map((reason, index) => (
                  <div
                    key={`${reason}-${index}`}
                    className="flex items-start gap-3 rounded-xl bg-slate-950/60 px-4 py-3"
                  >
                    <span
                      className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs ${tone.bullet}`}
                    >
                      {tone.bulletIcon}
                    </span>

                    <span className="text-sm text-slate-300">{reason}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Disclaimer */}
        <footer className="mt-auto pt-10 text-center">
          <p className="mx-auto max-w-2xl text-xs leading-5 text-slate-600">
            This tool analyzes URL text and its structural patterns only. It
            does not visit the website or guarantee that a URL is safe or
            malicious. The model was trained on a public dataset and can be
            wrong, especially on unusual or very new links.
          </p>
        </footer>
      </div>
    </main>
  );
}

export default App;
