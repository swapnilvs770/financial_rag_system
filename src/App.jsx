import { useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowUpRight, BookOpen, CheckCircle2, ChevronDown, CircleHelp,
  Clock3, Command, Database, FileSearch, FileText, Filter, Layers3,
  LoaderCircle, MessageSquareText, PanelLeftClose, Search, Send, ShieldCheck,
  SlidersHorizontal, Sparkles, XCircle
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function prettyDocument(item) {
  if (typeof item === "string") {
    return { id: item.replace(/\.pdf$/i, ""), name: item, filename: item };
  }
  const id = item?.document_id || item?.id || item?.filename || item?.name || "unknown-document";
  const name = item?.filename || item?.name || item?.title || id;
  return { ...item, id, name, filename: item?.filename || name };
}

function extractDocuments(payload) {
  if (Array.isArray(payload)) return payload.map(prettyDocument);
  for (const key of ["documents", "items", "data", "results"]) {
    if (Array.isArray(payload?.[key])) return payload[key].map(prettyDocument);
  }
  return [];
}

function App() {
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState("collection");
  const [selectedDocument, setSelectedDocument] = useState("");
  const [retrievalK, setRetrievalK] = useState(50);
  const [finalK, setFinalK] = useState(10);
  const [documents, setDocuments] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState("checking");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [activeNav, setActiveNav] = useState("Ask a question");

  useEffect(() => {
    let alive = true;
    async function loadData() {
      try {
        const [healthRes, docsRes] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/api/documents`)
        ]);
        if (!healthRes.ok) throw new Error("Health endpoint returned an error.");
        if (!alive) return;
        setApiStatus("connected");
        if (docsRes.ok) {
          const payload = await docsRes.json();
          const parsed = extractDocuments(payload);
          setDocuments(parsed);
          if (parsed.length) setSelectedDocument(parsed[0].id);
        }
      } catch {
        if (alive) setApiStatus("offline");
      } finally {
        if (alive) setDocumentsLoading(false);
      }
    }
    loadData();
    return () => { alive = false; };
  }, []);

  const answer = response?.answer;
  const sources = useMemo(() => {
    const candidates = [
      answer?.sources,
      response?.sources,
      response?.results,
      response?.retrieval?.results,
      response?.evidence
    ];
    return candidates.find(Array.isArray) || [];
  }, [answer, response]);

  async function askQuestion(nextQuestion = question) {
    const cleanQuestion = nextQuestion.trim();
    if (!cleanQuestion) {
      setError("Enter a question about the annual reports first.");
      return;
    }
    if (scope === "document" && !selectedDocument) {
      setError("Select a document, or switch the scope to the full collection.");
      return;
    }
    setLoading(true);
    setError("");
    setResponse(null);
    setQuestion(cleanQuestion);
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: cleanQuestion,
          scope,
          document_id: scope === "document" ? selectedDocument : null,
          retrieval_k: Number(retrievalK),
          final_k: Number(finalK)
        })
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload?.detail || payload?.message || `Request failed (${res.status}).`);
      }
      setResponse(payload);
    } catch (e) {
      setError(`${e.message || "Could not reach the API."} Check that FastAPI is running at ${API_BASE}.`);
    } finally {
      setLoading(false);
    }
  }

  const suggestions = [
    "How much did TCS spend on employee costs in FY2025?",
    "What was TCS's consolidated revenue from operations in FY2025?",
    "Compare employee benefit expenses in FY2024 and FY2025."
  ];

  function chooseSuggestion(text) {
    setQuestion(text);
    askQuestion(text);
  }

  return (
    <div className="min-h-screen bg-[#080c18] text-slate-100">
      <div className="flex min-h-screen">
        <aside className="hidden w-[250px] shrink-0 flex-col border-r border-white/[0.07] bg-[#0b1020] lg:flex">
          <div className="flex h-[76px] items-center gap-3 border-b border-white/[0.07] px-5">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-500/15 text-indigo-300 ring-1 ring-indigo-400/25">
              <Activity size={21} />
            </div>
            <div>
              <div className="text-[15px] font-semibold tracking-wide">FinSight</div>
              <div className="mt-0.5 text-[11px] text-slate-500">FINANCIAL RAG WORKSPACE</div>
            </div>
          </div>

          <div className="px-3 pt-6">
            <div className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">Workspace</div>
            {[
              { name: "Ask a question", icon: MessageSquareText },
              { name: "Report library", icon: BookOpen },
              { name: "Retrieval settings", icon: SlidersHorizontal }
            ].map(({ name, icon: Icon }) => (
              <button key={name} onClick={() => {
                setActiveNav(name);
                if (name === "Retrieval settings") setShowSettings(true);
                if (name === "Ask a question") document.getElementById("question-box")?.focus();
              }} className={`mb-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-[13px] transition ${activeNav === name ? "bg-indigo-500/15 text-indigo-200 ring-1 ring-indigo-400/15" : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"}`}>
                <Icon size={17} /> {name}
                {name === "Report library" && <span className="ml-auto rounded-md bg-white/[0.06] px-1.5 py-0.5 text-[10px]">{documents.length || "—"}</span>}
              </button>
            ))}
          </div>

          <div className="mx-4 mt-7 rounded-2xl border border-white/[0.08] bg-white/[0.025] p-4">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <Database size={15} className="text-indigo-300" /> Knowledge base
            </div>
            <div className="mt-4 text-2xl font-semibold tracking-tight">{documentsLoading ? "…" : documents.length || "—"}</div>
            <div className="mt-1 text-xs text-slate-500">Reports available through API</div>
            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
              <div className={`h-full rounded-full transition-all ${apiStatus === "connected" ? "w-full bg-emerald-400" : "w-1/3 bg-amber-400"}`} />
            </div>
            <div className="mt-2 flex items-center gap-1.5 text-[11px] text-slate-500">
              <span className={`h-1.5 w-1.5 rounded-full ${apiStatus === "connected" ? "bg-emerald-400" : apiStatus === "offline" ? "bg-rose-400" : "bg-amber-400"}`} />
              {apiStatus === "connected" ? "API connection healthy" : apiStatus === "offline" ? "API unavailable" : "Checking API"}
            </div>
          </div>

          <div className="mt-auto border-t border-white/[0.07] p-4">
            <div className="flex items-center gap-3 rounded-xl p-2">
              <div className="grid h-9 w-9 place-items-center rounded-full bg-gradient-to-br from-indigo-400 to-violet-600 text-xs font-bold text-white">FS</div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-slate-200">Research workspace</div>
                <div className="mt-0.5 truncate text-[11px] text-slate-500">Local development</div>
              </div>
              <ShieldCheck size={16} className="text-slate-600" />
            </div>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="flex h-[76px] items-center justify-between border-b border-white/[0.07] px-5 sm:px-8">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <span className="text-slate-600">Workspace</span><span className="text-slate-700">/</span><span className="text-slate-200">{activeNav}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] sm:flex ${apiStatus === "connected" ? "border-emerald-400/15 bg-emerald-400/[0.06] text-emerald-300" : "border-amber-400/15 bg-amber-400/[0.06] text-amber-300"}`}>
                <span className={`h-1.5 w-1.5 rounded-full ${apiStatus === "connected" ? "bg-emerald-400" : "bg-amber-400"}`} />
                {apiStatus === "connected" ? "Backend connected" : apiStatus === "offline" ? "Backend offline" : "Connecting"}
              </div>
              <button onClick={() => setShowSettings(!showSettings)} className="rounded-xl border border-white/10 p-2.5 text-slate-400 transition hover:bg-white/[0.05] hover:text-white" aria-label="Toggle retrieval settings">
                <SlidersHorizontal size={17} />
              </button>
            </div>
          </header>

          <div className="mx-auto max-w-[1050px] px-5 pb-16 pt-9 sm:px-8 sm:pt-12">
            <div className="mb-8">
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-400/20 bg-indigo-400/[0.07] px-3 py-1.5 text-[11px] font-medium text-indigo-200">
                <Sparkles size={13} /> ANNUAL REPORT INTELLIGENCE
              </div>
              <h1 className="max-w-3xl text-3xl font-semibold leading-tight tracking-[-0.04em] text-white sm:text-[42px]">
                Research financials.<br className="hidden sm:block" /> <span className="text-slate-500">Ground every answer in evidence.</span>
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-400 sm:text-[15px]">
                Ask questions across your annual report collection. Review extracted answers alongside the report pages and supporting text used by retrieval.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <StatCard icon={Layers3} label="Search mode" value={scope === "collection" ? "Full collection" : "Single report"} />
              <StatCard icon={FileText} label="Available reports" value={documentsLoading ? "Loading…" : String(documents.length || "—")} />
              <StatCard icon={ShieldCheck} label="Answer approach" value="Evidence-first" />
            </div>

            <section className="mt-8 overflow-hidden rounded-2xl border border-white/[0.09] bg-[#0d1324] shadow-glow">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-5 py-4 sm:px-6">
                <div className="flex items-center gap-2.5">
                  <div className="grid h-8 w-8 place-items-center rounded-lg bg-indigo-500/15 text-indigo-300"><FileSearch size={17} /></div>
                  <div>
                    <div className="text-sm font-semibold text-slate-100">Ask your reports</div>
                    <div className="mt-0.5 text-[11px] text-slate-500">Hybrid retrieval · FAISS + BM25</div>
                  </div>
                </div>
                <button onClick={() => setShowSettings(!showSettings)} className="inline-flex items-center gap-2 rounded-lg border border-white/[0.09] px-3 py-2 text-xs text-slate-400 hover:bg-white/[0.04] hover:text-slate-200">
                  <SlidersHorizontal size={14} /> Retrieval settings <ChevronDown size={13} className={showSettings ? "rotate-180 transition" : "transition"} />
                </button>
              </div>

              {showSettings && (
                <div className="grid gap-4 border-b border-white/[0.07] bg-white/[0.015] px-5 py-5 sm:grid-cols-2 sm:px-6">
                  <div>
                    <label className="mb-2 flex items-center justify-between text-xs font-medium text-slate-300"><span>Retrieval candidates</span><span className="font-mono text-indigo-300">{retrievalK}</span></label>
                    <input type="range" min="10" max="100" step="10" value={retrievalK} onChange={e => setRetrievalK(Number(e.target.value))} className="w-full accent-indigo-400" />
                    <p className="mt-1 text-[11px] text-slate-500">Number of chunks considered during retrieval.</p>
                  </div>
                  <div>
                    <label className="mb-2 flex items-center justify-between text-xs font-medium text-slate-300"><span>Final evidence chunks</span><span className="font-mono text-indigo-300">{finalK}</span></label>
                    <input type="range" min="3" max="20" step="1" value={finalK} onChange={e => setFinalK(Number(e.target.value))} className="w-full accent-indigo-400" />
                    <p className="mt-1 text-[11px] text-slate-500">Number of final chunks passed to the answer layer.</p>
                  </div>
                </div>
              )}

              <div className="p-5 sm:p-6">
                <div className="mb-4 grid gap-3 sm:grid-cols-[180px_minmax(0,1fr)]">
                  <div>
                    <label className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-slate-500">Search scope</label>
                    <div className="relative">
                      <select value={scope} onChange={e => setScope(e.target.value)} className="w-full appearance-none rounded-xl border border-white/10 bg-[#090e1b] px-3.5 py-3 pr-9 text-sm text-slate-200 outline-none focus:border-indigo-400/50">
                        <option value="collection">All reports</option>
                        <option value="document">One report</option>
                      </select>
                      <ChevronDown size={15} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500" />
                    </div>
                  </div>
                  {scope === "document" && (
                    <div>
                      <label className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-slate-500">Select report</label>
                      <div className="relative">
                        <select value={selectedDocument} onChange={e => setSelectedDocument(e.target.value)} className="w-full appearance-none rounded-xl border border-white/10 bg-[#090e1b] px-3.5 py-3 pr-9 text-sm text-slate-200 outline-none focus:border-indigo-400/50">
                          {documents.length === 0 && <option value="">No reports returned by API</option>}
                          {documents.map(doc => <option key={doc.id} value={doc.id}>{doc.name}</option>)}
                        </select>
                        <ChevronDown size={15} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500" />
                      </div>
                    </div>
                  )}
                </div>

                <label htmlFor="question-box" className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-slate-500">Your question</label>
                <div className="rounded-2xl border border-white/[0.12] bg-[#080d19] p-3 transition focus-within:border-indigo-400/50 focus-within:ring-4 focus-within:ring-indigo-400/[0.06]">
                  <textarea id="question-box" rows="3" value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askQuestion(); } }} placeholder="e.g. How much did TCS spend on employee costs in FY2025?" className="w-full resize-y bg-transparent px-2 py-1 text-sm leading-6 text-slate-100 outline-none placeholder:text-slate-600" />
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.07] pt-3">
                    <div className="flex items-center gap-2 text-[11px] text-slate-600"><Command size={13} /> Enter to ask <span className="text-slate-700">·</span> Shift + Enter for new line</div>
                    <button disabled={loading} onClick={() => askQuestion()} className="inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-500 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-950/30 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60">
                      {loading ? <LoaderCircle size={15} className="animate-spin" /> : <Send size={14} />}
                      {loading ? "Searching reports…" : "Ask FinSight"}
                    </button>
                  </div>
                </div>

                <div className="mt-5">
                  <div className="mb-2.5 flex items-center gap-2 text-[11px] font-medium text-slate-500"><Sparkles size={13} /> Try a suggested question</div>
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map(s => <button key={s} onClick={() => chooseSuggestion(s)} disabled={loading} className="rounded-full border border-white/[0.09] bg-white/[0.02] px-3 py-2 text-left text-[11px] leading-4 text-slate-400 transition hover:border-indigo-400/30 hover:bg-indigo-400/[0.06] hover:text-indigo-200 disabled:opacity-50">{s}</button>)}
                  </div>
                </div>
              </div>
            </section>

            {error && <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-400/20 bg-rose-400/[0.06] p-4 text-sm text-rose-200"><XCircle size={18} className="mt-0.5 shrink-0" /><div><div className="font-medium">Could not complete the query</div><p className="mt-1 text-xs leading-5 text-rose-200/80">{error}</p></div></div>}

            {loading && <div className="mt-6 flex items-center gap-3 rounded-xl border border-indigo-400/15 bg-indigo-400/[0.04] p-5 text-sm text-indigo-200"><LoaderCircle size={18} className="animate-spin" /><div><div className="font-medium">Searching your financial reports</div><div className="mt-1 text-xs text-slate-500">Retrieving relevant passages and preparing evidence…</div></div></div>}

            {response && !loading && (
              <section className="mt-7 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500">Research result</div>
                    <h2 className="mt-1 text-lg font-semibold text-white">Answer and evidence</h2>
                  </div>
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/15 bg-emerald-400/[0.06] px-3 py-1.5 text-[11px] text-emerald-300"><CheckCircle2 size={13} /> Response received</span>
                </div>
                <div className="rounded-2xl border border-white/[0.09] bg-[#0d1324] p-5 sm:p-6">
                  <div className="mb-3 flex items-center gap-2 text-xs font-medium text-indigo-200"><Sparkles size={15} /> Answer</div>
                  <p className="whitespace-pre-wrap text-[15px] leading-7 text-slate-200">{typeof answer === "string" ? answer : answer?.answer || response?.message || "The API returned a response, but no answer field was found. Inspect the raw response below."}</p>
                  {answer?.note && <p className="mt-4 rounded-lg border border-amber-300/10 bg-amber-300/[0.04] p-3 text-xs leading-5 text-amber-200/80">{answer.note}</p>}
                  {answer?.answer_type && <div className="mt-4 text-[11px] text-slate-500">Answer type: <span className="font-mono text-slate-400">{answer.answer_type}</span></div>}
                </div>

                {sources.length > 0 && <div>
                  <div className="mb-3 flex items-center justify-between"><h3 className="text-sm font-semibold text-slate-200">Supporting evidence <span className="ml-1 text-xs font-normal text-slate-500">({sources.length})</span></h3><span className="text-[11px] text-slate-600">Verify report pages before external use</span></div>
                  <div className="space-y-3">{sources.map((src, i) => <SourceCard key={src.chunk_id || src.id || `${src.filename}-${i}`} source={src} index={i} />)}</div>
                </div>}

                <details className="rounded-xl border border-white/[0.08] bg-white/[0.02]">
                  <summary className="cursor-pointer px-4 py-3 text-xs font-medium text-slate-400 hover:text-slate-200">Inspect raw API response</summary>
                  <pre className="max-h-[420px] overflow-auto border-t border-white/[0.07] p-4 text-[11px] leading-5 text-slate-400">{JSON.stringify(response, null, 2)}</pre>
                </details>
              </section>
            )}

            {!response && !loading && !error && (
              <div className="mt-7 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5">
                  <div className="mb-3 grid h-9 w-9 place-items-center rounded-xl bg-cyan-400/[0.08] text-cyan-300"><Search size={17} /></div>
                  <div className="text-sm font-medium text-slate-200">Hybrid retrieval</div>
                  <p className="mt-1.5 text-xs leading-5 text-slate-500">Combines semantic similarity and keyword matching to surface relevant annual-report passages.</p>
                </div>
                <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5">
                  <div className="mb-3 grid h-9 w-9 place-items-center rounded-xl bg-violet-400/[0.08] text-violet-300"><ShieldCheck size={17} /></div>
                  <div className="text-sm font-medium text-slate-200">Evidence-first answers</div>
                  <p className="mt-1.5 text-xs leading-5 text-slate-500">Review source pages and supporting text rather than relying on an untraceable response.</p>
                </div>
              </div>
            )}

            <footer className="mt-12 flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.07] pt-5 text-[11px] text-slate-600">
              <span>FinSight · Financial RAG prototype</span>
              <span className="flex items-center gap-1.5"><CircleHelp size={12} /> Answers depend on retrieved report content and backend capabilities.</span>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value }) {
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4">
      <div className="flex items-center gap-2 text-[11px] text-slate-500"><Icon size={14} /> {label}</div>
      <div className="mt-2 text-lg font-semibold tracking-tight text-slate-100">{value}</div>
    </div>
  );
}

function SourceCard({ source, index }) {
  const filename = source.filename || source.document_id || source.document || source.source || "Source document";
  const page = source.page_number ?? source.page ?? source.pdf_page;
  const chunk = source.chunk_id || source.id;
  const text = source.supporting_text || source.text || source.content || source.page_content || source.chunk_text;
  const basis = source.basis;
  const score = source.score ?? source.relevance_score;
  return (
    <article className="rounded-xl border border-white/[0.08] bg-[#0d1324] p-4 sm:p-5">
      <div className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-indigo-400/15 bg-indigo-400/[0.06] text-indigo-300"><FileText size={17} /></div>
        <div className="min-w-0 flex-1">
          <div className="break-words text-sm font-medium text-slate-200">{filename}</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {page != null && <span className="rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-1 text-[10px] text-slate-400">Page {page}</span>}
            {basis && <span className="rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-1 text-[10px] capitalize text-slate-400">{basis}</span>}
            {chunk && <span className="max-w-full break-all rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-1 font-mono text-[10px] text-slate-500">{chunk}</span>}
            {score != null && <span className="rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-1 text-[10px] text-slate-500">Score {Number(score).toFixed(4)}</span>}
          </div>
        </div>
        <span className="text-[10px] text-slate-600">#{index + 1}</span>
      </div>
      {text && <div className="mt-4 rounded-lg border border-white/[0.06] bg-black/10 p-3 text-xs leading-6 text-slate-400">{text}</div>}
      {!text && <p className="mt-3 text-xs text-slate-500">No supporting-text field was present in this source object.</p>}
    </article>
  );
}

export default App;