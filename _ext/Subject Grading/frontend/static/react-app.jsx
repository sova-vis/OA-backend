const { useEffect, useMemo, useRef, useState } = React;

const SUBJECT_OPTIONS = [
  "Chemistry 1011",
  "Physics 1016",
  "Mathematics 1014",
  "English 1123",
  "Islamiyat 2058",
  "Pakistan Studies 2059",
  "Biology 5090",
];

function sanitizeOptionalText(value) {
  const trimmed = (value || "").trim();
  return trimmed.length > 0 ? trimmed : null;
}

function sanitizeOptionalInt(value) {
  const trimmed = (value || "").trim();
  if (!trimmed) return null;
  const num = Number(trimmed);
  if (!Number.isInteger(num) || num <= 0) return null;
  return num;
}

function getApiBase() {
  const configured = typeof window !== "undefined" ? window.OA_API_BASE : null;
  if (configured && typeof configured === "string" && configured.trim()) {
    return configured.trim().replace(/\/$/, "");
  }
  const protocol = window.location.protocol || "http:";
  const host = window.location.hostname || "localhost";
  return `${protocol}//${host}:8001`;
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${Math.max(0, Math.min(100, value)).toFixed(1)}%`;
}

function safeJsonParse(text) {
  if (!text) return { ok: true, value: null };
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch (err) {
    return { ok: false, value: null, error: String(err) };
  }
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(id);
  }
}

function formatHttpError(status, payload) {
  if (payload && typeof payload === "object") {
    const detail = payload.detail;
    if (typeof detail === "string" && detail.trim()) return detail.trim();
    if (Array.isArray(detail)) return "Validation error.";
  }
  return `Request failed (${status}).`;
}

function inferSourceSummary(result) {
  if (!result) return null;
  const primary = result.primary_data_source || "--";
  const fallback = result.fallback_data_source || "--";
  const used = Boolean(result.fallback_used);
  return `Primary: ${primary}. Fallback: ${fallback}. Used fallback: ${used}.`;
}

function StatusBadge({ status }) {
  const value = status || "--";
  const cls = status ? `badge ${status}` : "badge unknown";
  return <span className={cls}>{value}</span>;
}

function ModeToggle({ mode, setMode, disabled }) {
  return (
    <div className="mode-toggle" role="tablist" aria-label="Input mode">
      <button
        type="button"
        className={mode === "typed" ? "tab active" : "tab"}
        onClick={() => setMode("typed")}
        disabled={disabled}
      >
        Typed Question (Mode B)
      </button>
      <button
        type="button"
        className={mode === "upload" ? "tab active" : "tab"}
        onClick={() => setMode("upload")}
        disabled={disabled}
      >
        Upload Question (Mode A)
      </button>
    </div>
  );
}

function App() {
  const STORAGE_KEY = "oa_frontend_state_v2";
  const abortRef = useRef(null);

  const [form, setForm] = useState({
    subject: "Chemistry 1011",
    question: "",
    student_answer: "",
    year: "",
    session: "",
    paper: "",
    variant: "",
    question_id: "",
    debug: true,
  });
  const [mode, setMode] = useState("typed"); // typed | upload
  const [upload, setUpload] = useState({
    file: null,
    page_number: "1",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [previewEdit, setPreviewEdit] = useState({ question_text: "", student_answer: "" });
  const [lastRequest, setLastRequest] = useState(null);

  const apiBase = useMemo(() => getApiBase(), []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        if (parsed.form) setForm((prev) => ({ ...prev, ...parsed.form }));
        if (parsed.mode === "typed" || parsed.mode === "upload") setMode(parsed.mode);
        if (parsed.upload) setUpload((prev) => ({ ...prev, ...parsed.upload, file: null }));
        if (parsed.result) setResult(parsed.result);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          mode,
          form,
          upload: { page_number: upload.page_number },
          result,
        })
      );
    } catch {
      // ignore
    }
  }, [mode, form, upload.page_number, result]);

  const onChange = (key) => (event) => {
    const value = key === "debug" ? event.target.checked : event.target.value;
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const onUploadChange = (key) => (event) => {
    const value = key === "file" ? event.target.files?.[0] || null : event.target.value;
    setUpload((prev) => ({ ...prev, [key]: value }));
  };

  const loadSample = () => {
    setForm((prev) => ({
      ...prev,
      subject: "Chemistry 1011",
      question:
        "An aqueous solution of zinc chloride is tested by adding reagents. Which observation is correct?",
      student_answer: "B",
      year: "",
      session: "",
      paper: "",
      variant: "",
      question_id: "",
      debug: true,
    }));
    setError("");
    setWarning("");
  };

  const buildOptionalFilters = () => {
    const payload = {
      subject: sanitizeOptionalText(form.subject),
    };

    const year = sanitizeOptionalInt(form.year);
    const session = sanitizeOptionalText(form.session);
    const paper = sanitizeOptionalText(form.paper);
    const variant = sanitizeOptionalText(form.variant);
    const questionId = sanitizeOptionalText(form.question_id);

    if (year !== null) payload.year = year;
    if (session !== null) payload.session = session;
    if (paper !== null) payload.paper = paper;
    if (variant !== null) payload.variant = variant;
    if (questionId !== null) payload.question_id = questionId;

    return payload;
  };

  const stopInFlight = () => {
    try {
      abortRef.current?.abort?.();
    } catch {
      // ignore
    }
    abortRef.current = null;
  };

  const submitTyped = async () => {
    setError("");
    setWarning("");
    stopInFlight();

    const payload = {
      question: form.question.trim(),
      student_answer: form.student_answer.trim(),
      ...buildOptionalFilters(),
      debug: Boolean(form.debug),
    };
    if (!payload.question || !payload.student_answer) {
      setError("Question and student answer are required.");
      return;
    }
    setLoading(true);
    setResult(null);
    setPreview(null);

    try {
      const response = await fetchWithTimeout(
        `${apiBase}/oa-level/evaluate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        300000
      );
      const text = await response.text();
      const parsed = safeJsonParse(text);
      const data = parsed.ok ? parsed.value : { raw: text };
      setLastRequest({ url: `${apiBase}/oa-level/evaluate`, mode: "typed", payload });

      if (!parsed.ok) setWarning(`Response was not valid JSON: ${parsed.error}`);
      if (!response.ok) {
        setError(formatHttpError(response.status, data));
        setResult(data);
        return;
      }
      setResult(data);
    } catch (err) {
      const msg = String(err);
      if (msg.includes("AbortError")) setError("Request timed out.");
      else setError(`Network error: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const submitUpload = async () => {
    setError("");
    setWarning("");
    stopInFlight();

    const file = upload.file;
    if (!file) {
      setError("Please choose a PDF or image file.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("File too large (max 20MB).");
      return;
    }

    const formData = new FormData();
    formData.append("file", file, file.name);
    formData.append("debug", String(Boolean(form.debug)));
    const filters = buildOptionalFilters();
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== null && v !== undefined) formData.append(k, String(v));
    });
    const pageNum = sanitizeOptionalInt(upload.page_number);
    formData.append("page_number", String(pageNum || 1));

    setLoading(true);
    setResult(null);
    try {
      const response = await fetchWithTimeout(
        `${apiBase}/oa-level/evaluate-from-image/preview`,
        { method: "POST", body: formData },
        300000
      );
      const text = await response.text();
      const parsed = safeJsonParse(text);
      const data = parsed.ok ? parsed.value : { raw: text };
      setLastRequest({ url: `${apiBase}/oa-level/evaluate-from-image/preview`, mode: "upload", payload: { ...filters, page_number: pageNum || 1 } });

      if (!parsed.ok) setWarning(`Response was not valid JSON: ${parsed.error}`);
      if (!response.ok) {
        setError(formatHttpError(response.status, data));
        setResult(data);
        return;
      }
      setPreview(data);
      setPreviewEdit({
        question_text: (data && (data.normalized_question_text || data.extracted_question_text)) || "",
        student_answer: (data && (data.normalized_student_answer || data.extracted_student_answer)) || "",
      });
    } catch (err) {
      const msg = String(err);
      if (msg.includes("AbortError")) setError("Request timed out.");
      else setError(`Network error: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const confirmPreviewAndGrade = async () => {
    if (!previewEdit.question_text.trim()) {
      setError("Question text is required before confirm.");
      return;
    }
    setError("");
    setWarning("");
    setLoading(true);
    setResult(null);
    const payload = {
      question_text: previewEdit.question_text.trim(),
      student_answer: (previewEdit.student_answer || "").trim(),
      ...buildOptionalFilters(),
      debug: Boolean(form.debug),
    };
    try {
      const response = await fetchWithTimeout(
        `${apiBase}/oa-level/evaluate-from-image/confirm`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        300000
      );
      const text = await response.text();
      const parsed = safeJsonParse(text);
      const data = parsed.ok ? parsed.value : { raw: text };
      if (!response.ok) {
        setError(formatHttpError(response.status, data));
        setResult(data);
        return;
      }
      setResult(data);
    } catch (err) {
      setError(`Network error: ${String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (mode === "upload") return submitUpload();
    return submitTyped();
  };

  const onRetry = async () => {
    if (!lastRequest) return;
    if (lastRequest.mode === "upload") return submitUpload();
    return submitTyped();
  };

  const copyText = async (label, text) => {
    const value = String(text || "").trim();
    if (!value) {
      setWarning(`Nothing to copy for ${label}.`);
      return;
    }
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
        setWarning(`Copied ${label}.`);
        return;
      }
    } catch {
      // ignore and fallback
    }
    try {
      const el = document.createElement("textarea");
      el.value = value;
      el.style.position = "fixed";
      el.style.opacity = "0";
      document.body.appendChild(el);
      el.focus();
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      setWarning(`Copied ${label}.`);
    } catch (err) {
      setWarning(`Copy failed: ${String(err)}`);
    }
  };

  const canSubmit =
    !loading &&
    ((mode === "typed" && form.question.trim() && form.student_answer.trim()) ||
      (mode === "upload" && upload.file));

  return (
    <main className="container">
      <section className="card">
        <h1>O/A Levels Evaluator</h1>
        <p className="muted">Minimal typed-question frontend (API: {apiBase})</p>
        {result ? (
          <p className="muted source-note" title="Derived from the response fields.">
            {inferSourceSummary(result)}
          </p>
        ) : (
          <p className="muted source-note">
            Choose a mode below. Mode B sends typed question text. Mode A uploads a PDF/image and OCR extracts question text.
            If the server just started, the first request may take 1–2 minutes while the model loads; use Retry if it times out.
          </p>
        )}

        <ModeToggle mode={mode} setMode={setMode} disabled={loading} />

        <form onSubmit={onSubmit} className="form">
          <label>
            Subject
            <select value={form.subject} onChange={onChange("subject")}>
              {SUBJECT_OPTIONS.map((subject) => (
                <option key={subject} value={subject}>
                  {subject}
                </option>
              ))}
            </select>
          </label>

          {mode === "typed" ? (
            <label>
              Question
              <textarea
                rows="4"
                value={form.question}
                onChange={onChange("question")}
                placeholder="Paste the exact question here"
              />
            </label>
          ) : (
            <label>
              Upload PDF/Image containing the question
              <input
                type="file"
                accept=".pdf,image/png,image/jpeg"
                onChange={onUploadChange("file")}
                disabled={loading}
              />
              <span className="helper">
                {upload.file
                  ? `Selected: ${upload.file.name} (${Math.ceil(upload.file.size / 1024)} KB)`
                  : "Accepted: PDF, PNG, JPG. Max 20MB."}
              </span>
            </label>
          )}

          {mode === "typed" ? (
            <label>
              Student Answer
              <textarea
                rows="3"
                value={form.student_answer}
                onChange={onChange("student_answer")}
                placeholder="Example: B"
              />
            </label>
          ) : null}

          <details>
            <summary>Optional filters</summary>
            <div className="grid">
              <label>
                Year
                <input value={form.year} onChange={onChange("year")} placeholder="2021" />
              </label>
              <label>
                Session
                <input value={form.session} onChange={onChange("session")} placeholder="Oct_Nov" />
                <span className="helper">Example: May_June, Oct_Nov</span>
              </label>
              <label>
                Paper
                <input value={form.paper} onChange={onChange("paper")} placeholder="Paper_1" />
                <span className="helper">Example: Paper_1, Paper_2</span>
              </label>
              <label>
                Variant
                <input value={form.variant} onChange={onChange("variant")} placeholder="Variant_2" />
                <span className="helper">Example: Variant_1, Variant_2, Variant_3</span>
              </label>
            </div>
            <label>
              Question ID (advanced)
              <input
                value={form.question_id}
                onChange={onChange("question_id")}
                placeholder="main|Mathematics 1014|2017|May_June|Paper_2|Variant_2|9(i)"
              />
            </label>
            {mode === "upload" ? (
              <label>
                Page number (Mode A)
                <input value={upload.page_number} onChange={onUploadChange("page_number")} placeholder="1" />
                <span className="helper">1-based page number for PDFs (default 1).</span>
              </label>
            ) : null}
          </details>

          <label className="check">
            <input type="checkbox" checked={form.debug} onChange={onChange("debug")} />
            Enable debug_trace in response
          </label>

          <div className="actions">
            <button type="submit" disabled={!canSubmit}>
              {loading ? "Evaluating..." : "Evaluate"}
            </button>
            <button type="button" onClick={loadSample} disabled={loading} className="secondary">
              Load Sample
            </button>
            <button type="button" onClick={onRetry} disabled={loading || !lastRequest} className="secondary">
              Retry
            </button>
            {mode === "upload" && preview ? (
              <button type="button" onClick={confirmPreviewAndGrade} disabled={loading} className="secondary">
                Confirm & Grade
              </button>
            ) : null}
          </div>
        </form>

        {error ? <p className="error">{error}</p> : null}
        {warning ? <p className="warning">{warning}</p> : null}
      </section>

      <section className="card">
        {mode === "upload" && preview ? (
          <>
            <h2>Mode A Preview</h2>
            <p className="muted">
              Auto-accept eligible: {String(Boolean(preview.auto_accept_eligible))} | reason: {preview.auto_accept_reason || "--"}
            </p>
            {Array.isArray(preview.vision_warnings) && preview.vision_warnings.length ? (
              <p className="warning">Warnings: {preview.vision_warnings.join(", ")}</p>
            ) : null}
            <label>
              Confirm/Edit Question
              <textarea rows="4" value={previewEdit.question_text} onChange={(e) => setPreviewEdit((p) => ({ ...p, question_text: e.target.value }))} />
            </label>
            <label>
              Confirm/Edit Student Answer
              <textarea rows="3" value={previewEdit.student_answer} onChange={(e) => setPreviewEdit((p) => ({ ...p, student_answer: e.target.value }))} />
            </label>
          </>
        ) : null}
        <h2>Result</h2>
        {!result ? <p className="muted">No result yet.</p> : null}
        {result ? (
          <>
            <div className="result-grid">
              <div>
                <strong>Status:</strong>{" "}
                <StatusBadge status={result.status} />
              </div>
              <div><strong>Score:</strong> {formatPercent(result.score_percent)}</div>
              <div><strong>Grade:</strong> {result.grade_label || "--"}</div>
              <div><strong>Match:</strong> {formatPercent((result.match_confidence || 0) * 100)}</div>
              <div><strong>Data source:</strong> {result.data_source || "--"}</div>
              <div><strong>Fallback used:</strong> {String(Boolean(result.fallback_used))}</div>
              <div><strong>Primary:</strong> {result.primary_data_source || "--"}</div>
              <div><strong>Fallback:</strong> {result.fallback_data_source || "--"}</div>
              <div><strong>Reference:</strong> {result.source_paper_reference || "--"}</div>
              <div><strong>Page:</strong> {result.page_number ?? "--"}</div>
              <div><strong>Matched ID:</strong> {result.matched_question_id || "--"}</div>
              <div className="full"><strong>Feedback:</strong> {result.feedback || "--"}</div>
            </div>

            <h3>Markscheme answer</h3>
            <div className="section-actions">
              <button
                type="button"
                className="mini secondary"
                onClick={() => copyText("markscheme answer", result.marking_scheme_answer)}
              >
                Copy markscheme
              </button>
              <button
                type="button"
                className="mini secondary"
                onClick={() => copyText("source reference", result.source_paper_reference)}
              >
                Copy reference
              </button>
            </div>
            <pre className="markscheme">{String(result.marking_scheme_answer || "--")}</pre>

            {Array.isArray(result.top_alternatives) && result.top_alternatives.length ? (
              <>
                <h3>Top alternatives</h3>
                <div className="alternatives">
                  {result.top_alternatives.map((alt) => (
                    <button
                      type="button"
                      key={alt.question_id}
                      className="alt"
                      onClick={() => {
                        setForm((prev) => ({ ...prev, question_id: alt.question_id }));
                        setWarning("Alternative selected. Click Evaluate (or Retry) to re-run using this question_id.");
                      }}
                      disabled={loading}
                      title="Click to set question_id (advanced) to this alternative"
                    >
                      <div className="alt-head">
                        <span className="alt-id">{alt.question_id}</span>
                        <span className="alt-score">{formatPercent((alt.match_confidence || 0) * 100)}</span>
                      </div>
                      <div className="alt-text">{alt.question_text}</div>
                      <div className="alt-ref">{alt.source_paper_reference}</div>
                    </button>
                  ))}
                </div>
              </>
            ) : null}

            {result.debug_trace ? (
              <>
                <h3>Debug trace</h3>
                <details>
                  <summary>Show debug_trace</summary>
                  <pre>{JSON.stringify(result.debug_trace, null, 2)}</pre>
                </details>
              </>
            ) : null}
          </>
        ) : null}

        <h3>Raw JSON</h3>
        {result ? (
          <div className="section-actions">
            <button
              type="button"
              className="mini secondary"
              onClick={() => copyText("raw JSON", JSON.stringify(result, null, 2))}
            >
              Copy JSON
            </button>
          </div>
        ) : null}
        <pre>{result ? JSON.stringify(result, null, 2) : "{}"}</pre>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
