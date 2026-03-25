const form = document.getElementById("eval-form");
const sampleBtn = document.getElementById("sample-btn");
const submitBtn = document.getElementById("submit-btn");
const output = document.getElementById("result-output");
const statusChip = document.getElementById("status-chip");
const scoreEl = document.getElementById("result-score");
const gradeEl = document.getElementById("result-grade");
const sourceEl = document.getElementById("result-source");
const confidenceEl = document.getElementById("result-match-confidence");
const referenceEl = document.getElementById("result-reference");
const pageEl = document.getElementById("result-page");
const feedbackEl = document.getElementById("result-feedback");
const expectedEl = document.getElementById("result-expected");
const missingEl = document.getElementById("result-missing");
const debugEl = document.getElementById("result-debug");

function setStatus(status) {
  statusChip.className = "chip";
  if (status === "accepted" || status === "review_required" || status === "failed" || status === "running") {
    statusChip.classList.add(status);
  } else {
    statusChip.classList.add("neutral");
  }
  statusChip.textContent = status || "idle";
}

function clearList(el, emptyText) {
  el.innerHTML = "";
  const li = document.createElement("li");
  li.textContent = emptyText;
  el.appendChild(li);
}

function renderList(el, items, emptyText) {
  el.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    clearList(el, emptyText);
    return;
  }
  items.slice(0, 8).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = String(item);
    el.appendChild(li);
  });
}

function asPercent(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${Math.max(0, Math.min(100, value)).toFixed(1)}%`;
  }
  return "--";
}

function asConfidence(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
  }
  return "--";
}

function resetSummary(message) {
  scoreEl.textContent = "--";
  gradeEl.textContent = "--";
  sourceEl.textContent = "--";
  confidenceEl.textContent = "--";
  referenceEl.textContent = "--";
  pageEl.textContent = "--";
  feedbackEl.textContent = message || "Run an evaluation to see feedback.";
  clearList(expectedEl, "No expected points yet.");
  clearList(missingEl, "No missing points yet.");
  debugEl.textContent = "Debug disabled.";
}

function renderDebugSummary(trace) {
  if (!trace || typeof trace !== "object") {
    debugEl.textContent = "Debug disabled.";
    return;
  }
  const timings = trace.timings_ms || {};
  const finalDecision = trace.final_decision || {};
  const parts = [
    `Total ${timings.total ?? "--"} ms`,
    `Primary ${timings.primary ?? "--"} ms`,
  ];
  if (typeof timings.fallback !== "undefined") {
    parts.push(`Fallback ${timings.fallback} ms`);
  }
  if (finalDecision.matched_question_id) {
    parts.push(`Match ID ${finalDecision.matched_question_id}`);
  }
  debugEl.textContent = parts.join(" | ");
}

function renderResult(data) {
  scoreEl.textContent = asPercent(data.score_percent);
  gradeEl.textContent = data.grade_label || "--";
  sourceEl.textContent = data.data_source || "--";
  confidenceEl.textContent = asConfidence(data.match_confidence);
  referenceEl.textContent = data.source_paper_reference || "--";
  pageEl.textContent = data.page_number ?? "--";
  feedbackEl.textContent = data.feedback || "No feedback generated.";
  renderList(expectedEl, data.expected_points, "No expected points returned.");
  renderList(missingEl, data.missing_points, "No missing points identified.");
  renderDebugSummary(data.debug_trace);
}

function optionalText(id) {
  const value = (document.getElementById(id).value || "").trim();
  return value || null;
}

function optionalInt(id) {
  const value = (document.getElementById(id).value || "").trim();
  if (!value) {
    return null;
  }
  const num = Number(value);
  if (!Number.isInteger(num) || num <= 0) {
    return null;
  }
  return num;
}

function buildPayload() {
  const payload = {
    question: (document.getElementById("question").value || "").trim(),
    student_answer: (document.getElementById("student-answer").value || "").trim(),
    subject: optionalText("subject"),
    debug: Boolean(document.getElementById("debug").checked),
  };

  const year = optionalInt("year");
  const session = optionalText("session");
  const paper = optionalText("paper");
  const variant = optionalText("variant");
  const questionId = optionalText("question-id");

  if (year) payload.year = year;
  if (session) payload.session = session;
  if (paper) payload.paper = paper;
  if (variant) payload.variant = variant;
  if (questionId) payload.question_id = questionId;

  return payload;
}

sampleBtn.addEventListener("click", () => {
  document.getElementById("subject").value = "Chemistry 1011";
  document.getElementById("question").value = "Which row correctly identifies the gas?";
  document.getElementById("student-answer").value = "D";
  document.getElementById("year").value = "";
  document.getElementById("session").value = "";
  document.getElementById("paper").value = "";
  document.getElementById("variant").value = "";
  document.getElementById("question-id").value = "";
  document.getElementById("debug").checked = true;
});

resetSummary("Run an evaluation to see feedback.");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = buildPayload();

  setStatus("running");
  submitBtn.disabled = true;
  submitBtn.textContent = "Evaluating...";
  resetSummary("Processing request...");
  output.textContent = JSON.stringify({ request: payload }, null, 2);

  try {
    const response = await fetch("/oa-level/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const rawBody = await response.text();
    let data = {};
    try {
      data = rawBody ? JSON.parse(rawBody) : {};
    } catch (_err) {
      data = { raw_response: rawBody };
    }
    if (!response.ok) {
      setStatus("failed");
      resetSummary("Request failed. Check error details below.");
      output.textContent = JSON.stringify(
        {
          error: `HTTP ${response.status}`,
          detail: data,
        },
        null,
        2
      );
      return;
    }

    setStatus(data.status || "done");
    renderResult(data);
    output.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    setStatus("failed");
    resetSummary("Network or server error.");
    output.textContent = JSON.stringify(
      {
        error: "Network or server error",
        detail: String(error),
      },
      null,
      2
    );
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Evaluate";
  }
});

setStatus("idle");
