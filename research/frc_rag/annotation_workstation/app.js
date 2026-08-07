const query = new URLSearchParams(window.location.search);
const queryToken = query.get("token") || "";
if (queryToken) sessionStorage.setItem("frc-review-token", queryToken);
const token = queryToken || sessionStorage.getItem("frc-review-token") || "";
window.history.replaceState({}, "", "/");

const elements = {
  app: document.querySelector("#app"),
  fatal: document.querySelector("#fatal-error"),
  fatalMessage: document.querySelector("#fatal-error-message"),
  saveStatus: document.querySelector("#save-status"),
  batchLabel: document.querySelector("#batch-label"),
  annotatorId: document.querySelector("#annotator-id"),
  progressValue: document.querySelector("#progress-value"),
  progressLabel: document.querySelector("#progress-label"),
  progressBar: document.querySelector("#progress-bar"),
  taskList: document.querySelector("#task-list"),
  main: document.querySelector("#review-main"),
  taskPosition: document.querySelector("#task-position"),
  conflictType: document.querySelector("#conflict-type"),
  question: document.querySelector("#question-title"),
  expectedBehavior: document.querySelector("#expected-behavior"),
  correctAnswer: document.querySelector("#correct-answer"),
  comparison: document.querySelector("#response-comparison"),
  preferenceFieldset: document.querySelector("#preference-fieldset"),
  preferenceOptions: document.querySelector("#preference-options"),
  notes: document.querySelector("#decision-notes"),
  previous: document.querySelector("#previous-button"),
  next: document.querySelector("#next-button"),
  save: document.querySelector("#save-button"),
  export: document.querySelector("#export-button"),
  finalize: document.querySelector("#finalize-button"),
  finalizedBanner: document.querySelector("#finalized-banner"),
  reopen: document.querySelector("#reopen-button"),
};

const ratingFields = [
  ["expected_behavior_adherence", "符合期望行为"],
  ["factual_grounding", "事实依据"],
  ["citation_correctness", "引用正确性"],
  ["answer_correctness", "答案正确性"],
];
const ratingChoices = [
  ["PASS", "通过"],
  ["FAIL", "不通过"],
  ["UNCERTAIN", "不确定"],
];
const preferenceChoices = [
  ["A", "回答 A"],
  ["B", "回答 B"],
  ["TIE", "同等"],
  ["NEITHER", "均不合格"],
  ["UNCERTAIN", "不确定"],
];

let state = null;
let currentIndex = 0;
let dirty = false;
let saving = null;
let saveTimer = null;

function node(tagName, className, text) {
  const value = document.createElement(tagName);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function clear(value) {
  while (value.firstChild) value.removeChild(value.firstChild);
}

function isPlaceholderIdentity(value) {
  const normalized = value.trim().toUpperCase();
  return value.trim().length < 3 || normalized.startsWith("REPLACE_WITH");
}

function decisionComplete(decision, task) {
  const allowed = new Set(["PASS", "FAIL", "UNCERTAIN"]);
  for (const alias of ["A", "B"]) {
    const rating = decision.ratings[alias];
    for (const [field] of ratingFields.slice(0, 3)) {
      if (!allowed.has(rating[field])) return false;
    }
    if (task.correct_answer === null) {
      if (rating.answer_correctness !== "NOT_APPLICABLE") return false;
    } else if (!allowed.has(rating.answer_correctness)) {
      return false;
    }
    const rationale = rating.rationale.trim();
    if (rationale.length < 3 || rationale.startsWith("REQUIRED")) return false;
  }
  return preferenceChoices.some(([value]) => value === decision.preference);
}

function localProgress() {
  const statuses = state.draft.decisions.map((decision, index) => ({
    review_task_id: decision.review_task_id,
    index: index + 1,
    complete: decisionComplete(decision, state.batch.tasks[index]),
  }));
  const completed = statuses.filter((item) => item.complete).length;
  const identityReady = !isPlaceholderIdentity(state.draft.annotator_id);
  return {
    completed_task_count: completed,
    total_task_count: statuses.length,
    identity_ready: identityReady,
    batch_complete: completed === statuses.length && identityReady,
    task_status: statuses,
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
    headers: {
      "X-Review-Token": token,
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error || `请求失败：${response.status}`);
    error.details = payload.details || {};
    throw error;
  }
  return payload;
}

function setSaveStatus(text, isError = false) {
  elements.saveStatus.textContent = text;
  elements.saveStatus.classList.toggle("save-error", isError);
}

function updateProgress() {
  state.progress = localProgress();
  const { completed_task_count: completed, total_task_count: total } = state.progress;
  elements.progressValue.textContent = `${completed} / ${total}`;
  elements.progressLabel.textContent = state.progress.batch_complete
    ? "本批内容完整"
    : state.progress.identity_ready
      ? "可继续填写"
      : "还需填写私有标识";
  elements.progressBar.style.width = `${total ? (completed / total) * 100 : 0}%`;
  const taskButtons = elements.taskList.querySelectorAll("button[data-task-index]");
  taskButtons.forEach((button) => {
    const index = Number(button.dataset.taskIndex);
    button.dataset.current = String(index === currentIndex);
    button.dataset.complete = String(state.progress.task_status[index].complete);
    button.setAttribute(
      "aria-label",
      `任务 ${index + 1}，${state.progress.task_status[index].complete ? "已完成" : "未完成"}`,
    );
  });
}

function scheduleSave() {
  dirty = true;
  setSaveStatus("有未保存修改");
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => void flushSave(), 500);
  updateProgress();
}

async function flushSave() {
  window.clearTimeout(saveTimer);
  if (saving) return saving;
  saving = (async () => {
    while (dirty) {
      dirty = false;
      setSaveStatus("正在保存到本机…");
      const payload = JSON.stringify(state.draft);
      try {
        const saved = await api("/api/draft", { method: "PUT", body: payload });
        state.status = saved.status;
        state.receipt = saved.receipt;
        state.progress = saved.progress;
        setSaveStatus("已保存到本机");
      } catch (error) {
        dirty = true;
        setSaveStatus(error.message || "保存失败", true);
        throw error;
      }
    }
  })();
  try {
    await saving;
  } finally {
    saving = null;
  }
}

function choice(name, value, label, current, onChange, disabled = false) {
  const wrapper = node("label", "choice-label");
  const input = document.createElement("input");
  input.type = "radio";
  input.name = name;
  input.value = value;
  input.checked = current === value;
  input.disabled = disabled;
  input.addEventListener("change", () => {
    if (input.checked) onChange(value);
  });
  wrapper.append(input, node("span", "", label));
  return wrapper;
}

function sourceDetails(source) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const summaryBody = node("span", "source-summary");
  summaryBody.append(
    node("span", "source-title", `${source.citation} ${source.title || "未命名来源"}`),
    node("span", "source-date", source.date || "日期未提供"),
  );
  summary.append(summaryBody);
  const content = node("div", "source-content");
  content.append(
    node("p", "source-url", source.url || "URL 未提供"),
    node("p", "source-text", source.text || "来源正文为空"),
  );
  details.append(summary, content);
  return details;
}

function responsePanel(alias, response, rating, taskId, locked) {
  const article = node("article", "response-panel");
  article.dataset.responseAlias = alias;
  const heading = node("header", "response-heading");
  heading.append(node("h3", "", `匿名回答 ${alias}`), node("span", "", `${response.sources.length} 条可见来源`));
  article.append(heading, node("p", "response-body", response.response));

  const sources = node("section", "source-list");
  sources.append(node("h4", "", "回答所依据的来源"));
  response.sources.forEach((source) => sources.append(sourceDetails(source)));
  article.append(sources);

  const form = node("div", "rating-form");
  for (const [field, fieldLabel] of ratingFields) {
    const group = node("fieldset", "rating-group");
    group.append(node("legend", "", fieldLabel));
    const options = node("div", "segmented-options");
    if (field === "answer_correctness" && rating[field] === "NOT_APPLICABLE") {
      options.append(node("span", "not-applicable", "无官方答案，不适用"));
    } else {
      for (const [value, label] of ratingChoices) {
        options.append(
          choice(
            `${taskId}-${alias}-${field}`,
            value,
            label,
            rating[field],
            (nextValue) => {
              rating[field] = nextValue;
              scheduleSave();
            },
            locked,
          ),
        );
      }
    }
    group.append(options);
    form.append(group);
  }

  const rationaleLabel = node("label", "rationale-field");
  rationaleLabel.append(node("span", "", `回答 ${alias} 的评分理由`));
  const rationale = document.createElement("textarea");
  rationale.rows = 4;
  rationale.maxLength = 4000;
  rationale.placeholder = "引用具体回答或来源，简要说明四项评分依据";
  rationale.value = rating.rationale.startsWith("REQUIRED") ? "" : rating.rationale;
  rationale.disabled = locked;
  rationale.setAttribute("aria-label", `回答 ${alias} 的评分理由`);
  rationale.addEventListener("input", () => {
    rating.rationale = rationale.value;
    scheduleSave();
  });
  rationaleLabel.append(rationale);
  form.append(rationaleLabel);
  article.append(form);
  return article;
}

function renderTaskList() {
  clear(elements.taskList);
  state.batch.tasks.forEach((task, index) => {
    const item = document.createElement("li");
    const button = node("button", "task-jump", String(index + 1));
    button.type = "button";
    button.dataset.taskIndex = String(index);
    button.title = task.conflict_type;
    button.addEventListener("click", () => void navigateTo(index));
    item.append(button);
    elements.taskList.append(item);
  });
}

function renderTask() {
  const task = state.batch.tasks[currentIndex];
  const decision = state.draft.decisions[currentIndex];
  const locked = state.status === "FINALIZED";
  elements.taskPosition.textContent = `任务 ${currentIndex + 1} / ${state.batch.tasks.length}`;
  elements.conflictType.textContent = task.conflict_type;
  elements.question.textContent = task.question;
  elements.expectedBehavior.textContent = task.expected_behavior;
  elements.correctAnswer.textContent = task.correct_answer ?? "未提供；答案正确性固定为不适用";

  clear(elements.comparison);
  for (const alias of ["A", "B"]) {
    const response = task.responses.find((item) => item.response_id === alias);
    elements.comparison.append(
      responsePanel(alias, response, decision.ratings[alias], task.review_task_id, locked),
    );
  }

  clear(elements.preferenceOptions);
  for (const [value, label] of preferenceChoices) {
    elements.preferenceOptions.append(
      choice(
        `${task.review_task_id}-preference`,
        value,
        label,
        decision.preference,
        (nextValue) => {
          decision.preference = nextValue;
          scheduleSave();
        },
        locked,
      ),
    );
  }
  elements.preferenceFieldset.disabled = locked;
  elements.notes.value = decision.notes;
  elements.notes.disabled = locked;
  elements.previous.disabled = currentIndex === 0;
  elements.next.disabled = currentIndex === state.batch.tasks.length - 1;
  elements.save.disabled = locked;
  elements.finalize.disabled = locked;
  elements.annotatorId.disabled = locked;
  elements.finalizedBanner.hidden = !locked;
  updateProgress();
  localStorage.setItem(`frc-review-position:${state.batch.operations_id}:${state.batch.batch_id}`, String(currentIndex));
}

async function navigateTo(index) {
  if (index < 0 || index >= state.batch.tasks.length || index === currentIndex) return;
  try {
    await flushSave();
  } catch {
    return;
  }
  currentIndex = index;
  renderTask();
  elements.main.focus({ preventScroll: true });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
}

async function finalizeBatch() {
  try {
    await flushSave();
    setSaveStatus("正在验证整批…");
    const finalized = await api("/api/finalize", {
      method: "POST",
      body: JSON.stringify(state.draft),
    });
    state.status = finalized.status;
    state.receipt = finalized.receipt;
    state.progress = finalized.progress;
    setSaveStatus("本批已验证完成");
    renderTask();
  } catch (error) {
    setSaveStatus(error.message || "整批验证失败", true);
    if (error.details?.first_review_task_id) {
      const index = state.batch.tasks.findIndex(
        (task) => task.review_task_id === error.details.first_review_task_id,
      );
      if (index >= 0) {
        currentIndex = index;
        renderTask();
      }
    }
  }
}

async function reopenBatch() {
  try {
    const reopened = await api("/api/reopen", { method: "POST" });
    state.status = reopened.status;
    state.receipt = null;
    setSaveStatus("已恢复修订状态");
    renderTask();
  } catch (error) {
    setSaveStatus(error.message || "无法恢复修订", true);
  }
}

async function exportDraft() {
  try {
    await flushSave();
    const response = await fetch("/api/export", {
      cache: "no-store",
      headers: { "X-Review-Token": token },
    });
    if (!response.ok) throw new Error("导出失败");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = state.draft_path_label;
    link.click();
    URL.revokeObjectURL(url);
    setSaveStatus("已导出当前草稿");
  } catch (error) {
    setSaveStatus(error.message || "导出失败", true);
  }
}

function installEvents() {
  elements.annotatorId.addEventListener("input", () => {
    state.draft.annotator_id = elements.annotatorId.value;
    scheduleSave();
  });
  elements.notes.addEventListener("input", () => {
    state.draft.decisions[currentIndex].notes = elements.notes.value;
    scheduleSave();
  });
  elements.previous.addEventListener("click", () => void navigateTo(currentIndex - 1));
  elements.next.addEventListener("click", () => void navigateTo(currentIndex + 1));
  elements.save.addEventListener("click", () => void flushSave());
  elements.finalize.addEventListener("click", () => void finalizeBatch());
  elements.reopen.addEventListener("click", () => void reopenBatch());
  elements.export.addEventListener("click", () => void exportDraft());
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden" && dirty) void flushSave();
  });
}

async function start() {
  if (!token) throw new Error("启动链接缺少本机会话令牌。");
  state = await api("/api/session");
  if (!state.privacy?.local_only || state.privacy?.routing_loaded || state.privacy?.method_identity_loaded) {
    throw new Error("工作台隐私边界验证失败。");
  }
  elements.batchLabel.textContent = `${state.batch.reviewer_slot} · 第 ${state.batch.batch_index} 批`;
  elements.annotatorId.value = state.draft.annotator_id.startsWith("REPLACE_WITH")
    ? ""
    : state.draft.annotator_id;
  const storedPosition = Number(
    localStorage.getItem(`frc-review-position:${state.batch.operations_id}:${state.batch.batch_id}`),
  );
  currentIndex = Number.isInteger(storedPosition)
    ? Math.min(Math.max(storedPosition, 0), state.batch.tasks.length - 1)
    : 0;
  renderTaskList();
  installEvents();
  renderTask();
  elements.app.setAttribute("aria-busy", "false");
  setSaveStatus(state.status === "FINALIZED" ? "本批已验证完成" : "草稿已从本机恢复");
}

start().catch((error) => {
  elements.app.hidden = true;
  elements.fatal.hidden = false;
  elements.fatalMessage.textContent = error.message || "未知错误";
});
