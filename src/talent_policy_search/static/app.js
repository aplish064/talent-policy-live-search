const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const statusEl = document.querySelector("#status");
const metaEl = document.querySelector("#meta");
const warningsEl = document.querySelector("#warnings");
const resultsEl = document.querySelector("#results");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();

  if (!query) {
    setStatus("请输入地区或学校。", "error");
    queryInput.focus();
    return;
  }

  setBusy(true);
  setStatus("正在实时检索官方来源，通常需要 1-3 分钟。", "loading");
  metaEl.replaceChildren();
  warningsEl.replaceChildren();
  resultsEl.replaceChildren();

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({query}),
    });

    if (!response.ok) {
      throw new Error(await responseErrorMessage(response));
    }

    const payload = await response.json();
    renderResponse(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "检索失败。";
    setStatus(message, "error");
  } finally {
    setBusy(false);
  }
});

function renderResponse(payload) {
  const resultCount = numberValue(payload.results_returned);
  const seconds = numberValue(payload.duration_seconds);
  const query = textValue(payload.query);
  const normalized = textValue(payload.normalized_query);
  const resultText = resultCount === 1 ? "1 条结果" : `${resultCount} 条结果`;

  setStatus(`完成：${resultText}，耗时 ${formatNumber(seconds)} 秒。结果未保存。`, "done");
  metaEl.replaceChildren(
    metric("查询", query),
    metric("规范化", normalized || query),
    metric("官方来源", numberValue(payload.official_sources_checked)),
    metric("候选页面", numberValue(payload.candidate_pages_seen)),
    metric("持久化", textValue(payload.persistence) || "none"),
  );
  warningsEl.replaceChildren(...arrayValue(payload.warnings).map(renderWarning));

  const results = arrayValue(payload.results);
  const summaryMarkdown = textValue(payload.summary_markdown);
  if (summaryMarkdown) {
    const nodes = [renderMarkdownSummary(summaryMarkdown)];
    if (results.length > 0) {
      nodes.push(renderStructuredResults(results));
    }
    resultsEl.replaceChildren(...nodes);
    return;
  }
  if (results.length === 0) {
    resultsEl.replaceChildren(emptyResult());
    return;
  }
  resultsEl.replaceChildren(...results.map(renderPolicy));
}

function renderMarkdownSummary(markdown) {
  const article = document.createElement("article");
  article.className = "markdown-summary";
  for (const line of markdown.split(/\r?\n/)) {
    article.append(renderMarkdownLine(line));
  }
  return article;
}

function renderMarkdownLine(line) {
  const trimmed = line.trim();
  if (!trimmed) {
    const spacer = document.createElement("div");
    spacer.className = "markdown-spacer";
    return spacer;
  }

  const headingMatch = /^(#{1,4})\s+(.+)$/.exec(trimmed);
  if (headingMatch) {
    const level = Math.min(headingMatch[1].length + 1, 4);
    const heading = document.createElement(`h${level}`);
    appendInlineMarkdown(heading, headingMatch[2]);
    return heading;
  }

  const bulletMatch = /^[-*]\s+(.+)$/.exec(trimmed);
  if (bulletMatch) {
    const item = document.createElement("p");
    item.className = "markdown-list-line";
    item.append(textNode("• "));
    appendInlineMarkdown(item, bulletMatch[1]);
    return item;
  }

  const numberedMatch = /^(\d+)\.\s+(.+)$/.exec(trimmed);
  if (numberedMatch) {
    const item = document.createElement("p");
    item.className = "markdown-list-line markdown-numbered-line";
    item.append(textNode(`${numberedMatch[1]}. `));
    appendInlineMarkdown(item, numberedMatch[2]);
    return item;
  }

  const paragraphNode = document.createElement("p");
  appendInlineMarkdown(paragraphNode, trimmed);
  return paragraphNode;
}

function appendInlineMarkdown(parent, text) {
  const linkPattern = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;
  let cursor = 0;
  for (const match of text.matchAll(linkPattern)) {
    if (match.index > cursor) {
      parent.append(textNode(stripBold(text.slice(cursor, match.index))));
    }
    const link = document.createElement("a");
    link.href = match[2];
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = stripBold(match[1]);
    parent.append(link);
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    parent.append(textNode(stripBold(text.slice(cursor))));
  }
}

function stripBold(text) {
  return text.replace(/\*\*/g, "");
}

function renderStructuredResults(results) {
  const details = document.createElement("details");
  details.className = "structured-results";
  const summary = document.createElement("summary");
  summary.textContent = "结构化结果";
  details.append(summary, ...results.map(renderPolicy));
  return details;
}

function renderWarning(warning) {
  const item = document.createElement("article");
  item.className = "warning";
  item.append(
    labeledText("来源", textValue(warning.source) || "unknown"),
    paragraph(textValue(warning.message) || "未提供警告详情。"),
  );
  return item;
}

function renderPolicy(policy) {
  const article = document.createElement("article");
  article.className = "policy";

  const header = document.createElement("header");
  header.className = "policy-header";

  const title = document.createElement("h2");
  const link = document.createElement("a");
  link.href = textValue(policy.official_url) || "#";
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = textValue(policy.title) || "未命名政策";
  title.append(link);

  const quality = document.createElement("div");
  quality.className = "quality";
  quality.append(
    scorePill("score", policy.score),
    scorePill("confidence", policy.confidence),
    scorePill("complete", policy.completeness),
  );

  header.append(title, quality);

  article.append(
    header,
    chipRow([
      textValue(policy.source_name),
      textValue(policy.matched_entity),
      textValue(policy.jurisdiction),
      ...arrayValue(policy.policy_types),
    ]),
    summaryBlock(policy),
    detailGrid(policy),
    evidenceList(policy),
  );

  return article;
}

function summaryBlock(policy) {
  const block = document.createElement("section");
  block.className = "summary";
  block.append(sectionTitle("摘要"), paragraph(textValue(policy.summary) || "未提供摘要。"));
  return block;
}

function detailGrid(policy) {
  const grid = document.createElement("section");
  grid.className = "detail-grid";
  grid.append(
    detailGroup("适用对象", arrayValue(policy.applicable_to).join("；") || "未提取"),
    benefitsGroup(policy),
    eligibilityGroup(policy),
    applicationGroup(policy.application || {}),
    datesGroup(policy.dates || {}),
  );
  return grid;
}

function benefitsGroup(policy) {
  const benefits = arrayValue(policy.benefits);
  if (benefits.length === 0) {
    return detailGroup("待遇", "未提取");
  }
  return richGroup(
    "待遇",
    benefits.map((benefit) => {
      const parts = [
        textValue(benefit.type),
        textValue(benefit.amount_text),
        textValue(benefit.currency),
      ].filter(Boolean);
      return {
        main: parts.join(" · ") || "未命名待遇",
        sub: textValue(benefit.evidence),
      };
    }),
  );
}

function eligibilityGroup(policy) {
  const items = arrayValue(policy.eligibility);
  if (items.length === 0) {
    return detailGroup("申报条件", "未提取");
  }
  return richGroup(
    "申报条件",
    items.map((item) => ({
      main: textValue(item.condition) || "未命名条件",
      sub: textValue(item.evidence),
    })),
  );
}

function applicationGroup(application) {
  const entryUrl = textValue(application.entry_url);
  const materials = arrayValue(application.materials);
  const lines = [];

  if (entryUrl) {
    const link = document.createElement("a");
    link.href = entryUrl;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = "申请入口";
    lines.push(link);
  }
  if (materials.length > 0) {
    lines.push(textNode(`材料：${materials.join("；")}`));
  }
  if (application.process) {
    lines.push(textNode(`流程：${textValue(application.process)}`));
  }
  if (application.evidence) {
    lines.push(textNode(`依据：${textValue(application.evidence)}`));
  }

  return mixedGroup("申请", lines.length > 0 ? lines : [textNode("未提取")]);
}

function datesGroup(dates) {
  const entries = [
    ["发布日期", dates.published_date],
    ["生效日期", dates.effective_date],
    ["截止日期", dates.deadline],
    ["有效期至", dates.valid_until],
  ]
    .map(([label, value]) => [label, textValue(value)])
    .filter(([, value]) => value);

  if (entries.length === 0) {
    return detailGroup("日期", "未提取");
  }

  return richGroup(
    "日期",
    entries.map(([label, value]) => ({main: `${label}：${value}`})),
  );
}

function evidenceList(policy) {
  const snippets = arrayValue(policy.evidence_snippets);
  const block = document.createElement("section");
  block.className = "evidence-block";
  block.append(sectionTitle("证据片段"));

  if (snippets.length === 0) {
    block.append(paragraph("未提取。"));
    return block;
  }

  const list = document.createElement("ul");
  list.className = "evidence";
  for (const snippet of snippets) {
    const item = document.createElement("li");
    item.textContent = textValue(snippet);
    list.append(item);
  }
  block.append(list);
  return block;
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const name = document.createElement("span");
  name.className = "metric-label";
  name.textContent = label;
  const data = document.createElement("strong");
  data.textContent = String(value || "0");
  item.append(name, data);
  return item;
}

function chipRow(values) {
  const row = document.createElement("div");
  row.className = "chips";
  const chips = values.filter(Boolean).map((value) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = value;
    return chip;
  });
  row.replaceChildren(...chips);
  return row;
}

function scorePill(label, value) {
  const pill = document.createElement("span");
  pill.className = "score-pill";
  pill.textContent = `${label} ${formatNumber(numberValue(value))}`;
  return pill;
}

function detailGroup(title, value) {
  const group = document.createElement("section");
  group.className = "detail-group";
  group.append(sectionTitle(title), paragraph(value));
  return group;
}

function richGroup(title, items) {
  const group = document.createElement("section");
  group.className = "detail-group";
  group.append(sectionTitle(title));

  const list = document.createElement("ul");
  list.className = "compact-list";
  for (const item of items) {
    const row = document.createElement("li");
    const main = document.createElement("span");
    main.textContent = item.main;
    row.append(main);
    if (item.sub) {
      const sub = document.createElement("small");
      sub.textContent = item.sub;
      row.append(sub);
    }
    list.append(row);
  }
  group.append(list);
  return group;
}

function mixedGroup(title, nodes) {
  const group = document.createElement("section");
  group.className = "detail-group";
  group.append(sectionTitle(title));
  for (const node of nodes) {
    const line = document.createElement("p");
    line.append(node);
    group.append(line);
  }
  return group;
}

function labeledText(label, value) {
  const row = document.createElement("p");
  row.className = "labeled-text";
  const name = document.createElement("strong");
  name.textContent = `${label}: `;
  row.append(name, textNode(value));
  return row;
}

function sectionTitle(text) {
  const heading = document.createElement("h3");
  heading.textContent = text;
  return heading;
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function emptyResult() {
  const item = document.createElement("article");
  item.className = "empty";
  item.textContent = "未返回匹配政策。";
  return item;
}

function textNode(text) {
  return document.createTextNode(text);
}

function textValue(value) {
  return typeof value === "string" ? value.trim() : "";
}

function numberValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function arrayValue(value) {
  return Array.isArray(value) ? value : [];
}

function formatNumber(value) {
  return value.toLocaleString("zh-CN", {maximumFractionDigits: 2});
}

function setStatus(message, state) {
  statusEl.textContent = message;
  statusEl.dataset.state = state;
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  queryInput.disabled = isBusy;
  submitButton.textContent = isBusy ? "检索中" : "检索";
}

async function responseErrorMessage(response) {
  try {
    const payload = await response.json();
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg).join("；")
      : payload.detail;
    return detail || `请求失败：${response.status}`;
  } catch {
    return `请求失败：${response.status}`;
  }
}
