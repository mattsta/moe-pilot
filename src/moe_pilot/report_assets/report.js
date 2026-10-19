"use strict";
const DATA = JSON.parse(document.getElementById("research-data").textContent);
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const colors = ["#006d77", "#345ca0", "#a24522", "#775a9d", "#526c34"];
const f = (v, n = 4) => (v == null ? "—" : Number(v).toFixed(n));
const domains = [...new Set(DATA.baseline.map((r) => r.domain))];
domains.forEach((d) =>
  $("domain").insertAdjacentHTML("beforeend", `<option>${esc(d)}</option>`),
);
for (let i = 0; i < 16; i++)
  $("layer").insertAdjacentHTML(
    "beforeend",
    `<option value="${i}">${i}</option>`,
  );
$("layer").value = "7";
$("metrics").innerHTML = [
  [Number(DATA.record.documents).toLocaleString(), "analysis documents"],
  [
    Number(DATA.record.valid_targets).toLocaleString(),
    "valid next-token targets",
  ],
  [f(DATA.record.baseline_mean_loss, 6), "baseline loss · nats/token"],
  [
    Number(DATA.record.conditions).toLocaleString(),
    "held-out intervention conditions",
  ],
]
  .map(
    ([v, k]) =>
      `<div class="metric"><strong>${v}</strong><span>${k}</span></div>`,
  )
  .join("");
function table(rows, fields) {
  return `<table><thead><tr>${fields.map(([k, t]) => `<th>${esc(t)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${fields.map(([k, t, n]) => `<td class="${typeof r[k] === "number" ? "num" : ""}">${typeof r[k] === "number" ? f(r[k], n ?? 4) : esc(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function svg(inner, w = 1000, h = 340) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" font-family="system-ui,sans-serif" font-size="13" fill="#182a38">${inner}</svg>`;
}
function axes(min, max, x, y, h) {
  let s = "";
  for (let i = 0; i < 6; i++) {
    const value = min + ((max - min) * i) / 5,
      px = x(value);
    s += `<line x1="${px}" y1="${y}" x2="${px}" y2="${h}" stroke="#e1e7e4"/><text x="${px}" y="${h + 25}" text-anchor="middle">${f(value, 3)}</text>`;
  }
  return s;
}
function forest(rows) {
  if (!rows.length)
    return '<p class="pending">No completed contrasts for this selection.</p>';
  const vals = rows.flatMap((r) => [r.ci95_low, r.ci95_high, 0]),
    low = Math.min(...vals),
    high = Math.max(...vals),
    pad = Math.max((high - low) * 0.12, 0.0005),
    min = low - pad,
    max = high + pad;
  const w = 1000,
    h = 80 + rows.length * 54,
    gutter = Math.max(
      180,
      Math.min(320, 30 + Math.max(...rows.map((row) => row.domain.length)) * 9),
    ),
    x = (v) => gutter + ((v - min) / (max - min)) * (920 - gutter);
  let s = axes(min, max, x, 20, h - 45);
  s += `<line x1="${x(0)}" x2="${x(0)}" y1="20" y2="${h - 45}" stroke="#526572" stroke-dasharray="4 4"/>`;
  rows.forEach((r, i) => {
    const y = 48 + i * 54,
      c = colors[Math.max(0, domains.indexOf(r.domain)) % colors.length];
    s += `<text x="${gutter - 25}" y="${y + 5}" text-anchor="end">${esc(r.domain)}</text><line x1="${x(r.ci95_low)}" x2="${x(r.ci95_high)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="3"/><circle cx="${x(r.paired_difference)}" cy="${y}" r="5" fill="${c}"><title>${esc(r.domain)}: ${f(r.paired_difference, 6)} [${f(r.ci95_low, 6)}, ${f(r.ci95_high, 6)}]</title></circle>`;
  });
  return svg(s, w, h);
}
function lines(
  rows,
  key,
  { nullKey = null, groups = "domain", height = 330 } = {},
) {
  const ds = [...new Set(rows.map((r) => r[groups] ?? "centered output"))],
    max =
      Math.max(
        ...rows.map((r) => Math.max(r[key], nullKey ? r[nullKey] : 0)),
        0.001,
      ) * 1.15;
  const x = (l) => 75 + (l / 15) * 840,
    y = (v) => height - 65 - (v / max) * (height - 110);
  let s = "";
  for (let i = 0; i < 5; i++) {
    let v = (max * i) / 4;
    s += `<line x1="75" y1="${y(v)}" x2="915" y2="${y(v)}" stroke="#e1e7e4"/><text x="60" y="${y(v) + 4}" text-anchor="end">${f(v, 3)}</text>`;
  }
  for (let l = 0; l < 16; l += 3)
    s += `<text x="${x(l)}" y="${height - 40}" text-anchor="middle">${l}</text>`;
  ds.forEach((d, i) => {
    const rr = rows
      .filter((r) => (r[groups] ?? "centered output") === d)
      .sort((a, b) => a.layer - b.layer);
    s += `<polyline fill="none" stroke="${colors[i % colors.length]}" stroke-width="3" points="${rr.map((r) => `${x(r.layer)},${y(r[key])}`).join(" ")}"/>`;
    if (nullKey)
      s += `<polyline fill="none" stroke="${colors[i % colors.length]}" stroke-dasharray="5 4" points="${rr.map((r) => `${x(r.layer)},${y(r[nullKey])}`).join(" ")}"/>`;
    rr.forEach(
      (r) =>
        (s += `<circle cx="${x(r.layer)}" cy="${y(r[key])}" r="3" fill="${colors[i % colors.length]}"><title>${esc(d)} · layer ${r.layer}: ${f(r[key], 6)}</title></circle>`),
    );
    s += `<text x="${75 + i * 175}" y="20" fill="${colors[i % colors.length]}">${esc(d)}</text>`;
  });
  s += `<text x="500" y="${height - 10}" text-anchor="middle">Decoder layer</text>`;
  return svg(s, 1000, height);
}
function bars(rows, key, label) {
  const max = Math.max(...rows.map((r) => r.upper ?? r[key]), 0.001) * 1.12,
    h = rows.length * 52 + 55;
  let s = "";
  rows.forEach((r, i) => {
    const y = i * 52 + 15,
      width = (r[key] / max) * 600;
    s += `<text x="180" y="${y + 20}" text-anchor="end">${esc(r[label])}</text><rect x="200" y="${y}" width="${width}" height="27" fill="${colors[i % 5]}"/><text x="${215 + ((r.upper ?? r[key]) / max) * 600}" y="${y + 20}">${f(r[key], 3)}</text>`;
    if (r.lower != null && r.upper != null) {
      const low = 200 + (r.lower / max) * 600,
        high = 200 + (r.upper / max) * 600;
      s += `<line x1="${low}" x2="${high}" y1="${y + 13}" y2="${y + 13}" stroke="#182a38" stroke-width="2"/><line x1="${low}" x2="${low}" y1="${y + 7}" y2="${y + 19}" stroke="#182a38"/><line x1="${high}" x2="${high}" y1="${y + 7}" y2="${y + 19}" stroke="#182a38"/>`;
    }
  });
  return svg(s, 1000, h);
}
function render() {
  const domain = $("domain").value,
    layer = Number($("layer").value),
    family = $("family").value,
    filter = (r) => domain === "all" || r.domain === domain;
  const rows = DATA.ablations.filter(
    (r) =>
      filter(r) &&
      r.scope === `layer-${layer}` &&
      r.expert_count === 1 &&
      r.control_family === family,
  );
  $("effect-chart").innerHTML = forest(rows);
  $("balance").innerHTML = table(
    rows.map((r) => ({
      ...r,
      gap:
        (Math.abs(r.specialist_routing_mass - r.control_routing_mass_mean) /
          Math.max(r.specialist_routing_mass, 1e-12)) *
        100,
    })),
    [
      ["domain", "Domain"],
      ["paired_difference", "Excess loss", 6],
      ["ci95_low", "95% low", 6],
      ["ci95_high", "95% high", 6],
      ["gap", "Mass gap (%)", 1],
    ],
  );
  $("effect-note").textContent =
    `Layer ${layer}, one expert, ${family}. The summary refers to the preselected layer-7 on-domain comparison; other selections are exploratory views.`;
  $("routing-chart").innerHTML = lines(
    DATA.routing.filter(filter),
    "jsd_vs_overall",
    { nullKey: "jsd_null_mean" },
  );
  $("baseline-chart").innerHTML = bars(
    DATA.baseline.filter(filter),
    "mean_loss_nats_per_token",
    "domain",
  );
  $("geometry-chart").innerHTML = lines(
    DATA.geometry,
    "mean_abs_cos_centered",
    { groups: "space" },
  );
}
["domain", "layer", "family"].forEach((id) =>
  $(id).addEventListener("change", render),
);
$("reset").onclick = () => {
  $("domain").value = "all";
  $("layer").value = "7";
  $("family").value = "on-domain-control";
  render();
};
$("print").onclick = () => window.print();
document.querySelectorAll("[data-download]").forEach(
  (button) =>
    (button.onclick = () => {
      const rows = DATA[button.dataset.download],
        keys = Object.keys(rows[0]),
        cell = (v) => '"' + String(v ?? "").replaceAll('"', '""') + '"',
        text = [
          keys.map(cell).join(","),
          ...rows.map((r) => keys.map((k) => cell(r[k])).join(",")),
        ].join("\n"),
        url = URL.createObjectURL(new Blob([text], { type: "text/csv" })),
        a = document.createElement("a");
      a.href = url;
      a.download = button.dataset.download + ".csv";
      a.click();
      URL.revokeObjectURL(url);
    }),
);
const modelControls = document.createElement("div");
modelControls.className = "filters";
modelControls.innerHTML =
  '<label>Model comparison<select id="model-metric"><option value="accuracy">Forced-choice accuracy (%)</option><option value="choice_nll">Choice loss (nats/question)</option><option value="loss">Text loss (nats/token)</option></select></label>';
$("models").insertBefore(modelControls, $("model-results"));
$("model-metric").onchange = renderStudy;
const modelNames = {
  "olmoe-0924": "0924 base",
  "olmoe-0125-base": "0125 base",
  "olmoe-0125-sft": "0125 SFT",
  "olmoe-0125-dpo": "0125 DPO",
  "olmoe-0125-instruct": "0125 final",
};
function renderStudy() {
  $("new-effects").innerHTML = "";
  $("progress").innerHTML = DATA.progress.length
    ? table(DATA.progress, [
        ["model", "Model"],
        ["partition", "Partition"],
        ["status", "State"],
        ["completed", "Saved docs", 0],
        ["documents", "Total docs", 0],
        ["seconds", "Seconds", 1],
        ["peak_gib", "Peak RSS GiB", 2],
        ["job_budget_gib", "Budget GiB", 1],
        ["reserve_gib", "Reserve GiB", 1],
      ])
    : '<p class="pending">New study has no completed inference batches yet. No new model comparison is claimed.</p>';
  if (DATA.policy)
    $("progress").nextElementSibling.textContent =
      `One accelerator job at a time; streaming weights; ${DATA.policy.max_memory_gib} GiB job budget; ${DATA.policy.reserve_memory_gib} GiB available-memory reserve. Partial batches are never counted as completed model comparisons.`;
  const stages = DATA.study.baseline_by_model_domain;
  $("model-results").innerHTML = stages?.length
    ? table(stages, [
        ["model", "Model"],
        ["domain", "Domain"],
        ["documents", "Documents", 0],
        ["loss", "Loss nats/token", 5],
        ["answer_loss", "Answer loss", 4],
        ["accuracy", "Choice accuracy", 3],
      ])
    : '<p class="pending">Post-training results are not complete yet. The charts above show the completed 0924 reconstruction.</p>';
  const metric = $("model-metric").value;
  const summary = (DATA.study.model_summary || []).filter(
    (row) => row.metric === metric,
  );
  if (summary.length) {
    const scale = metric === "accuracy" ? 100 : 1;
    const ordered = Object.keys(modelNames);
    const chartRows = summary
      .slice()
      .sort((a, b) => ordered.indexOf(a.model) - ordered.indexOf(b.model))
      .map((row) => ({
        ...row,
        label: modelNames[row.model] || row.model,
        value: row.estimate * scale,
        lower: row.ci_low * scale,
        upper: row.ci_high * scale,
      }));
    $("model-results").insertAdjacentHTML(
      "afterbegin",
      '<div class="chart" role="img" aria-label="Completed model comparison with confidence intervals">' +
        bars(chartRows, "value", "label") +
        '</div><p class="note">Only completed 180-question evaluations are included. Error bars are unadjusted 95% source/domain-stratified bootstrap intervals. Accuracy is forced-choice letter scoring under identical plain prompts; it is not a general reasoning score.</p>',
    );
  }
  const paired = DATA.study.posttraining_contrasts;
  if (paired?.length) {
    const rows = paired.map((row) => ({
      ...row,
      metric:
        row.metric === "accuracy"
          ? "Accuracy (percentage points)"
          : row.metric === "choice_nll"
            ? "Choice loss (nats/question)"
            : "Text loss (nats/token)",
      difference: row.difference * (row.metric === "accuracy" ? 100 : 1),
      ci_low: row.ci_low * (row.metric === "accuracy" ? 100 : 1),
      ci_high: row.ci_high * (row.metric === "accuracy" ? 100 : 1),
    }));
    $("model-results").insertAdjacentHTML(
      "beforeend",
      "<h3>Paired differences from 0125 base</h3>" +
        table(rows, [
          ["model", "Stage"],
          ["metric", "Metric"],
          ["difference", "Difference", 4],
          ["ci_low", "95% low", 4],
          ["ci_high", "95% high", 4],
        ]) +
        '<p class="note">These are observational training-stage contrasts on the same questions. Per-token differences are shown only when token IDs agree exactly.</p>',
    );
  }
  const balance = DATA.study.control_balance;
  if (balance?.length) {
    const passed = balance.filter(
      (row) => row.balance_passed === true || row.balance_passed === "True",
    ).length;
    $("new-effects").innerHTML =
      '<h3>Calibration control balance</h3><p class="caution">' +
      passed +
      " of " +
      balance.length +
      " domains passed every declared threshold (mass/usage ≤10%, norm ≤20%). Failed matches keep their intervention results exploratory and prevent semantic-specialization claims.</p>" +
      table(
        balance.map((row) => ({
          ...row,
          on_mass: row.on_mass * 100,
          off_mass: row.off_mass * 100,
          on_usage: row.on_usage * 100,
          off_usage: row.off_usage * 100,
          raw_output_norm: row.raw_output_norm * 100,
        })),
        [
          ["domain", "Domain"],
          ["expert", "Expert", 0],
          ["control", "Control", 0],
          ["on_mass", "On-mass gap (%)", 1],
          ["off_mass", "Off-mass gap (%)", 1],
          ["on_usage", "On-usage gap (%)", 1],
          ["off_usage", "Off-usage gap (%)", 1],
          ["raw_output_norm", "Norm gap (%)", 1],
        ],
      );
  }
  const effects = DATA.study.causal_effects;
  if (effects?.length)
    $("new-effects").innerHTML +=
      "<h3>New held-out interaction contrasts</h3>" +
      table(effects, [
        ["domain", "Domain"],
        ["mode", "Mode"],
        ["interaction", "On/off excess", 6],
        ["ci_low", "95% low", 6],
        ["ci_high", "95% high", 6],
        ["holm_p", "Holm p", 4],
        ["balance_passed", "Balance passed"],
      ]) +
      '<div class="chart" role="img" aria-label="New held-out primary interaction intervals">' +
      forest(
        effects
          .filter((row) => row.primary === true || row.primary === "True")
          .map((row) => ({
            ...row,
            domain: row.domain + " / " + row.mode,
            paired_difference: row.interaction,
            ci95_low: row.ci_low,
            ci95_high: row.ci_high,
          })),
      ) +
      '</div><p class="note">Primary endpoints are removal and rerouting across three domains (Holm family of six). Rescaling and substitution are exploratory secondary outcomes. Intervals are unadjusted. Zero-strength restoration is checked exactly before analysis. Source/lexical sensitivity coefficients are available in the complete CSV.</p><a href="data/study/causal_effects.csv">Download all intervention contrasts</a>';
  renderInternals();
  renderFormats();
}

function renderInternals() {
  const routing = (DATA.study.routing_shift || []).filter(
    (row) => row.domain === "all",
  );
  const norms = DATA.study.expert_norm_changes || [];
  let html = "";
  if (routing.length) {
    html +=
      '<h3>Layer-7 routing changes from 0125 base</h3><div class="chart" role="img" aria-label="Routing divergence from base">' +
      bars(
        routing.map((row) => ({
          ...row,
          label: modelNames[row.model],
          value: row.mean_jsd_bits,
          lower: row.ci_low,
          upper: row.ci_high,
        })),
        "value",
        "label",
      ) +
      '</div><p class="note">Mean per-question Jensen–Shannon divergence of selected-gate mass distributions, in bits, with 95% source/domain-stratified bootstrap intervals. Divergence measures a routing change; it does not measure improved reasoning.</p><a href="data/study/routing_shift.csv">Download routing by stage and domain</a>';
  }
  if (norms.length) {
    const models = [...new Set(norms.map((row) => row.model))];
    const limit = Math.max(
      ...norms.map((row) => Math.abs(row.mean_log_relative_norm_change)),
      0.001,
    );
    let shapes = "";
    models.forEach((model, i) => {
      shapes += `<text x="140" y="${65 + i * 55}" text-anchor="end">${esc(modelNames[model])}</text>`;
      norms
        .filter((row) => row.model === model)
        .forEach((row) => {
          const value = row.mean_log_relative_norm_change;
          const strength = Math.min(1, Math.abs(value) / limit);
          const color =
            value >= 0
              ? `rgb(${Math.round(255 - 90 * strength)},${Math.round(255 - 186 * strength)},${Math.round(255 - 221 * strength)})`
              : `rgb(${Math.round(255 - 203 * strength)},${Math.round(255 - 163 * strength)},${Math.round(255 - 95 * strength)})`;
          shapes += `<rect x="${160 + row.expert * 12}" y="${42 + i * 55}" width="12" height="38" fill="${color}"><title>${esc(modelNames[model])} / expert ${row.expert}: log ratio ${f(value, 5)}, fold ${f(row.relative_norm_fold_change, 4)}</title></rect>`;
        });
    });
    for (let expert = 0; expert < 64; expert += 8)
      shapes += `<text x="${166 + expert * 12}" y="${65 + models.length * 55}" text-anchor="middle">${expert}</text>`;
    shapes += `<text x="545" y="${100 + models.length * 55}" text-anchor="middle">Expert index · blue = smaller, red = larger relative norm · ±${f(limit, 3)} log units</text>`;
    html +=
      '<h3>Relative expert-output magnitude changes</h3><div class="chart" role="img" aria-label="Heatmap of relative expert-output norm changes">' +
      svg(shapes, 1000, 125 + models.length * 55) +
      '</div><p class="note">Within the 0125 lineage, matched token positions have model-specific incoming hidden states. Norms are divided by each document’s mean expert norm before log ratios. This measures output magnitude, not function direction, knowledge, or causal necessity. Hover a cell for its value.</p><a href="data/study/expert_norm_changes.csv">Download all expert values</a>';
  }
  $("internal-results").innerHTML = html;
}

function renderFormats() {
  const rows = DATA.study.format_effects || [];
  if (!rows.length) {
    $("format-results").innerHTML =
      '<p class="pending">No completed paired native-chat model comparison yet. Saved partial runs appear in execution progress.</p>';
    return;
  }
  const accuracy = rows
    .filter((row) => row.metric === "accuracy")
    .map((row) => ({
      ...row,
      domain: modelNames[row.model],
      paired_difference: 100 * row.estimate,
      ci95_low: 100 * row.ci_low,
      ci95_high: 100 * row.ci_high,
    }));
  const shown = rows.map((row) => ({
    ...row,
    metric:
      row.metric === "accuracy"
        ? "Accuracy (percentage points)"
        : row.metric === "choice_nll"
          ? "Choice loss (nats/question)"
          : row.metric === "aligned_loss"
            ? "Aligned-span loss (nats/token)"
            : "Routing divergence (bits)",
    estimate: row.estimate * (row.metric === "accuracy" ? 100 : 1),
    ci_low: row.ci_low * (row.metric === "accuracy" ? 100 : 1),
    ci_high: row.ci_high * (row.metric === "accuracy" ? 100 : 1),
  }));
  $("format-results").innerHTML =
    '<h3>Native-chat minus plain accuracy (percentage points)</h3><div class="chart" role="img" aria-label="Paired chat-format accuracy effects">' +
    forest(accuracy) +
    "</div>" +
    table(shown, [
      ["model", "Checkpoint"],
      ["metric", "Outcome"],
      ["estimate", "Estimate", 5],
      ["ci_low", "95% low", 5],
      ["ci_high", "95% high", 5],
      ["pairs", "Pairs", 0],
    ]) +
    '<p class="note">Signed effects are chat minus plain; lower loss is better. Routing divergence is nonnegative and has no direction. All intervals are paired, source/domain-stratified, and unadjusted. Whole-prompt per-token loss is deliberately not contrasted across different token sequences.</p>' +
    table(DATA.study.format_alignment || [], [
      ["model", "Checkpoint"],
      ["pairs", "Pairs", 0],
      ["minimum_question_coverage", "Minimum aligned fraction", 4],
      ["mean_question_coverage", "Mean aligned fraction", 4],
    ]) +
    '<a href="data/study/format_paired_documents.csv">Inspect every paired question effect</a>';
}
renderStudy();
const refreshButton = document.createElement("button");
refreshButton.textContent = "Refresh study results";
refreshButton.id = "refresh-study";
$("models").insertBefore(refreshButton, $("model-results"));
async function refreshStudy() {
  if (location.protocol === "file:") {
    refreshButton.textContent = "Use serve-report for live updates";
    return;
  }
  try {
    const response = await fetch("/api/study", { cache: "no-store" });
    if (!response.ok) throw new Error("Refresh unavailable");
    const snapshot = await response.json();
    Object.assign(DATA, snapshot);
    renderStudy();
    refreshButton.textContent = "Updated " + new Date().toLocaleTimeString();
  } catch (error) {
    refreshButton.textContent = "Refresh unavailable — saved results retained";
  }
}
refreshButton.onclick = refreshStudy;
if (location.protocol !== "file:") setInterval(refreshStudy, 30000);
const interactions = DATA.study.reconstruction_interactions;
if (interactions?.length) {
  const rows = interactions.map((r) => ({
    ...r,
    paired_difference: r.interaction,
    ci95_low: r.ci_low,
    ci95_high: r.ci_high,
  }));
  $("interaction-results").innerHTML =
    '<div class="eyebrow">NEW ANALYSIS OF COMPLETED INTERVENTIONS</div><h2>Do the effects exceed what happens off-domain?</h2><p>This contrast subtracts off-domain specialist-minus-control effects from on-domain effects. It uses all five layer-7, single-expert comparisons, 10,000 within-domain paired bootstrap replicates and 9,999 label permutations, with Holm correction across five tests.</p>' +
    '<div class="chart" role="img" aria-label="On-domain minus off-domain effects">' +
    forest(rows) +
    "</div>" +
    table(interactions, [
      ["domain", "Domain"],
      ["interaction", "On/off interaction", 6],
      ["ci_low", "95% low", 6],
      ["ci_high", "95% high", 6],
      ["holm_p", "Holm p", 4],
    ]) +
    '<p class="caution"></p><a href="data/study/reconstruction_interactions.csv">Download interaction table</a> · <a href="data/study/reconstruction_paired_documents.csv">Inspect all paired document effects</a>';
}
if (interactions?.length) {
  const surviving = interactions
    .filter((row) => row.holm_p < 0.05)
    .map((row) => row.domain);
  $("interaction-results").querySelector(".caution").textContent =
    `${surviving.length === 1 ? "Only " + surviving[0] : surviving.join(", ") || "No domain"} clears the Holm threshold in this sample using a studentized permutation test. This is a post-hoc reanalysis; source/domain confounding and large control-mass gaps remain. It does not establish semantic expert necessity. Confidence intervals are unadjusted.`;
}
render();
const exported = DATA.figures || {};
$("figure-downloads").innerHTML = Object.keys(exported).length
  ? "<h3>Export figures for reading and reuse</h3><ul>" +
    Object.entries(exported)
      .map(
        ([name, item]) =>
          "<li><strong>" +
          esc(name.replaceAll("_", " ")) +
          "</strong> · " +
          ["png", "svg", "pdf"]
            .map(
              (extension) =>
                '<a href="figures/' +
                esc(name) +
                "." +
                extension +
                '">' +
                extension.toUpperCase() +
                "</a>",
            )
            .join(" · ") +
          '<p class="note">' +
          esc(item.caption) +
          "</p></li>",
      )
      .join("") +
    "</ul>"
  : "";
