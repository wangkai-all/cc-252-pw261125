/* audio-sync UI: polls /api/state, /api/index, /api/segments; renders device
   controls, coverage timeline, gaps, alignment report, provenance. */
"use strict";

const $ = (s) => document.querySelector(s);
const fmt = (x, d = 2) => (x === null || x === undefined) ? "—" : Number(x).toFixed(d);

async function api(path, method = "GET", body) {
  const r = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  return r.json();
}

/* ---------------- devices ---------------- */

function deviceHtml(d) {
  const c = d.config;
  const on = c.running;
  return `<div class="device" data-id="${d.device_id}">
    <div class="row">
      <span class="name">${d.device_id}</span>
      <button class="${on ? "on" : "off"}" data-act="${on ? "stop" : "start"}">
        ${on ? "■ 停止" : "▶ 发送"}</button>
      <button data-act="reboot">重启(计数清零)</button>
      <span class="st">状态 ${d.status} · run ${d.run_id.slice(0, 6)} ·
        counter ${d.counter} · 积压 ${d.backlog}</span>
    </div>
    <div class="row">
      <label>采样率 <select data-cfg="sample_rate">
        ${[8000, 16000, 22050, 44100, 48000].map(r =>
          `<option ${r === c.sample_rate ? "selected" : ""}>${r}</option>`).join("")}
      </select></label>
      <label>段长ms <input data-cfg="segment_ms" type="number" value="${c.segment_ms}"></label>
      <label>块字节 <input data-cfg="chunk_size" type="number" value="${c.chunk_size}"></label>
      <label>丢块 <input data-cfg="drop_prob" type="number" step="0.01" min="0" max="1" value="${c.drop_prob}"></label>
      <label>损坏 <input data-cfg="corrupt_prob" type="number" step="0.01" min="0" max="1" value="${c.corrupt_prob}"></label>
      <label>重复 <input data-cfg="dup_prob" type="number" step="0.01" min="0" max="1" value="${c.dup_prob}"></label>
      <label>冲突 <input data-cfg="conflict_prob" type="number" step="0.01" min="0" max="1" value="${c.conflict_prob}"></label>
      <label>离线率 <input data-cfg="offline_prob" type="number" step="0.01" min="0" max="1" value="${c.offline_prob}"></label>
      <button data-act="apply">应用配置</button>
    </div>
  </div>`;
}

function renderDevices(state) {
  $("#devices").innerHTML = state.devices.map(deviceHtml).join("") ||
    "<p>无设备，点「+ 新设备」。</p>";
  document.querySelectorAll(".device").forEach(el => {
    const id = el.dataset.id;
    el.querySelectorAll("button").forEach(btn => btn.onclick = async () => {
      const act = btn.dataset.act;
      if (act === "apply") {
        const cfg = {};
        el.querySelectorAll("[data-cfg]").forEach(inp => {
          cfg[inp.dataset.cfg] = inp.dataset.cfg.endsWith("prob") ||
            inp.dataset.cfg === "offline_prob"
            ? parseFloat(inp.value) : parseInt(inp.value, 10);
        });
        await api(`/api/device/${id}/config`, "POST", cfg);
      } else {
        await api(`/api/device/${id}/${act}`, "POST", {});
      }
      refresh();
    });
  });
}

/* ---------------- timeline ---------------- */

function renderTimeline(idx) {
  const cv = $("#timeline");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!idx || !idx.report) { ctx.fillText("等待索引…", 10, 20); return; }
  const devices = Object.values(idx.report.devices);
  if (!devices.length) { ctx.fillText("暂无已校验片段", 10, 20); return; }
  let tMin = Infinity, tMax = -Infinity;
  for (const d of devices)
    for (const ep of d.epochs)
      for (const [a, b] of ep.covered_time) { tMin = Math.min(tMin, a); tMax = Math.max(tMax, b); }
  if (!isFinite(tMin)) { ctx.fillText("暂无可映射区间", 10, 20); return; }
  tMin -= 0.5; tMax += 0.5;
  const X = t => 8 + (t - tMin) / (tMax - tMin) * (cv.width - 16);
  const rowH = Math.min(46, (cv.height - 30) / devices.length);
  devices.forEach((d, i) => {
    const y = 24 + i * rowH;
    ctx.fillStyle = "#111"; ctx.font = "11px sans-serif";
    ctx.fillText(d.device, 8, y - 4);
    // full known span in faint grey
    for (const ep of d.epochs) {
      if (!ep.covered_time.length) continue;
      const lo = ep.covered_time[0][0], hi = ep.covered_time[ep.covered_time.length - 1][1];
      ctx.fillStyle = "#e5e7eb";
      ctx.fillRect(X(lo), y, X(hi) - X(lo), rowH * 0.45);
    }
    // covered
    ctx.fillStyle = "#22c55e";
    for (const ep of d.epochs)
      for (const [a, b] of ep.covered_time)
        ctx.fillRect(X(a), y, Math.max(1.5, X(b) - X(a)), rowH * 0.45);
    // gaps
    ctx.fillStyle = "#ef4444";
    for (const ep of d.epochs)
      for (const [a, b] of ep.time_gaps)
        ctx.fillRect(X(a), y, Math.max(1.5, X(b) - X(a)), rowH * 0.45);
    // unconfirmable / degraded
    ctx.fillStyle = "#f59e0b";
    for (const u of d.unconfirmable)
      if (u.time_range) ctx.fillRect(X(u.time_range[0]), y + rowH * 0.5,
        Math.max(2, X(u.time_range[1]) - X(u.time_range[0])), 4);
    // conflicts
    ctx.fillStyle = "#7c3aed";
    for (const c of d.conflicts) {
      const ep = d.epochs.find(e => e.run_id === c.run_id && e.epoch_seq === c.epoch_seq);
      const run = ep && ep.rate_runs.find(r =>
        c.counter_range[0] >= r.counter_range[0] && c.counter_range[0] < r.counter_range[1]);
      if (run && run.offset != null) {
        const t = c.counter_range[0] / run.rate + run.offset;
        ctx.fillRect(X(t) - 1, y, 3, rowH * 0.45);
      }
    }
  });
  ctx.fillStyle = "#6b7280"; ctx.font = "10px sans-serif";
  ctx.fillText(new Date(tMin * 1000).toLocaleTimeString(), 8, cv.height - 6);
  const lbl = new Date(tMax * 1000).toLocaleTimeString();
  ctx.fillText(lbl, cv.width - ctx.measureText(lbl).width - 8, cv.height - 6);
}

/* ---------------- tables ---------------- */

function renderGaps(idx) {
  const tb = $("#gaps tbody");
  const rows = [];
  if (idx && idx.report)
    for (const d of Object.values(idx.report.devices))
      for (const ep of d.epochs)
        for (const g of ep.time_gaps)
          rows.push(`<tr><td>${d.device}</td><td>${ep.run_id.slice(0, 6)}#${ep.epoch_seq}</td>
            <td class="mono">${fmt(g[0], 1)} … ${fmt(g[1], 1)}</td><td>${fmt(g[1] - g[0])}s</td></tr>`);
  tb.innerHTML = rows.join("") || `<tr><td colspan="4">无</td></tr>`;
}

function renderUnconfirmable(idx) {
  const tb = $("#unconfirm tbody");
  const rows = (idx && idx.report ? idx.report.unconfirmable : []).map(u =>
    `<tr><td>${u.device}</td><td>${u.reason}</td><td>${u.detail}</td></tr>`);
  tb.innerHTML = rows.join("") || `<tr><td colspan="3">无</td></tr>`;
}

function renderAlign(idx) {
  const el = $("#align");
  if (!idx || !idx.report) { el.innerHTML = ""; return; }
  const parts = [];
  for (const d of Object.values(idx.report.devices)) {
    for (const ep of d.epochs) {
      const runs = ep.rate_runs.map(r => `<tr>
        <td>${r.rate} Hz</td><td class="mono">[${r.counter_range[0]}, ${r.counter_range[1]})</td>
        <td>${r.n_anchors}</td>
        <td class="mono">${r.offset == null ? "—" : fmt(r.offset, 3)}</td>
        <td>${r.drift_ppm == null ? "—" : fmt(r.drift_ppm, 0) + " ppm"}</td>
        <td>${r.resid_rms == null ? "—" : fmt(r.resid_rms * 1000, 0) + " ms"}</td>
        <td>${r.resid_max == null ? "—" : fmt(r.resid_max * 1000, 0) + " ms"}</td>
        <td>${r.confidence}</td></tr>`).join("");
      parts.push(`<div class="align-epoch">
        <h3>${d.device} · run ${ep.run_id.slice(0, 6)} #${ep.epoch_seq} ·
          counter [${ep.counter_range[0]}, ${ep.counter_range[1]}) ·
          覆盖 ${fmt(ep.covered_time.reduce((s, [a, b]) => s + b - a, 0))}s</h3>
        <table><thead><tr><th>速率段</th><th>counter区间</th><th>锚点</th>
          <th>offset(s)</th><th>漂移</th><th>残差rms</th><th>残差max</th><th>置信</th>
        </tr></thead><tbody>${runs}</tbody></table></div>`);
    }
  }
  el.innerHTML = parts.join("") || "<p>暂无</p>";
}

function renderIndexMeta(state, idx) {
  const ix = state.indexer || {};
  $("#index-meta").innerHTML = idx && idx.report
    ? `校验版本 <b>${idx.report.validation_version}</b> ·
       已发布索引 <b>#${idx.index_id}</b> ·
       统计: 设备 ${idx.report.stats.devices}，片段 ${idx.report.stats.segments_used}，
       覆盖 ${fmt(idx.report.stats.coverage_seconds)}s，缺口 ${fmt(idx.report.stats.gap_seconds)}s，
       冲突 ${idx.report.stats.conflicts}，无法确认 ${idx.report.stats.unconfirmable} ·
       构建 ${ix.builds} 次（丢弃 ${ix.discarded_builds}）`
    : `索引尚未发布（校验版本 ${state.validation_version}）`;
}

function renderProvenance(idx) {
  const tb = $("#provenance tbody");
  const rows = (idx && idx.report ? idx.report.provenance : []).map(p =>
    `<tr><td>${p.segment_id}</td><td>${p.device}</td>
     <td class="mono">[${p.counter_range[0]}, ${p.counter_range[1]})</td>
     <td class="mono">${fmt(p.time_range[0], 1)} … ${fmt(p.time_range[1], 1)}</td>
     <td>${fmt(p.err * 1000, 0)}ms</td><td>${p.anchor ? "锚点" : "补传"}</td></tr>`);
  tb.innerHTML = rows.join("") || `<tr><td colspan="6">无</td></tr>`;
}

function renderSegments(segs) {
  const tb = $("#segments tbody");
  tb.innerHTML = segs.slice(-60).reverse().map(s =>
    `<tr><td>${s.segment_id}</td><td>${s.device_id}</td>
     <td class="mono">${(s.run_id || "").slice(0, 6)}</td>
     <td class="mono">${s.counter_start}+${s.nsamples}</td>
     <td>${s.sample_rate}</td><td>${s.backfill ? "是" : ""}</td>
     <td><span class="badge ${s.state}">${s.state}</span></td>
     <td>${s.state_reason || ""}</td>
     <td>${s.state === "validated"
        ? `<button data-retract="${s.segment_id}">撤回</button>` : ""}
        ${s.state === "retracted"
        ? `<button data-revalidate="${s.segment_id}">重新校验</button>` : ""}</td>
    </tr>`).join("");
  tb.querySelectorAll("[data-retract]").forEach(b => b.onclick = async () => {
    await api(`/api/segments/${b.dataset.retract}/retract`, "POST",
      { reason: "manual retract from UI" });
    refresh();
  });
  tb.querySelectorAll("[data-revalidate]").forEach(b => b.onclick = async () => {
    await api(`/api/segments/${b.dataset.revalidate}/revalidate`, "POST", {});
    refresh();
  });
}

/* ---------------- main loop ---------------- */

async function refresh() {
  try {
    const [state, idxRes, segs] = await Promise.all([
      api("/api/state"), api("/api/index").catch(() => null), api("/api/segments"),
    ]);
    $("#hdr-status").textContent =
      `校验版本 ${state.validation_version} · 已发布索引 #${state.published_index_id ?? "—"}`;
    renderDevices(state);
    renderIndexMeta(state, idxRes && idxRes.report ? idxRes : null);
    if (idxRes && idxRes.report) {
      renderTimeline(idxRes); renderGaps(idxRes);
      renderUnconfirmable(idxRes); renderAlign(idxRes); renderProvenance(idxRes);
    }
    renderSegments(segs.segments || []);
  } catch (e) { /* server restarting; next poll recovers */ }
}

$("#btn-create").onclick = async () => {
  const id = prompt("设备ID：", "dev-" + Math.floor(Math.random() * 1000));
  if (id) { await api("/api/device/create", "POST", { device_id: id }); refresh(); }
};
$("#btn-rebuild").onclick = async () => { await api("/api/index/rebuild", "POST", {}); };

refresh();
setInterval(refresh, 1500);
