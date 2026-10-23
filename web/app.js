// LazyDecider demo UI — renders from window.LAZY (baked by src/build_demo_data.py).
(function () {
  "use strict";
  const D = window.LAZY;
  const fmt = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

  /* ---- Headline stats ---- */
  (function stats() {
    const winner = D.results.find((r) => r.winner);
    const tinker = D.results.find((r) => r.model.startsWith("Tinker"));
    const nRuns = D.armA.length + D.armB.length;
    const cards = [
      { num: fmt(winner.acc, 3), lbl: "winner accuracy", win: true },
      { num: fmt(winner.kl, 3), lbl: "KL from gold ↓", win: true },
      { num: nRuns, lbl: "Gemma-proposed configs" },
      { num: "$0", lbl: "cost to search (local)" },
    ];
    document.getElementById("stats").innerHTML = cards
      .map((c) => `<div class="stat ${c.win ? "win" : ""}"><div class="num">${c.num}</div><div class="lbl">${c.lbl}</div></div>`)
      .join("");
  })();

  /* ---- Overnight replay ---- */
  const runsEl = document.getElementById("runs");
  const armSelect = document.getElementById("armSelect");
  const replayBtn = document.getElementById("replayBtn");
  let timers = [];

  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function runCard(r) {
    const chips = Object.entries(r.config)
      .map(([k, v]) => `<span class="chip">${k}=${v}</span>`)
      .join("");
    const decision = r.kept
      ? `<span class="decision keep">KEEP ✔</span>`
      : `<span class="decision revert">REVERT ✗</span>`;
    const loss = r.selectLoss != null ? `<div class="loss">select loss <b>${fmt(r.selectLoss)}</b>${
      r.selectAcc != null ? ` · acc <b>${fmt(r.selectAcc, 3)}</b>` : ""
    }</div>` : "";
    return `<div class="run ${r.kept ? "keep-run" : "revert-run"}">
      <div class="run-head"><span class="run-title">run ${r.run}</span>${decision}</div>
      <div class="reason">“${r.reason}”</div>
      <div class="cfg">${chips}</div>
      ${loss}
    </div>`;
  }

  function play(arm) {
    clearTimers();
    runsEl.innerHTML = "";
    const rows = D[arm];
    rows.forEach((r, i) => {
      timers.push(
        setTimeout(() => {
          const div = document.createElement("div");
          div.innerHTML = runCard(r);
          const node = div.firstElementChild;
          node.style.animationDelay = "0s";
          runsEl.appendChild(node);
          node.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, i * 1100)
      );
    });
  }

  armSelect.addEventListener("change", () => play(armSelect.value));
  replayBtn.addEventListener("click", () => play(armSelect.value));

  /* ---- Results table ---- */
  (function table() {
    const tbody = document.querySelector("#resultsTable tbody");
    tbody.innerHTML = D.results
      .map((r) => {
        const cls = r.winner ? "winner-row" : r.model.startsWith("Tinker") ? "tinker-row" : "";
        return `<tr class="${cls}">
          <td>${r.model}</td>
          <td class="role">${r.role}</td>
          <td>${fmt(r.acc, 3)}</td>
          <td>${fmt(r.kl)}</td>
          <td>${fmt(r.brier)}</td>
          <td>${fmt(r.ece)}</td>
        </tr>`;
      })
      .join("");
  })();

  /* ---- Links ---- */
  (function links() {
    const L = D.links;
    document.querySelector(".link-row").innerHTML = `
      <a href="${L.repo}" target="_blank" rel="noopener">⭐ GitHub repo</a>
      <a href="${L.model}" target="_blank" rel="noopener">🤗 Hugging Face model</a>
      <a href="${L.dataset}" target="_blank" rel="noopener">📊 typed-decisions dataset</a>`;
  })();

  /* ---- Audio ---- */
  (function audio() {
    const btn = document.getElementById("playBtn");
    const a = document.getElementById("narration");
    btn.addEventListener("click", () => {
      if (a.paused) {
        a.play();
        btn.textContent = "⏸ Pause narration";
        btn.classList.add("playing");
      } else {
        a.pause();
        btn.textContent = "▶ Play narration";
        btn.classList.remove("playing");
      }
    });
    a.addEventListener("ended", () => {
      btn.textContent = "▶ Play narration";
      btn.classList.remove("playing");
    });
  })();

  /* ---- Generalization: invoice case stepper ---- */
  (function cases() {
    const S = window.LAZY_SAMPLES || [];
    if (!S.length) return;
    const card = document.getElementById("caseCard");
    const counter = document.getElementById("caseCounter");
    let idx = 0;

    const pretty = (v) =>
      String(v).replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());

    function render() {
      const c = S[idx];
      const facts = Object.entries(c.summary)
        .map(([k, v]) => `<div class="fact"><span class="fk">${k}</span><span class="fv">${v}</span></div>`)
        .join("");
      const decisions = c.decisions
        .map((d) => `
          <tr class="${d.match ? "d-ok" : "d-miss"}">
            <td class="d-label">${d.label}</td>
            <td class="d-pred">${pretty(d.pred)} <span class="d-conf">${Math.round(d.conf * 100)}%</span></td>
            <td class="d-gold">${pretty(d.gold)}</td>
            <td class="d-mark">${d.match ? "✔" : "✗"}</td>
          </tr>`)
        .join("");
      card.innerHTML = `
        <div class="case-grid">
          <div class="case-scenario">
            <div class="case-tag">invoice · ${c.id.split("_").pop()}</div>
            <h3>The scenario</h3>
            <div class="facts">${facts}</div>
          </div>
          <div class="case-decisions">
            <h3>Model decision <span class="vs">vs. gold</span>
              <span class="score-pill ${c.hits === c.total ? "perfect" : ""}">${c.hits}/${c.total} match</span>
            </h3>
            <table class="dtable">
              <thead><tr><th>Decision</th><th>Model (confidence)</th><th>Gold</th><th></th></tr></thead>
              <tbody>${decisions}</tbody>
            </table>
          </div>
        </div>`;
      counter.textContent = `${idx + 1} / ${S.length}`;
    }

    document.getElementById("prevCase").addEventListener("click", () => {
      idx = (idx - 1 + S.length) % S.length;
      render();
    });
    document.getElementById("nextCase").addEventListener("click", () => {
      idx = (idx + 1) % S.length;
      render();
    });
    render();
  })();

  /* ---- Kick off ---- */
  play("armA");
})();
