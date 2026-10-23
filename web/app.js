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

  /* ---- Kick off ---- */
  play("armA");
})();
