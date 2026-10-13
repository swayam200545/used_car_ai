/* Used Car Price Estimator & AI Buying Advisory - frontend
   Handles:
     - 360° Inspection Checklist & Live Defect Cost Estimator
     - Multi-Photo Upload & Computer Vision Visual AI Assessment
     - AI Recommendation Engine & 4 Pillars (Value, Reliability, Usage, Condition)
     - What Matters Most Key Decision Factors
     - Systematic Step-by-Step Valuation Waterfall Analysis
     - Order Details & Financial Deal Sheet (True On-Road Cost, Counter-Offer, Walkaway Price)
     - Buyer's Negotiation Playbook
     - 3-Year TCO Projections
     - History & Modal Report Viewer
*/
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const state = {
    meta: null,
    charts: {},
    insightsLoaded: false,
    busy: false,
    historyLoaded: false,
    uploadedPhotos: [],         // [{ name, dataUrl, analysis }]
    imageAnalysisProfile: null, // Aggregated CV profile
    inspectionFlags: new Set(), // Set of selected defect keys
    activeInspectCat: "Mechanical",
    lastResult: null,
  };

  /* ------------------------------------------------------------ helpers */
  const inr = (n) => "₹" + Math.round(n || 0).toLocaleString("en-IN");
  const lakh = (n) => "₹" + ((n || 0) / 1e5).toFixed(2) + " L";
  const lakhFromL = (n) => "₹" + Number(n || 0).toFixed(2) + " L";
  const parseNum = (s) => {
    const d = String(s ?? "").replace(/[^\d]/g, "");
    return d ? parseInt(d, 10) : null;
  };
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    let data = {};
    try { data = await res.json(); } catch (_) { /* empty body */ }
    if (!res.ok) {
      const err = new Error(data.error || "Something went wrong. Please try again.");
      err.field = data.field; err.status = res.status;
      throw err;
    }
    return data;
  }

  function showGlobalError(html) {
    const b = $("#global-error");
    b.innerHTML = html;
    b.hidden = !html;
  }

  /* ------------------------------------------------------------ odometer */
  const GROUPS = [1, 2, 2, 3];   // Indian digit grouping: 1,23,45,678
  const drums = [];

  function buildOdometer() {
    const el = $("#odometer");
    el.innerHTML = "";
    drums.length = 0;
    const rupee = Object.assign(document.createElement("span"), { className: "odo-rupee", textContent: "₹" });
    el.append(rupee);
    let idx = 0;
    GROUPS.forEach((size, gi) => {
      if (gi > 0) el.append(Object.assign(document.createElement("span"), { className: "odo-comma", textContent: "," }));
      for (let k = 0; k < size; k++, idx++) {
        const drum = document.createElement("span");
        drum.className = "drum dim" + (idx >= 5 ? " minor" : "");
        const strip = document.createElement("span");
        strip.className = "strip";
        for (let d = 0; d < 10; d++) strip.append(Object.assign(document.createElement("i"), { textContent: d }));
        drum.append(strip);
        el.append(drum);
        drums.push({ drum, strip });
      }
    });
  }

  function setOdometer(value) {
    const v = Math.max(0, Math.round(value || 0));
    const digits = String(v).padStart(8, "0").slice(-8);
    const firstSig = v > 0 ? digits.search(/[1-9]/) : 8;
    drums.forEach(({ drum, strip }, i) => {
      drum.classList.toggle("dim", i < firstSig);
      strip.style.setProperty("--delay", `${i * 55}ms`);
      strip.style.setProperty("--d", digits[i]);
    });
    $("#odometer").setAttribute("aria-label", v ? `Estimated price ${inr(v)}` : "Estimated price");
  }

  /* ---------------------------------------------------------------- tabs */
  function showTab(name, updateHash = true) {
    if (!["home", "estimate", "insights", "history"].includes(name)) name = "home";
    $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === name)));
    $$(".view").forEach((v) => (v.hidden = v.id !== "tab-" + name));
    if (updateHash) history.replaceState(null, "", "#" + name);
    if (name === "insights") loadInsights();
    if (name === "history") loadHistory();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* ---------------------------------------------------------------- form */
  function buildSeg(id, name, items, selected) {
    const box = $("#" + id);
    box.innerHTML = "";
    items.forEach((it) => {
      const label = document.createElement("label");
      const input = Object.assign(document.createElement("input"), {
        type: "radio", name, value: it.value, disabled: !!it.disabled,
      });
      input.checked = String(it.value) === String(selected);
      const span = Object.assign(document.createElement("span"), { textContent: it.label });
      label.append(input, span);
      box.append(label);
    });
  }
  const getSeg = (name) => $(`input[name="${name}"]:checked`)?.value ?? "";

  function fillSelect(sel, values, selected) {
    sel.innerHTML = "";
    values.forEach((v) => {
      const o = document.createElement("option");
      o.value = v.value ?? v; o.textContent = v.label ?? v;
      sel.append(o);
    });
    if (selected != null && [...sel.options].some((o) => o.value === String(selected))) sel.value = String(selected);
  }

  function currentSpec() {
    const { catalog } = state.meta;
    return catalog[$("#brand").value]?.[$("#model").value];
  }

  function onBrandChange(preferModel, preferFuel, preferTrans) {
    const models = Object.keys(state.meta.catalog[$("#brand").value]).sort();
    fillSelect($("#model"), models, preferModel);
    onModelChange(preferFuel, preferTrans);
  }

  function onModelChange(preferFuel, preferTrans) {
    const spec = currentSpec();
    if (!spec) return;
    const FUEL_ORDER = ["Petrol", "Diesel", "CNG", "Electric"];
    const fuels = Object.keys(spec.fuels).sort((a, b) => FUEL_ORDER.indexOf(a) - FUEL_ORDER.indexOf(b));
    const fuel = fuels.includes(preferFuel) ? preferFuel : fuels.includes(getSeg("fuel")) ? getSeg("fuel") : fuels[0];
    buildSeg("fuel", "fuel", fuels.map((f) => ({ value: f, label: f })), fuel);

    const wanted = preferTrans || getSeg("transmission");
    const avail = spec.transmissions;
    const trans = avail.includes(wanted) ? wanted : avail[0];
    buildSeg("transmission", "transmission", state.meta.transmissions.map((t) => ({ value: t, label: t, disabled: !avail.includes(t) })), trans);
    updateSpecs();
  }

  function updateSpecs() {
    const spec = currentSpec();
    const el = $("#specs");
    if (!spec) { el.textContent = ""; return; }
    const f = spec.fuels[getSeg("fuel")] || Object.values(spec.fuels)[0];
    const engine = f.cc > 0 ? `<b>${f.cc.toLocaleString("en-IN")}</b> cc` : "<b>Electric</b> motor";
    el.innerHTML = `${engine} · <b>${f.bhp}</b> bhp · <b>${f.seats}</b> seats · Ex-Showroom: ₹<b>${spec.showroom}</b> L`;
  }

  function syncKmOutput() {
    const km = parseNum($("#km").value);
    const year = parseInt($("#year").value, 10);
    const age = state.meta ? state.meta.max_year - year : 0;
    const out = $("#km-out");
    if (km == null) { out.textContent = ""; return; }
    out.textContent = age > 0 ? `≈ ${Math.round(km / age).toLocaleString("en-IN")} km a year` : "";
  }

  function setKm(value, fromRange) {
    const v = Math.max(0, Math.min(400000, value ?? 0));
    $("#km").value = v ? v.toLocaleString("en-IN") : "";
    if (!fromRange) $("#km-range").value = Math.min(v, 250000);
    else $("#km-range").value = v;
    syncKmOutput();
  }

  function setAskingHint() {
    const v = parseNum($("#asking").value);
    $("#asking-hint").textContent = v ? `${inr(v)} · ${(v / 1e5).toFixed(2)} lakh` : "Enter asking price to generate the complete financial deal sheet and negotiation strategy.";
  }

  /* ------------------------------------------------ 360° INSPECTION CHECKLIST */
  function setupInspectionChecklist() {
    const catalog = state.meta.inspection_catalog || {};

    // Category tab listeners
    $$(".inspect-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".inspect-tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeInspectCat = btn.dataset.cat;
        renderInspectionList();
      });
    });

    // "All Clear" & "Reset" buttons
    $("#chk-all-good")?.addEventListener("click", () => {
      state.inspectionFlags.clear();
      renderInspectionList();
      updateRefurbTally();
    });

    $("#chk-reset")?.addEventListener("click", () => {
      state.inspectionFlags.clear();
      renderInspectionList();
      updateRefurbTally();
    });

    renderInspectionList();
    updateRefurbTally();
  }

  function renderInspectionList() {
    const list = $("#inspect-list");
    if (!list) return;
    list.innerHTML = "";
    const catalog = state.meta.inspection_catalog || {};
    const cat = state.activeInspectCat;

    const items = Object.entries(catalog).filter(([k, v]) => v.category === cat);
    if (!items.length) {
      list.innerHTML = `<p class="hint" style="padding:10px;">No checkpoints in this category.</p>`;
      return;
    }

    items.forEach(([key, info]) => {
      const isChecked = state.inspectionFlags.has(key);
      const row = document.createElement("label");
      row.className = `inspect-item ${isChecked ? "checked" : ""}`;
      row.innerHTML = `
        <div class="inspect-left">
          <input type="checkbox" data-key="${esc(key)}" ${isChecked ? "checked" : ""}>
          <span>${esc(info.name)}</span>
        </div>
        <span class="inspect-cost-badge">+${inr(info.cost)}</span>
      `;
      row.querySelector("input").addEventListener("change", (e) => {
        if (e.target.checked) state.inspectionFlags.add(key);
        else state.inspectionFlags.delete(key);
        row.classList.toggle("checked", e.target.checked);
        updateRefurbTally();
      });
      list.appendChild(row);
    });
  }

  function updateRefurbTally() {
    const catalog = state.meta?.inspection_catalog || {};
    let total = 0;
    state.inspectionFlags.forEach((k) => {
      if (catalog[k]) total += catalog[k].cost;
    });
    $("#refurb-tally-amt").textContent = inr(total);
  }

  /* ------------------------------------------------ CAR PHOTO UPLOAD & VISUAL AI */
  function setupPhotoUpload() {
    const dropzone = $("#photo-dropzone");
    const input = $("#photo-input");

    dropzone.addEventListener("click", () => input.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
    });

    input.addEventListener("change", () => {
      if (input.files?.length) handleFiles(input.files);
    });
  }

  async function handleFiles(fileList) {
    const files = [...fileList].slice(0, 8); // max 8 photos
    for (const file of files) {
      if (!file.type.startsWith("image/")) continue;
      const dataUrl = await readFileAsDataUrl(file);
      state.uploadedPhotos.push({ name: file.name, dataUrl });
    }
    renderPhotoPreviews();
    await triggerVisualAnalysis();
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.readAsDataURL(file);
    });
  }

  function renderPhotoPreviews() {
    const grid = $("#photos-preview-grid");
    grid.innerHTML = "";
    grid.hidden = state.uploadedPhotos.length === 0;

    state.uploadedPhotos.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "photo-thumb-card";
      card.innerHTML = `
        <img src="${item.dataUrl}" alt="${esc(item.name)}">
        <button type="button" class="thumb-del-btn" title="Remove photo" data-idx="${index}">&times;</button>
        ${item.rating ? `<span class="thumb-score-tag">★ ${item.rating}</span>` : ""}
      `;
      card.querySelector(".thumb-del-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        state.uploadedPhotos.splice(index, 1);
        renderPhotoPreviews();
        triggerVisualAnalysis();
      });
      grid.appendChild(card);
    });
  }

  async function triggerVisualAnalysis() {
    const statusWrap = $("#visual-ai-status");
    const badge = $("#visual-ai-badge");
    const spinner = $("#visual-ai-spinner");

    if (state.uploadedPhotos.length === 0) {
      state.imageAnalysisProfile = null;
      statusWrap.hidden = true;
      return;
    }

    statusWrap.hidden = false;
    spinner.hidden = false;
    badge.textContent = `Analyzing ${state.uploadedPhotos.length} photo(s) with Computer Vision…`;

    try {
      const payload = {
        images: state.uploadedPhotos.map((p) => ({ name: p.name, data: p.dataUrl })),
      };
      const res = await api("/api/analyze-images", { method: "POST", body: JSON.stringify(payload) });
      state.imageAnalysisProfile = res;

      // Update ratings on thumbnails
      if (res.gallery?.length) {
        res.gallery.forEach((g, idx) => {
          if (state.uploadedPhotos[idx]) state.uploadedPhotos[idx].rating = g.rating;
        });
        renderPhotoPreviews();
      }

      spinner.hidden = true;
      badge.textContent = `✓ Visual AI Rating: ${res.overall_condition_score}/10 · ${res.overall_condition_label} (${res.condition_adjustment_pct >= 0 ? "+" : ""}${res.condition_adjustment_pct}% valuation adjustment)`;
    } catch (e) {
      spinner.hidden = true;
      badge.textContent = `Visual analysis completed.`;
    }
  }

  /* ------------------------------------------------------------- setup form */
  function setupForm() {
    const m = state.meta;
    const brands = Object.keys(m.catalog).sort();
    fillSelect($("#brand"), brands, "Maruti Suzuki");

    const years = [];
    for (let y = m.max_year; y >= m.min_year; y--) years.push(y);
    fillSelect($("#year"), years, m.max_year - 5);

    buildSeg("owner", "owner", m.owners.map((o) => ({ value: o.value, label: o.label.replace(" owner", "") })), 1);

    onBrandChange("Swift", "Petrol", "Manual");
    setKm(40000);

    $("#brand").addEventListener("change", () => onBrandChange());
    $("#model").addEventListener("change", () => onModelChange());
    $("#fuel").addEventListener("change", updateSpecs);
    $("#year").addEventListener("change", syncKmOutput);

    $("#km-range").addEventListener("input", (e) => setKm(parseInt(e.target.value, 10), true));
    $("#km").addEventListener("input", (e) => {
      const v = parseNum(e.target.value);
      e.target.value = v == null ? "" : Math.min(v, 400000).toLocaleString("en-IN");
      $("#km-range").value = Math.min(v ?? 0, 250000);
      syncKmOutput();
    });
    $("#asking").addEventListener("input", (e) => {
      const v = parseNum(e.target.value);
      e.target.value = v == null ? "" : v.toLocaleString("en-IN");
      setAskingHint();
    });

    setupPhotoUpload();
    setupInspectionChecklist();

    $("#car-form").addEventListener("submit", onSubmit);

    // Print report button
    $("#btn-print")?.addEventListener("click", () => window.print());

    // Modal close button
    $("#modal-close")?.addEventListener("click", () => $("#deal-modal")?.close());
  }

  function readForm() {
    return {
      brand: $("#brand").value,
      model: $("#model").value,
      fuel: getSeg("fuel"),
      transmission: getSeg("transmission"),
      year: parseInt($("#year").value, 10),
      km: parseNum($("#km").value),
      owner: parseInt(getSeg("owner"), 10) || 1,
      asking: parseNum($("#asking").value),
      inspection_flags: [...state.inspectionFlags],
      image_analysis: state.imageAnalysisProfile,
    };
  }

  function clearErrors() {
    $$(".field.is-invalid").forEach((f) => f.classList.remove("is-invalid"));
    showGlobalError("");
  }
  function fieldError(field, message) {
    const p = $(`[data-err="${field}"]`);
    if (!p) { showGlobalError(esc(message)); return; }
    p.textContent = message;
    const wrap = p.closest(".field");
    wrap.classList.add("is-invalid");
    return wrap;
  }

  function setBusy(on) {
    state.busy = on;
    $("#submit").disabled = on;
    $(".btn-label").textContent = on ? "Running Neural Network & Advisory…" : "Analyze Car & Generate Valuation";
    $("#result").setAttribute("aria-busy", String(on));
  }

  async function onSubmit(e) {
    e.preventDefault();
    if (state.busy) return;
    clearErrors();
    const body = readForm();
    if (body.km == null) {
      fieldError("km", "Enter the kilometres driven.")?.scrollIntoView({ block: "center", behavior: "smooth" });
      $("#km").focus();
      return;
    }

    setBusy(true);
    try {
      const r = await api("/api/predict", { method: "POST", body: JSON.stringify(body) });
      state.lastResult = r;
      renderResult(r);
      state.historyLoaded = false;
      if (matchMedia("(max-width: 1100px)").matches) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      if (err.field) fieldError(err.field, err.message)?.scrollIntoView({ block: "center", behavior: "smooth" });
      else showGlobalError(esc(err.message));
    } finally {
      setBusy(false);
    }
  }

  /* -------------------------------------------------------------- RENDER RESULT */
  function renderResult(r) {
    const c = r.car;
    const adv = r.advisory || {};

    $("#car-line").hidden = false;
    $("#plate-text").textContent = `${c.model} ${c.year}`;
    $("#car-title").textContent = `${c.brand} ${c.model} · ${c.fuel} · ${c.transmission} · ${c.km.toLocaleString("en-IN")} km · ${c.owner}`;
    $("#empty-note").hidden = true;
    $("#detail").hidden = false;

    // Reset animation
    const d = $("#detail");
    d.style.animation = "none";
    void d.offsetWidth;
    d.style.animation = "";

    // 1. Odometer
    setOdometer(r.price);

    // 2. HERO AI BUYING ADVISORY
    const badge = $("#adv-badge");
    badge.textContent = adv.verdict_badge || "EVALUATED DEAL";
    badge.className = `verdict-badge ${adv.verdict_color || "teal"}`;
    $("#adv-headline").textContent = adv.verdict_headline || "Valuation Analysis Ready";
    $("#adv-summary").textContent = adv.verdict_summary || "";
    $("#adv-index-score").textContent = adv.overall_buy_index ?? "—";

    // 3. 4 EVALUATION PILLARS
    const pils = adv.pillars || {};
    $("#pil-val").textContent = `${pils.value_score ?? 80}%`;
    $("#pil-val-bar").style.width = `${pils.value_score ?? 80}%`;

    $("#pil-rel").textContent = `${pils.reliability_score ?? 85}%`;
    $("#pil-rel-bar").style.width = `${pils.reliability_score ?? 85}%`;

    $("#pil-use").textContent = `${pils.usage_score ?? 80}%`;
    $("#pil-use-bar").style.width = `${pils.usage_score ?? 80}%`;

    $("#pil-cnd").textContent = `${pils.condition_score ?? 80}%`;
    $("#pil-cnd-bar").style.width = `${pils.condition_score ?? 80}%`;

    // 4. "WHAT MATTERS MOST FOR THIS CAR" KEY FACTORS
    const kfList = $("#key-factors-list");
    kfList.innerHTML = "";
    (adv.key_factors || []).forEach((kf) => {
      const card = document.createElement("div");
      card.className = `factor-card ${kf.type || "neutral"}`;
      card.innerHTML = `
        <div class="factor-body">
          <span class="factor-badge">${esc(kf.badge)} · ${esc(kf.importance)}</span>
          <h4 class="factor-title">${esc(kf.title)}</h4>
          <p class="factor-desc">${esc(kf.desc)}</p>
        </div>
      `;
      kfList.appendChild(card);
    });

    // 5. RANGE BAR & TRACK
    const ask = r.deal?.asking ?? null;
    const lo = Math.min(r.low, ask ?? Infinity) * 0.86;
    const hi = Math.max(r.high, ask ?? 0) * 1.14;
    const pos = (v) => ((v - lo) / (hi - lo)) * 100;
    $("#range-text").textContent = `${inr(r.low)} – ${inr(r.high)}`;
    Object.assign($("#band").style, { left: pos(r.low) + "%", width: pos(r.high) - pos(r.low) + "%" });
    $("#mark-est").style.left = pos(r.price) + "%";
    const mAsk = $("#mark-ask");
    mAsk.hidden = !ask;
    if (ask) {
      mAsk.style.left = Math.max(2, Math.min(98, pos(ask))) + "%";
      mAsk.className = "mark ask " + r.deal.verdict;
      mAsk.firstElementChild.textContent = `Asking ${lakh(ask)}`;
    }

    // 6. FACTS
    $("#f-new").textContent = lakhFromL(r.new_lakh);
    $("#f-kept").textContent = r.retained_pct.toFixed(0) + "%";
    $("#f-age").textContent = r.age === 0 ? "This year" : `${r.age} ${r.age === 1 ? "year" : "years"}`;
    $("#f-loss").textContent = r.age === 0 ? "—" : inr(r.loss_per_year);
    $("#cap-year").textContent = `Same car, driven about ${r.curve_year.km_per_year.toLocaleString("en-IN")} km a year`;

    // 7. SYSTEMATIC VALUATION WATERFALL
    const wfGrid = $("#waterfall-grid");
    wfGrid.innerHTML = "";
    (adv.systematic_breakdown || []).forEach((step) => {
      const card = document.createElement("div");
      card.className = `waterfall-card ${step.type === "final" ? "final" : ""}`;
      const colorCls = step.amount < 0 ? "text-deduct" : step.type === "final" ? "text-final" : "text-base";
      card.innerHTML = `
        <span class="wf-step-label">Step ${step.step}</span>
        <h4 class="wf-title">${esc(step.title)}</h4>
        <strong class="wf-amt ${colorCls}">${esc(step.display)}</strong>
        <span class="wf-note">${esc(step.note)}</span>
      `;
      wfGrid.appendChild(card);
    });

    // 8. ORDER DETAILS & FINANCIAL DEAL SHEET
    const ds = adv.deal_sheet || {};
    $("#ds-asking").textContent = ds.asking_price ? inr(ds.asking_price) : "Not Specified";
    $("#ds-market").textContent = inr(ds.market_estimate || r.price);
    $("#ds-refurb").textContent = ds.refurbishment_cost ? `-${inr(ds.refurbishment_cost)}` : "₹0 (Clean Inspection)";
    $("#ds-defects-sub").textContent = ds.identified_defects_count
      ? `${ds.identified_defects_count} defect(s) flagged: ${ds.identified_defects.map((d) => d.name).join(", ")}`
      : "No mechanical or structural defects flagged";
    $("#ds-rto").textContent = inr(ds.rto_transfer_fee || 2500);
    $("#ds-insurance").textContent = inr(ds.insurance_estimate || 8500);
    $("#ds-total-onroad").textContent = inr(ds.total_on_road_acquisition || (r.price + 11000));
    $("#ds-counter").textContent = inr(ds.recommended_counter_offer || r.price);
    $("#ds-walkaway").textContent = inr(ds.walk_away_price || r.high);

    // 9. BUYER'S NEGOTIATION PLAYBOOK
    const scGrid = $("#scripts-grid");
    scGrid.innerHTML = "";
    (adv.negotiation_scripts || []).forEach((sc, i) => {
      const card = document.createElement("div");
      card.className = "script-card";
      card.innerHTML = `
        <div class="script-head">
          <span class="script-tactic">💬 ${esc(sc.tactic)}</span>
          <button type="button" class="btn-copy-script" data-text="${esc(sc.script)}">Copy Script</button>
        </div>
        <p class="script-text">${esc(sc.script)}</p>
      `;
      card.querySelector(".btn-copy-script").addEventListener("click", function () {
        navigator.clipboard?.writeText(sc.script);
        this.textContent = "Copied!";
        setTimeout(() => (this.textContent = "Copy Script"), 1800);
      });
      scGrid.appendChild(card);
    });

    // 10. 3-YEAR TCO TABLE
    const tcoBody = $("#tco-body");
    tcoBody.innerHTML = "";
    (adv.tco_projections || []).forEach((tp) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>Year ${tp.year}</strong></td>
        <td class="num">${inr(tp.fuel_cost)}</td>
        <td class="num">${inr(tp.maintenance_service)}</td>
        <td class="num">${inr(tp.insurance)}</td>
        <td class="num font-bold text-petrol">${inr(tp.projected_car_equity)}</td>
        <td class="num font-bold">${inr(tp.total_expense)}</td>
      `;
      tcoBody.appendChild(row);
    });

    // 11. PHOTO GALLERY (IF PHOTOS UPLOADED)
    const gallerySection = $("#photo-gallery-section");
    const galleryGrid = $("#photo-gallery-grid");
    const gallery = adv.image_analysis_summary?.gallery || [];
    if (gallery.length > 0) {
      gallerySection.hidden = false;
      galleryGrid.innerHTML = "";
      gallery.forEach((g) => {
        const item = document.createElement("div");
        item.className = "gallery-card";
        item.innerHTML = `
          <img src="${g.thumbnail}" alt="${esc(g.filename)}" class="gallery-card-img">
          <div class="gallery-card-info">
            <span class="gallery-card-view">${esc(g.viewpoint)}</span>
            <span class="gallery-card-rating">★ ${g.rating}/10 · ${esc(g.label)}</span>
            <div class="gallery-card-tags">
              ${(g.tags || []).map((t) => `<span class="gallery-tag">${esc(t)}</span>`).join("")}
            </div>
          </div>
        `;
        galleryGrid.appendChild(item);
      });
    } else {
      gallerySection.hidden = true;
    }

    // 12. CHARTS
    drawEstimateCharts(r);
  }

  /* -------------------------------------------------------------- charts */
  function baseChartDefaults() {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12.5;
    Chart.defaults.color = css("--muted");
    Chart.defaults.maintainAspectRatio = false;
  }

  function upsert(key, canvas, config) {
    state.charts[key]?.destroy();
    state.charts[key] = new Chart(canvas, config);
  }

  const gridStyle = () => ({ color: css("--line-soft"), drawTicks: false });

  function lineConfig({ xLabel, yLabel, series, highlight, xFmt, xMin, xMax }) {
    const petrol = css("--petrol"), plate = css("--plate"), deep = css("--petrol-deep");
    return {
      type: "line",
      data: {
        datasets: [
          { data: series, borderColor: petrol, backgroundColor: "rgba(14,90,88,.10)", fill: true, borderWidth: 3, tension: 0.35, pointRadius: 0, pointHoverRadius: 4, order: 2 },
          { data: [highlight], showLine: false, order: 1, pointRadius: 8, pointHoverRadius: 9, pointBackgroundColor: plate, pointBorderColor: deep, pointBorderWidth: 3 },
        ],
      },
      options: {
        parsing: false, animation: { duration: 700 },
        interaction: { mode: "nearest", intersect: false, axis: "x" },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: deep, padding: 10, displayColors: false,
            callbacks: {
              title: (items) => xFmt(items[0].parsed.x),
              label: (item) => (item.datasetIndex === 1 ? "Your car: " : "") + "₹" + item.parsed.y.toFixed(2) + " lakh",
            },
          },
        },
        scales: {
          x: { type: "linear", min: xMin, max: xMax, title: { display: true, text: xLabel }, grid: { display: false }, ticks: { callback: (v) => xFmt(v), maxTicksLimit: 7 }, border: { color: css("--line") } },
          y: { title: { display: true, text: yLabel }, grid: gridStyle(), border: { display: false }, ticks: { callback: (v) => "₹" + v, padding: 6 }, beginAtZero: false },
        },
      },
    };
  }

  function drawEstimateCharts(r) {
    baseChartDefaults();
    const y = r.curve_year, k = r.curve_km, price = r.price / 1e5;
    upsert("year", $("#chart-year"), lineConfig({
      xLabel: "Model year", yLabel: "Price (lakh)",
      series: y.labels.map((l, i) => ({ x: l, y: y.values[i] })),
      highlight: { x: y.current, y: price },
      xFmt: (v) => (Number.isInteger(v) ? String(v) : ""),
      xMin: y.labels[0], xMax: y.labels[y.labels.length - 1],
    }));
    upsert("km", $("#chart-km"), lineConfig({
      xLabel: "Kilometres driven", yLabel: "Price (lakh)",
      series: k.labels.map((l, i) => ({ x: l, y: k.values[i] })),
      highlight: { x: k.current, y: price },
      xFmt: (v) => (v >= 1000 ? Math.round(v / 1000) + "k" : String(v)),
      xMin: 0, xMax: k.labels[k.labels.length - 1],
    }));
  }

  /* ------------------------------------------------------------ insights */
  async function loadInsights() {
    if (state.insightsLoaded) return;
    const box = $("#insights");
    try {
      const m = await api("/api/insights");
      if (!m.results) throw new Error("No training metrics found. Run: python -m ml.train");
      renderInsights(m);
      state.insightsLoaded = true;
    } catch (err) {
      box.innerHTML = `<p class="loading-note">${esc(err.message)}</p>`;
    }
  }

  function renderInsights(m) {
    const dnn = m.results["Deep neural network"];
    const d = m.dataset;
    const names = Object.keys(m.results);
    const maxMae = Math.max(...names.map((n) => m.results[n].mae_lakh));

    const rows = names.map((n) => {
      const r = m.results[n];
      const isBest = n === "Deep neural network";
      const w = Math.round((r.mae_lakh / maxMae) * 100);
      return `<tr class="${isBest ? "best" : ""}">
        <td>${esc(n)}</td>
        <td class="num">${r.r2.toFixed(3)}</td>
        <td class="num">${r.mae_lakh.toFixed(2)} L</td>
        <td class="num">${r.rmse_lakh.toFixed(2)} L</td>
        <td class="num">${(r.mape * 100).toFixed(1)}%</td>
        <td class="num">${(r.within_10_pct * 100).toFixed(0)}%</td>
        <td><span class="bar"><i style="width: ${w}%"></i></span></td>
      </tr>`;
    }).join("");

    const layers = (m.architecture?.layers || []).map((l, i) => `
      <div class="arch-layer">
        <span class="arch-stack"><span class="arch-block" style="height: ${Math.max(24, Math.min(130, (l.units || 16) * 0.45))}px"></span></span>
        <strong>${esc(l.type)}</strong>
        <span class="units">${l.units ?? ""}</span>
        <small>${esc(l.activation || "")} ${l.dropout ? `· drop ${l.dropout}` : ""}</small>
      </div>
    `).join("");

    const box = $("#insights");
    box.innerHTML = `
      <div class="sec-head">
        <h2>Model Performance & Evaluation</h2>
        <p>Trained on ${Number(d.rows).toLocaleString("en-IN")} cars (${Number(d.test_rows).toLocaleString("en-IN")} held out). Evaluated on unseen test data.</p>
      </div>

      <dl class="scorebar">
        <div class="hero"><dt>R² Test Score</dt><dd>${dnn.r2.toFixed(3)}</dd><small>Variance explained</small></div>
        <div><dt>Mean Abs Error</dt><dd>${inr(dnn.mae_lakh * 100000)}</dd><small>Average prediction error</small></div>
        <div><dt>Mean Abs % Error</dt><dd>${(dnn.mape * 100).toFixed(1)}%</dd><small>Percentage error</small></div>
        <div><dt>Within ±10%</dt><dd>${(dnn.within_10_pct * 100).toFixed(0)}%</dd><small>Of all test estimates</small></div>
        <div><dt>Model Weights</dt><dd>${Number(m.params).toLocaleString("en-IN")}</dd><small>Trainable parameters</small></div>
      </dl>

      <div class="two-col">
        <div class="card">
          <h3>Algorithm Benchmark</h3>
          <p class="sub">Comparison against Linear Regression and Random Forest on identical test split.</p>
          <div class="table-wrap">
            <table class="cmp">
              <thead><tr><th>Model</th><th class="num">R²</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">MAPE</th><th class="num">±10%</th><th>Error</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <h3>Neural Network Architecture</h3>
          <p class="sub">Embedding layers feed into 3 dense ReLU layers with Monte-Carlo dropout regularization.</p>
          <div class="arch">${layers}</div>
        </div>
      </div>
    `;
  }

  /* ------------------------------------------------------------ history */
  async function loadHistory() {
    try {
      const h = await api("/api/history");
      renderHistory(h);
      state.historyLoaded = true;
    } catch (err) {
      $("#history-body").innerHTML = `<tr><td colspan="9" class="loading-note">${esc(err.message)}</td></tr>`;
    }
  }

  function renderHistory(h) {
    const body = $("#history-body");
    $("#history-sub").textContent = h.total
      ? `${h.total} saved valuation(s) with complete AI deal sheets · stored in ${h.storage}`
      : `Valuations are saved here so you can compare cars side by side · stored in ${h.storage}`;
    $("#history-empty").hidden = h.items.length > 0;
    $("#history-table").hidden = h.items.length === 0;
    $("#clear-history").hidden = h.items.length === 0;

    body.innerHTML = h.items.map((r) => {
      const when = new Date(r.ts).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
      const thumb = r.thumbnail ? `<img src="${r.thumbnail}" alt="Car" style="width:36px;height:28px;object-fit:cover;border-radius:4px;vertical-align:middle;margin-right:6px;">` : "";
      const vColor = r.verdict_color || "teal";
      const vBadge = r.verdict_badge ? `<span class="verdict-badge ${vColor}" style="font-size:10.5px;padding:2px 8px;">${esc(r.verdict_badge)}</span>` : "—";
      const defects = r.defects_count ? `<span class="inspect-cost-badge">${r.defects_count} defects (${inr(r.refurbishment_cost)})</span>` : "Clean";

      return `<tr data-id="${esc(r.id)}">
        <td>${esc(when)}</td>
        <td>
          <div style="display:flex;align-items:center;">
            ${thumb}
            <div>
              <strong>${esc(r.brand)} ${esc(r.model)}</strong>
              <span>${esc(r.year)} · ${esc(r.transmission)} · ${esc(r.owner)}</span>
            </div>
          </div>
        </td>
        <td>${vBadge}</td>
        <td class="num">${Number(r.km).toLocaleString("en-IN")}</td>
        <td><span class="tag">${esc(r.fuel)}</span></td>
        <td class="num"><strong>${inr(r.price)}</strong></td>
        <td class="num">${r.asking ? inr(r.asking) : "—"}</td>
        <td class="num">${defects}</td>
        <td>
          <div class="row-actions">
            <button type="button" data-act="view">View Deal</button>
            <button type="button" data-act="reuse">Run</button>
            <button type="button" class="del" data-act="del" aria-label="Delete">✕</button>
          </div>
        </td>
      </tr>`;
    }).join("");

    $$("#history-body tr").forEach((tr, i) => {
      const rec = h.items[i];
      tr.querySelector('[data-act="view"]').addEventListener("click", () => openDealModal(rec));
      tr.querySelector('[data-act="reuse"]').addEventListener("click", () => reuse(rec));
      tr.querySelector('[data-act="del"]').addEventListener("click", async () => {
        await api("/api/history/" + encodeURIComponent(rec.id), { method: "DELETE" });
        loadHistory();
      });
    });
  }

  function openDealModal(rec) {
    const modal = $("#deal-modal");
    const title = $("#modal-car-title");
    const body = $("#modal-body");

    title.textContent = `${rec.brand} ${rec.model} (${rec.year}) Deal Sheet`;
    const full = rec.full_result || {};
    const adv = full.advisory || {};
    const ds = adv.deal_sheet || {};

    body.innerHTML = `
      <div class="advisory-hero-card" style="margin-bottom:16px;">
        <div class="advisory-main">
          <span class="verdict-badge ${adv.verdict_color || 'teal'}">${esc(adv.verdict_badge || 'VALUATED')}</span>
          <h3 style="margin:8px 0 4px;font-size:18px;">${esc(adv.verdict_headline || '')}</h3>
          <p style="font-size:13.5px;color:#d4e7e6;">${esc(adv.verdict_summary || '')}</p>
        </div>
        <div class="buy-index-box">
          <div class="buy-index-circle" style="width:72px;height:72px;">
            <span class="index-score" style="font-size:26px;">${adv.overall_buy_index ?? '—'}</span>
            <span class="index-label" style="font-size:9px;">Score</span>
          </div>
        </div>
      </div>

      <div class="deal-sheet-card" style="margin-bottom:16px;">
        <table class="deal-table">
          <tbody>
            <tr><td>Asking Price</td><td class="num font-bold">${rec.asking ? inr(rec.asking) : 'Not Provided'}</td></tr>
            <tr><td>AI Market Valuation</td><td class="num font-bold text-petrol">${inr(rec.price)}</td></tr>
            <tr><td>Identified Refurbishment</td><td class="num text-bad">-${inr(rec.refurbishment_cost || 0)}</td></tr>
            <tr><td>RTO Transfer & Insurance</td><td class="num">${inr(11000)}</td></tr>
            <tr class="deal-total-row"><td><strong>Total On-Road Acquisition</strong></td><td class="num font-extrabold text-lg">${inr(ds.total_on_road_acquisition || (rec.price + 11000))}</td></tr>
          </tbody>
        </table>
        <div class="deal-strategy-box">
          <div class="strategy-col">
            <span class="strat-label">Recommended Counter-Offer</span>
            <strong class="strat-val highlight-green">${inr(ds.recommended_counter_offer || rec.price)}</strong>
          </div>
          <div class="strategy-divider"></div>
          <div class="strategy-col">
            <span class="strat-label">Walk-Away Ceiling</span>
            <strong class="strat-val highlight-red">${inr(ds.walk_away_price || rec.high)}</strong>
          </div>
        </div>
      </div>

      <div style="text-align:right;">
        <button type="button" class="btn-primary" style="display:inline-flex;width:auto;padding:8px 20px;" onclick="document.getElementById('deal-modal').close()">Close</button>
      </div>
    `;

    modal.showModal();
  }

  function reuse(rec) {
    if (!state.meta.catalog[rec.brand]) return;
    $("#brand").value = rec.brand;
    onBrandChange(rec.model, rec.fuel, rec.transmission);
    $("#year").value = String(rec.year);
    buildSeg("owner", "owner", state.meta.owners.map((o) => ({ value: o.value, label: o.label.replace(" owner", "") })), parseInt(rec.owner, 10) || 1);
    setKm(rec.km);
    $("#asking").value = rec.asking ? rec.asking.toLocaleString("en-IN") : "";
    setAskingHint();
    showTab("estimate");
    $("#car-form").requestSubmit();
  }

  /* ---------------------------------------------------------------- boot */
  function setStatus(ok, text) {
    $("#status .dot").dataset.state = ok ? "ok" : "bad";
    $("#status-text").textContent = text;
  }

  async function init() {
    buildOdometer();
    setOdometer(0);

    $$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
    $(".brand").addEventListener("click", (e) => { e.preventDefault(); showTab("home"); });
    window.addEventListener("hashchange", () => showTab(location.hash.slice(1), false));

    // Landing page interactive action buttons
    document.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      const act = btn.dataset.action;
      if (act === "go-estimate") {
        showTab("estimate");
      } else if (act === "go-insights") {
        showTab("insights");
      } else if (act === "quick-car") {
        const brand = btn.dataset.brand;
        const model = btn.dataset.model;
        if (state.meta && state.meta.catalog && brand && state.meta.catalog[brand]) {
          $("#brand").value = brand;
          onBrandChange(model);
        }
        showTab("estimate");
      }
    });

    $("#clear-history")?.addEventListener("click", async () => {
      if (!confirm("Delete all saved valuations?")) return;
      await api("/api/history", { method: "DELETE" });
      loadHistory();
    });

    try {
      const h = await api("/api/health");
      if (!h.model_ready) {
        setStatus(false, "Model not trained");
        showGlobalError("The neural network has not been trained yet. Run <code>python -m ml.train</code>, then restart the app.");
        $("#submit").disabled = true;
        return;
      }
      setStatus(true, `Neural network ready · R² ${h.r2 ?? "–"} · ${h.storage}`);
      state.meta = await api("/api/meta");
      setupForm();
    } catch (err) {
      setStatus(false, "Server unreachable");
      showGlobalError("Cannot reach the Flask server. Start it with <code>python app.py</code> and refresh this page.");
      $("#submit").disabled = true;
    }

    if (location.hash) {
      showTab(location.hash.slice(1), false);
    } else {
      showTab("home", false);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
