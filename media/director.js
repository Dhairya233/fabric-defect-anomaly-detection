/* Deterministic scene director.
 * Every frame is a pure function of time: Director.frame(t) sets the page to its state at
 * time t (seconds) and returns a signature string; identical signatures mean identical
 * pixels, so the renderer can reuse the previous screenshot.
 * Used for the title cards and, with a "program", to drive the real results page. */
window.Director = (() => {
  const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  let cfg = { program: [], fadeIn: 0.3, fadeOut: 0.3, duration: 0 };
  let S, overlay = {};
  const NS = "http://www.w3.org/2000/svg";

  function mk(tag, css, parent = document.body) {
    const e = document.createElement(tag);
    e.style.cssText = css;
    parent.appendChild(e);
    return e;
  }

  function load(c) {
    cfg = Object.assign(cfg, c);
    S = { cur: { x: 1180, y: 660 }, dirty: 0, chip: "", ripple: null, acts: cfg.program.map(() => ({})) };
    const st = document.createElement("style");
    st.textContent = "*{transition:none!important;animation:none!important;scroll-behavior:auto!important} html{scrollbar-width:none} ::-webkit-scrollbar{display:none}";
    document.head.appendChild(st);
    const fixed = "position:fixed;pointer-events:none;z-index:99999;";
    overlay.fade = mk("div", fixed + "inset:0;background:#F3EEE3;opacity:1");
    if (cfg.cursor) {
      overlay.cursor = mk("div", fixed + "left:0;top:0;width:28px;height:28px;margin:-3px 0 0 -3px");
      overlay.cursor.innerHTML = '<svg viewBox="0 0 24 24" width="28" height="28"><path d="M4 2l15 9-6.5 1.6L9.5 19z" fill="#1B2430" stroke="#F3EEE3" stroke-width="1.6" stroke-linejoin="round"/></svg>';
      overlay.ring = mk("div", fixed + "width:12px;height:12px;border-radius:50%;border:3px solid #A23B2A;opacity:0");
      overlay.hl = mk("div", fixed + "border:3px solid #A23B2A;border-radius:2px;opacity:0;box-shadow:0 0 0 3px rgba(243,238,227,.85)");
      overlay.chip = mk("div", fixed + "left:24px;bottom:22px;background:#1B2430;color:#F3EEE3;font:600 15px 'IBM Plex Sans',sans-serif;letter-spacing:.04em;text-transform:uppercase;padding:9px 14px;border-radius:2px;opacity:0;border-left:5px solid #A23B2A");
    }
    return true;
  }

  function target(a) {
    const el = document.querySelector(a.sel);
    if (!el) throw new Error("missing selector " + a.sel);
    return el;
  }

  function frame(t) {
    const sig = [];
    // reveal animations on cards
    document.querySelectorAll("[data-start]").forEach((el) => {
      const s = parseFloat(el.dataset.start), d = parseFloat(el.dataset.dur || "0.5");
      const e = el.dataset.end ? parseFloat(el.dataset.end) : Infinity;
      const pin = ease(clamp((t - s) / d));
      const pout = Number.isFinite(e) ? ease(clamp((t - e) / 0.4)) : 0;
      const p = pin * (1 - pout);
      const kind = el.dataset.anim || "fade";
      el.style.opacity = p.toFixed(3);
      if (kind === "rise") el.style.transform = `translateY(${((1 - pin) * 18).toFixed(1)}px)`;
      if (kind === "grow") el.style.setProperty("--p", pin.toFixed(3));
      if (kind === "draw") el.style.strokeDashoffset = ((1 - pin) * 1000).toFixed(1);
      sig.push(p.toFixed(3));
    });
    // program for driving the real page
    let hl = null;
    cfg.program.forEach((a, i) => {
      const st = S.acts[i];
      if (t < a.t) return;
      const u = a.dur ? clamp((t - a.t) / a.dur) : 1;
      if (a.do === "scrollTo") {
        if (st.from === undefined) {
          const el = target(a);
          st.from = window.scrollY;
          st.to = Math.max(0, el.getBoundingClientRect().top + window.scrollY - (a.margin ?? 70));
        }
        window.scrollTo(0, st.from + (st.to - st.from) * ease(u));
      } else if (a.do === "moveTo") {
        if (st.from === undefined) {
          const r = target(a).getBoundingClientRect();
          st.from = { ...S.cur };
          st.to = { x: r.left + r.width * (a.fx ?? 0.5), y: r.top + r.height * (a.fy ?? 0.5) };
        }
        S.cur = { x: st.from.x + (st.to.x - st.from.x) * ease(u), y: st.from.y + (st.to.y - st.from.y) * ease(u) };
      } else if (a.do === "click" && !st.done) {
        st.done = true;
        const el = target(a);
        if (el.tagName === "SELECT") { el.value = a.value; el.dispatchEvent(new Event("change", { bubbles: true })); }
        else el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
        S.dirty += 1;
        S.ripple = { t0: t, x: S.cur.x, y: S.cur.y };
      } else if (a.do === "highlight") {
        if (t <= a.t + a.dur) {
          st.el = st.el || target(a);
          hl = { el: st.el, pad: a.pad ?? 6 };
        }
      } else if (a.do === "chip") {
        S.chip = a.text;
      }
    });
    if (overlay.cursor) {
      overlay.cursor.style.transform = `translate(${S.cur.x.toFixed(1)}px,${S.cur.y.toFixed(1)}px)`;
      sig.push(S.cur.x.toFixed(1), S.cur.y.toFixed(1), String(S.dirty), S.chip, String(Math.round(window.scrollY)));
      const rp = S.ripple ? clamp((t - S.ripple.t0) / 0.45) : 1;
      overlay.ring.style.opacity = rp < 1 ? (1 - rp).toFixed(2) : "0";
      if (S.ripple) {
        const r = 12 + rp * 36;
        overlay.ring.style.width = overlay.ring.style.height = r + "px";
        overlay.ring.style.left = S.ripple.x - r / 2 + "px";
        overlay.ring.style.top = S.ripple.y - r / 2 + "px";
      }
      sig.push(rp.toFixed(2));
      if (hl) {
        const r = hl.el.getBoundingClientRect(), p = hl.pad;
        Object.assign(overlay.hl.style, { opacity: "1", left: r.left - p + "px", top: r.top - p + "px", width: r.width + 2 * p + "px", height: r.height + 2 * p + "px" });
        sig.push("hl" + Math.round(r.top) + ":" + Math.round(r.left));
      } else overlay.hl.style.opacity = "0";
      overlay.chip.textContent = S.chip;
      overlay.chip.style.opacity = S.chip ? "1" : "0";
    }
    const a = 1 - clamp(t / cfg.fadeIn) * 1;
    const b = cfg.duration ? clamp((t - (cfg.duration - cfg.fadeOut)) / cfg.fadeOut) : 0;
    const fade = Math.max(a, b);
    overlay.fade.style.opacity = fade.toFixed(3);
    sig.push(fade.toFixed(3));
    return sig.join("|");
  }

  return { load, frame };
})();
