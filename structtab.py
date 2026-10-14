"""The Structure tab: where a blade is stressed, how it bends and twists, and how it vibrates.

A 3D view of the analysed blade coloured by stress or safety factor (the weakest spot marked), animated
for the deflection under load and for each vibration mode; charts along the blade; a plain-language
diagnosis. Works on the Prop Generator's design or on the prop analysed on the dashboard (with a material
picked here). The GUI module passes itself in as G (no circular import at load time).
"""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
import traceback

import numpy as np

import structure
import structview

SOURCES = ("Generated design", "Dashboard model")
VIEWS = ("Stress", "Deflection", "Vibration modes")


def _hex(c):
    v = int(c[1:], 16)
    return np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], float)


def _ramp(stops, n=256):
    """(n, 3) colour table through the given hex stops (evenly spaced)."""
    cols = np.array([_hex(c) for c in stops])
    t = np.linspace(0, 1, n) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[:, None]
    return cols[i] * (1 - f) + cols[i + 1] * f


CMAP_SF = _ramp(["#ff2d55", "#ff6a3d", "#ffb547", "#ffe066", "#2dffb0", "#22c7ff", "#3b6bff"])
CMAP_MAG = _ramp(["#16224f", "#1f5fff", "#00d0ff", "#2dffb0", "#ffe066", "#ff8a3d", "#ff2d55"])
CMAP_DIV = _ramp(["#2f6bff", "#5fa8ff", "#c9d6ee", "#ff9a7a", "#ff2d55"])


class StressView(tk.Canvas):
    """The analysed blade in 3D, coloured by a stress field; drag to rotate, wheel to zoom. Animates the
    deflection under load (rest -> loaded, exaggerated) or a vibration mode."""

    N_SHADE = 24
    T_REQ = 0.55          # where the target safety factor sits on the colour bar

    def __init__(self, master, G):
        self.G = G
        super().__init__(master, bg=G.CARD, highlightthickness=0, bd=0, width=G.px(420), height=G.px(330))
        self.case = None
        self.field = "sf"
        self.view = "Stress"
        self.mode_i = 0
        self.playing = True
        self.yaw, self.elev, self.zoom = -0.55, math.radians(28), 1.0
        self.drag = None
        self.t0 = time.time()
        self.job = None
        self.visible = False
        self.img = None
        self._geo = None
        G.bind_redraw(self)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "drag", None))
        self.bind("<Double-Button-1>", self._reset)
        self.bind("<MouseWheel>", lambda e: self._zoom(e.delta / 120.0))
        self.bind("<Button-4>", lambda _e: self._zoom(1))
        self.bind("<Button-5>", lambda _e: self._zoom(-1))

    # ------------------------------------------------------------------ state
    def set_case(self, case):
        self.case = case
        self._prep()
        self.redraw()
        self._schedule()

    def set_field(self, key):
        self.field = key
        self._prep()
        self.redraw()

    def set_view(self, view, mode_i=None):
        self.view = view
        if mode_i is not None:
            self.mode_i = mode_i
        self.t0 = time.time()
        self.redraw()
        self._schedule()

    def set_playing(self, on):
        self.playing = on
        self._schedule()

    def show(self, on):
        self.visible = on
        self._schedule()

    def _schedule(self):
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None
        if self.visible and self.playing and self.case is not None and self.view != "Stress":
            self.job = self.after(33, self._tick)

    def _tick(self):
        self.job = None
        self.redraw()
        self._schedule()

    def _press(self, e):
        self.drag = (e.x, e.y, self.yaw, self.elev)

    def _motion(self, e):
        if self.drag:
            x, y, yaw, elev = self.drag
            self.yaw = yaw - (e.x - x) * 0.01
            self.elev = min(max(elev + (e.y - y) * 0.01, -1.45), 1.45)
            self.redraw()

    def _zoom(self, notches):
        self.zoom = min(max(self.zoom * (1.12 ** notches), 0.4), 6.0)
        self.redraw()

    def _reset(self, _e=None):
        self.yaw, self.elev, self.zoom = -0.55, math.radians(28), 1.0
        self.redraw()

    # ------------------------------------------------------------------ data
    def _prep(self):
        c = self.case
        if c is None:
            return
        F = c.fields()[self.field]
        # every other outline point on screen (the outline is closed: first and last are the same point)
        m0 = F.shape[1]
        self.sel = np.unique(np.r_[np.arange(0, m0, 2), m0 - 1])
        F = F[:, self.sel]
        n, m = F.shape
        # quads between stations i, i+1 and outline points j, j+1
        i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
        a, b, cc, d = i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1
        self.faces = np.concatenate([np.stack([a, b, cc], -1).reshape(-1, 3), np.stack([a, cc, d], -1).reshape(-1, 3)])
        fv = F.reshape(-1)[self.faces].mean(axis=1)
        lab, unit, low_bad, div = structview.FIELDS[self.field]
        st = c.st
        if self.field.startswith("sf"):
            # the target safety factor sits at the yellow -> green change; red at the blade's own weakest
            # spot if it is below the target, blue well above it
            req = st["safety_factor"]
            fmin = float(c.fields()[self.field].min())
            lo = min(fmin, req) / 1.05
            hi = max(4 * req, 2.5 * fmin)
            fv_ = np.maximum(fv, 1e-9)
            t = np.where(fv_ < req, self.T_REQ * np.log(fv_ / lo) / np.log(req / lo),
                         self.T_REQ + (1 - self.T_REQ) * np.log(fv_ / req) / np.log(hi / req))
            t = np.clip(t, 0, 1)
            self.cmap, self.crange = CMAP_SF, (lo, hi)
        elif div:
            vmax = max(np.abs(F).max(), 1e-9)
            t = np.clip(0.5 + 0.5 * fv / vmax, 0, 1)
            self.cmap, self.crange = CMAP_DIV, (-vmax, vmax)
        else:
            vmax = max(F.max(), 1e-9)
            t = np.clip(fv / vmax, 0, 1)
            self.cmap, self.crange = CMAP_MAG, (0.0, vmax)
        self.fidx = np.minimum((t * 255).round().astype(int), 255)
        self.label = lab
        self.unit = unit
        self.crit = c.critical()
        # the hub, a plain grey cylinder for scale
        R = c.R
        rh = max(c.hub_radius, 0.04 * R)
        hz = max(0.6 * float(np.ptp(st["points"][0, :, 1])), 0.06 * R)
        th = np.linspace(0, 2 * np.pi, 33)
        ring = np.stack([rh * np.cos(th), rh * np.sin(th)], -1)
        HV = np.concatenate([np.c_[ring, np.full(33, -hz / 2)], np.c_[ring, np.full(33, hz / 2)],
                             [[0, 0, -hz / 2], [0, 0, hz / 2]]])
        HF = []
        for k in range(32):
            HF += [[k, k + 1, 33 + k + 1], [k, 33 + k + 1, 33 + k], [66, k + 1, k], [67, 33 + k, 33 + k + 1]]
        self.hub_V, self.hub_F = HV, np.array(HF)

    def _surface(self):
        c = self.case
        info = ""
        if self.view == "Deflection":
            k = c.deflection_scale()
            s = 0.5 - 0.5 * math.cos(2 * math.pi * ((time.time() - self.t0) / 3.0)) if self.playing else 1.0
            V = c.surface(deflect=k * s)
            info = f"deflection under load × {k:.0f} (exaggerated) · {100 * s:3.0f} % load"
        elif self.view == "Vibration modes" and c.vb:
            modes = c.vb["modes"]
            md = modes[min(self.mode_i, len(modes) - 1)]
            amp = c.mode_amplitude(md)
            ph = math.sin(2 * math.pi * (time.time() - self.t0) / 1.2) if self.playing else 1.0
            V = c.surface(mode=md, amp=amp * ph)
            info = (f"{md['name']} mode · {md['freq_design']:.0f} Hz = {md['per_rev']:.2f} per rev"
                    + (f" · RESONANT with {md['nearest_order']}P" if md["resonant"] else
                       f" · {100 * md['margin']:.0f} % from {md['nearest_order']}P"))
        else:
            V = c.surface()
        return V, info

    # ------------------------------------------------------------------ drawing
    def _project(self, P, w, h, center, scale):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ce, se = math.cos(self.elev), math.sin(self.elev)
        Q = P - center
        x = Q[:, 0] * cy - Q[:, 1] * sy
        y = Q[:, 0] * sy + Q[:, 1] * cy
        z = Q[:, 2]
        up = y * se + z * ce
        near = z * se - y * ce
        s = scale * self.zoom
        return w / 2 + x * s, h / 2 - up * s, near

    def redraw(self):
        G = self.G
        px = G.px
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 80 or h < 80:
            return
        G.round_rect(self, 1, 1, w - 2, h - 2, px(12), fill=G.CARD, outline=G.BORDER)
        self.create_line(px(6), px(18), px(6), px(6), px(18), px(6), fill=G.CYAN, width=px(2))
        self.create_text(px(18), px(20), text=G.spaced("Blade structure"), anchor="w", font=G.cfont(11, "bold"),
                         fill=G.TEXT)
        c = self.case
        if c is None:
            self.create_text(w / 2, h / 2, text="Generate a prop, or run a simulation on the dashboard\n"
                                                "and press Analyse, to see where its blades are stressed",
                             fill=G.MUTED, font=G.cfont(12), justify="center")
            return
        x0, y0, x1, y1 = px(8), px(34), w - px(8), h - px(52)
        W, H = max(x1 - x0, 1), max(y1 - y0, 1)
        V3full, info = self._surface()
        V3 = V3full[:, self.sel]
        n, m, _ = V3.shape
        V = V3.reshape(-1, 3)
        allV = np.concatenate([V, self.hub_V])
        F = np.concatenate([self.faces, self.hub_F + len(V)])
        R = c.R
        center = np.array([0.45 * R, 0.0, 0.0])
        scale = min(W / (1.05 * R), H / (0.75 * R))
        X, Y, near = self._project(allV, W, H, center, scale)
        # face normals (blade outline runs TE -> upper -> LE -> lower, stations outward)
        p0, p1, p2 = allV[F[:, 0]], allV[F[:, 1]], allV[F[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ce, se = math.cos(self.elev), math.sin(self.elev)
        ny = nrm[:, 0] * sy + nrm[:, 1] * cy
        nn = nrm[:, 2] * se - ny * ce                       # towards the viewer
        nu = ny * se + nrm[:, 2] * ce
        nxv = nrm[:, 0] * cy - nrm[:, 1] * sy
        L = np.array([-0.35, 0.55, 0.75])
        L /= np.linalg.norm(L)
        lam = np.abs(nxv * L[0] + nu * L[1] + nn * L[2])    # two-sided (thin, twisted surfaces)
        inten = 0.30 + 0.70 * lam
        nb = len(self.faces)
        base = np.empty((len(F), 3))
        base[:nb] = self.cmap[self.fidx]
        base[nb:] = _hex(G.BORDER2) * 0.9 + 25
        rgb = np.clip(base * inten[:, None], 0, 255).astype(np.uint8)
        order = np.argsort(near[F].mean(axis=1))            # far first
        img = G.rasterize(X, Y, F[order], rgb[order], W, H, G._rgb(G.CARD))
        self.img = tk.PhotoImage(data=b"P6 %d %d 255\n" % (W, H) + img.tobytes(), format="PPM")
        self.create_image(x0, y0, anchor="nw", image=self.img)
        # the weakest spot
        i, j, r_c, x_c, side = self.crit
        Xc, Yc, _ = self._project(V3full[i, j][None, :], W, H, center, scale)
        cx, cy_ = float(Xc[0]) + x0, float(Yc[0]) + y0
        sf_c = c.fields()["sf"][i, j]
        rr = px(9)
        self.create_oval(cx - rr, cy_ - rr, cx + rr, cy_ + rr, outline=G.TEXT, width=px(2))
        self.create_oval(cx - px(3), cy_ - px(3), cx + px(3), cy_ + px(3), fill=G.TEXT, outline="")
        tag = f"weakest · SF {sf_c:.2f} · r {1e3 * r_c:.0f} mm"
        tf = G.cfont(10, "bold")
        tw = tf.measure(tag)
        lx = cx + px(14) if cx + px(14) + tw + px(10) < w - px(8) else cx - px(14) - tw - px(10)
        ly = max(y0 + px(4), cy_ - px(24))
        G.round_rect(self, lx, ly, lx + tw + px(10), ly + px(18), px(7), fill=G.CARD2, outline=G.BORDER2)
        self.create_text(lx + px(5), ly + px(9), text=tag, anchor="w", font=tf, fill=G.TEXT)
        # the colour bar
        bx0, bx1, by = px(16), min(w - px(16), px(16) + px(320)), h - px(30)
        bw, bh = int(bx1 - bx0), px(8)
        key = (id(self.cmap), bw, bh)
        if getattr(self, "_bar_key", None) != key:
            row = self.cmap[np.minimum((np.arange(bw) / max(bw - 1, 1) * 255).astype(int), 255)].astype(np.uint8)
            bar = np.repeat(row[None, :, :], bh, axis=0)
            self._bar_img = tk.PhotoImage(data=b"P6 %d %d 255\n" % (bw, bh) + bar.tobytes(), format="PPM")
            self._bar_key = key
        self.create_image(bx0, by, anchor="nw", image=self._bar_img)
        lo, hi = self.crange
        f10 = G.cfont(10)
        if self.field.startswith("sf"):
            req = c.st["safety_factor"]
            for frac, val in ((0.0, lo), (self.T_REQ, req), (1.0, hi)):
                self.create_text(bx0 + (bx1 - bx0) * frac, by + px(17), text=f"{val:.2g}" + (" (target)" if frac == 0.5
                                                                                            else ""),
                                 font=f10, fill=G.TEXT2)
        else:
            for frac in (0.0, 0.5, 1.0):
                val = lo + (hi - lo) * frac
                self.create_text(bx0 + (bx1 - bx0) * frac, by + px(17), text=f"{val:.3g}", font=f10, fill=G.TEXT2)
        self.create_text(bx1 + px(10), by + px(4), text=f"{self.label}" + (f" [{self.unit}]" if self.unit else ""),
                         anchor="w", font=f10, fill=G.TEXT2)
        if info:
            self.create_text(w - px(16), px(20), text=info, anchor="e", font=G.cfont(10), fill=G.AMBER if "RESONANT"
                             in info else G.TEXT2)
        else:
            self.create_text(w - px(16), px(20), text="drag to rotate · wheel to zoom · double-click resets",
                             anchor="e", font=G.cfont(10), fill=G.MUTED)


class StructTab:
    """Builds the tab in page and keeps it up to date."""

    def __init__(self, app, page, G):
        self.app, self.G, self.page = app, G, page
        self.case = None
        self.busy = False
        self.design = None          # latest generated design
        self.outcome = None         # latest dashboard result
        px = G.px
        v = app.v
        for k, val in dict(st_src=SOURCES[0], st_view=VIEWS[0], st_field=structview.FIELDS["sf"][0], st_mode="",
                           st_sup="Generic", st_proc="FDM 3D print", st_mat="PLA", st_sf="2", st_alt="30",
                           st_chA="Stress", st_chB="Deflection").items():
            v[k] = tk.StringVar(value=val)
        page.columnconfigure(0, weight=3, uniform="st")
        page.columnconfigure(1, weight=2, uniform="st")
        page.rowconfigure(1, weight=3)
        page.rowconfigure(2, weight=2)
        ctk = G.ctk
        # ---- controls
        card = ctk.CTkFrame(page, fg_color=G.CARD, corner_radius=12, border_width=1, border_color=G.BORDER)
        card.grid(row=0, column=0, columnspan=2, sticky="ew", padx=px(4), pady=px(4))
        b = tk.Frame(card, bg=G.CARD)
        b.pack(fill="x", padx=px(12), pady=px(8))
        app._label(b, "Analyse", "Generated design: the Prop Generator's latest design, with its material, "
                   "process and loads. Dashboard model: the prop analysed on the dashboard (STL / STEP or the "
                   "test prop), at its operating point, in the material and process picked here.").grid(
            row=0, column=0, sticky="w")
        app._seg(b, "st_src", list(SOURCES), command=lambda _v: self._source_changed()).grid(
            row=0, column=1, sticky="w", padx=(px(8), px(14)))
        self.mat_box = tk.Frame(b, bg=G.CARD)
        self.mat_box.grid(row=1, column=0, columnspan=5, sticky="w", pady=(px(6), 0))
        sup_names = list(structure.SUPPLIER_NAMES)
        self.sup_menu = app._option(self.mat_box, "st_sup", sup_names, width=110)
        self.sup_menu.grid(row=0, column=0, padx=(0, px(4)))
        self.proc_menu = app._option(self.mat_box, "st_proc", structure.supplier_processes("Generic"), width=150)
        self.proc_menu.grid(row=0, column=1, padx=px(4))
        self.mat_menu = app._option(self.mat_box, "st_mat", structure.supplier_materials("Generic", "FDM 3D print"),
                                    width=170)
        self.mat_menu.grid(row=0, column=2, padx=px(4))
        app._unit(self.mat_box, "safety factor").grid(row=0, column=3, padx=(px(10), px(2)))
        app._entry(self.mat_box, "st_sf", width=44).grid(row=0, column=4)
        app._unit(self.mat_box, "vibration load %").grid(row=0, column=5, padx=(px(10), px(2)))
        app._entry(self.mat_box, "st_alt", width=44).grid(row=0, column=6)
        v["st_sup"].trace_add("write", lambda *_: self._sup_changed())
        v["st_proc"].trace_add("write", lambda *_: self._proc_changed())
        self.run_btn = app._button(b, "Analyse", self.analyse, width=96, primary=True)
        self.run_btn.grid(row=0, column=2, sticky="w")
        b.columnconfigure(3, weight=1)
        self.status = ctk.CTkLabel(b, text="", font=G.font(11), text_color=G.MUTED, anchor="e", justify="right")
        self.status.grid(row=0, column=3, sticky="e", padx=(px(10), 0))
        # ---- 3D view with its own bar
        vcard = ctk.CTkFrame(page, fg_color=G.CARD, corner_radius=12, border_width=1, border_color=G.BORDER)
        vcard.grid(row=1, column=0, sticky="nsew", padx=px(4), pady=px(4))
        vcard.rowconfigure(1, weight=1)
        vcard.columnconfigure(0, weight=1)
        bar = tk.Frame(vcard, bg=G.CARD)
        bar.grid(row=0, column=0, sticky="ew", padx=px(10), pady=(px(8), 0))
        bar.columnconfigure(1, weight=1)
        app._seg(bar, "st_view", list(VIEWS), command=lambda _v: self._view_changed()).grid(row=0, column=0,
                                                                                            sticky="w")
        self.play_btn = app._button(bar, "Pause", self._toggle_play, width=70)
        self.play_btn.grid(row=0, column=2, sticky="e")
        row2 = tk.Frame(bar, bg=G.CARD)
        row2.grid(row=1, column=0, columnspan=3, sticky="w", pady=(px(6), 0))
        app._unit(row2, "colour").grid(row=0, column=0, padx=(0, px(4)))
        self.field_menu = app._option(row2, "st_field", [f[0] for f in structview.FIELDS.values()], width=190)
        self.field_menu.grid(row=0, column=1)
        v["st_field"].trace_add("write", lambda *_: self._field_changed())
        self.mode_lab = app._unit(row2, "mode")
        self.mode_lab.grid(row=0, column=2, padx=(px(10), px(4)))
        self.mode_menu = app._option(row2, "st_mode", ["—"], width=190)
        self.mode_menu.grid(row=0, column=3)
        v["st_mode"].trace_add("write", lambda *_: self._view_changed())
        self.view3d = StressView(vcard, G)
        self.view3d.grid(row=1, column=0, sticky="nsew", padx=px(3), pady=px(3))
        # ---- diagnosis
        dcard, db = app._card(page, "Diagnosis", "", accent=G.GREEN)
        dcard.grid(row=1, column=1, sticky="nsew", padx=px(4), pady=px(4))
        db.rowconfigure(0, weight=1)
        self.diag = ctk.CTkTextbox(db, font=G.font(12), fg_color=G.CARD, text_color=G.TEXT2, wrap="word",
                                   border_width=0, activate_scrollbars=True)
        self.diag.grid(row=0, column=0, sticky="nsew")
        tb = self.diag._textbox
        for tag, col in (("ok", G.GREEN), ("warn", G.AMBER), ("bad", G.RED), ("info", G.TEXT2), ("head", G.TEXT)):
            tb.tag_configure(tag, foreground=col, spacing1=G.px(3), spacing3=G.px(5))
        tb.tag_configure("head", font=G.font(13, "bold"))
        self.diag.configure(state="disabled")
        # ---- charts, each with its own switch
        charts = []
        for col, (key, opts, title, yl, yl2, empty) in enumerate((
                ("st_chA", ["Stress", "Safety factor"], "Stress along the blade", ("stress [MPa]", "MPa"), "",
                 "The stresses along the blade appear here"),
                ("st_chB", ["Deflection", "Campbell"], "Deflection and twist", ("deflection [mm]", "mm"),
                 ("twist [°]", "°"), "The deflection under load appears here"))):
            holder = tk.Frame(page, bg=G.BG)
            holder.grid(row=2, column=col, sticky="nsew", padx=px(4), pady=px(4))
            holder.rowconfigure(1, weight=1)
            holder.columnconfigure(0, weight=1)
            app._seg(holder, key, opts, command=lambda _v: self._charts()).grid(row=0, column=0, sticky="e",
                                                                                pady=(0, px(4)))
            ch = G.LineChart(holder, title, "r/R", yl, ylabel2=yl2, empty=empty)
            ch.grid(row=1, column=0, sticky="nsew")
            charts.append(ch)
        self.chA, self.chB = charts
        # the generator's Strength tile opens this tab on the design
        tile = getattr(app, "gen_tiles", {}).get("strength")
        if tile is not None:
            tile.configure(cursor="hand2")
            tile.bind("<Button-1>", lambda _e: self.open_design(), add="+")
            G.Tooltip(tile, "Click for the Structure tab: where the blade is stressed, how it bends and how it "
                            "vibrates.")
        self._source_changed()
        self._set_diag([("info", "Generate a prop in the Prop Generator, or run a simulation on the dashboard and "
                                 "press Analyse here. The blade is shown coloured by its safety factor, with the "
                                 "weakest spot marked; Deflection animates how it bends and twists under load, "
                                 "Vibration modes how it vibrates at each of its natural frequencies.")])

    # ------------------------------------------------------------------ controls
    def _source_changed(self):
        dash = self.app.v["st_src"].get() == SOURCES[1]
        if dash:
            self.mat_box.grid()
        else:
            self.mat_box.grid_remove()
        self.status.configure(text=self._source_text())
        self.analyse(quiet=True)

    def _source_text(self):
        if self.app.v["st_src"].get() == SOURCES[1]:
            o = self.outcome
            return "dashboard: " + (f"{o.name}" if o is not None else "no simulation yet")
        d = self.design
        if d is None:
            return "no generated design yet"
        return f"{d.name} · {d.structure['mat']['material']}, {d.structure['mat']['process'].lower()}"

    def _sup_changed(self):
        sup = self.app.v["st_sup"].get()
        procs = structure.supplier_processes(sup)
        self.proc_menu.configure(values=procs)
        if self.app.v["st_proc"].get() not in procs:
            self.app.v["st_proc"].set(procs[0])
        else:
            self._proc_changed()

    def _proc_changed(self):
        sup, proc = self.app.v["st_sup"].get(), self.app.v["st_proc"].get()
        mats = structure.supplier_materials(sup, proc)
        self.mat_menu.configure(values=mats or ["—"])
        if self.app.v["st_mat"].get() not in mats and mats:
            self.app.v["st_mat"].set(mats[0])

    def _field_changed(self):
        lab = self.app.v["st_field"].get()
        key = next((k for k, f in structview.FIELDS.items() if f[0] == lab), "sf")
        self.view3d.set_field(key)

    def _view_changed(self):
        view = self.app.v["st_view"].get()
        names = self._mode_names()
        mi = names.index(self.app.v["st_mode"].get()) if self.app.v["st_mode"].get() in names else 0
        for w_ in (self.mode_menu, self.mode_lab):
            if view == "Vibration modes":
                w_.grid()
            else:
                w_.grid_remove()
        if view == "Stress":
            self.play_btn.grid_remove()
        else:
            self.play_btn.grid()
        self.view3d.set_view(view, mi)

    def _toggle_play(self):
        on = not self.view3d.playing
        self.view3d.set_playing(on)
        self.play_btn.configure(text="Pause" if on else "Play")

    def _mode_names(self):
        c = self.case
        if c is None or not c.vb:
            return []
        return [f"{m['name']} · {m['freq_design']:.0f} Hz" + (" ⚠" if m["resonant"] else "") for m in c.vb["modes"]]

    def show(self, on):
        self.view3d.show(on)

    def open_design(self):
        if self.design is None:
            return
        self.app.v["st_src"].set(SOURCES[0])
        self._source_changed()
        self.app._show_tab("struct")

    # ------------------------------------------------------------------ inputs from the app
    def on_design(self, d):
        self.design = d
        if self.app.v["st_src"].get() == SOURCES[0]:
            self.analyse(quiet=True)
        else:
            self.status.configure(text=self._source_text())

    def on_outcome(self, o):
        self.outcome = o
        if self.app.v["st_src"].get() == SOURCES[1]:
            self.analyse(quiet=True)

    # ------------------------------------------------------------------ analysis
    def analyse(self, quiet=False):
        app = self.app
        src = app.v["st_src"].get()
        if src == SOURCES[0]:
            d = self.design
            if d is None:
                if not quiet:
                    app._show_banner("Generate a prop in the Prop Generator first.", "NOTHING TO ANALYSE")
                self.status.configure(text=self._source_text())
                return
            case = structview.from_design(d)
            if case is None:
                self.status.configure(text="this design has no strength data")
                return
            self._show(case)
            return
        o = self.outcome
        if o is None:
            if not quiet:
                app._show_banner("Run a simulation on the dashboard first (an STL / STEP, or the built-in test "
                                 "prop).", "NOTHING TO ANALYSE")
            self.status.configure(text=self._source_text())
            return
        try:
            sf = float(app.v["st_sf"].get().replace(",", ".") or 2)
            alt = float(app.v["st_alt"].get().replace(",", ".") or 30) / 100
        except ValueError:
            app._show_banner("Safety factor and vibration load must be numbers.", "CHECK INPUT")
            return
        mat, proc = app.v["st_mat"].get(), app.v["st_proc"].get()
        if self.busy:
            return
        self.busy = True
        self.status.configure(text="analysing…")

        def work():
            try:
                case = structview.analyse_outcome(o, mat, proc, safety_factor=max(sf, 1.0), alt_load=max(alt, 0.0),
                                                  fatigue=alt > 0)
                err = None
            except Exception as exc:  # noqa: BLE001
                case, err = None, f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            app._post(self._done, case, err)
        threading.Thread(target=work, daemon=True).start()

    def _done(self, case, err):
        self.busy = False
        if err:
            self.status.configure(text="analysis failed")
            self.app._show_banner(f"Structure analysis failed: {err}", "ANALYSIS FAILED")
            return
        self._show(case)

    def _show(self, case):
        self.case = case
        names = self._mode_names()
        self.mode_menu.configure(values=names or ["—"])
        if names and self.app.v["st_mode"].get() not in names:
            res = [n for n in names if n.endswith("⚠")]
            self.app.v["st_mode"].set(res[0] if res else names[0])
        self.view3d.set_case(case)
        self._field_changed()
        self._view_changed()
        st = case.st
        self.status.configure(text=self._source_text() + f" · SF {st['sf_overall']:.2f}"
                              + ("" if st["ok"] else f" (target {st['safety_factor']:g})"))
        self._set_diag(case.diagnosis(), head=True)
        self._charts()

    def _set_diag(self, items, head=False):
        tb = self.diag
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        for k, (lv, text) in enumerate(items):
            tag = lv if lv in ("ok", "warn", "bad", "info") else "info"
            if head and k == 0:
                tag = (tag, "head") if lv in ("ok", "warn", "bad") else "head"
            tb.insert("end", text + "\n", tag)
        tb.configure(state="disabled")

    def _charts(self):
        G = self.G
        c = self.case
        A, B = self.chA, self.chB
        if c is None:
            A.set_data(None)
            B.set_data(None)
            return
        st = c.st
        x = st["r"] / c.R
        i = c.critical()[0]
        mk = (float(x[i]), "weakest")
        if self.app.v["st_chA"].get() == "Stress":
            A.title = "Stress along the blade"
            A.set_data(x, [dict(y=st["von_mises"] / 1e6, label="peak", color=G.RED, axis="L", fill=True, unit="MPa"),
                           dict(y=st["sigma_bend_max"] / 1e6, label="bending", color=G.CYAN, axis="L", unit="MPa"),
                           dict(y=st["sigma_cf"] / 1e6, label="centrifugal", color=G.MAGENTA, axis="L", unit="MPa"),
                           dict(y=st["tau"] / 1e6, label="torsion", color=G.VIOLET, axis="L", unit="MPa")],
                       marker=mk, ylabel=("stress [MPa]", "MPa"), ylabel2="")
        else:
            A.title = "Safety factor along the blade"
            req = st["safety_factor"]
            cap = 6 * req
            A.set_data(x, [dict(y=np.minimum(st["sf_static"], cap), label="static", color=G.CYAN, axis="L"),
                           dict(y=np.minimum(st["sf_fatigue"], cap), label="fatigue", color=G.MAGENTA, axis="L")]
                       if st["fatigue"] else
                       [dict(y=np.minimum(st["sf_static"], cap), label="static", color=G.CYAN, axis="L")],
                       marker=mk, hlines=[(req, "L", G.AMBER, f"target {req:g}"), (1.0, "L", G.RED, "failure (1)")],
                       ylabel=("safety factor", "SF"), ylabel2="")
        if self.app.v["st_chB"].get() == "Deflection":
            B.title, B.xlabel, B.xfmt = "Deflection and twist under load", "r/R", None
            B.set_data(x, [dict(y=1e3 * st["deflection_a"], label="along the axis", color=G.CYAN, axis="L", unit="mm"),
                           dict(y=1e3 * st["deflection_t"], label="in the plane", color=G.GREEN, axis="L", unit="mm"),
                           dict(y=np.degrees(st["twist"]), label="twist", color=G.AMBER, axis="R", unit="°")],
                       ylabel=("deflection [mm]", "mm"), ylabel2=("twist [°]", "°"))
        else:
            vb = c.vb
            B.title, B.xlabel, B.xfmt = "Campbell diagram", "rotation speed [rpm]", lambda v_, s_: f"{v_:,.0f}"
            if not vb:
                B.set_data(None, empty="No vibration result for this blade")
                return
            xs = np.asarray(vb["speeds_rpm"], float)
            cols = {"flap": G.CYAN, "lag": G.MAGENTA, "torsion": G.VIOLET}
            ser, seen = [], set()
            for k in vb["orders"]:
                ser.append(dict(y=k * xs / 60, label="1P-4P" if k == vb["orders"][0] else "", color=G.MUTED,
                                axis="L", dash=(4, 4), unit="Hz"))
            fmax = 1.15 * max(vb["orders"]) * xs[-1] / 60
            for md in vb["modes"]:
                if md["freq"][0] > fmax and md["freq_design"] > fmax:
                    continue
                kind = md["kind"]
                ser.append(dict(y=np.minimum(md["freq"], fmax), label=kind if kind not in seen else "",
                                color=G.AMBER if md["resonant"] else cols[kind], axis="L", unit="Hz"))
                seen.add(kind)
            B.set_data(xs, ser, marker=(vb["rpm"], f"{vb['rpm']:,.0f} rpm"), ylabel=("frequency [Hz]", "f [Hz]"),
                       ylabel2="")
