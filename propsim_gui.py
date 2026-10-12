"""PropSim desktop app: a window over propsim.py and cfd3d.py (same calculations, no command line).

Run with  python propsim_gui.py  or build PropSim.exe with build_exe.bat.

Long work (the BEMT run, the 2D section analysis, OpenFOAM) runs in worker threads; they hand their
results to the Tk thread through a queue that after() polls, so Tk is only ever touched from its own
thread. All charts and the 3D preview are drawn on plain tk canvases (no matplotlib in the exe).
"""

from __future__ import annotations

import collections
import contextlib
import io
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
import zlib
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import propsim  # noqa: E402

try:
    import cfd3d  # noqa: E402
    CFD_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - the app works without the 3D CFD module
    cfd3d = None
    CFD_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

try:
    import propgen  # noqa: E402
    import hubgeom  # noqa: E402
    GEN_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - ... and without the propeller generator
    propgen = None
    GEN_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

APP_TITLE = "PropSim - propeller performance estimator"
G = propsim.G
N_TO_LBF = propsim.N_TO_LBF
NF_SIZES = ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]
MODEL_TYPES = (".stl", ".step", ".stp", ".obj", ".ply")

# --------------------------------------------------------------------------- theme

BG = "#05080f"        # window
PANEL = "#080e1b"     # control panel
CARD = "#0c1426"      # cards, tiles, charts
CARD2 = "#111c35"     # raised / selected
INPUT = "#060a15"
BORDER = "#1a2846"
BORDER2 = "#2a4372"
GRID = "#14213b"
TEXT = "#e6f0ff"
TEXT2 = "#a3b3d1"
MUTED = "#7489b3"     # >= 4.5:1 on every background, for small captions
CYAN = "#00e5ff"
CYAN_D = "#0a8fab"
VIOLET = "#8f63ff"
MAGENTA = "#ff2fd1"
GREEN = "#2dffb0"
AMBER = "#ffb547"
RED = "#ff4d6d"

UI_FONTS = ("Bahnschrift", "Segoe UI Variable Text", "Segoe UI", "Inter", "Roboto", "DejaVu Sans",
            "Helvetica", "Arial")
MONO_FONTS = ("Cascadia Mono", "Cascadia Code", "Consolas", "JetBrains Mono", "DejaVu Sans Mono",
              "Liberation Mono", "Courier New", "Courier")


class Theme:
    ui = "Helvetica"
    mono = "Courier"
    scale = 1.0       # customtkinter's DPI scaling; plain canvases have to apply it themselves
    _fonts: dict = {}

    @classmethod
    def init(cls, root):
        fams = set(tkfont.families(root))
        cls.ui = next((f for f in UI_FONTS if f in fams), "Helvetica")
        cls.mono = next((f for f in MONO_FONTS if f in fams), "Courier")
        try:
            cls.scale = float(ctk.ScalingTracker.get_window_scaling(root))
        except Exception:  # noqa: BLE001
            cls.scale = 1.0


def px(v) -> int:
    return int(round(v * Theme.scale))


def font(size, weight="normal", mono=False):
    """CTkFont for customtkinter widgets (they scale it with the window's DPI themselves)."""
    key = ("ctk", size, weight, mono)
    if key not in Theme._fonts:
        Theme._fonts[key] = ctk.CTkFont(family=Theme.mono if mono else Theme.ui, size=size, weight=weight)
    return Theme._fonts[key]


def cfont(size, weight="normal", mono=False):
    """Font for canvas text and ttk: a pixel size, scaled here because canvases do not scale."""
    key = ("tk", size, weight, mono)
    if key not in Theme._fonts:
        Theme._fonts[key] = tkfont.Font(family=Theme.mono if mono else Theme.ui, size=-px(size), weight=weight)
    return Theme._fonts[key]


def blend(a: str, b: str, t: float) -> str:
    """Colour t of the way from a to b; stands in for transparency, which Tk canvases lack."""
    t = min(max(t, 0.0), 1.0)
    ca, cb = int(a[1:], 16), int(b[1:], 16)
    ch = [round(((ca >> s) & 255) * (1 - t) + ((cb >> s) & 255) * t) for s in (16, 8, 0)]
    return "#%02x%02x%02x" % tuple(ch)


def ramp(t: float) -> str:
    """Load colour scale: cyan (low) -> violet -> magenta (high)."""
    t = min(max(t, 0.0), 1.0)
    return blend(CYAN, VIOLET, 2 * t) if t < 0.5 else blend(VIOLET, MAGENTA, 2 * t - 1)


def spaced(s: str) -> str:
    """Letter-spaced caps for captions (Tk has no letter-spacing option)."""
    return "\u2009".join(s.upper())     # thin space


def ellipsize(text: str, f, width: int) -> str:
    """text shortened to fit width: whole '·' parts dropped first, then whole words, so a number is
    never cut into a misleading shorter one ('14.4…' for 14.47 V)."""
    if f.measure(text) <= width:
        return text
    parts = re.split(r"(\s+·\s+)", text)          # [part, separator, part, ...]
    while len(parts) > 1:
        parts = parts[:-2]
        if f.measure("".join(parts)) <= width:
            return "".join(parts)
    words = parts[0].split(" ")
    while len(words) > 1:
        words.pop()
        t = " ".join(words).rstrip(" ,:;·") + "…"
        if f.measure(t) <= width:
            return t
    text = parts[0]                                 # one long word: cut it
    while text and f.measure(text + "…") > width:
        text = text[:-1]
    return text + "…"


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def round_rect(cv, x0, y0, x1, y1, r, **kw):
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
           x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return cv.create_polygon(pts, smooth=True, **kw)


def glow_line(cv, pts, color, width=2.0, bg=CARD, layers=((7, 0.88), (4, 0.66)), **kw):
    """Neon line: wide dim strokes under a thin bright one."""
    for w, t in layers:
        cv.create_line(pts, fill=blend(color, bg, t), width=px(w), capstyle="round", joinstyle="round", **kw)
    return cv.create_line(pts, fill=color, width=px(width), capstyle="round", joinstyle="round", **kw)


def hfade(cv, x0, x1, y, c0, c1, width=1, steps=32, **kw):
    """Horizontal line fading from c0 to c1."""
    if x1 - x0 < 4:
        return
    dx = (x1 - x0) / steps
    for i in range(steps):
        cv.create_line(x0 + i * dx, y, x0 + (i + 1) * dx + 1, y, fill=blend(c0, c1, i / (steps - 1)),
                       width=width, **kw)


def gradient_line(cv, x0, x1, y, stops, width=1, steps=90, **kw):
    """Horizontal line through several colour stops."""
    dx = (x1 - x0) / steps
    for i in range(steps):
        t = i / (steps - 1) * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        cv.create_line(x0 + i * dx, y, x0 + (i + 1) * dx + 1, y, fill=blend(stops[k], stops[k + 1], t - k),
                       width=width, **kw)


def hatch(cv, x0, y0, x1, y1, color, gap=7, **kw):
    """45 degree hatching inside a rectangle (the hazard look marks stalled spans)."""
    hgt = y1 - y0
    c = x0 - hgt
    while c < x1:
        xa, ya, xb, yb = c, y1, c + hgt, y0
        if xa < x0:
            xa, ya = x0, y1 - (x0 - c)
        if xb > x1:
            xb, yb = x1, y0 + (c + hgt - x1)
        if xb > xa:
            cv.create_line(xa, ya, xb, yb, fill=color, **kw)
        c += gap


def nice_ticks(lo, hi, n=5, inside=0):
    """Round axis limits and tick positions covering [lo, hi].

    inside: for an axis that is not widened to the round limits (x), the minimum number of ticks that
    must fall within [lo, hi]; the step gets finer until they do (r/R 0.16-0.98 would otherwise get
    the single tick 0.5)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        pad = abs(hi) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    steps = [m * mag * d for d in (0.1, 1) for m in (1, 2, 2.5, 5)] + [10 * mag]
    k = next(i for i, s in enumerate(steps) if s >= raw * 0.999)
    while k > 0 and math.floor(hi / steps[k] + 1e-9) - math.ceil(lo / steps[k] - 1e-9) + 1 < inside:
        k -= 1
    step = steps[k]
    t0, t1 = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    return t0, t1, step, np.arange(t0, t1 + step * 0.5, step)


def fmt_tick(v, step):
    if abs(v) >= 10000 and step >= 1000:
        return f"{v / 1000:g}k"
    lead = round(step / 10 ** math.floor(math.log10(step)), 3)
    d = max(0, -math.floor(math.log10(step) + 1e-9) + (1 if lead == 2.5 else 0))
    return f"{v:.{d}f}"


# one blade of the prop logo: (radius / R, angle offset) outline points, drawn as a smoothed polygon
LOGO_BLADE = ((0.22, 0.0), (0.55, 0.32), (0.86, 0.24), (0.9, -0.05), (0.6, -0.2), (0.22, -0.12))


def _smooth_closed(pts, n=8):
    """Tk's smooth=True outline of a closed polygon: quadratic Beziers between edge midpoints."""
    pts = np.asarray(pts, float)
    out = []
    for i in range(len(pts)):
        p0, p1, p2 = (pts[i - 1] + pts[i]) / 2, pts[i], (pts[i] + pts[(i + 1) % len(pts)]) / 2
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append((1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t ** 2 * p2)
    return np.vstack(out)


def logo_rgba(size, ss=4):
    """The prop logo as an RGBA uint8 image (window and exe icon); drawn with numpy, no PIL."""
    n = size * ss
    c = (np.arange(n) + 0.5) / n * 2 - 1                  # -1..1 across the image
    X, Y = np.meshgrid(c, c)
    rho = np.hypot(X, Y)
    img = np.zeros((n, n, 4))

    def paint(mask, color, alpha=1.0):
        rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float)
        img[mask, :3] = rgb
        img[mask, 3] = 255 * alpha
    paint(rho <= 0.97, CARD)
    paint(np.abs(rho - 0.89) <= 0.055, CYAN)              # rim
    for k in range(3):
        a = -math.pi / 2 + k * 2 * math.pi / 3
        poly = _smooth_closed([(0.95 * s * math.cos(a + o), 0.95 * s * math.sin(a + o)) for s, o in LOGO_BLADE])
        inside = np.zeros_like(rho, bool)                   # even-odd point-in-polygon
        for (xa, ya), (xb, yb) in zip(poly, np.roll(poly, -1, axis=0)):
            cross = (ya > Y) != (yb > Y)
            xc = xa + (Y - ya) * (xb - xa) / np.where(yb == ya, 1e-12, yb - ya)
            inside ^= cross & (X < xc)
        paint(inside, blend(CYAN, VIOLET, k / 2))
    paint(rho <= 0.13, TEXT)
    img = img.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))   # supersampled -> anti-aliased
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def png_bytes(rgba):
    """Minimal PNG encoder (Tk 8.6 reads PNG photo data)."""
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def ico_bytes(sizes=(16, 24, 32, 48, 64, 256)):
    """A Windows .ico of the logo with 32-bit BMP images (read by Windows, Tk and PyInstaller alike)."""
    images = []
    for s in sizes:
        rgba = logo_rgba(s)
        bgra = rgba[::-1, :, [2, 1, 0, 3]].tobytes()                      # bottom-up rows
        mask = b"\x00" * (((s + 31) // 32) * 4 * s)                        # AND mask unused with alpha
        images.append(struct.pack("<IiiHHIIiiII", 40, s, 2 * s, 1, 32, 0, len(bgra) + len(mask), 0, 0, 0, 0)
                      + bgra + mask)
    out = struct.pack("<HHH", 0, 1, len(sizes))
    off = 6 + 16 * len(sizes)
    for s, data in zip(sizes, images):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), off)
        off += len(data)
    return out + b"".join(images)


def fmt_si(v, digits=3):
    """Compact number for readouts."""
    if not math.isfinite(v):
        return "-"
    a = abs(v)
    if a >= 1e5:
        return f"{v / 1e3:.0f}k"
    if a >= 1e3:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.{digits}g}"


# --------------------------------------------------------------------------- small widgets

def bind_redraw(canvas):
    """Redraw a canvas widget when its size changes: once per idle round however many <Configure>
    events arrive (a window resize sends dozens), and not at all when only its position changed."""
    state = {"size": None, "job": None}

    def run():
        state["job"] = None
        try:
            canvas.redraw()
        except tk.TclError:
            pass

    def on_configure(e):
        size = (e.width, e.height)
        if size == state["size"]:
            return
        state["size"] = size
        if state["job"] is None:
            state["job"] = canvas.after_idle(run)

    canvas.bind("<Configure>", on_configure)


class SmoothScroll:
    """Mouse-wheel scrolling for customtkinter scrollable frames, replacing their own handler.

    customtkinter moves the content once per wheel event (20 px a notch on Windows) and recomputes
    the scroll region every time the content moves; each move repaints every widget in the panel,
    so a fast spin of the wheel queues more repaints than the screen can show and the panel lags,
    tears and keeps scrolling after the wheel has stopped. Here the wheel only adds to a pending
    distance, applied at most once per frame (~60 per second) in one move, and the scroll region
    is updated only when the content's size changes."""

    PX_PER_NOTCH = 60
    FRAME_MS = 16

    def __init__(self, root):
        self.root, self.frames, self.pending, self.job = root, [], {}, None
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")
        root.bind_all("<MouseWheel>", lambda e: self._wheel(e, e.delta / 120.0), add="+")
        root.bind_all("<Button-4>", lambda e: self._wheel(e, 1.0), add="+")
        root.bind_all("<Button-5>", lambda e: self._wheel(e, -1.0), add="+")

    def add(self, sc):
        cv = sc._parent_canvas          # customtkinter keeps the scrolling canvas here (no public API)
        cv.configure(yscrollincrement=1)
        size = {"v": None}

        def on_configure(e):
            if (e.width, e.height) != size["v"]:
                size["v"] = (e.width, e.height)
                cv.configure(scrollregion=cv.bbox("all"))
        tk.Frame.bind(sc, "<Configure>", on_configure)      # replaces customtkinter's per-move update
        self.frames.append((str(cv), cv))

    def _target(self, widget):
        try:
            path = str(widget)
        except Exception:  # noqa: BLE001
            return None
        for name, cv in self.frames:
            if path == name or path.startswith(name + "."):
                return cv
        return None

    def _wheel(self, e, notches):
        Tooltip.hide_all()
        # the widget under the pointer: Tk on Windows delivers wheel events to the focus widget
        try:
            w = self.root.winfo_containing(e.x_root, e.y_root)
        except (tk.TclError, KeyError, AttributeError):
            w = None
        if w is None or isinstance(w, str):
            return
        if isinstance(w, (tk.Text, ctk.CTkTextbox)) or isinstance(getattr(w, "master", None), ctk.CTkTextbox):
            return          # text boxes scroll themselves
        cv = self._target(w)
        if cv is None or cv.yview() == (0.0, 1.0):
            return
        self.pending[cv] = self.pending.get(cv, 0.0) - notches * px(self.PX_PER_NOTCH)
        if self.job is None:
            self.job = self.root.after(self.FRAME_MS, self._flush)

    def _flush(self):
        self.job = None
        for cv, d in list(self.pending.items()):
            n = int(d)
            self.pending[cv] = d - n
            if n:
                try:
                    cv.yview_scroll(n, "units")
                except tk.TclError:
                    self.pending.pop(cv, None)

class Tooltip:
    """Hover help for a widget. At most one is shown; scrolling, moving or resizing the window and
    switching tabs hide it (Tk sends no <Leave> when a widget moves away under a still pointer, so a
    tooltip would otherwise stay behind, floating where its widget used to be)."""

    _live: set = set()

    def __init__(self, widget, text):
        self.widget, self.text, self.win, self.job = widget, text, None, None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")

    @classmethod
    def hide_all(cls, _e=None):
        for t in list(cls._live):
            t._leave()

    def _enter(self, _e=None):
        if self.job is None:
            self.job = self.widget.after(450, self._show)
            Tooltip._live.add(self)

    def _leave(self, _e=None):
        if self.job:
            try:
                self.widget.after_cancel(self.job)
            except tk.TclError:
                pass
            self.job = None
        if self.win:
            try:
                self.win.destroy()
            except tk.TclError:
                pass
            self.win = None
        Tooltip._live.discard(self)

    def _pointer_inside(self):
        """Is the mouse still over the widget (it may have scrolled away)?"""
        try:
            x, y = self.widget.winfo_pointerxy()
            w = self.widget.winfo_containing(x, y)
        except (tk.TclError, KeyError):
            return False
        return w is not None and str(w).startswith(str(self.widget))

    def _show(self):
        self.job = None
        if self.win or not self.text or not self._pointer_inside():
            Tooltip._live.discard(self)
            return
        for t in list(Tooltip._live):         # one tooltip at a time
            if t is not self:
                t._leave()
        Tooltip._live.add(self)
        x = self.widget.winfo_rootx() + px(8)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + px(4)
        self.win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.configure(bg=CYAN_D)
        tk.Label(tw, text=self.text, justify="left", bg=CARD2, fg=TEXT, font=cfont(12), padx=px(10), pady=px(6),
                 wraplength=px(340)).pack(padx=1, pady=1)
        tw.wm_geometry(f"+{x}+{y}")


class CardHeader(tk.Canvas):
    """Card caption: index number, letter-spaced title and a hairline that fades out."""

    def __init__(self, master, title, index="", bg=CARD, accent=CYAN, right=""):
        super().__init__(master, height=px(34), bg=bg, highlightthickness=0, bd=0)
        self.title, self.index, self.bgc, self.accent = title, index, bg, accent
        self.right, self.right_color = right, MUTED
        bind_redraw(self)

    def set_right(self, text, color=MUTED):
        self.right, self.right_color = text, color
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        y = h // 2 + px(2)
        x = px(14)
        if self.index:
            t = self.create_text(x, y, text=self.index, fill=self.accent, font=cfont(11, "bold", True), anchor="w")
            x = self.bbox(t)[2] + px(8)
        t = self.create_text(x, y, text=spaced(self.title), fill=TEXT, font=cfont(12, "bold"), anchor="w")
        x = self.bbox(t)[2] + px(12)
        xr = w - px(14)
        if self.right:
            t = self.create_text(xr, y, text=self.right, fill=self.right_color, font=cfont(11), anchor="e")
            xr = self.bbox(t)[0] - px(10)
        hfade(self, x, xr, y, blend(self.accent, self.bgc, 0.25), self.bgc)


class GlowButton(tk.Canvas):
    """The primary action button: gradient fill, neon halo, a light sweep while busy."""

    def __init__(self, master, text, command, hint="", height=50, stops=(CYAN, VIOLET), bg=PANEL):
        super().__init__(master, height=px(height), bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.text, self.hint, self.command, self.stops, self.bgc = text, hint, command, stops, bg
        self.hover = self.busy = False
        self.busy_text = ""
        self.phase = 0.0
        bind_redraw(self)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", lambda _e: (not self.busy) and self.command())

    def _set_hover(self, on):
        self.hover = on
        self.redraw()

    def set_busy(self, busy, text="Working"):
        start = busy and not self.busy
        self.busy, self.busy_text = busy, text
        self.configure(cursor="watch" if busy else "hand2")
        self.redraw()
        if start:
            self._animate()

    def _animate(self):
        if not self.busy:
            return
        self.phase = (self.phase + 0.02) % 1.0
        self._draw_sweep()
        self.after(30, self._animate)

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20:
            return
        m, r = px(5), px(10)
        x0, y0, x1, y1 = m, m, w - m, h - m
        c0, c1 = self.stops
        if self.busy:
            c0, c1 = blend(c0, self.bgc, 0.75), blend(c1, self.bgc, 0.75)
        elif self.hover:
            c0, c1 = blend(c0, "#ffffff", 0.2), blend(c1, "#ffffff", 0.2)
        for i, t in enumerate((0.9, 0.8, 0.66)):          # halo
            g = m - i - 1
            round_rect(self, g, g, w - g, h - g, r + i + 1, outline=blend(c0, self.bgc, t), fill="")
        # gradient fill clipped to the rounded rectangle, one vertical line per pixel column
        for x in range(x0, x1):
            d = min(x - x0, x1 - 1 - x)
            inset = r - math.sqrt(max(r * r - (r - d) ** 2, 0)) if d < r else 0
            self.create_line(x, y0 + inset, x, y1 - inset, fill=blend(c0, c1, (x - x0) / max(x1 - x0, 1)))
        label = self.busy_text if self.busy else self.text
        self.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=spaced(label), fill=TEXT if self.busy else "#03101a",
                         font=cfont(15, "bold"))
        if self.hint and not self.busy:
            self.create_text(x1 - px(12), (y0 + y1) // 2, text=self.hint, fill=blend("#03101a", c1, 0.12),
                             font=cfont(10, "bold"), anchor="e")
        if self.busy:
            self._draw_sweep()

    def _draw_sweep(self):
        self.delete("sweep")
        w, h = self.winfo_width(), self.winfo_height()
        m = px(5)
        xc = m + (w - 2 * m) * self.phase
        for k, t in ((px(40), 0.8), (px(22), 0.55), (px(8), 0.1)):
            a, b = max(m + px(8), xc - k), min(w - m - px(8), xc + k)
            if b > a:
                self.create_line(a, h - m - px(2), b, h - m - px(2), fill=blend(CYAN, self.bgc, t),
                                 width=px(2), tags="sweep")


class TabBar(tk.Canvas):
    """Tabs as captions with a neon underline under the selected one."""

    def __init__(self, master, tabs, command, bg=BG):
        super().__init__(master, height=px(44), bg=bg, highlightthickness=0, bd=0)
        self.tabs, self.command, self.bgc = tabs, command, bg
        self.sel, self.hover, self.badges, self.hit = tabs[0][0], None, {}, []
        self.right_text = ""
        bind_redraw(self)
        self.bind("<Motion>", lambda e: self._set_hover(self._at(e.x)))
        self.bind("<Leave>", lambda _e: self._set_hover(None))
        self.bind("<Button-1>", self._click)

    def _at(self, x):
        return next((k for k, a, b in self.hit if a <= x <= b), None)

    def _set_hover(self, k):
        if k != self.hover:
            self.hover = k
            self.configure(cursor="hand2" if k else "")
            self.redraw()

    def _click(self, e):
        k = self._at(e.x)
        if k:
            self.command(k)

    def select(self, key):
        self.sel = key
        self.redraw()

    def set_badge(self, key, color):
        self.badges[key] = color
        self.redraw()

    def set_right(self, text):
        self.right_text = text
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        base = h - px(6)
        hfade(self, 0, w, base, BORDER2, self.bgc, steps=60)
        x = px(6)
        self.hit = []
        f = cfont(12, "bold")
        for key, label in self.tabs:
            text = spaced(label)
            tw = f.measure(text)
            on = key == self.sel
            col = CYAN if on else (TEXT if key == self.hover else MUTED)
            if on:
                for dx, dy in ((-1, 0), (1, 0), (0, 1)):
                    self.create_text(x + dx, base - px(16) + dy, text=text, fill=blend(CYAN, self.bgc, 0.82),
                                     font=f, anchor="w")
            self.create_text(x, base - px(16), text=text, fill=col, font=f, anchor="w")
            if self.badges.get(key):
                bx = x + tw + px(9)
                self.create_oval(bx - px(3), base - px(19), bx + px(3), base - px(13), fill=self.badges[key], outline="")
                tw += px(12)
            if on:
                glow_line(self, (x, base, x + tw, base), CYAN, width=2, bg=self.bgc, layers=((8, 0.9), (5, 0.72)))
            self.hit.append((key, x - px(8), x + tw + px(8)))
            x += tw + px(34)
        if self.right_text:
            f2 = cfont(11)
            self.create_text(w - px(8), base - px(16), text=ellipsize(self.right_text, f2, max(w - x - px(20), 20)),
                             fill=MUTED, font=f2, anchor="e")


class StatusBar(tk.Canvas):
    """Bottom status line with an animated scanner while work is running."""

    COLORS = dict(idle=MUTED, busy=CYAN, ok=GREEN, error=RED, warn=AMBER)

    def __init__(self, master, bg=BG):
        super().__init__(master, height=px(30), bg=bg, highlightthickness=0, bd=0)
        self.bgc, self.text, self.state, self.right = bg, "Ready", "idle", ""
        self.phase = 0
        bind_redraw(self)

    def set(self, text, state=None, right=None):
        self.text = text
        if right is not None:
            self.right = right
        if state and state != self.state:
            start = state == "busy" and self.state != "busy"
            self.state = state
            if start:
                self.after(60, self._animate)
        self.redraw()

    def _animate(self):
        if self.state != "busy":
            return
        self.phase += 1
        self._draw_scanner()
        self.after(60, self._animate)

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        gradient_line(self, 0, w, 0, (blend(CYAN, self.bgc, 0.6), blend(VIOLET, self.bgc, 0.75), self.bgc))
        col = self.COLORS.get(self.state, MUTED)
        x = px(16) + 12 * px(7) + px(12)
        f = cfont(12)
        xr = w
        if self.right:
            t = self.create_text(w - px(14), h // 2, text=self.right, fill=MUTED, font=cfont(11, mono=True), anchor="e")
            xr = self.bbox(t)[0]
        self.create_text(x, h // 2, text=ellipsize(self.text, f, xr - x - px(20)), anchor="w",
                         fill=TEXT if self.state in ("busy", "idle") else col, font=f)
        self._draw_scanner()

    def _draw_scanner(self):
        self.delete("scan")
        h = self.winfo_height()
        col = self.COLORS.get(self.state, MUTED)
        n = 12
        pos = abs((self.phase % (2 * n - 2)) - (n - 1)) if self.state == "busy" else None
        for i in range(n):
            x = px(16) + i * px(7)
            if pos is None:
                c = blend(col, self.bgc, 0.15 if i == 0 else 0.82)
            else:
                c = blend(col, self.bgc, min(0.88, 0.1 + 0.25 * abs(i - pos)))
            self.create_rectangle(x, h // 2 - px(4), x + px(4), h // 2 + px(4), fill=c, outline="", tags="scan")


class AppHeader(tk.Canvas):
    """Title bar: prop logo (spins while busy), glowing title, status chips, gradient rule."""

    def __init__(self, master, bg=BG):
        super().__init__(master, height=px(64), bg=bg, highlightthickness=0, bd=0)
        self.bgc, self.chips, self.angle, self.busy = bg, {}, 0.0, False
        bind_redraw(self)

    def set_chip(self, key, text, color):
        self.chips[key] = (text, color)
        self.redraw()

    def set_busy(self, busy):
        start = busy and not self.busy
        self.busy = busy
        if start:
            self._spin()

    def _spin(self):
        if not self.busy:
            return
        self.angle += 0.22
        self._logo()
        self.after(40, self._spin)

    def _logo(self):
        self.delete("logo")
        cx, cy, r = px(36), self.winfo_height() // 2 - px(2), px(17)
        self.create_oval(cx - r - px(3), cy - r - px(3), cx + r + px(3), cy + r + px(3),
                         outline=blend(CYAN, self.bgc, 0.8), width=px(3), tags="logo")
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=CYAN, width=px(1.5), tags="logo")
        for k in range(3):
            a = self.angle + k * 2 * math.pi / 3
            pts = []
            for s, off in LOGO_BLADE:
                pts += [cx + r * s * math.cos(a + off), cy + r * s * math.sin(a + off)]
            self.create_polygon(pts, smooth=True, fill=blend(CYAN, VIOLET, k / 2), outline="", tags="logo")
        self.create_oval(cx - px(3), cy - px(3), cx + px(3), cy + px(3), fill=TEXT, outline="", tags="logo")

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self._logo()
        x, y = px(68), h // 2 - px(9)
        title = spaced("PropSim")
        tf = cfont(23, "bold")
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (2, 0), (-2, 0)):
            self.create_text(x + dx, y + dy, text=title, fill=blend(CYAN, self.bgc, 0.78), font=tf, anchor="w")
        self.create_text(x, y, text=title, fill=TEXT, font=tf, anchor="w")
        self.create_text(x + px(1), y + px(21), text=spaced("Propeller performance estimator"), fill=MUTED,
                         font=cfont(10, "bold"), anchor="w")
        xr = w - px(18)
        f = cfont(11, "bold")
        for key in reversed(list(self.chips)):
            text, col = self.chips[key]
            x0 = xr - f.measure(text) - px(34)
            round_rect(self, x0, h // 2 - px(14), xr, h // 2 + px(12), px(13), fill=CARD,
                       outline=blend(col, self.bgc, 0.55))
            self.create_oval(x0 + px(11), h // 2 - px(5), x0 + px(19), h // 2 + px(3), fill=col, outline="")
            self.create_text(x0 + px(25), h // 2 - px(1), text=text, fill=TEXT2, font=f, anchor="w")
            xr = x0 - px(10)
        gradient_line(self, 0, w, h - px(3), (blend(CYAN, self.bgc, 0.75), blend(VIOLET, self.bgc, 0.8),
                                              blend(MAGENTA, self.bgc, 0.85)), width=px(4))
        gradient_line(self, 0, w, h - px(2), (CYAN, VIOLET, MAGENTA, self.bgc), width=1)


class OptionCard(ctk.CTkFrame):
    """A selectable card: radio button with a title, a badge and a description."""

    def __init__(self, master, variable, value, title, desc, badge="", command=None):
        super().__init__(master, fg_color=INPUT, border_color=BORDER, border_width=1, corner_radius=9)
        self.variable, self.value, self.command = variable, value, command
        self.columnconfigure(1, weight=1)
        self.dot = tk.Canvas(self, width=px(18), height=px(18), bg=INPUT, highlightthickness=0, bd=0)
        self.dot.grid(row=0, column=0, padx=(px(12), px(8)), pady=(px(10), 0), sticky="w")
        self.title = ctk.CTkLabel(self, text=title, font=font(13, "bold"), text_color=TEXT, anchor="w", height=22)
        self.title.grid(row=0, column=1, sticky="w", pady=(px(9), 0))
        self.badge = ctk.CTkLabel(self, text=f" {badge} ", font=font(10, "bold"), text_color=CYAN, fg_color=CARD2,
                                  corner_radius=8, height=18)
        if badge:
            self.badge.grid(row=0, column=2, padx=(px(4), px(10)), pady=(px(9), 0), sticky="e")
        self.desc = ctk.CTkLabel(self, text=desc, font=font(11), text_color=MUTED, anchor="w", justify="left",
                                 wraplength=250)
        self.desc.grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, px(10)), pady=(0, px(9)))
        for wdg in (self, self.dot, self.title, self.desc, self.badge):
            wdg.bind("<Button-1>", self._pick, add="+")
            wdg.configure(cursor="hand2")
        variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def _pick(self, _e=None):
        self.variable.set(self.value)
        if self.command:
            self.command()

    def refresh(self):
        on = self.variable.get() == self.value
        bgc = CARD2 if on else INPUT
        self.configure(fg_color=bgc, border_color=CYAN_D if on else BORDER)
        d, s = self.dot, px(18)
        d.configure(bg=bgc)
        d.delete("all")
        d.create_oval(px(2), px(2), s - px(2), s - px(2), outline=CYAN if on else BORDER2, width=px(2))
        if on:
            d.create_oval(px(6), px(6), s - px(6), s - px(6), fill=CYAN, outline="")
        self.title.configure(text_color=TEXT if on else TEXT2)


# --------------------------------------------------------------------------- results widgets

class KpiTile(tk.Canvas):
    """One headline number: caption, value + unit, a sub line and optionally a bar."""

    def __init__(self, master, label, accent=CYAN, hero=False, bg=BG):
        super().__init__(master, height=px(94), width=px(140), bg=bg, highlightthickness=0, bd=0)
        self.label, self.accent, self.hero, self.bgc = label, accent, hero, bg
        self.d = dict(value="—", unit="", sub="", extra=(), bar=None, warn=False)
        self.stale = False          # inputs changed since this value was computed
        bind_redraw(self)

    def set(self, value="—", unit="", sub="", extra=(), bar=None, warn=False, label=None):
        if label:
            self.label = label
        self.d = dict(value=value, unit=unit, sub=sub, extra=extra, bar=bar, warn=warn)
        self.redraw()

    def set_stale(self, stale):
        if stale != self.stale:
            self.stale = stale
            self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 40:
            return
        a, d = self.accent, self.d
        if self.hero:
            round_rect(self, 1, 1, w - 2, h - 2, px(12), fill=blend(CARD, a, 0.035), outline=blend(a, self.bgc, 0.72),
                       width=px(3))
            round_rect(self, 2, 2, w - 3, h - 3, px(11), fill="", outline=blend(a, self.bgc, 0.35))
        else:
            round_rect(self, 1, 1, w - 2, h - 2, px(12), fill=CARD, outline=BORDER)
        m, L = px(6), px(12)
        self.create_line(m, m + L, m, m, m + L, m, fill=a, width=px(2))
        self.create_line(w - m - L, h - m, w - m, h - m, w - m, h - m - L, fill=blend(a, CARD, 0.5), width=px(2))
        self.create_text(px(17), px(19), text=spaced(self.label), anchor="w", font=cfont(10, "bold"), fill=MUTED)
        x_right = w - px(14)
        if self.hero and d["extra"]:
            ef, uf = cfont(15, "bold", True), cfont(11)
            y = px(28)
            col_w = 0
            for val, unit in d["extra"]:
                t = self.create_text(x_right, y, text=unit, anchor="e", font=uf, fill=MUTED)
                x = self.bbox(t)[0] - px(5)
                t2 = self.create_text(x, y, text=val, anchor="e", font=ef, fill=TEXT2)
                col_w = max(col_w, x_right - self.bbox(t2)[0])
                y += px(21)
            x_right -= col_w + px(14)
            self.create_line(x_right + px(6), px(18), x_right + px(6), h - px(16), fill=BORDER)
        vf = cfont(32 if self.hero else 24, "bold", True)
        uf = cfont(14 if self.hero else 12)
        base = int(h * 0.64)
        col = AMBER if d["warn"] else (blend(TEXT, a, 0.12) if self.hero else TEXT)
        if self.stale and d["value"] != "—":
            col = blend(col, CARD, 0.45)
        x0 = px(16)
        uw = uf.measure(d["unit"]) + px(6) if d["unit"] else 0
        val = ellipsize(str(d["value"]), vf, max(px(30), x_right - x0 - uw))
        self.create_text(x0, base - vf.metrics("ascent"), text=val, anchor="nw", font=vf, fill=col)
        if d["unit"]:
            x = x0 + vf.measure(val) + px(5)
            self.create_text(x, base - uf.metrics("ascent"), text=d["unit"], anchor="nw", font=uf, fill=TEXT2)
        if d["sub"]:
            sf = cfont(11)
            self.create_text(x0, h - px(16), text=ellipsize(d["sub"], sf, x_right - x0), anchor="w", font=sf,
                             fill=AMBER if d["warn"] else TEXT2)
        if d["bar"] is not None:
            frac = d["bar"]
            y = h - px(6)
            bx0, bx1 = px(18), w - px(24)
            self.create_line(bx0, y, bx1, y, fill=BORDER, width=px(2))
            self.create_line(bx0, y, bx0 + (bx1 - bx0) * min(max(frac, 0), 1.0), y,
                             fill=AMBER if frac > 1.0001 else a, width=px(2))


class ModelStrip(tk.Canvas):
    """Model name, key geometry as chips, and warning lines."""

    def __init__(self, master, bg=BG):
        super().__init__(master, height=px(36), bg=bg, highlightthickness=0, bd=0)
        self.bgc = bg
        self.name, self.chips, self.warnings = "", [], []
        self.empty, self.stale = "", False
        bind_redraw(self)

    def set(self, name, chips, warnings):
        self.name, self.chips, self.warnings = name, chips, warnings[:3]
        self.configure(height=px(36) + px(22) * len(self.warnings))
        self.redraw()

    def set_empty(self, text):
        self.empty = text
        if not self.name:
            self.redraw()

    def set_stale(self, stale):
        if stale != self.stale:
            self.stale = stale
            self.redraw()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        y = px(17)
        if not self.name:
            self.create_text(px(4), y, anchor="w", font=cfont(13), fill=MUTED, text=ellipsize(self.empty, cfont(13), w - px(8)))
            return
        nf = cfont(15, "bold")
        t = self.create_text(px(4), y, text=ellipsize(self.name, nf, w // 3), anchor="w", font=nf,
                             fill=TEXT2 if self.stale else TEXT)
        x = self.bbox(t)[2] + px(14)
        f = cfont(11)
        chips = ([("inputs changed · press RUN to update", AMBER)] if self.stale else []) + self.chips
        for text, col in chips:
            tw = f.measure(text)
            if x + tw + px(20) > w:
                break
            round_rect(self, x, y - px(11), x + tw + px(18), y + px(11), px(11),
                       fill=blend(col, CARD, 0.85) if col == AMBER else CARD, outline=blend(col, self.bgc, 0.55))
            self.create_text(x + px(9), y, text=text, anchor="w", font=f, fill=AMBER if col == AMBER else TEXT2)
            x += tw + px(26)
        y += px(30)
        for text, col in self.warnings:
            if col == AMBER:
                self.create_polygon(px(5), y + px(6), px(12), y - px(7), px(19), y + px(6), fill=col, outline="")
                self.create_text(px(12), y + px(1), text="!", font=cfont(9, "bold"), fill=BG)
            else:
                self.create_oval(px(5), y - px(7), px(19), y + px(7), outline=col, width=px(1.5))
                self.create_text(px(12), y, text="i", font=cfont(9, "bold"), fill=col)
            self.create_text(px(28), y, text=ellipsize(text, f, w - px(32)), anchor="w", font=f, fill=col)
            y += px(22)


class LineChart(tk.Canvas):
    """Neon line chart with an optional right-hand axis and a hover readout, on a plain canvas.

    series: dicts with y, label, color, axis ("L"/"R"), and optionally fill, dash, unit, dots and hl
    (bool mask of points to mark). ylabel / ylabel2: a string, or alternatives longest first for short
    charts. compact: tighter margins for a deliberately small chart. apart: keep the left and right curves
    apart at the marker (or right end) when two quantities grow together."""

    def __init__(self, master, title, xlabel, ylabel, ylabel2="", empty="Run a simulation to see this chart",
                 zero=True, xfmt=None, compact=False, apart=False, bg=BG):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, height=px(140), width=px(240))
        self.title, self.xlabel, self.ylabel, self.ylabel2 = title, xlabel, ylabel, ylabel2
        self.empty, self.zero, self.xfmt, self.compact, self.apart, self.bgc = empty, zero, xfmt, compact, apart, bg
        self.x, self.series, self.marker, self.bands, self.hlines = None, [], None, [], []
        self.band_label, self.yfrom = "", None
        self.geo = None
        bind_redraw(self)
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda _e: self.delete("hover"))

    def set_data(self, x=None, series=(), marker=None, bands=(), hlines=(), band_label="", empty=None, ylabel=None,
                 ylabel2=None, yfrom=None):
        """yfrom: scale the y axes to the points at x >= yfrom only (earlier ones are drawn clamped to
        the edge), so a start-up transient does not flatten the part that matters."""
        self.yfrom = yfrom
        if empty:
            self.empty = empty
        if ylabel is not None:
            self.ylabel = ylabel
        if ylabel2 is not None:
            self.ylabel2 = ylabel2
        self.x = None if x is None else np.asarray(x, float)
        self.series = [dict(s, y=np.asarray(s["y"], float)) for s in series]
        self.marker, self.bands, self.hlines, self.band_label = marker, list(bands), list(hlines), band_label
        self.redraw()

    def _axis(self, which, n=4):
        keep = self.x >= self.yfrom if self.yfrom is not None else np.ones(len(self.x), bool)
        ys = [s["y"][np.isfinite(s["y"]) & keep] for s in self.series if s.get("axis", "L") == which]
        ys += [np.array([h[0]]) for h in self.hlines if h[1] == which]
        ys = [y for y in ys if len(y)]
        if not ys:
            return None
        lo, hi = float(min(y.min() for y in ys)), float(max(y.max() for y in ys))
        if self.zero:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        else:
            pad = 0.08 * (hi - lo or abs(hi) or 1)
            lo, hi = lo - pad, hi + pad
        return nice_ticks(lo, hi, n)

    def _axis_color(self, ax):
        return next((s["color"] for s in self.series if s.get("axis", "L") == ax), TEXT2)

    def _separate(self, axes, xref, n, gap=0.14):
        """Raise one axis's upper limit until the left and right curves are at least gap of the plot
        height apart at xref (thrust and power both rise with rpm and would land on the same pixels)."""
        left = [s for s in self.series if s.get("axis", "L") == "L"][:1]
        right = [s for s in self.series if s.get("axis") == "R"]
        if not left or not right:
            return axes

        def frac(lim, s):
            return (float(np.interp(xref, self.x, s["y"])) - lim[0]) / (lim[1] - lim[0] or 1)

        def ok(aL, aR):
            return all(abs(frac(aL, left[0]) - frac(aR, r)) >= gap for r in right)
        for k in range(7):
            for ax in ("R", "L"):
                t0, t1, step, _ = axes[ax]
                trial = dict(axes)
                trial[ax] = nice_ticks(t0, t1 + k * step, n) if k else axes[ax]
                if ok(trial["L"], trial["R"]):
                    return trial
        return axes

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 80 or h < 60:
            return
        round_rect(self, 1, 1, w - 2, h - 2, px(12), fill=CARD, outline=BORDER)
        self.create_line(px(6), px(18), px(6), px(6), px(18), px(6), fill=CYAN, width=px(2))
        t = self.create_text(px(18), px(20), text=spaced(self.title), anchor="w", font=cfont(11, "bold"), fill=TEXT)
        title_end = self.bbox(t)[2] + px(16)
        dual = bool(self.ylabel2)
        short = self.compact or h < px(170)     # x-axis title moves onto the tick row
        L, R, T, B = px(60), px(60 if dual else 20), px(36 if short else 44), px(24 if short else 38)
        x0, x1, y0, y1 = L, w - R, T, h - B
        self.geo = None
        if self.x is None or len(self.x) < 2 or not self.series:
            t = self.create_text(w // 2, max((y0 + y1) // 2, px(30) + (h - px(30)) // 2), text=self.empty, fill=MUTED,
                                 font=cfont(12), width=w - px(40), justify="center")
            ta, tb = self.bbox(t)[1] - px(5), self.bbox(t)[3] + px(5)
            if y1 - y0 > 30:
                for i in range(5):
                    y = y0 + (y1 - y0) * i / 4
                    if not ta <= y <= tb:           # no grid line through the message
                        self.create_line(x0, y, x1, y, fill=GRID)
            return
        if x1 - x0 < 40 or y1 - y0 < 30:
            return
        # legend, right-aligned on the title line
        lf = cfont(11)
        xr = w - px(16)
        items = [(s["label"], s["color"], s.get("dash")) for s in reversed(self.series)]
        if self.bands and self.band_label:
            items.insert(0, (self.band_label, AMBER, "band"))
        if self.hlines and short:       # no room to label the reference lines in place
            items.insert(0, (self.hlines[0][3].split(" ")[0], [hl[2] for hl in self.hlines], "ref"))
        for label, col, dash in items:
            if not label:               # series kept out of the legend
                continue
            if xr - lf.measure(label) - px(22) < title_end:
                break
            t = self.create_text(xr, px(20), text=label, anchor="e", font=lf, fill=TEXT2)
            bx = self.bbox(t)[0] - px(6)
            if dash == "band":
                hatch(self, bx - px(12), px(15), bx, px(25), blend(AMBER, CARD, 0.35), gap=px(4))
                self.create_rectangle(bx - px(12), px(15), bx, px(25), outline=AMBER)
            elif dash == "ref":         # one short dash per reference line, in its own colour
                seg = px(16) / len(col)
                for j, c in enumerate(col):
                    self.create_line(bx - px(16) + j * seg, px(20), bx - px(16) + (j + 0.6) * seg, px(20), fill=c,
                                     width=px(2))
            else:
                self.create_line(bx - px(16), px(20), bx, px(20), fill=col, width=px(2), dash=dash or "")
            xr = bx - px(30)
        xlo, xhi = float(np.nanmin(self.x)), float(np.nanmax(self.x))
        nx = max(3, min(7, (x1 - x0) // px(75)))
        if self.marker is not None and self.marker[0] >= xhi - 1e-9 * abs(xhi):
            # the operating point is the last sweep point: give it room, ideally up to the next round tick
            st0 = nice_ticks(xlo, xhi, nx, inside=3)[2]
            nxt = (math.floor(xhi / st0 + 1e-9) + 1) * st0
            xhi = nxt if nxt - xhi <= 0.15 * (xhi - xlo) else xhi + 0.05 * (xhi - xlo)
        _, _, xstep, xt = nice_ticks(xlo, xhi, nx, inside=3)
        n = 3 if short else 4
        axes = {"L": self._axis("L", n), "R": self._axis("R", n) if dual else None}
        if self.apart and axes["L"] and axes["R"]:
            axes = self._separate(axes, self.marker[0] if self.marker is not None else float(np.nanmax(self.x)), n)

        def sx(v):
            return x0 + (v - xlo) / (xhi - xlo or 1) * (x1 - x0)

        def sy(v, ax):
            lo, hi = axes[ax][0], axes[ax][1]
            return min(max(y1 - (v - lo) / (hi - lo or 1) * (y1 - y0), y0), y1)    # off-scale points sit on the edge

        for a, b in self.bands:
            ax_, bx_ = max(x0, sx(a)), min(x1, sx(b))
            if bx_ > ax_:
                hatch(self, ax_, y0, bx_, y1, blend(AMBER, CARD, 0.8), gap=px(8))
                self.create_line(ax_, y0, bx_, y0, fill=AMBER, width=px(2))
        tf = cfont(10, mono=True)
        for ax in ("L", "R"):
            if not axes[ax]:
                continue
            lo, hi, st, ticks = axes[ax]
            tc = blend(self._axis_color(ax), TEXT, 0.35)
            every = 2 if len(ticks) > 1 and (y1 - y0) / (len(ticks) - 1) < px(15) else 1
            for j, v in enumerate(ticks):
                y = sy(v, ax)
                label = fmt_tick(v, st) if j % every == 0 else ""
                if ax == "L":
                    self.create_line(x0, y, x1, y, fill=GRID if abs(v) > 1e-12 else BORDER2)
                    self.create_text(x0 - px(7), y, text=label, anchor="e", font=tf, fill=tc)
                else:
                    self.create_text(x1 + px(9), y, text=label, anchor="w", font=tf, fill=tc)
        af = cfont(11)
        ty = y1 + px(11 if short else 12)
        xt_left = w
        if short:           # x-axis title at the right end of the tick row; ticks under it are skipped
            t = self.create_text(x1, ty, text=self.xlabel, font=af, fill=MUTED, anchor="e")
            xt_left = self.bbox(t)[0] - px(8)
        else:
            self.create_text((x0 + x1) // 2, h - px(11), text=self.xlabel, font=af, fill=MUTED)
        for v in xt:
            if xlo - 1e-9 <= v <= xhi + 1e-9:
                x = sx(v)
                self.create_line(x, y0, x, y1, fill=GRID)
                label = (self.xfmt or fmt_tick)(v, xstep)
                if x + tf.measure(label) / 2 < xt_left:
                    self.create_text(x, ty, text=label, font=tf, fill=MUTED)
        self.create_line(x0, y1, x1, y1, fill=BORDER2)
        room = 2 * min((y0 + y1) / 2 - px(30), h - px(6) - (y0 + y1) / 2)     # below the title, above the edge

        def fit(label):     # the longest alternative that fits the chart's height
            alts = (label,) if isinstance(label, str) else label
            return next((a for a in alts if af.measure(a) <= room), alts[-1])
        self.create_text(px(15), (y0 + y1) // 2, text=fit(self.ylabel), font=af, angle=90, fill=self._axis_color("L"))
        if dual:
            self.create_text(w - px(15), (y0 + y1) // 2, text=fit(self.ylabel2), font=af, angle=270,
                             fill=self._axis_color("R"))
        for yv, ax, col, label in self.hlines:
            if axes.get(ax):
                y = sy(yv, ax)
                self.create_line(x0, y, x1, y, fill=col, dash=(5, 4))
                # right-axis labels at the right end, so two reference lines at similar heights stay legible
                lx, anchor = (x1 - px(6), "e") if ax == "R" else (x0 + px(6), "w")
                if not short:
                    self.create_text(lx, y - px(8), text=label, anchor=anchor, font=cfont(10), fill=col)
        for s in self.series:
            ax = s.get("axis", "L")
            if not axes.get(ax):
                continue
            ok = np.isfinite(s["y"]) & np.isfinite(self.x)
            pts = [c for xv, yv in zip(self.x[ok], s["y"][ok]) for c in (sx(xv), sy(yv, ax))]
            if len(pts) < 4:
                continue
            if s.get("fill"):
                yb = sy(min(max(0.0, axes[ax][0]), axes[ax][1]), ax)
                self.create_polygon([pts[0], yb] + pts + [pts[-2], yb], fill=blend(s["color"], CARD, 0.87), outline="")
            if s.get("dash"):
                self.create_line(pts, fill=s["color"], width=px(1.6), dash=s["dash"])
            else:
                glow_line(self, pts, s["color"], width=2.2)
            if s.get("dots"):       # sparse samples (CFD force history): show where they are
                for cx, cy in zip(pts[::2], pts[1::2]):
                    self.create_oval(cx - px(2.5), cy - px(2.5), cx + px(2.5), cy + px(2.5), fill=s["color"], outline="")
            hl = s.get("hl")
            if hl is not None:
                for i in np.flatnonzero(np.asarray(hl)[ok]):
                    cx, cy = pts[2 * i], pts[2 * i + 1]
                    self.create_oval(cx - px(3.5), cy - px(3.5), cx + px(3.5), cy + px(3.5), fill=AMBER, outline=CARD)
        if self.marker is not None:
            mx_v, label = self.marker
            mx = sx(mx_v)
            self.create_line(mx, y0, mx, y1, fill=TEXT2, dash=(2, 4))
            # right-axis markers first, so the left (primary) one is never the hidden one
            for s in sorted(self.series, key=lambda s: s.get("axis", "L") == "L"):
                ax = s.get("axis", "L")
                if axes.get(ax) and len(s["y"]):
                    cy = sy(float(np.interp(mx_v, self.x, s["y"])), ax)
                    self.create_oval(mx - px(7), cy - px(7), mx + px(7), cy + px(7), outline=blend(s["color"], CARD, 0.55),
                                     width=px(2))
                    self.create_oval(mx - px(3.5), cy - px(3.5), mx + px(3.5), cy + px(3.5), fill=s["color"], outline="")
            mf = cfont(10, "bold")
            tw = mf.measure(label)
            lx = mx - tw - px(16) if mx + tw + px(16) > x1 else mx + px(6)
            round_rect(self, lx, y1 - px(24), lx + tw + px(10), y1 - px(6), px(8), fill=CARD2, outline=BORDER2)
            self.create_text(lx + px(5), y1 - px(15), text=label, anchor="w", font=mf, fill=TEXT)
        self.geo = (x0, x1, y0, y1, axes, sx, sy)

    def _hover(self, e):
        self.delete("hover")
        if not self.geo:
            return
        x0, x1, y0, y1, axes, sx, sy = self.geo
        if not (x0 <= e.x <= x1 and y0 <= e.y <= y1):
            return
        i = int(np.argmin(np.abs([sx(v) - e.x for v in self.x])))
        X = sx(self.x[i])
        self.create_line(X, y0, X, y1, fill=BORDER2, tags="hover")
        name, _, unit = self.xlabel.partition(" [")
        lines = [(f"{name}  {fmt_si(self.x[i], 4)} {unit.rstrip(']')}".rstrip(), TEXT)]
        for s in self.series:
            ax = s.get("axis", "L")
            if not axes.get(ax) or not np.isfinite(s["y"][i]):
                continue
            Y = sy(s["y"][i], ax)
            self.create_oval(X - px(4), Y - px(4), X + px(4), Y + px(4), fill=s["color"], outline=CARD, tags="hover")
            lines.append((f"{s['label']}  {fmt_si(s['y'][i], 4)} {s.get('unit', '')}", s["color"]))
        f = cfont(11, mono=True)
        bw = max(f.measure(t) for t, _ in lines) + px(16)
        bh = px(18) * len(lines) + px(8)
        bx = X + px(12) if X + px(12) + bw < x1 else X - px(12) - bw
        by = max(y0, min(e.y - bh // 2, y1 - bh))
        round_rect(self, bx, by, bx + bw, by + bh, px(8), fill=CARD2, outline=BORDER2, tags="hover")
        for k, (t, c) in enumerate(lines):
            self.create_text(bx + px(8), by + px(13) + k * px(18), text=t, anchor="w", font=f, fill=c, tags="hover")


def section_loop(sec, tc, camber, xs):
    """Closed section outline (upper TE -> LE, lower LE -> TE) sampled at chord stations xs."""
    if sec is not None and len(sec) >= 8:
        sec = np.asarray(sec, float)
        le = int(np.argmin(sec[:, 0]))
        up, lo = sec[:le + 1][::-1], sec[le:]

        def surf(p):
            o = np.argsort(p[:, 0])
            return np.interp(xs, p[o, 0], p[o, 1])
        yu, yl = surf(up), surf(lo)
    else:   # no traced outline: NACA 4-digit shape with the measured thickness and camber
        yt = 5 * tc * (0.2969 * np.sqrt(xs) - 0.126 * xs - 0.3516 * xs ** 2 + 0.2843 * xs ** 3 - 0.1015 * xs ** 4)
        yc = 4 * camber * xs * (1 - xs)
        yu, yl = yc + yt, yc - yt
    return np.column_stack([np.r_[xs[::-1], xs[1:]], np.r_[yu[::-1], yl[1:]]])


def rasterize(X, Y, F, rgb, W, H, bg):
    """Flat-shaded triangles (given far to near) into an (H, W, 3) uint8 image, scanline by scanline and
    vectorised over all triangles; nearer ones overwrite farther ones (painter's algorithm). Tk draws a
    few thousand canvas polygons far too slowly for a turning model; this takes milliseconds."""
    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    if len(F) == 0:
        return img
    x, y = X[F], Y[F]
    ys0 = np.maximum(np.ceil(y.min(1) - 0.5).astype(np.int64), 0)
    ys1 = np.minimum(np.floor(y.max(1) - 0.5).astype(np.int64), H - 1)
    nrow = ys1 - ys0 + 1
    tid = np.nonzero(nrow > 0)[0]
    nr = nrow[tid]
    if nr.sum() == 0:
        return img
    t = np.repeat(tid, nr)
    row = ys0[t] + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
    yc = row + 0.5
    xl = np.full(len(t), 1e9)
    xr = np.full(len(t), -1e9)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        xa, ya, xb, yb = x[t, a], y[t, a], x[t, b], y[t, b]
        lo, hi = np.minimum(ya, yb), np.maximum(ya, yb)
        on = (yc >= lo) & (yc <= hi) & (hi > lo)
        xi = xa + (yc - ya) * (xb - xa) / np.where(hi > lo, yb - ya, 1.0)
        xl = np.where(on, np.minimum(xl, xi), xl)
        xr = np.where(on, np.maximum(xr, xi), xr)
    c0 = np.maximum(np.ceil(xl - 0.5).astype(np.int64), 0)
    c1 = np.minimum(np.floor(xr - 0.5).astype(np.int64), W - 1)
    n = np.maximum(c1 - c0 + 1, 0)
    keep = n > 0
    t, row, c0, n = t[keep], row[keep], c0[keep], n[keep]
    tot = int(n.sum())
    if tot:
        col = np.repeat(c0, n) + (np.arange(tot) - np.repeat(np.cumsum(n) - n, n))
        img.reshape(-1, 3)[np.repeat(row, n) * W + col] = rgb[np.repeat(t, n)]
    return img


def _rgb(hexcol):
    v = int(hexcol[1:], 16)
    return (v >> 16) & 255, (v >> 8) & 255, v & 255


class PropView(tk.Canvas):
    """Wireframe of the analysed blades (measured sections placed at their chord and blade angle),
    coloured by local thrust. Turns in the prop's own sense for a while after a new result or a touch,
    then rests (a constant 25 fps redraw costs an idle laptop ~15 % of a core); drag to rotate, wheel to zoom."""

    NX = 9    # chordwise points per surface
    SPIN_S = 20.0

    def __init__(self, master, bg=BG):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, width=px(320), height=px(300))
        self.yaw, self.elev, self.zoom = 0.6, math.radians(34), 1.0
        self.lines, self.items, self.P = [], [], None
        self.spin, self.info, self.has_load = 1, "", False
        self.drag, self.resume_at, self.empty_phase = None, 0.0, 0.0
        self.t_label, self.job, self.active_until = None, None, 0.0
        self.empty_text = "Awaiting model"
        self.solid, self.mode, self.toggle_box, self.bg_top = None, "model", None, None
        bind_redraw(self)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._reset)
        self.bind("<MouseWheel>", lambda e: self._zoom(1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda _e: self._zoom(1))
        self.bind("<Button-5>", lambda _e: self._zoom(-1))
        self.bind("<Enter>", lambda _e: self.wake(), add="+")
        self.after(500, self.wake)

    def wake(self, seconds=None):
        """Animate for a while (new result, pointer over the view, dashboard shown again)."""
        self.active_until = max(self.active_until, time.time() + (seconds or self.SPIN_S))
        if self.job is None:
            self.job = self.after(40, self._tick)

    def set_empty(self, text):
        self.empty_text = text
        if self.P is None:
            self._draw_empty()

    def clear(self):
        self.lines, self.P, self.solid = [], None, None
        self.redraw()
        self.wake()

    # shaded solid: the real model (every feature), painter's algorithm with back faces culled
    N_SHADE = 24

    def set_solid(self, V, F, frame=None):
        """Show this triangle mesh (model units, e.g. the generator's mm) shaded, in the frame of the last
        set_model (frame: a function mapping model points to the analysis frame); set_model first."""
        if self.P is None or V is None or len(F) == 0:
            self.solid = None
            return
        R = self.solid_R
        Vn = (frame(np.asarray(V, float)) if frame else np.asarray(V, float)) / R
        F = np.asarray(F, int)
        a, b, c = Vn[F[:, 0]], Vn[F[:, 1]], Vn[F[:, 2]]
        Nf = np.cross(b - a, c - a)
        Nf /= np.maximum(np.linalg.norm(Nf, axis=1), 1e-12)[:, None]
        rc = np.clip(np.hypot(*((a + b + c) / 3)[:, :2].T), 0, 1)
        # colour along the radius: violet hub to cyan tip, darker or lighter with the light
        base = [blend(VIOLET, CYAN, t) for t in np.linspace(0, 1, 8)]
        self.palette = np.array([[_rgb(blend(blend(CARD, col, 0.18 + 0.82 * k / (self.N_SHADE - 1)), "#ffffff",
                                             max(0.0, (k / (self.N_SHADE - 1) - 0.85) * 1.6)))
                                  for k in range(self.N_SHADE)] for col in base], np.uint8)
        self.solid = dict(V=Vn, F=F, N=Nf, ci=np.minimum((rc * 8).astype(int), 7))
        if self.mode == "model":
            self.redraw()

    def _toggle_mode(self):
        self.mode = "sections" if self.mode == "model" else "model"
        self.redraw()

    def set_model(self, geom, mesh, res=None, pod=None):
        """pod: a generated single blade's counterweight (propgen PropDesign.counterweight), drawn opposite
        the blade."""
        R, n, B = float(geom.radius), len(geom.r), int(geom.blades)
        self.solid_R, self.solid = R, None
        self.spin = geom.spin_sign or 1
        th_sign = geom.thrust_sign or 1
        nx = self.NX
        xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nx)))
        secs = geom.sections if geom.sections is not None else []
        loops = [section_loop(secs[i] if i < len(secs) else None, geom.thickness[i], geom.camber[i], xs)
                 for i in range(n)]
        if res is not None:
            dT = np.asarray(res.stations["dT"], float)
            load = np.clip(dT / max(float(np.nanmax(dT)), 1e-12), 0, 1)
            stalled = np.asarray(res.stations.get("stalled", np.zeros(n, bool)), bool)
        else:
            load, stalled = np.zeros(n), np.zeros(n, bool)
        show = sorted(set(range(0, n, max(1, n // 13))) | {n - 1})
        # a swept blade: sections displaced along the direction of travel (negative = swept back)
        sw = getattr(geom, "sweep", None)
        off = np.zeros(n)
        if sw is not None and len(sw) == n:
            t_ = np.tan(np.asarray(sw, float))
            off[1:] = -np.cumsum(0.5 * (t_[1:] + t_[:-1]) * np.diff(np.asarray(geom.r, float)))
        lines = []      # (points (m, 3) in units of R, kind, colour)
        ea = np.array([0.0, 0.0, float(th_sign)])
        for k in range(B):
            th = 2 * math.pi * k / B
            er = np.array([math.cos(th), math.sin(th), 0.0])
            et = self.spin * np.array([-math.sin(th), math.cos(th), 0.0])
            P = np.empty((n, len(loops[0]), 3))
            for i in range(n):
                c, b = geom.chord[i], geom.beta[i]
                ec = -math.cos(b) * et - math.sin(b) * ea     # leading edge -> trailing edge
                en = -math.sin(b) * et + math.cos(b) * ea     # towards the suction side
                s = (loops[i][:, 0] - 0.25) * c
                P[i] = geom.r[i] * er + off[i] * et + s[:, None] * ec + (loops[i][:, 1] * c)[:, None] * en
            P /= R
            for i in show:
                lines.append((np.vstack([P[i], P[i][:1]]), "sec", AMBER if stalled[i] else ramp(load[i])))
            lines.append((P[:, nx - 1], "le", CYAN))
            lines.append((P[:, 0], "te", VIOLET))
            lines.append((P[:, nx - 4], "mid", blend(CYAN, VIOLET, 0.5)))
            lines.append((P[:, nx + 2], "mid", blend(CYAN, VIOLET, 0.5)))
        # hub, sized from the model's own vertices near the axis
        rh = float(geom.hub_radius) / R
        try:
            v = geom.to_analysis_frame(mesh.vertices) / R
            near = v[np.hypot(v[:, 0], v[:, 1]) < max(rh * 0.9, 0.05), 2]
            zl, zh = (float(near.min()), float(near.max())) if len(near) else (-0.06, 0.06)
            zl, zh = zl - 0.5 * (zl + zh), zh - 0.5 * (zl + zh)
        except Exception:  # noqa: BLE001
            zl, zh = -0.06, 0.06
        zl, zh = max(zl, -0.25), min(zh, 0.25)
        a = np.linspace(0, 2 * math.pi, 41)
        rhub = rh * 0.85
        for z in (zl, zh):
            lines.append((np.column_stack([rhub * np.cos(a), rhub * np.sin(a), np.full_like(a, z)]), "hub", TEXT2))
        for t in np.linspace(0, 2 * math.pi, 9)[:-1]:
            lines.append((np.array([[rhub * math.cos(t), rhub * math.sin(t), zl],
                                    [rhub * math.cos(t), rhub * math.sin(t), zh]]), "hub", TEXT2))
        if pod:
            # prolate ellipsoid along the flight path at the opposite side, nose tilted by the helix angle
            er = np.array([-1.0, 0.0, 0.0])
            et = self.spin * np.array([0.0, -1.0, 0.0])
            psi = math.radians(pod["tilt_deg"])
            ex = math.cos(psi) * et + math.sin(psi) * np.array([0.0, 0.0, float(th_sign)])
            en = np.cross(ex, er)
            C = (pod["radius_mm"] * er + np.array([0.0, 0.0, th_sign * pod["z_mm"]])) / (R * 1e3)
            ax, ay, az = (pod[k] / 2 / (R * 1e3) for k in ("length_mm", "width_mm", "height_mm"))
            t = np.linspace(0, 2 * math.pi, 33)[:, None]
            lines.append((C + ax * np.cos(t) * ex + ay * np.sin(t) * er, "pod", MAGENTA))
            lines.append((C + ax * np.cos(t) * ex + az * np.sin(t) * en, "pod", MAGENTA))
            for f in (-0.6, 0.0, 0.6):
                k = math.sqrt(1 - f * f)
                lines.append((C + f * ax * ex + k * (ay * np.cos(t) * er + az * np.sin(t) * en), "pod", MAGENTA))
        a = np.linspace(0, 2 * math.pi, 49)
        for i in range(len(a) - 1):        # in pieces, so the near side can be drawn brighter than the far side
            seg = a[i:i + 2]
            lines.append((np.column_stack([np.cos(seg), np.sin(seg), np.zeros(2)]), "disk", CYAN))
        arc = np.linspace(0, 0.9, 16) * self.spin - math.pi / 2
        lines.append((np.column_stack([1.1 * np.cos(arc), 1.1 * np.sin(arc), np.zeros_like(arc)]), "spin", GREEN))
        lines.append((np.array([[0, 0, 0], [0, 0, th_sign * 0.55]]), "thrust", MAGENTA))
        lines.append((np.array([[0, 0, -0.45], [0, 0, 0.45]]), "axis", BORDER2))
        self.lines = lines
        self.P = np.vstack([ln[0] for ln in lines])
        self.thrust_tip = np.array([[0, 0, th_sign * 0.62]])
        pitch75 = geom.pitch[int(np.argmin(abs(geom.r - 0.75 * R)))]
        self.info = f"Ø {2 * R * 1e3:.0f} mm  ·  {B} blade{'s' * (B != 1)}" + (" + counterweight" if pod else "") \
            + f"  ·  pitch {pitch75 * 1e3:.0f} mm" \
            + (f"  ·  {'CCW' if self.spin * th_sign > 0 else 'CW'} seen from the front" if geom.spin_sign else "")
        self.has_load = res is not None
        self.redraw()
        self.wake()

    def _press(self, e):
        tb = self.toggle_box
        if tb and tb[0] <= e.x <= tb[2] and tb[1] <= e.y <= tb[3]:
            self._toggle_mode()
            return
        self.drag = (e.x, e.y, self.yaw, self.elev)
        self.wake()

    def _motion(self, e):
        if self.drag:
            x, y, yaw, elev = self.drag
            self.yaw = yaw - (e.x - x) * 0.01
            self.elev = min(max(elev + (e.y - y) * 0.01, -1.45), 1.45)
            self._update()

    def _release(self, _e):
        self.drag = None
        self.resume_at = time.time() + 2.5
        self.wake(self.SPIN_S + 2.5)

    def _reset(self, _e=None):
        self.yaw, self.elev, self.zoom = 0.6, math.radians(34), 1.0
        self._update()

    def _zoom(self, d):
        self.zoom = min(max(self.zoom * (1.12 if d > 0 else 1 / 1.12), 0.4), 3.0)
        self._update()

    def _tick(self):
        self.job = None
        if not self.winfo_viewable() or time.time() > self.active_until:
            return          # resting until wake()
        self.job = self.after(40, self._tick)
        if self.P is None:
            self.empty_phase += 0.04
            self._draw_empty()
        elif self.drag is None and time.time() >= self.resume_at:
            self.yaw += self.spin * 0.011        # _project rotates the model by +yaw: turn it its own way
            self._update()

    def _project(self, P):
        w, h = self.winfo_width(), self.winfo_height()
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ce, se = math.cos(self.elev), math.sin(self.elev)
        x = P[:, 0] * cy - P[:, 1] * sy
        y = P[:, 0] * sy + P[:, 1] * cy
        z = P[:, 2]
        up = y * se + z * ce
        near = z * se - y * ce
        s = min(0.42 * w, 0.36 * h / max(abs(se), 0.5)) * self.zoom
        k = 1 + 0.2 * near                       # perspective: the near side larger
        return w / 2 + x * s * k, h / 2 + px(4) - up * s * k, near

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 60:
            return
        round_rect(self, 1, 1, w - 2, h - 2, px(12), fill=CARD, outline=BORDER)
        self.create_line(px(6), px(18), px(6), px(6), px(18), px(6), fill=CYAN, width=px(2))
        self.bg_top = self.find_all()[-1] if self.find_all() else None
        self.create_text(px(18), px(20), text=spaced("3D model"), anchor="w", font=cfont(11, "bold"), fill=TEXT)
        self.items = []
        self.toggle_box = None
        solid = self.solid is not None and self.mode == "model"
        if self.solid is not None:
            # a two-way switch: the real (shaded) model or the analysed blade sections
            f = cfont(10, "bold")
            labels = (("sections", "SECTIONS"), ("model", "MODEL"))
            x0 = w - px(14) - sum(f.measure(lb) for _, lb in labels) - px(10) * len(labels)
            bb = (x0, px(8), w - px(8), px(32))
            self.create_rectangle(bb[0] + px(4), bb[1] + px(2), bb[2] - px(2), bb[3] - px(2), outline=BORDER,
                                  fill=CARD)
            x = w - px(14)
            for key, label in labels:
                t = self.create_text(x, px(20), text=label, anchor="e", font=f,
                                     fill=CYAN if self.mode == key else MUTED)
                x = self.bbox(t)[0] - px(10)
            self.toggle_box = bb
        else:
            self.create_text(w - px(16), px(20), text="BEMT blade sections", anchor="e", font=cfont(11), fill=MUTED)
        if self.P is None:
            self._draw_empty()
            return
        for pts, kind, col in ([] if solid else self.lines):
            item = dict(kind=kind, col=col, n=len(pts))
            if kind in ("le", "te"):
                item["g"] = self.create_line(0, 0, 0, 0, width=px(5), capstyle="round", fill=blend(col, CARD, 0.8))
            item["id"] = self.create_line(
                0, 0, 0, 0, fill=col, capstyle="round", joinstyle="round",
                width=px(1.6 if kind in ("le", "te", "thrust", "spin") else 1.1),
                dash=(3, 4) if kind in ("disk", "axis") else "",
                arrow="last" if kind in ("thrust", "spin") else "none", arrowshape=(px(9), px(11), px(4)))
            self.items.append(item)
        self.t_label = self.create_text(0, 0, text="T", font=cfont(12, "bold"), fill=MAGENTA)
        # footer text, shortened where the colour key on the right would overlap it
        x1, f10, f11 = w - px(16), cfont(10), cfont(11)
        room1 = (x1 - px(96) - f10.measure("dT/dr") if self.has_load else x1) - px(16)
        room2 = (x1 - px(90) if self.has_load else x1) - px(16)
        self.create_text(px(16), h - px(31), text=ellipsize(self.info, f11, room1), anchor="w", font=f11, fill=TEXT2)
        hint = next((t for t in ("drag to rotate · wheel to zoom · double-click resets",
                                 "drag · wheel · double-click resets", "drag to rotate") if f10.measure(t) <= room2), "")
        self.create_text(px(16), h - px(14), text=hint, anchor="w", font=f10, fill=MUTED)
        if self.has_load and not solid:          # colour key
            y = h - px(31)
            for i in range(40):
                xx = x1 - px(80) + i * px(80) / 40
                self.create_line(xx, y, xx + px(3), y, fill=ramp(i / 39), width=px(5))
            self.create_text(x1 - px(86), y, text="dT/dr", anchor="e", font=cfont(10), fill=MUTED)
            self.create_rectangle(x1 - px(80), h - px(18), x1 - px(72), h - px(10), fill=AMBER, outline="")
            self.create_text(x1 - px(67), h - px(14), text="stalled", anchor="w", font=cfont(10), fill=MUTED)
        self._update()

    def _update(self):
        if self.solid is not None and self.mode == "model" and self.P is not None:
            self._draw_solid()
            return
        if self.P is None or not self.items:
            return
        X, Y, near = self._project(self.P)
        o = 0
        for it in self.items:
            n = it["n"]
            xy = np.column_stack([X[o:o + n], Y[o:o + n]]).ravel().tolist()
            if it["kind"] in ("sec", "le", "te", "mid", "hub", "disk"):     # depth cue: far lines fade
                depth = float(np.clip(near[o:o + n].mean(), -1, 1))
                fade = 0.75 * (1 - depth) / 2 + (0.35 if it["kind"] == "disk" else 0.0)
                self.itemconfigure(it["id"], fill=blend(it["col"], CARD, min(fade, 0.9)))
                if it["kind"] == "disk":
                    self.itemconfigure(it["id"], dash=(3, 4) if depth < 0 else "")
            o += n
            self.coords(it["id"], xy)
            if "g" in it:
                self.coords(it["g"], xy)
        tx, ty, _ = self._project(self.thrust_tip)
        self.coords(self.t_label, float(tx[0]) + px(10), float(ty[0]))

    def _draw_solid(self, quality=False):
        """The shaded model as one image (fast render while it turns, anti-aliased once it rests)."""
        S = self.solid
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 80:
            return
        x0, y0, x1, y1 = px(8), px(34), w - px(8), h - px(42)
        W, H = max(x1 - x0, 1), max(y1 - y0, 1)
        ss = 2 if quality else 1
        V, F, Nf = S["V"], S["F"], S["N"]
        X, Y, near = self._project(V)
        X, Y = (X - x0) * ss, (Y - y0) * ss
        # face normals in view space (the rotation _project uses): facing the viewer if 'near' > 0
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ce, se = math.cos(self.elev), math.sin(self.elev)
        nx = Nf[:, 0] * cy - Nf[:, 1] * sy
        ny = Nf[:, 0] * sy + Nf[:, 1] * cy
        nn = Nf[:, 2] * se - ny * ce
        nu = ny * se + Nf[:, 2] * ce
        vis = np.nonzero(nn > -0.02)[0]
        vis = vis[np.argsort(near[F[vis]].mean(axis=1))]       # far first
        L = np.array([-0.45, 0.55, 0.70])
        L /= np.linalg.norm(L)
        inten = np.clip(0.22 + 0.78 * np.clip(nx[vis] * L[0] + nu[vis] * L[1] + nn[vis] * L[2], 0, 1), 0, 1)
        k = np.minimum((inten * (self.N_SHADE - 1)).round().astype(int), self.N_SHADE - 1)
        rgb = self.palette[S["ci"][vis], k]
        img = rasterize(X, Y, F[vis], rgb, W * ss, H * ss, _rgb(CARD))
        if ss > 1:
            img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3)).astype(np.uint8)
        self.solid_img = tk.PhotoImage(data=b"P6 %d %d 255\n" % (W, H) + img.tobytes(), format="PPM")
        if not self.find_withtag("solid"):
            self.create_image(x0, y0, anchor="nw", tags="solid")
            if self.bg_top:
                self.tag_raise("solid", self.bg_top)        # above the card, below the captions
        self.itemconfigure("solid", image=self.solid_img)
        self.coords("solid", x0, y0)
        if not quality:
            if getattr(self, "hq_job", None):
                self.after_cancel(self.hq_job)
            self.hq_job = self.after(250, self._hq)

    def _hq(self):
        self.hq_job = None
        if self.solid is not None and self.mode == "model" and self.P is not None:
            self._draw_solid(quality=True)

    def _draw_empty(self):
        self.delete("empty")
        w, h = self.winfo_width(), self.winfo_height()
        if self.P is not None or w < 60:
            return
        cx, cy, r = w / 2, h / 2, min(w, h) * 0.3
        for i in range(3):
            a0 = math.degrees(self.empty_phase) * (1 + i * 0.6) + i * 40
            rr = r * (1 - i * 0.2)
            self.create_arc(cx - rr, cy - rr * 0.42, cx + rr, cy + rr * 0.42, start=a0, extent=110 - i * 20,
                            style="arc", outline=blend((CYAN, VIOLET, MAGENTA)[i], CARD, 0.45), width=px(2), tags="empty")
        self.create_text(cx, cy, text=spaced(self.empty_text), font=cfont(12, "bold"), fill=MUTED, tags="empty")


class SectionView(tk.Canvas):
    """One blade section at its blade angle, with the relative wind (at phi) arriving."""

    def __init__(self, master, bg=BG):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, height=px(220), width=px(300))
        self.data = None
        bind_redraw(self)

    def set(self, data):
        self.data = data
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60:
            return
        round_rect(self, 1, 1, w - 2, h - 2, px(12), fill=CARD, outline=BORDER)
        self.create_line(px(6), px(18), px(6), px(6), px(18), px(6), fill=CYAN, width=px(2))
        self.create_text(px(18), px(20), text=spaced("Blade section"), anchor="w", font=cfont(11, "bold"), fill=TEXT)
        d = self.data
        if not d:
            self.create_text(w / 2, h / 2, text="Run a simulation to see a blade section", fill=MUTED, font=cfont(12))
            return
        self.create_text(w - px(16), px(20), text=d["title"], anchor="e", font=cfont(11, mono=True), fill=TEXT2)
        loop, beta, phi = d["loop"], d["beta"], d["phi"]
        s, n = loop[:, 0] - 0.25, loop[:, 1]
        cb, sb = math.cos(beta), math.sin(beta)
        X = s * cb + n * sb          # blade moving to the left, upstream is up
        Y = -s * sb + n * cb
        sc = min((w - px(150)) / 1.1, (h - px(100)) / 0.9)
        cx, cy = w / 2 + px(36), h / 2 + px(4)
        self.create_line(px(16), cy, w - px(16), cy, fill=BORDER2, dash=(4, 4))
        self.create_text(w - px(18), cy - px(9), text="plane of rotation", anchor="e", font=cfont(10), fill=MUTED)
        pts = [c for x, y in zip(X, Y) for c in (cx + x * sc, cy - y * sc)]
        self.create_polygon(pts, fill=blend(CYAN, CARD, 0.86), outline="")
        glow_line(self, pts, CYAN, width=1.8)
        le = (cx - 0.25 * cb * sc, cy - 0.25 * sb * sc)
        te = (cx + 0.75 * cb * sc, cy + 0.75 * sb * sc)
        self.create_line(le[0], le[1], te[0], te[1], fill=MUTED, dash=(2, 3))
        if phi is not None:
            cp, sp = math.cos(phi), math.sin(phi)
            for k in (-1, 0, 1):
                ox, oy = le[0] - px(14) - k * sp * px(16), le[1] + k * cp * px(16)
                L = px(64)
                self.create_line(ox - cp * L, oy - sp * L, ox, oy, fill=MAGENTA if k == 0 else blend(MAGENTA, CARD, 0.5),
                                 width=px(1.6), arrow="last", arrowshape=(px(8), px(10), px(3)))
        f = cfont(11, mono=True)
        x, y = px(16), h - px(16)
        for label, col in d["stats"]:
            t = self.create_text(x, y, text=label, anchor="w", font=f, fill=col)
            x = self.bbox(t)[2] + px(16)


# --------------------------------------------------------------------------- the app

class InputError(ValueError):
    def __init__(self, key, msg):
        super().__init__(msg)
        self.key = key


class App:
    TABS = (("dash", "Dashboard"), ("geom", "Geometry"), ("report", "Report"), ("cfd", "3D CFD"),
            ("gen", "Prop Generator"))
    STEPS = (("blockMesh", "base mesh"), ("features", "features"), ("decomposePar", "decompose"),
             ("snappyHexMesh", "snappy"), ("topoSet", "MRF zone"), ("checkMesh", "check"), ("solver", "solve"),
             ("reconstructPar", "merge"))

    def __init__(self, root: ctk.CTk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.outcome = None
        self.last_key = None          # argv (without report-only flags) of the shown outcome
        self.sim_busy = False
        self.sim_t0 = 0.0
        self.entries = {}
        self.err_key = None
        self.tab = "dash"
        self.stale, self.stale_job, self.result_label = False, None, ""

        self.v = {k: tk.StringVar() for k in (
            "model", "units", "axis", "blades", "hub", "mode", "tip", "rpm", "power",
            "kv", "cells", "cellv", "voltage", "rm", "i0", "esc", "qmax", "drive",
            "airspeed", "rho", "polar", "clmax", "clalpha", "cd0", "aero", "nf_model", "ncrit",
            "cfd_res", "cfd_cores", "cfd_iters", "cfd_air")}
        defaults = dict(units="mm", axis="z", mode="tip", tip="150", rpm="8000", power="400",
                        cellv="3.7", esc="95", drive="85", airspeed="0", rho="1.225",
                        clmax="1.2", clalpha="5.7", aero="analytic", nf_model="large", ncrit="9",
                        cfd_res="coarse", cfd_cores="Auto", cfd_air="Auto")
        for k, val in defaults.items():
            self.v[k].set(val)
        self.test_prop = tk.BooleanVar(value=False)
        self.show_geom = tk.BooleanVar(value=False)
        self.sweep = tk.BooleanVar(value=True)
        for var in (*self.v.values(), self.test_prop):
            var.trace_add("write", self._inputs_changed)

        # 3D CFD state; the worker writes cfd_step / cfd_iter / cfd_info as plain attributes and
        # appends log lines to cfd_lines, which the pump drains
        self.cfd_install = None
        self.cfd_busy = self.cfd_running = self.cfd_done_ok = False
        self.cfd_stop = threading.Event()
        self.cfd_lines: collections.deque = collections.deque()
        self.cfd_case = None          # (case_dir, inputs key) of the case written in this session
        self.cfd_info = {}
        self.cfd_bemt = None          # (thrust, torque, power) of the BEMT run the case was made from
        self.cfd_out = None           # ... that run's Outcome, and its one-line description
        self.cfd_desc = None
        self.cfd_shown = False        # a CFD result is in the comparison table
        self.foam_hint = ""           # install hint last written to the log
        self.cfd_step, self.cfd_iter, self.cfd_iter_t0, self.cfd_t0 = "", 0, None, 0.0
        self.bar_mode = "determinate"
        self.blink, self.step_drawn = 0, None
        self.closing = False

        root.configure(fg_color=BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self.header = AppHeader(root)
        self.header.grid(row=0, column=0, sticky="ew")
        body = tk.Frame(root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_left(body)
        sep = tk.Canvas(body, width=px(2), bg=BG, highlightthickness=0, bd=0)
        sep.grid(row=0, column=1, sticky="ns")
        sep.bind("<Configure>", lambda _e: self._draw_sep(sep))
        self._build_right(body)
        self.status = StatusBar(root)
        self.status.grid(row=2, column=0, sticky="ew")

        self.header.set_chip("aero", "AERO · ANALYTIC", CYAN)
        self.header.set_chip("foam", "OPENFOAM · CHECKING", MUTED)
        self._mode_changed()
        self._aero_changed()
        self._source_changed()
        self.root.bind("<Return>", self._on_return)
        # after every scrollable frame exists (each adds a global wheel handler, replaced here)
        self.scroller = SmoothScroll(self.root)
        for sc in (self.left_sc, getattr(self, "gen_sc", None)):
            if sc is not None:
                self.scroller.add(sc)
        # the window moved or was resized: no tooltip left floating where its widget used to be
        self.root.bind("<Configure>", lambda e: e.widget is self.root and Tooltip.hide_all(), add="+")
        self.root.bind("<Unmap>", lambda e: e.widget is self.root and Tooltip.hide_all(), add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._pump()
        self.root.after(700, self._cfd_detect)
        self._report_intro()

    # ================================================================== layout
    def _draw_sep(self, cv):
        cv.delete("all")
        h = cv.winfo_height()
        for i in range(40):
            t = i / 39
            cv.create_line(0, h * i / 40, 0, h * (i + 1) / 40 + 1, width=2,
                           fill=blend(blend(CYAN, VIOLET, t), BG, 0.55 + 0.4 * t))

    def _card(self, parent, title, index="", accent=CYAN):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.header = CardHeader(card, title, index, accent=accent)
        card.header.pack(fill="x", padx=px(3), pady=(px(3), 0))
        body = tk.Frame(card, bg=CARD, bd=0, highlightthickness=0)   # (a plain frame: one native window)
        body.pack(fill="both", expand=True, padx=px(12), pady=(px(2), px(12)))
        body.columnconfigure(0, weight=1)
        return card, body

    def _entry(self, parent, key, width=96, mono=True, justify="right"):
        e = ctk.CTkEntry(parent, textvariable=self.v[key], width=width, height=30, font=font(13, mono=mono),
                         fg_color=INPUT, border_color=BORDER, text_color=TEXT, border_width=1, corner_radius=7,
                         justify=justify)
        e.bind("<FocusIn>", lambda _e: e.configure(border_color=RED if self.err_key == key else CYAN), add="+")
        e.bind("<FocusOut>", lambda _e: e.configure(border_color=RED if self.err_key == key else BORDER), add="+")
        self.v[key].trace_add("write", lambda *_: self._field_edited(key))
        self.entries[key] = e
        return e

    def _field_edited(self, key):
        """Typing in the flagged field takes the error mark off it."""
        if key == self.err_key:
            self.err_key = None
            e = self.entries[key]
            e.configure(border_color=CYAN if self.root.focus_get() is getattr(e, "_entry", None) else BORDER)

    # Form captions are plain Tk labels: one native window each instead of a customtkinter label's
    # three (frame, canvas, label). The input panel has dozens of them, and every native window is
    # repainted on each scroll step, so this makes scrolling much lighter.
    _BLANK = None

    @staticmethod
    def _bg_of(w):
        """The colour a widget's children are drawn on (customtkinter frames may be transparent)."""
        while w is not None:
            try:
                fg = w.cget("fg_color")
            except (ValueError, tk.TclError, AttributeError):
                fg = None
            if fg not in (None, "transparent"):
                return fg[1] if isinstance(fg, (tuple, list)) else fg
            if fg is None:
                try:
                    return w.cget("bg")
                except tk.TclError:
                    pass
            w = w.master
        return CARD

    def _plain_label(self, parent, text, size, color, width=0):
        if App._BLANK is None:
            App._BLANK = tk.PhotoImage(width=1, height=1)
        # with an image, width and height are in pixels (the same box a customtkinter label has)
        return tk.Label(parent, text=text, font=cfont(size), fg=color, bg=self._bg_of(parent), anchor="w",
                        image=App._BLANK, compound="left", width=width or 0, height=px(28), bd=0, padx=0, pady=0,
                        highlightthickness=0)

    def _label(self, parent, text, tip=""):
        lab = self._plain_label(parent, text, 13, TEXT2)
        if tip:
            Tooltip(lab, tip)
        return lab

    def _unit(self, parent, text):
        # at least 62 px wide (so the entries line up), wider for a longer text
        return self._plain_label(parent, text, 11, MUTED, width=max(px(62), cfont(11).measure(text) + px(4)))

    def _field(self, parent, row, label, key, unit="", tip=""):
        self._label(parent, label, tip).grid(row=row, column=0, sticky="w", pady=px(3))
        e = self._entry(parent, key)
        e.grid(row=row, column=1, sticky="e", pady=px(3))
        self._unit(parent, unit).grid(row=row, column=2, sticky="w", padx=(px(8), 0))
        return e

    def _seg(self, parent, key, values, command=None):
        return ctk.CTkSegmentedButton(
            parent, values=values, variable=self.v[key], command=command, height=30, font=font(12, "bold"),
            fg_color=INPUT, selected_color=CYAN_D, selected_hover_color="#0ba7c7", unselected_color=INPUT,
            unselected_hover_color=CARD2, text_color=TEXT, text_color_disabled=MUTED, corner_radius=7)

    def _switch(self, parent, text, var, command=None):
        return ctk.CTkSwitch(parent, text=text, variable=var, onvalue=True, offvalue=False, command=command,
                             font=font(13), text_color=TEXT2, progress_color=CYAN_D, button_color=TEXT,
                             button_hover_color=CYAN, fg_color=BORDER, switch_width=38, switch_height=18)

    def _option(self, parent, key, values, width=96):
        return ctk.CTkOptionMenu(parent, values=values, variable=self.v[key], width=width, height=30, font=font(12),
                                 dropdown_font=font(12), fg_color=INPUT, button_color=BORDER, button_hover_color=BORDER2,
                                 dropdown_fg_color=CARD2, dropdown_hover_color=BORDER2, dropdown_text_color=TEXT,
                                 text_color=TEXT, corner_radius=7)

    def _button(self, parent, text, command, accent=CYAN, width=80, height=34, primary=False):
        b = ctk.CTkButton(parent, text=text, command=command, width=width, height=height, font=font(12, "bold"),
                          corner_radius=8, border_width=1, border_color=blend(accent, BG, 0.45),
                          fg_color=blend(accent, BG, 0.8 if primary else 0.9), hover_color=blend(accent, BG, 0.7),
                          text_color=TEXT, text_color_disabled=blend(MUTED, CARD, 0.25))
        b.colors = (blend(accent, BG, 0.8 if primary else 0.9), blend(accent, BG, 0.45))
        return b

    @staticmethod
    def _enable(btn, on):
        """Enable or disable a _button, with a flat grey look when disabled (not just greyed text)."""
        fill, border = btn.colors if on else (CARD, BORDER)
        if btn.cget("state") != ("normal" if on else "disabled") or btn.cget("fg_color") != fill:
            btn.configure(state="normal" if on else "disabled", fg_color=fill, border_color=border)

    def _build_left(self, body):
        left = tk.Frame(body, bg=PANEL, width=px(388))
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.left_sc = sc = ctk.CTkScrollableFrame(left, fg_color=PANEL, corner_radius=0, scrollbar_button_color=BORDER,
                                                   scrollbar_button_hover_color=BORDER2)
        sc.grid(row=0, column=0, sticky="nsew", padx=(px(8), 0), pady=(px(10), 0))
        pad = dict(fill="x", padx=(px(4), px(8)), pady=(0, px(10)))

        # --- model
        card, b = self._card(sc, "Propeller model", "01")
        card.pack(**pad)
        row = ctk.CTkFrame(b, fg_color="transparent")
        row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, px(6)))
        row.columnconfigure(0, weight=1)
        self._entry(row, "model", width=200, mono=False, justify="left").grid(row=0, column=0, sticky="ew")
        self._button(row, "Browse…", self._browse_model, width=78, height=30).grid(row=0, column=1, padx=(px(6), 0))
        self._switch(b, "Use built-in test prop (10 × 4.5 in)", self.test_prop, self._source_changed).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(px(2), px(8)))
        self._label(b, "Model units", "Units the model file was drawn in.").grid(row=2, column=0, sticky="w", pady=px(3))
        self.units_seg = self._seg(b, "units", ["mm", "cm", "m", "in"])
        self.units_seg.grid(row=2, column=1, columnspan=2, sticky="ew", pady=px(3))
        self._label(b, "Spin axis", "Model axis the propeller spins about.").grid(row=3, column=0, sticky="w", pady=px(3))
        self.axis_seg = self._seg(b, "axis", ["x", "y", "z"])
        self.axis_seg.grid(row=3, column=1, columnspan=2, sticky="ew", pady=px(3))
        self._field(b, 4, "Blades", "blades", "auto", "Number of blades. Blank = detected from the model.")
        self._field(b, 5, "Hub cut-off r/R", "hub", "auto",
                    "Where the blade analysis starts, as a fraction of the tip radius (the hub/spinner is "
                    "excluded). Blank = detected.")

        # --- operating point
        card, b = self._card(sc, "Operating point", "02")
        card.pack(**pad)
        self.mode_entries = {}
        for i, (key, label, field, unit, tip) in enumerate((
                ("tip", "Tip speed", "tip", "m/s", "Set the blade tip speed."),
                ("rpm", "RPM", "rpm", "rpm", "Set the rotational speed."),
                ("motor", "Motor power", "power", "W rated",
                 "Motor rated (maximum) input power; needs the motor KV. With a battery voltage this is the "
                 "max-thrust point; without one PropSim solves for the voltage."))):
            rb = ctk.CTkRadioButton(b, text=label, variable=self.v["mode"], value=key, command=self._mode_changed,
                                    font=font(13), text_color=TEXT2, fg_color=CYAN, hover_color=CYAN_D,
                                    border_color=BORDER2, radiobutton_width=17, radiobutton_height=17,
                                    border_width_unchecked=2, border_width_checked=5)
            rb.grid(row=i, column=0, sticky="w", pady=px(3))
            Tooltip(rb, tip)
            e = self._entry(b, field)
            e.grid(row=i, column=1, sticky="e", pady=px(3))
            self._unit(b, unit).grid(row=i, column=2, sticky="w", padx=(px(8), 0))
            self.mode_entries[key] = e

        # --- aerodynamics
        card, b = self._card(sc, "Aerodynamics", "03", accent=VIOLET)
        card.pack(**pad)
        OptionCard(b, self.v["aero"], "analytic", "Analytic", "Linear lift with a stall model and "
                   "Reynolds-based drag.", "INSTANT", self._aero_changed).grid(row=0, column=0, columnspan=3, sticky="ew")
        OptionCard(b, self.v["aero"], "neuralfoil", "2D section CFD · NeuralFoil",
                   "Viscous 2D analysis of every measured blade section at its own Reynolds number. "
                   "Takes a few seconds.", "2D", self._aero_changed).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(px(6), 0))
        self.nf_row = ctk.CTkFrame(b, fg_color="transparent")
        self.nf_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(px(8), 0))
        self.nf_row.columnconfigure(0, weight=1)
        self._label(self.nf_row, "Network size", "NeuralFoil network: larger is slightly more accurate and "
                    "slower.").grid(row=0, column=0, sticky="w")
        self._option(self.nf_row, "nf_model", self._nf_sizes(), width=130).grid(row=0, column=1, sticky="e")
        self._label(self.nf_row, "Ncrit (transition)", "Boundary-layer transition setting as in XFOIL: 9 = calm air "
                    "(typical), 4-6 = turbulent air. Matters most for small, slow props (low Reynolds number)."
                    ).grid(row=1, column=0, sticky="w", pady=(px(6), 0))
        self._entry(self.nf_row, "ncrit", width=130).grid(row=1, column=1, sticky="e", pady=(px(6), 0))
        ctk.CTkLabel(b, text="This is 2D section analysis feeding the BEMT, not full 3D CFD. For a 3D "
                             "simulation of the whole propeller use the 3D CFD tab (OpenFOAM).",
                     font=font(11), text_color=MUTED, anchor="w", justify="left", wraplength=300).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(px(8), 0))

        # --- flight and airfoil
        card, b = self._card(sc, "Flight and airfoil", "04")
        card.pack(**pad)
        self._field(b, 0, "Airspeed", "airspeed", "m/s", "Forward (axial) airspeed; 0 = static thrust (hover).")
        self._field(b, 1, "Air density", "rho", "kg/m³", "1.225 at sea level, ~1.11 at 1000 m, ~1.0 at 2000 m.")
        self._field(b, 2, "Cl max", "clmax", "", "Stall lift coefficient (analytic model).")
        self._field(b, 3, "Lift slope", "clalpha", "1/rad", "Section lift-curve slope (analytic model).")
        self._field(b, 4, "Cd0", "cd0", "auto", "Profile drag. Blank = from thickness and Reynolds number.")
        self._label(b, "Polar CSV", "Optional CSV of alpha_deg, cl, cd (e.g. exported from XFOIL), used for all "
                    "stations.").grid(row=5, column=0, sticky="w", pady=px(3))
        pr = ctk.CTkFrame(b, fg_color="transparent")
        pr.grid(row=5, column=1, columnspan=2, sticky="ew", pady=px(3))
        pr.columnconfigure(0, weight=1)
        self._entry(pr, "polar", width=110, mono=False, justify="left").grid(row=0, column=0, sticky="ew")
        self._button(pr, "…", self._browse_polar, width=34, height=30).grid(row=0, column=1, padx=(px(6), 0))

        # --- motor and battery
        card, b = self._card(sc, "Motor and battery", "05", accent=MAGENTA)
        card.pack(**pad)
        self._field(b, 0, "KV", "kv", "rpm/V", "Motor KV. Needed for motor mode; in the other modes it adds "
                    "current, motor voltage and throttle.")
        self._field(b, 1, "LiPo cells", "cells", "S", "Battery cell count (or set the battery voltage).")
        self._field(b, 2, "Volts per cell", "cellv", "V", "3.7 V nominal, 4.2 V full.")
        self._field(b, 3, "Battery voltage", "voltage", "V", "Overrides the cell count.")
        self._field(b, 4, "Rm", "rm", "Ω · auto", "Motor winding resistance. Blank = estimated.")
        self._field(b, 5, "I0", "i0", "A · auto", "Motor no-load current. Blank = estimated.")
        self._field(b, 6, "ESC efficiency", "esc", "%")
        self._field(b, 7, "Max torque", "qmax", "N·m · none",
                    "The motor's maximum torque (datasheet). The prop is never run harder than this: the "
                    "simulation throttles back and the Prop Generator designs within it. Blank = no limit.")
        self._field(b, 8, "Motor+ESC eff.", "drive", "% no KV",
                    "Overall motor + ESC efficiency used for electrical power when no KV is given.")

        # --- report options
        card, b = self._card(sc, "Report options", "06")
        card.pack(**pad)
        self._switch(b, "RPM sweep table", self.sweep).grid(row=0, column=0, sticky="w", pady=px(3))
        self._switch(b, "Blade geometry table", self.show_geom).grid(row=1, column=0, sticky="w", pady=px(3))

        # --- actions (always visible)
        foot = tk.Frame(left, bg=PANEL)
        foot.grid(row=1, column=0, sticky="ew", padx=(px(12), px(14)), pady=(px(8), px(12)))
        foot.columnconfigure((0, 1), weight=1, uniform="foot")
        self.run_btn = GlowButton(foot, "Run simulation", self._run, hint="ENTER ⏎")
        self.run_btn.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._button(foot, "Save results…", self._save_text).grid(row=1, column=0, sticky="ew",
                                                                       padx=(0, px(4)), pady=(px(8), 0))
        self._button(foot, "Export station CSV…", self._save_csv).grid(row=1, column=1, sticky="ew",
                                                                            padx=(px(4), 0), pady=(px(8), 0))

    def _build_right(self, body):
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=2, sticky="nsew", padx=(px(16), px(18)), pady=(px(6), px(8)))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self.banner = ctk.CTkFrame(right, fg_color="#1a0a17", border_color=RED, border_width=1, corner_radius=10)
        self.banner.columnconfigure(1, weight=1)
        self.banner_title = ctk.CTkLabel(self.banner, text="ERROR", font=font(12, "bold"), text_color=RED)
        self.banner_title.grid(row=0, column=0, sticky="nw", padx=(px(14), px(10)), pady=px(8))
        self.banner_msg = ctk.CTkLabel(self.banner, text="", font=font(13), text_color=TEXT, anchor="w",
                                       justify="left", wraplength=760)
        self.banner_msg.grid(row=0, column=1, sticky="w", pady=px(8))
        ctk.CTkButton(self.banner, text="✕", width=28, height=28, fg_color="transparent", hover_color=CARD2,
                      text_color=TEXT2, font=font(13), command=self._hide_banner).grid(
            row=0, column=2, padx=px(8), pady=px(5), sticky="ne")
        self.tabbar = TabBar(right, self.TABS, self._show_tab)
        self.tabbar.grid(row=1, column=0, sticky="ew", pady=(0, px(8)))
        holder = tk.Frame(right, bg=BG)
        holder.grid(row=2, column=0, sticky="nsew")
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        self.pages = {}
        for key, _ in self.TABS:
            f = tk.Frame(holder, bg=BG)
            f.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = f
        self._build_dashboard(self.pages["dash"])
        self._build_geometry(self.pages["geom"])
        self._build_report(self.pages["report"])
        self._build_cfd(self.pages["cfd"])
        self._build_gen(self.pages["gen"])
        self._show_tab("dash")

    def _show_tab(self, key):
        Tooltip.hide_all()
        for k, f in self.pages.items():
            if k == key:
                f.grid()
            else:
                f.grid_remove()
        self.tabbar.select(key)
        self.tab = key
        if key == "cfd":
            self._cfd_update_op()
        elif key == "dash":
            self.view3d.wake()
        elif key == "gen":
            self.gen_view.wake()

    def _build_dashboard(self, page):
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)
        self.strip = ModelStrip(page)
        self.strip.grid(row=0, column=0, sticky="ew", pady=(0, px(6)))
        tiles = tk.Frame(page, bg=BG)
        tiles.grid(row=1, column=0, sticky="ew")
        for c in range(5):
            tiles.columnconfigure(c, weight=1, uniform="tiles")
        spec = (("thrust", "Thrust", 0, 0, 2, CYAN, True), ("torque", "Shaft torque", 0, 2, 1, VIOLET, False),
                ("power", "Shaft power", 0, 3, 1, MAGENTA, False), ("elec", "Electrical power", 0, 4, 1, MAGENTA, False),
                ("rpm", "Speed", 1, 0, 1, CYAN, False), ("tip", "Tip speed", 1, 1, 1, CYAN, False),
                ("fom", "Figure of merit", 1, 2, 1, GREEN, False), ("current", "Current", 1, 3, 1, VIOLET, False),
                ("throttle", "Throttle", 1, 4, 1, GREEN, False))     # amber is kept for warnings
        self.tiles = {}
        for key, label, r, c, span, col, hero in spec:
            t = KpiTile(tiles, label, col, hero)
            t.grid(row=r, column=c, columnspan=span, sticky="ew", padx=px(4), pady=px(4))
            self.tiles[key] = t
        lower = tk.Frame(page, bg=BG)
        lower.grid(row=2, column=0, sticky="nsew", pady=(px(6), 0))
        lower.columnconfigure(0, weight=9, uniform="low")
        lower.columnconfigure(1, weight=11, uniform="low")
        lower.rowconfigure((0, 1), weight=1, uniform="lowr")
        self.view3d = PropView(lower)
        self.view3d.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(px(4), px(6)), pady=px(4))
        self.sweep_chart = LineChart(lower, "RPM sweep", "speed [rpm]", ("thrust [N]", "T [N]"), ("power [W]", "P [W]"),
                                     xfmt=lambda v, s: f"{v:,.0f}", apart=True)
        self.sweep_chart.grid(row=0, column=1, sticky="nsew", padx=(px(6), px(4)), pady=px(4))
        self.radial_chart = LineChart(lower, "Radial distribution", "r / R", ("dT/dr [N/m]", "[N/m]"),
                                      ("angle of attack [°]", "α [°]"))
        self.radial_chart.grid(row=1, column=1, sticky="nsew", padx=(px(6), px(4)), pady=px(4))

    def _build_geometry(self, page):
        page.columnconfigure((0, 1), weight=1, uniform="geom")
        page.rowconfigure(0, weight=3)
        page.rowconfigure(1, weight=2)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("PS.Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT2,
                        rowheight=px(25), borderwidth=0, font=cfont(12, mono=True))
        style.configure("PS.Treeview.Heading", background=CARD2, foreground=CYAN, relief="flat",
                        font=cfont(11, "bold"), borderwidth=0, padding=(px(4), px(7)))
        style.map("PS.Treeview", background=[("selected", "#123a52")], foreground=[("selected", TEXT)])
        style.map("PS.Treeview.Heading", background=[("active", BORDER)])
        style.layout("PS.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        frame = ctk.CTkFrame(page, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=px(4), pady=px(4))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        self.geom_hdr = CardHeader(frame, "Stations", right="run a simulation to fill the table")
        self.geom_hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=px(3), pady=(px(3), 0))
        cols = (("st", "#", 26), ("rR", "r/R", 46), ("r", "r mm", 50), ("c", "chord mm", 64), ("b", "β °", 44),
                ("p", "pitch mm", 62), ("tc", "t/c", 44), ("cam", "camber", 56), ("a", "α °", 44),
                ("cl", "Cl", 46), ("cd", "Cd", 54), ("re", "Re", 48), ("m", "Mach", 46), ("dT", "dT/dr N/m", 70))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings", style="PS.Treeview",
                                 selectmode="browse")
        for k, label, wdt in cols:
            self.tree.heading(k, text=label, anchor="e")
            self.tree.column(k, width=px(wdt), minwidth=px(30), anchor="e", stretch=True)
        self.tree.tag_configure("odd", background=blend(CARD, CARD2, 0.55))
        self.tree.tag_configure("stall", foreground=AMBER)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(px(10), 0), pady=(px(4), px(10)))
        ys = ctk.CTkScrollbar(frame, command=self.tree.yview, button_color=BORDER, button_hover_color=BORDER2)
        ys.grid(row=1, column=1, sticky="ns", padx=(0, px(4)), pady=(px(4), px(10)))
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._geom_select())
        self.section_view = SectionView(page)
        self.section_view.grid(row=1, column=0, sticky="nsew", padx=(px(4), px(6)), pady=px(4))
        self.dist_chart = LineChart(page, "Chord and blade angle", "r / R", ("chord [mm]", "c [mm]"),
                                    ("blade angle [°]", "β [°]"))
        self.dist_chart.grid(row=1, column=1, sticky="nsew", padx=(px(6), px(4)), pady=px(4))

    def _build_report(self, page):
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        self.report = ctk.CTkTextbox(page, font=font(13, mono=True), wrap="none", fg_color=CARD, text_color=TEXT2,
                                     border_width=1, border_color=BORDER, corner_radius=12,
                                     scrollbar_button_color=BORDER, scrollbar_button_hover_color=BORDER2)
        self.report.grid(row=0, column=0, sticky="nsew", padx=px(4), pady=px(4))
        for tag, col in (("head", CYAN), ("warn", AMBER), ("note", MUTED), ("err", RED), ("num", TEXT)):
            self.report.tag_config(tag, foreground=col)

    def _set_report(self, text, error=False):
        r = self.report
        r.configure(state="normal")
        r.delete("1.0", "end")
        for line in text.splitlines():
            s = line.strip()
            if error:
                tag = "err"
            elif s.startswith(("Model:", "Operating point", "Electrical", "rpm ")) or line.startswith("  r/R"):
                tag = "head"
            elif s.startswith("warning:") or "<--" in s:
                tag = "warn"
            elif s.startswith(("note:", "(")):
                tag = "note"
            else:
                tag = "num"
            r.insert("end", "  " + line + "\n", tag)
        r.configure(state="disabled")

    def _report_intro(self):
        self._set_report(
            "\nPick an STL/STEP model (or switch on the built-in test prop), choose a mode and press RUN.\n"
            "STEP files take up to a minute the first time; after that they are cached.\n\n"
            "Results are estimates from blade element momentum theory (BEMT). With the analytic airfoil\n"
            "model they are typically within about 15-25 % of a thrust stand. '2D section CFD - NeuralFoil'\n"
            "analyses each measured blade section with a viscous 2D model; it is not 3D CFD. For a full 3D\n"
            "simulation of the propeller use the 3D CFD tab (OpenFOAM).\n")

    def _build_cfd(self, page):
        page.columnconfigure((0, 1), weight=1, uniform="cfd")
        page.rowconfigure(1, weight=2)      # spare height goes to the convergence chart and the log
        page.rowconfigure(2, weight=3)
        # OpenFOAM status: one line (a plain card, no caption: the tab already says 3D CFD, and the
        # convergence chart needs the height at 1280 x 800)
        card = ctk.CTkFrame(page, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, columnspan=2, sticky="ew", padx=px(4), pady=px(4))
        b = ctk.CTkFrame(card, fg_color="transparent")
        b.pack(fill="both", expand=True, padx=px(14), pady=px(9))
        b.columnconfigure(1, weight=1)
        self.foam_dot = tk.Canvas(b, width=px(14), height=px(14), bg=CARD, highlightthickness=0, bd=0)
        self.foam_dot.grid(row=0, column=0, padx=(px(2), px(10)), sticky="w")
        self.foam_text = ctk.CTkLabel(b, text="Checking for OpenFOAM…", font=font(13, "bold"), text_color=TEXT,
                                      anchor="w", justify="left", wraplength=560)
        self.foam_text.grid(row=0, column=1, sticky="w")
        self.foam_steps_btn = self._button(b, "Details ▾", self._toggle_foam_steps, accent=AMBER, width=118)
        self.foam_btn = self._button(b, "Detect again", lambda: self._cfd_detect(refresh=True), width=110)
        self.foam_btn.grid(row=0, column=3, sticky="e")
        self._set_foam_dot(MUTED)

        # case setup
        card, b = self._card(page, "Case setup", "A")
        card.header.set_right("RANS k-ω SST · MRF rotor")
        card.grid(row=1, column=0, sticky="nsew", padx=px(4), pady=px(4))
        b.columnconfigure(1, weight=1)
        self.cfd_op = ctk.CTkLabel(b, text="", font=font(12), text_color=TEXT2, anchor="w", justify="left",
                                   wraplength=360)
        self.cfd_op.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, px(8)))
        self._label(b, "Resolution", "Mesh resolution preset (cells on the 10x4.5 in test prop). Coarse is a "
                    "quick check (minutes); medium and fine take hours. The case is steady RANS k-ω SST with the "
                    "rotor in a rotating MRF zone, meshed with snappyHexMesh.").grid(
            row=1, column=0, sticky="w", pady=px(3))
        self._seg(b, "cfd_res", ["coarse", "medium", "fine"], command=lambda _v: self._cfd_res_changed()).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=px(3), padx=(px(10), 0))
        self.cfd_res_info = ctk.CTkLabel(b, text="", font=font(11), text_color=MUTED, anchor="e", height=16)
        self.cfd_res_info.grid(row=2, column=1, columnspan=2, sticky="e")
        self._label(b, "CPU cores", "Parallel OpenFOAM processes. Auto = as many of this PC's cores as the mesh can use well (any core count works; if parallel can't start it runs on 1 core).").grid(
            row=3, column=0, sticky="w", pady=px(3))
        self._option(b, "cfd_cores", ["Auto"] + [str(i) for i in range(1, (os.cpu_count() or 1) + 1)]).grid(
            row=3, column=2, sticky="e", pady=px(3))
        self._label(b, "Iterations", "Solver iterations. Blank = the preset for the resolution.").grid(
            row=4, column=0, sticky="w", pady=px(3))
        self._unit(b, "blank = preset").grid(row=4, column=1, sticky="e", padx=px(8))
        self._entry(b, "cfd_iters").grid(row=4, column=2, sticky="e", pady=px(3))
        self._label(b, "Air", "Air model of the 3D CFD. Compressible (rhoSimpleFoam): the density changes with "
                    "pressure, as in real air - needed for fast props, where incompressible CFD underestimates the "
                    "loads (by about 1/√(1-M²) at the blade's Mach number M). Incompressible (simpleFoam): faster "
                    "and a little more robust, fine at low tip speeds. Auto: compressible above tip Mach "
                    f"{getattr(cfd3d, 'COMPRESSIBLE_AUTO_MACH', 0.5):g}.").grid(row=5, column=0, sticky="w",
                                                                              pady=px(3))
        self._seg(b, "cfd_air", ["Auto", "Incompressible", "Compressible"]).grid(
            row=5, column=1, columnspan=2, sticky="ew", pady=px(3), padx=(px(10), 0))
        btns = ctk.CTkFrame(b, fg_color="transparent")
        btns.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(px(12), 0))
        for c, wt in ((0, 4), (1, 5), (2, 3)):    # "Run OpenFOAM" needs the most room at the minimum window size
            btns.columnconfigure(c, weight=wt, uniform="cfdb")
        self.cfd_write_btn = self._button(btns, "Write case…", lambda: self._cfd_start(run=False))
        self.cfd_write_btn.grid(row=0, column=0, sticky="ew", padx=(0, px(4)))
        self.cfd_run_btn = self._button(btns, "▶  Run OpenFOAM", lambda: self._cfd_start(run=True), accent=GREEN,
                                        primary=True)
        self.cfd_run_btn.grid(row=0, column=1, sticky="ew", padx=px(4))
        self.cfd_stop_btn = self._button(btns, "■  Stop", self._cfd_stop_click, accent=RED)
        self.cfd_stop_btn.grid(row=0, column=2, sticky="ew", padx=(px(4), 0))
        self.cfd_load_btn = self._button(btns, "Load results…", self._cfd_load)
        self.cfd_load_btn.grid(row=1, column=0, sticky="ew", padx=(0, px(4)), pady=(px(8), 0))
        self.cfd_open_btn = self._button(btns, "Open folder", self._cfd_open_folder)
        self.cfd_open_btn.grid(row=1, column=1, sticky="ew", padx=px(4), pady=(px(8), 0))

        # results
        card, b = self._card(page, "CFD vs BEMT", "B", accent=VIOLET)
        card.grid(row=1, column=1, sticky="nsew", padx=px(4), pady=px(4))
        self.cfd_res_hdr = card.header
        tbl = ctk.CTkFrame(b, fg_color="transparent")
        tbl.grid(row=0, column=0, sticky="ew")
        tbl.columnconfigure((1, 2, 3), weight=1, uniform="cmp")
        for c, t in enumerate(("", "BEMT", "3D CFD", "Δ")):
            hl = ctk.CTkLabel(tbl, text=spaced(t), font=font(10, "bold"), text_color=MUTED, anchor="e", height=18)
            hl.grid(row=0, column=c, sticky="e" if c else "w", padx=(0, px(4)))
            if c == 2:
                self.cfd_col_hdr = hl
                Tooltip(hl, "The CFD forces on the blades alone (outside the BEMT's hub cut-off), like the BEMT; "
                            "the hub's own share is listed below the table.")
        self.cfd_cells = {}
        for r, (key, label, unit) in enumerate((("T", "Thrust", "N"), ("Q", "Torque", "N·m"), ("P", "Power", "W")), 1):
            ctk.CTkLabel(tbl, text=f"{label} [{unit}]", font=font(13), text_color=TEXT2, anchor="w").grid(
                row=r, column=0, sticky="w", pady=px(1))
            cells = []
            for c, col in ((1, TEXT), (2, CYAN), (3, TEXT2)):
                lab = ctk.CTkLabel(tbl, text="—", font=font(15, "bold", mono=True), text_color=col, anchor="e")
                lab.grid(row=r, column=c, sticky="e", padx=(0, px(4)), pady=px(1))
                cells.append(lab)
            self.cfd_cells[key] = cells
        self.cfd_msg = ctk.CTkLabel(b, text="No CFD result yet.", font=font(11), text_color=MUTED, anchor="w",
                                    justify="left", wraplength=380)
        self.cfd_msg.grid(row=1, column=0, sticky="w", pady=(px(4), 0))
        self.hist_chart = LineChart(b, "Convergence", "iteration", ("Δ vs BEMT [%]", "Δ [%]"), zero=False, compact=True,
                                    apart=True, empty="Force history appears while the solver runs", bg=CARD)
        self.hist_chart.configure(height=px(120))
        self.hist_chart.grid(row=2, column=0, sticky="nsew", pady=(px(6), 0))
        b.rowconfigure(2, weight=1)

        # progress + log
        card, b = self._card(page, "Run", "C")
        card.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=px(4), pady=px(4))
        b.rowconfigure(2, weight=1)
        self.stepper = tk.Canvas(b, height=px(44), bg=CARD, highlightthickness=0, bd=0)
        self.stepper.grid(row=0, column=0, sticky="ew")
        self.stepper.bind("<Configure>", lambda _e: self._draw_stepper())
        self.cfd_prog_row = pr = ctk.CTkFrame(b, fg_color="transparent")
        pr.grid(row=1, column=0, sticky="ew", pady=(px(2), px(6)))
        pr.columnconfigure(0, weight=1)
        self.cfd_bar = ctk.CTkProgressBar(pr, height=6, progress_color=CYAN, fg_color=INPUT, corner_radius=3)
        self.cfd_bar.grid(row=0, column=0, sticky="ew", padx=(0, px(12)))
        self._bar_set(0)
        self.cfd_prog = ctk.CTkLabel(pr, text="idle", font=font(12, mono=True), text_color=TEXT2, anchor="e")
        self.cfd_prog.grid(row=0, column=1, sticky="e")
        self.cfd_log = ctk.CTkTextbox(b, font=font(12, mono=True), wrap="word", fg_color=INPUT, text_color=TEXT2,
                                      border_width=1, border_color=BORDER, corner_radius=8, height=100,
                                      scrollbar_button_color=BORDER, scrollbar_button_hover_color=BORDER2)
        self.cfd_log.grid(row=2, column=0, sticky="nsew")
        for tag, col in (("step", CYAN), ("err", RED), ("app", "#c3a6ff"), ("warn", AMBER), ("hint", TEXT),
                         ("note", MUTED)):
            self.cfd_log.tag_config(tag, foreground=col)
        self.cfd_log.configure(state="disabled")
        # OpenFOAM setup steps: shown in place of the log on request (nothing runs without OpenFOAM, and
        # the page has no spare height for a separate panel at 1280 x 800)
        self.foam_steps = ctk.CTkTextbox(b, font=font(12, mono=True), wrap="word", fg_color=INPUT, text_color=TEXT,
                                         border_width=1, border_color=blend(AMBER, CARD, 0.5), corner_radius=8,
                                         height=100, scrollbar_button_color=BORDER, scrollbar_button_hover_color=BORDER2)
        self.foam_steps_open = False
        self.v["cfd_res"].trace_add("write", lambda *_: self._cfd_res_changed())
        self._cfd_res_changed()
        self._cfd_buttons()
        if cfd3d is None:
            self._cfd_log_lines([f"The 3D CFD module could not be loaded ({CFD_IMPORT_ERROR})."], "err")
        else:
            self._cfd_log_lines(["Write or run a case; the OpenFOAM output streams here."], "note")

    def _toggle_foam_steps(self, show=None):
        self.foam_steps_open = (not self.foam_steps_open) if show is None else show
        idle_parts = (self.cfd_log, self.stepper, self.cfd_prog_row)     # nothing to track without OpenFOAM
        if self.foam_steps_open:
            for w in idle_parts:
                w.grid_remove()
            self.foam_steps.grid(row=0, column=0, rowspan=3, sticky="nsew")
        else:
            self.foam_steps.grid_remove()
            for w in idle_parts:
                w.grid()
        self.foam_steps_btn.configure(text="Hide details" if self.foam_steps_open else "Details ▾")

    def _bar_set(self, v):
        """Progress bar value; at 0 the bar's rounded end would show as a stray dot, so hide the fill."""
        self.cfd_bar.set(v)
        self.cfd_bar.configure(progress_color=CYAN if v > 0 else INPUT)

    # ================================================================== plumbing
    def _post(self, fn, *args):
        """Run fn(*args) on the Tk thread; callable from worker threads."""
        self.q.put((fn, args))

    def _pump(self):
        try:
            for _ in range(200):
                fn, args = self.q.get_nowait()
                try:
                    fn(*args)
                except Exception as e:  # noqa: BLE001 - a display bug must not stop the pump
                    traceback.print_exc()
                    self.status.set(f"Display error: {type(e).__name__}: {e}", "error")
        except queue.Empty:
            pass
        if self.cfd_lines:
            batch = []
            while self.cfd_lines and len(batch) < 400:
                batch.append(self.cfd_lines.popleft())
            self._cfd_log_lines(batch)
        if self.cfd_running:
            self._cfd_progress()
        # 20 Hz while work is in flight; idle, a slower poll keeps the app near 0 % CPU
        self.root.after(50 if self.sim_busy or self.cfd_busy or not self.q.empty() else 200, self._pump)

    def tk_error(self, exc, val, tb):
        """Tk callback exceptions go to the status bar (the exe has no console)."""
        traceback.print_exception(exc, val, tb)
        self.status.set(f"Internal error: {exc.__name__}: {val}", "error")

    def _on_return(self, e):
        if not isinstance(e.widget, tk.Text):
            self._run()

    def _show_banner(self, msg, title="ERROR", color=RED):
        self.banner.configure(border_color=color, fg_color=blend(color, BG, 0.9))
        self.banner_title.configure(text=title, text_color=color)
        self.banner_msg.configure(text=msg)
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, px(8)))

    def _hide_banner(self):
        self.banner.grid_remove()

    def _flag_entry(self, key):
        if self.err_key in self.entries:
            self.entries[self.err_key].configure(border_color=BORDER)
        self.err_key = key
        if key in self.entries:
            e = self.entries[key]
            e.configure(border_color=RED)
            e.focus_set()
            self._scroll_to(e)

    def _scroll_to(self, widget):
        """Scroll the input panel so widget is in view (an error in card 05 is below the fold)."""
        sc = self.left_sc
        if not str(widget).startswith(str(sc)):
            return
        self.root.update_idletasks()
        cv = sc._parent_canvas      # CTkScrollableFrame keeps its canvas here; no public scroll-to API
        top, view, total = widget.winfo_rooty() - sc.winfo_rooty(), cv.winfo_height(), max(sc.winfo_height(), 1)
        first = cv.yview()[0] * total
        if top < first or top + widget.winfo_height() > first + view:
            cv.yview_moveto(max(0.0, (top - view * 0.3) / total))

    # ================================================================== inputs
    @staticmethod
    def _nf_sizes():
        try:
            import sectionfoil
            return list(sectionfoil.available_model_sizes()) or NF_SIZES
        except Exception:  # noqa: BLE001 - stub or missing weights: offer the standard names
            return NF_SIZES

    def _mode_changed(self):
        mode = self.v["mode"].get()
        for key, e in self.mode_entries.items():
            on = key == mode
            e.configure(state="normal" if on else "disabled", text_color=TEXT if on else MUTED,
                        fg_color=INPUT if on else PANEL)

    def _aero_changed(self):
        nf = self.v["aero"].get() == "neuralfoil"
        if nf:
            self.nf_row.grid()
        else:
            self.nf_row.grid_remove()
        if self.outcome is None:        # once there is a result the chip describes it (see _show_outcome)
            self._aero_chip(nf)

    def _aero_chip(self, nf):
        self.header.set_chip("aero", "AERO · " + ("NEURALFOIL 2D" if nf else "ANALYTIC"), VIOLET if nf else CYAN)

    def _source_changed(self):
        tp = self.test_prop.get()
        self._idle_texts()
        self.entries["model"].configure(state="disabled" if tp else "normal", text_color=MUTED if tp else TEXT,
                                        fg_color=PANEL if tp else INPUT)
        for seg in (self.units_seg, self.axis_seg):
            seg.configure(state="disabled" if tp else "normal", selected_color=BORDER if tp else CYAN_D)

    def _start_dir(self):
        """Folder for file dialogs: the model's own folder, else the home folder (not the exe's)."""
        d = os.path.dirname(self.v["model"].get().strip().strip('"'))
        return d if d and os.path.isdir(d) else os.path.expanduser("~")

    def _stem(self):
        o = self.outcome
        if o is None or o.args.test_prop:
            return "testprop"
        return os.path.splitext(os.path.basename(o.args.model))[0]

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Open propeller model", initialdir=self._start_dir(), filetypes=[
            ("3D models", "*.stl *.STL *.step *.stp *.STEP *.STP *.obj *.ply"), ("All files", "*.*")])
        if p:
            self.v["model"].set(p)
            self.test_prop.set(False)
            self._source_changed()

    def _browse_polar(self):
        p = filedialog.askopenfilename(title="Airfoil polar CSV (alpha_deg, cl, cd)", initialdir=self._start_dir(),
                                       filetypes=[("CSV", "*.csv *.txt"), ("All files", "*.*")])
        if p:
            self.v["polar"].set(p)

    def _num(self, key, label, kind=float, lo=None, hi=None, required=False, open_lo=False, big=False):
        """The field as a number. A comma is read as a decimal point (12,5), except where it looks
        like a thousands separator in a field whose values run into the thousands (big: '8,000' rpm
        must not quietly run at 8 rpm); there the user is asked to retype it."""
        s = self.v[key].get().strip()
        if not s:
            if required:
                raise InputError(key, f"{label}: enter a value.")
            return None
        if "," in s and "." in s:
            raise InputError(key, f"{label}: '{s}' mixes ',' and '.'; type the number without a thousands "
                                  f"separator, e.g. 1234.5.")
        if (big or kind is int) and re.fullmatch(r"[-+]?[1-9]\d{0,2}(,\d{3})+", s):
            raise InputError(key, f"{label}: '{s}' looks like it has a thousands separator; type "
                                  f"{s.replace(',', '')} (a comma is read as a decimal point).")
        try:
            x = int(s) if kind is int else float(s.replace(",", "."))   # accept a decimal comma
        except ValueError:
            raise InputError(key, f"{label}: '{s}' is not a {'whole ' if kind is int else ''}number.") from None
        if not math.isfinite(x):
            raise InputError(key, f"{label}: '{s}' is not a finite number.")
        if lo is not None and (x <= lo if open_lo else x < lo):
            raise InputError(key, f"{label} must be {'greater than' if open_lo else 'at least'} {lo:g} (got {s}).")
        if hi is not None and x > hi:
            raise InputError(key, f"{label} must be at most {hi:g} (got {s}).")
        return x

    def _argv(self):
        """propsim command-line arguments from the form; raises InputError naming the field to fix."""
        a = []
        if self.test_prop.get():
            a.append("--test-prop")
        else:
            path = self.v["model"].get().strip().strip('"')
            if not path:
                raise InputError("model", "Choose a model file (Browse...), or switch on the built-in test prop.")
            if not os.path.isfile(path):
                raise InputError("model", f"Model file not found: {path}")
            if not path.lower().endswith(MODEL_TYPES):
                raise InputError("model", "Unsupported model type: use .stl, .step/.stp, .obj or .ply.")
            a += [path, "--units", self.v["units"].get(), "--axis", self.v["axis"].get()]
        blades = self._num("blades", "Blades", int, lo=1)
        if blades is not None:
            a += ["--blades", str(blades)]
        hub = self._num("hub", "Hub cut-off r/R", lo=0, hi=0.9)
        if hub is not None:
            a += ["--hub-ratio", repr(hub)]
        mode = self.v["mode"].get()
        kv = self._num("kv", "KV", lo=0, open_lo=True, big=True)
        if mode == "tip":
            a += ["--tip-speed", repr(self._num("tip", "Tip speed", lo=0, open_lo=True, required=True))]
        elif mode == "rpm":
            a += ["--rpm", repr(self._num("rpm", "RPM", lo=0, open_lo=True, required=True, big=True))]
        else:
            power = self._num("power", "Motor power", lo=0, open_lo=True, required=True, big=True)
            if kv is None:
                raise InputError("kv", "Motor mode needs the motor KV (under Motor and battery).")
            a += ["--motor-power", repr(power)]
        if kv is not None:
            a += ["--kv", repr(kv)]
        voltage = self._num("voltage", "Battery voltage", lo=0, open_lo=True)
        cells = self._num("cells", "LiPo cells", int, lo=1)
        cellv = self._num("cellv", "Volts per cell", lo=0, open_lo=True)
        if voltage is not None:
            a += ["--voltage", repr(voltage)]
        elif cells is not None:
            a += ["--cells", str(cells), "--cell-voltage", repr(cellv if cellv is not None else 3.7)]
        for key, flag, label, open_lo in (
                ("rm", "--rm", "Rm", False), ("i0", "--i0", "I0", False), ("cd0", "--cd0", "Cd0", False),
                ("airspeed", "--airspeed", "Airspeed", False), ("rho", "--rho", "Air density", True),
                ("clmax", "--cl-max", "Cl max", True), ("clalpha", "--cl-alpha", "Lift slope", True)):
            x = self._num(key, label, lo=0, open_lo=open_lo)
            if x is not None:
                a += [flag, repr(x)]
        polar = self.v["polar"].get().strip().strip('"')
        if polar:
            if not os.path.isfile(polar):
                raise InputError("polar", f"Polar file not found: {polar}")
            a += ["--polar", polar]
        esc = self._num("esc", "ESC efficiency", lo=1, hi=100)
        if esc is not None:
            a += ["--esc-eff", repr(esc / 100)]
        qmax = self._num("qmax", "Max torque", lo=0, open_lo=True)
        if qmax is not None:
            a += ["--max-torque", repr(qmax)]
        drive = self._num("drive", "Motor+ESC efficiency", lo=1, hi=100)
        if drive is not None:
            a += ["--drive-eff", repr(drive / 100)]
        if self.v["aero"].get() == "neuralfoil":
            a += ["--aero", "neuralfoil", "--nf-model", self.v["nf_model"].get()]
            ncrit = self._num("ncrit", "Ncrit", lo=1, hi=14)
            if ncrit is not None:
                a += ["--ncrit", repr(ncrit)]
        if self.sweep.get():
            a.append("--sweep")
        if self.show_geom.get():
            a.append("--show-geometry")
        extra = getattr(self, "gen_model_extra", None)
        if extra and not self.test_prop.get() and \
                os.path.normcase(os.path.abspath(self.v["model"].get().strip().strip('"'))) == extra[0]:
            a += extra[1]             # a generated prop with features: their aerodynamic effects
        return a

    @staticmethod
    def _key(argv):
        """The arguments that decide the physics of a run (the report-only flags do not)."""
        return tuple(x for x in argv if x not in ("--sweep", "--show-geometry"))

    def _inputs_changed(self, *_):
        """Any form change: re-check (debounced) whether the shown result still matches the form."""
        if self.stale_job is None:
            self.stale_job = self.root.after(250, self._check_stale)

    def _check_stale(self):
        self.stale_job = None
        self._idle_texts()
        if self.outcome is None:
            return
        try:
            stale = self._key(self._argv()) != self.last_key
        except InputError:
            stale = True
        if stale != self.stale:
            self.stale = stale
            self.strip.set_stale(stale)
            for t in self.tiles.values():
                t.set_stale(stale)
            self.tabbar.set_right(self.result_label + ("  ·  previous run, inputs changed" if stale else ""))
        if self.tab == "cfd":
            self._cfd_update_op()

    def _idle_texts(self):
        """Empty-state texts that say what happens next."""
        if self.sim_busy:
            strip, view = "Running the simulation…", "Analysing"
        elif self.test_prop.get() or self.v["model"].get().strip():
            strip, view = "Press RUN (or Enter) to analyse " + ("the built-in test prop." if self.test_prop.get()
                                                                 else "the model."), "Awaiting run"
        else:
            strip, view = ("No model yet: pick an STL / STEP file or switch on the built-in test prop, then press RUN.",
                           "Awaiting model")
        self.strip.set_empty(strip)
        self.view3d.set_empty(view)

    def _input_error(self, e: InputError):
        self._flag_entry(e.key)
        self._show_banner(str(e), "CHECK INPUT", AMBER)
        self.status.set(str(e), "warn")

    # ================================================================== simulation
    def _run(self):
        if self.sim_busy:
            return
        if self.gen_busy == "step" and (self.v["model"].get().lower().endswith((".step", ".stp"))
                                        and not self.test_prop.get()):
            self._show_banner("A STEP export is running (it uses the same CAD kernel); run again when it has "
                              "finished.", "BUSY", AMBER)
            return
        try:
            argv = self._argv()
        except InputError as e:
            self._input_error(e)
            return
        self._flag_entry(None)
        self._hide_banner()
        self.sim_busy = True
        self._idle_texts()
        self.sim_t0 = time.time()
        self.run_btn.set_busy(True, "Running")
        self.header.set_busy(True)
        self.status.set("Running 2D section analysis + BEMT…" if "neuralfoil" in argv else "Running…",
                        "busy", right="")
        threading.Thread(target=self._sim_worker, args=(argv,), daemon=True).start()
        self._gen_buttons()

    @staticmethod
    def _simulate(argv, say):
        """propsim.simulate with argparse's complaints and the library's SystemExit messages (e.g.
        a STEP file without gmsh) turned into InputError (worker thread)."""
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                args = propsim.parse_args(argv)
        except SystemExit:
            lines = [ln for ln in err.getvalue().splitlines() if ln.strip()]
            raise InputError(None, lines[-1] if lines else "invalid input") from None
        try:
            return propsim.simulate(args=args, progress=say)
        except SystemExit as e:
            raise InputError(None, str(e.code) if e.code not in (None, 0) else "the simulation stopped") from None

    def _sim_worker(self, argv):
        try:
            out = self._simulate(argv, lambda m: self._post(self.status.set, m))
            self._post(self._sim_done, out, argv, None, "")
        except BaseException as e:  # noqa: BLE001 - anything, so the window always hears back
            self._post(self._sim_done, None, argv, self._describe(e, argv), traceback.format_exc())

    @staticmethod
    def _describe(e, argv=()):
        if isinstance(e, NotImplementedError):
            if "neuralfoil" in argv:
                return ("2D section analysis (NeuralFoil) is not available in this build yet. "
                        "Choose 'Analytic' under Aerodynamics.")
            return f"Not available in this build: {e}"
        if isinstance(e, FileNotFoundError):
            return f"File not found: {e.filename or e}"
        if isinstance(e, MemoryError):
            return "Out of memory. Try a coarser model."
        if isinstance(e, ValueError):
            return str(e)
        if isinstance(e, SystemExit):
            return str(e.code) if e.code not in (None, 0) else "the calculation stopped"
        return f"{type(e).__name__}: {e}"

    def _sim_done(self, out, argv, err, tb):
        self.sim_busy = False
        self._gen_buttons()
        self.run_btn.set_busy(False)
        self.header.set_busy(self.cfd_busy)
        self._idle_texts()
        dt = time.time() - self.sim_t0
        if err:
            self._show_banner(err)
            self.status.set(f"Failed: {err.splitlines()[0]}", "error", right=f"{dt:.1f} s")
            self._set_report(f"\nThe simulation failed:\n\n{err}\n\n{tb}", error=True)
            self._check_stale()
            return
        self.outcome, self.last_key = out, self._key(argv)
        try:
            self._show_outcome(out)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._show_banner(f"Results computed but could not be displayed: {type(e).__name__}: {e}")
        r = out.res
        self.status.set(f"Done: {out.name}  ·  {r.thrust:.2f} N at {r.rpm:,.0f} rpm  ·  {r.power:.1f} W shaft",
                        "ok", right=f"{dt:.1f} s")

    def _show_outcome(self, o):
        r, a, g, e = o.res, o.args, o.geom, o.elec
        self._set_report(o.text.lstrip("\n"))
        R = g.radius
        i75 = int(np.argmin(abs(g.r - 0.75 * R)))
        nf = getattr(a, "aero", "analytic") == "neuralfoil"
        chips = [(f"Ø {2 * R * 1e3:.1f} mm · {2 * R / 0.0254:.2f} in", CYAN),
                 (f"{g.blades} blades", CYAN),
                 (f"pitch {g.pitch[i75] * 1e3:.0f} mm · {g.pitch[i75] / 0.0254:.2f} in", CYAN),
                 ("NeuralFoil 2D sections" if nf else ("polar file" if a.polar else "analytic airfoil"),
                  VIOLET if nf else CYAN),
                 ("hover" if r.airspeed == 0 else f"{r.airspeed:g} m/s", CYAN),
                 (f"ρ {a.rho:g} kg/m³", CYAN)]
        warns = [(ln.strip()[8:].strip(), AMBER) for ln in o.text.splitlines() if ln.strip().startswith("warning:")]
        if e is not None and e.get("throttle", 0) > 1.0001:
            warns.append(("the motor needs more voltage than the battery has for this point", AMBER))
        if o.limited_by:
            warns.append((f"limited by {o.limited_by}", CYAN))
        self.strip.set(o.name, chips, warns)
        self._show_tiles(o)
        sw = o.sweep
        if sw:
            series = [dict(y=[s["thrust"] for s in sw], label="thrust", color=CYAN, axis="L", fill=True, unit="N"),
                      dict(y=[s["power"] for s in sw], label="shaft power", color=MAGENTA, axis="R", unit="W")]
            if "battery_power" in sw[0]:
                series.append(dict(y=[s["battery_power"] for s in sw], label="battery", color=VIOLET, axis="R",
                                   dash=(6, 4), unit="W"))
            self.sweep_chart.set_data([s["rpm"] for s in sw], series, marker=(r.rpm, f"OP {r.rpm:,.0f} rpm"))
        st = r.stations
        xr = np.asarray(st["r"]) / R
        stalled = np.asarray(st["stalled"], bool)
        self.radial_chart.set_data(
            xr, [dict(y=st["dT"], label="dT/dr", color=CYAN, axis="L", fill=True, unit="N/m"),
                 dict(y=np.degrees(st["alpha"]), label="α", color=MAGENTA, axis="R", unit="°", hl=stalled)],
            bands=self._bands(xr, stalled), band_label="stalled")
        try:
            self.view3d.set_model(g, o.mesh, r)
        except Exception as ex:  # noqa: BLE001 - the preview is a nicety
            traceback.print_exc()
            self.view3d.clear()
            self.status.set(f"3D preview unavailable: {ex}", "warn")
        self._show_geometry(o)
        self._aero_chip(nf)
        self.result_label = f"{o.name}  ·  {time.strftime('%H:%M:%S')}"
        self.stale = None           # force the redraw of the stale marks
        self._check_stale()
        self._cfd_update_op()

    @staticmethod
    def _bands(x, mask):
        """Runs of True in mask as (x0, x1) spans reaching halfway to the neighbouring stations."""
        out, i, n = [], 0, len(x)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((x[i] - (x[i] - x[i - 1]) / 2 if i > 0 else x[i],
                        x[j] + (x[j + 1] - x[j]) / 2 if j + 1 < n else x[j]))
            i = j + 1
        return out

    def _show_tiles(self, o):
        r, a, e, t = o.res, o.args, o.elec, self.tiles
        t["thrust"].set(f"{r.thrust:.2f}", "N", f"CT {r.ct:.4f} · CP {r.cp:.4f}",
                        extra=[(f"{r.thrust / G * 1000:,.0f}", "g"), (f"{r.thrust / G:.3f}", "kgf"),
                               (f"{r.thrust * N_TO_LBF:.3f}", "lbf")], warn=r.thrust < 0)
        t["torque"].set(f"{r.torque:.4f}", "N·m", "at the shaft")
        t["power"].set(f"{r.power:.1f}", "W", f"{r.thrust / G * 1000 / max(r.power, 1e-9):.1f} gf/W shaft")
        if e is not None:
            pe = e["battery_power"]
            t["elec"].set(f"{pe:.1f}", "W", f"{r.thrust / G * 1000 / max(pe, 1e-9):.1f} gf/W · "
                          f"drive {e['drive_efficiency'] * 100:.0f} %" if "drive_efficiency" in e else
                          f"{r.thrust / G * 1000 / max(pe, 1e-9):.1f} gf/W", label="Battery power")
        else:
            pe = r.power / a.drive_eff
            t["elec"].set(f"{pe:.1f}", "W", f"at {a.drive_eff * 100:.0f} % motor+ESC", label="Electrical power")
        t["rpm"].set(f"{r.rpm:,.0f}", "rpm", f"{r.rpm / 60:.1f} rev/s")
        t["tip"].set(f"{r.tip_speed:.1f}", "m/s", f"Mach {r.tip_mach:.2f}" + ("  above 0.7" if r.tip_mach > 0.7 else ""),
                     warn=r.tip_mach > 0.7)
        if r.airspeed > 0:
            J = r.airspeed / (r.rpm / 60 * 2 * o.geom.radius)
            t["fom"].set(f"{r.efficiency * 100:.1f}", "%", f"advance ratio J {J:.3f}", label="Prop efficiency")
        else:
            t["fom"].set(f"{r.figure_of_merit:.2f}", "", f"{r.stalled_fraction * 100:.0f} % of span stalled",
                         label="Figure of merit", warn=r.stalled_fraction > 0.4)
        if e is not None:
            cur = e.get("battery_current", e["current"])
            t["current"].set(f"{cur:.2f}", "A", f"motor {e['current']:.1f} A at {e['motor_voltage']:.2f} V",
                             label="Battery current" if "battery_current" in e else "Motor current")
        else:
            t["current"].set("—", "", "give the motor KV", label="Current")
        if e is not None and "throttle" in e:
            thr = e["throttle"]
            lim = o.limited_by or ""
            sub = ("throttled to torque limit" if "torque limit" in lim and thr < 0.9995 else
                   "throttled to motor rating" if "throttled" in lim else
                   f"at {e['battery_voltage']:.2f} V (solved)" if "voltage solved" in lim else
                   f"full throttle, {e['battery_voltage']:.2f} V" if "full throttle" in lim else
                   f"of {e['battery_voltage']:.2f} V battery")
            t["throttle"].set(f"{thr * 100:.1f}" if thr < 0.9995 else f"{thr * 100:.0f}", "%", sub, bar=thr,
                              warn=thr > 1.0001)
        else:
            t["throttle"].set("—", "", "give KV + battery")

    def _show_geometry(self, o):
        g, st, R = o.geom, o.res.stations, o.geom.radius
        self.tree.delete(*self.tree.get_children())
        stalled = np.asarray(st["stalled"], bool)
        for i in range(len(g.r)):
            vals = (i + 1, f"{g.r[i] / R:.3f}", f"{g.r[i] * 1e3:.1f}", f"{g.chord[i] * 1e3:.2f}",
                    f"{math.degrees(g.beta[i]):.2f}", f"{g.pitch[i] * 1e3:.1f}", f"{g.thickness[i]:.3f}",
                    f"{g.camber[i]:.3f}", f"{math.degrees(st['alpha'][i]):.2f}", f"{st['cl'][i]:.3f}",
                    f"{st['cd'][i]:.4f}", f"{st['re'][i] / 1e3:.0f}k", f"{st['mach'][i]:.3f}", f"{st['dT'][i]:.2f}")
            tags = (("odd",) if i % 2 else ()) + (("stall",) if stalled[i] else ())
            self.tree.insert("", "end", iid=str(i), values=vals, tags=tags)
        self.geom_hdr.set_right(f"{len(g.r)} stations · amber = stalled")
        xr = g.r / R
        self.dist_chart.set_data(xr, [dict(y=g.chord * 1e3, label="chord", color=CYAN, axis="L", fill=True, unit="mm"),
                                      dict(y=np.degrees(g.beta), label="β", color=VIOLET, axis="R", unit="°")])
        i75 = int(np.argmin(abs(g.r - 0.75 * R)))
        self.tree.selection_set(str(i75))
        self.tree.see(str(i75))
        self._geom_select()

    def _geom_select(self):
        o, sel = self.outcome, self.tree.selection()
        if not o or not sel:
            return
        i = int(sel[0])
        g, st = o.geom, o.res.stations
        xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 60)))
        secs = g.sections if g.sections is not None else []
        loop = section_loop(secs[i] if i < len(secs) else None, g.thickness[i], g.camber[i], xs)
        alpha, phi, stall = float(st["alpha"][i]), float(st["phi"][i]), bool(st["stalled"][i])
        self.section_view.set(dict(
            loop=loop, beta=float(g.beta[i]), phi=phi,
            title=f"#{i + 1}  r/R {g.r[i] / g.radius:.3f}  chord {g.chord[i] * 1e3:.1f} mm",
            stats=[(f"β {math.degrees(g.beta[i]):.1f}°", VIOLET),
                   (f"φ {math.degrees(phi):.1f}°", MAGENTA),
                   (f"α {math.degrees(alpha):.1f}°" + (" stalled" if stall else ""), AMBER if stall else CYAN),
                   (f"Re {st['re'][i] / 1e3:.0f}k", TEXT2)]))

    # ================================================================== save / export
    def _save_text(self):
        if not self.outcome:
            self._show_banner("Run a simulation first.", "NOTHING TO SAVE", AMBER)
            return
        kind = tk.StringVar(self.root, "Text report")
        p = filedialog.asksaveasfilename(title="Save results", initialdir=self._start_dir(),
                                         initialfile=f"{self._stem()}_results", typevariable=kind,
                                         filetypes=[("Text report", "*.txt"), ("JSON summary", "*.json")])
        if not p:
            return
        # the format follows the extension typed, else the file type picked in the dialog
        as_json = kind.get().startswith("JSON")
        ext = os.path.splitext(p)[1].lower()
        if ext in (".txt", ".json"):
            as_json = ext == ".json"
        elif not ext:
            p += ".json" if as_json else ".txt"
            if os.path.exists(p) and not messagebox.askyesno(APP_TITLE, f"{p} already exists. Replace it?"):
                return
        o = self.outcome
        try:
            with open(p, "w", encoding="utf-8") as f:
                if as_json:
                    json.dump(propsim.summary_dict(o.res, o.elec, o.geom, o.limited_by, o.args), f, indent=2,
                              default=float)
                else:
                    f.write(o.text.lstrip("\n"))
            self.status.set(f"Saved {p}", "ok")
        except OSError as e:
            self._show_banner(f"Could not save: {e}")

    def _save_csv(self):
        if not self.outcome:
            self._show_banner("Run a simulation first.", "NOTHING TO EXPORT", AMBER)
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                         title="Export per-station results", initialdir=self._start_dir(),
                                         initialfile=f"{self._stem()}_stations.csv")
        if not p:
            return
        try:
            propsim.write_csv(p, self.outcome.res)
            self.status.set(f"Saved {p}", "ok")
        except OSError as e:
            self._show_banner(f"Export failed: {e}")

    # ================================================================== 3D CFD
    def _set_foam_dot(self, color):
        d, s = self.foam_dot, px(14)
        d.delete("all")
        d.create_oval(0, 0, s, s, fill=blend(color, CARD, 0.7), outline="")
        d.create_oval(px(3), px(3), s - px(3), s - px(3), fill=color, outline="")

    def _cfd_detect(self, refresh=False):
        if cfd3d is None:
            self._cfd_detected(None, f"3D CFD module not available: {CFD_IMPORT_ERROR}")
            return
        self.foam_text.configure(text="Checking for OpenFOAM…")
        self._set_foam_dot(AMBER)
        self._enable(self.foam_btn, False)

        def work():
            try:
                try:
                    inst = cfd3d.detect_openfoam(refresh=True) if refresh else cfd3d.detect_openfoam()
                except TypeError:            # interface without refresh
                    inst = cfd3d.detect_openfoam()
                self._post(self._cfd_detected, inst, "")
            except Exception as e:  # noqa: BLE001
                self._post(self._cfd_detected, None, f"Detecting OpenFOAM failed: {type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _cfd_detected(self, inst, err):
        self.cfd_install = inst
        self._enable(self.foam_btn, cfd3d is not None)
        if inst is not None and inst.found:
            self.foam_text.configure(text=f"Ready  ·  {inst.message or 'OpenFOAM found'}")
            self._set_foam_dot(GREEN)
            self.foam_hint = ""
            self.foam_steps_btn.grid_remove()
            self._toggle_foam_steps(False)
            self.header.set_chip("foam", "CFD · BUILT-IN" if getattr(inst, "mode", "") == "bundled" else
                                 f"OPENFOAM · {getattr(inst, 'version', '') or 'READY'}", GREEN)
        else:
            msg = err or (inst.message if inst is not None else "") or "No OpenFOAM installation was found."
            self.foam_text.configure(text=err.split("\n")[0] if err else
                                     ("CFD engine missing  ·  the 'cfd-engine' folder must sit next to PropSim.exe; "
                                      "'Details' says more" if getattr(sys, "frozen", False) else
                                      "OpenFOAM not found  ·  cases can still be written; 'Details' shows how to "
                                      "install it"))
            self._set_foam_dot(RED if err else AMBER)
            if msg != self.foam_hint:
                self.foam_hint = msg
                t = self.foam_steps
                t.configure(state="normal")
                t.delete("1.0", "end")
                t.insert("end", msg)
                t.configure(state="disabled")     # still selectable, so the commands can be copied
                self._cfd_log_lines(["==> no CFD engine found: 'Details' (top right) says why and how to fix it"],
                                    "warn")
            self.foam_steps_btn.grid(row=0, column=2, sticky="e", padx=(px(8), px(8)))
            self.header.set_chip("foam", "OPENFOAM · NOT FOUND", AMBER)
        self._cfd_buttons()

    def _cfd_buttons(self):
        busy, avail = self.cfd_busy, cfd3d is not None
        found = bool(self.cfd_install is not None and self.cfd_install.found)
        self._enable(self.cfd_write_btn, avail and not busy)
        self._enable(self.cfd_run_btn, avail and found and not busy)
        self._enable(self.cfd_stop_btn, self.cfd_running)
        self._enable(self.cfd_load_btn, avail and not busy)
        self._enable(self.cfd_open_btn, bool(self.cfd_case))

    def _cfd_res_changed(self):
        pre = getattr(cfd3d, "RESOLUTIONS", {}).get(self.v["cfd_res"].get()) if cfd3d else None
        self.cfd_res_info.configure(
            text=f"{pre.get('cells', '?')} cells · {pre.get('iterations', '?')} iterations" if pre else "")

    @staticmethod
    def _op_text(out):
        r = out.res
        return (f"{out.name}  ·  {r.rpm:,.0f} rpm  ·  " + ("hover" if r.airspeed == 0 else f"{r.airspeed:g} m/s")
                + f"  ·  ρ {out.args.rho:g}  ·  tip Mach {r.tip_mach:.2f}")

    def _cfd_source(self, out):
        """The BEMT run a case is made from; the tab describes it while the case runs or is shown."""
        r = out.res
        self.cfd_out, self.cfd_desc, self.cfd_bemt = out, self._op_text(out), (r.thrust, r.torque, r.power,
                                                                               r.stalled_fraction)
        self._cfd_update_op()

    def _cfd_update_op(self):
        """Case description and BEMT column: the case being run or shown, else what the next case would use."""
        o = self.outcome
        if self.cfd_busy or self.cfd_shown:
            text = self.cfd_desc or "Running the BEMT for the case…"
            if self.cfd_out is not None and o is not self.cfd_out:
                text += "\nThe dashboard now shows another BEMT run; this tab stays with the case."
            bemt = self.cfd_bemt
        else:
            try:
                current = self._key(self._argv())
            except InputError:
                current = None
            if o is None:
                text = "No simulation yet: the case uses the model and RPM of the current inputs."
            else:
                text = self._op_text(o)
                if current is not None and current != self.last_key:
                    text += "\nInputs changed since the last run: the BEMT runs again first."
            bemt = (o.res.thrust, o.res.torque, o.res.power) if o is not None else None
        self.cfd_op.configure(text=text)
        bemt = tuple(bemt[:3]) if bemt else bemt
        if not self.cfd_shown:      # until a CFD result is in, the table shows what the BEMT predicts
            for (k, fmt), v in zip((("T", "{:.2f}"), ("Q", "{:.4f}"), ("P", "{:.1f}")), bemt or (None,) * 3):
                self.cfd_cells[k][0].configure(text=fmt.format(v) if v is not None else "—")

    def _cfd_settings(self):
        iters = self._num("cfd_iters", "CFD iterations", int, lo=1)
        cores = self.v["cfd_cores"].get()
        return dict(resolution=self.v["cfd_res"].get(), n_procs=0 if cores == "Auto" else int(cores),
                    air=self.v["cfd_air"].get().lower(),
                    iterations=iters or 0)

    def _cfd_start(self, run):
        if self.cfd_busy or cfd3d is None:
            return
        try:
            argv = self._argv()
            settings = self._cfd_settings()
        except InputError as e:
            self._input_error(e)
            return
        key = (self._key(argv), tuple(sorted(settings.items())))
        case_dir = folder = None
        replace = False
        if run and self.cfd_case and self.cfd_case[1] == key and os.path.isdir(self.cfd_case[0]):
            case_dir = self.cfd_case[0]           # run the case already written for exactly these inputs
            if self._case_has_output(case_dir) and not messagebox.askyesno(
                    APP_TITLE, f"Run this case again?\n\n{case_dir}\n\nIts current mesh and CFD results are "
                               "replaced."):
                return
        else:
            folder = filedialog.askdirectory(title="Folder for the OpenFOAM case (a new subfolder is made if it is "
                                                   "not empty)", initialdir=self._start_dir(), mustexist=False)
            if not folder:
                return
            if self._is_case(folder) and self._case_has_output(folder):
                replace = messagebox.askyesnocancel(
                    APP_TITLE, f"{folder}\n\nholds a PropSim case that has already been meshed or run.\n\n"
                               "Yes: replace it (its mesh and CFD results are deleted).\n"
                               "No: keep it and write the new case into a new subfolder.\n"
                               "Cancel: do nothing.")
                if replace is None:
                    return
        self._hide_banner()
        self._toggle_foam_steps(False)            # the log shows the progress
        self.cfd_busy, self.cfd_running, self.cfd_done_ok = True, False, False
        self.cfd_stop.clear()
        self.cfd_step, self.cfd_iter, self.cfd_iter_t0 = "", 0, None
        self.cfd_t0 = time.time()
        # the comparison shown belongs to the previous case until this run produces forces
        self.cfd_shown = False
        for _b, c_, d_ in self.cfd_cells.values():
            c_.configure(text="—")
            d_.configure(text="—", text_color=TEXT2)
        self.cfd_res_hdr.set_right("")
        self.cfd_msg.configure(text="Waiting for the solver's first forces…" if run else "No CFD result yet.",
                               text_color=MUTED)
        self.cfd_out = self.cfd_desc = self.cfd_bemt = None
        if self.outcome is not None and self.last_key == key[0]:
            self._cfd_source(self.outcome)
        self.hist_chart.set_data(None, empty="Force history appears while the solver runs")
        self.header.set_busy(True)
        self.tabbar.set_badge("cfd", MAGENTA)
        self._cfd_buttons()
        self._draw_stepper()
        self.bar_mode = "indeterminate"
        self.cfd_bar.configure(mode="indeterminate", progress_color=CYAN)
        self.cfd_bar.start()
        self.cfd_prog.configure(text="preparing")
        self.status.set("3D CFD: preparing the case…", "busy")
        self._cfd_update_op()
        threading.Thread(target=self._cfd_worker, args=(argv, settings, key, case_dir, folder, run, replace),
                         daemon=True).start()

    def _cfd_worker(self, argv, settings, key, case_dir, folder, run, replace=False):
        def say(text, tag="app"):
            self._post(self._cfd_log_lines, [text], tag)
        try:
            out = self.outcome if self.outcome is not None and self.last_key == self._key(argv) else None
            if out is None:
                say("==> running the BEMT for the current inputs")
                out = self._simulate(argv, lambda m: self._post(self.status.set, "3D CFD: " + m))
                self._post(self._sim_from_cfd, out, argv)
            self._post(self._cfd_source, out)
            if case_dir is None:
                case_dir = self._case_dir_for(folder, out, settings["resolution"], replace)
                st = cfd3d.CaseSettings(rpm=float(out.res.rpm), airspeed=float(out.args.airspeed),
                                        rho=float(out.args.rho), bemt=cfd3d.bemt_record(out, argv), **settings)
                say(f"==> writing the case for {out.res.rpm:,.0f} rpm into {case_dir}")
                info = cfd3d.write_case(case_dir, out.mesh, out.geom, st) or {}
                self.cfd_info = info
                self.cfd_case = (case_dir, key)
                lines = [f"case written: {info.get('case_dir', case_dir)}"]
                if info:
                    lines.append(f"  {info.get('flavor', '?')} syntax, {info.get('resolution', '?')} mesh"
                                 + (f" (~{info['est_cells'] / 1e6:.2f} M cells)" if "est_cells" in info else "")
                                 + f", {info.get('n_procs', '?')} processes, {info.get('iterations', '?')} iterations")
                    lines.append(f"  {info.get('air', 'incompressible')} air ({info.get('solver', '?')}), tip Mach "
                                 f"{info.get('tip_mach', 0):.2f}")
                lines += [f"  note: {n}" for n in info.get("notes", [])]
                allrun = os.path.join(case_dir, "Allrun")
                lines.append("  to run it on another machine with OpenFOAM:  bash Allrun" if os.name == "nt" else
                             f"  to run it outside PropSim:  bash \"{allrun}\"")
                self._post(self._cfd_log_lines, lines, "app")
            if not run:
                found = bool(getattr(self.cfd_install, "found", False))
                self._post(self._cfd_finished, None, case_dir, "",
                           "Case written. " + ("Run it here with 'Run OpenFOAM', or copy the folder to another machine "
                                               "with OpenFOAM and run Allrun." if found else
                                               "Copy the folder to a machine with OpenFOAM and run Allrun there, or "
                                               "see 'Details' (top right)."))
                return
            inst = self.cfd_install
            if inst is None or not inst.found:
                self._post(self._cfd_finished, None, case_dir, "OpenFOAM was not found, so the case cannot run "
                           "here. It has been written; run Allrun where OpenFOAM is installed.", "")
                return
            self.cfd_running = True
            self._post(self._cfd_buttons)
            done = threading.Event()
            threading.Thread(target=self._cfd_poll, args=(case_dir, done), daemon=True).start()
            try:
                res = cfd3d.run_case(case_dir, inst, on_line=self._cfd_on_line, stop_event=self.cfd_stop)
            finally:
                done.set()
            self._post(self._cfd_finished, res, case_dir, "", "")
        except NotImplementedError:
            self._post(self._cfd_finished, None, case_dir, "3D CFD is not implemented in this build "
                       "(cfd3d is a stub).", "")
        except BaseException as e:  # noqa: BLE001 - anything, so the tab is never left busy
            if not isinstance(e, InputError):
                self._post(self._cfd_log_lines, traceback.format_exc().splitlines(), "err")
            self._post(self._cfd_finished, None, case_dir, self._describe(e, argv), "")

    def _sim_from_cfd(self, out, argv):
        self.outcome, self.last_key = out, self._key(argv)
        self._show_outcome(out)

    @staticmethod
    def _is_case(d):
        return os.path.exists(os.path.join(d, getattr(cfd3d, "CASE_INFO", "propsim_case.json")))

    @staticmethod
    def _case_has_output(d):
        """True when a case folder holds a mesh or solver output (minutes to hours of work that
        cfd3d.write_case would delete)."""
        try:
            names = os.listdir(d)
        except OSError:
            return False
        return (any(n == "postProcessing" or n.startswith(("log.", "processor")) for n in names)
                or os.path.isdir(os.path.join(d, "constant", "polyMesh")))

    @classmethod
    def _case_dir_for(cls, folder, out, resolution, replace=False):
        """The chosen folder if it is empty, a PropSim case with nothing meshed or run in it yet, or
        one the user agreed to replace; else a new subfolder, so no result is deleted unasked."""
        folder = os.path.abspath(folder)
        if (not os.path.isdir(folder) or not os.listdir(folder) or replace
                or (cls._is_case(folder) and not cls._case_has_output(folder))):
            return folder
        a = out.args
        stem = "testprop" if a.test_prop else os.path.splitext(os.path.basename(a.model))[0]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "prop"
        base = os.path.join(folder, f"cfd_{stem}_{out.res.rpm:.0f}rpm_{resolution}")
        cand, k = base, 2
        while os.path.isdir(cand) and os.listdir(cand):
            cand, k = f"{base}_{k}", k + 1
        return cand

    def _cfd_on_line(self, line):
        """An OpenFOAM output line (worker thread): queue it and track the step and iteration."""
        self.cfd_lines.append(line)
        if line.startswith("==> "):
            word = line[4:].split(" ")[0].rstrip(":")
            if word not in ("stop", "stopped", "stopping", "note", "reading") and " done:" not in line:
                self.cfd_step = word
        else:
            m = re.match(r"^Time = (\d+)\s*$", line)
            if m and self.cfd_step == "solver":     # checkMesh and friends print "Time = 0" too
                self.cfd_iter = int(m.group(1))
                if self.cfd_iter_t0 is None:
                    self.cfd_iter_t0 = (time.time(), self.cfd_iter)

    def _cfd_poll(self, case_dir, done):
        """Live forces while the solver runs (worker thread)."""
        while not done.wait(5.0):
            if self.cfd_step != "solver" or self.cfd_iter < 5:
                continue
            try:
                res = cfd3d.read_results(case_dir)
            except Exception:  # noqa: BLE001 - files being written; try again next time
                continue
            if res.ok:
                self._post(self._cfd_show_result, res, False)

    def _cfd_progress(self):
        """Progress text, bar and stepper from the worker's step / iteration (Tk thread)."""
        el = time.time() - self.cfd_t0
        total = int(self.cfd_info.get("iterations", 0) or 0)
        step, it = self.cfd_step, self.cfd_iter
        if step == "solver" and it and total:
            if self.bar_mode != "determinate":
                self.cfd_bar.stop()
                self.cfd_bar.configure(mode="determinate")
                self.bar_mode = "determinate"
            self._bar_set(min(it / total, 1.0))
            text = f"solver {it} / {total}"
            if self.cfd_iter_t0 and it > self.cfd_iter_t0[1]:
                ips = (it - self.cfd_iter_t0[1]) / max(time.time() - self.cfd_iter_t0[0], 1e-6)
                text += f"  ·  {ips:.2f} it/s  ·  ETA {hms((total - it) / max(ips, 1e-9))}"
        else:
            text = step or "starting"
        text += f"  ·  {hms(el)}"
        if text != self.cfd_prog.cget("text"):
            self.cfd_prog.configure(text=text)
        blink = int(el * 2) % 2
        if self.step_drawn != step or blink != self.blink:
            self.step_drawn, self.blink = step, blink
            self._draw_stepper()
            self.status.set(("3D CFD stopping: " if self.cfd_stop.is_set() else "3D CFD running: ") + text, "busy")

    def _draw_stepper(self):
        cv = self.stepper
        cv.delete("all")
        w = cv.winfo_width()
        if w < 100:
            return
        names = [s for s, _ in self.STEPS]
        cur = self.cfd_step.replace("reconstructParMesh", "reconstructPar")
        if cur == "finished" or (self.cfd_done_ok and not self.cfd_running):
            idx = len(names)
        else:
            idx = names.index(cur) if cur in names else -1
        n = len(names)
        x0, x1, y = px(30), w - px(30), px(13)
        cv.create_line(x0, y, x1, y, fill=BORDER, width=px(2))
        if idx > 0:
            glow_line(cv, (x0, y, x0 + (x1 - x0) * min(idx, n - 1) / (n - 1), y), CYAN, width=2)
        for i, (_, label) in enumerate(self.STEPS):
            x = x0 + (x1 - x0) * i / (n - 1)
            if i < idx:
                cv.create_oval(x - px(6), y - px(6), x + px(6), y + px(6), fill=CYAN, outline="")
                col = TEXT2
            elif i == idx and not self.cfd_running:          # stopped or failed here
                cv.create_oval(x - px(6), y - px(6), x + px(6), y + px(6), fill=AMBER, outline="")
                col = AMBER
            elif i == idx:
                r = px(9 if self.blink else 7)
                cv.create_oval(x - r, y - r, x + r, y + r, outline=blend(MAGENTA, CARD, 0.4), width=px(2))
                cv.create_oval(x - px(5), y - px(5), x + px(5), y + px(5), fill=MAGENTA, outline="")
                col = MAGENTA
            else:
                cv.create_oval(x - px(5), y - px(5), x + px(5), y + px(5), fill=CARD, outline=BORDER2, width=px(2))
                col = MUTED
            cv.create_text(x, y + px(19), text=label, font=cfont(10, "bold"), fill=col)

    def _cfd_log_lines(self, lines, tag=None, scroll=True):
        t = self.cfd_log
        t.configure(state="normal")
        for ln in lines:
            tg = tag or ("step" if ln.startswith("==> ") else ("err" if re.search(r"ERROR|FATAL", ln) else None))
            t.insert("end", ln + "\n", tg)
        n = int(t.index("end-1c").split(".")[0])
        if n > 5000:            # keep the widget fast on long runs; the full logs are in the case folder
            t.delete("1.0", f"{n - 4000}.0")
        t.configure(state="disabled")
        if scroll:
            t.see("end")
            t.after(60, lambda: t.see("end"))     # again once a banner or relayout has resized the log

    def _cfd_show_result(self, res, final):
        bemt = self.cfd_bemt
        self.cfd_shown = True
        hub = getattr(res, "hub_thrust", None) is not None
        # like for like: the BEMT covers the blades outside its hub cut-off, so the CFD column does too
        bt, bq = (res.blade_thrust, res.blade_torque) if hub else (res.thrust, res.torque)
        bp = res.power * bq / res.torque if abs(res.torque) > 1e-12 else 0.0
        vals = (bt, bq, bp)
        self.cfd_col_hdr.configure(text=spaced("CFD blades" if hub else "3D CFD"))
        for i, (k, fmt) in enumerate((("T", "{:.2f}"), ("Q", "{:.4f}"), ("P", "{:.1f}"))):
            b_, c_, d_ = self.cfd_cells[k]
            b_.configure(text=fmt.format(bemt[i]) if bemt else "—")
            c_.configure(text=fmt.format(vals[i]) if res.ok else "—")
            if bemt and res.ok and abs(bemt[i]) > 1e-12:
                d = (vals[i] - bemt[i]) / abs(bemt[i]) * 100
                d_.configure(text=f"{d:+.1f} %", text_color=GREEN if abs(d) < 10 else (AMBER if abs(d) < 25 else RED))
            else:
                d_.configure(text="—", text_color=TEXT2)
        conv = ("converged" if res.converged else "stopped" if final and self.cfd_stop.is_set()
                else "not converged" if final else "running")
        self.cfd_res_hdr.set_right(f"{res.iterations} it · {conv}", GREEN if res.converged else AMBER)
        notes = list(getattr(res, "notes", []))
        if final and bemt and len(bemt) > 3 and bemt[3] > 0.25:
            notes.append(f"{bemt[3] * 100:.0f} % of the span is stalled in the BEMT, whose 2D polars are pessimistic "
                         "there (rotation delays stall): the CFD is the better estimate")
        self.cfd_msg.configure(text="\n".join([res.message or ""] + [f"• {n}" for n in notes]),
                               text_color=TEXT2 if res.ok else RED)
        if len(res.history) == 1:
            it = res.history[0][0]
            self.hist_chart.set_data(None, empty=f"One force sample so far (iteration {it}); "
                                                 "the history needs two")
        elif res.history:
            h = np.asarray(res.history, float)
            dots = len(h) <= 40
            # zoom on the last 70 % of the iterations: the first samples are far off and would flatten the rest
            yfrom = h[-1, 0] * 0.3 if np.count_nonzero(h[:, 0] >= h[-1, 0] * 0.3) >= 2 else None
            if hub:         # the hub share as averaged (a constant offset over the history)
                h[:, 1] -= res.hub_thrust
                h[:, 2] -= res.hub_torque
            if bemt and abs(bemt[0]) > 1e-12 and abs(bemt[1]) > 1e-12:
                # both as % off the BEMT on one axis: the table's delta column over the iterations, and no
                # two co-rising curves drawn on top of each other (thrust, the key one, is drawn last)
                dT = (h[:, 1] - bemt[0]) / abs(bemt[0]) * 100
                dQ = (h[:, 2] - bemt[1]) / abs(bemt[1]) * 100
                self.hist_chart.set_data(h[:, 0], [dict(y=dQ, label="torque", color=MAGENTA, unit="%", dash=(6, 3),
                                                        dots=dots),
                                                   dict(y=dT, label="thrust", color=CYAN, unit="%", dots=dots)],
                                         hlines=[(0.0, "L", TEXT2, "BEMT")], ylabel=("Δ vs BEMT [%]", "Δ [%]"),
                                         ylabel2="", yfrom=yfrom)
            else:
                self.hist_chart.set_data(h[:, 0], [dict(y=h[:, 1], label="thrust", color=CYAN, axis="L", unit="N",
                                                        dots=dots),
                                                   dict(y=h[:, 2], label="torque", color=MAGENTA, axis="R",
                                                        unit="N·m", dots=dots)],
                                         ylabel=("thrust [N]", "T [N]"), ylabel2=("torque [N·m]", "Q [N·m]"),
                                         yfrom=yfrom)

    def _cfd_finished(self, res, case_dir, err, note):
        if self.cfd_lines:          # the last OpenFOAM lines go before the summary
            self._cfd_log_lines([self.cfd_lines.popleft() for _ in range(len(self.cfd_lines))])
        self.cfd_busy, self.cfd_running = False, False
        self.cfd_bar.stop()
        self.cfd_bar.configure(mode="determinate")
        self.bar_mode = "determinate"
        self.header.set_busy(self.sim_busy)
        self.tabbar.set_badge("cfd", None)
        # a stopped run never reached 'merge', whatever forces it has: the tracker marks the stop
        self.cfd_done_ok = bool(res is not None and res.ok and not err and not self.cfd_stop.is_set())
        self._cfd_buttons()
        el = time.time() - self.cfd_t0
        if err:
            self._bar_set(0)
            self._cfd_log_lines([f"==> {err}"], "err")
            self._show_banner(err, "3D CFD", AMBER if "not found" in err else RED)
            self.status.set(f"3D CFD: {err}", "error")
            self.cfd_prog.configure(text="failed")
        elif res is None:
            self._bar_set(0)
            self._cfd_log_lines([f"==> {note}"], "app")
            self.status.set(f"3D CFD: case written to {case_dir}", "ok")
            self.cfd_prog.configure(text="case written")
        else:
            stopped = self.cfd_stop.is_set()
            self._bar_set(1 if res.ok and not stopped else 0)
            self._cfd_show_result(res, True)
            if res.ok:
                msg = f"CFD thrust {res.thrust:.2f} N, torque {res.torque:.4f} N·m, power {res.power:.1f} W"
                self._cfd_log_lines([f"==> {msg}", f"    {res.message}"], "app")
                self.status.set(f"3D CFD: {msg}" + (" (stopped early)" if stopped else ""), "warn" if stopped else "ok")
            elif stopped:
                self._cfd_log_lines([f"==> stopped: {res.message}"], "app")
                self.status.set("3D CFD stopped before the solver produced forces.", "warn")
                self.cfd_msg.configure(text="Stopped before the solver produced forces.", text_color=AMBER)
            else:
                self._cfd_log_lines([f"==> CFD failed: {res.message}"], "err")
                self.status.set(f"3D CFD failed: {res.message}", "error")
                self._show_banner(res.message or "The OpenFOAM run failed; see the log.", "3D CFD", RED)
            self.cfd_prog.configure(text=("stopped after " if stopped else "done in ") + hms(el))
        self._draw_stepper()
        if self.closing:
            self.root.destroy()

    def _cfd_stop_click(self):
        if self.cfd_running and not self.cfd_stop.is_set():
            self.cfd_stop.set()
            self._cfd_log_lines(["==> stop requested"], "app")
            self.status.set("3D CFD: stopping (the solver writes its current state first)…", "busy")

    def _cfd_load(self):
        if cfd3d is None or self.cfd_busy:
            return
        d = filedialog.askdirectory(title="Open an OpenFOAM case written by PropSim", mustexist=True,
                                    initialdir=os.path.dirname(self.cfd_case[0]) if self.cfd_case else self._start_dir())
        if not d:
            return

        base = self._cfd_aero_argv()

        def work():
            try:
                bemt, how = self._cfd_case_bemt(d, base)
                res = cfd3d.read_results(d, hub_radius=bemt[4] if bemt and bemt[4] else None)
                self._post(self._cfd_loaded, res, d, "", bemt, how)
            except NotImplementedError:
                self._post(self._cfd_loaded, None, d, "Reading CFD results is not implemented in this build.")
            except Exception as e:  # noqa: BLE001
                self._post(self._cfd_loaded, None, d, f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _cfd_aero_argv(self):
        """The aerodynamic model settings of the form (not the model, speed or flight condition), for
        re-running the BEMT of a loaded case (Tk thread)."""
        out = []
        for key, flag, label in (("hub", "--hub-ratio", "Hub cut-off r/R"), ("clmax", "--cl-max", "Cl max"),
                                 ("clalpha", "--cl-alpha", "Lift slope"), ("cd0", "--cd0", "Cd0")):
            try:
                x = self._num(key, label, lo=0)
            except InputError:
                x = None
            if x is not None:
                out += [flag, repr(x)]
        polar = self.v["polar"].get().strip().strip('"')
        if polar and os.path.isfile(polar):
            out += ["--polar", polar]
        if self.v["aero"].get() == "neuralfoil":
            out += ["--aero", "neuralfoil", "--nf-model", self.v["nf_model"].get()]
        return out

    def _cfd_case_bemt(self, d, base):
        """(thrust, torque, power, stalled fraction, hub radius) of the BEMT for a case folder, and how
        it was found: stored with the case when PropSim wrote it, else run now on the case's own model
        (constant/triSurface/prop.stl, in metres along +z) at its rpm, airspeed and air density (worker)."""
        try:
            with open(os.path.join(d, cfd3d.CASE_INFO)) as f:
                info = json.load(f)
        except (OSError, ValueError):
            return None, "no propsim_case.json in the folder: compared with the dashboard's BEMT run"
        b = info.get("bemt")
        if b:
            return (b["thrust"], b["torque"], b["power"], b.get("stalled_fraction", 0.0),
                    b.get("hub_radius", info.get("hub_radius", 0.0))), "the BEMT stored with the case"
        stl = os.path.join(d, "constant", "triSurface", "prop.stl")
        if not os.path.isfile(stl):
            return None, "the case has no constant/triSurface/prop.stl: compared with the dashboard's BEMT run"
        argv = [stl, "--units", "m", "--axis", "z", "--rpm", repr(float(info["rpm"])),
                "--airspeed", repr(float(info.get("airspeed", 0.0))), "--rho", repr(float(info.get("rho", 1.225)))]
        if info.get("blades"):
            argv += ["--blades", str(int(info["blades"]))]
        argv += base
        self._post(self.status.set, "3D CFD: running the BEMT for the loaded case's own model…", "busy")
        try:
            out = self._simulate(argv, lambda m: self._post(self.status.set, "3D CFD: " + m, "busy"))
        except Exception as e:  # noqa: BLE001 - fall back to the dashboard's run, and say why
            traceback.print_exc()
            return None, f"the BEMT could not be run on the case's model ({self._describe(e)}): compared with the " \
                         "dashboard's BEMT run"
        r = out.res
        return ((r.thrust, r.torque, r.power, r.stalled_fraction, float(out.geom.hub_radius)),
                "the BEMT re-run now on the case's own model and conditions (current aerodynamic settings; "
                "features such as serrations are not known to an older case)")

    def _cfd_loaded(self, res, d, err, bemt=None, how=""):
        if err or res is None:
            self._show_banner(err or "No results found.", "3D CFD", RED)
            return
        self.cfd_case = (d, None)
        self.cfd_stop.clear()         # a stop of an earlier run says nothing about these results
        self._cfd_buttons()
        o = self.outcome
        own = bemt is not None
        self.cfd_out = None if own else o
        if own:
            self.cfd_bemt = bemt[:4]
        else:
            self.cfd_bemt = (o.res.thrust, o.res.torque, o.res.power, o.res.stalled_fraction) if o is not None else None
        self.cfd_desc = (f"Loaded case: {os.path.basename(os.path.abspath(d))}"
                         + (f"  ·  {res.rpm:,.0f} rpm" if res.rpm else "") + f"\nBEMT column: {how}")
        self._cfd_show_result(res, True)
        self._cfd_update_op()
        self._cfd_log_lines([f"==> results of {d}: {res.message}", f"    BEMT: {how}"]
                            + [f"    note: {n}" for n in res.notes], "app" if res.ok else "err")
        self.status.set(f"Loaded CFD case {os.path.basename(os.path.abspath(d))}", "ok")
        if not own and self.outcome is not None and res.rpm and abs(res.rpm - self.outcome.res.rpm) > 1:
            self._show_banner(f"The case ran at {res.rpm:,.0f} rpm but the last BEMT run is at "
                              f"{self.outcome.res.rpm:,.0f} rpm, so the comparison is not like for like.",
                              "NOTE", AMBER)

    def _cfd_open_folder(self):
        if not self.cfd_case:
            return
        path = self.cfd_case[0]
        try:
            if os.name == "nt":
                os.startfile(path)  # noqa: S606
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            self._show_banner(f"Could not open {path}: {e}", "NOTE", AMBER)

    # ================================================================== propeller generator
    GEN_GOALS = {"Max thrust": "max_thrust", "Efficiency": "efficiency", "Target thrust": "target_thrust"}

    def _build_gen(self, page):
        self.gen = None               # (PropDesign, verify dict) shown
        self.gen_busy = ""            # "", "design" or "step"
        self.gen_t0 = 0.0
        self.gen_cmp = None           # (ranking rows, spec) of the last "Test all airfoils"
        self.gen_cancel = threading.Event()
        self.gen_airfoils = ({v["label"]: k for k, v in propgen.AIRFOILS.items()} if propgen
                             else {"Clark-Y": "clarky"})
        for k, val in dict(gen_goal="Max thrust", gen_src="Motor (left panel)", gen_target="", gen_rpm="",
                           gen_pshaft="", gen_blades="2", gen_cw="Solid (prop material)", gen_dens="1.24",
                           gen_sup="Generic", gen_mat="PLA", gen_proc="FDM 3D print", gen_uts="60", gen_mod="3.5", gen_sf="2",
                           gen_alt="30", gen_pitch="", gen_pmode="At 75 % R",
                           gen_sweep="", gen_sweep0="0.35", gen_serr="", gen_serr_w="15", gen_serr_span="0.5-0.95",
                           gen_tub="", gen_tub_w="30", gen_tub_span="0.3-0.9", gen_tip="None",
                           gen_wl_h="6", gen_wl_cant="80", gen_wl_dir="Up", gen_rake="45", gen_rake_len="12",
                           gen_dmax="254", gen_dfix="", gen_mach="0.6",
                           gen_airfoil=next(iter(self.gen_airfoils)), gen_spin="CCW", gen_bore="5",
                           gen_hub="", gen_hub_h="", gen_edge="Sharp", gen_edge_mm="1", gen_spinner="None",
                           gen_spinner_len="", gen_pocket="", gen_tol="0", gen_bore_dep="", gen_hang="", gen_dflat="", gen_setscrew="",
                           gen_hex="", gen_hex_d="3", gen_holes="", gen_hole_d="3.2", gen_pcd="16",
                           gen_head="None", gen_head_d="6", gen_head_dep="3", gen_fillet="", gen_offset="0",
                           gen_step="Exact").items():
            self.v[k] = tk.StringVar(value=val)
        page.columnconfigure(0, weight=0)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)

        # --- design brief (left), a fixed width so the results get the room
        hold = tk.Frame(page, bg=BG, width=px(330))
        hold.grid(row=0, column=0, sticky="ns")
        hold.grid_propagate(False)
        hold.rowconfigure(0, weight=1)
        hold.columnconfigure(0, weight=1)
        card = ctk.CTkFrame(hold, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, sticky="nsew", padx=px(4), pady=px(4))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        hdr = CardHeader(card, "Design brief", "A", accent=GREEN)
        hdr.grid(row=0, column=0, sticky="ew", padx=px(3), pady=(px(3), 0))
        self.gen_sc = sc = ctk.CTkScrollableFrame(card, fg_color=CARD, corner_radius=0, scrollbar_button_color=BORDER,
                                                  scrollbar_button_hover_color=BORDER2)
        sc.grid(row=1, column=0, sticky="nsew", padx=(px(8), px(2)))
        b = ctk.CTkFrame(sc, fg_color="transparent")
        b.pack(fill="both", expand=True, padx=(px(4), px(6)))
        b.columnconfigure(1, weight=1)
        r = 0
        self._label(b, "Goal", "Max thrust: the prop that turns the available power into the most thrust. "
                    "Efficiency: most thrust per watt (best propulsive efficiency in forward flight). Target "
                    "thrust: the prop that makes the thrust you enter with the least power.").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        self._seg(b, "gen_goal", list(self.GEN_GOALS), command=lambda _v: self._gen_fields()).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(px(3), px(8)))
        r += 1
        self._label(b, "Power from", "Motor: uses KV, motor power, battery, Rm/I0 and ESC efficiency from "
                    "card 05 on the left. RPM + power: a fixed shaft operating point.").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        self._seg(b, "gen_src", ["Motor (left panel)", "RPM + power"], command=lambda _v: self._gen_fields()).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(px(3), px(4)))
        r += 1
        self.gen_src_info = ctk.CTkLabel(b, text="", font=font(11), text_color=MUTED, anchor="w", justify="left",
                                         wraplength=290)
        self.gen_src_info.grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, px(4)))
        r += 1
        self.gen_rows = {}

        def row(key, label, unit, tip, rr):
            lab = self._label(b, label, tip)
            lab.grid(row=rr, column=0, sticky="w", pady=px(3))
            prev = self.entries.get(key)
            e = self._entry(b, key, width=88)
            if prev is not None:          # a second view of a left-panel field: errors still point there
                self.entries[key] = prev
            e.grid(row=rr, column=1, sticky="e", pady=px(3))
            u = self._unit(b, unit)
            u.grid(row=rr, column=2, sticky="w", padx=(px(8), 0))
            self.gen_rows[key] = (lab, e, u)
            return lab, e, u

        for key, label, unit, tip in (
                ("gen_target", "Target thrust", "N", "Thrust the prop must make (Target thrust goal)."),
                ("gen_rpm", "Speed", "rpm", "Shaft speed. Blank = PropSim picks the best speed under the tip "
                 "Mach limit."),
                ("gen_pshaft", "Shaft power", "W", "Power available at the shaft."),
                ("gen_dmax", "Max diameter", "mm", "Largest prop that fits (frame, ground clearance). With "
                 "winglets this is the overall diameter, winglets included."),
                ("gen_dfix", "Fixed diameter", "mm · auto", "Blank = PropSim picks the best diameter up to the "
                 "max."),
                ("gen_pitch", "Pitch", "mm · auto", "Blank = the optimum blade angles. Or set the pitch you want, "
                 "in mm or with 'in' (e.g. 6in or 6\"): the blade is twisted to it and the chord adapts so the "
                 "power is still absorbed. The report says what it costs against the optimum.")):
            row(key, label, unit, tip, r)
            r += 1
        self._label(b, "Pitch mode", "At 75 % R: the set pitch is the pitch at 75 % of the radius (how props are "
                    "labelled), with the optimum twist shape along the blade. Constant: the same geometric "
                    "pitch from root to tip (a helical blade, like many simple props).").grid(
            row=r, column=0, sticky="w", pady=px(3))
        self._seg(b, "gen_pmode", ["At 75 % R", "Constant"]).grid(row=r, column=1, columnspan=2, sticky="ew",
                                                                  pady=px(3))
        r += 1
        for key, label, unit, tip in (
                ("gen_mach", "Max tip Mach", "", "Helical tip speed limit as a Mach number, up to "
                 f"{propgen.MAX_TIP_MACH if propgen else 0.98:g}. 0.6 keeps noise and compressibility losses "
                 "low. Above about 0.75 the tips go transonic: PropSim thins them and adds wave drag (an "
                 "engineering estimate). Above 1 the tips are supersonic: experimental, extremely loud; use "
                 "the Experimental (high speed) airfoils. Blade sweep (below) delays the transonic drag rise."),
                ("gen_bore", "Shaft bore", "mm", "Hole through the hub for the motor shaft or prop adapter."),
                ("gen_hub", "Hub diameter", "mm · auto", "Blank = sized from the bore and the diameter (and big "
                 "enough for the mounting holes or hex pocket). A set diameter is used as is: holes that do not fit "
                 "it are left out, with a HUB note."),
                ("airspeed", "Airspeed", "m/s", "Design airspeed (forward flight); 0 = static thrust (hover). "
                 "The same setting as card 04 on the left: changing one changes both."),
                ("rho", "Air density", "kg/m³", "1.225 at sea level, ~1.11 at 1000 m, ~1.0 at 2000 m. The same "
                 "setting as card 04 on the left.")):
            row(key, label, unit, tip, r)
            r += 1
        self._label(b, "Blades", "1 = a single blade balanced by a counterweight opposite it: the least "
                    "blade interference and tip loss for its blade area, used on some gliders and record "
                    "models. 2+ blades usually give more thrust for the same diameter.").grid(
            row=r, column=0, sticky="w", pady=px(3))
        self._seg(b, "gen_blades", ["1", "2", "3", "4", "5"], command=lambda _v: self._gen_cw_rows()).grid(
            row=r, column=1, columnspan=2, sticky="ew", pady=px(3))
        r += 1
        # single blade: the counterweight (shown only for 1 blade)
        cw_lab = self._label(b, "Counterweight", "Balances the single blade, statically and dynamically "
                             "(its centre of mass sits opposite the blade's). Solid: a pod of the prop's own "
                             "material (moulded, machined or printed with it). A metal rod insert keeps the pod "
                             "small: a hole is left for a standard rod cut to the length shown, then pressed in, "
                             "bonded or moulded in.")
        cw_opt = self._option(b, "gen_cw", list(self.GEN_CW), width=150)
        cw_lab.grid(row=r, column=0, sticky="w", pady=px(3))
        cw_opt.grid(row=r, column=1, columnspan=2, sticky="e", pady=px(3))
        self.gen_cw_widgets = (cw_lab, cw_opt)
        self._gen_cw_rows()
        r += 1
        self._label(b, "Airfoil", "Section shape; it gets thicker toward the root for strength.").grid(
            row=r, column=0, sticky="w", pady=px(3))
        self._option(b, "gen_airfoil", list(self.gen_airfoils), width=150).grid(
            row=r, column=1, columnspan=2, sticky="e", pady=px(3))
        r += 1
        self.gen_af_info = ctk.CTkLabel(b, text="", font=font(11), text_color=MUTED, anchor="w", justify="left",
                                        wraplength=290)
        self.gen_af_info.grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, px(4)))
        self.v["gen_airfoil"].trace_add("write", lambda *_: self._gen_af_info())
        self._gen_af_info()
        r += 1
        self._label(b, "Spin", "Rotation seen from the front (the side the thrust points to). Most "
                    "motors' 'normal' props are CCW.").grid(row=r, column=0, sticky="w", pady=px(3))
        self._seg(b, "gen_spin", ["CCW", "CW"]).grid(row=r, column=1, columnspan=2, sticky="ew", pady=px(3))
        r += 1
        # ---- hub
        ctk.CTkLabel(b, text="HUB", font=font(11, "bold"), text_color=CYAN, anchor="w").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(px(10), px(2)))
        r += 1
        self.gen_hub_widgets = {}

        def hrow(key, label, unit, tip, group=None):
            nonlocal r
            w_ = row(key, label, unit, tip, r)
            if group:
                self.gen_hub_widgets.setdefault(group, []).extend(w_)
            r += 1

        def hseg(key, label, values, tip, group=None):
            nonlocal r
            lab = self._label(b, label, tip)
            lab.grid(row=r, column=0, sticky="w", pady=px(3))
            sg = self._seg(b, key, values, command=lambda _v: self._gen_hub_rows())
            sg.grid(row=r, column=1, columnspan=2, sticky="ew", pady=px(3))
            if group:
                self.gen_hub_widgets.setdefault(group, []).extend([lab, sg])
            r += 1

        hrow("gen_hub_h", "Hub height", "mm · auto", "Blank = sized from the diameter (and grown for the root "
             "fillet, the blade offset and the edges, or for strength).")
        hseg("gen_edge", "Hub edges", ["Sharp", "Chamfer", "Round"], "The hub's outer top and bottom edges: "
             "sharp, chamfered or rounded.")
        hrow("gen_edge_mm", "Edge size", "mm", "Chamfer size or edge radius.", "edge")
        hseg("gen_spinner", "Spinner", ["None", "Cone", "Dome", "Ogive"], "A streamlined nose on the front "
             "(thrust side) of the hub. In fast forward flight it removes most of the bare hub's drag (the 3D CFD "
             "shows it). Without a pocket the bore ends under it.")
        hrow("gen_spinner_len", "Spinner length", "mm · auto", "Blank = from the hub size (dome 1, cone 1.6, "
             "ogive 2 hub radii).", "spinner")
        hrow("gen_pocket", "Spinner pocket", "mm · none", "Diameter of a pocket inside the spinner for the end "
             "of the shaft or a prop nut (the bore then continues into it).", "spinner")
        hrow("gen_tol", "Bore tolerance", "mm", "Added to the shaft bore: + for a clearance fit (FDM holes "
             "print about 0.1-0.3 mm small), - for a press fit.")
        hrow("gen_bore_dep", "Bore depth", "mm · through", "Blank = the bore goes all the way through. A depth "
             "makes a blind bore from the bottom (the motor side), e.g. 40 for a shaft that should not come out at "
             "the front. The D-flat follows it, and the set screw moves to the middle of a short bore.")
        hrow("gen_dflat", "D-flat", "mm · none", "Depth of a flat on the bore for a D-shaft (0.5 on a 5 mm "
             "shaft is common). It faces the set screw.")
        hrow("gen_setscrew", "Set screw", "mm · none", "Diameter of a radial hole at mid-height, between two "
             "blades, for a set screw (tap drill size, e.g. 2.5 for M3).")
        hrow("gen_hex", "Nut pocket", "mm AF · none", "Hex pocket in the front face for a prop nut, across "
             "flats (e.g. 8 for an M5 nut).")
        hrow("gen_hex_d", "Pocket depth", "mm", "Depth of the hex pocket.", "hex")
        hrow("gen_holes", "Mount holes", "· none", "Number of mounting holes on a bolt circle (e.g. 4 for "
             "motors with a threaded bell and no shaft). With a spinner they go on through it, wide enough "
             "for the screw heads. An auto hub grows to hold them; a hub diameter you set must be at least "
             "the bolt circle + the hole (or head recess) + 2 mm, or they are left out with a HUB note.")
        hrow("gen_hole_d", "Hole size", "mm", "Hole diameter (3.2 = M3 clearance, 2.2 = M2).", "holes")
        hrow("gen_pcd", "Bolt circle", "mm", "Diameter of the circle through the hole centres (motors: 12, 16, "
             "19 mm are common).", "holes")
        hrow("gen_hang", "Bolt angle", "° · auto", "Rotates the bolt circle: the angle of the first hole from "
             "blade 1, counter-clockwise seen from the front (for a CW prop, mirrored with the prop). Blank = half "
             "the hole spacing from the blade (4 holes: 45°, 3 holes: 60°, 2 holes: 90°).", "holes")
        hseg("gen_head", "Screw heads", ["None", "Countersink", "Counterbore"], "A recess on the front for "
             "the screw heads.", "holes")
        hrow("gen_head_d", "Head size", "mm", "Diameter of the head recess.", "head")
        hrow("gen_head_dep", "Head depth", "mm", "Depth of the counterbore.", "cbore")
        hrow("gen_fillet", "Root fillet", "mm · none", "A rounded fillet where each blade meets the hub: "
             "less stress concentration at the root and a stronger, cleaner print.")
        hrow("gen_offset", "Blade offset", "mm", "Moves the blades along the axis: + towards the thrust side "
             "(front), - towards the motor.")
        self.v["gen_holes"].trace_add("write", lambda *_: self._gen_hub_rows())
        self.v["gen_hex"].trace_add("write", lambda *_: self._gen_hub_rows())
        self._gen_hub_rows()
        hseg("gen_step", "STEP surface", ["Exact", "Smooth"], "Exact: every blade is a B-spline surface "
             "through every point of the STL's surface, so the STEP keeps all its details (tubercles, teeth, "
             "winglets). Smooth: the same from a coarser grid, a lighter file with slightly smoother surfaces. "
             "Both are one closed solid.")
        # ---- material, manufacturing, strength
        ctk.CTkLabel(b, text="MATERIAL & STRENGTH", font=font(11, "bold"), text_color=GREEN, anchor="w").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(px(10), px(2)))
        r += 1
        self._label(b, "Supplier", "Whose catalogue to pick from. Generic: typical grades of the common "
                    "materials. JLC3DP: its SLA resins, MJF / SLS nylons, FDM plastics, full-colour WJP resin and "
                    "SLM / binder-jet metals. PCBWay: its resins, nylons, FDM plastics, SLM metals and the CNC "
                    "metals and plastics. SendCutSend: its laser / waterjet / router-cut sheets (flat parts "
                    "only) and CNC-machined 6061. The Process and Material lists then show only what that "
                    "supplier offers (for Generic, the materials usually made by the chosen process). Values are the suppliers' published data where they give them, otherwise "
                    "typical values for the grade: check the datasheet for anything critical.").grid(
            row=r, column=0, sticky="w", pady=px(3))
        self._option(b, "gen_sup", list(propgen.structure.SUPPLIER_NAMES) if propgen else ["Generic"],
                     width=150).grid(row=r, column=1, columnspan=2, sticky="e", pady=px(3))
        r += 1
        self._label(b, "Process", "How it is made. Each process scales the material's strength and fatigue "
                    "strength: FDM printing about 0.65 (printed flat, 100 % infill), resin and powder-bed "
                    "printing 0.9, material jetting (WJP / PolyJet) 0.85, injection moulding 0.9 (knit lines, "
                    "voids), CNC machining 1.0, composite layup 0.9, metal printing (SLM/DMLS) 0.95 with about "
                    "half the fatigue strength as built (porosity, rough surface), metal binder jet 0.9 with half "
                    "the fatigue strength. Sheet cutting (laser, waterjet, router) makes flat parts only: a "
                    "twisted blade cannot be cut from sheet.").grid(row=r, column=0, sticky="w", pady=px(3))
        self.gen_proc_menu = self._option(b, "gen_proc", list(propgen.PROCESSES) if propgen else ["FDM 3D print"],
                                          width=150)
        self.gen_proc_menu.grid(row=r, column=1, columnspan=2, sticky="e", pady=px(3))
        r += 1
        self._label(b, "Material", "What the prop is made of: sets the density (centrifugal load, counterweight) "
                    "and the strength the blade is checked against. Generic: printing plastics, "
                    "injection-moulding nylons (plain, glass or carbon filled), POM, PC, PEEK, resin, composites, "
                    "wood and boards, aluminium, titanium, steels, stainless, tool steels, brass, copper. "
                    "Pick a Supplier to see its own grades (for example JLC3DP PA12-HP or PCBWay Somos Taurus). "
                    "Rubbers, foams and gasket sheets are listed because the sheet suppliers cut them, but they "
                    "are not blade materials. Edit density, strength or modulus for your exact grade (the "
                    "material then shows Custom).").grid(row=r, column=0, sticky="w", pady=px(3))
        self.gen_mat_menu = self._option(b, "gen_mat", list(self.GEN_MATERIALS) + ["Custom"], width=150)
        self.gen_mat_menu.grid(row=r, column=1, columnspan=2, sticky="e", pady=px(3))
        r += 1
        for key, label, unit, tip in (
                ("gen_dens", "Density", "g/cm³", "Density of the prop material. The counterweight balance assumes a "
                 "fully solid part: if you 3D print it, use 100 % infill."),
                ("gen_uts", "Strength", "MPa", "Tensile strength of the moulded / bulk material (the process "
                 "factor is applied on top)."),
                ("gen_mod", "Modulus", "GPa", "Young's modulus: sets how far the blade bends and twists, and the "
                 "centrifugal relief of the bending."),
                ("gen_sf", "Safety factor", "", "Design safety factor on the strength (static and fatigue). 2 is "
                 "a sensible minimum for model props; use 3+ for big, fast or people-carrying props, or "
                 "untested materials."),
                ("gen_alt", "Vibration load", "%", "Alternating load for the fatigue check, as a percentage of the "
                 "aerodynamic bending: 30 for a prop in axial flow; more for edgewise / forward flight on a "
                 "multirotor, gusts or a vibrating mount. 0 = static check only.")):
            row(key, label, unit, tip, r)
            r += 1
        self.gen_respect = tk.BooleanVar(value=True)
        sw = self._switch(b, "Design within strength limits", self.gen_respect)
        sw.grid(row=r, column=0, columnspan=3, sticky="w", pady=(px(4), px(2)))
        Tooltip(sw, "On: under-strength sections are thickened (up to 30 % t/c on the blade, 55 % at the root), "
                    "the hub grows if the root needs it, and if that is still not enough a slower or smaller prop "
                    "is chosen. Off: the blade is designed for aerodynamics only and the stresses are just "
                    "reported.")
        r += 1
        self.v["gen_mat"].trace_add("write", lambda *_: self._gen_material())
        self.v["gen_sup"].trace_add("write", lambda *_: self._gen_supplier())
        self.v["gen_proc"].trace_add("write", lambda *_: self._gen_process())
        self._gen_supplier()
        for k_ in ("gen_dens", "gen_uts", "gen_mod"):
            self.v[k_].trace_add("write", lambda *_: self._gen_density_typed())
        # ---- experimental blade features
        ctk.CTkLabel(b, text="EXPERIMENTAL FEATURES", font=font(11, "bold"), text_color=AMBER, anchor="w").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(px(10), px(2)))
        r += 1
        for key, label, unit, tip in (
                ("gen_sweep", "Sweep", "° · off", "Scimitar blade: the blade curves back (against the rotation), "
                 "growing to this sweep angle at the tip. It lowers the Mach number the sections see, so it "
                 "delays the transonic drag rise of fast tips (simple sweep theory). It also lowers the lift "
                 "slope, and swept blades bend and twist more under load. Negative = forward sweep. Blank = none."),
                ("gen_sweep0", "Sweep from", "r/R", "Where the sweep begins (0.35 = 35 % of the radius)."),
                ("gen_serr", "Serrations", "% c · off", "Serrated (sawtooth) trailing edge: tooth depth as a "
                 "percentage of the local chord (10-25 is typical). Mainly a noise feature: up to several dB less "
                 "trailing-edge noise; no high-speed benefit, a little more drag (the blade is widened to keep "
                 "its lift). Blank = none."),
                ("gen_serr_w", "Tooth width", "% c", "Spanwise width of one tooth, as a percentage of the local "
                 "chord. Narrow, deep teeth (width about the depth or less) reduce noise most."),
                ("gen_serr_span", "Teeth span", "r/R", "Where the teeth run, as from-to fractions of the radius, "
                 "e.g. 0.5-0.95."),
                ("gen_tub", "Tubercles", "% c · off", "Leading-edge tubercles (humpback-whale bumps): amplitude as "
                 "a percentage of the local chord (2-8 is typical). They make stall gentle, which helps a prop "
                 "working at high angles of attack (e.g. a fast prop at static thrust); a little more drag at "
                 "the design point. Blank = none."),
                ("gen_tub_w", "Wavelength", "% c", "Spacing of the tubercles as a percentage of the local chord "
                 "(25-50 is typical)."),
                ("gen_tub_span", "Tubercle span", "r/R", "Where the tubercles run, e.g. 0.3-0.9.")):
            row(key, label, unit, tip, r)
            r += 1
        self._label(b, "Tip", "Winglet: the blade tip turns up (towards the thrust side) or down; it reduces "
                    "the tip loss a little but adds its own drag (within a fixed overall diameter a plain "
                    "longer blade usually does as well). Raked tip: the outer blade sweeps back strongly; "
                    "modelled as sweep.").grid(row=r, column=0, sticky="w", pady=px(3))
        self._seg(b, "gen_tip", ["None", "Winglet", "Raked"], command=lambda _v: self._gen_tip_rows()).grid(
            row=r, column=1, columnspan=2, sticky="ew", pady=px(3))
        r += 1
        self.gen_tip_widgets = {"Winglet": [], "Raked": []}
        for key, label, unit, tip, kind in (
                ("gen_wl_h", "Winglet length", "% R", "Length of the winglet along its span, as a percentage "
                 "of the tip radius (4-10 is typical).", "Winglet"),
                ("gen_wl_cant", "Winglet cant", "°", "Angle between the winglet and the blade: 90 = at right "
                 "angles, smaller = leaning outwards.", "Winglet"),
                ("gen_rake", "Rake angle", "°", "Extra sweep of the raked tip.", "Raked"),
                ("gen_rake_len", "Rake length", "% R", "Length of the raked part, as a percentage of the radius.",
                 "Raked")):
            self.gen_tip_widgets[kind].extend(row(key, label, unit, tip, r))
            r += 1
        wl_lab = self._label(b, "Winglet side", "Up: towards the thrust side (+Z, the blade's suction side). "
                             "Down: towards the motor.")
        wl_lab.grid(row=r, column=0, sticky="w", pady=px(3))
        wl_seg = self._seg(b, "gen_wl_dir", ["Up", "Down"])
        wl_seg.grid(row=r, column=1, columnspan=2, sticky="ew", pady=px(3))
        self.gen_tip_widgets["Winglet"] += [wl_lab, wl_seg]
        r += 1
        self._gen_tip_rows()
        ctk.CTkLabel(b, text="Features are built into the model and their effect is a rough engineering estimate "
                     "(see the notes after Generate). The aerodynamics choice comes from card 03 on the left.",
                     font=font(11), text_color=MUTED, anchor="w", justify="left", wraplength=290).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(px(6), 0))
        foot = ctk.CTkFrame(card, fg_color="transparent")
        foot.grid(row=2, column=0, sticky="ew", padx=px(12), pady=(px(6), px(12)))
        foot.columnconfigure((0, 1), weight=1, uniform="genb")
        self.gen_btn = GlowButton(foot, "Generate", self._gen_run, height=46, stops=(GREEN, CYAN), bg=CARD)
        self.gen_btn.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.gen_cmp_btn = self._button(foot, f"Test all {len(self.gen_airfoils)} airfoils", self._gen_compare,
                                        accent=GREEN, primary=True)
        self.gen_cmp_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(px(8), 0))
        Tooltip(self.gen_cmp_btn, "Designs the ideal prop for this brief with every airfoil in the library "
                "(each gets its own best diameter, rpm, chord and twist), ranks them by the goal, then "
                "generates and checks the winner. Uses all CPU cores. Press again to stop.")
        self.gen_stl_btn = self._button(foot, "Export STL…", lambda: self._gen_export("stl"))
        self.gen_stl_btn.grid(row=2, column=0, sticky="ew", padx=(0, px(4)), pady=(px(8), 0))
        self.gen_step_btn = self._button(foot, "Export STEP…", lambda: self._gen_export("step"))
        self.gen_step_btn.grid(row=2, column=1, sticky="ew", padx=(px(4), 0), pady=(px(8), 0))
        self.gen_use_btn = self._button(foot, "Analyse it in PropSim  →", self._gen_use, accent=VIOLET)
        self.gen_use_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(px(8), 0))

        # --- results (right)
        right = tk.Frame(page, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(px(4), 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self.gen_strip = ModelStrip(right)
        self.gen_strip.grid(row=0, column=0, sticky="ew", pady=(0, px(6)))
        tiles = tk.Frame(right, bg=BG)
        tiles.grid(row=1, column=0, sticky="ew")
        for c in range(4):
            tiles.columnconfigure(c, weight=1, uniform="gtiles")
        self.gen_tiles = {}
        for key, label, rr, c, span, col, hero in (
                ("size", "Propeller", 0, 0, 1, GREEN, False), ("rpm", "Speed", 0, 1, 1, CYAN, False),
                ("strength", "Strength", 0, 2, 1, AMBER, False), ("check", "Model check", 0, 3, 1, GREEN, False),
                ("thrust", "Thrust", 1, 0, 1, CYAN, False), ("power", "Shaft power", 1, 1, 1, MAGENTA, False),
                ("eff", "Figure of merit", 1, 2, 1, GREEN, False), ("batt", "Battery", 1, 3, 1, VIOLET, False)):
            t = KpiTile(tiles, label, col, hero)
            t.grid(row=rr, column=c, columnspan=span, sticky="ew", padx=px(4), pady=px(4))
            self.gen_tiles[key] = t
        lower = tk.Frame(right, bg=BG)
        lower.grid(row=2, column=0, sticky="nsew", pady=(px(6), 0))
        lower.columnconfigure(0, weight=6, uniform="glow")
        lower.columnconfigure(1, weight=5, uniform="glow")
        lower.rowconfigure((0, 1), weight=1, uniform="glowr")
        self.gen_view = PropView(lower)
        self.gen_view.empty_text = "Awaiting design"
        self.gen_view.grid(row=0, column=0, sticky="nsew", padx=(px(4), px(6)), pady=px(4))
        self.gen_shape = LineChart(lower, "Blade shape", "r / R", ("chord [mm]", "c [mm]"),
                                   ("blade angle [°]", "β [°]"), empty="Generate a propeller to see its blade")
        self.gen_shape.grid(row=0, column=1, sticky="nsew", padx=(px(6), px(4)), pady=px(4))
        scan_box = tk.Frame(lower, bg=BG)
        scan_box.grid(row=1, column=1, sticky="nsew", padx=(px(6), px(4)), pady=px(4))
        scan_box.columnconfigure(0, weight=1)
        scan_box.rowconfigure(1, weight=1)
        self.gen_chart_kind = tk.StringVar(value="Diameter scan")
        seg = ctk.CTkSegmentedButton(
            scan_box, values=["Diameter scan", "Vibration"], variable=self.gen_chart_kind,
            command=lambda _v: self._gen_chart(), height=24, font=font(11, "bold"), fg_color=INPUT,
            selected_color=CYAN_D, selected_hover_color="#0ba7c7", unselected_color=INPUT,
            unselected_hover_color=CARD2, text_color=TEXT, text_color_disabled=MUTED, corner_radius=7)
        seg.grid(row=0, column=0, sticky="e", pady=(0, px(3)))
        for btn in getattr(seg, "_buttons_dict", {}).values():     # the segmented button itself takes no bindings
            Tooltip(btn, "Diameter scan: the diameters the generator tried. Vibration (Campbell diagram): the "
                         "blade's natural frequencies against rotation speed, with the 1P-4P excitation lines; "
                         "where a mode line crosses an excitation line at the operating speed the blade resonates.")
        self.gen_scan = LineChart(scan_box, "Diameter scan", "diameter [mm]", ("thrust [N]", "T [N]"),
                                  ("efficiency", "η"), empty="The diameters tried appear here",
                                  xfmt=lambda v, s: f"{v:,.0f}")
        self.gen_scan.grid(row=1, column=0, sticky="nsew")
        self._gen_chart_d = None
        self.gen_text = ctk.CTkTextbox(lower, font=font(11, mono=True), wrap="none", fg_color=CARD, text_color=TEXT2,
                                       border_width=1, border_color=BORDER, corner_radius=12,
                                       scrollbar_button_color=BORDER, scrollbar_button_hover_color=BORDER2)
        self.gen_text.grid(row=1, column=0, sticky="nsew", padx=(px(4), px(6)), pady=px(4))
        self.gen_text.tag_config("warn", foreground=AMBER)
        self.gen_text.tag_config("ok", foreground=GREEN)
        self._gen_set_text([("Design a propeller for your setup:\nset the goal and limits, then press "
                             "Generate.\n\nPropSim works out the best blade\n(minimum induced loss, each section "
                             "at its\nbest lift-to-drag), then checks it by slicing\nthe generated 3D model and "
                             "analysing it\nlike any other prop. Export it as STL or\nSTEP for manufacturing or CAD.", None)])
        if propgen is None:
            self._gen_set_text([(f"The propeller generator could not be loaded ({GEN_IMPORT_ERROR}).", "warn")])
        self.gen_strip.set_empty("No design yet")
        for k in ("gen_goal", "gen_src", "kv", "power", "cells", "cellv", "voltage", "qmax", "airspeed"):
            self.v[k].trace_add("write", lambda *_: self._gen_fields())
        self._gen_fields()
        self._gen_buttons()

    def _gen_set_text(self, parts):
        self.gen_text.configure(state="normal")
        self.gen_text.delete("1.0", "end")
        for txt, tag in parts:
            self.gen_text.insert("end", txt, tag) if tag else self.gen_text.insert("end", txt)
        self.gen_text.configure(state="disabled")

    def _gen_fields(self):
        goal = self.GEN_GOALS.get(self.v["gen_goal"].get(), "max_thrust")
        motor = self.v["gen_src"].get().startswith("Motor")
        show = {"gen_target": goal == "target_thrust", "gen_rpm": not motor,
                "gen_pshaft": not motor and goal != "target_thrust"}
        for key, on in show.items():
            for w in self.gen_rows[key]:
                w.grid() if on else w.grid_remove()
        if motor:
            g = {k: self.v[k].get().strip() for k in ("kv", "power", "cells", "cellv", "voltage", "qmax", "airspeed")}
            def num(s):
                try:
                    return float(s.replace(",", "."))
                except ValueError:
                    return None
            nc, cv = num(g["cells"]), num(g["cellv"] or "3.7")
            batt = (f"{g['voltage']} V battery" if g["voltage"] else
                    f"{g['cells']}S" + (f" ({nc * cv:.1f} V)" if nc and cv else "") if g["cells"] else "no battery set")
            txt = (f"Motor {g['kv'] or '?'} KV · {g['power'] or '?'} W rated"
                   + (f" · max {g['qmax']} N·m" if g["qmax"] else "") + f" · {batt}"
                   if g["kv"] or g["power"] else "Set the motor KV, rated power and battery in card 05 on the left.")
        else:
            txt = "A fixed shaft operating point (no motor model)."
        try:
            v = float(self.v["airspeed"].get().replace(",", ".") or 0)
        except ValueError:
            v = 0.0
        self.gen_src_info.configure(text=txt + ("  ·  static (hover)" if v == 0 else f"  ·  designed for {v:g} m/s"))

    GEN_MATERIALS = dict(propgen.MATERIALS) if propgen else {"PLA": 1.24}

    def _gen_mat_values(self, m):
        import structure as st_mod
        v = st_mod.MATERIAL_DB[m]
        return {"gen_dens": f"{v[0]:g}", "gen_uts": f"{v[2]:g}", "gen_mod": f"{v[1]:g}"}

    def _gen_material(self):
        m = self.v["gen_mat"].get()
        if m in self.GEN_MATERIALS:
            self._setting_density = True
            for k, val in self._gen_mat_values(m).items():
                self.v[k].set(val)
            self._setting_density = False
            import structure as st_mod
            procs = st_mod.MATERIAL_DB[m][5]
            sup = self.v["gen_sup"].get() if "gen_sup" in self.v else "Generic"
            if sup in st_mod.SUPPLIERS:
                procs = [p for p in st_mod.supplier_processes(sup) if m in st_mod.SUPPLIERS[sup][p]] or procs
            if self.v["gen_proc"].get() not in procs:
                self._gen_in_material = True       # keep the material while the process follows it
                try:
                    self.v["gen_proc"].set(procs[0])
                finally:
                    self._gen_in_material = False

    def _gen_supplier(self):
        """Supplier picked: limit the process list to what it offers, then the material list."""
        st_mod = getattr(propgen, "structure", None) if propgen else None
        if st_mod is None or not hasattr(self, "gen_proc_menu"):
            return
        sup = self.v["gen_sup"].get()
        procs = st_mod.supplier_processes(sup)
        self.gen_proc_menu.configure(values=procs)
        mat, proc = self.v["gen_mat"].get(), self.v["gen_proc"].get()
        own = [p for p in procs if mat in st_mod.supplier_materials(sup, p)]   # keep the material if offered
        want = proc if (proc in own or (not own and proc in procs)) else (own or procs)[0]
        if want != proc:
            self.v["gen_proc"].set(want)     # the trace refreshes the materials
        else:
            self._gen_process()

    def _gen_process(self):
        """Process (or supplier) changed: show the supplier's materials for it; keep the material if offered."""
        st_mod = getattr(propgen, "structure", None) if propgen else None
        if st_mod is None or not hasattr(self, "gen_mat_menu"):
            return
        sup, proc = self.v["gen_sup"].get(), self.v["gen_proc"].get()
        mats = st_mod.supplier_materials(sup, proc)       # Generic: the materials usually made this way
        self.gen_mat_menu.configure(values=mats + ["Custom"])
        m = self.v["gen_mat"].get()
        if sup != "Generic" and m not in mats and mats and not getattr(self, "_gen_in_material", False):
            self.v["gen_mat"].set(mats[0])

    def _gen_density_typed(self):
        if getattr(self, "_setting_density", False):
            return
        m = self.v["gen_mat"].get()
        if m == "Custom" or m not in self.GEN_MATERIALS:
            return
        ref = self._gen_mat_values(m)
        try:
            same = all(abs(float(self.v[k].get().replace(",", ".")) - float(ref[k])) < 1e-9 for k in ref)
        except ValueError:
            same = False
        if not same:
            self.v["gen_mat"].set("Custom")

    def _gen_tip_rows(self):
        t = self.v["gen_tip"].get()
        for kind, ws in self.gen_tip_widgets.items():
            for w in ws:
                w.grid() if kind == t else w.grid_remove()

    def _gen_span(self, key, label):
        s = self.v[key].get().strip().replace(" ", "").replace("–", "-")
        m = re.fullmatch(r"([0-9]*[.,]?[0-9]+)-([0-9]*[.,]?[0-9]+)", s)
        if not m:
            raise InputError(key, f"{label}: type from-to fractions of the radius, e.g. 0.5-0.95.")
        a, b = (float(x.replace(",", ".")) for x in m.groups())
        if not 0.1 <= a < b <= 1.0:
            raise InputError(key, f"{label}: from 0.1 to 1.0, the first number smaller (got {s}).")
        return a, b

    def _gen_features(self):
        """propgen BladeFeatures from the form; raises InputError naming the field to fix."""
        import features as feat_mod
        f = feat_mod.BladeFeatures()
        f.sweep_deg = self._num("gen_sweep", "Sweep", lo=-30, hi=60) or 0.0
        f.sweep_start = self._num("gen_sweep0", "Sweep from", lo=0.15, hi=0.9) or 0.35
        d = self._num("gen_serr", "Serration depth", lo=0, hi=40)
        if d:
            f.serration_depth = d / 100
            f.serration_width = (self._num("gen_serr_w", "Tooth width", lo=4, hi=100, required=True)) / 100
            f.serration_start, f.serration_end = self._gen_span("gen_serr_span", "Teeth span")
        a = self._num("gen_tub", "Tubercle amplitude", lo=0, hi=15)
        if a:
            f.tubercle_amp = a / 100
            f.tubercle_wavelength = (self._num("gen_tub_w", "Tubercle wavelength", lo=10, hi=150,
                                               required=True)) / 100
            f.tubercle_start, f.tubercle_end = self._gen_span("gen_tub_span", "Tubercle span")
        t = self.v["gen_tip"].get()
        if t == "Winglet":
            f.tip = "winglet"
            f.winglet_height = self._num("gen_wl_h", "Winglet length", lo=2, hi=25, required=True) / 100
            f.winglet_cant = self._num("gen_wl_cant", "Winglet cant", lo=30, hi=100, required=True)
            f.winglet_dir = "up" if self.v["gen_wl_dir"].get() == "Up" else "down"
        elif t == "Raked":
            f.tip = "raked"
            f.rake_deg = self._num("gen_rake", "Rake angle", lo=10, hi=65, required=True)
            f.rake_len = self._num("gen_rake_len", "Rake length", lo=4, hi=30, required=True) / 100
        try:
            return f.validate()
        except ValueError as exc:
            raise InputError("gen_sweep", str(exc)) from None

    def _gen_pitch(self):
        s = self.v["gen_pitch"].get().strip().lower().replace(" ", "")
        if not s:
            return None
        inch = s.endswith(("in", '"', "inch"))
        num = re.sub(r'(inch|in|"|mm)$', "", s)
        try:
            x = float(num.replace(",", "."))
        except ValueError:
            raise InputError("gen_pitch", f"Pitch: '{s}' is not a number (mm, or add 'in' for inches, e.g. "
                                          "6in).") from None
        mm = x * 25.4 if inch else x
        if not 5 <= mm <= 5000:
            raise InputError("gen_pitch", f"Pitch must be between 5 and 5000 mm (got {mm:g} mm).")
        return mm

    GEN_CW = {"Solid (prop material)": "solid", "Steel rod insert": "steel", "Brass rod insert": "brass",
              "Lead rod insert": "lead", "Tungsten rod insert": "tungsten", "None (balance it yourself)": "none"}

    def _gen_cw_rows(self):
        one = self.v["gen_blades"].get() == "1"
        for w in self.gen_cw_widgets:
            w.grid() if one else w.grid_remove()

    def _gen_af_info(self):
        k = self.gen_airfoils.get(self.v["gen_airfoil"].get())
        if propgen is None or k is None:
            return
        a = propgen.AIRFOILS[k]
        self.gen_af_info.configure(text=f"{a['group']} · {a['tc_native'] * 100:.1f} % thick, "
                                        f"{a['camber'] * 100:.1f} % camber: {a['desc']}")

    def _gen_buttons(self):
        have = self.gen is not None and not self.gen_busy
        self._enable(self.gen_cmp_btn, propgen is not None and self.gen_busy in ("", "compare"))
        self._enable(self.gen_stl_btn, have)
        self._enable(self.gen_step_btn, have)
        self._enable(self.gen_use_btn, have and not self.sim_busy)

    def _gen_hub_rows(self):
        """Show the hub fields that apply."""
        if not hasattr(self, "gen_hub_widgets"):
            return

        def num(k):
            try:
                return float(self.v[k].get().replace(",", ".") or 0)
            except ValueError:
                return 0.0
        holes = num("gen_holes") > 0
        on = {"edge": self.v["gen_edge"].get() != "Sharp", "spinner": self.v["gen_spinner"].get() != "None",
              "hex": num("gen_hex") > 0, "holes": holes,
              "head": holes and self.v["gen_head"].get() != "None",
              "cbore": holes and self.v["gen_head"].get() == "Counterbore"}
        for g, ws in self.gen_hub_widgets.items():
            for w in ws:
                w.grid() if on.get(g, True) else w.grid_remove()

    def _gen_hub(self):
        """hubgeom.HubOptions from the form."""
        def n(key, label, lo=0.0, hi=500.0, default=0.0, **kw):
            x = self._num(key, label, lo=lo, hi=hi, **kw)
            return default if x is None else x
        holes = int(n("gen_holes", "Mount holes", hi=12))
        if abs(holes - n("gen_holes", "Mount holes", hi=12)) > 1e-9:
            raise InputError("gen_holes", "Mount holes: a whole number (0 to 12).")
        try:
            return hubgeom.HubOptions(
                edge=self.v["gen_edge"].get().lower(), edge_mm=n("gen_edge_mm", "Edge size", hi=50, default=1.0),
                spinner=self.v["gen_spinner"].get().lower(),
                spinner_length_mm=self._num("gen_spinner_len", "Spinner length", lo=0, open_lo=True, hi=1000),
                spinner_pocket_mm=n("gen_pocket", "Spinner pocket", hi=200),
                holes=holes, hole_mm=n("gen_hole_d", "Hole size", lo=0.3, hi=50, default=3.2),
                hole_circle_mm=n("gen_pcd", "Bolt circle", lo=1, hi=500, default=16.0),
                hole_angle_deg=self._num("gen_hang", "Bolt angle", lo=-360, hi=360),
                hole_head=self.v["gen_head"].get().lower(), head_mm=n("gen_head_d", "Head size", hi=60, default=6.0),
                head_depth_mm=n("gen_head_dep", "Head depth", hi=100, default=3.0),
                bore_tolerance_mm=n("gen_tol", "Bore tolerance", lo=-2, hi=2),
                bore_depth_mm=n("gen_bore_dep", "Bore depth", hi=1000),
                dflat_mm=n("gen_dflat", "D-flat", hi=50), set_screw_mm=n("gen_setscrew", "Set screw", hi=30),
                nut_hex_mm=n("gen_hex", "Nut pocket", hi=100), nut_depth_mm=n("gen_hex_d", "Pocket depth", hi=100,
                                                                             default=3.0),
                fillet_mm=n("gen_fillet", "Root fillet", hi=30),
                blade_offset_mm=n("gen_offset", "Blade offset", lo=-200, hi=200))
        except ValueError as exc:
            raise InputError("gen_hub", str(exc)) from None

    def _gen_spec(self):
        """propgen.DesignSpec from the form; raises InputError naming the field to fix."""
        goal = self.GEN_GOALS.get(self.v["gen_goal"].get(), "max_thrust")
        motor = self.v["gen_src"].get().startswith("Motor")
        kw = dict(goal=goal, airspeed=self._num("airspeed", "Airspeed", lo=0, hi=propgen.MAX_AIRSPEED) or 0.0,
                  rho=self._num("rho", "Air density", lo=0.05, hi=2.0, required=True),
                  blades=int(self.v["gen_blades"].get()),
                  counterweight=self.GEN_CW.get(self.v["gen_cw"].get(), "solid"),
                  material_density=self._num("gen_dens", "Density", lo=0.3, hi=25.0, required=True),
                  material=self.v["gen_mat"].get(), process=self.v["gen_proc"].get(),
                  strength_mpa=self._num("gen_uts", "Strength", lo=1, hi=5000, required=True),
                  modulus_gpa=self._num("gen_mod", "Modulus", lo=0.05, hi=500, required=True),
                  safety_factor=self._num("gen_sf", "Safety factor", lo=1, hi=10, required=True),
                  alt_load=(self._num("gen_alt", "Vibration load", lo=0, hi=150) or 0.0) / 100,
                  fatigue=bool(self._num("gen_alt", "Vibration load", lo=0, hi=150)),
                  respect_strength=bool(self.gen_respect.get()),
                  pitch_mm=self._gen_pitch(),
                  pitch_mode="constant" if self.v["gen_pmode"].get() == "Constant" else "p75",
                  features=self._gen_features(),
                  max_diameter_mm=self._num("gen_dmax", "Max diameter", lo=30, hi=3000, required=True),
                  diameter_mm=self._num("gen_dfix", "Fixed diameter", lo=30, hi=3000),
                  max_tip_mach=self._num("gen_mach", "Max tip Mach", lo=0.15, hi=propgen.MAX_TIP_MACH,
                                         required=True),
                  bore_mm=self._num("gen_bore", "Shaft bore", lo=0.5, hi=200) or 5.0,
                  hub_diameter_mm=self._num("gen_hub", "Hub diameter", lo=3, hi=500),
                  hub_height_mm=self._num("gen_hub_h", "Hub height", lo=2, hi=500),
                  hub=self._gen_hub(),
                  airfoil=self.gen_airfoils.get(self.v["gen_airfoil"].get(), "clarky"),
                  spin=self.v["gen_spin"].get().lower(), aero=self.v["aero"].get(),
                  nf_model=self.v["nf_model"].get(), ncrit=self._num("ncrit", "Ncrit", lo=1, hi=14) or 9.0)
        if kw["airspeed"] >= 0.9 * kw["max_tip_mach"] * propgen.A_SOUND:
            raise InputError("gen_mach", f"At {kw['airspeed']:g} m/s the tip Mach limit must be above "
                                         f"{kw['airspeed'] / (0.9 * propgen.A_SOUND):.2f}: the blade tips also "
                                         "move with the airspeed. Raise Max tip Mach or lower the airspeed.")
        if kw["diameter_mm"] and kw["diameter_mm"] > kw["max_diameter_mm"]:
            kw["max_diameter_mm"] = kw["diameter_mm"]
        if kw["hub_diameter_mm"] and kw["hub_diameter_mm"] <= kw["bore_mm"] + 1:
            raise InputError("gen_hub", "Hub diameter must be larger than the shaft bore (by at least 1 mm).")
        if goal == "target_thrust":
            kw["target_thrust_N"] = self._num("gen_target", "Target thrust", lo=0, open_lo=True, hi=5000,
                                              required=True)
        if motor:
            kw.update(source="motor", kv=self._num("kv", "KV", lo=0, open_lo=True, required=True, big=True),
                      motor_power_W=self._num("power", "Motor power", lo=0, open_lo=True, required=True, big=True),
                      rm=self._num("rm", "Rm", lo=0), i0=self._num("i0", "I0", lo=0),
                      max_torque_Nm=self._num("qmax", "Max torque", lo=0, open_lo=True),
                      esc_eff=(self._num("esc", "ESC efficiency", lo=1, hi=100) or 95.0) / 100)
            volts = self._num("voltage", "Battery voltage", lo=0, open_lo=True)
            cells = self._num("cells", "LiPo cells", kind=int, lo=1, hi=30)
            if volts:
                kw["voltage"] = volts
            elif cells:
                kw["cells"] = cells
                kw["cell_voltage"] = self._num("cellv", "Volts per cell", lo=0.5, hi=5) or 3.7
            else:
                raise InputError("cells", "A motor design needs the battery: enter the LiPo cell count or the "
                                          "battery voltage in card 05.")
        else:
            kw.update(source="shaft", rpm=self._num("gen_rpm", "Speed", lo=100, hi=200000, big=True))
            if goal != "target_thrust":
                kw["shaft_power_W"] = self._num("gen_pshaft", "Shaft power", lo=0, open_lo=True, hi=1e6,
                                                required=True, big=True)
        return propgen.DesignSpec(**kw)

    def _gen_run(self):
        if self.gen_busy or propgen is None:
            return
        try:
            spec = self._gen_spec()
        except InputError as e:
            self._input_error(e)
            return
        self._flag_entry(None)
        self._hide_banner()
        self.gen_busy = "design"
        self.gen_t0 = time.time()
        self.gen_btn.set_busy(True, "Designing")
        self.header.set_busy(True)
        self._gen_buttons()
        self.status.set("Designing the propeller…", "busy", right="")
        threading.Thread(target=self._gen_worker, args=(spec,), daemon=True).start()

    def _gen_compare(self):
        if self.gen_busy == "compare":
            self.gen_cancel.set()
            self.gen_cmp_btn.configure(text="Stopping…")
            return
        if self.gen_busy or propgen is None:
            return
        try:
            spec = self._gen_spec()
        except InputError as e:
            self._input_error(e)
            return
        self._flag_entry(None)
        self._hide_banner()
        self.gen_busy = "compare"
        self.gen_cancel = threading.Event()
        self.gen_t0 = time.time()
        self.gen_btn.set_busy(True, "Testing airfoils")
        self.gen_cmp_btn.configure(text="Stop the test")
        self.header.set_busy(True)
        self._gen_buttons()
        self.status.set(f"Testing {len(propgen.AIRFOILS)} airfoils…", "busy", right="")
        threading.Thread(target=self._gen_cmp_worker, args=(spec, self.gen_cancel), daemon=True).start()

    def _gen_cmp_worker(self, spec, cancel):
        from dataclasses import replace
        say = lambda m: self._post(self.status.set, m.strip(" .") + "…", "busy")  # noqa: E731

        def prog(i, n, row):
            say(f"Testing airfoils: {i} of {n} done ({row['label']})")
        try:
            rows = propgen.compare_airfoils(spec, progress=prog, cancel=cancel)
            best = next((r for r in rows if r["ok"]), None)
            if best is None:
                raise RuntimeError("Stopped before any airfoil finished." if cancel.is_set() else
                                   "No airfoil gave a design: " + (rows[0]["error"] if rows else "nothing ran"))
            self._post(self.v["gen_airfoil"].set, best["label"])
            say(f"Best: {best['label']}. Generating it")
            d = propgen.design(replace(spec, airfoil=best["airfoil"]), progress=say)
            say("Checking the design: slicing its 3D model and analysing it")
            v = propgen.verify(d, progress=say)
            try:
                v["preview"] = propgen.preview_mesh(d)
            except Exception:  # noqa: BLE001 - the shaded view is a nicety
                traceback.print_exc()
            self._post(self._gen_done, d, v, None, "", (rows, spec, cancel.is_set()))
        except BaseException as e:  # noqa: BLE001 - the window must always hear back
            self._post(self._gen_done, None, None, self._describe(e), traceback.format_exc())

    def _gen_worker(self, spec):
        say = lambda m: self._post(self.status.set, m.strip(" .") + "…", "busy")  # noqa: E731
        try:
            d = propgen.design(spec, progress=say)
            say("Checking the design: slicing its 3D model and analysing it")
            v = propgen.verify(d, progress=say)
            try:
                v["preview"] = propgen.preview_mesh(d)
            except Exception:  # noqa: BLE001 - the shaded view is a nicety
                traceback.print_exc()
            self._post(self._gen_done, d, v, None, "")
        except BaseException as e:  # noqa: BLE001 - the window must always hear back
            self._post(self._gen_done, None, None, self._describe(e), traceback.format_exc())

    def _gen_done(self, d, v, err, tb, cmp=None):
        was = self.gen_busy
        self.gen_busy = ""
        self.gen_btn.set_busy(False)
        self.gen_cmp_btn.configure(text=f"Test all {len(self.gen_airfoils)} airfoils")
        if was == "design":
            self.gen_cmp = None
        elif cmp is not None:
            self.gen_cmp = cmp
        self.header.set_busy(self.cfd_busy or self.sim_busy)
        dt = time.time() - self.gen_t0
        if err:
            self._show_banner(err, "DESIGN FAILED")
            self.status.set(f"Design failed: {err.splitlines()[0]}", "error", right=f"{dt:.1f} s")
            self._gen_set_text([(f"The design failed:\n\n{err}\n\n{tb}", "warn")])
            self._gen_buttons()
            return
        self.gen = (d, v)
        try:
            self._gen_show(d, v)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._show_banner(f"Design made but could not be displayed: {type(e).__name__}: {e}")
        self._gen_buttons()
        if cmp is not None:
            n_ok = sum(r["ok"] for r in cmp[0])
            self.status.set(f"Best of {n_ok} airfoils{' (stopped early)' if cmp[2] else ''}: {d.airfoil_label}. "
                            f"Designed {d.name}: {d.thrust_N:.2f} N at {d.rpm:,.0f} rpm", "ok", right=f"{dt:.1f} s")
        else:
            self.status.set(f"Designed {d.name}: {d.thrust_N:.2f} N at {d.rpm:,.0f} rpm", "ok", right=f"{dt:.1f} s")

    def _gen_show(self, d, v):
        D = d.diameter_mm
        nb = f"{d.blades} blade{'s' * (d.blades > 1)}"
        cw = d.counterweight
        chips = [(f"Ø {D:.0f} mm · pitch {d.pitch75_mm:.0f} mm", GREEN), (nb, CYAN),
                 (d.airfoil_label, CYAN), (d.spin.upper(), CYAN),
                 ("hover" if d.airspeed == 0 else f"{d.airspeed:g} m/s", CYAN),
                 (f"hub Ø {d.hub_diameter_mm:.1f} · bore {d.bore_mm:g} mm", CYAN)]
        minor = getattr(d, "minor_notes", [])
        warns = ([(n, AMBER) for n in d.notes if n not in minor] + [(n, CYAN) for n in d.notes if n in minor])[:3]
        if d.limited_by:
            warns.insert(0, (f"diameter set by {d.limited_by}", CYAN))
        if cw:
            warns.insert(0, (f"counterweight {cw['mass_g']:.1f} g opposite the blade" + (
                f": {cw['kind']} rod Ø{cw['rod_diameter_mm']:g} × {cw['rod_length_mm']:.1f} mm in the pod"
                if cw["rod_diameter_mm"] > 0 else ", solid") + f" ({cw['material_density']:g} g/cm³, fully solid "
                "part); details in the report", MAGENTA))
        f = d.spec.features
        if f.any:
            chips.insert(1, (" · ".join(x for x, on in (("swept", abs(f.sweep_deg) > 0.05), ("serrated", f.serrated),
                                                     ("tubercles", f.tubercles), ("winglet", f.winglet),
                                                     ("raked tip", f.raked)) if on), AMBER))
        if d.spec.pitch_mm:
            chips[0] = (chips[0][0] + (" (set, 75 % R)" if d.spec.pitch_mode == "p75" else " (set, constant)"), GREEN)
        chips.insert(1, (f"{d.spec.material} · {d.spec.process.lower()}", GREEN))
        self.gen_strip.set(d.name, chips, warns)
        t = self.gen_tiles
        def _in(x):
            return f"{x / 25.4:.1f}".rstrip("0").rstrip(".")
        t["size"].set(f"{_in(D)}×{_in(d.pitch75_mm)}", "in",
                      f"{D:.0f} mm diameter · {d.pitch75_mm:.0f} mm pitch · {nb}")
        t["rpm"].set(f"{d.rpm / 1000:.1f}k" if d.rpm >= 10000 else f"{d.rpm:,.0f}", "rpm",
                     f"{d.rpm:,.0f} rpm · Mach {d.tip_mach:.2f}")
        t["thrust"].set(f"{d.thrust_N:.2f}", "N", f"{d.thrust_N / G * 1000:,.0f} g")
        t["power"].set(f"{d.shaft_power_W:.1f}", "W", f"torque {d.torque_Nm:.4f} N·m")
        if d.airspeed > 0:
            t["eff"].set(f"{d.efficiency * 100:.1f}", "%", "propulsive efficiency", label="Prop efficiency")
        else:
            t["eff"].set(f"{d.figure_of_merit:.2f}", "", f"{d.thrust_N / G * 1000 / max(d.shaft_power_W, 1e-9):.1f} "
                         "gf/W shaft", label="Figure of merit")
        e = d.electrical
        if e:
            t["batt"].set(f"{e['battery_power']:.0f}", "W", f"{e['battery_current']:.1f} A · throttle "
                          f"{e['throttle'] * 100:.0f} %", bar=min(e["throttle"], 1.0), warn=e["throttle"] > 1.0001)
        else:
            t["batt"].set("—", "", "no motor model")
        rel = v.get("rel_error", {})
        dT = rel.get("thrust_N", float("nan")) * 100
        t["check"].set(f"{dT:+.1f}", "%", "thrust, sliced model" if v.get("ok") else "check failed: see notes",
                       warn=not v.get("ok"))
        stt, vb = d.structure, d.vibration
        # the resonance verdict first: the narrow tile drops trailing '·' parts
        if vb is None:
            vib = ""
        elif vb["ok"]:
            vib = "no resonance · "
        else:
            m0 = vb["resonant"][0]
            vib = f"RESONANT {m0['name'].split('. ')[-1]} ≈ {m0['nearest_order']}P · "
        if stt:
            t["strength"].set(f"{stt['sf_overall']:.2f}" if stt["sf_overall"] < 99.5 else ">99", "SF",
                              vib + ("" if stt["ok"] else f"need {d.spec.safety_factor:g} · ")
                              + f"{stt['governing']} · {stt['mat']['material'][:18]}",
                              bar=min(stt["sf_overall"] / max(2 * d.spec.safety_factor, 1e-9), 1.0),
                              warn=not stt["ok"] or (vb is not None and not vb["ok"]))
        else:
            t["strength"].set("—", "", vib + "not checked", warn=vb is not None and not vb["ok"])
        x = np.asarray(d.r_mm) / (D / 2)
        self.gen_shape.set_data(x, [dict(y=d.chord_mm, label="chord", color=GREEN, axis="L", fill=True, unit="mm"),
                                    dict(y=d.beta_deg, label="β", color=VIOLET, axis="R", unit="°")])
        self._gen_chart_d = d
        self._gen_chart()
        o = v.get("outcome")
        if o is not None:
            try:
                self.gen_view.set_model(o.geom, o.mesh, o.res, pod=d.counterweight)
                pv = v.get("preview")
                if pv is not None:
                    self.gen_view.set_solid(pv[0], pv[1], frame=o.geom.to_analysis_frame)
            except Exception as ex:  # noqa: BLE001 - the preview is a nicety
                traceback.print_exc()
                self.gen_view.clear()
                self.status.set(f"3D preview unavailable: {ex}", "warn")
        parts = []
        if self.gen_cmp is not None and self.gen_cmp[0] and self.gen_cmp[0][0].get("airfoil") == d.airfoil:
            rows, spec, stopped = self.gen_cmp
            lines = propgen.compare_text(rows, spec).splitlines()
            if stopped:
                lines[0] += f"  (stopped early: {len(rows)} of {len(propgen.AIRFOILS)} tested)"
            parts += [("\n".join(lines[:2]) + "\n", None), (lines[2] + "  <- generated below\n", "ok"),
                      ("\n".join(lines[3:]) + "\n\n", None)]
        parts.append((propgen.verify_text(v).rstrip() + "\n\n", "ok" if v.get("ok") else "warn"))
        if d.notes:
            parts.append(("".join(f"note: {n}\n" for n in d.notes) + "\n", "warn"))
        parts.append((propgen.summary_text(d).lstrip("\n"), None))
        self._gen_set_text(parts)

    def _gen_chart(self):
        """The lower-right generator chart: the diameter scan or the Campbell (vibration) diagram."""
        d = self._gen_chart_d
        ch = self.gen_scan
        if self.gen_chart_kind.get() == "Vibration":
            ch.title, ch.xlabel, ch.xfmt = "Campbell", "rotation speed [rpm]", lambda v, s: f"{v:,.0f}"
            vb = d.vibration if d is not None else None
            if not vb:
                ch.set_data(None, empty="Generate a propeller to see its blade frequencies"
                            if d is None else "No vibration check for this design (see the report)")
                return
            xs = np.asarray(vb["speeds_rpm"], float)
            cols = {"flap": CYAN, "lag": MAGENTA, "torsion": VIOLET}
            seen, ser = set(), []
            for k in vb["orders"]:
                ser.append(dict(y=k * xs / 60, label="1P-4P" if k == vb["orders"][0] else "", color=MUTED,
                                axis="L", dash=(4, 4), unit="Hz"))
            fmax = 1.15 * max(vb["orders"]) * xs[-1] / 60
            for md in vb["modes"]:
                if md["freq"][0] > fmax and md["freq_design"] > fmax:
                    continue
                kind = md["kind"]
                ser.append(dict(y=np.minimum(md["freq"], fmax), label=kind if kind not in seen else "",
                                color=AMBER if md["resonant"] else cols[kind], axis="L", unit="Hz"))
                seen.add(kind)
            ch.set_data(xs, ser, marker=(vb["rpm"], f"{vb['rpm']:,.0f} rpm"), ylabel=("frequency [Hz]", "f [Hz]"),
                        ylabel2="")
            return
        ch.title, ch.xlabel, ch.xfmt = "Diameter scan", "diameter [mm]", lambda v, s: f"{v:,.0f}"
        rows = sorted(d.scan, key=lambda s: s["diameter_mm"]) if d is not None else []
        if not rows:
            ch.set_data(None, empty="The diameters tried appear here")
            return
        D = d.diameter_mm
        if rows:
            xs = [s["diameter_mm"] for s in rows]
            eff = [s["fom"] if d.airspeed == 0 else s["efficiency"] for s in rows]
            bad = np.array([not s["feasible"] for s in rows])
            tgt = d.spec.goal == "target_thrust"
            ch.set_data(xs, [dict(y=[s["power_W"] if tgt else s["thrust_N"] for s in rows],
                                             label="shaft power" if tgt else "thrust", color=MAGENTA if tgt else CYAN,
                                             axis="L", fill=True, unit="W" if tgt else "N"),
                                        dict(y=eff, label="FM" if d.airspeed == 0 else "η", color=GREEN, axis="R")],
                                   marker=(D, f"chosen {D:.0f} mm"), bands=self._bands(np.asarray(xs, float), bad),
                                   band_label="not feasible",
                                   ylabel=("shaft power [W]", "P [W]") if tgt else ("thrust [N]", "T [N]"),
                                   ylabel2=("figure of merit", "FM") if d.airspeed == 0 else ("efficiency", "η"))

    def _gen_file(self, ext):
        d = self.gen[0]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"prop_{d.diameter_mm / 25.4:.1f}x{d.pitch75_mm / 25.4:.1f}_"
                                               f"{d.blades}b_{d.spin}")
        return stem + ext

    def _gen_export(self, kind):
        if not self.gen or self.gen_busy:
            return
        ext = ".stl" if kind == "stl" else ".step"
        p = filedialog.asksaveasfilename(title=f"Export propeller as {kind.upper()}", initialdir=self._start_dir(),
                                         initialfile=self._gen_file(ext), defaultextension=ext,
                                         filetypes=[("STL mesh", "*.stl")] if kind == "stl" else
                                         [("STEP solid", "*.step *.stp")])
        if not p:
            return
        d = self.gen[0]
        if kind == "stl":
            try:
                propgen.export_stl(d, p, quality="fine")
                self.status.set(f"Saved {p} (millimetres, spin axis Z)", "ok")
            except Exception as e:  # noqa: BLE001
                self._show_banner(f"Could not save the STL: {self._describe(e)}")
            return
        if self.sim_busy and self.v["model"].get().lower().endswith((".step", ".stp")):
            self._show_banner("A STEP model is being loaded; export again when the simulation has finished.",
                              "BUSY", AMBER)
            return
        self.gen_busy = "step"
        self.gen_t0 = time.time()
        self._gen_buttons()
        self.header.set_busy(True)
        self.status.set("Building the STEP solid (B-spline blades + hub, fused; up to a few minutes for blades "
                        "with tubercles)…", "busy", right="")

        def work():
            try:
                info = propgen.export_step(d, p, progress=lambda m: self._post(self.status.set, m, "busy"),
                                           mode=self.v["gen_step"].get().lower())
                self._post(self._gen_step_done, p, info, None)
            except BaseException as e:  # noqa: BLE001
                traceback.print_exc()
                self._post(self._gen_step_done, p, None, self._describe(e))
        threading.Thread(target=work, daemon=True).start()

    def _gen_step_done(self, p, info, err):
        self.gen_busy = ""
        self.header.set_busy(self.cfd_busy or self.sim_busy)
        self._gen_buttons()
        if err:
            self._show_banner(f"Could not export the STEP file: {err}")
            self.status.set("STEP export failed", "error")
        else:
            ck = (info or {}).get("check") or {}
            nvol = (info or {}).get("volumes", 1)
            bad = ck.get("open_edges", 0) or nvol != 1
            self.status.set(f"Saved {p} ({info.get('mode', 'exact')}, millimetres, spin axis Z): "
                            + (f"{nvol} solid(s), {ck.get('open_edges', '?')} open edges - check it before use"
                               if bad else "one closed solid"),
                            "warn" if bad else "ok", right=f"{time.time() - self.gen_t0:.1f} s")

    def _gen_use(self):
        """Load the generated prop into the main simulation (as an STL in the temp folder) and run it."""
        if not self.gen or self.gen_busy or self.sim_busy:
            return
        d = self.gen[0]
        p = os.path.join(tempfile.gettempdir(), "propsim_" + self._gen_file(".stl"))
        try:
            # without the details the slicer cannot read (teeth, tubercles, winglet); their effects go to the
            # analysis as options instead, the way the generator's own check does it
            propgen.build_mesh(d, "normal", details=False, hub_cuts=False).export(p)
            import features as feat_mod
            self.gen_model_extra = (os.path.normcase(os.path.abspath(p)), feat_mod.analysis_argv(d.spec.features))
        except Exception as e:  # noqa: BLE001
            self._show_banner(f"Could not write the model: {self._describe(e)}")
            return
        self.test_prop.set(False)
        for k, val in (("model", p), ("units", "mm"), ("axis", "z"), ("blades", ""), ("hub", "")):
            self.v[k].set(val)
        if d.spec.source == "motor" and d.spec.goal == "max_thrust":
            self.v["mode"].set("motor")
        else:
            self.v["mode"].set("rpm")
            self.v["rpm"].set(f"{d.rpm:.0f}")
        self._mode_changed()
        self._source_changed()
        self._show_tab("dash")
        self._run()

    def _on_close(self):
        if self.cfd_running and not self.closing:
            if not messagebox.askyesno(APP_TITLE, "A 3D CFD run is in progress. Stop it and quit?\n\n"
                                       "(The solver first writes its current state, which can take up to a minute.)"):
                return
            self.closing = True
            self.cfd_stop.set()
            self.status.set("Stopping OpenFOAM before closing (close again to quit at once)…", "busy")
            return
        self.root.destroy()


def set_icon(root):
    """The prop logo as the window / taskbar icon (else Windows shows customtkinter's logo)."""
    try:
        if sys.platform.startswith("win"):
            # iconbitmap wants an .ico file; it also stops customtkinter from setting its own icon
            path = os.path.join(tempfile.gettempdir(), "PropSim_icon_1.ico")
            if not os.path.exists(path):
                with open(path + ".tmp", "wb") as f:
                    f.write(ico_bytes((16, 20, 24, 32, 40, 48, 64)))
                os.replace(path + ".tmp", path)
            root.iconbitmap(default=path)
        else:
            root._icons = [tk.PhotoImage(master=root, data=png_bytes(logo_rgba(s))) for s in (64, 32, 16)]
            root.iconphoto(True, *root._icons)
    except Exception:  # noqa: BLE001 - an icon is not worth failing over
        traceback.print_exc()


def fit_window(root, w=1360, h=860, min_w=1200, min_h=760):
    """Size the window in customtkinter's logical units, but never larger than the screen: at 150 %
    Windows scaling a 1920 x 1080 laptop has 1280 x 720 logical pixels, less the taskbar."""
    sc = Theme.scale or 1.0
    sw, sh = root.winfo_screenwidth() / sc, root.winfo_screenheight() / sc
    aw, ah = int(sw - 24), int(sh - 100)          # room for a title bar (~32) and the taskbar (~48)
    w, h = min(w, aw), min(h, ah)
    root.minsize(min(min_w, aw), min(min_h, ah))
    # centred above the taskbar; +x+y are physical pixels and place the frame, title bar included
    x, y = max(0, int((sw - w) / 2 * sc)), max(0, int((sh - 48 - h - 32) / 2 * sc))
    root.geometry(f"{w}x{h}+{x}+{y}")
    if h < min_h:                                 # a small screen: use all of it
        with contextlib.suppress(tk.TclError):
            root.state("zoomed")


def main():
    # a windowed exe has no console: give stray prints and warnings somewhere to go
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    ctk.set_appearance_mode("dark")
    # Keep the DPI scaling the window starts with. customtkinter otherwise re-checks the monitor's
    # scaling every 0.1 s and, when the window is dragged onto (or partly over) a monitor with other
    # scaling, fades it out and rebuilds every widget - repeatedly while it is dragged back and forth
    # - and the canvases, drawn at the starting scale, would no longer match the widgets.
    ctk.ScalingTracker.update_loop_running = True
    root = ctk.CTk()
    root.title(APP_TITLE)
    Theme.init(root)
    set_icon(root)
    fit_window(root)
    app = App(root)
    root.report_callback_exception = app.tk_error
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()     # "Test all airfoils" worker processes in the frozen app
    main()
