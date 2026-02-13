# Project: SAXS Data Processor (MuDPaW / PLI tools)

## Scope
GUI tools (Tkinter) and processing routines to rescale space–time PLI diagrams, align PLI/SAXS/rheology timelines, and output steady-state stitched panels and JSON metadata. Focus on `PLI_rescale_space_time_diagram.py`.

## Core files
- `PLI_rescale_space_time_diagram.py` — main GUI + preview + ENAGE stitching
- `Time_stamper_PLI_v1.py` — legacy time-source reader: `compute_pair_widths_from_times(...)`
- Others: helpers for stitching, cropping, path resolution

## Modes
1. **Triggered steady-state intervals**
   - Input: FPS + time-source CSV (“Time of Day” column only; ignore “Shear Rate”).
   - Per step: transient run → blank (NaN/empty) → steady run.
   - Compute four absolutes per step (relative to t₀):  
     - T_begin = first time of transient run  
     - T_end   = last time before blank  
     - S_begin = first time after blank  
     - S_end   = last time of steady run  
   - Build hidden T/S frames:  
     `T = (T_end − T_begin) * fps`, `S = (S_end − S_begin) * fps`  
   - Preview mapping: red = END (S_end), cyan = BEGIN (T_begin).  
     Pixel mapping: `x(t) = end_x − (t_last − t) * fps`. Gaps: ~30 px per 1 s at 29.97 fps.

2. **Not triggered — reference rheology file**
   - Show 2 columns (Begin/End seconds per shear segment).
   - Hidden T/S built from total − steady-width (px).

3. **Manual**
   - Show blank Begin/End (seconds) rows; ENGAGE derives T/S = (End−Begin)*fps split by steady width.

## Preview rules (critical)
- **No “top-level” preview code**; all preview drawing must live inside class methods.
- Use `self.img` and `self.img_width` in preview scope (not `img`).
- Horizontal crop lines: `self.ax.axhline(...)`; vertical end line: `self.ax.axvline(...)`.
- After drawing lines, always call `self.canvas.draw_idle()` or `self.canvas.draw()`.

## Common fixups
- If VS Code shows the file “all red”: check for stray module-level `self.` or `img.` code (delete), duplicated encoding lines, or `self.axhline` typos (should be `self.ax.axhline`).

## Preferences
- Favor explicit, surgical patches.  
- Keep Tkinter layout intact.  
- Don’t break ENAGE stitching path.
