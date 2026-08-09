"""
Energy spectrum computation for proton radiography tracks (MATLAB-compatible).

Reference:
  P. Martin et al., "Absolute Calibration of Fujifilm BAS-TR Image Plate
  Response to Laser Driven Protons up to 40 MeV",
  Review of Scientific Instruments 93, 053303 (2022).

Pipeline (matching the MATLAB reference):
  1. Row-wise PSL sampling: sum all pixels within the track polygon for each
     y-row.  R = y  (y is the dispersion direction in magnetic spectrometer).
  2. IP fading correction:  PSL ← (t_ref / t)^k × PSL  (both signal & bg)
  3. Background subtraction:  PSL_net = PSL_signal - PSL_bg
  4. R (mm) → E (MeV) via user-provided lookup table (spline interp)
  5. Aluminium filter correction (range-energy residual method)
  6. PSL → proton number via Martin et al. formula (piecewise)
  7. Fixed dE binning → dN/dE/dΩ
"""

import os
import tempfile
import numpy as np
import cv2
from scipy.interpolate import interp1d

# Debug log file
_DEBUG_LOG = r"C:\Users\Lenovo\Desktop\log.txt"
def _dbg(msg):
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ---------------------------------------------------------------------------
# Physical defaults — override via config / function arguments
# ---------------------------------------------------------------------------
PIXEL_TO_UM = 50.0            # μm / pixel (IP scanner)

# IP fading (BAS-TR):  PSL_corrected = (t_ref / t)^k × PSL_raw
IP_FADING_EXPONENT = -0.161   # k  (set to 0 to disable)
IP_FADING_REF_MIN = 30.0      # t_ref in minutes

# Aluminium filter
AL_THICKNESS_CM = 0.0015      # cm  (15 μm)
AL_DENSITY_G_CM3 = 2.7        # g/cm³

# Energy binning (MATLAB: 60 → 1 MeV, step -0.2)
DEFAULT_D_E_MEV = -0.2
DEFAULT_E_BIN_MAX = 60.0
DEFAULT_E_BIN_MIN = 1.0

# Analytic R→E fallback (used when no r_to_e_path table is provided)
A_R2E = 0.018                 # E(MeV) ≈ A × R_pixel^B
B_R2E = 1.75

# Martin et al. PSL → N piecewise coefficients
#   E < 1.6 MeV:  N = PSL / (C_LO × E^P_LO)
#   E ≥ 1.6 MeV:  N = PSL / (C_HI × E^P_HI)
MARTIN_E_THRESH = 1.6          # MeV
MARTIN_C_LO = 0.151
MARTIN_P_LO = 0.6
MARTIN_C_HI = 0.284
MARTIN_P_HI = -0.75

# ---------------------------------------------------------------------------
# Table loaders
# ---------------------------------------------------------------------------


def load_r_to_e_table(path):
    """Load an R→E lookup table.

    Expected file format: four whitespace-separated columns (only first two used)
        R(m)  E(MeV)  [col3]  [col4]
    R is converted: m → mm (×1000, matching MATLAB convention:
    ``r = r_e_proton(:,1)*1000``).

    Returns a callable ``f(r_mm) -> E_MeV`` (cubic spline interpolation).
    """
    data = np.loadtxt(path, encoding="utf-8")
    r_col = data[:, 0] * 1000.0  # m → mm (MATLAB: r = r_e_proton(:,1)*1000)
    e_col = data[:, 1]           # MeV
    return interp1d(r_col, e_col, kind="cubic", fill_value="extrapolate")


def load_al_range_energy_table(path, sheet_name=None):
    """Load an Aluminium range-energy table (text or Excel).

    For plain-text files: whitespace-separated columns
        E(MeV)  Range(g/cm²)  [optional …]

    For Excel files (.xlsx / .xls): reads the specified sheet_name
    (defaults to the first sheet if None).
    Defaults to columns 0 (E) and 2 (Range), matching the MATLAB convention
    ``data = xlsread('质子穿透.xlsx','Al')`` where E=col1, Range=col3.
    Falls back to columns 0 and 1 if the sheet has fewer than 3 columns.

    Returns
    -------
    tuple[callable, callable]
        (E→Range, Range→E)  — both linear interpolation, extrapolating.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        import pandas as pd
        kwargs = {"header": None}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        df = pd.read_excel(path, **kwargs)
        data = df.values
        # MATLAB convention: E in col 0, Range in col 2
        if data.shape[1] >= 3:
            e_col = data[:, 0]
            r_col = data[:, 2]
        else:
            e_col = data[:, 0]
            r_col = data[:, 1]
        # Force numeric, coerce non-numeric (headers etc.) to NaN
        e_col = np.asarray(pd.to_numeric(e_col, errors="coerce"), dtype=float)
        r_col = np.asarray(pd.to_numeric(r_col, errors="coerce"), dtype=float)
    else:
        data = np.loadtxt(path, encoding="utf-8")
        e_col = data[:, 0]      # MeV
        r_col = data[:, 1]      # g/cm²

    # Drop rows with NaN
    mask = np.isfinite(e_col) & np.isfinite(r_col)
    e_col = e_col[mask]
    r_col = r_col[mask]

    # MATLAB uses 'spline' (cubic) for Al range-energy interpolation.
    e_to_r = interp1d(e_col, r_col, kind="cubic", fill_value="extrapolate")
    r_to_e = interp1d(r_col, e_col, kind="cubic", fill_value="extrapolate")
    return e_to_r, r_to_e

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polygon_to_binary_mask(points, h, w):
    """Create a uint8 binary mask from a polygon point list [(x,y), …]."""
    if not points or len(points) < 3:
        return None
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _rowwise_psl_sum(image, mask):
    """Sum PSL values of all masked pixels for each row y.

    Returns
    -------
    y_vals : np.ndarray (N,)  — row indices where mask has pixels
    psl_sums : np.ndarray (N,)  — sum of PSL in that row
    """
    ys = np.where(mask.sum(axis=1) > 0)[0]
    if len(ys) == 0:
        return np.array([]), np.array([])
    psl_sums = np.array([
        image[y][mask[y] > 0].sum() for y in ys
    ], dtype=np.float64)
    return ys.astype(np.float64), psl_sums


def _centerline_x_at_y(centerline, y):
    """Interpolate the centerline x position at a given y."""
    cl = [(float(p[0]), float(p[1])) for p in centerline]
    if cl[0][1] > cl[-1][1]:
        cl = list(reversed(cl))
    if len(cl) < 2:
        return cl[0][0] if cl else 0.0
    if y <= cl[0][1]:
        return cl[0][0]
    if y >= cl[-1][1]:
        return cl[-1][0]
    for i in range(len(cl) - 1):
        x1, y1 = cl[i]
        x2, y2 = cl[i + 1]
        if y1 <= y <= y2:
            if abs(y2 - y1) < 1e-12:
                return x1
            t = (y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)
    return cl[-1][0]


def pixel_to_psl(ql, R=50.0, S=4000.0, L=5.0, G=None):
    """Convert raw pixel value QL to PSL.

    $$PSL = \\left( \\frac{R}{100} \\right)^2 \\times \\frac{4000}{S}
           \\times 10^{L \\left( \\frac{QL}{2^G - 1} - \\frac{1}{2} \\right)}$$

    Args:
        ql: raw pixel value (0 … 2^G - 1), int or float array
        R: resolution (default 50)
        S: sensitivity (default 4000)
        L: latitude / dynamic range (default 5)
        G: bit depth (default None → auto-detect from ql.dtype:
           uint8→8, uint16→16, etc.)

    Returns PSL value(s), same shape as ql.
    """
    arr = np.asarray(ql)
    if G is None:
        G = arr.dtype.itemsize * 8      # uint8→8, uint16→16, etc.
    ql = arr.astype(np.float64)
    ql_max = (2 ** G) - 1               # e.g. 65535 for 16-bit
    return (R / 100.0) ** 2 * (4000.0 / S) * 10.0 ** (L * (ql / ql_max - 0.5))

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_track_spectrum(
    track_polygon,          # list of [x, y] in global IP image coords
    bg_polygon,             # list of [x, y] in global coords, or None
    ip_image,               # (H, W) float64 PSL image
    bg_centerline=None,     # list of [x, y] global background centerline
    r_to_e_func=None,       # f(R_mm) -> E_MeV
    al_e_to_range=None,     # f(E_MeV) -> Range_g_cm2
    al_range_to_e=None,     # f(Range_g_cm2) -> E_MeV
    pixel_to_um=PIXEL_TO_UM,
    al_thickness_cm=AL_THICKNESS_CM,
    al_density=AL_DENSITY_G_CM3,
    solid_angle=1.0,        # sr
    t_shot=None,
    t_scan=None,
    ip_fading_exponent=IP_FADING_EXPONENT,
    ip_fading_ref_min=IP_FADING_REF_MIN,
    dE_MeV=DEFAULT_D_E_MEV,
    E_bin_max=DEFAULT_E_BIN_MAX,
    E_bin_min=DEFAULT_E_BIN_MIN,
    source_y=0.0,           # source centroid Y (pixel), origin for R→E
    skip_bg=False,          # if True, skip bg subtraction (use raw PSL)
    debug_raw_image=None,   # (H, W) float64 raw QL image for per-row dump
    debug_label=None,       # track label for debug file naming
):
    """Compute dN/dE/dΩ for a single track (MATLAB-compatible pipeline).

    Parameters
    ----------
    track_polygon : list of [x, y]
        Vertices of the track ribbon polygon in global IP image coordinates.
    bg_polygon : list of [x, y] or None
        Vertices of the background strip polygon in global coordinates.
    ip_image : np.ndarray (H, W) float64
        Raw PSL image.
    r_to_e_func : callable, optional
        ``E_MeV = f(R_mm)``.  If None, falls back to analytic approx.
    al_e_to_range, al_range_to_e : callable, optional
        Al range-energy interpolators.  If None, skip Al correction.
    pixel_to_um : float
    al_thickness_cm : float
    al_density : float — g/cm³
    solid_angle : float — sr
    t_shot, t_scan : datetime.datetime, optional
    ip_fading_exponent : float
        Power-law exponent for IP fading.  Set to 0 to disable.
    ip_fading_ref_min : float
        Reference time in minutes.
    dE_MeV : float
        Bin width in MeV (negative means descending: 60→1).
    E_bin_max, E_bin_min : float
        Energy range.

    Returns
    -------
    dict or None
    """
    h, w = ip_image.shape

    # --- Guard: energy origin is meaningless without a valid source_y -----
    # R→E dispersion is measured FROM the source centroid; source_y must be
    # a real, positive pixel coordinate. 0.0 (the old default when no source
    # was found) silently produced a spurious spectrum — refuse it here too.
    if source_y is None or not np.isfinite(source_y) or source_y <= 0.0:
        _dbg(f"[spectrum] INVALID source_y={source_y!r} — cannot determine "
             "energy origin, aborting")
        return None

    # --- R→E function -----------------------------------------------------
    if r_to_e_func is None:
        def _r2e(r_pixel):
            return A_R2E * (r_pixel ** B_R2E)
        r_to_e_func = _r2e

    # --- Step 1: Row-wise PSL sampling from track polygon ------------------
    track_mask = _polygon_to_binary_mask(track_polygon, h, w)
    if track_mask is None or track_mask.sum() == 0:
        return None

    # Debug: pixel-level mask widths (per-row column count)
    t_mask_rows, t_mask_cols = np.where(track_mask)
    if len(t_mask_rows) > 0:
        from collections import Counter
        t_row_widths = Counter(t_mask_rows)
        t_w_vals = list(t_row_widths.values())
        _dbg(f"[mask] track: rows={len(t_row_widths)}  width min/avg/max = {min(t_w_vals)}/{np.mean(t_w_vals):.1f}/{max(t_w_vals)}")

    # --- Debug: dump per-row raw QL pixel values to log directory ----------
    if debug_raw_image is not None and debug_label is not None:
        try:
            ys_dbg = np.where(track_mask.sum(axis=1) > 0)[0]
            log_path = os.path.join(
                r"C:\Users\Lenovo\Desktop\log", debug_label
            )
            with open(log_path, "w", encoding="utf-8") as _f:
                _f.write(f"# Track: {debug_label}\n")
                _f.write(f"# {'row_y':>5s}  raw_QL_values\n")
                for y in ys_dbg:
                    vals = debug_raw_image[y][track_mask[y] > 0]
                    val_str = " ".join(f"{int(round(float(v)))}" for v in vals)
                    _f.write(f"{int(y):5d}  [{val_str}]\n")
        except Exception as e:
            _dbg(f"[debug] failed to write raw pixels for {debug_label}: {e}")

    R_y, PSL_signal = _rowwise_psl_sum(ip_image, track_mask)
    if len(R_y) == 0:
        return None

    # R = distance from source along dispersion direction (pixel → mm)
    _dbg(f"[R_mm] R_y(pixel): min={R_y.min():.1f}  max={R_y.max():.1f}  n={len(R_y)}")
    _dbg(f"[R_mm] source_y(pixel)={source_y:.1f}  pixel_to_um={pixel_to_um}")
    _dbg(f"[R_mm] |R_y - source_y|(pixel): min={np.abs(R_y - source_y).min():.1f}  max={np.abs(R_y - source_y).max():.1f}")
    R_mm = np.abs(R_y - source_y) * pixel_to_um / 1000.0
    _dbg(f"[R_mm] R_mm: min={R_mm.min():.4f}  max={R_mm.max():.4f}")

    # --- MATLAB alignment: drop R <= 3.0 mm rows (low-energy end near the
    #     source — heavy scattering / below Al filter cutoff / unreliable
    #     R→E extrapolation).  MATLAB: index1 = find(R_PSL(:,1)>3.0).
    R_MIN_MM = 3.0
    _valid = R_mm > R_MIN_MM
    _n_dropped = len(R_y) - int(_valid.sum())
    if _n_dropped > 0:
        _dbg(f"[R_mm] dropped {_n_dropped}/{len(R_y)} rows with R_mm <= {R_MIN_MM} (MATLAB alignment)")
    R_y = R_y[_valid]
    R_mm = R_mm[_valid]
    PSL_signal = PSL_signal[_valid]
    if len(R_y) == 0:
        return None

    # --- Step 2: IP fading correction (signal) ----------------------------
    dt_min = None
    fading = 1.0
    if ip_fading_exponent != 0 and t_shot is not None and t_scan is not None:
        dt_min = (t_scan - t_shot).total_seconds() / 60.0
        if dt_min > 0:
            fading = (ip_fading_ref_min / dt_min) ** ip_fading_exponent
            PSL_signal = PSL_signal * fading

    # --- Step 3: Background subtraction (bg also gets fading）-------------
    PSL_signal_raw = PSL_signal.copy()  # save before overwriting
    bg_row_to_psl = {}
    if bg_centerline is not None and len(bg_centerline) >= 2:
        for i in range(len(R_y)):
            yi = int(round(R_y[i]))
            track_cols = np.where(track_mask[yi] > 0)[0]
            if len(track_cols) == 0:
                continue
            bg_x = _centerline_x_at_y(bg_centerline, R_y[i])
            width = len(track_cols)
            start = int(round(bg_x - width / 2.0))
            if start < 0:
                start = 0
            if start + width > w:
                start = max(0, w - width)
            cols = np.arange(start, start + width)
            if len(cols) == 0:
                continue
            bg_psl = float(ip_image[yi, cols].sum())
            if fading != 1.0:
                bg_psl *= fading
            bg_row_to_psl[yi] = bg_psl
        _dbg(f"[bg] aligned bg rows: {len(bg_row_to_psl)}")
    elif bg_polygon is not None and len(bg_polygon) >= 3:
        bg_mask = _polygon_to_binary_mask(bg_polygon, h, w)
        if bg_mask is not None and bg_mask.sum() > 0:
            # Debug: pixel-level bg mask width
            bg_mask_rows, bg_mask_cols = np.where(bg_mask)
            if len(bg_mask_rows) > 0:
                from collections import Counter
                bg_row_widths = Counter(bg_mask_rows)
                bg_w_vals = list(bg_row_widths.values())
                _dbg(f"[mask] bg   : rows={len(bg_row_widths)}  width min/avg/max = {min(bg_w_vals)}/{np.mean(bg_w_vals):.1f}/{max(bg_w_vals)}")

            bg_y, PSL_bg = _rowwise_psl_sum(ip_image, bg_mask)
            if fading != 1.0:
                PSL_bg = PSL_bg * fading

            # --- Direct same-Y subtraction (no interpolation) ---
            # Background polygon's Y range covers track's Y range, and both
            # have the same width, so for each track row R_y[i] we take the
            # bg PSL sum at the SAME row int(round(R_y[i])) and subtract
            # directly.  No interp1d smoothing/extrapolation involved.
            for _yy, _pp in zip(bg_y, PSL_bg):
                bg_row_to_psl[int(round(_yy))] = float(_pp)
            _dbg(f"[bg] bg rows covered: {len(bg_row_to_psl)}  y=[{int(min(bg_row_to_psl))},{int(max(bg_row_to_psl))}]")


    PSL_bg_interp = np.zeros(len(R_y), dtype=np.float64)
    _missing = 0
    for i in range(len(R_y)):
        yi = int(round(R_y[i]))
        if yi in bg_row_to_psl:
            PSL_bg_interp[i] = bg_row_to_psl[yi]
        else:
            _missing += 1  # row not in bg mask: leave bg=0
    if _missing > 0:
        _dbg(f"[bg] WARNING: {_missing}/{len(R_y)} track rows had no matching bg row (left as 0)")
    if bg_row_to_psl:
        PSL_signal = PSL_signal - PSL_bg_interp

    PSL_net = np.maximum(PSL_signal, 0.0) if not skip_bg else PSL_signal_raw

    # --- Step 3.5: Convert row-sum PSL → PSL/mm² ---------------------------
    # Martin et al. 2022 calibration coefficients were fitted against
    # Multi Gauge PSL/mm² values.  Our _rowwise_psl_sum() returns the sum
    # of PSL/pixel across the track width at each row.  We must divide by
    # the row area (width_in_pixels × pixel_area_mm²) to get PSL/mm².
    pixel_area_mm2 = (pixel_to_um / 1000.0) ** 2
    row_widths_pixels = track_mask.sum(axis=1)[R_y.astype(np.int64)]
    # Safety: clip to ≥1 to avoid division by zero
    row_widths_pixels = np.maximum(row_widths_pixels, 1)
    row_area_mm2 = row_widths_pixels * pixel_area_mm2
    PSL_mm2 = PSL_net / row_area_mm2
    # 不除以面积
    # PSL_mm2 = PSL_net
    # Debug: dump per-row PSL to file (only for normal run, not skip_bg)
    if not skip_bg:
        _bg_arr = locals().get("PSL_bg_interp", None)
        with open(r"C:\Users\Lenovo\Desktop\1.txt", "w", encoding="utf-8") as _f:
            _f.write(f"{'row':>5s}  {'R_mm':>10s}  {'PSL_raw':>14s}  {'PSL_bg':>14s}  {'PSL_net':>14s}  {'PSL_mm2':>14s}\n")
            for _j in range(len(R_y)):
                _bg_val = _bg_arr[_j] if _bg_arr is not None else 0.0
                _f.write(f"{_j:5d}  {R_mm[_j]:10.4f}  {PSL_signal_raw[_j]:14.6e}  {_bg_val:14.6e}  {PSL_net[_j]:14.6e}  {PSL_mm2[_j]:14.6e}\n")

    # --- Step 4: R(mm) → E_ip (MeV, energy at IP plate) ------------------
    E_ip = np.asarray(r_to_e_func(R_mm), dtype=np.float64)
    # Clip non-physical negative energies from extrapolation beyond table range
    E_ip = np.maximum(E_ip, 1e-6)

    # --- Step 5: Al filter correction → true incident energy --------------
    if al_e_to_range is not None and al_range_to_e is not None:
        R01 = np.asarray(al_e_to_range(E_ip), dtype=np.float64)
        R02 = R01 - al_density * al_thickness_cm
        R02 = np.maximum(R02, 0.0)
        E_incident = np.asarray(al_range_to_e(R02), dtype=np.float64)
    else:
        E_incident = E_ip

    # --- Step 6: PSL → proton number (Martin et al. 2022) -----------------
    # Note: PSL_mm2 (PSL/mm²) is used here because Martin's calibration
    # coefficients (C_LO, P_LO, C_HI, P_HI) were fitted against Multi Gauge
    # PSL/mm² data.
    N_proton = np.zeros(len(PSL_mm2), dtype=np.float64)
    for i in range(len(PSL_mm2)):
        E2 = E_incident[i]
        if E2 <= 0:
            N_proton[i] = 0.0
        elif E2 < MARTIN_E_THRESH:
            N_proton[i] = PSL_mm2[i] / (MARTIN_C_LO * E2 ** MARTIN_P_LO)
        else:
            N_proton[i] = PSL_mm2[i] / (MARTIN_C_HI * E2 ** MARTIN_P_HI)

    # --- Step 7: Fixed dE binning → dN/dE/dΩ ------------------------------
    abs_dE = abs(dE_MeV)
    # Edges from E_bin_max down to E_bin_min (MATLAB: 60:-0.2:1)
    edges = np.arange(E_bin_max, E_bin_min - abs_dE * 0.5, -abs_dE)[::-1]
    if len(edges) < 2:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])

    dN_dE = np.zeros(len(centers), dtype=np.float64)
    # MATLAB bin rule: EEE is descending [60, 59.8, ..., 1]; for each proton,
    #   bin i_dN matches when E2 <= E <= E1, and first-match + break means
    #   boundary values go to the higher-energy bin.
    # Here edges is ascending [1.0, 1.2, ..., 60.0], so we use left-closed-
    # right-open [edges[i], edges[i+1]) for all bins except the last, which
    # includes the upper edge E_bin_max.
    for i in range(len(centers)):
        if i < len(centers) - 1:
            mask = (E_incident >= edges[i]) & (E_incident < edges[i + 1])
        else:
            mask = (E_incident >= edges[i]) & (E_incident <= edges[i + 1])
        if mask.any():
            dN_dE[i] = N_proton[mask].sum() / abs_dE

    sa = max(solid_angle, 1e-12)
    dN_dE_dOmega = dN_dE / sa

    # Debug: print pipeline stats to terminal
    print("=== Spectrum pipeline stats ===")
    print(f"  R_mm:  min={R_mm.min():.2f}  max={R_mm.max():.2f}  n={len(R_mm)}")
    print(f"  PSL_signal:  min={PSL_signal.min():.4e}  max={PSL_signal.max():.4e}  mean={PSL_signal.mean():.4e}")
    print(f"  PSL_net:  min={PSL_net.min():.4e}  max={PSL_net.max():.4e}  sum={PSL_net.sum():.4e}")
    print(f"  E_ip:        min={E_ip.min():.4f}  max={E_ip.max():.4f}")
    print(f"  E_incident:  min={E_incident.min():.4f}  max={E_incident.max():.4f}  n>0={(E_incident > 0).sum()}")
    print(f"  N_proton:  min={N_proton.min():.4e}  max={N_proton.max():.4e}  sum={N_proton.sum():.4e}")
    print(f"  dN_dE_dOmega:  min={dN_dE_dOmega.min():.4e}  max={dN_dE_dOmega.max():.4e}  sum={dN_dE_dOmega.sum():.4e}")
    print(f"  solid_angle={solid_angle:.6e}  dt={dt_min or 0:.1f} min  fading={fading:.4f}")
    # Dump bin coverage: which E bins have data?
    nonzero_bins = np.where(dN_dE > 0)[0]
    print(f"  Non-zero bins: {len(nonzero_bins)}/{len(centers)}  E_range=[{E_incident.min():.3f}, {E_incident.max():.3f}] MeV  skip_bg={skip_bg}")
    if len(nonzero_bins) > 0:
        print(f"  First bin: E={centers[nonzero_bins[0]]:.1f}  dN_dE_dOmega={dN_dE_dOmega[nonzero_bins[0]]:.4e}")
        print(f"  Last bin:  E={centers[nonzero_bins[-1]]:.1f}  dN_dE_dOmega={dN_dE_dOmega[nonzero_bins[-1]]:.4e}")
    # Print first 10 and last 5 bin values
    n_show = min(len(centers), 10)
    print(f"  --- First {n_show} bins ---")
    print(f"  {'E_center':>8s}  {'dN_dE_dOmega':>14s}")
    for i in range(n_show):
        print(f"  {centers[i]:8.2f}  {dN_dE_dOmega[i]:14.6e}")
    if len(centers) > n_show + 5:
        print(f"  ... ({len(centers) - n_show - 5} bins omitted) ...")
        for i in range(-5, 0):
            print(f"  {centers[i]:8.2f}  {dN_dE_dOmega[i]:14.6e}")
    # Also log bin coverage
    _dbg(f"[bins] skip_bg={skip_bg}  E_incident=[{E_incident.min():.3f},{E_incident.max():.3f}] MeV  non-zero={len(nonzero_bins)}/{len(centers)} bins")
    for i in range(len(centers)):
        _dbg(f"  bin {i}: E={centers[i]:.2f}  dNdEdO={dN_dE_dOmega[i]:.6e}")

    return {
        "E_edges": edges,
        "E_centers": centers,
        "dN_dE_dOmega": dN_dE_dOmega,
        "N_total": float(N_proton.sum()),
        "E_min": float(edges[0]),
        "E_max": float(edges[-1]),
    }

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_spectrum_csv(result, dir_path, label):
    """Save spectrum as CSV: E_center(MeV), dN/dE/dΩ."""
    path = os.path.join(dir_path, f"{label}_spectrum.csv")
    np.savetxt(
        path,
        np.column_stack([result["E_centers"], result["dN_dE_dOmega"]]),
        delimiter=",",
        header="E_center_MeV,dN_dE_dOmega",
        fmt="%.6e",
        comments="",
    )


def save_spectrum_plot(result, dir_path, label, ylim_min=5e8, ylim_max=1e13,
                       cutoff_energy_MeV=62.0):
    """Save spectrum plot as PNG (log-scale Y, MATLAB风格).

    Args:
        ylim_min, ylim_max: Y-axis limits for log-scale plots.
            Default 5e8, 1e13 (MATLAB: axis([0 70 0.5e9 1e13])).
        cutoff_energy_MeV: Vertical dashed line (MATLAB: hh=62).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = result["E_centers"]
    spec = result["dN_dE_dOmega"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(E, spec, "b-", linewidth=1.5, label="H+")
    ax.set_xlabel("Proton energy (MeV)", fontsize=14, fontname="Times New Roman")
    ax.set_ylabel("dN/dE/dΩ (H+/MeV/sr)", fontsize=14, fontname="Times New Roman")
    ax.set_title(f"{label}  Energy Spectrum", fontsize=14)
    positive = spec > 0
    if positive.any():
        ax.set_yscale("log")
        ax.set_ylim(ylim_min, ylim_max)
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel("dN/dE/dΩ (H+/MeV/sr) — ALL ZERO", fontsize=12, color="red")
    ax.set_xlim(0, 70)            # MATLAB: axis([0 70 ...])

    # High-energy cutoff (MATLAB: hh=62; plot([hh hh],[1e7 1e15],'k--'))
    if cutoff_energy_MeV is not None and 0 < cutoff_energy_MeV < 70:
        ax.axvline(x=cutoff_energy_MeV, color="k", linestyle="--",
                   linewidth=1.5, label="Cutoff")
        ax.legend(loc="upper right", fontsize=11,
                  prop={"family": "Times New Roman"})

    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    path = os.path.join(dir_path, f"{label}_spectrum.png")
    plt.show(block=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
