# -*- coding: utf-8 -*-
"""能量谱计算工具（供 X-AnyLabeling 调用和命令行使用）。"""
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def polygon_centroid(pts):
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 3:
        return pts.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if abs(area) < 1e-6:
        return pts.mean(axis=0)
    cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6.0 * area)
    cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6.0 * area)
    return np.array([cx, cy])


def centerline_to_band_polygon(centerline, half_width):
    if len(centerline) < 2:
        return centerline
    n = len(centerline)
    left, right = [], []
    for i in range(n):
        if i == 0:
            dx = centerline[1][0] - centerline[0][0]
            dy = centerline[1][1] - centerline[0][1]
        elif i == n - 1:
            dx = centerline[-1][0] - centerline[-2][0]
            dy = centerline[-1][1] - centerline[-2][1]
        else:
            dx = centerline[i + 1][0] - centerline[i - 1][0]
            dy = centerline[i + 1][1] - centerline[i - 1][1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        nx, ny = -dy / length, dx / length
        left.append((centerline[i][0] + nx * half_width, centerline[i][1] + ny * half_width))
        right.append((centerline[i][0] - nx * half_width, centerline[i][1] - ny * half_width))
    if len(left) < 2:
        return centerline
    return left + list(reversed(right))


def get_band_polygon(shape):
    pts = shape.get("points", [])
    if not pts:
        return []
    shape_type = shape.get("shape_type", "")

    # X-AnyLabeling 保存格式: centerline/track_width 可能在顶级或 other_data 内
    cl = shape.get("centerline") or (shape.get("other_data") or {}).get("centerline")
    tw = shape.get("track_width") or (shape.get("other_data") or {}).get("track_width", 10)

    if shape_type == "polygon" and len(pts) >= 3:
        return [(p[0], p[1]) for p in pts]

    if cl and len(cl) >= 2:
        return centerline_to_band_polygon(cl, tw / 2.0)

    return [(p[0], p[1]) for p in pts]


def row_sum_within_polygon(image, polygon, y):
    H, W = image.shape
    if y < 0 or y >= H or len(polygon) < 3:
        return 0.0
    pts = np.array(polygon, dtype=np.float64)
    n = len(pts)
    xs = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            if abs(y2 - y1) < 1e-8:
                continue
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    total = 0.0
    for i in range(0, len(xs) - 1, 2):
        x_start = max(0, int(np.floor(xs[i])))
        x_end = min(W - 1, int(np.ceil(xs[i + 1])))
        if x_end >= x_start:
            total += float(image[y, x_start:x_end + 1].sum())
    return total


def _get_plate_id(shape):
    """plate_id 可能在顶级或 other_data 内 (X-AnyLabeling 两种保存格式)"""
    pid = shape.get("plate_id")
    if pid is not None:
        return pid
    return (shape.get("other_data") or {}).get("plate_id")


def _get_plate_bbox(shape):
    bbox = shape.get("plate_bbox")
    if bbox is not None:
        return bbox
    return (shape.get("other_data") or {}).get("plate_bbox")


def group_by_plate(shapes):
    known = {}
    orphans = []

    for s in shapes:
        pid = _get_plate_id(s)
        bbox = _get_plate_bbox(s)
        label = s["label"]

        if pid is not None:
            if pid not in known:
                known[pid] = {"tracks": [], "sources": [], "bbox": bbox}
            elif bbox and known[pid]["bbox"] is None:
                known[pid]["bbox"] = bbox
            if label.startswith("track"):
                known[pid]["tracks"].append(s)
            elif label.startswith("source"):
                known[pid]["sources"].append(s)
        else:
            orphans.append(s)

    for s in orphans:
        s_pts = np.array(s["points"], dtype=np.float64)
        sc = s_pts.mean(axis=0) if len(s_pts) > 0 else np.array([0.0, 0.0])
        label = s["label"]
        assigned = False
        for pid, pd in known.items():
            bbox = pd["bbox"]
            if bbox and bbox[0] <= sc[0] <= bbox[0] + bbox[2] and bbox[1] <= sc[1] <= bbox[1] + bbox[3]:
                if label.startswith("track"):
                    pd["tracks"].append(s)
                elif label.startswith("source"):
                    pd["sources"].append(s)
                assigned = True
                break
        if not assigned:
            pid = -1
            if pid not in known:
                known[pid] = {"tracks": [], "sources": [], "bbox": None}
            if label.startswith("track"):
                known[pid]["tracks"].append(s)
            elif label.startswith("source"):
                known[pid]["sources"].append(s)

    return {pid: {"tracks": p["tracks"], "sources": p["sources"]}
            for pid, p in known.items()}


def _ccw(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    return (_ccw(a, b, c) * _ccw(a, b, d) < 0 and
            _ccw(c, d, a) * _ccw(c, d, b) < 0)


def _polygon_is_simple(polygon):
    """True if no non-adjacent edges intersect."""
    n = len(polygon)
    if n < 4:
        return True
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            c, d = polygon[j], polygon[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def load_image_as_uint8(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
    if img.dtype != np.uint8:
        log_img = np.log1p(img.astype(np.float32))
        vmin, vmax = np.percentile(log_img, [1, 99.9])
        img = np.clip((log_img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    return img


def compute_spectra(image_path, shapes, output_dir, scale=1.0, smooth=3,
                    progress_callback=None):
    """计算所有 plate 的能谱，输出 CSV+PNG。

    Args:
        image_path: 原始图片路径
        shapes:    shape 字典列表
        output_dir: 输出目录
        scale:    能量比例因子
        smooth:   滑动平均窗口
        progress_callback: 可选回调 (message: str)
    Returns:
        dict: {"plates": valid_count, "tracks": total_computed, "csv_files": [...], "png_files": [...]}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if progress_callback:
            progress_callback("matplotlib 未安装，无法生成能谱图")
        return None

    img = load_image_as_uint8(image_path)
    H, W = img.shape
    plates = group_by_plate(shapes)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    valid_count = 0
    csv_files = []
    png_files = []

    for pid in sorted(plates.keys()):
        pd = plates[pid]
        n_src = len(pd["sources"])
        n_trk = len(pd["tracks"])

        if n_src != 1 or n_trk == 0:
            continue

        # source 多边形很小，直接用各顶点 y 均值作为原点 y
        src_pts = pd["sources"][0]["points"]
        origin_y = sum(p[1] for p in src_pts) / max(len(src_pts), 1)
        print(f"[energy_spectrum] Plate {pid} source={pd['sources'][0].get('label','?')} "
              f"origin_y={origin_y:.0f} n_tracks={n_trk}")

        # 打印每条轨迹底部 x 坐标，验证排序
        for ti, track in enumerate(pd["tracks"]):
            cl = track.get("centerline") or (track.get("other_data") or {}).get("centerline")
            if cl:
                bottom = max(cl, key=lambda p: p[1])
                x = bottom[0]
            else:
                pts = track.get("points", [])
                if pts:
                    max_y = max(p[1] for p in pts)
                    x = min(p[0] for p in pts if p[1] == max_y)
                else:
                    x = 0.0
            print(f"  [{ti}] {track.get('label','?')} bottom_x={x:.0f}")

        for ti, track in enumerate(pd["tracks"]):
            band = get_band_polygon(track)
            if len(band) < 3:
                continue

            if not _polygon_is_simple(band):
                if progress_callback:
                    progress_callback(
                        f"Plate {pid} {track.get('label', f'track_{ti+1}')}: band 自相交，跳过"
                    )
                continue

            band_np = np.array(band, dtype=np.float64)
            y_min = max(0, int(band_np[:, 1].min()))
            y_max = min(H - 1, int(band_np[:, 1].max()))

            # 调试
            e_min = (origin_y - y_max) * scale
            e_max = (origin_y - y_min) * scale
            print(f"[energy_spectrum] Plate {pid} {track.get('label', '?')}: "
                  f"origin_y={origin_y:.0f} band y=[{y_min},{y_max}] energy=[{e_min:.0f},{e_max:.0f}]")
            if progress_callback:
                progress_callback(
                    f"Plate {pid} {track.get('label', '?')}: energy=[{e_min:.0f},{e_max:.0f}]"
                )

            energies, counts = [], []
            for y in range(y_min, y_max + 1):
                # 从下往上扫：下方 y 大，上方 y 小。能量 = origin_y - y
                energy = (origin_y - y) * scale
                if energy < 0:
                    continue
                count = row_sum_within_polygon(img, band, y)
                if count > 0:
                    energies.append(energy)
                    counts.append(count)

            if not energies:
                continue

            # 按能量从小到大排序
            energies = np.array(energies, dtype=np.float64)
            counts = np.array(counts, dtype=np.float64)
            order = np.argsort(energies)
            energies = energies[order]
            counts = counts[order]

            if smooth > 1 and len(counts) >= smooth:
                kernel = np.ones(smooth) / smooth
                counts = np.convolve(counts, kernel, mode="same")

            c_max = counts.max()

            # CSV
            csv_path = output_dir / f"spectrum_{stem}_plate{pid}_track{ti + 1}.csv"
            np.savetxt(csv_path, np.column_stack([energies, counts]),
                       delimiter=",", fmt="%.3f,%.1f", header="energy,count", comments="")
            csv_files.append(str(csv_path))

            # 单轨迹 PNG
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(energies, counts, color="#2196F3", linewidth=1.5)
            ax.fill_between(energies, 0, counts, alpha=0.2, color="#2196F3")
            ax.set_xlabel("Energy (px)")
            ax.set_ylabel("Particle Count")
            ax.set_title(f"Plate {pid} Track {ti + 1}  —  {len(energies)} pts, max={c_max:.0f}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png_path = output_dir / f"spectrum_{stem}_plate{pid}_track{ti + 1}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            png_files.append(str(png_path))

        valid_count += 1

    if progress_callback:
        progress_callback(
            f"能谱计算完成: {valid_count} 个 plate, {len(csv_files)} 条轨迹"
        )

    return {
        "plates": valid_count,
        "tracks": len(csv_files),
        "csv_files": csv_files,
        "png_files": png_files,
    }
