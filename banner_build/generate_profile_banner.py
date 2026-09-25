import os
import sys
import math
import re
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, label
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from svgpath2mpl import parse_path

def fmt(num):
    s = f"{num:.1f}"
    return s[:-2] if s.endswith(".0") else s

def run_pipeline():
    print("=== Phase 1 Banner Generator Starting ===")
    
    photo_path = r"D:\Downloads\image0.jpg"
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Source photo not found at {photo_path}")
    
    img = Image.open(photo_path)
    w, h = img.size
    print(f"Original photo size: {w}x{h}")
    
    # 1. Crop head + shoulders, target aspect 300 / 340
    target_aspect = 300.0 / 340.0
    crop_h = int(w / target_aspect)
    top = (h - crop_h) // 2
    bottom = top + crop_h
    cropped = img.crop((0, top, w, bottom))
    resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
    print("Cropped to head+shoulders and resized to 300x340 grid")
    
    # 2. Preprocessing:
    # Contrast 1.3x only, with autocontrast(cutoff=1) + UnsharpMask(radius=3, percent=140)
    gray = ImageOps.grayscale(resized)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
    # 3. Segmentation for Dark Mode:
    # Background removal using color distance, binary closing, fill holes, keep largest component
    arr_rgb = np.array(resized).astype(np.float32)
    bg_sample = arr_rgb[:15, :].mean(axis=(0,1))
    dist = np.linalg.norm(arr_rgb - bg_sample, axis=-1)
    mask_raw = dist > 30.0
    closed = binary_closing(mask_raw, structure=np.ones((5,5)))
    filled = binary_fill_holes(closed)
    lbl, num = label(filled)
    sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
    largest_idx = np.argmax(sizes) + 1
    subject_mask = (lbl == largest_idx)
    
    # 4. Floyd-Steinberg dither (serpentine)
    # Dark mode: dots draw lit subject, hard-clear error bleed at mask edge
    arr_dark = np.array(gray).astype(np.float32) / 255.0
    arr_dark[~subject_mask] = 0.0
    
    # Light mode: dots draw dark parts of photo, keep background
    arr_light = 1.0 - (np.array(gray).astype(np.float32) / 255.0)
    
    def dither_serpentine(in_arr, mask=None):
        buf = in_arr.copy()
        h_grid, w_grid = buf.shape
        out = np.zeros((h_grid, w_grid), dtype=np.uint8)
        for r in range(h_grid):
            is_even = (r % 2 == 0)
            cols = range(w_grid) if is_even else range(w_grid - 1, -1, -1)
            for c in cols:
                if mask is not None and not mask[r, c]:
                    buf[r, c] = 0.0
                    continue
                old_val = buf[r, c]
                new_val = 1.0 if old_val >= 0.5 else 0.0
                out[r, c] = int(new_val)
                err = old_val - new_val
                if is_even:
                    if c + 1 < w_grid and (mask is None or mask[r, c + 1]):
                        buf[r, c + 1] += err * (7.0 / 16.0)
                    if r + 1 < h_grid:
                        if c - 1 >= 0 and (mask is None or mask[r + 1, c - 1]):
                            buf[r + 1, c - 1] += err * (3.0 / 16.0)
                        if mask is None or mask[r + 1, c]:
                            buf[r + 1, c] += err * (5.0 / 16.0)
                        if c + 1 < w_grid and (mask is None or mask[r + 1, c + 1]):
                            buf[r + 1, c + 1] += err * (1.0 / 16.0)
                else:
                    if c - 1 >= 0 and (mask is None or mask[r, c - 1]):
                        buf[r, c - 1] += err * (7.0 / 16.0)
                    if r + 1 < h_grid:
                        if c + 1 < w_grid and (mask is None or mask[r + 1, c + 1]):
                            buf[r + 1, c + 1] += err * (3.0 / 16.0)
                        if mask is None or mask[r + 1, c]:
                            buf[r + 1, c] += err * (5.0 / 16.0)
                        if c - 1 >= 0 and (mask is None or mask[r + 1, c - 1]):
                            buf[r + 1, c - 1] += err * (1.0 / 16.0)
        return out
    
    dither_dark = dither_serpentine(arr_dark, mask=subject_mask)
    dither_light = dither_serpentine(arr_light, mask=None)
    
    dark_dots_count = int(np.sum(dither_dark))
    light_dots_count = int(np.sum(dither_light))
    dark_ink_coverage = dark_dots_count / (300.0 * 340.0)
    light_ink_coverage = light_dots_count / (300.0 * 340.0)
    print(f"Dither completed: Dark dots = {dark_dots_count} ({dark_ink_coverage*100:.2f}% ink), Light dots = {light_dots_count} ({light_ink_coverage*100:.2f}% ink)")
    
    build_dir = r"C:\Users\Lenovo\Shivxng3010\banner_build"
    np.save(os.path.join(build_dir, "portrait_dark_dither.npy"), dither_dark)
    np.save(os.path.join(build_dir, "portrait_light_dither.npy"), dither_light)
    
    y_dark, x_dark = np.where(dither_dark == 1)
    dots_dark = np.vstack([x_dark, y_dark]).T
    
    y_light, x_light = np.where(dither_light == 1)
    dots_light = np.vstack([x_light, y_light]).T
    
    # 5. Intro groups: 60 interleaved groups (run-paired for optimal byte density)
    np.random.seed(42)
    grid_groups_dark = np.random.randint(0, 60, (340, 150))
    intro_groups_dark = grid_groups_dark[dots_dark[:, 1], dots_dark[:, 0] // 2]
    
    grid_groups_light = np.random.randint(0, 60, (340, 150))
    intro_groups_light = grid_groups_light[dots_light[:, 1], dots_light[:, 0] // 2]
    
    np.save(os.path.join(build_dir, "intro_groups_dark.npy"), intro_groups_dark)
    np.save(os.path.join(build_dir, "intro_groups_light.npy"), intro_groups_light)
    
    def calc_evenness_metric(dots, groups, num_groups=60, bins=(4, 4)):
        xb = np.linspace(0, 300, bins[0] + 1)
        yb = np.linspace(0, 340, bins[1] + 1)
        h_tot, _, _ = np.histogram2d(dots[:, 0], dots[:, 1], bins=[xb, yb])
        p_tot = h_tot / (np.sum(h_tot) + 1e-9)
        
        tvs = []
        for g in range(num_groups):
            mask_g = (groups == g)
            h_g, _, _ = np.histogram2d(dots[mask_g, 0], dots[mask_g, 1], bins=[xb, yb])
            p_g = h_g / (np.sum(h_g) + 1e-9)
            tv = 0.5 * np.sum(np.abs(p_g - p_tot))
            tvs.append(tv)
        return float(np.mean(tvs))
    
    evenness_dark = calc_evenness_metric(dots_dark, intro_groups_dark)
    evenness_light = calc_evenness_metric(dots_light, intro_groups_light)
    print(f"Intro Evenness Metric: Dark = {evenness_dark:.4f}, Light = {evenness_light:.4f} (Target: ~0.05)")
    
    # 6. Drift bands: 94 groups with Gaussian noise (sigma ~ 4)
    np.random.seed(42)
    dots_dark_noisy = dots_dark + np.random.normal(0, 4.0, dots_dark.shape)
    centroids_dark, labels_dark = kmeans2(dots_dark_noisy.astype(np.float32), 94, minit='points', iter=15)
    
    dots_light_noisy = dots_light + np.random.normal(0, 4.0, dots_light.shape)
    centroids_light, labels_light = kmeans2(dots_light_noisy.astype(np.float32), 94, minit='points', iter=15)
    
    np.save(os.path.join(build_dir, "drift_bands_dark.npy"), labels_dark)
    np.save(os.path.join(build_dir, "drift_bands_light.npy"), labels_light)
    
    def calc_straight_boundary_metric(h_grid, w_grid, dots, labels, run_len=7):
        lmap = np.full((h_grid, w_grid), -1, dtype=int)
        lmap[dots[:, 1], dots[:, 0]] = labels
        bh = (lmap[:-1, :] != lmap[1:, :]) & (lmap[:-1, :] != -1) & (lmap[1:, :] != -1)
        bv = (lmap[:, :-1] != lmap[:, 1:]) & (lmap[:, :-1] != -1) & (lmap[:, 1:] != -1)
        tot = np.sum(bh) + np.sum(bv)
        
        run_count = 0
        for r in range(bh.shape[0]):
            cur = 0
            for val in bh[r, :]:
                if val: cur += 1
                else:
                    if cur >= run_len: run_count += cur
                    cur = 0
            if cur >= run_len: run_count += cur
            
        for c in range(bv.shape[1]):
            cur = 0
            for val in bv[:, c]:
                if val: cur += 1
                else:
                    if cur >= run_len: run_count += cur
                    cur = 0
            if cur >= run_len: run_count += cur
            
        return float(run_count / tot) if tot > 0 else 0.0
    
    boundary_dark = calc_straight_boundary_metric(340, 300, dots_dark, labels_dark, 7)
    boundary_light = calc_straight_boundary_metric(340, 300, dots_light, labels_light, 7)
    print(f"Straight-Boundary Metric: Dark = {boundary_dark:.4f}, Light = {boundary_light:.4f} (Target: ~0.01 organic)")
    
    # 7. Logos & Travellers Layer (~900 dots)
    python_d = 'M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z'
    py_path = parse_path(python_d)
    
    pytorch_d = 'M12.005 0L4.952 7.053a9.865 9.865 0 0 0 0 14.022 9.866 9.866 0 0 0 14.022 0c3.984-3.9 3.986-10.205.085-14.023l-1.744 1.743c2.904 2.905 2.904 7.634 0 10.538s-7.634 2.904-10.538 0-2.904-7.634 0-10.538l4.647-4.646.582-.665zm3.568 3.899a1.327 1.327 0 0 0-1.327 1.327 1.327 1.327 0 0 0 1.327 1.328A1.327 1.327 0 0 0 16.9 5.226 1.327 1.327 0 0 0 15.573 3.9z'
    torch_path = parse_path(pytorch_d)
    
    cx, cy = 150.0, 170.0
    logo_size = 185.0
    s_logo = logo_size / 24.0
    x_off = cx - (logo_size / 2.0)
    y_off = cy - (logo_size / 2.0)
    
    gx, gy = np.meshgrid(np.linspace(0, 24, 180), np.linspace(0, 24, 180))
    pts_unit = np.vstack([gx.ravel(), gy.ravel()]).T
    
    mask_py = py_path.contains_points(pts_unit)
    pts_py = pts_unit[mask_py] * s_logo + np.array([x_off, y_off])
    
    mask_torch = torch_path.contains_points(pts_unit)
    pts_torch = pts_unit[mask_torch] * s_logo + np.array([x_off, y_off])
    
    img_code = Image.new('L', (240, 240), 0)
    d_draw = ImageDraw.Draw(img_code)
    sc = 240.0 / 24.0
    lw = int(2.4 * sc)
    d_draw.line([(6*sc, 8*sc), (2*sc, 12*sc), (6*sc, 16*sc)], fill=255, width=lw, joint='round')
    d_draw.line([(18*sc, 8*sc), (22*sc, 12*sc), (18*sc, 16*sc)], fill=255, width=lw, joint='round')
    d_draw.line([(14.5*sc, 4*sc), (9.5*sc, 20*sc)], fill=255, width=lw)
    
    arr_c = np.array(img_code)
    cy_idx, cx_idx = np.where(arr_c > 128)
    pts_code = np.vstack([cx_idx / 240.0 * logo_size + x_off, cy_idx / 240.0 * logo_size + y_off]).T
    
    np.random.seed(42)
    c_py, _ = kmeans2(pts_py, 900, minit='points', iter=15)
    c_torch, _ = kmeans2(pts_torch, 900, minit='points', iter=15)
    c_code, _ = kmeans2(pts_code, 900, minit='points', iter=15)
    
    cost12 = cdist(c_py, c_torch, 'sqeuclidean')
    _, idx12 = linear_sum_assignment(cost12)
    c_torch_matched = c_torch[idx12]
    
    cost23 = cdist(c_torch_matched, c_code, 'sqeuclidean')
    _, idx23 = linear_sum_assignment(cost23)
    c_code_matched = c_code[idx23]
    
    c1_centroid = c_py.mean(axis=0)
    print(f"First logo (Python) centroid: {c1_centroid}")
    
    np.save(os.path.join(build_dir, "traveller_p1.npy"), c_py)
    np.save(os.path.join(build_dir, "traveller_p2.npy"), c_torch_matched)
    np.save(os.path.join(build_dir, "traveller_p3.npy"), c_code_matched)
    
    # 8. SVG Grid and Helpers
    s_grid = 1.18
    x0_grid = 45.0
    y0_grid = 105.4
    
    def encode_runs_for_dots(dot_list):
        if len(dot_list) == 0:
            return ""
        sorted_dots = dot_list[np.lexsort((dot_list[:, 0], dot_list[:, 1]))]
        d_parts = []
        i = 0
        n = len(sorted_dots)
        while i < n:
            r = sorted_dots[i, 1]
            c_start = sorted_dots[i, 0]
            c_end = c_start
            while i + 1 < n and sorted_dots[i+1, 1] == r and sorted_dots[i+1, 0] == c_end + 1:
                c_end += 1
                i += 1
            length = c_end - c_start + 1
            x = x0_grid + c_start * s_grid
            y = y0_grid + r * s_grid + (s_grid / 2.0)
            w_run = length * s_grid
            d_parts.append(f"M{fmt(x)},{fmt(y)}h{fmt(w_run)}")
            i += 1
        return "".join(d_parts)
    
    loop_dur = 14.2
    kt = [
        0.0,
        3.0 / loop_dur,
        4.3 / loop_dur,
        6.3 / loop_dur,
        7.6 / loop_dur,
        9.6 / loop_dur,
        10.9 / loop_dur,
        12.9 / loop_dur,
        1.0
    ]
    kt_str = ";".join([f"{t:.4f}" for t in kt])
    
    rows = [
        ("Subject", "Shivang Soni"),
        ("Role", "Student"),
        ("Origin", "Bhopal, IN"),
        ("Education", "BS in Data Science (IITM)"),
        ("Status", "Learning"),
        ("ToolChain", "Git Bash, VS Code, Figma"),
        ("Core.Lang", "Python"),
        ("Core.Frontend", "HTML, CSS, React"),
        ("Core.Backend", "Node.js"),
        ("Core.Database", "PostgreSQL"),
        ("Core.Infra", "Docker, Vercel"),
        ("Grid.Mail", "shivangsoni30@gmail.com"),
        ("Grid.Portfolio", "Coming Soon"),
        ("Grid.GitHub", "github.com/Shivxng3010"),
    ]
    
    def build_banner_svg(theme="dark"):
        is_dark = (theme == "dark")
        
        bg_color = "#0A101F" if is_dark else "#F8FAFC"
        chrome_color = "#22D3EE" if is_dark else "#0891B2"
        portrait_color = "#A78BFA" if is_dark else "#7C3AED"
        accent_color = "#10B981" if is_dark else "#059669"
        border_color = "#1E293B" if is_dark else "#CBD5E1"
        label_color = "#94A3B8" if is_dark else "#475569"
        dots_color = "#334155" if is_dark else "#CBD5E1"
        val_color = "#F8FAFC" if is_dark else "#0F172A"
        panel_bg = "#0D1527" if is_dark else "#FFFFFF"
        
        active_dots = dots_dark if is_dark else dots_light
        active_labels = labels_dark if is_dark else labels_light
        active_intro = intro_groups_dark if is_dark else intro_groups_light
        
        svg_lines = []
        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">')
        svg_lines.append(f'<defs>')
        svg_lines.append(f'<style>')
        svg_lines.append(f'  @import url("https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&amp;display=swap");')
        svg_lines.append(f'  .mono {{ font-family: "Fira Code", Menlo, Monaco, Consolas, monospace; }}')
        svg_lines.append(f'  .label {{ font-size: 14px; fill: {label_color}; font-weight: 500; }}')
        svg_lines.append(f'  .leader {{ font-size: 14px; fill: {dots_color}; letter-spacing: 2px; }}')
        svg_lines.append(f'  .val {{ font-size: 14px; fill: {val_color}; font-weight: 600; }}')
        svg_lines.append(f'</style>')
        svg_lines.append(f'</defs>')
        
        svg_lines.append(f'<rect width="1180" height="610" rx="12" fill="{bg_color}" stroke="{border_color}" stroke-width="1.5"/>')
        
        svg_lines.append(f'<rect width="1180" height="42" rx="12" fill="{panel_bg}"/>')
        svg_lines.append(f'<rect y="30" width="1180" height="12" fill="{panel_bg}"/>')
        svg_lines.append(f'<line x1="0" y1="42" x2="1180" y2="42" stroke="{border_color}" stroke-width="1"/>')
        
        svg_lines.append(f'<circle cx="25" cy="21" r="6" fill="#EF4444"/>')
        svg_lines.append(f'<circle cx="45" cy="21" r="6" fill="#F59E0B"/>')
        svg_lines.append(f'<circle cx="65" cy="21" r="6" fill="#10B981"/>')
        
        svg_lines.append(f'<text x="590" y="26" text-anchor="middle" class="mono" font-size="13" fill="{label_color}" font-weight="500">profile.sh --live</text>')
        
        # Left frame: VISUAL.MAP
        svg_lines.append(f'<g id="portrait-frame">')
        svg_lines.append(f'<rect x="32" y="58" width="380" height="526" rx="8" fill="{panel_bg}" stroke="{border_color}" stroke-width="1"/>')
        svg_lines.append(f'<text x="45" y="82" class="mono" font-size="13" fill="{chrome_color}" font-weight="700" letter-spacing="1">VISUAL.MAP</text>')
        svg_lines.append(f'<text x="398" y="82" text-anchor="end" class="mono" font-size="11" fill="{label_color}">300×340 [DITHER]</text>')
        svg_lines.append(f'<line x1="32" y1="94" x2="412" y2="94" stroke="{border_color}" stroke-width="1"/>')
        
        svg_lines.append(f'<rect x="42" y="102" width="360" height="408" rx="4" fill="{bg_color}" stroke="{border_color}" stroke-width="1"/>')
        svg_lines.append(f'<text x="45" y="534" class="mono" font-size="11" fill="{label_color}">STATE: DUAL-PHASE · OPTIMAL TRANSPORT</text>')
        svg_lines.append(f'<text x="45" y="556" class="mono" font-size="11" fill="{chrome_color}">MORPH: PYTHON → PYTORCH → &lt;/&gt;</text>')
        svg_lines.append(f'</g>')
        
        # Right frame: SYSTEM.INFO
        svg_lines.append(f'<g id="system-info">')
        svg_lines.append(f'<rect x="432" y="58" width="716" height="526" rx="8" fill="{panel_bg}" stroke="{border_color}" stroke-width="1"/>')
        svg_lines.append(f'<text x="452" y="84" class="mono" font-size="13" fill="{chrome_color}" font-weight="700" letter-spacing="1">SYSTEM.INFO</text>')
        
        svg_lines.append(f'<circle cx="585" cy="80" r="5" fill="#EF4444">')
        svg_lines.append(f'  <animate attributeName="opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite"/>')
        svg_lines.append(f'</circle>')
        svg_lines.append(f'<circle cx="585" cy="80" r="5" fill="none" stroke="#EF4444" stroke-width="1.5">')
        svg_lines.append(f'  <animate attributeName="r" values="5;10" dur="1.5s" repeatCount="indefinite"/>')
        svg_lines.append(f'  <animate attributeName="opacity" values="0.8;0" dur="1.5s" repeatCount="indefinite"/>')
        svg_lines.append(f'</circle>')
        svg_lines.append(f'<text x="598" y="85" class="mono" font-size="12" fill="#EF4444" font-weight="700">LIVE</text>')
        
        svg_lines.append(f'<rect x="990" y="66" width="142" height="30" rx="15" fill="{chrome_color}" fill-opacity="0.12" stroke="{chrome_color}" stroke-width="1"/>')
        svg_lines.append(f'<text x="1061" y="86" text-anchor="middle" class="mono" font-size="14" fill="{chrome_color}" font-weight="600">@Shivxng3010</text>')
        
        svg_lines.append(f'<line x1="432" y1="104" x2="1148" y2="104" stroke="{border_color}" stroke-width="1"/>')
        
        x_left = 454
        x_right = 1128
        char_w = 8.4
        row_y = 135
        spacing = 23
        
        for idx, (label_text, val_text) in enumerate(rows):
            if idx == 6 or idx == 11:
                row_y += 10
                svg_lines.append(f'<line x1="454" y1="{row_y - 14}" x2="1128" y2="{row_y - 14}" stroke="{border_color}" stroke-dasharray="3,3" stroke-width="1"/>')
            
            l_w = len(label_text) * char_w
            v_w = len(val_text) * char_w
            gap = 10.0
            x_dots = x_left + l_w + gap
            x_val = x_right - v_w
            dots_w = x_val - gap - x_dots
            dot_spacing = 7.0
            num_dots = max(2, int(dots_w / dot_spacing))
            leader_str = "." * num_dots
            
            svg_lines.append(f'<g class="mono">')
            svg_lines.append(f'  <text x="{x_left}" y="{row_y}" class="label">{label_text}</text>')
            svg_lines.append(f'  <text x="{fmt(x_dots)}" y="{row_y}" class="leader">{leader_str}</text>')
            svg_lines.append(f'  <text x="{fmt(x_val)}" y="{row_y}" textLength="{fmt(v_w)}" lengthAdjust="spacingAndGlyphs" class="val">{val_text}</text>')
            svg_lines.append(f'</g>')
            
            row_y += spacing
        
        svg_lines.append(f'<line x1="454" y1="520" x2="1128" y2="520" stroke="{border_color}" stroke-width="1"/>')
        svg_lines.append(f'<text x="454" y="546" class="mono" font-size="11" fill="{label_color}">SYS.INTEGRITY: 100% · PROFILE: DATA SCIENCE &amp; ML · STACK: PYTHON/REACT</text>')
        svg_lines.append(f'</g>')
        
        # ANIMATION LAYER 1: Intro (~3.2s, once)
        svg_lines.append(f'<g id="portrait-intro" fill="none" stroke="{portrait_color}" stroke-width="{s_grid}" shape-rendering="crispEdges">')
        svg_lines.append(f'  <animate attributeName="opacity" to="0" begin="3.2s" dur="0.05s" fill="freeze"/>')
        
        for g in range(60):
            group_dots = active_dots[active_intro == g]
            d_str = encode_runs_for_dots(group_dots)
            t_fade = (g / 59.0) * 1.8
            svg_lines.append(f'  <g opacity="0">')
            svg_lines.append(f'    <animate attributeName="opacity" values="0;1" dur="0.25s" begin="{t_fade:.2f}s" fill="freeze"/>')
            svg_lines.append(f'    <path d="{d_str}"/>')
            svg_lines.append(f'  </g>')
        svg_lines.append(f'</g>')
        
        # ANIMATION LAYER 2: Loop Portrait (~14.2s loop)
        svg_lines.append(f'<g id="portrait-loop" opacity="0" fill="none" stroke="{portrait_color}" stroke-width="{s_grid}" shape-rendering="crispEdges">')
        svg_lines.append(f'  <animate attributeName="opacity" to="1" begin="3.2s" dur="0.05s" fill="freeze"/>')
        
        cx_c1, cy_c1 = c1_centroid[0], c1_centroid[1]
        
        for k in range(94):
            band_dots = active_dots[active_labels == k]
            if len(band_dots) == 0:
                continue
            d_str = encode_runs_for_dots(band_dots)
            b_mean = band_dots.mean(axis=0)
            dx = 0.42 * (cx_c1 - b_mean[0]) * s_grid
            dy = 0.42 * (cy_c1 - b_mean[1]) * s_grid
            
            trans_vals = f"0 0; 0 0; {fmt(dx)} {fmt(dy)}; {fmt(dx)} {fmt(dy)}; {fmt(dx)} {fmt(dy)}; {fmt(dx)} {fmt(dy)}; {fmt(dx)} {fmt(dy)}; 0 0; 0 0"
            op_vals = "1; 1; 0; 0; 0; 0; 0; 1; 1"
            
            svg_lines.append(f'  <g>')
            svg_lines.append(f'    <animateTransform attributeName="transform" type="translate" values="{trans_vals}" keyTimes="{kt_str}" dur="{loop_dur}s" repeatCount="indefinite" begin="3.2s"/>')
            svg_lines.append(f'    <animate attributeName="opacity" values="{op_vals}" keyTimes="{kt_str}" dur="{loop_dur}s" repeatCount="indefinite" begin="3.2s"/>')
            svg_lines.append(f'    <path d="{d_str}"/>')
            svg_lines.append(f'  </g>')
        svg_lines.append(f'</g>')
        
        # ANIMATION LAYER 3: Travellers (~900 dots)
        travel_color = chrome_color
        svg_lines.append(f'<g id="travellers" fill="{travel_color}">')
        trav_op = "0; 0; 1; 1; 1; 1; 1; 1; 0"
        svg_lines.append(f'  <animate attributeName="opacity" values="{trav_op}" keyTimes="{kt_str}" dur="{loop_dur}s" repeatCount="indefinite" begin="3.2s"/>')
        
        for i in range(900):
            p1 = c_py[i]
            p2 = c_torch_matched[i]
            p3 = c_code_matched[i]
            
            x1 = x0_grid + p1[0] * s_grid
            y1 = y0_grid + p1[1] * s_grid
            x2 = x0_grid + p2[0] * s_grid
            y2 = y0_grid + p2[1] * s_grid
            x3 = x0_grid + p3[0] * s_grid
            y3 = y0_grid + p3[1] * s_grid
            
            vx = f"{fmt(x1)};{fmt(x1)};{fmt(x1)};{fmt(x1)};{fmt(x2)};{fmt(x2)};{fmt(x3)};{fmt(x3)};{fmt(x1)}"
            vy = f"{fmt(y1)};{fmt(y1)};{fmt(y1)};{fmt(y1)};{fmt(y2)};{fmt(y2)};{fmt(y3)};{fmt(y3)};{fmt(y1)}"
            
            svg_lines.append(f'  <circle cx="{fmt(x1)}" cy="{fmt(y1)}" r="1.3">')
            svg_lines.append(f'    <animate attributeName="cx" values="{vx}" keyTimes="{kt_str}" dur="{loop_dur}s" repeatCount="indefinite" begin="3.2s"/>')
            svg_lines.append(f'    <animate attributeName="cy" values="{vy}" keyTimes="{kt_str}" dur="{loop_dur}s" repeatCount="indefinite" begin="3.2s"/>')
            svg_lines.append(f'  </circle>')
            
        svg_lines.append(f'</g>')
        
        svg_lines.append(f'</svg>')
        return "\n".join(svg_lines)
    
    dark_svg = build_banner_svg("dark")
    light_svg = build_banner_svg("light")
    
    dark_path = os.path.join(build_dir, "dark.svg")
    light_path = os.path.join(build_dir, "light.svg")
    repo_dark = r"C:\Users\Lenovo\Shivxng3010\dark.svg"
    repo_light = r"C:\Users\Lenovo\Shivxng3010\light.svg"
    
    with open(dark_path, "w", encoding="utf-8") as f:
        f.write(dark_svg)
    with open(light_path, "w", encoding="utf-8") as f:
        f.write(light_svg)
        
    with open(repo_dark, "w", encoding="utf-8") as f:
        f.write(dark_svg)
    with open(repo_light, "w", encoding="utf-8") as f:
        f.write(light_svg)
        
    dark_kb = len(dark_svg.encode("utf-8")) / 1024.0
    light_kb = len(light_svg.encode("utf-8")) / 1024.0
    
    print(f"dark.svg generated: {dark_kb:.1f} KB")
    print(f"light.svg generated: {light_kb:.1f} KB")
    print("=== Phase 1 Banner Generation Finished Successfully ===")
    
    return {
        "dark_dots": dark_dots_count,
        "light_dots": light_dots_count,
        "dark_ink_coverage": dark_ink_coverage,
        "light_ink_coverage": light_ink_coverage,
        "evenness_dark": evenness_dark,
        "evenness_light": evenness_light,
        "boundary_dark": boundary_dark,
        "boundary_light": boundary_light,
        "dark_kb": dark_kb,
        "light_kb": light_kb
    }

if __name__ == "__main__":
    run_pipeline()
