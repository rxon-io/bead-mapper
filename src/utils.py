import pandas as pd
import numpy as np
from collections import Counter

# --- PDF ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def color_count_report(used_indices: np.ndarray, palette_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a 1D array of palette indices (by blocks, row-major) OR a 2D index map.
    Returns a color count table (brand, code, color_name, r, g, b, count),
    sorted by brand / code / name.
    """
    if used_indices.ndim == 2:
        flat = used_indices.ravel()
    else:
        flat = used_indices
    cnt = Counter(flat.tolist())
    rows = []
    for idx, n in cnt.items():
        row = palette_df.iloc[int(idx)]
        rows.append({
            "palette_idx": int(idx),
            "brand": row.get("brand", ""),
            "code": row.get("code", ""),
            "color_name": row.get("color_name", ""),
            "r": int(row["r"]), "g": int(row["g"]), "b": int(row["b"]),
            "count": int(n)
        })
    df = pd.DataFrame(rows)
    # neat sorting: brand -> code (as string to preserve leading letters) -> name
    if not df.empty:
        df = df.sort_values(["brand", "code", "color_name"], kind="stable").reset_index(drop=True)
    return df


def build_legend(used_index_map: np.ndarray, palette_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a legend and color numbering. The legend number (legend_id) is assigned
    in order of first appearance, left-to-right, top-to-bottom.
    Returns a DataFrame with columns:
    [legend_id, palette_idx, brand, code, color_name, r, g, b, count]
    """
    # order of appearance
    flat = used_index_map.ravel().tolist()
    seen = {}
    order = []
    for idx in flat:
        if idx not in seen:
            seen[idx] = True
            order.append(int(idx))

    # count occurrences
    cnt = Counter(flat)

    rows = []
    for i, pidx in enumerate(order, start=1):
        row = palette_df.iloc[int(pidx)]
        rows.append({
            "legend_id": i,
            "palette_idx": int(pidx),
            "brand": row.get("brand", ""),
            "code": row.get("code", ""),
            "color_name": row.get("color_name", ""),
            "r": int(row["r"]), "g": int(row["g"]), "b": int(row["b"]),
            "count": int(cnt[pidx]),
        })
    return pd.DataFrame(rows)


def generate_pattern_pdf(
    index_map: np.ndarray,
    legend_df: pd.DataFrame,
    filename: str = "pattern.pdf",
    cell_mm: float = 6.0,
    show_cell_color: bool = True,
    show_numbers_on_cells: bool = True
) -> str:
    """
    Generates a PDF with a grid (color numbers inside cells) and a legend.

    index_map: 2D numpy array (rows x cols) with palette_idx for each cell.
    legend_df: table from build_legend() with columns
               [legend_id, palette_idx, brand, code, color_name, r, g, b, count].
    filename: output PDF path.
    cell_mm: cell size in millimeters.
    show_cell_color: fill cells with the actual bead color.
    show_numbers_on_cells: print legend numbers inside each cell.
    """
    # prepare mapping: palette_idx -> legend_id and -> RGB
    pid_to_legend = {int(r.palette_idx): int(r.legend_id) for _, r in legend_df.iterrows()}
    pid_to_rgb = {int(r.palette_idx): (int(r.r), int(r.g), int(r.b)) for _, r in legend_df.iterrows()}

    # page size
    page_w, page_h = landscape(A4)  # in points (pt). 1 pt = 1/72 inch
    mm_to_pt = 72.0 / 25.4
    cell = cell_mm * mm_to_pt

    rows, cols = index_map.shape

    # compute grid area
    margin = 20 * mm_to_pt
    grid_w_pt = cols * cell
    grid_h_pt = rows * cell

    # if grid is too large — scale cell down (keeping proportions)
    max_grid_w = page_w - 2 * margin
    max_grid_h = page_h - 60 * mm_to_pt  # leave space for legend
    scale = min(1.0, max_grid_w / max(1.0, grid_w_pt), max_grid_h / max(1.0, grid_h_pt))
    if scale < 1.0:
        cell *= scale
        grid_w_pt = cols * cell
        grid_h_pt = rows * cell

    start_x = (page_w - grid_w_pt) / 2.0
    start_y = page_h - margin  # top edge of the grid

    c = canvas.Canvas(filename, pagesize=landscape(A4))

    # --- Draw cells ---
    for r in range(rows):
        for ccol in range(cols):
            pid = int(index_map[r, ccol])
            x = start_x + ccol * cell
            y = start_y - (r + 1) * cell
            if show_cell_color:
                rr, gg, bb = pid_to_rgb.get(pid, (220, 220, 220))
                c.setFillColorRGB(rr/255.0, gg/255.0, bb/255.0)
                c.rect(x, y, cell, cell, fill=1, stroke=0)
            # cell border
            c.setLineWidth(0.3)
            c.setStrokeColorRGB(0, 0, 0)
            c.rect(x, y, cell, cell, fill=0, stroke=1)

            if show_numbers_on_cells:
                # legend number
                lid = pid_to_legend.get(pid, 0)
                c.setFillColor(colors.black if (rr + gg + bb) > 382 else colors.white)  # contrast text
                c.setFont("Helvetica-Bold", max(6, int(cell * 0.28)))  # adaptive size
                text_x = x + cell * 0.32
                text_y = y + cell * 0.28
                c.drawString(text_x, text_y, str(lid))

    # --- Legend ---
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    legend_title_y = start_y - (rows * cell) - 12
    c.drawString(margin, legend_title_y, "Legend:")

    c.setFont("Helvetica", 9)
    y = legend_title_y - 10
    line_h = 12

    # legend columns: №, brand/code/name, RGB, count
    for _, row in legend_df.iterrows():
        lid = int(row["legend_id"])
        brand = row["brand"]
        code = row["code"]
        name = row["color_name"]
        rr, gg, bb = int(row["r"]), int(row["g"]), int(row["b"])
        count = int(row["count"])

        # swatch
        c.setFillColorRGB(rr/255.0, gg/255.0, bb/255.0)
        c.rect(margin, y - 8, 16, 8, fill=1, stroke=1)
        c.setFillColor(colors.black)

        c.drawString(
            margin + 22,
            y,
            f"{lid}. {brand} {code} — {name}  RGB({rr},{gg},{bb})   x{count}"
        )

        y -= line_h
        # move to a new page if space runs out
        if y < 30:
            c.showPage()
            page_w, page_h = landscape(A4)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, page_h - 40, "Legend (continued):")
            c.setFont("Helvetica", 9)
            y = page_h - 60

    c.save()
    return filename
