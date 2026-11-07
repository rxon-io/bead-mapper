import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io

from src.inference import load_model, autoencode_image, grid_average, apply_palette_grid
from src.utils import color_count_report

from src.utils import generate_pattern_pdf

# --- Page settings ---
st.set_page_config(page_title="AI Bead Mapper", layout="wide")
st.title("🎨 AI Bead Mapper — bead pattern generator")


# --- Sidebar ---
st.sidebar.header("⚙️ Settings")

# Model
model_path = st.sidebar.text_input("Model path", value="models/autoencoder_model_best.pth")
use_autoencoder = st.sidebar.checkbox("Pass through autoencoder", value=True)

# Grid
grid = st.sidebar.slider("Cell size (px)", min_value=4, max_value=32, value=8, step=2)

# Palette
try:
    palette = pd.read_csv("data/palettes/palette.csv")
except Exception as e:
    st.error(f"Failed to load palette: {e}")
    st.stop()

# Brands in the palette
brands = sorted(palette["brand"].dropna().unique().tolist())
brand_filter = st.sidebar.multiselect(
    "Select brands for color matching:",
    options=brands,
    default=brands
)

# Filter by brand
palette = palette[palette["brand"].isin(brand_filter)].reset_index(drop=True)
if palette.empty:
    st.error("❌ No available colors in the selected palettes. Select at least one brand in the filter.")
    st.stop()


# --- Model (lazy loading) ---
_model = None
_device = None
if use_autoencoder:
    try:
        _model, _device = load_model(model_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        use_autoencoder = False


# --- Image upload ---
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    # Display the original image
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)  # ✅ Fixed

    # 1) Autoencoder (optional)
    proc = img
    if use_autoencoder and _model is not None:
        proc = autoencode_image(img, _model, _device)
    with col2:
        st.subheader("After autoencoder")
        st.image(proc, use_container_width=True)  # ✅ Fixed

    # 2) Grid averaging
    grid_img = grid_average(proc, grid, grid)
    with col3:
        st.subheader("Grid average")
        st.image(grid_img, use_container_width=True)  # ✅ Fixed

        st.markdown("---")
    st.subheader("🎯 Match colors from palette")

    # 3) Palette color matching
    bead_img, used_indices = apply_palette_grid(grid_img, grid, grid, palette)
    st.image(bead_img, caption="Bead Map", use_container_width=True)  # ✅ Fixed

    # --- Build 2D index map (rows x cols) ---
    W, H = grid_img.size  # PIL: (width, height)
    rows = H // grid
    cols = W // grid
    index_map = used_indices.reshape(rows, cols)  # now 2D

    # 4) Report + legend
    from src.utils import color_count_report, build_legend, generate_pattern_pdf

    report = color_count_report(index_map, palette)  # counts using the 2D map
    legend_df = build_legend(index_map, palette)     # assigns legend_id

    st.subheader("📋 List of used colors")
    st.dataframe(report, use_container_width=True)

    # --- Download buttons: PNG/CSV/PDF ---
    import io
    buf_img = io.BytesIO()
    bead_img.save(buf_img, format="PNG")
    st.download_button(
        "💾 Download bead image (PNG)",
        data=buf_img.getvalue(),
        file_name="bead_map.png",
        mime="image/png",
    )

    buf_csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📋 Download color list (CSV)",
        data=buf_csv,
        file_name="colors.csv",
        mime="text/csv",
    )

    if st.button("📄 Generate PDF with numbers on cells"):
        pdf_path = "bead_pattern.pdf"
        generate_pattern_pdf(
            index_map=index_map,
            legend_df=legend_df,
            filename=pdf_path,
            cell_mm=6.0,                  # 5–7 mm can be used depending on desired scale
            show_cell_color=True,
            show_numbers_on_cells=True
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF (with numbers and legend)",
                data=f.read(),
                file_name="bead_pattern.pdf",
                mime="application/pdf"
            )

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎨 AI Bead Mapper | Made with ❤️ for bead art enthusiasts <br> by Xeniya Zabaranilova and Erik Starodubtsev</p>
    </div>
    """,
    unsafe_allow_html=True
)