import streamlit as st
import json
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from lib import GLN_IO as gl
import GLN_optimization as gopt
import GLN_simulation_1 as gsimu
import pickle

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from io import BytesIO
from PIL import Image as PILImage
import os


from datetime import datetime
import base64
from utils.scdt_Report import create_filled_report, save_to_json
#from reportlab.lib.utils import ImageReader
from dataclasses import dataclass, asdict
from aas.aas_client import AASClient
from dataclasses import dataclass, asdict
from typing import List, Optional
import yaml
from pathlib import Path

#-------------------------------------for the report generation and submission to the AAS-----------------
def load_config(path: str = "config.yaml") -> dict:
    """Carga el archivo config.yaml y devuelve un diccionario."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

@dataclass
class AASConfig:
    base_url:str
    client_id: str
    client_secret: str
    aas_id_short: str = "SCDTReports_AAS"
    submodel_id_short: str = "Reports"

def _draw_multiline_text(c: canvas.Canvas, x: int, y: int, text: str, leading: int = 14, max_width: int = 520):
    """
    render text  (\n).
    """
    text_obj = c.beginText(x, y)
    text_obj.setLeading(leading)
    for line in text.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

def _scale_to_width(img_path, max_width, max_height):
    # calculate the size to fit the content in the space
    with PILImage.open(img_path) as im:
        w, h = im.size
    ratio = min(max_width / w, max_height / h)
    return (w * ratio, h * ratio)

def build_pdf(title, notes, image_paths, header_data):
    """
    title: str
    notes: str (
    image_paths: list[str]
    header_data: dict (title/type/date/parameters)
    """
    buf = BytesIO()

    # Márgenes y doc
    PAGE_SIZE = A4
    margin = 2*cm
    doc = SimpleDocTemplate(
        buf,
        pagesize=PAGE_SIZE,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.2*cm, bottomMargin=1.8*cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleH1",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="Meta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        leading=12,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12, spaceAfter=6
    ))

    story = []

    # --- Portada / encabezado visible en página 1 ---
    story.append(Paragraph(header_data.get("title", title), styles["TitleH1"]))
    meta_line = f"{header_data.get('type','')} — {header_data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))}"
    story.append(Paragraph(meta_line, styles["Meta"]))

    # Parámetros en tabla (si hay)
    params = header_data.get("parameters", {}) or {}
    if params:
        data = [["Parementer", "Value"]] + [[str(k), str(v)] for k, v in params.items()]
        tbl = Table(data, colWidths=[6*cm, None])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F0F0")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,0), "LEFT"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FBFBFB")]),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

    # --- Gráficos ---
    story.append(Paragraph("Graphs from simulation", styles["SectionTitle"]))
    usable_w = PAGE_SIZE[0] - doc.leftMargin - doc.rightMargin
    usable_h = PAGE_SIZE[1] - doc.topMargin - doc.bottomMargin - 1*cm  # margen de seguridad

    for p in image_paths or []:
        if p and os.path.exists(p):
            iw, ih = _scale_to_width(p, usable_w, usable_h)
            img = Image(p, width=iw, height=ih)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        else:
            story.append(Paragraph(f"[Aviso] Image not founded: {p}", styles["Meta"]))

    # Salto antes de notas
    story.append(PageBreak())

    # --- Notas / comentarios (flujo multi-página) ---
    story.append(Paragraph("Comments", styles["SectionTitle"]))
    # Puedes dividir en párrafos por líneas en blanco para mejor justificado
    for para in (notes or "").split("\n\n"):
        story.append(Paragraph(para.strip(), styles["Body"]))
        story.append(Spacer(1, 0.2*cm))

    # --- Header/Footer en todas las páginas ---
    def _header_footer(canvas, doc_):
        canvas.saveState()
        # Header
        header_txt = header_data.get("title", title)
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc_.leftMargin, PAGE_SIZE[1] - 1.2*cm, header_txt[:90])
        # Footer
        footer_txt = f"Generated: {header_data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))}"
        canvas.drawRightString(PAGE_SIZE[0] - doc_.rightMargin, 1.0*cm, f"{footer_txt}  ·  Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes 


# def build_pdf(
#     title: str,
#     subtitle_lines: List[str],
#     image_paths: List[str],
#     notes: Optional[str] = None,
#     input_values: Optional[str] = None,         
#     small_kpi_images: Optional[List[str]] = None, 
#     extra_section: Optional[str] = None           
# ) -> bytes:
#     """
#     Crea un PDF en memoria con título, subtítulo (líneas), imágenes (si existen) y notas.
#     Devuelve bytes del PDF.
#     """
#     buf = io.BytesIO()
#     c = canvas.Canvas(buf, pagesize=letter)
#     width, height = letter

#     # Portada
#     c.setFont("Helvetica-Bold", 20)
#     c.drawString(50, height - 60, title)

#     c.setFont("Helvetica", 12)
#     y = height - 85
#     for line in subtitle_lines:
#         c.drawString(50, y, line)
#         y -= 16

#     # Imágenes (máximo dos por página)
#     x_img, w_img, h_img, y_start, y_gap = 50, 500, 230, y - 20, 20
#     current_y = y_start
#     items_in_page = 0

#     for p in image_paths:
#         if os.path.exists(p):
#             if items_in_page == 2:
#                 c.showPage()
#                 c.setFont("Helvetica-Bold", 16)
#                 c.drawString(50, height - 50, "Charts")
#                 current_y = height - 90
#                 items_in_page = 0
#             c.drawImage(p, x_img, current_y - h_img, width=w_img, height=h_img, preserveAspectRatio=True, anchor='n')
#             current_y -= (h_img + y_gap)
#             items_in_page += 1

#     # Notas
#     if notes:
#         c.showPage()
#         c.setFont("Helvetica-Bold", 16)
#         c.drawString(50, height - 50, "Notes / Comments")
#         c.setFont("Helvetica", 12)
#         _draw_multiline_text(c, 50, height - 80, notes, leading=14)

#     c.save()
#     buf.seek(0)
#     return buf.read()

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_small_kpi_plot(fig, path):
    fig.set_size_inches(1.18, 1.57)  # 3x4 cm
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def create_and_optionally_store_report(
    strType: str,
    strTitle: str,
    strAnalysis: str,
    pdf_bytes: bytes,
    save_json_locally: bool,
    json_path: str,
    upload_to_aas: bool,
    aas_cfg: Optional[AASConfig] = None,
):
    """
    Build the pdf as blob Base64, 
    save JSON ig requested and upload to the AAS if requested
    """
    pdf_blob = base64.b64encode(pdf_bytes).decode("utf-8")

    scdt_Report = create_filled_report(
        strType=strType,
        strTitle=strTitle,
        strAnalysis=strAnalysis,
        pdfBlob=pdf_blob
    )

    if save_json_locally:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        save_to_json(scdt_Report, json_path)

    if upload_to_aas:
        if aas_cfg is None:
            raise ValueError("AASConfig mandatory si upload_to_aas=True.")

        reportID = scdt_Report.idShort
        report_json = json.dumps(asdict(scdt_Report))

        client = AASClient(
            base_url=aas_cfg.base_url,
            client_id=aas_cfg.client_id,
            client_secret=aas_cfg.client_secret
        )
        token_info = client.authenticate()
        # Puedes loguear condicionalmente:
        # st.write("Token obtenido:", token_info)

        result = client.update_submodel_element_value(
            aas_id_short=aas_cfg.aas_id_short,
            sm_id_short=aas_cfg.submodel_id_short,
            se_id_short_path=reportID,
            value=report_json
        )
        # st.write("Resultado AAS:", result)

    return scdt_Report

#-------------------------------manage the error in forecasting---------------------------------


def interpret_mape(mape_in):
    
    
    if isinstance(mape_in, set) and len(mape_in) == 1:
        mape_in = next(iter(mape_in))  # Extrae el único valor
        print(type(mape_in))
        mape=mape_in*100
    else:
        exit(0)  
    
    if mape < 10:
        st.write(f"MAPE: {mape:.2f}% presents an excellent accuracy as the error is <10%")
    elif 10 <= mape < 20:
        st.write(f"MAPE: {mape:.2f}% presents a good accuracy as the error is between 10% and 20%")
    elif 20 <= mape < 50:
        st.write(f"MAPE: {mape:.2f}% presents an acceptable accuracy as the error is between 20% and 50%")
    else:
        st.write(f"MAPE: {mape:.2f}% presents a poor accuracy as the error is >50%")



def check_serializability(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

# ---main page with tabs---
df_products_material = gl.read_products_BOM()
batch_size_init = int(df_products_material["BatchSize"].values[0])
df_materials_suppliers = gl.read_material_suppliers()
df_supplier_material = df_materials_suppliers[(df_materials_suppliers["SupplierId"]=="1") & (df_materials_suppliers["MaterialId"]=="1")]
reorder_quantity_init = int(df_supplier_material["ReorderQuantity"].iloc[0])   
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(layout="wide", page_title="R3Group Supply Chain Proof of Concept")
config = load_config()

#-----------config AAS-----------
  #config server AAS
aas_url = config["base_url"]
aas_client = config["client_id"]
aas_secret = client_secret = config["client_secret"]
aas_id_short = "SCDTReports_AAS"
aas_sm_id_short = "Reports"

st.markdown(
    """
    <div style="display: flex; align-items: right; justify-content: space-between; padding: 10px 0;">
        <img src="http://r3group-project.com/wp-content/uploads/2023/04/REGROUP-Logo-project.png" alt="Logo" style="height: 60px;">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Menú superior tipo tabs ---
tabs = st.tabs(["🏠 Home","Prediction", "📊 Simulation", "⚙️ Optimization"])

# --- Home Page ---
with tabs[0]:
    st.title("Supply Chain Simulation App")
    st.markdown("""
    Welcome to the **supply chain digital twin tool**.

    This application allows you to:
    - Enter parameters
    - Launch simulations
    - Compare results between scenarios
    """)
    st.divider()

     
 # ====================== PREDICTION TAB ======================
with tabs[1]:
    st.title("Price Prediction")

    # 🔹 Input Section
    st.subheader("Input parameters")
    input_col, help_col = st.columns([2, 1])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with input_col:
        raw_material_file = st.file_uploader(
            "Upload raw material prices (CSV)",
            type="csv",
            help="CSV file with crude oil historical prices. Columns: Date, Price."
        )
        main_material_file = st.file_uploader(
            "Upload supplier material prices (CSV)",
            type="csv",
            help="CSV file with supplier-specific supplier material prices. Columns: Date, Price."
        )

        if st.button("Start prediction"):
            # Persistimos en session_state
            st.session_state['run_id'] = __import__('time').strftime("%Y%m%d_%H%M%S")
            st.session_state['raw_path'] = None
            st.session_state['main_path'] = None

            # Guardar los CSV en BASE_DIR
            if raw_material_file:
                raw_path = os.path.join(BASE_DIR, raw_material_file.name)
                with open(raw_path, "wb") as f:
                    f.write(raw_material_file.getbuffer())
                st.session_state['raw_path'] = raw_path
                st.success(f"Raw material prices saved to {raw_path}")

            if main_material_file:
                main_path = os.path.join(BASE_DIR, main_material_file.name)
                with open(main_path, "wb") as f:
                    f.write(main_material_file.getbuffer())
                st.session_state['main_path'] = main_path
                st.success(f"Supplier material prices saved to {main_path}")

            # Ejecutar el generador SOLO si tenemos ambos CSV
            if st.session_state['raw_path'] and st.session_state['main_path']:
                # Registrar tiempos de modificación previos (si existen)
                raw_png = os.path.join(OUTPUT_DIR, "raw_material_prices.png")
                pred_png = os.path.join(OUTPUT_DIR, "predictions.png")
                prev_raw_mtime = os.path.getmtime(raw_png) if os.path.exists(raw_png) else 0
                prev_pred_mtime = os.path.getmtime(pred_png) if os.path.exists(pred_png) else 0

                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            os.path.join(BASE_DIR, "GRAL_forecasting_v1.1.py"),
                            st.session_state['raw_path'],
                            st.session_state['main_path']
                        ],
                        cwd=BASE_DIR,            # 🔒 asegura rutas relativas del script
                        capture_output=True,
                        text=True,
                        check=True               # 🔒 lanza excepción si returncode != 0
                    )
                    st.info("Prediction process launched. See results below.")
                    #st.session_state['stdout'] = result.stdout
                    #st.session_state['stderr'] = result.stderr

                    # Validar que los PNG se han actualizado
                    ok_raw = os.path.exists(raw_png) and os.path.getmtime(raw_png) > prev_raw_mtime
                    ok_pred = os.path.exists(pred_png) and os.path.getmtime(pred_png) > prev_pred_mtime

                    if not (ok_raw and ok_pred):
                        st.warning("The generator ran but output images did not update. Check logs below.")

                except subprocess.CalledProcessError as e:
                    st.error("Prediction script failed.")
                    st.code(e.stdout or "", language="text")
                    st.code(e.stderr or "", language="text")

    with help_col:
        with st.expander("ℹ️ What files should I upload?"):
            st.markdown("""
- **Raw material prices** → CSV with crude oil time series.
- **Supplier material prices** → CSV with supplier-specific material historical prices.

The model uses **XGBoost** for forecasting raw material prices (Stage 1),
then predicts **supplier material prices** from the last 12 forecasted values (Stage 2).
            """)

    st.divider()
    # 🔹 Results Section
    st.subheader("Forecast Results")
    result_tabs = st.tabs(["📈 Raw material forecast", "📉 Supplier material price forecast"])

    def show_image_bytes(path: str, caption: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.image(f.read(), caption=caption)
        else:
            st.info(f"Not found: {path}")

    with result_tabs[0]:
        raw_png = os.path.join(OUTPUT_DIR, "raw_material_prices.png")
        show_image_bytes(raw_png, "Historical and forecasted raw material prices")

        # scores.pickle robusto
        pkl_path = os.path.join(OUTPUT_DIR, 'scores.pickle')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as handle:
                    resultsDict = pickle.load(handle)
                # ⚠️ usar una única clave consistente
                mape_key = 'XGBoost' if 'XGBoost' in resultsDict else ('xgb' if 'xgb' in resultsDict else None)
                if mape_key:
                    mape_error = {resultsDict[mape_key]['mape']}
                    interpret_mape(mape_error)
                else:
                    st.warning("MAPE key not found in scores.pickle")
            except Exception as ex:
                st.warning(f"Could not read scores.pickle: {ex}")

        with st.expander("ℹ️ Explanation"):
            st.markdown("""
This graph shows the last 12 **raw material price forecast** generated using XGBoost.
They will be used to project supplier material prices.
            """)

    with result_tabs[1]:
        pred_png = os.path.join(OUTPUT_DIR, "predictions.png")
        show_image_bytes(pred_png, "Predicted supplier material prices vs raw material forecast")

        pkl_path = os.path.join(OUTPUT_DIR, 'scores.pickle')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as handle:
                    resultsDict = pickle.load(handle)
                mape_key = 'XGBoost' if 'XGBoost' in resultsDict else ('xgb' if 'xgb' in resultsDict else None)
                if mape_key:
                    mape_error = {resultsDict[mape_key]['mape']}
                    interpret_mape(mape_error)
                else:
                    st.warning("MAPE key not found in scores.pickle")
            except Exception as ex:
                st.warning(f"Could not read scores.pickle: {ex}")

        with st.expander("ℹ️ Explanation"):
            st.markdown("""
This graph compares **supplier-specific material prices** against the **predicted supplier material prices**.
The forecasted values are derived via simple linear regression from the last 12 XGBoost raw material forecasts.
            """)

    # show logs
    if 'stdout' in st.session_state or 'stderr' in st.session_state:
        st.divider()
        st.subheader("Generator logs")
        st.code(st.session_state.get('stdout', ''), language="text")
        st.code(st.session_state.get('stderr', ''), language="text")
        
    # enable if we have results
    if 'scores.pickle' in os.listdir(OUTPUT_DIR):

        # Obtener MAPE
        try:
            with open(os.path.join(OUTPUT_DIR, 'scores.pickle'), 'rb') as handle:
                resultsDict = pickle.load(handle)
            mape_key = 'XGBoost' if 'XGBoost' in resultsDict else ('xgb' if 'xgb' in resultsDict else None)
            if mape_key:
                mape_value = resultsDict[mape_key]['mape'] * 100  # porcentaje
            else:
                mape_value = None
        except Exception:
            mape_value = None

        
    # ====================== FORECASTING REPORT ======================
    st.subheader("📄 Export Forecast Report")

    # Inputs del usuario (requerimiento #3)
    rpt_type_fore = "Forecasting"
    rpt_title_fore = st.text_input("Report title (strTitle)", value="Forecast Report")
    rpt_comments_fore = st.text_area("Comments", value="Generated report including uploaded files, charts, and notes.")

    # Opciones
    #save_json_fore = st.checkbox("Save JSON locally", value=False, key="save_json_fore")
    #upload_aas_fore = st.checkbox("Upload to AAS Server", value=True, key="upload_json_fore")
  
       
    if st.button("📄 Generate Forecast PDF"):
        creation_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # --- MAPE  ---
        mape_str = "N/A"
        mape_model = "N/A"
        try:
            scores_path = os.path.join(OUTPUT_DIR, 'scores.pickle')
            if os.path.exists(scores_path):
                with open(scores_path, 'rb') as handle:
                    resultsDict = pickle.load(handle)
                mape_key = 'XGBoost' if 'XGBoost' in resultsDict else ('xgb' if 'xgb' in resultsDict else None)
                if mape_key and 'mape' in resultsDict[mape_key]:
                    mape_value = float(resultsDict[mape_key]['mape']) * 100
                    mape_str = f"{mape_value:.2f}%"
                    mape_model = mape_key
        except Exception:           
            pass

        # --- Files info (si no hay, N/A) ---
        raw_file = os.path.basename(st.session_state.get('raw_path', 'N/A'))
        supp_file = os.path.basename(st.session_state.get('main_path', 'N/A'))

       # --- Header info (UNIFIED) ---
        header_data = {
            "title": rpt_title_fore,
            "type": rpt_type_fore,  # "Forecasting"
            "date": creation_date,
            "parameters": {
                "Model": mape_model,
                "MAPE": mape_str,
                "Raw material CSV": raw_file,
                "Supplier material CSV": supp_file
            }
        }


        # --- Images from pipeline---
        forecast_imgs = [
            os.path.join(OUTPUT_DIR, "raw_material_prices.png"),
            os.path.join(OUTPUT_DIR, "predictions.png"),
        ]

        # 1) build PDF
        pdf_bytes = build_pdf(
            title=rpt_title_fore,
            header_data=header_data,
            image_paths=forecast_imgs,
            notes=rpt_comments_fore
        )

        # 2) Create object + JSON + (opcional) subir AAS
        json_name = "output/report_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       
        aas_cfg = AASConfig(
            base_url=aas_url,
            client_id=aas_client,
            client_secret=aas_secret,
            aas_id_short=aas_id_short,
            submodel_id_short=aas_sm_id_short
        )

        scdt_report = create_and_optionally_store_report(
            strType=rpt_type_fore,
            strTitle=rpt_title_fore,
            strAnalysis=rpt_comments_fore,
            pdf_bytes=pdf_bytes,
            save_json_locally=True,
            json_path=json_name,
            upload_to_aas=True,
            aas_cfg=aas_cfg
        )

        # 3) Download button
        st.download_button(
            label="Download Forecast PDF",
            data=pdf_bytes,
            file_name="forecast_report.pdf",
            mime="application/pdf"
        )
        st.success("Forecast report generated successfully.")


# --- Simulation Page ---
with tabs[2]:
    st.header("Scenario Simulation")

    # 🔹 Generic function for KPI charts
    def plot_kpi_distribution(df_kpis, title):
        """Show KPI chart:
           - Pie chart if all values >= 0
           - Bar chart if there are negative values
        """
        fig, ax = plt.subplots()

        if (df_kpis["Value"] < 0).any():
            # At least one negative → use bar chart
            colors = ["green" if v >= 0 else "red" for v in df_kpis["Value"]]
            ax.bar(df_kpis["KPI"], df_kpis["Value"], color=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Value")
            ax.set_title(f"{title} (bar chart - negatives detected)")
            plt.xticks(rotation=45, ha="right")
        else:
            # All values >= 0 → use pie chart
            ax.pie(df_kpis["Value"], labels=df_kpis["KPI"], autopct='%1.1f%%')
            ax.set_title(f"{title} (distribution)")

        return fig

    # 🔹 Input layout
    input_col, help_col = st.columns([2, 1])

    with input_col:
        st.subheader("Input parameters: Stress Test")

        demand_var = st.number_input(
            "Demand variation (-1.0 to 1.0)", 
            value=0.0, step=0.1,
            help="Percentage change in demand (e.g., 0.2 = +20%)."
        )

        start_week = st.number_input(
            "Week starting the supply shortage", 
            value=-1, step=1,
            help="Week when supply interruption begins. -1 = no interruption."
        )

        end_week = st.number_input(
            "Week ending the supply shortage", 
            value=-1, step=1,
            help="Week when supply interruption ends. -1 = no interruption."
        )

        if st.button("🚀 Start simulation"):
            params_as_is = {
                "SimulationName": "AS_IS",
                "DemandVariation": 0,
                "InitWeekShortage": -1,
                "EndWeekShortage": -1,
                "BatchSize": batch_size_init,
                "ReorderQuantity": reorder_quantity_init
            }

            input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        
            with open(input_file, "w") as f:
                json.dump(params_as_is, f)
                # Leer y validar
            with open(input_file, "r") as f:
                data = json.load(f)
                print("Contenido AS IS:", data)
            df_as_is = gsimu.main()             
            name='output/simulation'
            simulation_data=gl.write_output_json(df_as_is,name + '.json')
            st.session_state.simulation_as_is_done = True

            params_to_be = {
                "SimulationName": "STRESS_TEST",
                "DemandVariation": demand_var,
                "InitWeekShortage": start_week,
                "EndWeekShortage": end_week,
                "BatchSize": batch_size_init,
                "ReorderQuantity": reorder_quantity_init
            }

            with open(input_file, "w") as f:
                json.dump(params_to_be, f)
                # Leer y validar
            with open(input_file, "r") as f:
                data = json.load(f)
                print("Contenido AS IS:", data)
            df_to_be = gsimu.main()
            st.session_state.simulation_to_be_done = True
            simulation_data=gl.write_output_json(df_to_be,name + '.json')
            gl.compare_simulations(name + '.json')

    with help_col:
        with st.expander("ℹ️ About the input parameters"):
            st.markdown("""
            - **Demand variation**: percentage change in total product demand.  
              Example: `0.2 = +20%`, `-0.3 = -30%`.  
            - **Supply shortage (start & end week)**: period when **no material is delivered by supplier**.  
              If both values are `-1`, **no shortage is applied**.
            """)

    st.divider()

    # 🔹 Comparative graphs
    if st.session_state.get("simulation_as_is_done") and st.session_state.get("simulation_to_be_done"):
        st.success("Simulation completed ✅")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.subheader("AS-IS Scenario")
            ind_file = os.path.join(BASE_DIR, "output", "AS_IS_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "AS_IS_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Inventory & demand evolution (AS-IS)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Inventory on-had  vs in-transit (AS-IS)")

        with graph_col2:
            st.subheader("Stress Test Scenario")
            ind_file = os.path.join(BASE_DIR, "output", "STRESS_TEST_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "STRESS_TEST_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Inventory & demand evolution (Stress Test)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Inventory on-hand vs in-transit (Stress Test)")

        st.divider()

        # 🔹 KPIs
        st.subheader("Scenario KPI Comparison")

        output_file = os.path.join(BASE_DIR, "output", "comparison.json")
        if os.path.exists(output_file):
            with open(output_file) as f:
                df = json.load(f)

            last_scenario = list(df.keys())[-1]
            items = list(df[last_scenario].items())

            financial_kpis = pd.DataFrame(items[3:6], columns=["KPI", "Value"])
            operational_kpis = pd.DataFrame(items[-7:], columns=["KPI", "Value"])

            kpi_tabs = st.tabs(["💰 Financial KPIs", "⚙️ Operational KPIs"])

            # --- Financial KPIs
            with kpi_tabs[0]:
                
                st.markdown("**AS-IS vs Stress Test (Bar chart)**")
                fig1, ax = plt.subplots()
                ax.bar(financial_kpis["KPI"], financial_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Financial KPIs comparison")               
                st.pyplot(fig1)

                fig1_path = os.path.join(OUTPUT_DIR, "financial_kpis.png")
                save_fig(fig1, fig1_path)

            # --- Operational KPIs
            with kpi_tabs[1]:
               
                st.markdown("**AS-IS vs Stress Test (Bar chart)**")
                fig2, ax = plt.subplots()
                ax.bar(operational_kpis["KPI"], operational_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Operational KPIs comparison")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig2)

                fig2_path = os.path.join(OUTPUT_DIR, "operational_kpis.png")
                save_fig(fig2, fig2_path)

            with st.expander("ℹ️ KPI explanation"):
                st.markdown("""
                - **Financial KPIs**: costs, margins, and economic efficiency.  
                - **Operational KPIs**: inventory levels, lead time, demand fulfillment, etc.  

                👉 Pie chart is shown only if all values are positive.  
                👉 If there are negatives, a bar chart is used (green = positive, red = negative).  
                👉 The bar chart directly compares **AS-IS vs Stress Test**.
                """)

     # ====================== Simulation REPORT ======================
    st.subheader("📄 Export Simulation Report")

    # Inputs shown in the report
    sim_demand_variation = demand_var,
    sim_start_week = start_week,
    sim_end_week = end_week,
    #sim_batch_size = batch_size,
    #sim_reorder_qty = reorder_quantity

    # Inputs from user
    rpt_type_simu = "Simulation"
    rpt_title_simu = st.text_input("Report title (strTitle)", value="Simulation Report")
    rpt_comments_simu = st.text_area("Simulation notes", value="Generated report including uploaded files, charts, and notes.")

    # User Options
    #save_json_simu = st.checkbox("Save JSON locally", value=False, key="save_json_simu")
    #upload_aas_simu = st.checkbox("Upload to AAS Server",value=True, key="upload_aas_simu")
  
    if st.button("📄 Generate Simulation PDF"):
        try:
            # ICON
            icon_path = "assets/icon.png"

            # Images for the PDF
            simulation_imgs = [
                os.path.join(OUTPUT_DIR, "AS_IS_individual.png"),
                #os.path.join(OUTPUT_DIR, "AS_IS_combined.png"),
                os.path.join(OUTPUT_DIR, "STRESS_TEST_individual.png"),
                #os.path.join(OUTPUT_DIR, "STRESS_TEST_combined.png")
                os.path.join(OUTPUT_DIR, "operational_kpis.png"),
                os.path.join(OUTPUT_DIR, "financial_kpis.png")
               
            ]           

           # Header info
            header_data = {
                "title": rpt_title_simu,
                "type": rpt_type_simu,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "parameters": {
                    "Demand Variation (percentage +/-[0-1]": sim_demand_variation,
                    "Shortage Start Week": sim_start_week,
                    "Shortage End Week": sim_end_week
                    #"Batch Size": sim_batch_size,
                    #"Reorder Quantity": sim_reorder_qty
                }
            }     

                       
            # 1) build PDF
            pdf_bytes = build_pdf(
                title=rpt_title_simu,
                header_data=header_data,
                image_paths=simulation_imgs,
                notes=rpt_comments_simu
            )

            # 2) Create object + JSON + (opcional) subir AAS
            json_name = "output/report_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       
            aas_cfg = AASConfig(
                base_url=aas_url,
                client_id=aas_client,
                client_secret=aas_secret,
                aas_id_short=aas_id_short,
                submodel_id_short=aas_sm_id_short
            )
                   

            scdt_report = create_and_optionally_store_report(
                strType=rpt_type_simu,
                strTitle=rpt_title_simu,
                strAnalysis=rpt_comments_simu,
                pdf_bytes=pdf_bytes,
                save_json_locally=True,
                json_path=json_name,
                upload_to_aas=True,
                aas_cfg=aas_cfg
            )

            #  3) Download button
            st.download_button(
                label="Download Simulation PDF",
                data=pdf_bytes,
                file_name="simulation_report.pdf",
                mime="application/pdf"
            )

            st.success("PDF generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating the simulation PDF: {e}")

with tabs[3]:
  
    st.header("Scenario Simulation Optimization")    
     
    st.session_state.simulation_conventional_stress_test_done = False
    st.session_state.simulation_optimal_stress_test_done = False

    # 🔹 Input layout
    input_col, value_col  = st.columns([2, 2])

    # Layout 1: Inputs
    with input_col:
        st.subheader("Input parameters: Optimal Stress Test")

        demand_var_opt = st.number_input(
            "Demand variation (-1.0 to 1.0)", 
            value=0.0, step=0.1,
            help="Percentage change in demand for optimization."
        )

        # Build parameter dictionaries
        params_conventional = {
            "SimulationName": "CONVENTIONAL_STRESS_TEST",
            "DemandVariation": demand_var_opt,
            "InitWeekShortage": -1,
            "EndWeekShortage": -1,
            "BatchSize": batch_size_init,
            "ReorderQuantity": reorder_quantity_init
        }

        EBQ = gopt.EBQ(demand_var_opt)
        EOQ = gopt.EOQ(demand_var_opt)

        params_optimal = {
            "SimulationName": "OPTIMAL_STRESS_TEST",
            "DemandVariation": demand_var_opt,
            "InitWeekShortage": -1,
            "EndWeekShortage": -1,
            "BatchSize": EBQ,
            "ReorderQuantity": EOQ
        }
    
    if st.button("🚀 Start optimization simulation"):
        # Conventional run
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")        
        with open(input_file, "w") as f:
            json.dump(params_conventional, f)
            # Leer y validar
        with open(input_file, "r") as f:
            data = json.load(f)
            print("Contenido conventional:", data)

        df_conventional=gsimu.main()
        name='output/simulation'
        simulation_data=gl.write_output_json(df_conventional,name + '.json')
        st.session_state.simulation_conventional_stress_test_done = True
         
        # Optimal run
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        with open(input_file, "w") as f:            
            json.dump(params_optimal, f)
        with open(input_file, "r") as f:
            data = json.load(f)
            print("\nContenido optimal:", data)
     
        df_optimal=gsimu.main()       
        st.session_state.simulation_optimal_stress_test_done = True        
        simulation_data=gl.write_output_json(df_optimal,name + '.json')
        gl.compare_simulations(name + '.json')

    if st.button("🧹 Reset optimization"):
        st.session_state.simulation_conventional_stress_test_done = False
        st.session_state.simulation_optimal_stress_test_done = False
        st.rerun()
    
    with value_col:
        st.subheader("Batch & Order Quantities")
        st.write(f"📦 **Initial Batch Size**: {batch_size_init} units")
        st.write(f"📦 **Optimized Batch Size (EBQ)**: {EBQ} kg supplier material")
        st.write(f"📦 **Initial Reorder Quantity**: {reorder_quantity_init} units")
        st.write(f"📦 **Optimized Reorder Quantity (EOQ)**: {EOQ} kg supplier material")

        with st.expander("ℹ️ Explanation of EBQ & EOQ"):
            st.markdown("""
            - **EBQ (Economic Batch Quantity)**: the optimal production lot size,  
              recalculated according to demand variation.  
            - **EOQ (Economic Order Quantity)**: the optimal minimum supplier order quantity.  

            These values are updated dynamically as the demand variation changes.
            """)


    st.divider()
   
  # 🔹 Show optimization results
    if st.session_state.get("simulation_optimal_stress_test_done"):
        st.success("Optimization simulation completed ✅")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.subheader("Conventional Stress Test")
            ind_file = os.path.join(BASE_DIR, "output", "CONVENTIONAL_STRESS_TEST_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "CONVENTIONAL_STRESS_TEST_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Conventional stress test (individual)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Conventional stress test (combined)")

        with graph_col2:
            st.subheader("Optimal Stress Test")
            ind_file = os.path.join(BASE_DIR, "output", "OPTIMAL_STRESS_TEST_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "OPTIMAL_STRESS_TEST_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Optimal stress test (individual)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Optimal stress test (combined)")

        st.divider()
        
        # 🔹 KPIs
        st.subheader("Scenario KPI Comparison")

        output_file = os.path.join(BASE_DIR, "output", "comparison.json")
        if os.path.exists(output_file):
            with open(output_file) as f:
                df = json.load(f)

            last_scenario = list(df.keys())[-1]
            items = list(df[last_scenario].items())

            financial_kpis = pd.DataFrame(items[3:6], columns=["KPI", "Value"])
            operational_kpis = pd.DataFrame(items[-7:], columns=["KPI", "Value"])

            kpi_tabs = st.tabs(["💰 Financial KPIs", "⚙️ Operational KPIs"])

            # --- Financial KPIs
            with kpi_tabs[0]:
                
                st.markdown("**Conventional Stress Test vs Optimal Stress Test (Bar chart)**")
                fig1, ax = plt.subplots(figsize=(6, 4))
                ax.bar(financial_kpis["KPI"], financial_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Financial KPIs comparison")
                st.pyplot(fig1)

                fig1_path = os.path.join(OUTPUT_DIR, "financial_kpis_opt.png")
                save_fig(fig1, fig1_path)

            # --- Operational KPIs
            with kpi_tabs[1]:
               
                st.markdown("**Conventional Stress Test vs Optimal Stress Test (Bar chart)**")
                fig2, ax = plt.subplots(figsize=(6, 4))
                ax.bar(operational_kpis["KPI"], operational_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Operational KPIs comparison")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig2)

                fig2_path = os.path.join(OUTPUT_DIR, "operational_kpis_opt.png")
                save_fig(fig2, fig2_path)

            with st.expander("ℹ️ KPI explanation"):
                st.markdown("""
                - **Financial KPIs**: costs, margins, and economic efficiency.  
                - **Operational KPIs**: inventory levels, lead time, demand fulfillment, etc.  

                👉 Pie chart is shown only if all values are positive.  
                👉 If there are negatives, a bar chart is used (green = positive, red = negative).  
                👉 The bar chart directly compares **Conventional Stress Test vs Optimal Stress Test (Bar chart)**.
                """)  
         
      # ====================== Optimization REPORT ======================
    st.subheader("📄 Export Optimization Report")

    # Inputs shown in the report
    opt_demand_variation = demand_var_opt
    opt_EOQ_before =reorder_quantity_init
    opt_EBQ_before =batch_size_init
    opt_EOQ_after =EOQ
    opt_EBQ_after =EBQ
     

    # Inputs from user
    rpt_type_opt = "Optimization - Simulation"
    rpt_title_opt = st.text_input("Report title (strTitle)", value="Optimization - Simulation Report")
    rpt_comments_opt = st.text_area("Optimization - Simulation notes", value="Generated report including uploaded files, charts, and notes.")

    # User Options
    #save_json_simu = st.checkbox("Save JSON locally", value=False, key="save_json_simu")
    #upload_aas_simu = st.checkbox("Upload to AAS Server",value=True, key="upload_aas_simu")
  
    if st.button("📄 Generate Optimization - Simulation PDF"):
        try:
            # ICON
            icon_path = "assets/icon.png"


            # Images for the PDF
            optimization_imgs = [
                os.path.join(OUTPUT_DIR, "CONVENTIONAL_STRESS_TEST_individual.png"),
                #os.path.join(OUTPUT_DIR, "AS_IS_combined.png"),
                os.path.join(OUTPUT_DIR, "OPTIMAL_STRESS_TEST_individual.png"),
                #os.path.join(OUTPUT_DIR, "STRESS_TEST_combined.png")
                os.path.join(OUTPUT_DIR, "operational_kpis_opt.png"),
                os.path.join(OUTPUT_DIR, "financial_kpis_opt.png")
               
            ]           
   

           # Header info
            header_data = {
                "title": rpt_title_opt,
                "type": rpt_type_opt,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "parameters": {
                    "Demand Variation (percentage +/-[0-1]": sim_demand_variation,
                    "EOQ before optimization (kg)": opt_EOQ_before,
                    "EOQ after optimization (kg)": EOQ,
                    "EBQ before optimization (pieces)": opt_EBQ_before,
                    "EBQ after optimization (pieces)": EBQ
                   
                }
            }     

                       
            # 1) build PDF
            pdf_bytes = build_pdf(
                title=rpt_title_opt,
                header_data=header_data,
                image_paths=optimization_imgs,
                notes=rpt_comments_opt
            )

            # 2) Create object + JSON + (opcional) subir AAS
            json_name = "output/report_optimization_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       
            aas_cfg = AASConfig(
                base_url=aas_url,
                client_id=aas_client,
                client_secret=aas_secret,
                aas_id_short=aas_id_short,
                submodel_id_short=aas_sm_id_short
            )
                   

            scdt_report = create_and_optionally_store_report(
                strType=rpt_type_opt,
                strTitle=rpt_title_opt,
                strAnalysis=rpt_comments_opt,
                pdf_bytes=pdf_bytes,
                save_json_locally=True,
                json_path=json_name,
                upload_to_aas=True,
                aas_cfg=aas_cfg
            )

            #  3) Download button
            st.download_button(
                label="Download Optimization Simulation PDF",
                data=pdf_bytes,
                file_name="optimization_simulation_report.pdf",
                mime="application/pdf"
            )

            st.success("PDF generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating the optimization simulation PDF: {e}")


    st.markdown(
    """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f0f2f6;
            color: #333;
            text-align: center;
            padding: 10px;
            font-size: 13px;
            border-top: 1px solid #ccc;
            z-index: 100;
        }
        .footer img {
            height: 40px;
            vertical-align: middle;
            margin-right: 10px;
        }
        </style>

        <div class="footer">
            <img src="https://r3group-project.com/wp-content/uploads/2023/05/European_Union_flag_with_blue_background_and_yellow_stars_generated.jpg" alt="EU Logo">
            The R3GROUP project has received funding from the European Union’s Horizon Europe programme under grant agreement No. 101091869.
        </div>
     """,
     unsafe_allow_html=True
    )



