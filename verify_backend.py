from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings
import math
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
STATIC_REPORT_DIR = os.path.join(STATIC_FOLDER, "reports")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    """Health check endpoint."""
    return "Flask verification server is running."

@app.route('/verify', methods=['POST'])
def verify_watermarks():
    """
    Receives initial and extracted watermark images.
    Computes PSNR, SSIM, Correlation, determines status,
    and generates a PDF verification report.
    """
    initial_watermark_file = request.files.get('initial_watermark')
    extracted_watermark_file = request.files.get('extracted_watermark')

    if not initial_watermark_file or not extracted_watermark_file:
        return "Both watermark images are required", 400

    # Decode images using OpenCV
    initial_watermark = cv2.imdecode(np.frombuffer(initial_watermark_file.read(), np.uint8), cv2.IMREAD_COLOR)
    extracted_watermark = cv2.imdecode(np.frombuffer(extracted_watermark_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if initial_watermark is None or extracted_watermark is None:
        return "Error reading the images", 400

    try:
        # Resize extracted watermark to match initial watermark size
        extracted_watermark = cv2.resize(
            extracted_watermark,
            (initial_watermark.shape[1], initial_watermark.shape[0])
        )

        # --- Metric Calculations ---

        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((initial_watermark.astype("float") - extracted_watermark.astype("float")) ** 2)
        PIXEL_MAX = 255.0
        psnr_value = 999.99 if mse == 0 else 10 * math.log10((PIXEL_MAX ** 2) / mse)

        # SSIM (Structural Similarity Index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim_value = ssim(initial_watermark, extracted_watermark, channel_axis=-1)

        # Correlation Coefficient
        correlation = np.corrcoef(initial_watermark.flatten(), extracted_watermark.flatten())[0, 1]

        # Determine image authenticity
        status = "Authentic" if (psnr_value >= 30 and ssim_value >= 0.95 and correlation >= 0.98) else "Tampered"

        # --- PDF Report Generation ---

        os.makedirs(STATIC_REPORT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        report_filename = f"report_{timestamp}.pdf"
        report_path = os.path.join(STATIC_REPORT_DIR, report_filename)

        # Save temporary images
        initial_img_path = os.path.join(STATIC_REPORT_DIR, f"initial_{timestamp}.png")
        extracted_img_path = os.path.join(STATIC_REPORT_DIR, f"extracted_{timestamp}.png")
        cv2.imwrite(initial_img_path, initial_watermark)
        cv2.imwrite(extracted_img_path, extracted_watermark)

        # Create PDF canvas
        pdf = canvas.Canvas(report_path, pagesize=A4)
        width, height = A4

        # Title
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(100, height - 80, "MarkMesh – Verification Report")

        # Image section labels
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(100, height - 130, "Initial Watermark:")
        pdf.drawString(320, height - 130, "Extracted Watermark:")

        # Display images on PDF
        pdf.drawImage(initial_img_path, 100, height - 280, width=150, height=150, preserveAspectRatio=True)
        pdf.drawImage(extracted_img_path, 320, height - 280, width=150, height=150, preserveAspectRatio=True)

        # Metric table data
        table_data = [
            ["Metric", "Value"],
            ["PSNR", f"{psnr_value:.2f}"],
            ["SSIM", f"{ssim_value:.4f}"],
            ["Correlation Coefficient", f"{correlation:.4f}"],
            ["Status", status]
        ]

        # Table style
        table = Table(table_data, colWidths=[200, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#212529')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.75, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica')
        ]))

        table.wrapOn(pdf, width, height)
        table.drawOn(pdf, 100, height - 430)

        # Footer with timestamp
        pdf.setFont("Helvetica", 10)
        pdf.drawString(100, 100, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save the PDF report
        pdf.save()

        # Delete temporary image files
        os.remove(initial_img_path)
        os.remove(extracted_img_path)

        # Return metrics and download link
        download_link = f"http://localhost:5002/download/{report_filename}"
        result = {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'correlation': correlation,
            'status': status,
            'downloadLink': download_link
        }

        return jsonify(result)

    except Exception as e:
        print("Error during verification:", str(e))
        return "Internal server error during verification.", 500

@app.route('/download/<filename>', methods=['GET'])
def download_report(filename):
    """
    Serves the generated PDF report file for download.
    """
    return send_from_directory(STATIC_REPORT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
