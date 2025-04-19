from flask import Flask, request, send_file, jsonify
import numpy as np
import os
from io import BytesIO
import traceback
import cv2
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Create a directory to store extracted images (optional use)
os.makedirs("extracted", exist_ok=True)

# --- Configuration ---
ALPHA = 0.07  # Must match the alpha used during embedding
EXPECTED_WATERMARK_SHAPE = (32, 32)  # Final expected watermark size for output
# ---

def recover_watermark_channel(channel_data, alpha=ALPHA):
    """
    Recovers watermark signal from a single image channel (e.g., Y, Cr, or Cb).
    Uses even-odd row difference based on the embedding strategy.
    """
    try:
        # Validate input
        if channel_data is None or channel_data.ndim != 2:
            print(f"Error: Invalid channel data: {type(channel_data)}")
            return None

        h, w = channel_data.shape

        # Ensure the height is even for row-pair processing
        if h % 2 != 0:
            print(f"Warning: Channel height {h} is odd. Cropping.")
            channel_data = channel_data[:-1, :]
            h -= 1
            if h == 0:
                return None

        # Convert to float for calculations
        channel_data_float = channel_data.astype(np.float32)

        # Extract even and odd rows
        even = channel_data_float[::2, :]
        odd = channel_data_float[1::2, :]

        # Ensure shapes still match
        if even.shape != odd.shape:
            print("Error: Mismatched row shapes.")
            return None

        # Recover watermark: difference of even and odd rows, scaled by alpha
        wm_float = (even - odd) / (2 * alpha)

        # Normalize watermark for visibility (stretch to 0–255)
        wm_normalized = cv2.normalize(wm_float, None, 0, 255, cv2.NORM_MINMAX)

        return wm_normalized.astype(np.uint8)

    except Exception as e:
        print(f"Error in watermark channel recovery: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def home():
    """
    Default endpoint to confirm the server is running.
    """
    return jsonify({"message": "Stealth Extract server is running!"})

@app.route('/extract', methods=['POST'])
def extract_watermark():
    """
    Extracts an embedded watermark from an uploaded image.
    Processes all 3 YCrCb channels and reconstructs watermark in color.
    """
    try:
        # --- File Input Validation ---
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # --- Load Image Using OpenCV ---
        image_stream = image_file.stream.read()
        np_image = np.frombuffer(image_stream, np.uint8)
        image_bgr = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({"error": "Invalid image format or unable to decode"}), 400

        # --- Convert BGR to YCrCb Color Space ---
        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(image_ycrcb)

        print(f"Input Channel shapes - Y: {y_channel.shape}, Cr: {cr_channel.shape}, Cb: {cb_channel.shape}")

        # --- Recover Watermark from All Three Channels ---
        watermark_y = recover_watermark_channel(y_channel)
        watermark_cr = recover_watermark_channel(cr_channel)
        watermark_cb = recover_watermark_channel(cb_channel)

        if watermark_y is None or watermark_cr is None or watermark_cb is None:
            errors = []
            if watermark_y is None: errors.append("Y channel recovery failed.")
            if watermark_cr is None: errors.append("Cr channel recovery failed.")
            if watermark_cb is None: errors.append("Cb channel recovery failed.")
            return jsonify({"error": "Failed to extract watermark.", "details": errors}), 500

        # --- Merge Recovered Channels to Form YCrCb Watermark ---
        recovered_ycrcb = cv2.merge([watermark_y, watermark_cr, watermark_cb])
        print(f"Merged watermark shape: {recovered_ycrcb.shape}")

        # --- Convert to BGR for viewing ---
        try:
            recovered_bgr = cv2.cvtColor(recovered_ycrcb, cv2.COLOR_YCrCb2BGR)
        except cv2.error as cvt_error:
            print("Color conversion failed, returning grayscale Y watermark.")
            is_success, buffer = cv2.imencode(".png", watermark_y)
            if not is_success:
                return jsonify({"error": "Failed to encode fallback grayscale watermark"}), 500
            byte_io = BytesIO(buffer)
            byte_io.seek(0)
            return send_file(byte_io, mimetype='image/png', download_name='extracted_watermark_gray.png')

        # --- Resize to Expected Output Dimensions ---
        final_watermark = cv2.resize(recovered_bgr, EXPECTED_WATERMARK_SHAPE[::-1], interpolation=cv2.INTER_LINEAR)
        print(f"Final watermark size: {final_watermark.shape}")

        # --- Encode and Send Output ---
        is_success, buffer = cv2.imencode(".png", final_watermark)
        if not is_success:
            return jsonify({"error": "Failed to encode extracted watermark"}), 500

        byte_io = BytesIO(buffer)
        byte_io.seek(0)

        return send_file(
            byte_io,
            mimetype='image/png',
            download_name='extracted_watermark_rgb.png',
            as_attachment=True
        )

    except cv2.error as cv_err:
        print("OpenCV Error:")
        print(traceback.format_exc())
        return jsonify({"error": f"Image processing error: {cv_err}"}), 500
    except Exception as e:
        print("Unexpected error in /extract:")
        print(traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app (default to port 5000 for extraction)
    app.run(host='0.0.0.0', port=5000, debug=True)
