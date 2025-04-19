from flask import Flask, request, send_file
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Create directory for storing embedded images if it doesn't exist
os.makedirs("embedded", exist_ok=True)

# Allowed image dimensions for validation
ALLOWED_COVER_DIMENSIONS = [(512, 512, 3)]
ALLOWED_WATERMARK_DIMENSIONS = [(32, 32, 3)]

# Embedding strength (alpha): controls visibility and extractability
ALPHA = 0.07

def is_valid_image(image_array, allowed_dimensions):
    """
    Checks if the image dimensions match allowed dimensions.
    """
    return image_array.shape in allowed_dimensions

def rgb_to_ycbcr_lossless(img_bgr):
    """
    Converts an RGB image to YCbCr color space using a transformation matrix.
    """
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]], dtype=np.float64)
    return np.dot(img_bgr, matrix.T)

def ycbcr_to_rgb_lossless(img_ycrcb):
    """
    Converts a YCbCr image back to RGB color space.
    """
    matrix = np.array([[1.0, 0.0, 1.402],
                       [1.0, -0.344136, -0.714136],
                       [1.0, 1.772, 0.0]], dtype=np.float64)
    return np.dot(img_ycrcb, matrix.T)

def embedd_matrix(COVER, WATERMARK, alpha=ALPHA):
    """
    Embeds a watermark into the luminance (Y) channel of the cover image using even-odd row splitting.
    """
    even = COVER[::2]
    odd = COVER[1::2]

    # Resize watermark if dimensions do not match
    if even.shape != WATERMARK.shape:
        WATERMARK = cv2.resize(WATERMARK, (even.shape[1], even.shape[0]))

    # Smooth watermark for better imperceptibility
    WATERMARK = cv2.GaussianBlur(WATERMARK, (5, 5), 0)
    WATERMARK = np.float64(WATERMARK) * alpha

    # Average + watermark for even rows, average - watermark for odd rows
    Aeven = (even + odd) / 2 + WATERMARK
    Aodd = (even + odd) / 2 - WATERMARK

    # Update COVER matrix with modified rows
    COVER[::2] = Aeven
    COVER[1::2] = Aodd
    return COVER

@app.route('/')
def home():
    """
    Default route to confirm the server is active.
    """
    return "Stealth Embed server is running!"

@app.route('/embed', methods=['POST'])
def embed_watermark():
    """
    Handles POST request to embed a watermark into an uploaded cover image.
    """
    try:
        # Load cover and watermark images from request
        image_file = request.files['image']
        watermark_file = request.files['watermark']

        # Convert uploaded images to RGB format
        image = Image.open(image_file).convert('RGB')
        watermark = Image.open(watermark_file).convert('RGB')

        # Convert to numpy arrays
        image_np = np.array(image)
        watermark_np = np.array(watermark)

        # Validate image dimensions
        if not is_valid_image(image_np, ALLOWED_COVER_DIMENSIONS):
            return "Invalid cover image dimensions.", 400
        if not is_valid_image(watermark_np, ALLOWED_WATERMARK_DIMENSIONS):
            return "Invalid watermark dimensions.", 400

        # Resize watermark to match image
        watermark_resized = cv2.resize(watermark_np, (image_np.shape[1], image_np.shape[0]))

        # Convert to YCbCr and embed watermark into Y channel
        ycbcr_image = rgb_to_ycbcr_lossless(image_np)
        ycbcr_image[:, :, 0] = embedd_matrix(ycbcr_image[:, :, 0].copy(), watermark_resized[:, :, 0])

        # Convert back to RGB and clamp pixel values
        embedded_image = ycbcr_to_rgb_lossless(ycbcr_image)
        embedded_image = np.clip(embedded_image, 0, 255).astype(np.uint8)

        # Save result to in-memory buffer and return it
        byte_io = BytesIO()
        Image.fromarray(embedded_image).save(byte_io, 'PNG')
        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png')

    except Exception as e:
        # Log and return any exceptions
        import traceback
        print(traceback.format_exc())
        return str(e), 400

if __name__ == '__main__':
    # Start the Flask development server
    app.run(port=5001, debug=True)
