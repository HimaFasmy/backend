import requests
import pytest
import os

# Define the Flask API base URL
BASE_URL = "http://127.0.0.1:5000"

# Sample test files
TEST_IMAGE = "test_image.jpg"
TEST_WATERMARK = "watermark.png"

@pytest.fixture
def setup_files():
    """ Ensure test files exist before running tests. """
    assert os.path.exists(TEST_IMAGE), f"Test image '{TEST_IMAGE}' not found!"
    assert os.path.exists(TEST_WATERMARK), f"Test watermark '{TEST_WATERMARK}' not found!"

def test_integration_watermarking(setup_files):
    """ Test full integration from embedding to extraction. """
    
    # 1️ **Embed Watermark**
    embed_url = f"{BASE_URL}/embed"
    embed_files = {
        "image": open(TEST_IMAGE, "rb"),
        "watermark": open(TEST_WATERMARK, "rb")
    }
    embed_response = requests.post(embed_url, files=embed_files)
    
    assert embed_response.status_code == 200, " Embedding failed!"
    assert embed_response.headers["Content-Type"] == "image/png"

    # Save watermarked image for next test
    watermarked_file = "watermarked_output.png"
    with open(watermarked_file, "wb") as f:
        f.write(embed_response.content)
    
    print("Embedding successful!")

    # 2️ **Extract Watermark**
    extract_url = f"{BASE_URL}/extract"
    extract_files = {"embedded_image": open(watermarked_file, "rb")}
    extract_response = requests.post(extract_url, files=extract_files)
    
    assert extract_response.status_code == 200, " Extraction failed!"
    assert extract_response.headers["Content-Type"] == "image/png"

    # Save extracted watermark image
    extracted_watermark = "extracted_watermark.png"
    with open(extracted_watermark, "wb") as f:
        f.write(extract_response.content)

    print(" Extraction successful!")

    # 3️ **Verify Output Files Exist**
    assert os.path.exists(watermarked_file), " Watermarked image not saved!"
    assert os.path.exists(extracted_watermark), "Extracted watermark image not saved!"

    print("\n Full integration test passed successfully!")

