import threading
import subprocess
import os

# --- Functions to launch each backend service ---

def run_embed():
    """Launch the embed server on port 5001."""
    print("Starting embed server on port 5001...")
    subprocess.run(["python", "embed_backend.py"], cwd=os.getcwd())

def run_extract():
    """Launch the extract server on port 5000."""
    print("Starting extract server on port 5000...")
    subprocess.run(["python", "extract_backend.py"], cwd=os.getcwd())

def run_verify():
    """Launch the verify server on port 5002."""
    print("Starting verify server on port 5002...")
    subprocess.run(["python", "verify_backend.py"], cwd=os.getcwd())

# --- Main Execution ---

if __name__ == "__main__":
    # Create separate threads for each backend server
    thread_embed = threading.Thread(target=run_embed)
    thread_extract = threading.Thread(target=run_extract)
    thread_verify = threading.Thread(target=run_verify)

    # Start all threads
    thread_embed.start()
    thread_extract.start()
    thread_verify.start()

    # Wait for all threads to complete (blocks main thread)
    thread_embed.join()
    thread_extract.join()
    thread_verify.join()
