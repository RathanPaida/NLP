import os
import sys
import time
import requests

def download_file_with_resume(url, destination_path, max_retries=20):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # First get total size
    r_head = requests.head(url, allow_redirects=True, timeout=15)
    total_size = int(r_head.headers.get('content-length', 0))
    print(f"Target URL: {url}")
    print(f"Total file size: {total_size / (1024*1024):.2f} MB")
    
    retries = 0
    while retries < max_retries:
        downloaded = os.path.getsize(destination_path) if os.path.exists(destination_path) else 0
        if total_size > 0 and downloaded >= total_size:
            print(f"File completely downloaded: {downloaded / (1024*1024):.2f} MB.")
            return True
            
        headers = {}
        file_mode = 'wb'
        if downloaded > 0:
            headers['Range'] = f"bytes={downloaded}-"
            file_mode = 'ab'
            print(f"Resuming download from {downloaded / (1024*1024):.2f} MB ({downloaded / total_size * 100:.1f}%)...")
            
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
                r.raise_for_status()
                # If server doesn't support Range and returned 200 instead of 206, rewrite from 0
                if downloaded > 0 and r.status_code == 200:
                    print("Server does not support Range requests, downloading from start.")
                    file_mode = 'wb'
                    downloaded = 0
                    
                chunk_size = 1024 * 1024 # 1 MB
                t_last_log = time.time()
                bytes_since_log = 0
                with open(destination_path, file_mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            bytes_since_log += len(chunk)
                            now = time.time()
                            if now - t_last_log >= 3.0:
                                speed = (bytes_since_log / (1024*1024)) / (now - t_last_log)
                                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                                print(f"Progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s", flush=True)
                                t_last_log = now
                                bytes_since_log = 0
                                
            if total_size > 0 and downloaded >= total_size:
                print(f"Successfully finished download: {downloaded / (1024*1024):.2f} MB.")
                return True
        except Exception as e:
            retries += 1
            print(f"Connection error: {e}. Retrying ({retries}/{max_retries}) in 3 seconds...", flush=True)
            time.sleep(3)
            
    print("Max retries exceeded.")
    return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python resilient_downloader.py <url> <destination_path>")
    else:
        download_file_with_resume(sys.argv[1], sys.argv[2])

