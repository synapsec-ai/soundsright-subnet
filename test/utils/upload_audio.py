import glob 
import requests 
import os 

def upload_audio(noisy_dir, timeout=500,) -> bool:
    """
    Upload audio files to the API.

    Returns:
        bool: True if operation was successful, False otherwise
    """
    url = f"http://127.0.0.1:6500/upload-audio/"
    
    files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    
    print(f"files: {files}")

    try:
        with requests.Session() as session:
            file_payload = [
                ("files", (os.path.basename(file), open(file, "rb"), "audio/wav"))
                for file in files
            ]
            
            print(f"files_payload: {file_payload}")

            response = session.post(url, files=file_payload, timeout=timeout)

            for _, file in file_payload:
                file[1].close()  # Ensure all files are closed after the request

            response.raise_for_status()
            data = response.json()
            
            print(f"response data: {data}")

            sorted_files = sorted([file[1][0] for file in file_payload])
            print(f"sorted_files: {sorted_files}")
            sorted_response = sorted(data["uploaded_files"])
            print(f"sorted_response: {sorted_response}")
            outcome = sorted_files == sorted_response and data["status"]
            print(f"sorted_files == sorted_response: {outcome}")
            return sorted_files == sorted_response and data["status"]

    except requests.RequestException as e:
        print(f"Uploading audio to model failed because: {e}")
        return False
    except Exception as e:
        print(f"Uploading audio to model failed because: {e}")
        return False

def main():    
    noisy_dir = f"{os.path.expanduser('~')}/.SoundsRight/data/noise/16000"

    upload_audio(noisy_dir=noisy_dir)
    
if __name__ == "__main__":
    main()