import os 
import requests
import zipfile 
import shutil
from soundsright.base.utils import subnet_logger 

def download_arni(arni_path: str, log_level: str = "INFO", partial: bool = False) -> None:
    """Downloads ARNI RIR dataset. [1] 
    
    [1] Prawda, Karolina, Schlecht, Sebastian J., & Välimäki, Vesa.
    Dataset of impulse responses from variable acoustics room Arni at Aalto Acoustic Labs [Data set]. 
    Zenodo, 2022. 
    https://doi.org/10.5281/zenodo.6985104
    
    Args:
        :param arni_path: (str): Output path to save ARNI files.
        :param log_level: (str, optional): Log level for operations. Defaults to "INFO".
        :param partial: (bool, optional): Set to True if you only want to partially download the dataset for testing purposes. Defaults to False.
        
    Raises:
        Exception: Raised if download fails in any way.
    """
    # URLs and filenames
    files = [
        "https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_0-5.zip?download=1",
        "https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_6-15.zip?download=1",
        "https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_16-25.zip?download=1",
        "https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_26-35.zip?download=1",
        "https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_36-45.zip?download=1",
        "https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_46-55.zip?download=1",
    ]
    
    if partial:
        files = files[0:2]

    if not os.path.exists(arni_path):
        os.makedirs(arni_path)

    # Download each file
    for url in files:
        try:
            # Get the file name from the URL
            zip_filename = url.split('/')[-1].split('?')[0]
            zip_filepath = os.path.join(arni_path, zip_filename)

            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save the file to disk
            with open(zip_filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            # Extract the zip file
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                extract_dir = os.path.join(arni_path, zip_filename.replace('.zip', ''))
                zip_ref.extractall(extract_dir)

            # Move all .wav files to arni_path (including subdirectories)
            for root, _, files_in_dir in os.walk(extract_dir):
                for file in files_in_dir:
                    if file.endswith('.wav'):
                        source_file = os.path.join(root, file)
                        destination_file = os.path.join(arni_path, file)
                        shutil.move(source_file, destination_file)

            # Clean up: Delete the extracted directory and the .zip file
            shutil.rmtree(extract_dir)
            os.remove(zip_filepath)
            subnet_logger(
                severity="TRACE",
                message=f"Downloaded portion of Arni dataset from url: {url}",
                log_level=log_level
            )

        except Exception as e:
            subnet_logger(
                severity="ERROR",
                message=f"Error downloading or processing {url}: {e}",
                log_level=log_level
            )
            raise e

def download_wham(wham_path: str, log_level:str = "INFO") -> None:
    """Downloads WHAM! 48kHz noise dataset. [2]
    
    [2] Wichern, Gordon, Antognini, Joe, Flynn, Michael, Zhu, Licheng Richard, 
    McQuinn, Emmett, Crow, Dwight, Manilow, Ethan, & Le Roux, Jonathan.
    WHAM!: Extending Speech Separation to Noisy Environments. 
    In Proceedings of Interspeech, September 2019.
    http://wham.whisper.ai/

    Args:
        :param wham_path: (str): Path to save WHAM! 48kHz dataset
        :param log_level: (str, optional): Log level for operations. Defaults to "INFO".

    Raises:
        Exception: Raised if download fails in any way.
    """
    try:
        url = 'https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip'
        file_name = 'high_res_wham.zip'
        file_path = os.path.join(wham_path, file_name)

        # Send a GET request to the URL and stream the content
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Check if the request was successful
            # Open a local file in write-binary mode
            with open(file_path, 'wb') as f:
                # Write the content in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

        # Unzip the file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(wham_path)

        # Define the directory containing the extracted audio files
        extracted_audio_dir = os.path.join(wham_path, 'high_res_wham', 'audio')

        # Move all .wav files from the extracted directory to wham_path
        for file_name in os.listdir(extracted_audio_dir):
            if file_name.endswith('.wav'):
                src_path = os.path.join(extracted_audio_dir, file_name)
                dest_path = os.path.join(wham_path, file_name)
                shutil.move(src_path, dest_path)

        # Remove the high_res_wham directory after moving the .wav files
        shutil.rmtree(os.path.join(wham_path, 'high_res_wham'))

        # Delete the zip file after extraction
        os.remove(file_path)
        
    except Exception as e:
        subnet_logger(
            severity="ERROR",
            message=f"WHAM download failed. Exception: {e}",
            log_level=log_level
        )
        raise e

# Check dataset directories, create if they do not exist. Then download WHAM and Arni datasets
def dataset_download(wham_path: str, arni_path: str, log_level: str = "INFO", partial: bool = False) -> bool:
    """Downloads ARNI and WHAM! 48kHz datasets. 

    Args:
        :param wham_path: (str): Path to save WHAM! 48kHz dataset
        :param arni_path: (str): Output path to save ARNI files.
        :param log_level: (str, optional): Log level for operations. Defaults to "INFO".
        :param partial: (bool, optional): Set to True if you only want to partially download the dataset for testing purposes. Defaults to False.

    Returns:
        bool: _description_
    """
    # Check if dataset directories exist, create if not
    for directory in [wham_path, arni_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Check if Arni dataset already downloaded, download if not
    if not any(file.endswith(".wav") for file in os.listdir(arni_path)):
        try:
            subnet_logger(
                severity="INFO",
                message="Starting download of ARNI dataset.",
                log_level=log_level
            )
            
            download_arni(arni_path=arni_path, log_level=log_level, partial=partial)
            subnet_logger(
                severity="INFO",
                message="Arni dataset download complete.",
                log_level=log_level
            )
        except:
            subnet_logger(
                severity="ERROR",
                message="Arni datset download failed. Please contact subnet owners if error persists. Exiting neuron.",
                log_level=log_level
            )
            return False
    else:
        subnet_logger(
                severity="INFO",
                message="Arni dataset has already been downloaded.",
                log_level=log_level
            )

    # Check if WHAM dataset already downloaded, download if not
    if not any(file.endswith(".wav") for file in os.listdir(wham_path)):
        try:
            subnet_logger(
                severity="INFO",
                message="Starting download of WHAM dataset.",
                log_level=log_level
            )
            download_wham(wham_path=wham_path)
            subnet_logger(
                severity="INFO",
                message="WHAM dataset download complete.",
                log_level=log_level
            )
        except:
            subnet_logger(
                severity="ERROR",
                message="WHAM datset download failed. Please contact subnet owners if error persists. Exiting neuron.",
                log_level=log_level
            )
            return False
    else:
        subnet_logger(
                severity="INFO",
                message="WHAM dataset has already been downloaded.",
                log_level=log_level
            )
        
    return True