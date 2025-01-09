import os 
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import convolve
import pyloudnorm as pyln
from typing import List
from scipy import stats

import soundsright.base.utils as Utils
import soundsright.base.data as Data

def _obtain_random_rir_from_arni(arni_dir_path: str) -> str:
    """Returns random RIR from Arni dataset as a list.

    Args:
       :param arni_dir_path: (str): Path to ARNI dataset.

    Returns:
        str: Path to .wav file in ARNI dataset.
    """
    # Get all .wav files in the ARNI directory (including subdirectories if needed)
    wav_files = [os.path.join(root, f) for root, dirs, files in os.walk(arni_dir_path) for f in files if f.endswith('.wav')]
    
    # Raise an error if no .wav files are found
    if not wav_files:
        raise ValueError(f"No .wav files found in the directory {arni_dir_path}.")
    
    # Select and return a random .wav file
    return random.choice(wav_files)
    
def calc_rt60(h, sr, rt='t30') -> float: 
    """
    RT60 measurement routine acording to Schroeder's method [1].

    [1] M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.

    Adapted from https://github.com/python-acoustics/python-acoustics/blob/99d79206159b822ea2f4e9d27c8b2fbfeb704d38/acoustics/room.py#L156
    
    Args:
        :param h: (np.ndarray): The RIR signal.
        :param sr: (int): The sample rate.
        :param rt: (str): The RT60 calculation to make. Default is 't30'
    """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    h_abs = np.abs(h) / np.max(np.abs(h))

    # Schroeder integration
    sch = np.cumsum(h_abs[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch)+1e-20)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / sr
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60

def _convolve_tts_with_random_rir(
    tts_path: str, 
    arni_dir_path: str, 
    reverb_dir_path: str, 
    max_rt60: float = 2.0,
) -> None:
    """
    Convolves a mono audio file with a random RIR from the Arni dataset and saves the output.
    The RIR is resampled to match the audio's sample rate before convolution.
    
    This method was adapted from the generation of the EARS-Reverb dataset [2]
    
    [2] J. Richter, Y.-C. Wu, S. Krenn, S. Welker, B. Lay, S. Watanabe, A. Richard, and T. Gerkmann, 
    "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation," 
    in Proc. ISCA Intertts, pp. 4873-4877, 2024.
    
    Here is a link to the code: https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_reverb.py
    
    Args:
        :param tts_path: (str): Path to the input TTS audio file.
        :param arni_dir_path: (str); Path to the directory containing the Arni RIR dataset.
        :param reverb_dir_path: (str): Path to save the convolved output file.
        
    Returns:
        None
    """
    # Obtain output path from tts_path 
    output_path = os.path.join(reverb_dir_path, os.path.basename(tts_path))
    
    tts, tts_sr = sf.read(tts_path)
    meter = pyln.Meter(tts_sr)

    # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples
    rt60 = np.inf
    while rt60 > max_rt60:
        rir_file = _obtain_random_rir_from_arni(arni_dir_path=arni_dir_path)

        rir, sr = sf.read(rir_file, always_2d=True)
        
        # Take random channel if file is multi-channel
        channel = np.random.randint(0, rir.shape[1])
        rir = rir[:,channel]
        assert sr == 44100
        rir = librosa.resample(rir, orig_sr=sr, target_sr=tts_sr)

        # Cut RIR to get direct path at the beginning
        max_index = np.argmax(np.abs(rir))
        rir = rir[max_index:]

        # Normalize RIRs in range [0.1, 0.7]
        if np.max(np.abs(rir)) < 0.1:
            rir = 0.1 * rir / np.max(np.abs(rir))
        elif np.max(np.abs(rir)) > 0.7:
            rir = 0.7 * rir / np.max(np.abs(rir))

        rt60 = calc_rt60(rir, sr=sr)

        mixture = convolve(tts, rir)[:len(tts)]

        # normalize mixture
        loudness_tts = meter.integrated_loudness(tts)
        loudness_mixture = meter.integrated_loudness(mixture)
        delta_loudness = loudness_tts - loudness_mixture
        gain = np.power(10.0, delta_loudness/20.0)
        # if gain is inf sample again
        if np.isinf(gain):
            rt60 = np.inf
        mixture = gain * mixture

    if np.max(np.abs(mixture)) > 1.0:
        mixture = mixture / np.max(np.abs(mixture))

    sf.write(output_path, mixture, tts_sr)

def convolve_all_tts_with_random_rir(tts_dir_path: str, arni_dir_path: str, reverb_dir_path: str) -> None:
    """Generates the entire reverberant database.

    Args:
        :param tts_dir_path: (str): Path to clean TTS dataset.
        :param arni_dir_path: (str): Path to ARNI datasrt.
        :param reverb_dir_path: (str): Path to save reverberant dataset.
    
    Returns:
        None
    """
    tts_paths = [os.path.join(tts_dir_path, f) for f in os.listdir(tts_dir_path) if f.endswith('.wav')]
    for tts_path in tts_paths:
        _convolve_tts_with_random_rir(tts_path=tts_path, arni_dir_path=arni_dir_path, reverb_dir_path=reverb_dir_path)

def _obtain_random_noise_from_wham(wham_dir_path: str) -> str:
    """
    Returns the full path of a randomly selected .wav file from the specified directory.
    
    Args:
        :param directory: (str): The path to the directory containing .wav files.

    Returns:
        str: The full path to the randomly selected .wav file.

    Raises:
        ValueError: If no .wav files are found in the directory.
    """
    # List all files in the directory
    files_in_directory = os.listdir(wham_dir_path)
    
    # Filter out only .wav files
    wav_files = [file for file in files_in_directory if file.lower().endswith('.wav')]
    
    if not wav_files:
        raise ValueError(f"No .wav files found in the directory: {wham_dir_path}")
    
    # Choose a random .wav file
    random_wav_file = random.choice(wav_files)
    
    # Get the full path
    full_path = os.path.join(wham_dir_path, random_wav_file)
    
    return full_path

def _add_random_wham_noise_to_tts(
    tts_path: str, 
    wham_dir_path: str, 
    noise_dir_path: str, 
    min_snr: float = -2.5, 
    max_snr: float = 17.5, 
    ramp_time_in_ms: int = 10
    ) -> None:
    """
    Adds random WHAM noise to a TTS audio file with a specified SNR.
    
    This method was adapted from the generation of the EARS-WHAM dataset [2]
    
    [2] J. Richter, Y.-C. Wu, S. Krenn, S. Welker, B. Lay, S. Watanabe, A. Richard, and T. Gerkmann, 
    "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation," 
    in Proc. ISCA Intertts, pp. 4873-4877, 2024.
    
    Here is a link to the code: https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_wham.py
    
    Args:
        :param tts_path: (str): Path to the TTS .wav file.
        :param wham_dir_path: (str): Path to the directory containing WHAM .wav files.
        :param noise_dir_path: (str): Path where the output noisy TTS should be saved.
        :param min_snr: (float): Minimum SNR (in dB).
        :param max_snr: (float): Maximum SNR (in dB).
        :param ramp_time_in_ms: (float): Duration of the ramp at the start and end in milliseconds.
    
    Returns:
        None
    """
    # Get a random noise .wav file path
    noise_wav_path = _obtain_random_noise_from_wham(wham_dir_path)
    
    # Load the TTS and noise audio files
    tts_audio, tts_sr = librosa.load(tts_path, sr=None)
    noise_audio, noise_sr = librosa.load(noise_wav_path, sr=None)
    
    # Resample noise if needed to match the TTS sampling rate
    if noise_sr != tts_sr:
        noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=tts_sr)
    
    # If noise is longer than the TTS audio, select a random segment
    if len(noise_audio) > len(tts_audio):
        max_start = len(noise_audio) - len(tts_audio)
        start_idx = np.random.randint(0, max_start)
        noise_audio = noise_audio[start_idx:start_idx + len(tts_audio)]
    else:
        # Ensure noise is the same length as the TTS audio (loop if necessary)
        repeats = int(np.ceil(len(tts_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)[:len(tts_audio)]
    
    # Choose a random SNR value between min_snr and max_snr
    snr_dB = random.uniform(min_snr, max_snr)
    
    # Perform loudness normalization to match the target SNR
    meter = pyln.Meter(tts_sr)
    loudness_tts = meter.integrated_loudness(tts_audio)
    loudness_noise = meter.integrated_loudness(noise_audio)
    
    # Calculate the required gain for the noise
    target_loudness = loudness_tts - snr_dB
    delta_loudness = target_loudness - loudness_noise
    gain = np.power(10.0, delta_loudness / 20.0)
    noise_scaled = gain * noise_audio
    
    # Mix the TTS audio with the scaled noise
    mixture = tts_audio + noise_scaled
    
    # Adjust for clipping by increasing SNR if needed
    while np.max(np.abs(mixture)) >= 1.0:
        snr_dB += 1  # Increase SNR to reduce noise level
        target_loudness = loudness_tts - snr_dB
        delta_loudness = target_loudness - loudness_noise
        gain = np.power(10.0, delta_loudness / 20.0)
        noise_scaled = gain * noise_audio
        mixture = tts_audio + noise_scaled
    
    # Apply ramps at beginning and end
    ramp_duration = ramp_time_in_ms / 1000.0  # Convert ramp time to seconds
    ramp_samples = int(ramp_duration * tts_sr)
    ramp = np.linspace(0, 1, ramp_samples)
    
    # Apply ramps to the mixture
    mixture[:ramp_samples] *= ramp
    mixture[-ramp_samples:] *= ramp[::-1]
    
    # Apply ramps to the original tts for consistency
    tts_audio[:ramp_samples] *= ramp
    tts_audio[-ramp_samples:] *= ramp[::-1]
    
    filename = os.path.basename(tts_path)
    noise_path = os.path.join(noise_dir_path, filename)

    # Save the resulting mixture to the specified noise_dir_path
    sf.write(noise_path, mixture, tts_sr)

def add_random_wham_noise_to_all_tts(tts_dir_path: str, wham_dir_path: str, noise_dir_path: str) -> None:
    """Generates full noisy dataset.

    Args:
        :param tts_dir_path: (str): Path to clean TTS dataset.
        :param wham_dir_path: (str): Path to WHAM! dataset.
        :param noise_dir_path: (str): Path to noisy dataset.
        
    Returns:
        None
    """
    tts_paths = [os.path.join(tts_dir_path, f) for f in os.listdir(tts_dir_path) if f.endswith('.wav')]
    for tts_path in tts_paths:
        _add_random_wham_noise_to_tts(tts_path=tts_path, wham_dir_path=wham_dir_path, noise_dir_path=noise_dir_path)
            
def create_noise_and_reverb_data_for_all_sampling_rates(
    tts_base_path: str, 
    arni_dir_path: str, 
    reverb_base_path: str,
    wham_dir_path: str, 
    noise_base_path: str, 
    tasks: List[str],
    log_level: str) -> None:
    """Takes TTS dataset and applies noise and/or reverb 
    to generate noise and/or reverb datasets.

    Args:
        :param tts_base_path: (str): Path to clean TTs dataset.
        :param arni_dir_path: (str): _description_
        :param reverb_base_path: (str): _description_
        :param wham_dir_path: (str): _description_
        :param noise_base_path: (str): _description_
        :param tasks: (List[str]): _description_
    
    Returns:
        None
    """
    
    # Iterate through each sub-directory in the TTS base path
    for dir_name in os.listdir(tts_base_path):
        tts_dir_path = os.path.join(tts_base_path, dir_name)
        
        # Ensure it is a directory
        if os.path.isdir(tts_dir_path):
            # Define the corresponding reverb and noise sub-directory paths
            reverb_dir_path = os.path.join(reverb_base_path, dir_name)
            noise_dir_path = os.path.join(noise_base_path, dir_name)
            
            if 'DENOISING' in tasks:
                # Create the noise sub-directory if it does not exist
                if not os.path.exists(noise_dir_path):
                    os.makedirs(noise_dir_path)
                add_random_wham_noise_to_all_tts(tts_dir_path=tts_dir_path, wham_dir_path=wham_dir_path, noise_dir_path=noise_dir_path)   
                
                Utils.subnet_logger(
                    severity="DEBUG",
                    message=f"Denoising dataset created in directory: {noise_dir_path}",
                    log_level=log_level,
                )

            if 'DEREVERBERATION' in tasks:
                # Create the reverb sub-directory if it does not exist
                if not os.path.exists(reverb_dir_path):
                    os.makedirs(reverb_dir_path)
                convolve_all_tts_with_random_rir(tts_dir_path=tts_dir_path, arni_dir_path=arni_dir_path, reverb_dir_path=reverb_dir_path)
                
                Utils.subnet_logger(
                    severity="DEBUG",
                    message=f"Dereverberation dataset created in directory: {reverb_dir_path}",
                    log_level=log_level,
                )
    
def reset_all_data_directories(tts_base_path: str, reverb_base_path: str, noise_base_path: str) -> bool:
    """
    Removes all .wav files from the subdirectories of the specified base paths.

    Args:
        :param tts_base_path: Base path containing subdirectories with .wav files.
        :param reverb_base_path: Base path containing subdirectories with .wav files.
        :param noise_base_path: Base path containing subdirectories with .wav files.
        
    Returns:
        bool: True if operation was successful, False if not.
    """
    # List of all base paths to process
    base_paths = [tts_base_path, reverb_base_path, noise_base_path]
    
    for base_path in base_paths:
        # Iterate through each subdirectory within the base path
        for subdir_name in os.listdir(base_path):
            subdir_path = os.path.join(base_path, subdir_name)
            
            # Check if it is indeed a directory
            if os.path.isdir(subdir_path):
                # Iterate through all files in the subdirectory
                for file_name in os.listdir(subdir_path):
                    # Check if the file is a .wav file
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(subdir_path, file_name)
                        
                        # Remove the .wav file
                        try:
                            os.remove(file_path)
                        except OSError as e:
                            Utils.subnet_logger(
                                severity="ERROR",
                                message=f"Error removing file located at: {file_path}"
                            )
                            return False
    
    return True
                        
def generate_dataset_for_miner(
    clean_dir: str, 
    sample_rate: int, 
    n: int, 
    task: str, 
    reverb_data_dir: str | None = None, 
    noise_data_dir: str | None = None, 
    noise_dir: str | None = None, 
    reverb_dir: str | None = None
) -> None:
    """Function to generate fine-tuning datasets for miners.

    Args:
        clean_dir (str): Path to clean TTS dataset.
        sample_rate (int): Sample rate.
        n (int): Number of elements in each dataset.
        task (str): "denoising" or "dereverberation".
        reverb_data_dir (str | None, optional): ARNI dataset path. Defaults to None (for if you are only looking to generate a noisy dataset).
        noise_data_dir (str | None, optional): WHAM! dataset path. Defaults to None (for if you are only looking to generate a reverberant dataset).
        noise_dir (str | None, optional): Noisy dataset path. Defaults to None (for if you are only looking to generate a reverberant dataset).
        reverb_dir (str | None, optional): Reverb dataset path. Defaults to None (for if you are only looking to generate a noisy dataset).

    Raises:
        Exception: Raised if there is an issue during either download or generation.
        
    Returns: 
        None
    """
    assert task in ['denoising', 'dereverberation', 'both'], "Input argument: task must be one of: 'denoising', 'dereverberation', 'both'"
    assert isinstance(sample_rate, int), "Input argument: sample_rate must be of type int"
    assert sample_rate in [16000], "Input argument: sample_rate must be 16000"
    assert reverb_data_dir or noise_data_dir, "At least one of input arguments: reverb_data_dir or noise_data_dir must be specified. If you want to generate both reverb and noise datasets (inputting 'both' into task), then both must be specified."
    
    dirs_to_make = []
    for d in [clean_dir, noise_dir, reverb_dir, noise_data_dir, reverb_data_dir]:
        if d:
            dirs_to_make.append(d)
    
    for directory in dirs_to_make:
        if not os.path.exists(directory):
            os.makedirs(directory)    
        
    tts_handler = Data.TTSHandler(
        tts_base_path=clean_dir, 
        sample_rates = [sample_rate]
    )
    
    tts_handler.create_openai_tts_dataset(
        sample_rate = sample_rate,
        n=n,
        for_miner=True
    )
    
    if task.lower() == "denoising":
        if not any(file.endswith(".wav") for file in os.listdir(noise_data_dir)):
            try:
                Data.download_wham(wham_path=noise_data_dir)
            except Exception as e:
                raise e("Noise dataset download failed.")
            
            add_random_wham_noise_to_all_tts(tts_dir_path=clean_dir, wham_dir_path=noise_data_dir, noise_dir_path=noise_dir)
            
    elif task.lower == 'dereverberation': 
        if not any(file.endswith(".wav") for file in os.listdir(reverb_data_dir)):
            try:
                Data.download_arni(arni_path=reverb_data_dir)
            except Exception as e:
                raise e("Reverb dataset download failed")
            
            convolve_all_tts_with_random_rir(tts_dir_path=clean_dir, arni_dir_path=reverb_data_dir, reverb_dir_path=reverb_dir)
            
    else: 
        Data.dataset_download(
            wham_path=noise_data_dir,
            arni_path=reverb_data_dir,
        )
        add_random_wham_noise_to_all_tts(tts_dir_path=clean_dir, wham_dir_path=noise_data_dir, noise_dir_path=noise_dir)
        convolve_all_tts_with_random_rir(tts_dir_path=clean_dir, arni_dir_path=reverb_data_dir, reverb_dir_path=reverb_dir)