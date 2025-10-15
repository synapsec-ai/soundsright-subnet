import librosa
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import numpy as np
from scipy import stats 
import os 
from typing import List, Tuple

import soundsright.base.utils as Utils

def si_sdr_components(s_hat, s, n):
    """
    Adapted from SGMSE+ [1,2,3,4]
    
    [1] Richter, Julius, de Oliveira, Danilo, & Gerkmann, Timo.
        Investigating Training Objectives for Generative Speech Enhancement.  
        arXiv preprint, https://arxiv.org/abs/2409.10753, 2024.  
    
    [2] Richter, Julius, Welker, Simon, Lemercier, Jean-Marie, Lay, Bunlong, & Gerkmann, Timo.  
        Speech Enhancement and Dereverberation with Diffusion-based Generative Models.  
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31, 2351–2364.  
        https://doi.org/10.1109/TASLP.2023.3285241, 2023.  
    
    [3] Richter, Julius, Wu, Yi-Chiao, Krenn, Steven, Welker, Simon, Lay, Bunlong, Watanabe, Shinji, Richard, Alexander, & Gerkmann, Timo.
        EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation. 
        In ISCA Interspeech, 2024.
    
    [4] Welker, Simon, Richter, Julius, & Gerkmann, Timo.
        Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain. 
        In Proceedings of Interspeech 2022, 2928–2932.  
        https://doi.org/10.21437/Interspeech.2022-10653, 2022.  
    """
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
    """
    Adapted from SGMSE+ [1,2,3,4]
    
    [1] Richter, Julius, de Oliveira, Danilo, & Gerkmann, Timo.
        Investigating Training Objectives for Generative Speech Enhancement.  
        arXiv preprint, https://arxiv.org/abs/2409.10753, 2024.  
    
    [2] Richter, Julius, Welker, Simon, Lemercier, Jean-Marie, Lay, Bunlong, & Gerkmann, Timo.  
        Speech Enhancement and Dereverberation with Diffusion-based Generative Models.  
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31, 2351–2364.  
        https://doi.org/10.1109/TASLP.2023.3285241, 2023.  
    
    [3] Richter, Julius, Wu, Yi-Chiao, Krenn, Steven, Welker, Simon, Lay, Bunlong, Watanabe, Shinji, Richard, Alexander, & Gerkmann, Timo.
        EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation. 
        In ISCA Interspeech, 2024.
    
    [4] Welker, Simon, Richter, Julius, & Gerkmann, Timo.
        Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain. 
        In Proceedings of Interspeech 2022, 2928–2932.  
        https://doi.org/10.21437/Interspeech.2022-10653, 2022.  
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar

def calculate_si_sdr_for_directories(clean_directory: str, enhanced_directory: str, noisy_directory: str, sample_rate: int, log_level: str, confidence_level: float = 0.95) -> dict:
    """
    Calculate SI_SDR scores for all matching audio files in the given directories and compute the average SI_SDR score with a confidence interval.

    Parameters:
        :param clean_directory: (str): Path to the directory containing clean reference audio files.
        :param enhanced_directory: (str): Path to the directory containing enhanced degraded audio files.
        :param noisy_directory: (str): Path to the directory containing the noisy audio files.
        :param sample_rate: {int): Sampling rate to use for SI_SDR calculation (8000 or 16000 Hz).
        :param confidence_level: (float): Confidence level for the confidence interval (default is 0.95 for 95%).

    Returns:
        dict
    """

    # Get list of audio files in both directories
    clean_files = sorted([f for f in os.listdir(clean_directory) if f.lower().endswith('.wav')])
    enhanced_files = sorted([f for f in os.listdir(enhanced_directory) if f.lower().endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(noisy_directory) if f.lower().endswith('.wav')])
    # Match files by name
    matched_files = set(clean_files).intersection(enhanced_files, noisy_files)
    if not matched_files or len(matched_files) <= (0.95*len(clean_files)):
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error calculating SI_SDR: No matching audio files found in the provided directories.",
            log_level=log_level
        )
        raise ValueError("No matching audio files found in the provided directories.")

    si_sdr_scores = []

    for file_name in matched_files:
        try:
            clean_audio_path = os.path.join(clean_directory, file_name)
            enhanced_audio_path = os.path.join(enhanced_directory, file_name)
            noisy_audio_path = os.path.join(noisy_directory, file_name)
            
            # Load the clean audio file
            clean_audio, clean_sr = sf.read(clean_audio_path)
            # Load the enhanced audio file
            enhanced_audio, enhanced_sr = sf.read(enhanced_audio_path)
            # Load the noisy audio file
            noisy_audio, noisy_sr = sf.read(noisy_audio_path)
            
            if clean_sr != enhanced_sr or clean_sr != noisy_sr:
                continue

            # Ensure the signals have the same length
            if len(clean_audio) != len(enhanced_audio) or len(noisy_audio) != len(enhanced_audio):
                continue

            # Convert to float32 type
            clean_audio = clean_audio.astype(np.float32)
            enhanced_audio = enhanced_audio.astype(np.float32)
            noisy_audio = noisy_audio.astype(np.float32)
            noise = noisy_audio - clean_audio

            # Calculate the SI_SDR score
            si_sdr_score = float(energy_ratios(enhanced_audio, clean_audio, noise)[0])
            si_sdr_scores.append(si_sdr_score)
        
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Exception occured during calculation of SI-SDR: {e}",
                log_level=log_level
            )
            continue

    if not si_sdr_scores:
        Utils.subnet_logger(
            severity="ERROR",
            message="No SI_SDR scores were calculated. Check your audio files and directories.",
            log_level=log_level,
        )
        raise ValueError("No SI_SDR scores were calculated. Check your audio files and directories.")

    # Calculate average SI_SDR score
    average_si_sdr = np.mean(si_sdr_scores)

    # Calculate confidence interval
    n = len(si_sdr_scores)
    stderr = stats.sem(si_sdr_scores)
    t_score = stats.t.ppf((1 + confidence_level) / 2.0, df=n - 1)
    margin_of_error = t_score * stderr
    confidence_interval = (average_si_sdr - margin_of_error, average_si_sdr + margin_of_error)

    output = {
        "scores":si_sdr_scores, 
        "average":average_si_sdr, 
        "confidence_interval":confidence_interval,
    }
    
    Utils.subnet_logger(
        severity="INFO",
        message=f"SI_SDR metrics output: {output}",
        log_level=log_level,
    )
    
    return output
    
def calculate_si_sir_for_directories(clean_directory: str, enhanced_directory: str, noisy_directory: str, sample_rate: int, log_level: str, confidence_level: float = 0.95) -> dict:
    """
    Calculate SI_SIR scores for all matching audio files in the given directories and compute the average SI_SIR score with a confidence interval.

    Parameters:
        :param clean_directory: (str): Path to the directory containing clean reference audio files.
        :param enhanced_directory: (str): Path to the directory containing enhanced degraded audio files.
        :param noisy_directory: (str): Path to the directory containing the noisy audio files.
        :param sample_rate: {int): Sampling rate to use for SI_SIR calculation (8000 or 16000 Hz).
        :param confidence_level: (float): Confidence level for the confidence interval (default is 0.95 for 95%).

    Returns:
        dict
    """

    # Get list of audio files in both directories
    clean_files = sorted([f for f in os.listdir(clean_directory) if f.lower().endswith('.wav')])
    enhanced_files = sorted([f for f in os.listdir(enhanced_directory) if f.lower().endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(noisy_directory) if f.lower().endswith('.wav')])
    # Match files by name
    matched_files = set(clean_files).intersection(enhanced_files, noisy_files)
    if not matched_files:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error calculating SI_SIR: No matching audio files found in the provided directories.",
            log_level=log_level
        )
        raise ValueError("No matching audio files found in the provided directories.")

    si_sir_scores = []

    for file_name in matched_files or len(matched_files) <= (0.95*len(clean_files)):
        try:
            clean_audio_path = os.path.join(clean_directory, file_name)
            enhanced_audio_path = os.path.join(enhanced_directory, file_name)
            noisy_audio_path = os.path.join(noisy_directory, file_name)
            
            # Load the clean audio file
            clean_audio, clean_sr = sf.read(clean_audio_path)
            # Load the enhanced audio file
            enhanced_audio, enhanced_sr = sf.read(enhanced_audio_path)
            # Load the noisy audio file
            noisy_audio, noisy_sr = sf.read(noisy_audio_path)
            
            if clean_sr != enhanced_sr or clean_sr != noisy_sr:
                continue

            # Ensure the signals have the same length
            if len(clean_audio) != len(enhanced_audio) or len(noisy_audio) != len(enhanced_audio):
                continue

            # Convert to float32 type
            clean_audio = clean_audio.astype(np.float32)
            enhanced_audio = enhanced_audio.astype(np.float32)
            noisy_audio = noisy_audio.astype(np.float32)
            noise = noisy_audio - clean_audio

            # Calculate the SI_SIR score
            si_sir_score = float(energy_ratios(enhanced_audio, clean_audio, noise)[1])
            si_sir_scores.append(si_sir_score)
        
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Exception occured during calculation of SI-SIR: {e}",
                log_level=log_level
            )
            continue

    if not si_sir_scores:
        Utils.subnet_logger(
            severity="ERROR",
            message="No SI_SIR scores were calculated. Check your audio files and directories.",
            log_level=log_level,
        )
        raise ValueError("No SI_SIR scores were calculated. Check your audio files and directories.")

    # Calculate average SI_SIR score
    average_si_sir = np.mean(si_sir_scores)

    # Calculate confidence interval
    n = len(si_sir_scores)
    stderr = stats.sem(si_sir_scores)
    t_score = stats.t.ppf((1 + confidence_level) / 2.0, df=n - 1)
    margin_of_error = t_score * stderr
    confidence_interval = (average_si_sir - margin_of_error, average_si_sir + margin_of_error)

    output = {
        "scores":si_sir_scores, 
        "average":average_si_sir, 
        "confidence_interval":confidence_interval,
    }
    
    Utils.subnet_logger(
        severity="INFO",
        message=f"SI_SIR metrics output: {output}",
        log_level=log_level,
    )
    
    return output
    
def calculate_si_sar_for_directories(clean_directory: str, enhanced_directory: str, noisy_directory: str, sample_rate: int, log_level: str, confidence_level: float = 0.95) -> dict:
    """
    Calculate SI_SAR scores for all matching audio files in the given directories and compute the average SI_SAR score with a confidence interval.

    Parameters:
        :param clean_directory: (str): Path to the directory containing clean reference audio files.
        :param enhanced_directory: (str): Path to the directory containing enhanced degraded audio files.
        :param noisy_directory: (str): Path to the directory containing the noisy audio files.
        :param sample_rate: {int): Sampling rate to use for SI_SAR calculation (8000 or 16000 Hz).
        :param confidence_level: (float): Confidence level for the confidence interval (default is 0.95 for 95%).

    Returns:
        dict
    """

    # Get list of audio files in both directories
    clean_files = sorted([f for f in os.listdir(clean_directory) if f.lower().endswith('.wav')])
    enhanced_files = sorted([f for f in os.listdir(enhanced_directory) if f.lower().endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(noisy_directory) if f.lower().endswith('.wav')])
    # Match files by name
    matched_files = set(clean_files).intersection(enhanced_files, noisy_files)
    if not matched_files or len(matched_files) <= (0.95*len(clean_files)):
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error calculating SI_SAR: No matching audio files found in the provided directories.",
            log_level=log_level
        )
        raise ValueError("No matching audio files found in the provided directories.")

    si_sar_scores = []

    for file_name in matched_files:
        try:
            clean_audio_path = os.path.join(clean_directory, file_name)
            enhanced_audio_path = os.path.join(enhanced_directory, file_name)
            noisy_audio_path = os.path.join(noisy_directory, file_name)
            
            # Load the clean audio file
            clean_audio, clean_sr = sf.read(clean_audio_path)
            # Load the enhanced audio file
            enhanced_audio, enhanced_sr = sf.read(enhanced_audio_path)
            # Load the noisy audio file
            noisy_audio, noisy_sr = sf.read(noisy_audio_path)
            
            if clean_sr != enhanced_sr or clean_sr != noisy_sr:
                continue

            # Ensure the signals have the same length
            if len(clean_audio) != len(enhanced_audio) or len(noisy_audio) != len(enhanced_audio):
                continue

            # Convert to float32 type
            clean_audio = clean_audio.astype(np.float32)
            enhanced_audio = enhanced_audio.astype(np.float32)
            noisy_audio = noisy_audio.astype(np.float32)
            noise = noisy_audio - clean_audio

            # Calculate the SI_SAR score
            si_sar_score = float(energy_ratios(enhanced_audio, clean_audio, noise)[2])
            si_sar_scores.append(si_sar_score)
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Exception occured during calculation of SI-SAR: {e}",
                log_level=log_level
            )
            continue

    if not si_sar_scores:
        Utils.subnet_logger(
            severity="ERROR",
            message="No SI_SAR scores were calculated. Check your audio files and directories.",
            log_level=log_level,
        )
        raise ValueError("No SI_SAR scores were calculated. Check your audio files and directories.")

    # Calculate average SI_SAR score
    average_si_sar = np.mean(si_sar_scores)

    # Calculate confidence interval
    n = len(si_sar_scores)
    stderr = stats.sem(si_sar_scores)
    t_score = stats.t.ppf((1 + confidence_level) / 2.0, df=n - 1)
    margin_of_error = t_score * stderr
    confidence_interval = (average_si_sar - margin_of_error, average_si_sar + margin_of_error)

    output = {
        "scores":si_sar_scores, 
        "average":average_si_sar, 
        "confidence_interval":confidence_interval,
    }
    
    Utils.subnet_logger(
        severity="INFO",
        message=f"SI_SAR metrics output: {output}",
        log_level=log_level,
    )
    
    return output

def calculate_pesq_for_directories(clean_directory: str, enhanced_directory: str, sample_rate: int, log_level: str, confidence_level: float = 0.95) -> dict:
    """
    Calculate PESQ scores for all matching audio files in the given directories and compute the average PESQ score with a confidence interval.

    Parameters:
        :param clean_directory: (str): Path to the directory containing clean reference audio files.
        :param enhanced_directory: (str): Path to the directory containing enhanced degraded audio files.
        :param sample_rate: {int): Sampling rate to use for PESQ calculation (8000 or 16000 Hz).
        :param confidence_level: (float): Confidence level for the confidence interval (default is 0.95 for 95%).

    Returns:
        dict
    """

    # Get list of audio files in both directories
    clean_files = sorted([f for f in os.listdir(clean_directory) if f.lower().endswith('.wav')])
    
    enhanced_files = sorted([f for f in os.listdir(enhanced_directory) if f.lower().endswith('.wav')])

    # Match files by name
    matched_files = set(clean_files).intersection(enhanced_files)
    if not matched_files or len(matched_files) <= (0.95*len(clean_files)):
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error calculating PESQ: No matching audio files found in the provided directories.",
            log_level=log_level
        )
        raise ValueError("No matching audio files found in the provided directories.")

    pesq_scores = []

    for file_name in matched_files:
        try:
            clean_audio_path = os.path.join(clean_directory, file_name)
            enhanced_audio_path = os.path.join(enhanced_directory, file_name)
            
            # Load the clean audio file
            clean_audio, clean_sr = sf.read(clean_audio_path)
            
            # Load the enhanced audio file
            enhanced_audio, enhanced_sr = sf.read(enhanced_audio_path)
            
            if clean_sr != enhanced_sr:
                continue

            # Ensure the signals have the same length
            if len(clean_audio) != len(enhanced_audio):
                continue

            # Convert to float32 type
            clean_audio = clean_audio.astype(np.float32)
            enhanced_audio = enhanced_audio.astype(np.float32)

            # Set the mode based on the sample rate
            if sample_rate == 8000:
                mode = 'nb'  # Narrow-band
            elif sample_rate == 16000:
                mode = 'wb'  # Wide-band
            else:
                raise ValueError("Unsupported sample rate. Use 8000 or 16000 Hz.")

            # Calculate the PESQ score
            pesq_score = float(pesq(sample_rate, clean_audio, enhanced_audio, mode))
            
            pesq_scores.append(pesq_score)
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Exception occured during calculation of PESQ: {e}",
                log_level=log_level
            )
            continue

    if not pesq_scores:
        Utils.subnet_logger(
            severity="ERROR",
            message="No PESQ scores were calculated. Check your audio files and directories.",
            log_level=log_level,
        )
        raise ValueError("No PESQ scores were calculated. Check your audio files and directories.")

    # Calculate average PESQ score
    average_pesq = np.mean(pesq_scores)

    # Calculate confidence interval
    n = len(pesq_scores)
    stderr = stats.sem(pesq_scores)
    t_score = stats.t.ppf((1 + confidence_level) / 2.0, df=n - 1)
    margin_of_error = t_score * stderr
    confidence_interval = (average_pesq - margin_of_error, average_pesq + margin_of_error)

    output = {
        "scores":pesq_scores,
        "average":average_pesq,
        "confidence_interval":confidence_interval
    }
    
    Utils.subnet_logger(
        severity="INFO",
        message=f"PESQ metrics output: {output}",
        log_level=log_level,
    )
    
    return output

def calculate_estoi_for_directories(clean_directory: str, enhanced_directory: str, sample_rate: int, log_level: str, confidence_level: float = 0.95) -> dict:
    """
    Calculate ESTOI scores for all matching audio files in the given directories and compute the average ESTOI score with a confidence interval.

    Parameters:
        :param clean_directory: (str): Path to the directory containing clean reference audio files.
        :param enhanced_directory: (str): Path to the directory containing enhanced degraded audio files.
        :param sample_rate: {int): Sampling rate to use for ESTOI calculation (8000 or 16000 Hz).
        :param confidence_level: (float): Confidence level for the confidence interval (default is 0.95 for 95%).

    Returns:
        dict
    """

    # Get list of audio files in both directories
    clean_files = sorted([f for f in os.listdir(clean_directory) if f.lower().endswith('.wav')])
    enhanced_files = sorted([f for f in os.listdir(enhanced_directory) if f.lower().endswith('.wav')])

    # Match files by name
    matched_files = set(clean_files).intersection(enhanced_files)
    if not matched_files or len(matched_files) <= (0.95*len(clean_files)):
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error calculating ESTOI: No matching audio files found in the provided directories.",
            log_level=log_level
        )
        raise ValueError("No matching audio files found in the provided directories.")

    estoi_scores = []

    for file_name in matched_files:
        try:
            clean_audio_path = os.path.join(clean_directory, file_name)
            enhanced_audio_path = os.path.join(enhanced_directory, file_name)
            
            # Load the clean audio file
            clean_audio, clean_sr = sf.read(clean_audio_path)
            # Load the enhanced audio file
            enhanced_audio, enhanced_sr = sf.read(enhanced_audio_path)

            if clean_sr != enhanced_sr:
                continue

            # Ensure the signals have the same length
            if len(clean_audio) != len(enhanced_audio):
                continue

            # Convert to float32 type
            clean_audio = clean_audio.astype(np.float32)
            enhanced_audio = enhanced_audio.astype(np.float32)

            # Calculate the ESTOI score
            estoi_score = float(stoi(x=clean_audio, y=enhanced_audio, fs_sig=sample_rate))
            estoi_scores.append(estoi_score)
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Exception occured during calculation of ESTOI: {e}",
                log_level=log_level
            )
            continue

    if not estoi_scores:
        Utils.subnet_logger(
            severity="ERROR",
            message="No ESTOI scores were calculated. Check your audio files and directories.",
            log_level=log_level,
        )
        raise ValueError("No ESTOI scores were calculated. Check your audio files and directories.")

    # Calculate average ESTOI score
    average_estoi = np.mean(estoi_scores)

    # Calculate confidence interval
    n = len(estoi_scores)
    stderr = stats.sem(estoi_scores)
    t_score = stats.t.ppf((1 + confidence_level) / 2.0, df=n - 1)
    margin_of_error = t_score * stderr
    confidence_interval = (average_estoi - margin_of_error, average_estoi + margin_of_error)

    output = {
        "scores":estoi_scores, 
        "average":average_estoi, 
        "confidence_interval":confidence_interval,
    }
    
    Utils.subnet_logger(
        severity="INFO",
        message=f"ESTOI metrics output: {output}",
        log_level=log_level,
    )
    
    return output
    
def calculate_metrics_dict(task: str, sample_rate: int, clean_directory: str, enhanced_directory: str, noisy_directory: str, log_level: str) -> dict:
    
    metrics_dict = {}
    
    if sample_rate == 16000:
            
        try:
            # PESQ
            metrics_dict['PESQ'] = calculate_pesq_for_directories(
                clean_directory=clean_directory,
                enhanced_directory=enhanced_directory,
                sample_rate=sample_rate,
                log_level=log_level,
            )
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Error calculating PESQ: {e}"
            )
            metrics_dict['PESQ'] = {}
            
        try:
            # ESTOI
            metrics_dict['ESTOI'] = calculate_estoi_for_directories(
                clean_directory=clean_directory,
                enhanced_directory=enhanced_directory,
                sample_rate=sample_rate,
                log_level=log_level,
            )
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Error calculating ESTOI: {e}"
            )
            metrics_dict['ESTOI'] = {}

    elif sample_rate == 48000:

        if "denoising" in task.lower():
                
            try:
                # SI_SIR
                metrics_dict['SI_SIR'] = calculate_si_sir_for_directories(
                    clean_directory=clean_directory,
                    enhanced_directory=enhanced_directory,
                    noisy_directory=noisy_directory,
                    sample_rate=sample_rate,
                    log_level=log_level,
                )
            except Exception as e:
                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Error calculating SI-SIR: {e}"
                )
                metrics_dict['SI_SIR'] = {}
            
            try:
                # SI_SAR
                metrics_dict['SI_SAR'] = calculate_si_sar_for_directories(
                    clean_directory=clean_directory,
                    enhanced_directory=enhanced_directory,
                    noisy_directory=noisy_directory,
                    sample_rate=sample_rate,
                    log_level=log_level,
                )
            except Exception as e:
                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Error calculating SI-SAR: {e}"
                )
                metrics_dict['SI_SAR'] = {}

        try:
            # SI_SDR
            metrics_dict['SI_SDR'] = calculate_si_sdr_for_directories(
                clean_directory=clean_directory,
                enhanced_directory=enhanced_directory,
                noisy_directory=noisy_directory,
                sample_rate=sample_rate,
                log_level=log_level,
            )
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Error calculating SI-SDR: {e}"
            )
            metrics_dict['SI_SDR'] = {}
            
        try:
            # ESTOI
            metrics_dict['ESTOI'] = calculate_estoi_for_directories(
                clean_directory=clean_directory,
                enhanced_directory=enhanced_directory,
                sample_rate=sample_rate,
                log_level=log_level,
            )
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Error calculating ESTOI: {e}"
            )
            metrics_dict['ESTOI'] = {}
    
    return metrics_dict