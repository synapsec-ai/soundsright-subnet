import random
import os
from openai import OpenAI
import librosa
import soundfile as sf
from wonderwords import RandomSentence
from typing import List

from soundsright.base.utils import subnet_logger 
from soundsright.base.templates import (
    TOPICS,
    EMOTIONS
)

from dotenv import load_dotenv 
load_dotenv() 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Handles all TTS-related operations
class TTSHandler:
    
    def __init__(self, tts_base_path: str, sample_rates: List[int], log_level: str = "INFO"):
        self.tts_base_path = tts_base_path
        self.sample_rates = sample_rates
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)
        self.openai_voices = ['alloy','echo','fable','onyx','nova','shimmer']
        self.log_level = log_level
        self.rs = RandomSentence()

    # Generates unique sentences for TTS 
    def _generate_random_sentence(self, n: int=8) -> str:
        output = ""
        for _ in range(0,2):
            output += self.rs.sentence() + " "
        return output
        
    # Generates one output TTS file at correct sample rate
    def _do_single_openai_tts_query(self, tts_file_path: str, sample_rate: int, voice: str = 'random'):
        # voice control
        if voice == 'random' or voice not in self.openai_voices:
            voice = random.choice(self.openai_voices)
        # define openai call params
        params = {
            'model':'tts-1-hd',
            'voice':voice,
            'input':self._generate_random_sentence()
        }
        # call openai with client 
        try:
            response=self.openai_client.audio.speech.create(**params)
            response.stream_to_file(tts_file_path)
            subnet_logger(
                severity="TRACE",
                message=f"Obtained TTS audio file: {tts_file_path} from OpenAI.",
                log_level=self.log_level
            )
        # raise error if it fails
        except Exception as e:
            subnet_logger(
                severity="ERROR",
                message="Could not get TTS audio file from OpenAI, please check configuration.",
                log_level=self.log_level
            )
        # resample in place if necessary
        try:
            # Load the generated TTS audio file
            audio_data, sr = librosa.load(tts_file_path, sr=None)
            
            # Check if sample rate matches
            if sr != sample_rate:
                # Resample the audio
                audio_data_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                
                # Write the resampled audio back to the same file
                sf.write(tts_file_path, audio_data_resampled, sample_rate)
                
                # Log the resampling action
                subnet_logger(
                    severity="TRACE",
                    message=f"Resampled audio file '{tts_file_path}' from {sr} Hz to {sample_rate} Hz.",
                    log_level=self.log_level
                )
        except Exception as e:
            subnet_logger(
                severity="ERROR",
                message=f"Error during resampling of '{tts_file_path}': {e}",
                log_level=self.log_level
            )

    # Creates TTS dataset of length n at specified sample rate
    def create_openai_tts_dataset(self, sample_rate: int, n:int, for_miner: bool = False):
        # define output file location and make directory if it doesn't exist
        if for_miner: 
            output_dir = self.tts_base_path
        else:
            output_dir = os.path.join(self.tts_base_path, str(sample_rate))
        os.makedirs(output_dir, exist_ok=True)
        # count to n and make files
        for i in range(n):
            self._do_single_openai_tts_query(
                tts_file_path = os.path.join(output_dir, (str(i) + ".wav")),
                sample_rate=sample_rate
            )

    # Create TTS dataset of length n for all sample rates
    def create_openai_tts_dataset_for_all_sample_rates(self, n:int, seed:int):

        random.seed(seed)
            
        for sample_rate in self.sample_rates: 
            self.create_openai_tts_dataset(
                sample_rate=sample_rate,
                n=n
            )