import os
import time
import glob
import asyncio
import shutil
import hashlib
import subprocess
import bittensor as bt
from typing import List
import socket 

# Import custom modules
import soundsright.base.benchmarking as Benchmarking
import soundsright.base.models as Models
import soundsright.base.utils as Utils

class ModelEvaluationHandler:

    def __init__(
            self,
            eval_cache: dict,
            image_hotkey_list: list,
            hotkeys: list,
            competitions_list: list,
            ports_list: list,
            model_path: str,
            reverb_path: str,
            noise_path: str,
            tts_path: str,
            model_output_path: str,
            cuda_directory: str,
            use_docker: bool,
            log_level: str,
        ):

        # Dataset paths
        self.reverb_path = reverb_path
        self.noise_path = noise_path
        self.tts_path = tts_path
        self.model_base_path = model_path
        self.base_model_output_path = model_output_path

        # Eval cache
        self.eval_cache = eval_cache
        self.image_hotkey_list = image_hotkey_list
        self.competitions_list = competitions_list
        self.ports_list = ports_list
        self.current_ports_list = []
        self.models_per_iteration = 3
        self.tasks = []

        # Timeout calculations
        self.base_timeouts = {
            "status":60,
            "prepare":10,
            "upload":3,
            "enhance":400,
            "download":3,
            "reset":10,
        }

        self.timeout_multipliers = {
            "status":[1,1,1],
            "prepare":[1,0.95,0.95],
            "upload":[1,0.7,0.4],
            "enhance":[1,0.95,0.95],
            "download":[1,0.7,0.4],
            "reset":[1, 0.7, 0.4]
        }

        # Misc
        self.cuda_directory = cuda_directory
        self.log_level = log_level
        self.hotkeys = hotkeys
        self.use_docker = use_docker

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Initialized model evaluator with model per iteration: {self.models_per_iteration}, image hotkey list: {self.image_hotkey_list}, competitions list: {self.competitions_list}, ports list: {self.ports_list}, eval cache: {self.eval_cache}",
            log_level=self.log_level,
        )

    def _is_port_free(self, port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return True
            except OSError:
                return False
            
    def _kill_process_on_port(self, port: int) -> bool:
        try:
            result = subprocess.run(
                ["sudo", "fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Killed existing process on port: {port}",
                    log_level=self.log_level
                )
                return True
            else:
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Failed to kill existing process on port: {port}",
                    log_level=self.log_level
                )
                return False

        except Exception as e:
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Failed to kill existing process on port: {port} because: {e}",
                log_level=self.log_level
            )
        
        return False

    def calculate_timeouts(self, concurrent_length: int, sample_rate: int):

        sr_multiplier = 1
        if sample_rate==48000:
            sr_multiplier = 1.5

        timeouts = {}
        index = concurrent_length - 1
        if index < 0:
            index = 0
            concurrent_length = 1
        if index > 2:
            index = 2
            concurrent_length = 3

        for key in self.base_timeouts.keys():
            if key == "enhance":
                timeouts[key] = self.base_timeouts[key] * self.timeout_multipliers[key][index] * concurrent_length * sr_multiplier
            else: 
                timeouts[key] = self.base_timeouts[key] * self.timeout_multipliers[key][index] * concurrent_length

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Determined timeouts for model evaluation: {timeouts} with eval length: {concurrent_length}",
            log_level=self.log_level
        )

        return timeouts

    def prepare_directory(self, dir_path):
        """
        Creates directory if it does not exist, removes contents if it does
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Created directory: {dir_path}",
                log_level=self.log_level
            )

        else:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)

                except Exception as e:
                    Utils.subnet_logger(
                        severity="ERROR",
                        message=f"Failed to delete item at path: {item_path} because: {e}",
                        log_level=self.log_level
                    )

    def _reset_dir(self, directory: str) -> None:
        """Removes all files and sub-directories in an inputted directory

        Args:
            directory (str): Directory to reset.
        """
        # Check if the directory exists
        if not os.path.exists(directory):
            return

        # Loop through all the files and subdirectories in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if it's a file or directory and remove accordingly
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Failed to delete {file_path}. Reason: {e}",
                    log_level=self.log_level
                )

    def get_next_eval_round(self):

        hotkeys = self.image_hotkey_list[:self.models_per_iteration]
        self.image_hotkey_list = self.image_hotkey_list[self.models_per_iteration:]

        competitions = self.competitions_list[:self.models_per_iteration]
        self.competitions_list = self.competitions_list[self.models_per_iteration:]

        self.current_ports_list = self.ports_list[:self.models_per_iteration]
        self.ports_list = self.ports_list[self.models_per_iteration:]

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Next model evaluation round obtained. Hotkeys: {hotkeys} Competitions: {competitions} Ports: {self.current_ports_list}",
            log_level=self.log_level
        )

        return hotkeys, competitions, self.current_ports_list

    def validate_all_noisy_files_are_enhanced(self, task_path: str, model_output_path: str):
        noisy_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(task_path, '*.wav'))])
        enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(model_output_path, '*.wav'))])
        return noisy_files == enhanced_files
    
    async def get_tasks(self, hotkeys: list, competitions: list, ports: list):

        self.tasks = []
        for hotkey, competition, port in zip(hotkeys, competitions, ports):
            if port != 0:
                task = asyncio.create_task(
                    self.run_model_evaluation(
                        hotkey=hotkey,
                        competition=competition,
                        port=port,
                        concurrent_length=len(hotkeys)
                    )
                )
                self.tasks.append(task)

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Obtained tasks for next model eval round: {self.tasks}",
            log_level=self.log_level
        )

    async def run_model_evaluation(self, hotkey: str, competition: str, port: int, concurrent_length: int):

        if not self._is_port_free(port):
            
            if not self._kill_process_on_port(port):
                Utils.subnet_logger(
                    severity="WARNING",
                    message=f"Port {port} already in use and process was unable to be stopped; skipping {hotkey}.",
                    log_level=self.log_level,
                )
                self._reset_dir(directory=model_output_path)
                return hotkey, False

        tag_name = f"{hotkey}_{competition}".lower()
        competition_components = competition.split("_")
        task, sample_rate = competition_components[0], competition_components[1].replace("HZ", "")

        Utils.subnet_logger(
            severity="INFO",
            message=f"Starting model evaluation for miner: {hotkey} for competition: {competition} with port: {port} with tag name: {tag_name}",
            log_level=self.log_level
        )

        if "denoising" in task.lower():
            dataset_path = os.path.join(self.noise_path, sample_rate)
        else:
            dataset_path = os.path.join(self.reverb_path, sample_rate)

        model_output_path = os.path.join(self.base_model_output_path, hotkey)

        self.prepare_directory(dir_path=model_output_path)

        timeouts = self.calculate_timeouts(concurrent_length=concurrent_length, sample_rate=int(sample_rate))

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Starting model container for miner: {hotkey}",
            log_level=self.log_level,
        )

        try: 
            start_status = await Utils.do_start_container_async(
                tag_name=tag_name,
                cuda_directory=self.cuda_directory,
                port=port,
                log_level=self.log_level,
                use_docker=self.use_docker,
            )

        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Model container start failed for miner because of: {e}",
                log_level=self.log_level,
            )
            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        if not start_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Model container start failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Model container start successful for miner: {hotkey}. Now checking API status with timeout: {timeouts['status']}.",
            log_level=self.log_level,
        )
        
        init_status = await Utils.check_container_status_async(port=port, log_level=self.log_level, timeout=timeouts["status"])

        if not init_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"API check failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"API check successful for miner: {hotkey}. Now preparing model with timeout: {timeouts['prepare']}.",
            log_level=self.log_level,
        )
        
        prepare_status = await Utils.prepare_async(port=port, log_level=self.log_level, timeout=timeouts["prepare"])

        if not prepare_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Model preparation failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Model preparation successful for miner: {hotkey}. Now uploading noisy files with timeout: {timeouts['prepare']}.",
            log_level=self.log_level,
        )
        
        upload_status = await Utils.upload_audio_async(noisy_dir=dataset_path, port=port, log_level=self.log_level, timeout=timeouts["prepare"])

        if not upload_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Noisy file upload failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Noisy file upload successful for miner: {hotkey}. Now enhancing model with timeout: {timeouts['enhance']}.",
            log_level=self.log_level,
        )
        
        enhance_status = await Utils.enhance_audio_async(port=port, log_level=self.log_level, timeout=timeouts["enhance"])

        if not enhance_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Enhnacement failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Enhancement successful for miner: {hotkey}. Now downloading enhanced files with timeout: {timeouts['download']}.",
            log_level=self.log_level,
        )
        
        download_status = await Utils.download_enhanced_async(enhanced_dir=model_output_path, port=port, log_level=self.log_level, timeout=timeouts["download"])

        if not download_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Download failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Download successful for miner: {hotkey}. Now verifying output.",
            log_level=self.log_level,
        )

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Download successful for miner: {hotkey}. Now resetting model files with timeout: {timeouts['reset']}.",
            log_level=self.log_level,
        )

        reset_status = await Utils.reset_model_async(port=port, log_level=self.log_level, timeout=timeouts["reset"])

        if not reset_status:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Resetting model files failed for miner: {hotkey}.",
                log_level=self.log_level,
            )

            return hotkey, False

        if not self.validate_all_noisy_files_are_enhanced(
            task_path=dataset_path,
            model_output_path=model_output_path
        ):
            
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Output verification failed for miner: {hotkey}",
                log_level=self.log_level,
            )

            self._reset_dir(directory=model_output_path)
            return hotkey, False
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Output verification successful for miner: {hotkey}",
            log_level=self.log_level,
        )
        
        return hotkey, True
    
    async def run_eval_group(self, hotkeys: list, competitions: list):

        Utils.handle_iptables(ports=self.current_ports_list, log_level=self.log_level)

        outcomes = await asyncio.gather(*self.tasks, return_exceptions=True)

        output_benchmarks = []
        output_competitions = []

        for outcome in outcomes:

            if isinstance(outcome, Exception):
                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Model eval task raised: {repr(outcome)}",
                    log_level=self.log_level
                )
                continue

            hotkey = outcome[0]
            result = outcome[1]

            if hotkey and isinstance(hotkey, str) and hotkey in hotkeys:

                index = hotkeys.index(hotkey)
                competition = competitions[index]
                model_output_path = os.path.join(self.base_model_output_path, hotkey)
                competition_components = competition.split("_")
                task, sample_rate = competition_components[0], competition_components[1].replace("HZ", "")
                tts_path = os.path.join(self.tts_path, sample_rate)

                if "denoising" in task.lower():
                    dataset_path = os.path.join(self.noise_path, sample_rate)
                else:
                    dataset_path = os.path.join(self.reverb_path, sample_rate)

                cache_entry = self.get_entry_from_cache(hotkey=hotkey)

                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Cache entry for model: {cache_entry}",
                    log_level=self.log_level,
                )

                if cache_entry and isinstance(cache_entry, dict):

                    if result: 

                        Utils.subnet_logger(
                            severity="TRACE",
                            message=f"Calculating model benchmarks for sample rate: {sample_rate} and task: {task}",
                            log_level=self.log_level
                        )
                        Utils.subnet_logger(
                            severity="TRACE",
                            message=f"Clean path: {tts_path}. Files in clean path: {os.listdir(tts_path)}",
                            log_level=self.log_level
                        )
                        Utils.subnet_logger(
                            severity="TRACE",
                            message=f"Noisy path: {dataset_path}. Files in noisy path: {os.listdir(dataset_path)}",
                            log_level=self.log_level
                        )
                        Utils.subnet_logger(
                            severity="TRACE",
                            message=f"Model output path: {model_output_path}. Files in path: {os.listdir(model_output_path)}",
                            log_level=self.log_level
                        )

                        metrics_dict = Benchmarking.calculate_metrics_dict(
                            clean_directory=tts_path,
                            enhanced_directory=model_output_path,
                            noisy_directory=dataset_path,
                            sample_rate=int(sample_rate),
                            task=task,
                            log_level=self.log_level,
                        )
                    else: 
                        metrics_dict = {}

                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Metrics dict for hotkey: {hotkey}: {metrics_dict}",
                        log_level=self.log_level,
                    )

                    model_metadata = cache_entry["response_data"]

                    model_benchmark = {
                        'hotkey':hotkey,
                        'hf_model_name':model_metadata['hf_model_name'],
                        'hf_model_namespace':model_metadata['hf_model_namespace'],
                        'hf_model_revision':model_metadata['hf_model_revision'],
                        'model_hash':cache_entry["hash"],
                        'block':cache_entry["block"],
                        'metrics':metrics_dict,
                    }

                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Model benchmark for hotkey: {hotkey}: {model_benchmark}",
                        log_level=self.log_level,
                    )

                    output_benchmarks.append(model_benchmark)
                    output_competitions.append(competition)

                # Reset model output and model repo directories following evaluation
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Resetting directory: {model_output_path}",
                    log_level=self.log_level
                )
                self._reset_dir(directory=model_output_path)
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Directory reset: {model_output_path}",
                    log_level=self.log_level
                )

                model_dir = os.path.join(self.model_base_path, hotkey)
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Resetting directory: {model_dir}",
                    log_level=self.log_level
                )
                self._reset_dir(model_dir)
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Directory reset: {model_dir}",
                    log_level=self.log_level
                )

        return output_benchmarks, output_competitions

    def get_entry_from_cache(self, hotkey: str):

        for comp_list in self.eval_cache.values():

            for entry in comp_list:

                if self.hotkeys[entry["uid"]] == hotkey:

                    return entry
                
        return None