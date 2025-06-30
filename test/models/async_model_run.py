import os
import sys
import time
import glob
import subprocess
from huggingface_hub import snapshot_download
import asyncio
import soundsright.base.utils as Utils
from soundsright.base.data import create_noise_and_reverb_data_for_all_sampling_rates, dataset_download, TTSHandler

class AsyncModelRunTester:

    def __init__(self):

        self.seed = 69

        self.base_path = os.path.join(os.path.expanduser("~"), ".SoundsRight")
        self.tts_base_path = os.path.join(self.base_path,'test_data','tts')
        self.noise_base_path = os.path.join(self.base_path,'test_data','noise')
        self.reverb_base_path = os.path.join(self.base_path,'test_data','reverb')
        self.arni_path = os.path.join(self.base_path,'test_data','arni')
        self.wham_path = os.path.join(self.base_path,'test_data','wham')
        self.enhanced1 = os.path.join(self.base_path, 'test_data','enhanced1')
        self.enhanced2 = os.path.join(self.base_path, 'test_data','enhanced2')
        self.enhanced3 = os.path.join(self.base_path, 'test_data','enhanced3')
        self.enhanced4 = os.path.join(self.base_path, 'test_data','enhanced4')
        self.enhanced5 = os.path.join(self.base_path, 'test_data','enhanced5')
        self.enhanced6 = os.path.join(self.base_path, 'test_data','enhanced6')
        self.enhanced7 = os.path.join(self.base_path, 'test_data','enhanced7')
        self.enhanced8 = os.path.join(self.base_path, 'test_data','enhanced8')
        self.enhanced9 = os.path.join(self.base_path, 'test_data','enhanced9')
        self.enhanced10 = os.path.join(self.base_path, 'test_data','enhanced10')
        self.model_path_1 = os.path.join(self.base_path,'models','model1')
        self.model_path_2 = os.path.join(self.base_path,'models','model2')
        self.model_path_3 = os.path.join(self.base_path,'models','model3')
        self.model_path_4 = os.path.join(self.base_path,'models','model4')
        self.model_path_5 = os.path.join(self.base_path,'models','model5')
        self.model_path_1 = os.path.join(self.base_path,'models','model6')
        self.model_path_2 = os.path.join(self.base_path,'models','model7')
        self.model_path_3 = os.path.join(self.base_path,'models','model8')
        self.model_path_4 = os.path.join(self.base_path,'models','model9')
        self.model_path_5 = os.path.join(self.base_path,'models','model10')
        self.output_path = os.path.join(self.base_path,'outputs')
        self.output_txt_path = os.path.join(self.output_path,'eval_results.txt')

        self.tags = [
            "model1",
            "model2",
            "model3",
            "model4",
            "model5",
        ]

        self.dereverb_tags = [
            "model6",
            "model7",
            "model8",
            "model9",
            "model10",
        ]

        self.paths = [
            self.model_path_1, 
            self.model_path_2, 
            self.model_path_3, 
            self.model_path_4, 
            self.model_path_5,
        ]

        self.dereverb_paths = [
            self.model_path_6, 
            self.model_path_7, 
            self.model_path_8, 
            self.model_path_9, 
            self.model_path_10
        ]

        self.output_paths = [
            self.enhanced1,
            self.enhanced2,
            self.enhanced3,
            self.enhanced4,
            self.enhanced5,
        ]

        self.output_paths_dereverb = [
            self.enhanced6,
            self.enhanced7,
            self.enhanced8,
            self.enhanced9,
            self.enhanced10,
        ]

        self.ports = [
            6501,
            6502,
            6503,
            6504,
            6505,
        ]

        self.dereverb_ports = [
            6506,
            6507,
            6508,
            6509,
            6510,
        ]

        for directory in [
            self.output_path, 
            self.tts_base_path, 
            self.noise_base_path, 
            self.reverb_base_path, 
            self.arni_path, 
            self.wham_path, 
            self.enhanced1, 
            self.enhanced2, 
            self.enhanced3, 
            self.enhanced4, 
            self.enhanced5, 
            self.enhanced6,
            self.enhanced7,
            self.enhanced8,
            self.enhanced9,
            self.enhanced10,
            self.model_path_1, 
            self.model_path_2, 
            self.model_path_3, 
            self.model_path_4, 
            self.model_path_5, 
            self.model_path_6, 
            self.model_path_7, 
            self.model_path_8, 
            self.model_path_9, 
            self.model_path_10
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        for path, port in zip(self.paths, self.ports):
            if not os.listdir(path):
                snapshot_download(repo_id="synapsecai/SoundsRightModelTemplate", local_dir=path, revision="DENOISING_16000HZ")
                Utils.replace_string_in_directory(directory=path, old_string="6500", new_string=str(port))

        for path, port in zip(self.dereverb_paths, self.dereverb_ports):
            if not os.listdir(path):
                snapshot_download(repo_id="synapsecai/SoundsRightModelTemplate", local_dir=path, revision="DEREVERBERATION_16000HZ")
                Utils.replace_string_in_directory(directory=path, old_string="6500", new_string=str(port))

        self.sample_rates = [16000]

        dataset_download(wham_path=self.wham_path, arni_path=self.arni_path, partial=True)

        self.TTSHandler = TTSHandler(
            tts_base_path=self.tts_base_path, 
            sample_rates=self.sample_rates
        )

        if not os.listdir(os.path.join(self.tts_base_path, "16000")):

            self.TTSHandler.create_openai_tts_dataset_for_all_sample_rates(n=30, seed=self.seed)

            create_noise_and_reverb_data_for_all_sampling_rates(
                tts_base_path = self.tts_base_path,
                arni_dir_path = self.arni_path,
                reverb_base_path=self.reverb_base_path,
                wham_dir_path=self.wham_path,
                noise_base_path=self.noise_base_path,
                tasks=['DENOISING', 'DEREVERBERATION'],
                log_level="TRACE",
                seed=100
            )

        if not self.run_async_build():
            print("model build failed")
            sys.exit()

    async def build_container_async(self, tag_name: str, directory: str) -> bool:
        """
        Build one miner model image async, return True if operation was successful and False otherwise
        """

        dockerfile_path = None

        # Search for docker-compose.yml in the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if file == "Dockerfile":
                    dockerfile_path = os.path.join(root, file)
                    break
            if dockerfile_path:
                break

        if not dockerfile_path:
            return False
        
        if not os.path.isfile(dockerfile_path):
            return False

        try:

            print(f"building container for tag_name: {tag_name} and directory: {directory}")
            
            process = await asyncio.create_subprocess_exec(
                "podman", "build",
                "-t", tag_name,
                "--file", dockerfile_path,
                "--no-cache",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(dockerfile_path),
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10000)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False

            if process.returncode != 0:
                print("build failed")
                return False

        except Exception as e:
            print(f"build failed because: {e}")
            return False
        
        print(f"build successful for tag_name: {tag_name} and directory: {directory}")
        return True

    async def build_containers_async(self):

        paths = self.paths + self.dereverb_paths
        tags = self.tags + self.dereverb_tags

        tasks = []

        for tag, path in zip(tags, paths):

            task = asyncio.create_task(self.build_container_async(tag_name=tag,directory=path))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)

        return outputs
    
    def run_async_build(self):

        outputs = asyncio.run(self.build_containers_async())

        return len(outputs) == 10
    
    def validate_all_noisy_files_are_enhanced(self, noise_path, enhanced_path):
        noisy_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(noise_path, "16000", '*.wav'))])
        enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(enhanced_path, '*.wav'))])
        return noisy_files == enhanced_files
    
    async def run_model_evaluation(self, tag, port, enhanced_path):

        start_time = time.time()

        print(f"starting model eval for tag: {tag}")

        start_start_time = time.time()

        status1 = await Utils.start_container_replacement_async(
            tag_name=tag,
            port=port,
            cuda_directory="/usr/local/cuda-12.6",
            log_level="TRACE"
        )

        model_start_time = time.time() - start_start_time

        if not status1:
            return False
        
        print(f"{tag} container started")

        status_start_time = time.time()
        
        status2 = await Utils.check_container_status_async(port=port, log_level="TRACE")

        status_check_time = time.time() - status_start_time

        if not status2:
            return False 
        
        print(f"{tag} status check successful")

        prepare_start_time = time.time()

        status3 = await Utils.prepare_async(port=port, log_level="TRACE")

        prepare_time = time.time() - prepare_start_time

        if not status3:
            return False
        
        print(f"{tag} preparation successful")

        upload_start_time = time.time()
        
        status4 = await Utils.upload_audio_async(noisy_dir=os.path.join(self.noise_base_path, "16000"), port=port, log_level="TRACE")

        upload_time = time.time() - upload_start_time

        if not status4:
            return False
        
        print(f"{tag} audio upload successful")

        enhance_start_time = time.time()
        
        status5 = await Utils.enhance_audio_async(port=port, log_level="TRACE")

        enhance_time = time.time() - enhance_start_time

        if not status5:
            return False
        
        print(f"{tag} audio enhancement successful")

        download_start_time = time.time()
        
        status6 = await Utils.download_enhanced_async(port=port, enhanced_dir=enhanced_path, log_level="TRACE")

        download_time = time.time() - download_start_time

        if not status6:
            return False 
        
        print(f"{tag} enhanced file download successful")
        
        status7 = self.validate_all_noisy_files_are_enhanced(noise_path=self.noise_base_path, enhanced_path=enhanced_path)

        if not status7:
            return False 
        
        print(f"{tag} noisy and enhanced file validation successful")
        
        completion_time = time.time() - start_time

        subprocess.run(
            ["podman", "rm", "-f", tag],
            check=True,
            capture_output=True
        )
        
        return completion_time, model_start_time, status_check_time, prepare_time, upload_time, enhance_time, download_time
    
    async def run_eval_group(self, count):

        tasks = []

        for i in range(count):
            
            task = asyncio.create_task(self.run_model_evaluation(
                tag=self.tags[i],
                port=self.ports[i],
                enhanced_path=self.output_paths[i]
            ))
            tasks.append(task)

        start_time = time.time()

        outputs = await asyncio.gather(*tasks)

        completion_time = time.time() - start_time

        return outputs, completion_time
    
    async def run_model_evaluation_dereverb(self, tag, port, enhanced_path):

        start_time = time.time()

        print(f"starting model eval for tag: {tag}")

        start_start_time = time.time()

        status1 = await Utils.start_container_replacement_async(
            tag_name=tag,
            port=port,
            cuda_directory="/usr/local/cuda-12.6",
            log_level="TRACE"
        )

        model_start_time = time.time() - start_start_time

        if not status1:
            return False
        
        print(f"{tag} container started")

        status_start_time = time.time()
        
        status2 = await Utils.check_container_status_async(port=port, log_level="TRACE")

        status_check_time = time.time() - status_start_time

        if not status2:
            return False 
        
        print(f"{tag} status check successful")

        prepare_start_time = time.time()

        status3 = await Utils.prepare_async(port=port, log_level="TRACE")

        prepare_time = time.time() - prepare_start_time

        if not status3:
            return False
        
        print(f"{tag} preparation successful")

        upload_start_time = time.time()
        
        status4 = await Utils.upload_audio_async(noisy_dir=os.path.join(self.reverb_base_path, "16000"), port=port, log_level="TRACE")

        upload_time = time.time() - upload_start_time

        if not status4:
            return False
        
        print(f"{tag} audio upload successful")

        enhance_start_time = time.time()
        
        status5 = await Utils.enhance_audio_async(port=port, log_level="TRACE")

        enhance_time = time.time() - enhance_start_time

        if not status5:
            return False
        
        print(f"{tag} audio enhancement successful")

        download_start_time = time.time()
        
        status6 = await Utils.download_enhanced_async(port=port, enhanced_dir=enhanced_path, log_level="TRACE")

        download_time = time.time() - download_start_time

        if not status6:
            return False 
        
        print(f"{tag} enhanced file download successful")
        
        status7 = self.validate_all_noisy_files_are_enhanced(noise_path=self.reverb_base_path, enhanced_path=enhanced_path)

        if not status7:
            return False 
        
        print(f"{tag} noisy and enhanced file validation successful")
        
        completion_time = time.time() - start_time

        subprocess.run(
            ["podman", "rm", "-f", tag],
            check=True,
            capture_output=True
        )
        
        return completion_time, model_start_time, status_check_time, prepare_time, upload_time, enhance_time, download_time
    
    async def run_eval_group_dereverb(self, count):

        tasks = []

        for i in range(count):
            
            task = asyncio.create_task(self.run_model_evaluation_dereverb(
                tag=self.dereverb_tags[i],
                port=self.dereverb_ports[i],
                enhanced_path=self.output_paths_dereverb[i]
            ))
            tasks.append(task)

        start_time = time.time()

        outputs = await asyncio.gather(*tasks)

        completion_time = time.time() - start_time

        return outputs, completion_time
    
    def save_lines_to_file(self, lines):
        
        with open(self.output_txt_path, 'w', encoding='utf-8') as f:
            for line in lines:
                print(line)
                f.write(f"{line}\n")

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
    
    def run_eval_test(self):

        lines = []

        for i in range(5):

            count = i + 1
            outputs, completion_time = asyncio.run(self.run_eval_group(count))

            completion_times = []
            model_start_times = []
            status_times = []
            prepare_times = []
            upload_times = []
            enhance_times = []
            download_times = []

            for k in outputs: 
                completion_times.append(k[0])
                model_start_times.append(k[1])
                status_times.append(k[2])
                prepare_times.append(k[3])
                upload_times.append(k[4])
                enhance_times.append(k[5])
                download_times.append(k[6])

            line1 = f"(DENOISING) Total completion time for {count} models: {completion_time}. Individual completion times: {completion_times}. Average completion time per model: {completion_time/len(completion_times)}"
            print(line1)
            lines.append(line1)

            line2 = f"Average start time: {sum(model_start_times) / len(model_start_times)}. Individual model start times: {model_start_times}"
            print(line2)
            lines.append(line2)

            line3 = f"Average status check time: {sum(status_times) / len(status_times)}. Individual status check times: {status_times}"
            print(line3)
            lines.append(line3)

            line4 = f"Average preparation time: {sum(prepare_times) / len(prepare_times)}. Individual preparation times: {prepare_times}"
            print(line4)
            lines.append(line4)

            line5 = f"Average audio upload time: {sum(upload_times) / len(upload_times)}. Individual audio upload times: {upload_times}"
            print(line5)
            lines.append(line5)

            line6 = f"Average enhancement time: {sum(enhance_times) / len(enhance_times)}. Individual enhancement times: {enhance_times}"
            print(line6)
            lines.append(line6)

            line7 = f"Average audio download time: {sum(download_times) / len(download_times)}. Individual audio download times: {download_times}"
            print(line7)
            lines.append(line7)

        for path in self.output_paths:
            self._reset_dir(directory=path)

        for i in range(5):

            count = i + 1
            outputs, completion_time = asyncio.run(self.run_eval_group_dereverb(count))

            completion_times = []
            model_start_times = []
            status_times = []
            prepare_times = []
            upload_times = []
            enhance_times = []
            download_times = []

            for k in outputs: 
                completion_times.append(k[0])
                model_start_times.append(k[1])
                status_times.append(k[2])
                prepare_times.append(k[3])
                upload_times.append(k[4])
                enhance_times.append(k[5])
                download_times.append(k[6])

            line1 = f"(DEREVERBERATION) Total completion time for {count} models: {completion_time}. Individual completion times: {completion_times}. Average completion time per model: {completion_time/len(completion_times)}"
            print(line1)
            lines.append(line1)

            line2 = f"Average start time: {sum(model_start_times) / len(model_start_times)}. Individual model start times: {model_start_times}"
            print(line2)
            lines.append(line2)

            line3 = f"Average status check time: {sum(status_times) / len(status_times)}. Individual status check times: {status_times}"
            print(line3)
            lines.append(line3)

            line4 = f"Average preparation time: {sum(prepare_times) / len(prepare_times)}. Individual preparation times: {prepare_times}"
            print(line4)
            lines.append(line4)

            line5 = f"Average audio upload time: {sum(upload_times) / len(upload_times)}. Individual audio upload times: {upload_times}"
            print(line5)
            lines.append(line5)

            line6 = f"Average enhancement time: {sum(enhance_times) / len(enhance_times)}. Individual enhancement times: {enhance_times}"
            print(line6)
            lines.append(line6)

            line7 = f"Average audio download time: {sum(download_times) / len(download_times)}. Individual audio download times: {download_times}"
            print(line7)
            lines.append(line7)

        self.save_lines_to_file(lines=lines)


if __name__ == "__main__":

    tester = AsyncModelRunTester()
    tester.run_eval_test()