---
title: Upcoming Changes with 48 kHz Competitions
parent: Mining
nav_order: 5
layout: page
---
# Upcoming Changes with 48 kHz Competitions

The subnet team has been working to prepare the new update to 48 kHz competitions with the upcoming subnet `v2.0.0`! This is a huge leap forward, and with the update will come a few changes for both miners and validators. While the update is still being tested, we wanted to provide a list of what to expect so everyone involved in the subnet can prepare accordingly. 

## Changes for Miners

Mining on the subnet will change in the following ways:

### Dataset Generation

[ElevenLabs](https://elevenlabs.io/) will now be used instead of OpenAI for the dataset generation. What plan you use is up to you, but we recommend getting at least a [Pro Plan](https://elevenlabs.io/pricing) as this will allow you to get PCM audio output with the API. Without this, there will be some loss when converting the MP3 output to WAV due to MP3's inherent compression.

If you wish to generate datasets ahead of the update, the `generate_dataset.py` script in the `release/2.0.0` branch of the subnet repository has already been updated. Add your ElevenLabs API key to the .env file and you should be good to go. 

Keep in mind that the following input arguments should be adjusted, on top of the ones already specified in the [generate dataset](generate_dataset.html) portion of the docs:

| Argument | Description | Always Required |
| :------: | :---------: | :--: | :------: |
| --sample_rate | Sample rate, either 16000 or 48000. Default is 48000. | No |
| --output_format | If your ElevenLabs subscription is below that of a Pro plan, you will need to adjust this to output MP3 (this will automatically get converted to .wav). Your input will be one of: mp3_44100_32, mp3_44100_64, mp3_44100_96, mp3_44100_128 or mp3_44100_192. The last portion of the input signifies the bitrate of the output audio. If you have a Pro Plan or above, ignore this section. The default is: pcm_44100 |  Yes |

### Model Formatting

The models submitted to validators will now need one more API endpoint -- `/reset/`. This endpoint will remove the noisy and enhanced files currently cached in the model container in preparation for a new batch of enhancement. Here are some drop-ins for adjusted/added functions in the `app/app.py` file in the [model template](https://huggingface.co/synapsecai/SoundsRightModelTemplate). The template itself has also been updated if you wish to clone the repository directly and work from there.

The endpoint needs to be added to ModelAPI._setup_routes:

```py
    def _setup_routes(self):
        """
        Setup API routes:
        
        /status/ : Communicates API status
        /upload-audio/ : Upload audio files, save to noisy audio directory
        /enhance/ : Enhance audio files, save to enhanced audio directory
        /download-enhanced/ : Download enhanced audio files
        /reset/ : Reset noisy and enhanced file cache
        """
        self.app.get("/status/")(self.get_status)
        self.app.post("/prepare/")(self.prepare)
        self.app.post("/upload-audio/")(self.upload_audio)
        self.app.post("/enhance/")(self.enhance_audio)
        self.app.get("/download-enhanced/")(self.download_enhanced)
        self.app.post("/reset/")(self.reset)
```

And a new function, ModelAPI.reset, must be added:

```py
    def reset(self):
        """
        Removes all audio files in preparation for another batch of enhancement.
        """
        for directory in [self.noisy_audio_path, self.enhanced_audio_path]:
            if not os.path.isdir(directory):
                continue

            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"Error removing {filepath}: {e}")
                        return {"status": False, "noisy": os.listdir(self.noisy_audio_path), "enhanced": os.listdir(self.enhanced_audio_path)}
        return {"status": True, "noisy": os.listdir(self.noisy_audio_path), "enhanced": os.listdir(self.enhanced_audio_path)}
```

Models will not be benchmarked if this endpoint is not provided, so please ensure that all of your models are up to date.

### Updated Scoring Mechanism

The metrics used for benchmarking will now depend on the task/sample rate:

**Denoising, 16kHz**: PESQ, ESTOI
**Dereverberation, 16kHz**: PESQ, ESTOI
**Denoising, 48kHz**: SI-SIR, SI-SAR, SI-SDR, ESTOI
**Dereverberation, 48kHz**: SI-SDR, ESTOI

This will result in a grand total of 10 competitions--the same as before, but spread out differently.

The weights assigned to the scores of different competition winners will also change:

| Sample Rate | Task | Benchmarking Metric | % of Total Score | 
| ----------- | ---- | ------ | --------------------------- |
| 16 kHz | Denoising | PESQ |  |
| 16 kHz | Denoising | ESTOI |  |
| 16 kHz | Dereverberation | PESQ |  |
| 16 kHz | Dereverberation | ESTOI |  |
| 48 kHz | Denoising | ESTOI |  |
| 48 kHz | Denoising | SI-SDR |  |
| 48 kHz | Denoising | SI-SAR |  |
| 48 kHz | Denoising | SI-SIR |  |
| 48 kHz | Dereverberation | ESTOI |  |
| 48 kHz | Dereverberation | SI-SDR |  |

## Changes for Validators

The specs required for validating on the subnet will remain the same, however a [Pro Plan with ElevenLabs](https://elevenlabs.io/pricing) will be required to generate the new datasets.

In addition to this, the OpenAI subscription will no longer be required. 

When updating to `v2.0.0`, please copy over the new `.env` file and fill in the ElevenLabs API key accordingly.