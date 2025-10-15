---
title: Fine-Tuning Competitions
parent: Subnet
nav_order: 1
layout: page
---
# SoundsRight Competitions

Each individual competition in the subnet is denoted by a unique **sample rate**, **task** and **benchmarking metric**. This doc serves to explain what each of these components are.

## Sample Rate

What we percieve as sound is fundamentally a wave propogating through the air. To represent this digitally we take samples of the signal and mesh them together, much like how a video is comprised of individual frames. Continuing this analogy, the sample rate of digital audio is akin to the frame rate of a video.

Where sample rates differ from frame rates are in how many samples are taken per second--so much so that sample rates are often represented in kHz. Another unique property of the sample rate is that the higher the sample rate, the higher the sound frequencies that can be digitally represented.

The table below denotes a few commonly used sample rates and their applications.

| Sample Rate | Details | Common Applications |
| ----------- | ------- | ------------------- |
| 8 kHz | The minimum sample rate for intelligible human speech, often used in applications with limited bandwith. Also known as narrowband audio. | Telephone calls, intercom systems, VoIP |
| 16 kHz | A good sample rate for capturing human speech while maintaining smaller file sizes. Also known as wideband audio. | Speech recognition and transcription, VoIP |
| 44.1 kHz | A sample rate that covers the entire range of human hearing. | CD's, Spotify, audiobooks |
| 48 kHz | A sample rate that covers the entire range of human hearing and also divides evenly with video frame rates to make syncing easier. | Films, television, high-quality digital media (live-streaming, Youtube, etc.) |

Currently, the subnet hosts competitions for 16kHz and 48kHz sample rates.

## Tasks

The subnet currently hosts competitions for two tasks--**denoising** and **dereverberation**. 

### Denoising

The task of denoising involves isolating speech from any background noise present in the recording. The subnet uses the [WHAM! noise dataset](http://wham.whisper.ai/) to add noise to clean text-to-speech outputs in evaluation datasets.

### Dereverberation

The task of dereverberation involves removing any reverberation from speech (an echo from a large room, etc.). The subnet convolves text-to-speech outputs with room impulse responses from the [Arni dataset](https://zenodo.org/records/6985104) to generate reverberant speech.

## Evaluation Metrics

There are a multitude of metrics to assess the quality of audio. Below are the metrics used in the subnet's competitions:

### PESQ (Perceptual Evaluation of Speech Quality)

This metric's aim is to quanitify a person's percieved quality of speech, and is useful as a holistic determination of the quality of speech enhancement performed.

It is important to note that PESQ only works for 8kHz and 16kHz audio.

### ESTOI (Extended Short-Time Objective Intelligibility)

This metric's aim is to quantify the intelligibility of speech--how easy it is the understand the speech itself.

### SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

This metric determines how much distortion is present in the audio. Distortion can be thought of as unwanted changes to the speech signal as a result of the enhancement operation.

### SI-SAR (Scale-Invariant Signal-to-Artifacts Ratio)

This metric determines the level of artifacts present in the audio. Artifacts can be thought of as new, unwanted components introduced as a result of the speech enhancement operation.

### SI-SIR (Scale-Invariant Signal-to-Interference Ratio)

This metric determines the level of interference present in the audio. Interference can be thought of as unwanted audio from outside sources still present in the recording, such as the noise from a crowded room.