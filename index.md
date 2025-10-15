---
title: Home
nav_order: 1
layout: page
---
<h1 align="center">SoundsRight (SN 105)</h1>
<h2 align="center">| <a href="https://soundsright.ai">Website & Leaderboard</a> | <a href="https://bittensor.com/">Bittensor</a> |</h2>

<h1 align="center">Bittensor's Speech Enhancement Subnet</h1>

If you are unfamiliar with how Bittensor works, please check out [this primer](https://docs.bittensor.com/learn/bittensor-building-blocks) first!

The SoundsRight subnet promotes the research and development of speech enhancement models through two-day fine-tuning competitions, powered by the decentralized Bittensor ecosystem. 

Miners in the subnet will upload their fine-tuned models to HuggingFace, and the subnet's validators are in charge of downloading the models, benchmarking their performance and determining the best model. 

**Each competition is winner-takes-most.**

<h1 align="center">Fine-Tuning Competitions</h1>

The table below outlines the two-day competitions currently being held by the subnet. Competitions are distinguished by the sample rate of the testing data, the task and the metric used for benchmarking.

| Sample Rate | Task | Benchmarking Metric | % of Total Score | 
| ----------- | ---- | ------------------- | ---------------- |
| 16 kHz | Denoising | PESQ | 7.8 |
| 16 kHz | Denoising | ESTOI | 5.2 |
| 16 kHz | Dereverberation | PESQ | 7.2 |
| 16 kHz | Dereverberation | ESTOI | 4.8 |
| 48 kHz | Denoising | ESTOI | 20 |
| 48 kHz | Denoising | SI-SDR | 10 |
| 48 kHz | Denoising | SI-SAR | 10 |
| 48 kHz | Denoising | SI-SIR | 10 |
| 48 kHz | Dereverberation | ESTOI | 15 |
| 48 kHz | Dereverberation | SI-SDR | 10 |

For more details about sample rates, tasks and metrics, please reference the [competition docs](subnet/competitions.html).

<h1 align="center">Getting Started</h1>

To get started with mining or validating in the subnet, please reference the following documentation:

Additionally, if you would like to start in testnet, SoundsRight is testnet 271.