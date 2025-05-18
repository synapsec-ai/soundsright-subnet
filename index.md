---
title: Home
nav_order: 1
layout: page
---
<h1 align="center">SoundsRight (SN 105)</h1>
<h2 align="center">| <a href="https://soundsright.ai">Website & Leaderboard</a> | <a href="https://docs.soundsright.ai">Docs</a> | <a href="https://bittensor.com/">Bittensor</a> |</h2>

<h1 align="center">Bittensor's Speech Enhancement Subnet</h1>

If you are unfamiliar with how Bittensor works, please check out [this primer](https://docs.bittensor.com/learn/bittensor-building-blocks) first!

The SoundsRight subnet incentivizes the research and development of speech enhancement models through daily fine-tuning competitions, powered by the decentralized Bittensor ecosystem. 

Miners in the subnet will upload their fine-tuned models to HuggingFace, and the subnet's validators are in charge of downloading the models, benchmarking their performance and rewarding miners accordingly. 

**Each competition is winner-takes-all.**

<h1 align="center">Fine-Tuning Competitions</h1>

The table below outlines the daily competitions currently being held by the subnet. Competitions are distinguished by the sample rate of the testing data, the task and the metric used for benchmarking.

| Sample Rate | Task | Benchmarking Metric | % of Total Miner Incentives | 
| ----------- | ---- | ------ | --------------------------- |
| 16 kHz | Denoising | PESQ | 15 |
| 16 kHz | Denoising | ESTOI | 12.5 |
| 16 kHz | Denoising | SI-SDR | 7.5 |
| 16 kHz | Denoising | SI-SAR | 7.5 |
| 16 kHz | Denoising | SI-SIR | 7.5 |
| 16 kHz | Dereverberation | PESQ | 15 |
| 16 kHz | Dereverberation | ESTOI | 12.5 |
| 16 kHz | Dereverberation | SI-SDR | 7.5 |
| 16 kHz | Dereverberation | SI-SAR | 7.5 |
| 16 kHz | Dereverberation | SI-SIR | 7.5 |

For more details about sample rates, tasks and metrics, please reference the [competition docs](docs/subnet/competitions.md).

<h1 align="center">Getting Started</h1>

To get started with mining or validating in the subnet, please reference the following documentation:

Additionally, if you would like to start in testnet, SoundsRight is testnet 271.

<h3 align="center">Subnet Documentation</h3>
 
<h4 align="center">| <a href="docs/subnet/subnet_architecture.md">Subnet Architecture</a> | <a href="docs/subnet/competitions.md">Competitions</a> | <a href="docs/subnet/roadmap.md">Roadmap</a> | <a href="docs/subnet/citations.md">Citations</a> |</h4>
 
<h3 align="center">Miner Documentation</h3>
 
<h4 align="center">| <a href="docs/mining/running_miner.md">Running a Miner</a> | <a href="docs/mining/generate_data.md">Generating a Fine-Tuning Dataset</a> | <a href="docs/mining/model_formatting.md">Model Formatting</a> |</h4>
 
<h3 align="center">Validator Documentation</h3>
 
<h4 align="center">| <a href="docs/validating/running_validator.md">Running a Validator</a> | <a href="docs/validating/validator_architecture.md">Validator Architecture</a> |</h4>