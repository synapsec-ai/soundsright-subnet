---
title: Validator Architecture Overview
parent: Validating
nav_order: 2
---
# Validator Architecture Overview

Validators on the subnet are in charge of benchmarking miner models and assigning weights for miners who submit the top performing models for each competition. Competitions span one day, and below is a diagram illustrating what the validator does during each:

```mermaid
sequenceDiagram
    participant Miner
    participant HuggingFace
    participant Bittensor Chain
    participant Validator
    participant Subnet Website

    Validator->>Validator: Generate new benchmarking dataset
    Validator->>Validator: Benchmark SGMSE+ on new dataset
    Validator->>Subnet Website: Report SGMSE+ benchmark results
    Validator->>Miner: Send Synapse requesting model information
    Miner->>Validator: Return Synapse containing model information
    Bittensor Chain->>Validator: Obtain model metadata<br>submitted by miner
    HuggingFace->>Validator: Download model
    Validator->>Validator: Obtain hash of model directory
    Validator->>Validator: Confirm model ownership by miner<br>using chain metadata and model hash
    Validator->>Validator: Confirms model container is safe to run
    Validator->>Validator: Runs model container<br>and benchmarks model
    Validator->>Validator: Iterates through all miners<br>and assigns scores per competition
    Validator->>Subnet Website: Submits miner model<br>benchmarking results
    Validator->>Bittensor Chain: Sets weights for miners
    Bittensor Chain->>Miner: Assigns incentives 
    Subnet Website->>Subnet Website: Generates leaderboard and<br>results of miner benchmarks<br>against standard (SGMSE+)
```