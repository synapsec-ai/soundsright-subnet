---
title: Subnet Architecture
parent: Subnet
nav_order: 2
layout: page
---
# Subnet Architecture

There are two main entities in the subnet:

1. **Miners** upload fine-tuned speech enhancement models to HuggingFace.
2. **Validators** benchmark models and determine the miners whose models perform the best.

Here is a diagram of the overarching process:

```mermaid
sequenceDiagram
    participant Miner
    participant HuggingFace
    participant Bittensor Chain
    participant Validator
    participant Subnet Website

    Miner->>Miner: Fine-tunes a speech enhancement model
    Miner->>HuggingFace: Uploads model
    Miner->>Bittensor Chain: Writes model metadata
    Validator->>Miner: Sends Synapse requesting model information
    Miner->>Validator: Returns Synapse containing model information
    Validator->>Bittensor Chain: References model metadata
    Bittensor Chain-->>Validator: Confirms model ownership
    HuggingFace->>Validator: Downloads model
    Validator->>Validator: Benchmarks model on locally generated dataset
    Validator->>Subnet Website: Reports benchmarking results
    Subnet Website->>Subnet Website: Constructs competition leaderboards
    Validator->>Bittensor Chain: Sets weights for miners
```