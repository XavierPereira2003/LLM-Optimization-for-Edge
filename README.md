

# LLM Optimisation for Edge

This repository provides tools, scripts, and models designed to optimise large language models (LLMs) for deployment on edge devices. The focus is on reducing storage requirements, enhancing inference efficiency, and facilitating fine-tuning of LLMs in resource-constrained environments such as edge and embedded systems.

## Contents

- **LoRa**: Contains Low-Rank Adaptation (LoRa) implementations for the `BoolQ` and `COPA` tasks from the SuperGLUE dataset, specifically adapted for the `flan-t5-small` model.
- **Expo**: Contains project files for a demonstration showcased at ProjectExpo, including example scripts for running LLM inference on edge devices.

## Installation

To set up and run the code from the Expo project, follow these steps:

1. Create the environment and install dependencies:
   ```bash
   conda env create --file environment.yml
   conda activate llm_optimisation
   cd Expo
   ```

2. Open separate terminals to run each component of the Expo demonstration:

   - In Terminal 1, start the client:
     ```bash
     python client.py
     ```

   - In Terminal 2, run the BoolQ service:
     ```bash
     python boolq.py
     ```

   - In Terminal 3, start the COPA service:
     ```bash
     python copa_service.py
     ```