# Refracted Skew Brownian Motion (RSBM)

Code accompanying the paper:

> **A note on Refracted Skew Brownian Motion with an application**  
> Zaniar Ahmadi and Xiaowen Zhou  
> Concordia University  


This repository contains a Python implementation for working with **Refracted Skew Brownian Motion (RSBM)** and **truncated normal distributions**, including simulation, sampling, and parameter fitting utilities.

The code is organized as a small reusable package with a simple entry-point script for experiments and plots.

---

## 1. Repository structure

```text
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ sbm_truncnorm/
│  ├─ sbm.py          # Skew Brownian Motion definitions and simulation
│  ├─ truncnorm.py    # Truncated normal distribution utilities
│  ├─ sampling.py     # Sampling routines
│  ├─ fit.py          # Parameter estimation / fitting methods
│  ├─ utils.py        # Helper and utility functions
│
└─ main.py            # Entry point for running experiments / generating plots
```

### Modules Overview

sbm.py
: Core implementation of Refracted Skew Brownian Motion, including PDF and CDF.

truncnorm.py
: Functions related to truncated normal distributions.

sampling.py
: Sampling RSBM.

fit.py
: Parameter estimation and fitting procedures.

utils.py
: Helper functions, including objective functions and plotting.

main.py
: Example script to run simulations, fit models, and generate plots.

---

## Installation

Follow the steps below to set up the environment and run the project.

### 1. Clone the repository

```bash
git clone https://github.com/zaniara3/RefractedSkewBrownianMotion.git
cd 
```

### 2. Create and activate a virtual environment (recommended)

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\activate
```
### 3. Install the required Python packages

```bash
pip install -r requirements.txt
```


