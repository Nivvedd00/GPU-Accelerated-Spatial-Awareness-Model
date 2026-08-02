# Local Setup for Video Depth Estimation Notebook

This directory contains the exact Jupyter Notebook project from your friend's Google Colab, optimized to run locally on your Windows environment.

## Folder Structure
```
dws_depth_detection/
├── Video_Depth_estimation.ipynb   # The modified Jupyter Notebook
├── requirements.txt               # Dependencies required to run the cells
└── README.md                      # Setup instructions (this file)
```

---

## Quick Start Guide

### Step 1: Install Dependencies
Open your PowerShell or Command Prompt, navigate to this project directory, and run the following command to install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Open the Notebook locally
You can run this notebook using one of two methods:

#### Method A: Using VS Code (Recommended)
1. Open **VS Code**.
2. Install the **Python** and **Jupyter** extensions from the Extensions tab (`Ctrl+Shift+X`).
3. Open this folder: `C:\Users\ROG\.gemini\antigravity\scratch\dws_depth_detection`.
4. Open the `Video_Depth_estimation.ipynb` file.
5. In the top-right corner, select your Python environment as the Jupyter kernel, and you can run the cells directly!

#### Method B: Using classic Jupyter Notebook Web Browser interface
If you do not use VS Code:
1. In your command line, run:
   ```bash
   jupyter notebook
   ```
2. A browser window will open automatically. Click on `Video_Depth_estimation.ipynb` to open and run it.

---

## Crucial Windows Modifications Made
To make the Colab code work locally, the following changes were applied:
1. **Paths**: All temporary Colab paths like `/content/dwsonder` and `/content/stair2.mp4` have been replaced with relative local paths (`./dwsonder` and `./stair2.mp4`).
2. **PyTorch Hub Security Bypass**: Added `trust_repo=True` to the `torch.hub.load()` calls. Without this, PyTorch would freeze the cells to wait for console keyboard inputs to trust the Intel-MiDaS repository.
3. **Local Kaggle Setup Cell**: A helper cell has been added at the very top of the notebook. If you need to download and extract the Kaggle `dwsonder` dataset locally, simply uncomment the code in the first cell, fill in your Kaggle Username/Key, and run it!
