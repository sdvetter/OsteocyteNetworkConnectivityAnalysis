# Osteocyte Network Connectivity Analysis

This repository contains code for training deep learning models and performing
connectivity analysis of osteocyte networks using MONAI and PyTorch.
A pre-trained Attention-UNet model (`.pth`) is provided in the repo.


## Connectivity Analysis

In the notebook **connectivity_analysis**, you will find an implementation and example of how to run a connectivity analysis on your own dataset using the pre-trained model provided.

## Training

The Python scripts `attentionunet` and `swinunet` show examples of using your own training and validation data to train your own models.

---

## Environment Setup

This code was originally developed using a MONAI + PyTorch stack from 2022–2023.
The environment below reproduces the original setup and allows running the code without modification. Make sure to use Python 3.9.x.

### Python venv + pip

Below is a single set of commands that creates and activates a virtual environment, installs all dependencies, and adds it to Jupyter if needed.

```bash
# Check Python version
python --version
# or
py -3.9 --version

# Create a virtual environment
py -3.9 -m venv ONCA
# or if python points to 3.9
python -m venv ONCA

# Activate the virtual environment
# Windows
ONCA\Scripts\activate
# macOS / Linux
source ONCA/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install Jupyter and add venv as a kernel
pip install notebook ipykernel
python -m ipykernel install --user --name=ONCA --display-name "Python 3.9 (ONCA)"

# Start Jupyter Notebook
jupyter notebook
```

### Notes

* The `requirements.txt` contains pinned package versions for reproducibility.
* Using `venv` ensures the environment is isolated and fully reproducible.
* GPU support with PyTorch is automatically enabled if a CUDA-compatible device is available.
* All notebooks must use the **Python 3.9 (ONCA)** kernel to run correctly.
