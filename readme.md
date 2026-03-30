# Disentangling Hardness from Noise: An Uncertainty-Driven Model-Agnostic Framework for Long-Tailed Remote Sensing Classification (ICME 2026)

This is the official implementation of the paper: **"Disentangling Hardness from Noise: An Uncertainty-Driven Model-Agnostic Framework for Long-Tailed Remote Sensing Classification"**.

## 📌 Introduction
Remote sensing images often suffer from **long-tailed distributions** and **heterogeneous data quality** (e.g., cloud occlusion, sensor noise). This project proposes an uncertainty-aware framework based on **Evidential Deep Learning (EDL)**.
* **Uncertainty Disentanglement**: Decouples Epistemic Uncertainty (**EU**) and Aleatoric Uncertainty (**AU**).
* **EU-based Reweighting**: Dynamically emphasizes under-learned tail samples.
* **AU-based Label Smoothing**: Mitigates the impact of noisy or ambiguous data.

## 🛠️ Environment Setup
The experiments were conducted using **PyTorch 2.1** and **CUDA 12.2**.
```bash
# Create environment
conda create -n dual_rs python=3.9
conda activate dual_rs

# Install dependencies
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 📊 Data Preparation
We evaluate our framework on three representative benchmarks:
* **DOTA**: 15 categories, 98,906 training samples.
* **DIOR**: 20 categories, 68,025 training samples.
* **FGSC-23**: 23 categories (fine-grained ship classification).

Please organize your data as follows:
```text
data/
├── DOTA/
├── DIOR/
└── FGSC-23/
```

## 🚀 Training and Evaluation
The default backbone is **ResNet-50**. Hyperparameters are set to $\sigma=3$ and $\lambda=0.2$ (Default).

### Training
```bash
python train.py --dataset FGSC-23 --batch_size 64 --lr 1e-3
```

### Evaluation
```bash
python eval.py --dataset FGSC-23 --checkpoint ./weights/best_model.pth
```

## 📈 Main Results
Our method achieves state-of-the-art performance:
| Dataset | Top-1 Acc (%) | Head Acc (%) | Tail Acc (%) |
| :--- | :---: | :---: | :---: |
| **DOTA** | 96.66% | 97.13% | 89.18% |
| **DIOR** | 91.07% | 90.54% | 87.47% |
| **FGSC-23** | 79.98% | 78.21% | 82.72% |


## 📝 Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{anonymous2026uncertainty,
  title={Uncertainty-aware Long-tailed Remote Sensing Image Classification},
  author={Anonymous},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}
```

