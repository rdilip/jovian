# Jovian: Bidirectional Autoregressive Protein Backbone Generation

**Jovian** is a bidirectional autoregressive model for **protein backbone generation** that  
1. Learns both **sequence length** and **motif positioning** by conditioning on prediction direction.  
2. Enables **simple, zero-shot motif scaffolding** without motif-specific training.  
3. Achieves **10–40× faster** inference than comparable models, while matching or exceeding diffusion-based methods on standard benchmarks.

<p align="center">
<img src="assets/jovian_overview.png" alt="Jovian Overview" width="600">
</p>

---

## ✨ Key Features
- **Variable-length generation** — no fixed size needed at inference time.
- **Zero-shot conditional design** — perform motif scaffolding without retraining.
- **Speed** — 10–40× faster than diffusion models in unconditional design.
- **Long-sequence stability** — maintains robustness for large proteins.
- **Competitive accuracy** — matches or surpasses diffusion-based methods on standard benchmarks.

---

## 📜 Paper

> **Jovian: Bidirectional Autoregressive Models Enable Zero-Shot Motif Scaffolding in Protein Design**  
> [Paper Link (NeurIPS 2025)](https://arxiv.org/abs/XXXX.XXXXX) – Coming soon  
> Rohit K. Dilip, et al.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/jovian.git
cd jovian
conda create -n jovian python=3.10
conda activate jovian
pip install -r requirements.txt
