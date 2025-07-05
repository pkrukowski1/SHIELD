# 🛡️ SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense

**Authors**: Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek  
🎓 *GMUM — Jagiellonian University*

📄 **[Read the full paper on arXiv](https://arxiv.org/abs/2506.08255)**  

---

## 🧭 Table of Contents

- [Abstract](#abstract)
- [Method Overview](#method-overview)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Setup](#setup)
  - [Launching Experiments](#launching-experiments)
  - [Running Sweeps](#running-sweeps)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## 🧠 Abstract

Continual learning under adversarial conditions remains a major challenge, as most existing methods either lack robustness guarantees or fail to scale to complex settings. In this work, we propose a novel framework that combines **Interval Bound Propagation (IBP)** with a **hypernetwork-based continual learning architecture**, enabling **certified robustness against adversarial attacks** across sequential tasks.

Our approach generates task-specific model parameters via a shared hypernetwork, conditioned only on task embeddings—a design that avoids storing previous task data or full model copies and remains parameter-efficient over time.

To further boost certified performance, we introduce **Interval MixUp**, a new training strategy that blends virtual example interpolation with provable guarantees using interval arithmetic. This allows us to maintain robustness even on interpolated synthetic inputs.

We evaluate SHIELD on a diverse set of benchmarks:
- Permuted MNIST
- Rotated MNIST
- Split CIFAR-100
- Split miniImageNet
- TinyImageNet

SHIELD achieves **state-of-the-art or highly competitive performance**, sometimes **doubling the robust accuracy** of prior methods under strong attacks such as **AutoAttack**. These results underscore the scalability and effectiveness of our approach for real-world, adversarially robust continual learning.

---

## 🧪 Method Overview

SHIELD integrates three key components:

1. **Hypernetwork-based continual learning**  
   Learns task-specific weights through a shared hypernetwork without storing previous data or models.

2. **Interval Bound Propagation (IBP)**  
   Enables certified robustness by computing bounds on adversarial perturbations using interval arithmetic.

3. **Interval MixUp (IM)**  
   A novel strategy that mixes intervals of different samples to generate provably robust synthetic data, improving generalization and certification.

The architecture and training loop are designed to be **modular, scalable**, and compatible with **PyTorch Lightning + Hydra** for flexible configuration and reproducibility.

---

## 📊 Results

### 📈 Datasets

- **Permuted MNIST** – 10 tasks x 10 classes
- **Rotated MNIST** – 10 tasks x 10 classes
- **Split CIFAR-100** – 10 tasks × 10 classes  
- **Split miniImageNet** – 10 tasks × 10 classes  
- **TinyImageNet** – 40 tasks × 5 classes

### 🔐 Robustness Evaluation

- Evaluated under **AutoAttack**, **PGD**, **FGSM**, and on clean samples.
- SHIELD with **Interval MixUp** improves certified accuracy while maintaining low forgetting.
- Outperforms all baselines in average robust accuracy and backward transfer (BWT) across all benchmarks.

---

## ⚙️ Getting Started

### 🛠️ Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/gmum/SHIELD.git
   cd SHIELD

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate shield
   ```

3. **Install remaining Python packages**
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables
   ```bash
   cp example.env .env
   # Edit `.env` to configure WANDB, dataset paths, etc.
   ```

### 🚀 Launching Experiments

To launch a default experiment with Hydra:
   ```bash
   WANDB_MODE=offline HYDRA_FULL_ERROR=1 python src/main.py --config-name=config
   ```
   > 💡 Use WANDB_MODE=online to enable live logging to Weights & Biases

You can also launch predefined experiments for specific datasets:
   ```bash
   # For Permuted MNIST
   ./scripts/permuted_mnist/train.sh

   # For Rotated MNIST
   ./scripts/rotated_mnist/train.sh

   # For Split CIFAR-100
   ./scripts/split_cifar_100/train.sh

   # For Split miniImageNet
   ./scripts/split_mini_imagenet/train.sh

   # For TinyImageNet
   ./scripts/tiny_imagenet/train.sh
   ```

---

## 🙏 Acknowledgements

- Project structure adapted from [Bartłomiej Sobieski’s template](https://github.com/sobieskibj/templates/tree/master)
- Inspired by and built upon [HyperMask](https://github.com/gmum/HyperMask)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for more information.

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{krukowski2025shieldsecurehypernetworksincremental,
      title={SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense}, 
      author={Patryk Krukowski and Łukasz Gorczyca and Piotr Helm and Kamil Książek and Przemysław Spurek},
      year={2025},
      eprint={2506.08255},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.08255}, 
}
```

---

## ✉️ Contact
Questions, suggestions or issues?  
Open an issue or contact the authors directly via [GMUM](https://gmum.net/).