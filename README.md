
# 🛡️ SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense

**Authors**: Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek  
🎓 *GMUM — Jagiellonian University*

📄 **[Read the full paper on arXiv](https://www.arxiv.org/abs/2506.08255)**  

---

## 🧭 Table of Contents

- [Abstract](#abstract)
- [Getting Started](#getting-started)
  - [Setup](#setup)
  - [Launching Experiments](#launching-experiments)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

---

## 🧠 Abstract

Continual learning under adversarial conditions remains a major challenge, as most existing methods either lack robustness guarantees or fail to scale to complex settings. In this work, we propose a novel framework that combines Interval Bound Propagation (IBP) with a hypernetwork-based continual learning architecture, enabling certified robustness against adversarial attacks across sequential tasks. Our approach generates task-specific model parameters via a shared hypernetwork, conditioned only on task embeddings—a design that avoids storing previous task data or full model copies and remains parameter-efficient across time. To further boost certified performance, we introduce Interval MixUp, a new training strategy that blends virtual example interpolation with provable guarantees using interval arithmetic. This allows us to maintain robustness even on interpolated synthetic inputs. We evaluate our method on a diverse set of benchmarks—Permuted MNIST, Rotated MNIST, Split CIFAR-100, and Split miniImageNet—the latter being a particularly challenging scenario for robust continual learning. Across all datasets, our model achieves state-of-the-art or highly competitive performance, sometimes doubling the robust accuracy of existing methods under strong attacks such as AutoAttack. These results underscore the scalability and effectiveness of our approach in real-world, adversarially robust continual learning.

### 📊 Evaluation

Tested on:
- Permuted MNIST
- Rotated MNIST
- Split CIFAR-100
- Split miniImageNet
- TinyImageNet

Across all datasets, SHIELD achieves **state-of-the-art** or highly competitive performance, often **doubling robust accuracy** vs. existing methods under strong attacks like **AutoAttack**.

---

## ⚙️ Getting Started

### 🛠️ Setup

1. **Create and activate the environment**  
   ```bash
   conda env create -f environment.yml
   conda activate shield
   ```

2. **Configure environment variables**  
   ```bash
   cp example.env .env
   # Edit `.env` with your preferred settings (paths, tracking, etc.)
   ```

---

### 🚀 Launching Experiments

```bash
WANDB_MODE=offline HYDRA_FULL_ERROR=1 python src/main.py --config-name config
```

> 💡 Use `WANDB_MODE=online` to enable live tracking with [Weights & Biases](https://wandb.ai)

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