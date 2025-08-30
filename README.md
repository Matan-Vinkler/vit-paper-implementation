# Vision Transformer (ViT) — Paper Reimplementation in PyTorch

From-scratch implementation of *[An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)* (Dosovitskiy et al., 2020).

This project demonstrates my ability to read a research paper, translate it into working code, and validate the implementation with training experiments.

---

## 📌 Project Goals

* Faithfully reimplement ViT components as described in the paper.
* Train and evaluate on CIFAR-10 to validate correctness.
* Provide unit tests, training loop, and visualizations for reproducibility.

---

## 🏗️ Implementation Details

### Components implemented

* **Patchify / Unpatchify**
* **Patch Projection**
* **CLS token addition**
* **Learnable Positional Embeddings**
* **Multi-Head Self Attention (MSA)**
* **Feedforward MLP block**
* **Transformer Encoder Block (Pre-LN, residuals, dropout)**
* **Vision Transformer (stack of L encoder blocks)**
* **Classification head (Linear / MLP)**

### Training setup

* Optimizer: **AdamW**
* LR schedule: **linear warmup + cosine decay**
* Label smoothing: optional
* Augmentations: RandomResizedCrop, RandomHorizontalFlip (optionally RandAugment)
* Mixed precision (AMP) training

---

## 📊 Results (CIFAR-10 @ 224×224)

| Model Config | Patch Size | Depth | Embed Dim | Heads | Head Type | Epochs | Val Acc |
| ------------ | ---------- | ----- | --------- | ----- | --------- | ------ | ------- |
| ViT-Tiny     | 16         | 6     | 128       | 4     | Linear    | 30     | \~60%   |
| ViT-Tiny     | 16         | 6     | 128       | 4     | MLP       | 30     | \~58%   |

> ⚠️ ViT underperforms CNNs (e.g. ResNet) on small datasets, consistent with the paper’s claim that **large-scale data is required** for ViTs to shine.

---

## 🔍 Next Steps / Extensions

* Fine-tune at higher resolution with **positional embedding interpolation**.
* Add attention visualizations.
* Train on a larger dataset (Tiny-ImageNet or ImageNet-100) to show ViT scaling.

---

## 🚀 How to Run (Google Colab)

1. **Open in Colab**

   * Upload this notebook (`vit-implementation-tiny.ipynb`) to [Google Colab](https://colab.research.google.com/).
   * [Or open directly from GitHub.](https://colab.research.google.com/github/Matan-Vinkler/vit-paper-implementation/blob/main/vit_implementation_tiny.ipynb)

2. **Set runtime to GPU**

   * In Colab: `Runtime` → `Change runtime type` → `T4 GPU`.

3. **Install dependencies** (run this in the first code cell):

   ```bash
   !pip install numpy torch torchvision matplotlib
   ```

4. **Run the notebook cells in order**

   * Sections are organized as:
     1. Imports & utils
     2. Patchify / Tokenizer tests
     3. Attention & MLP blocks
     4. Encoder & VisionTransformer class
     5. Training loop (CIFAR-10)

5. **Expected runtime**

   * With `embed_dim=128`, `depth=6`, and CIFAR-10: \~2–3 minutes per epoch on a Colab GPU.
   * After \~30 epochs, you should reach **\~58–60% validation accuracy**.

---

## 📖 References

* Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
* Official JAX/Flax implementation: [google-research/vision\_transformer](https://github.com/google-research/vision_transformer)
