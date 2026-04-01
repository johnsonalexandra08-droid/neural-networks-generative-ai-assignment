# Neural Networks & Generative AI Assignment
**M10 | Applied Artificial Intelligence**  
**Author:** Alexandra Johnson

---

## Overview

This repository contains my M10 assignment exploring neural network fundamentals and generative AI techniques through hands-on experimentation with the Fashion-MNIST dataset.

---

## Repository Structure

```
├── neural_networks_generative_ai.ipynb   # Main notebook (Parts A, B, D)
└── README.md
```

---

## Assignment Parts

### Part A — CNN on Fashion-MNIST
Built a Convolutional Neural Network using TensorFlow/Keras with:
- Two convolutional blocks (Conv2D → BatchNorm → MaxPool → Dropout)
- Dense head with 128 units (ReLU) + softmax output
- Trained for up to 15 epochs with early stopping on validation accuracy

**Baseline Test Accuracy: ~91–92%**

### Part B — Data Augmentation
Applied `ImageDataGenerator` with:
- Horizontal flip
- Rotation ±10°
- Width/height shift ±10%
- Zoom range ±10%

Retrained the identical CNN architecture on augmented data and compared test accuracy and validation curves to the baseline.

**Key finding:** Augmentation reduced overfitting and produced smoother validation loss curves. The augmented model generalized more reliably, especially for visually similar categories (Shirt vs. T-shirt, Coat vs. Pullover).

### Part D — Reflection

**What I learned:** Data augmentation acts as a lightweight form of generative AI — it synthesizes realistic training variations without requiring additional labeled data. The primary benefit is regularization: the model learns features that are robust to translation, rotation, and scale rather than memorizing exact pixel patterns.

**Challenges:** Properly separating the validation set before fitting the `ImageDataGenerator` was critical to avoid evaluating on augmented images. Dropout tuning also required iteration to balance regularization vs. convergence speed.

**Real-world application (Emerson / Fisher Valves):** In manufacturing quality inspection, defect images are rare and expensive to collect. Data augmentation would allow training robust visual inspection models on small labeled defect datasets. Advanced generative approaches (VAEs/GANs) could even synthesize synthetic defect images for rare failure modes — improving classifier coverage on safety-critical components.

---

## How to Run

**Option 1 — Google Colab (recommended):**  
Upload the `.ipynb` file to [colab.research.google.com](https://colab.research.google.com) and run all cells. TensorFlow is pre-installed.

**Option 2 — Local:**
```bash
pip install tensorflow numpy matplotlib scikit-learn
jupyter notebook neural_networks_generative_ai.ipynb
```

---

## Results Summary

| Model | Test Accuracy |
|---|---|
| Baseline CNN (no augmentation) | ~91–92% |
| Augmented CNN | ~92–93% |

Augmentation provided consistent improvement in generalization, particularly visible in smoother validation curves and reduced gap between training and validation accuracy.

---

## References
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Ch. 10–12
- [Keras ImageDataGenerator documentation](https://keras.io/api/preprocessing/image/)
- [Fashion-MNIST dataset — Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
