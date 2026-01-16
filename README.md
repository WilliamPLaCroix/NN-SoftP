# NN-SoftP (Neural Networks Software Project)

This is a deep learning framework developed for the neural networks software project at Saarland University (Winter Semester 2023-2024). The repository’s primary goal is to provide a scalable and resource-efficient pipeline for **Fake News Detection** by leveraging state-of-the-art Large Language Models (LLMs).

### Project Goals

The project aims accurately identify misinformation in Taiwanses news sources. Additionally, by focusing on **parameter-efficient** workflows, it demonstrates how large-scale transformer architectures can be adapted for specialized classification tasks (like veracity checking) with minimal resources.

### Technical Implementation & Methods

To realize these goals, the project integrates several advanced machine learning techniques:

* **PEFT & LoRA:** The project utilizes **Parameter-Efficient Fine-Tuning (PEFT)**, specifically **LoRA (Low-Rank Adaptation)**. Instead of updating billions of parameters in the base LLM, the system injects small, trainable low-rank matrices into the Transformer layers. This reduces the number of trainable parameters by over 90%, enabling fine-tuning on consumer-grade hardware while maintaining high performance.
* **Quantization:** To further lower the barrier to entry, the system employs **quantization** (e.g., 4-bit or 8-bit precision). By reducing the numerical precision of the model weights, the project enables **QLoRA** workflows—where the base model remains quantized in memory while LoRA adapters are trained—drastically reducing VRAM requirements for both fine-tuning and real-time inference.
* **Symbolic-Neural Hybridization:** As detailed in the project's technical `Report.pdf`, the model structure is designed to be interpretable, mapping neural outputs to logical clusters to better understand the decision-making process behind a "Fake" or "Real" news classification.

By combining these methods, this project serves as a high-performance prototype for efficient, accurate, and scalable news verification.

---

Authors:
- Ho-Hsuan Wang
- Jakob Gürtler
- Philipp Hawlitscheck
- William LaCroix

Special acknowledgement to Prof. Dr. Dietrich Klakow and Vagrant Gautam for their help and inspiration.

License: MIT License
