

## 🧠 About This Repository

This repository releases our large language model for decompilation, built upon [LLM4Decompile](https://github.com/albertan017/LLM4Decompile) and trained with [ColossalAI](https://github.com/hpcaitech/ColossalAI).

In this work, we are the **first** to incorporate **control flow graphs (CFGs)** and **memory-to-data mappings** into the input of a large language model to enhance decompilation quality. This structural information guides the model toward more accurate and semantically meaningful code generation.

We achieve **state-of-the-art performance** across multiple datasets and evaluation metrics, including ExeBench and HumanEval. The training process, preprocessing pipeline, and evaluation steps are all provided in this repository, with data split, alignment, and transformation clearly organized in dedicated folders.

The training code is adapted from the official [ColossalAI LLaMA examples](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA), and the core script for training is `train/run_ms2decllm_train.sh`.

📄 Our paper is available on arXiv: [https://arxiv.org/pdf/2503.07215](https://arxiv.org/pdf/2503.07215)


