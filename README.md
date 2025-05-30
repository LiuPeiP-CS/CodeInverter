---

## 🧠 About This Repository

This repository contains the official implementation of our paper 📄 [PDF](https://arxiv.org/pdf/2503.07215), introducing the **CodeInverter Suite** — a novel framework for enhancing neural decompilation via structural program analysis. We provide full source code, a large-scale dataset ([CID](https://huggingface.co/datasets/CodeInverter/CID)), and pretrained models ([CIM-1.3B](https://huggingface.co/CodeInverter/CIM-1.3b), [CIM-6.7B](https://huggingface.co/CodeInverter/CIM-6.7b)).

---

### 🚀 Highlights

The **CodeInverter Suite** makes three key contributions:

#### 1. **CIW: Code-Informed Wrapper**

We are the **first to incorporate control flow graphs (CFGs)** and **memory-to-data mappings** into the input space of large language models for code decompilation. This structured guidance enables models to generate **more accurate and semantically aligned code**.

> Extensive ablation studies demonstrate the significant performance gains of CIW across various backbone LLMs, validating the value of structural input for decompilation.

#### 2. **CID: CodeInverter Dataset**

We release **CID**, a large-scale dataset containing **8.69 million function-level samples**, each annotated with both **control flow graphs** and **memory-to-data mappings**. This is the first dataset of its kind to support structurally aware training for decompilation models.

> CID provides a rich foundation for advancing decompilation research in both supervised learning and pretraining settings.

#### 3. **CIM: CodeInverter Models (1.3B & 6.7B)**

Built upon the [LLM4Decompile](https://github.com/albertan017/LLM4Decompile) architecture and trained using [ColossalAI](https://github.com/hpcaitech/ColossalAI), we introduce:

* **CIM-1.3B** – Lightweight and fast, optimized for efficiency.
* **CIM-6.7B** – More capable while remaining resource-efficient.

We evaluate these models on **ExeBench** and **HumanEval** using rigorous metrics:

* ✅ *Re-compilation*
* 🔁 *Re-executability*
* 📝 *Edit Similarity*
* 📊 *Pass\@k*

> Remarkably, **CIM models outperform much larger models like DeepSeek-V3 (up to 100× in size)** while achieving **state-of-the-art (SOTA)** performance in both correctness and efficiency.

---

### 🛠️ Training

The training pipeline is adapted from [ColossalAI’s LLaMA examples](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA).
To start training, use:

```bash
train/run_ms2decllm_train.sh
```

---


## 📊 Experimental Results

We evaluate our models on HumanEval and ExeBench datasets. The comparison is shown below.

### 🔹 Table 1: Decompilation results on HumanEval using only Assembly instructions (without CIW) (%)
| Metric     | Model       | O0 (32)   | O1 (32)   | O2 (32)   | O3 (32)   | AVG (32)  | O0 (64)   | O1 (64)   | O2 (64)   | O3 (64)   | AVG (64)  |
| ---------- | ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **Re-com** | GPT-4o      | 85.98     | 82.32     | 83.54     | 78.05     | 82.47     | 89.02     | 77.44     | 85.98     | 79.27     | 82.93     |
|            | Deepseek-V3 | **94.51** | **87.80** | **85.98** | **89.63** | **89.48** | **95.73** | **88.41** | **89.02** | **84.15** | **89.33** |
|            | Qwen-plus   | 25.61     | 37.80     | 41.46     | 46.34     | 37.80     | 73.17     | 58.54     | 61.69     | 62.20     | 63.90     |
| **Re-exe** | GPT-4o      | 22.56     | 11.59     | 13.41     | 9.51      | 14.27     | 33.54     | 10.98     | 14.02     | 9.76      | 17.08     |
|            | Deepseek-V3 | **57.93** | **29.27** | **33.54** | **35.37** | **39.03** | **66.46** | **36.59** | **37.80** | **37.80** | **44.66** |
|            | Qwen-plus   | 0.00      | 3.05      | 4.27      | 4.88      | 3.05      | 19.51     | 7.93      | 5.49      | 7.93      | 10.22     |
| **ES**     | GPT-4o      | 34.92     | 31.54     | 31.37     | 29.88     | 31.93     | 39.02     | 30.02     | 31.88     | 29.69     | 32.65     |
|            | Deepseek-V3 | **42.65** | **34.16** | **34.29** | **33.17** | **36.07** | **44.58** | **34.14** | **35.75** | **33.84** | **37.08** |
|            | Qwen-plus   | 10.64     | 15.78     | 16.42     | 15.71     | 14.64     | 27.44     | 22.06     | 21.57     | 22.29     | 23.34     |


### 🔹 Table 2-4: Comparisons between our decompilation LLMs with the baselines on HumanEval (%)

#### Metric: Re-com (Re-compilation Accuracy)

| Model                | 32-bit O0 | O1        | O2        | O3        | AVG       | 64-bit O0 | O1        | O2        | O3        | AVG       |
| -------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| GPT-4o +CIW          | 95.73     | 91.46     | **92.07** | **93.90** | **93.29** | **97.56** | 91.46     | **94.51** | 84.76     | 92.07     |
| Deepseek-V3 +CIW     | **96.34** | **92.07** | 89.63     | 90.85     | 92.22     | **97.56** | 87.80     | 85.37     | 78.66     | 87.35     |
| Qwen-plus +CIW       | 90.24     | 86.59     | 85.98     | 91.46     | 88.57     | 89.63     | 87.20     | 87.20     | 70.12     | 83.54     |
| LLM4Decompile 1.3B\* | 9.15      | 25.00     | 17.68     | 18.90     | 17.68     | 56.10     | 58.54     | 54.27     | 56.10     | 56.25     |
| LLM4Decompile 6.7B\* | 7.32      | 34.15     | 35.98     | 35.98     | 28.35     | 71.95     | 80.49     | 75.00     | 75.00     | 75.61     |
| FAE†                 | -         | -         | -         | -         | -         | 92.07     | **93.29** | 92.07     | **93.90** | **92.84** |
| CIM-1.3B +CIW        | 85.98     | 87.20     | 90.24     | 89.36     | 88.26     | 90.85     | 87.80     | 87.80     | 86.59     | 88.26     |
| CIM-6.7B +CIW        | 89.02     | 90.85     | 90.24     | 92.07     | 90.55     | 93.29     | 91.46     | 93.29     | 92.07     | 92.53     |

---

#### Metric: Re-exe (Re-execution Accuracy)

| Model                | 32-bit O0 | O1        | O2        | O3        | AVG       | 64-bit O0 | O1        | O2        | O3        | AVG       |
| -------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| GPT-4o +CIW          | 32.32     | 28.05     | 24.39     | 24.39     | 27.29     | 45.12     | 29.27     | 23.17     | 19.51     | 29.27     |
| Deepseek-V3 +CIW     | 54.27     | 40.24     | 34.76     | 34.15     | 40.86     | 71.34     | 45.73     | 45.73     | 42.07     | 51.22     |
| Qwen-plus +CIW       | 22.56     | 17.07     | 15.85     | 17.68     | 18.29     | 26.83     | 13.41     | 12.20     | 12.80     | 16.31     |
| LLM4Decompile 1.3B\* | 0.00      | 1.22      | 0.61      | 0.61      | 0.61      | 25.61     | 10.37     | 7.32      | 9.76      | 13.26     |
| LLM4Decompile 6.7B\* | 0.61      | 1.22      | 0.61      | 0.61      | 0.76      | 39.02     | 26.22     | 30.49     | 28.05     | 30.94     |
| FAE†                 | -         | -         | -         | -         | -         | 71.95     | 53.66     | 48.78     | 45.73     | 55.03     |
| CIM-1.3B +CIW        | 65.24     | 34.15     | 35.37     | 31.71     | 41.62     | 71.34     | 39.63     | 42.07     | 40.24     | 48.32     |
| CIM-6.7B +CIW        | **75.61** | **53.05** | **53.05** | **50.00** | **57.93** | **80.49** | **57.93** | **56.71** | **53.05** | **62.05** |

---

#### Metric: ES (Edit Similarity)

| Model                | 32-bit O0 | O1        | O2        | O3        | AVG       | 64-bit O0 | O1        | O2        | O3        | AVG       |
| -------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| GPT-4o +CIW          | 40.81     | 35.47     | 36.28     | 35.41     | 36.99     | 44.48     | 36.45     | 36.86     | 34.11     | 37.98     |
| Deepseek-V3 +CIW     | 45.47     | 36.15     | 36.98     | 35.33     | 38.48     | **51.36** | 36.33     | 37.61     | 33.61     | 39.73     |
| Qwen-plus +CIW       | 39.71     | 34.80     | 35.76     | 34.37     | 36.16     | 43.59     | 35.75     | 35.14     | 31.63     | 36.53     |
| LLM4Decompile 1.3B\* | 14.05     | 13.42     | 11.23     | 11.08     | 12.45     | 30.14     | 19.89     | 18.81     | 20.26     | 22.27     |
| LLM4Decompile 6.7B\* | 29.03     | 20.52     | 21.80     | 21.15     | 23.12     | 41.42     | 32.07     | 32.03     | 31.68     | 34.30     |
| CIM-1.3B +CIW        | 47.45     | 35.88     | 37.03     | 36.33     | 39.17     | 47.56     | 35.67     | 37.64     | 37.31     | 39.55     |
| CIM-6.7B +CIW        | **51.06** | **40.16** | **39.96** | **40.48** | **42.92** | 49.16     | **40.25** | **39.54** | **39.02** | **41.99** |

---


### 🔹 Table 5-7: Comparison of our decompilation models (CIM-\*) against LLM-based baselines on ExeBench. † Results taken from the original paper. (%)

#### Re-com

| Metric | Model             | O0                  | O1        | O2        | O3        | AVG       | O0                  | O1        | O2        | O3        | AVG       |
| ------ | ----------------- | ------------------- | --------- | --------- | --------- | --------- | ------------------- | --------- | --------- | --------- | --------- |
|        |                   | **ExeBench 32-bit** |           |           |           |           | **ExeBench 64-bit** |           |           |           |           |
| Re-com | GPT-4o + CIW      | 80.07               | **85.25** | **84.80** | **85.13** | **83.81** | 90.54               | 88.34     | 88.09     | 87.28     | 88.56     |
|        | Deepseek-V3 + CIW | 84.30               | 81.30     | 80.15     | 81.41     | 81.79     | **93.32**           | **89.68** | **89.98** | **88.68** | **90.42** |
|        | Qwen-plus + CIW   | 71.97               | 78.68     | 78.66     | 80.00     | 77.33     | 82.84               | 82.78     | 83.20     | 81.41     | 82.56     |
|        | CIM-1.3B + CIW    | 84.14               | 73.55     | 74.72     | 73.28     | 76.42     | 88.33               | 70.76     | 70.72     | 69.99     | 74.95     |
|        | CIM-6.7B + CIW    | **86.87**           | 73.77     | 73.01     | 73.52     | 76.79     | 89.49               | 70.57     | 70.20     | 68.14     | 74.60     |

---

#### Re-exe

| Metric | Model               | O0                  | O1        | O2        | O3        | AVG       | O0                  | O1        | O2        | O3        | AVG       |
| ------ | ------------------- | ------------------- | --------- | --------- | --------- | --------- | ------------------- | --------- | --------- | --------- | --------- |
|        |                     | **ExeBench 32-bit** |           |           |           |           | **ExeBench 64-bit** |           |           |           |           |
| Re-exe | GPT-4o + CIW        | 30.87               | 30.20     | 28.27     | 29.08     | 29.61     | 43.99               | 25.61     | 23.61     | 22.06     | 28.82     |
|        | Deepseek-V3 + CIW   | 43.83               | 36.55     | 34.26     | 34.51     | 37.29     | 59.16               | 36.48     | 32.89     | 30.86     | 39.85     |
|        | Qwen-plus + CIW     | 22.05               | 22.54     | 21.22     | 21.28     | 21.77     | 32.96               | 19.85     | 17.33     | 16.33     | 21.62     |
|        | LLM4Decompile 1.3B† | --                  | --        | --        | --        | --        | 17.86               | 13.62     | 13.20     | 13.28     | 14.49     |
|        | LLM4Decompile 6.7B† | --                  | --        | --        | --        | --        | 22.89               | 16.60     | 16.18     | 16.25     | 17.98     |
|        | CIM-1.3B + CIW      | 56.09               | 35.33     | 33.94     | 32.49     | 39.46     | 65.20               | 36.32     | 33.34     | 32.55     | 41.85     |
|        | CIM-6.7B + CIW      | **64.74**           | **42.86** | **40.60** | **40.66** | **47.21** | **72.13**           | **40.42** | **36.57** | **35.45** | **46.14** |

---

#### ES

| Metric | Model             | O0                  | O1        | O2        | O3        | AVG       | O0                  | O1        | O2        | O3        | AVG       |
| ------ | ----------------- | ------------------- | --------- | --------- | --------- | --------- | ------------------- | --------- | --------- | --------- | --------- |
|        |                   | **ExeBench 32-bit** |           |           |           |           | **ExeBench 64-bit** |           |           |           |           |
| ES     | GPT-4o + CIW      | 52.02               | 40.53     | 38.68     | 38.44     | 42.42     | 46.53               | 41.24     | 39.85     | 40.16     | 41.95     |
|        | Deepseek-V3 + CIW | 54.93               | 44.73     | 42.97     | 42.70     | 46.33     | 60.95               | 44.31     | 41.81     | 41.20     | 47.07     |
|        | Qwen-plus + CIW   | 43.62               | 40.39     | 39.15     | 39.06     | 40.56     | 50.98               | 39.87     | 38.59     | 37.70     | 41.79     |
|        | CIM-1.3B + CIW    | 66.14               | 50.02     | 49.03     | 48.03     | 53.31     | 67.58               | 50.27     | 49.21     | 48.28     | 53.84     |
|        | CIM-6.7B + CIW    | **69.94**           | **53.04** | **51.28** | **50.68** | **56.24** | **67.94**           | **52.72** | **51.03** | **50.48** | **55.54** |

---



### 🔹 Table 8: Comparisons between our decompilation LLMs with the baselines on HumanEval 64-bit. † Results taken from the original paper. (%)


#### Pass\@1 and Pass\@10 Scores

| Model                | O0          | O1        | O2        | O3        | AVG       | O0           | O1        | O2        | O3        | AVG       |
| -------------------- | ----------- | --------- | --------- | --------- | --------- | ------------ | --------- | --------- | --------- | --------- |
|                      | **Pass\@1** |           |           |           |           | **Pass\@10** |           |           |           |           |
| GPT-4o †             | 21.34       | 18.29     | 14.48     | 13.05     | 16.79     | 29.94        | 26.74     | 21.42     | 19.88     | 24.50     |
| LLM4Decompile 1.3B † | 15.30       | 8.26      | 9.36      | 8.38      | 10.33     | 21.79        | 15.23     | 16.17     | 13.70     | 16.72     |
| Nova 1.3B †          | 37.53       | 21.71     | 22.68     | 18.75     | 25.17     | 49.38        | 34.84     | 36.95     | 32.03     | 38.30     |
| LLM4Decompile 6.7B † | 29.97       | 19.05     | 20.46     | 18.32     | 21.95     | 40.40        | 27.75     | 28.85     | 28.51     | 31.38     |
| Nova 6.7B †          | 48.78       | 30.58     | 30.85     | 27.23     | 34.36     | 57.47        | 47.45     | 43.03     | 39.68     | 46.91     |
| CIM-1.3B + CIW       | 70.64       | 38.41     | 38.63     | 37.26     | 46.23     | 82.71        | 59.75     | 57.74     | 55.70     | 63.97     |
| **CIM-6.7B + CIW**   | **79.66**   | **57.56** | **55.70** | **52.96** | **61.47** | **90.35**    | **76.79** | **71.48** | **67.68** | **76.57** |



## 📚 Citation

If you find our work useful for your research or product, please consider citing:

```bibtex
@article{chen2024codeinverter,
  title={The CodeInverter Suite: Control-Flow and Data-Mapping Augmented Binary Decompilation with LLMs},
  author={Peipei Liu, Sun Jian, Rongkang Sun, Li Chen, zhaoteng yan, Zhang Peizheng, Dapeng Sun, Dawei Wang, Xiaoling Zhang, Dan Li},
  journal={arXiv preprint arXiv:2503.07215},
  year={2025}
}
```

---
