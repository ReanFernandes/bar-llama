# Bar-Llama: Efficient Supervised Fine-Tuning for Legal Reasoning

This repository contains the work and resources related to the paper "A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam". This project investigates the effectiveness of fine-tuning smaller language models (Llama 2 7B and Llama 3 8B) for legal reasoning tasks, specifically focusing on the Multi-state Bar Examination (MBE).

**ArXiv Paper:** [A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam](https://arxiv.org/abs/2504.04945)

### Description
Legal reasoning presents unique challenges for Large Language Models (LLMs) due to the complexity of domain-specific knowledge and reasoning processes. This project explores how supervised fine-tuning (SFT) can enhance the performance of Llama 2 7B and Llama 3 8B models on MBE questions using a limited dataset. The models were evaluated on the 2022 MBE questions licensed from JD Advising, the same dataset used in the "GPT-4 passes the Bar exam" study. The methodology involved collecting approximately 200 questions per legal domain across 7 domains and distilling the dataset using Llama 3 (70B) into a structured IRAC (Issue, Rule, Application, Conclusion) format. The study compares non-fine-tuned models with their SFT counterparts, analyzing accuracy, prompt adherence, and option selection biases.

### Key Features & Contributions
* Investigates the fine-tuning of smaller, open-weight LLMs (Llama 2 7B, Llama 3 8B) for legal reasoning on consumer-grade hardware.
* Curated and released a fine-tuning dataset of 1,514 MBE questions, available in both un-distilled and structured IRAC formats.
* Released a family of Supervised Fine-tuned (SFT) adapters optimized for MBE performance with Llama 2 7B and Llama 3 8B models.
* Demonstrates that domain-specific SFT can help smaller models achieve close to human baseline performance with limited resources.
* Analyzes the impact of dataset distillation (IRAC format) on model performance.
* Studies the effect of fine-tuning on response parsing reliability and mitigation of option selection biases.
* Consolidates performance across multiple variables: prompt type (few-shot vs. zero-shot), answer ordering, response format (JSON, Markdown, Numbered list), and decoding temperatures.

### Models Used
* Llama 2 7B
* Llama 3 8B
* Llama 3 70B (for dataset distillation)

### Dataset
The fine-tuning dataset consists of 1,514 Multi-state Bar Examination (MBE) questions gathered from online study materials, covering 7 legal domains: Constitutional Law, Contract Law, Criminal Law and Procedure, Evidence, Real Property, Tort Law, and Civil Procedure.
* The dataset was meticulously verified to ensure no overlap with the test set licensed from JD Advising.
* Two versions of the dataset were created:
    * Original raw explanations.
    * Explanations restructured into the IRAC (Issue, Rule, Application, Conclusion) format using Llama 3 70B.
* The dataset is available at: [MBE-exam-questions on Hugging Face](https://huggingface.co/datasets/HolySaint/MBE-exam-questions)

### Methodology
* **Supervised Fine-Tuning (SFT):** Applied to Llama 2 7B and Llama 3 8B models.
* **Q-LoRa:** Utilized for fine-tuning due to its lower memory requirements, allowing for larger batch sizes on a single NVIDIA Tesla V100 PCIe 32 GB GPU.
* **IRAC Distillation:** Explanations in the dataset were transformed into the IRAC format to guide the reasoning process.
* **Evaluation:** Models were benchmarked on 2022 MBE questions from JD Advising. Performance was analyzed based on accuracy, adherence to response format, and option selection bias.

### Results
* Domain-specific SFT significantly enhanced model performance even with minimal training data.
* Llama 3 (8B) improved from an untrained baseline of 35.8% to 52.5% accuracy with just 20 samples per domain.
* Llama 2 (7B) improved from 18.5% to 36.8% accuracy, requiring 225 samples to reach its peak.
* Fine-tuning dramatically reduced response parsing failures:
    * Llama 2: from 42.7% to 5.3% with 10 samples.
    * Llama 3: from 30.5% to 1.6% with 10 samples.
* The structured IRAC format showed benefits for Llama 3's learning pattern.
* Fine-tuning helped mitigate inherent option selection biases in the models.

The project provides:
* The curated SFT dataset (see [Dataset](#dataset) section).
* Fine-tuned model adapters for Llama 2 7B and Llama 3 8B [on huggingface.](https://huggingface.co/HolySaint/bar-Llama-adapters/tree/main)
### Limitations
* The study does not analyze the correlation between the factuality of generated explanations and the model's prediction correctness due to evaluation complexities.
* The research was confined to a fine-tuning setting and did not explore advanced test-time inference strategies as an alternative to dataset creation and model fine-tuning.
* Performance on the MBE does not directly translate to mastery over law or readiness for real-life legal support without further guardrails and human verification. The fine-tuned models are specialized for the MBE question format.

### Future Scope
* Analyzing the correlation between predicted options and generated explanations using higher-capability models or a labeled preference dataset for further alignment (e.g., DPO).
* Investigating advanced test-time inference strategies (e.g., chain-of-thought decoding) as potentially more efficient paths to improving legal reasoning capabilities.
* Adapting these models for broader legal applications, such as chat-based legal assistants, which would require different approaches.

### Citation
If you use the work or resources from this project, please cite the original paper:
```
@misc{fernandes2025llamawalksbarefficient,
      title={A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam}, 
      author={Rean Fernandes and André Biedenkapp and Frank Hutter and Noor Awad},
      year={2025},
      eprint={2504.04945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04945}, 
}```
