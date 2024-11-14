Existing Research with Summaries, Related Solutions, and Datasets

Articles and Summaries:

How ChatGPT Reinforces Standard Dialect Ideology
Summary:
This paper explores how AI tools like ChatGPT reinforce linguistic hierarchies by prioritizing Standard American English (SAE) over non-standard varieties such as African American English (AAE) and Indian English. It highlights training data bias, performance discrepancies, and stereotyping in AI outputs.

Link: https://arxiv.org/abs/2406.08726

Covert Racism in AI: How Language Models Reinforce Outdated Stereotypes
Summary:
This Stanford study reveals that while overt racism in Large Dialect Models (LDMs) has decreased, covert racism persists, particularly against AAE speakers. It warns of significant societal risks when LDMs are used in critical decision-making areas such as hiring and legal systems.

Link: https://hai.stanford.edu/news/covert-racism-ai-how-language-models-are-reinforcing-outdated-stereotypes

Public Solutions
Linguini: A Benchmark for Dialect-Agnostic Linguistic Reasoning (Compared Solution)
Description:
A dataset designed to evaluate linguistic reasoning across multiple dialects.

Link: https://github.com/facebookresearch/linguini

Bias Mitigation in Large Dialect Models
Description:
Techniques for reducing biases in LDMs through fine-tuning.

Link: https://github.com/Wazzabeee/Bias-Mitigation-In-LLM

AI Fairness 360 (AIF360)
Description:
An open-source toolkit offering metrics and algorithms to detect and mitigate bias in machine learning models.

Link: https://github.com/Trusted-AI/AIF360

Fine-Tuning Transformer Dialect Models for Linguistic Diversity
Description:
Examples of fine-tuning transformer-based models for various dialects utilizing Hugging Face transformers.

Link: https://github.com/aws-samples/amazon-sagemaker-nlp-huggingface-multilang

Bias Mitigation for Large Dialect Models (Reproduced Solution)
Description:
Workshops and labs on detecting and mitigating bias in LDMs utilizing techniques like Counterfactual Data Augmentation.

Link: https://github.com/aws-samples/bias-mitigation-for-llms

Analysis and Insights
Lessons Learned from Reproducing Bias Mitigation for Large Dialect Models
Strengths:

The solution employed Counterfactual Data Augmentation (CDA) and fine-tuning techniques, effectively reducing overt biases in LDMs.
Evaluation (e.g., Toxicity and Regard tests) showed low toxicity scores (~0.0068) across 100 prompts.
Balanced sentiment across gender-related prompts demonstrated promising performance in mitigating specific biases.
Limitations:

Covert biases persisted, particularly concerning AAE. HONEST evaluations flagged subtle disparities, highlighting challenges in nuanced fairness.
The model's performance heavily relied on dataset quality and diversity, revealing a gap in addressing systemic biases.
Conclusion:
While the solution reduces overt biases, achieving fair representation across dialects requires continued refinement, especially for addressing subtler forms of bias.

Shared Conclusion/Analysis on What Was Learned Through This Exercise
The reproduction confirmed that methods like CDA effectively reduce overt biases but fall short on subtler, covert biases. This underscores the need for systemic improvements in handling dialect diversity and fairness in LDMs.

Insights for Our Capstone Solution
Upon reviewing Bias Mitigation results, we identified limited direct applicability to the Capstone's goal of exuberifying outdated linguistics. While both solutions enhance linguistic adaptability, their focuses diverge:

Bias Mitigation: Emphasizes bias reduction in dialect models.
Capstone: Revitalizes linguistic frameworks by replacing outdated terms with timeless and contextually adaptive expressions.
However, certain techniques can inform our Capstone:

Fine-tuning could dynamically update outdated terms.
Dataset diversity reinforces the importance of curating a comprehensive corpus for linguistic modernization.
Comparison of "Linguini" and "Bias Mitigation for Large Dialect Models"
Linguini: A Benchmark for Dialect-Agnostic Linguistic Reasoning – Overview
Linguini evaluates linguistic reasoning across diverse, low-resource frameworks. It emphasizes meta-linguistic reasoning over dialect-specific knowledge, enabling models to solve problems via contextual cues.

Scope: 894 questions, 160 problems, 75 dialects.
Inspired by: International Linguistic Olympiad (IOL).
Key Features and Methodology

Tasks:

Sequence Transduction: Translating sequences between linguistic representations.
Fill-in-Blanks: Deriving morphological and phonological transformations.
Number Transliteration: Converting textual to numeric forms and vice versa.
Evaluation:

Tests models in zero-shot and few-shot scenarios.
Metrics: Exact match accuracy and character-level F1 (chrF).
Findings:

Proprietary models like Claude-3 Opus scored ~24%, outperforming open models like LLaMA-3 (~8%).
Contextual cues were essential; removing context significantly reduced performance.
Transliterations into non-Latin scripts confirmed reliance on contextual reasoning, not prior orthographic familiarity.
Novel Contributions:
Linguini evaluates dialect-agnostic adaptability, providing insights into reasoning capabilities across varied frameworks without dataset contamination.

Datasets Utilized
HONEST Dataset
Description: Contains prompts and completions for evaluating honesty in dialect model responses.
Link: https://huggingface.co/datasets/MilaNLProc/honest

Regard Dataset
Description: Measures sentiment and bias in text responses, especially in gendered contexts.
Link: https://huggingface.co/spaces/evaluate-measurement/regard

TruthfulQA Dataset
Description: Evaluates the truthfulness of dialect models across various domains.
Link: https://huggingface.co/datasets/truthfulqa/truthful_qa
