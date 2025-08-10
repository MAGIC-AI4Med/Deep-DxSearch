<div align="center">

# *DiagRL*: Optimizing Workflow in Evidence-based Diagnosis through Reinforcement Learning

[![Notion](https://img.shields.io/badge/blog-black?style=for-the-badge&logo=notion)]() [![Arxiv](https://img.shields.io/badge/paper-A82F27?style=for-the-badge&logo=arxiv)]() [![Model](https://img.shields.io/badge/model-4169E1?style=for-the-badge&logo=huggingface)]() 

We introduce DiagRL, focusing on clinical presentation-based diagnosis, which requires combinatorial analysis of symptoms, evidence-based associations between symptoms and diseases, and differential diagnosis ranking. Unlike prior inference-only agentic systems, DiagRL jointly optimizes retrieval and reasoning in an end-to-end fashion, enabling the development of retrieval-aware diagnostic strategies. It leverages a large language model (LLM) as the decision-making core and operates through five structured action modes—<reason>, <lookup>, <match>, <search>, <diagnose>—which support stepwise evidence acquisition and transparent clinical reasoning.



## Key Insights

- We use the *LLM-based reinforcement learning* approach to enable the agent to learn *when and how to retrieve information*, and *how to optimize the reasoning paths* through rule-based supervision tailored for diagnosis tasks.
- We open-source a large-scale **disease-symptom(phenotype) guideline** from authoritative resources
- We open-source a processed **patient record database** collected from 5 datasets.
- We open-source our **model checkpoint** which trained on multi-center diagnosis tasks in Huggingface. We hope this can Promote the development of agentic disease diagnosis.
- Diagnostic workflow with retrieval corpus and performance compared to SOTAs here:

![performOverview](D:\Joy\GithubDesktop\DiagRL\assets\performOverview.png)

