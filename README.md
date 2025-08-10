<div align="center">

# *DiagRL*: Optimizing Workflow in Evidence-based Diagnosis through Reinforcement Learning

(**Under updating ...**)

[![Notion](https://img.shields.io/badge/blog-black?style=for-the-badge&logo=notion)]() [![Arxiv](https://img.shields.io/badge/paper-A82F27?style=for-the-badge&logo=arxiv)]() [![Model](https://img.shields.io/badge/model-4169E1?style=for-the-badge&logo=huggingface)]() 

</div>
We introduce DiagRL, focusing on clinical presentation-based diagnosis, which requires combinatorial analysis of symptoms, evidence-based associations between symptoms and diseases, and differential diagnosis ranking. Unlike prior inference-only agentic systems, DiagRL jointly optimizes retrieval and reasoning in an end-to-end fashion, enabling the development of retrieval-aware diagnostic strategies. It leverages a large language model (LLM) as the decision-making core and operates through five structured action modes—\<reason>, \<lookup>, \<match>, \<search>, \<diagnose>—which support stepwise evidence acquisition and transparent clinical reasoning.



## Key Insights

- We use the *LLM-based reinforcement learning* approach to enable the agent to learn *when and how to retrieve information*, and *how to optimize the reasoning paths* through rule-based supervision tailored for diagnosis tasks.
- We open-source a large-scale **disease-symptom(phenotype) guideline** from authoritative resources
- We open-source a processed **patient record database** collected from 5 datasets.
- We open-source our **model checkpoint** which trained on multi-center diagnosis tasks in Huggingface. We hope this can Promote the development of agentic disease diagnosis.
- Diagnostic workflow with retrieval corpus and performance compared to SOTAs here:

![performOverview](D:\Joy\GithubDesktop\DiagRL\assets\performOverview.png)



## Data Resources

**MIMIC-IV-note.** The official data repository is https://physionet.org/content/mimiciv/3.1/ . Pay attention to the ethics statement. Here we use a subset of it and further processed into the common, rare part.

**PMC-Patients.** This data is fully achievable at https://huggingface.co/datasets/zhengyun21/PMC-Patients . We further process it to a more cleaned subset of common disease related patient records.

**MedDialog.** This data is from https://huggingface.co/datasets/bigbio/meddialog . Users need to download through their provided scripts as the data is continuously growing. We further process it to fit our task.

**RareArena.** This data is also public available at https://huggingface.co/datasets/THUMedInfo/RareArena . It is a subset of PMC-Patients that only contain the rare disease related patients. We use the official version.

**RareBench.** This data is achievable at https://huggingface.co/datasets/chenxz/RareBench and we use the official version without further processing.

**Mendeley.** This data is achievable at https://data.mendeley.com/datasets/rjgjh8hgrt/2 for free. we use the English version for zeroshot test.

**Xinhua Hospital.** This is the in-house data for zero-shot only due to privacy concerns.



## Retrieval Corpus

**Disease-symptom(phenotype) Guidelines.** It contains 16,371 common and rare diseases with their typical phenotypes. These diseases are from former public datasets and the ICD10data. We search these diseases for their introduction from authoritative resources such as [MAYO CLINIC](https://www.mayoclinic.org/diseases-conditions), [NCBI](https://www.ncbi.nlm.nih.gov/), etc. Then we use GPT-4o to summarize the main information of these contents to structural information guideline. The processed file could be found at: (to upload).

**Patient Record Database.** We select a subset of former datasets as the multi-center patient records. We processed these data from their case report to extracted symptoms / phenotypes through GPT-4o and DeepSeek-V3. The processed database could be found at: (to upload)

**Medical Retrieval Collection.** This is borrowed from [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG). We sincerely appreciate their impressive work!
