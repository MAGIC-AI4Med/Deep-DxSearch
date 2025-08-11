<div align="center">



# *DiagRL*: Optimizing Workflow in Evidence-based Diagnosis through Reinforcement Learning (Still Updating ...)

(**Under updating ...**)

[![Notion](https://img.shields.io/badge/blog-black?style=for-the-badge&logo=notion)]() [![Arxiv](https://img.shields.io/badge/paper-A82F27?style=for-the-badge&logo=arxiv)]() [![Model](https://img.shields.io/badge/model-4169E1?style=for-the-badge&logo=huggingface)]() 

</div>
We introduce **DiagRL**, focusing on clinical presentation-based diagnosis, which requires combinatorial analysis of symptoms, evidence-based associations between symptoms and diseases, and differential diagnosis ranking. Unlike prior inference-only agentic systems, **DiagRL jointly optimizes retrieval and reasoning in an end-to-end fashion**, enabling the development of retrieval-aware diagnostic strategies. It leverages a large language model (LLM) as the decision-making core and operates through five structured action modes—***reason***, ***lookup***, ***match***, ***search***, ***diagnose***—which support stepwise evidence acquisition and transparent clinical reasoning.



## Key Insights

- We use the *LLM-based reinforcement learning* approach to enable the agent to learn ***when and how to retrieve information***, and ***how to optimize the reasoning paths*** through rule-based supervision tailored for diagnosis tasks.
- We open-source a large-scale **disease-symptom(phenotype) guideline** based on reliable resources
- We open-source a processed **patient record database** collected from 5 datasets.
- We open-source our **model checkpoint** which trained on multi-center diagnosis tasks in Huggingface. We hope this can Promote the development of agentic disease diagnosis.
- Diagnostic workflow with retrieval corpus and performance compared to SOTAs here:

<img src="https://github.com/MAGIC-AI4Med/DiagRL/blob/main/assets/performOverview.png"/> 

![performOverview](D:\Joy\GithubDesktop\DiagRL\assets\performOverview.png)



## Direct Usage



## Installation (Updating ...)

It is recommended that CUDA version >=12.1. If you encounter errors during installation, please adjust corresponding package version and refer to the [verl](https://verl.readthedocs.io/en/v0.2.x/start/install.html) document. Our project is currently based on verl v0.2x.

### Step1: Install the backbone requirements

Note that the following minimum installation is already sufficient for running DiagRL under basement settings.

```bash
# Initialize th Anaconda Environment
conda create -n DiagRL python==3.10
conda activate DiagRL

# Install Basic Verl-required Packages
cd ./your/path/to/DiagRL/verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
pip3 install -e .

# Install Customized Packages
cd ..
pip3 install -r requirements.txt
```

If you can not resolve the package contradiction, please follow the debug feedback. Violations to requirements.txt may not lead to inevitable failure

### Step2 (Optional): Install retriever environments

This server installation is borrowed from [Search-R1](https://github.com/PeterGriffinJin/Search-R1). While we process data from PubMed, Wiki and Textbook for retriever customization.

```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

### Step3 (Optional): Install LLM summarizer environments

We use SGLang to deploy a LLM summarizer to offer summarization service given a long context document from PubMed, Wikipedia, etc.

```bash
# Basic Installation
conda create -n llmServer python==3.10
conda activate llmServer
pip3 install uv
uv pip install "sglang[all]>=0.5.0rc0"
```

A quick test can be conducted during the training of DiagRL:

```python
# A quick test
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-14b-instruct --host 0.0.0.0

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path qwen/qwen2.5-14b-instruct \
 --host 0.0.0.0
"""
)

wait_for_server(f"http://localhost:{port}")
```

If you meet the CUDA out of memory problem, either:

- check whether the KV cache is correctly cleared after each rollout.
- check whether the memory utilization parameter is appropriately set.



## Quick Start (Updating ...)

Here we  use a example trained on MIMIC-IV-Common datasets to demonstrate how to training based on Qwen2.5-7B-Instruct.



### Training



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

<img src="https://github.com/MAGIC-AI4Med/DiagRL/blob/main/assets/Retrieval.png"/> 

![Retrieval](D:\Joy\GithubDesktop\DiagRL\assets\Retrieval.png)



## Benchmark

We show the main results of DiagRL and compare it to other frameworks. For more details, please refer to our paper.

<img src="https://github.com/MAGIC-AI4Med/DiagRL/blob/main/assets/comparison.png"/> 

![comparison](D:\Joy\GithubDesktop\DiagRL\assets\comparison.png)

**Table: Main diagnosis performance.**  
We calculate top-1 and top-5 accuracy among common and rare disease diagnosis datasets and compare our \ModelName{} with other representative models. “Env” means we allow the model to use our proposed environment as assistance. All results are shown in percentage.

| Model              | MIMIC-C Acc@1 | MIMIC-C Acc@5 | PMC-Patient Acc@1 | PMC-Patient Acc@5 | MedDialog Acc@1 | MedDialog Acc@5 | MIMIC-R Acc@1 | MIMIC-R Acc@5 | RareArena Acc@1 | RareArena Acc@5 | RareBench Acc@1 | RareBench Acc@5 |
| ------------------ | ------------- | ------------- | ----------------- | ----------------- | --------------- | --------------- | ------------- | ------------- | --------------- | --------------- | --------------- | --------------- |
| Qwen-2.5-14B       | 8.80          | 12.40         | 17.73             | 27.66             | 17.87           | 32.34           | 7.93          | 16.71         | 6.53            | 13.23           | 18.07           | 31.38           |
| Baichuan-M1        | 11.8          | 14.48         | 26.95             | 39.84             | 26.81           | 38.85           | 8.35          | 19.25         | 10.69           | 21.63           | 26.93           | 44.79           |
| DeepSeek-R1        | 5.65          | 15.32         | 29.62             | 41.52             | 28.34           | 40.96           | 12.05         | 23.90         | 10.98           | 22.56           | 28.22           | 50.83           |
| GPT-4o             | 6.43          | 9.82          | 23.51             | 36.10             | 22.59           | 36.01           | 7.65          | 15.58         | 12.83           | 23.10           | 24.25           | 43.54           |
| Qwen14B (Env)      | 13.22         | 15.91         | 24.38             | 35.57             | 24.69           | 36.22           | 16.54         | 24.33         | 10.08           | 15.47           | 34.70           | 59.20           |
| GPT-4o (Env)       | 15.07         | 21.25         | 28.64             | 38.38             | 25.86           | 39.41           | 20.47         | 29.05         | 11.24           | 19.32           | 40.11           | 63.28           |
| **Ours (Llama8B)** | 21.05         | 27.83         | 34.15             | 45.74             | 35.51           | 46.92           | 42.00         | 55.02         | 22.41           | 29.95           | 64.33           | 73.86           |
| **Ours (Qwen7B)**  | 33.09         | 42.87         | **41.41**         | 46.80             | **49.28**       | 55.34           | **52.44**     | 61.53         | 25.97           | 35.32           | 64.47           | 79.51           |
| **Ours (Qwen14B)** | **35.22**     | **46.83**     | 40.29             | **47.75**         | 48.81           | **60.04**       | 52.11         | **64.57**     | **28.14**       | **39.22**       | **70.48**       | **82.96**       |



**Table: Diagnosis performance compared to other frameworks.**  
We use GPT-4o as the large language model base for MedRAG and MAC framework, for other frameworks, we just follow their official settings during benchmarking. All results are shown in percentage.

| Framework  | Category       | MIMIC-C Acc@1 | MIMIC-C Acc@5 | PMC-Patient Acc@1 | PMC-Patient Acc@5 | MedDialog Acc@1 | MedDialog Acc@5 | MIMIC-R Acc@1 | MIMIC-R Acc@5 | RareArena Acc@1 | RareArena Acc@5 | RareBench Acc@1 | RareBench Acc@5 |
| ---------- | -------------- | ------------- | ------------- | ----------------- | ----------------- | --------------- | --------------- | ------------- | ------------- | --------------- | --------------- | --------------- | --------------- |
| MedCPT     | CLIP-based     | 0.00          | 0.81          | 7.80              | 17.73             | 6.81            | 17.45           | 4.79          | 8.38          | 1.80            | 2.99            | 4.82            | 11.45           |
| MedGemma   | Foundation     | 18.60         | 29.00         | 26.95             | 39.01             | 20.43           | 32.77           | 12.57         | 21.56         | 10.78           | 19.76           | 28.92           | 54.82           |
| MedRAG     | RAG-based      | 4.03          | 10.48         | 25.53             | 37.58             | 22.13           | 34.04           | 8.98          | 21.56         | 16.77           | 21.68           | 33.73           | 53.03           |
| COD        | COT-Agent      | 0.81          | 7.26          | 11.35             | 21.99             | 70.64           | 90.64           | 4.19          | 19.16         | 2.99            | 11.98           | 2.41            | 8.43            |
| MAC        | Multi-Agent    | 4.03          | 10.74         | 28.06             | 30.66             | 24.03           | 29.07           | 16.17         | 24.69         | 15.66           | 17.07           | 35.54           | 43.98           |
| **DiagRL** | **Agentic RL** | **35.22**     | **46.83**     | **40.29**         | **47.75**         | **48.81**       | **60.04**       | **52.11**     | **64.57**     | **28.14**       | **39.22**       | **70.48**       | **82.96**       |



## Acknowledgements

We thank Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory, and Xinhua Hospital for their fundings, computation and data support.

This training Implementation is based on [verl](https://github.com/volcengine/verl). The base LLMs are from [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) series. The retrieval serving is based on [FastAPI](https://github.com/fastapi/fastapi). The LLM service is based on [SGLang](https://github.com/sgl-project/sglang).  The retrieval corpus include a part of [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) as components. We sincerely appreciate their contributions.



## Citation & Contact



If you encounter any question, please raise a issue in this repository or directly contact three-world@sjtu.edu.cn
