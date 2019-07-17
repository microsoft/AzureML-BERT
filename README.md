# BERT on Azure Machine Learning Service
This repo contains end-to-end recipes to [pretrain](#pretrain) and [finetune](#finetune) the [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) language representation model using [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/). 

## BERT
BERT is a language representation model that is distinguished by its capacity to effectively capture deep and subtle textual relationships in a corpus. In the original paper, the authors demonstrate that the BERT model could be easily adapted to build state-of-the-art models for a number of NLP tasks, including text classification, named entity recognition and question answering. In this repo, we provide notebooks that allow a developer to pretrain a BERT model from scratch on a corpus, as well as to fine-tune an existing BERT model to solve a specialized task. A brief [introduction to BERT](docs/bert-intro.md) is available in this repo for a quick start on BERT. 

### Pretrain
###### Challenges in BERT Pretraining
Pretraining a BERT language representation model to the desired level of accuracy is quite challenging; as a result, most developers start from a BERT model that was pre-trained on a standard corpus (such as Wikipedia), instead of training it from scratch. This strategy works well if the final model is being trained on a corpus that is similar to the corpus used in the pre-train step; however, if the problem involves a specialized corpus that's quite different from the standard corpus, the results won't be optimal. Additionally, to advance language representation beyond BERT’s accuracy, users will need to change the model architecture, training data, cost function, tasks, and optimization routines. All these changes need to be explored at large parameter and training data sizes. In the case of BERT-large, this could be quite substantial as it has 340 million parameters and trained over a very large document corpus. To support this with GPUs, machine learning engineers will need distributed training support to train these large models. However, due to the complexity and fragility of configuring these distributed environments, even expert tweaking can end up with inferior results from the trained models.

To address these issues, this repo is publishing a workflow for pretraining BERT-large models. Developers can now build their own language representation models like BERT using their domain-specific data on GPUs, either with their own hardware or using Azure Machine Learning service. The pretrain recipe in this repo includes the dataset and preprocessing scripts so anyone can experiment with building their own general purpose language representation models beyond BERT. Overall this is a stable, predictable recipe that converges to a good optimum for researchers to try explorations on their own.

###### Implementation 
The pretraining recipe in this repo is based on the [PyTorch Pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT) package from [Hugging Face](https://huggingface.co/). The implementation in this pretraining recipe includes optimization techniques such as `gradient accumulation` (gradients are accumulated for smaller mini-batches before updating model weights) and [`mixed precision training`](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html). The notebook and python modules for pretraining are available at [pretrain](./pretrain/) directory.

###### Data Preprocessing
Data preparation is one of the important steps in any Machine Learning project. For BERT pretraining, document-level corpus is needed. The quality of the data used for pretraining directly impacts the quality of the trained models. To make the data preprocessing easier and for repeatability of results, data preprocessing code is included in the repo. It may be used to pre-process Wikipedia corpus or other datasets for pretraining. Refer to additional information at [data preparation for pretraining](docs/dataprep.md) for details on that.

### Finetune
The finetuning recipe in this repo shows how to finetune the BERT language representation model using Azure Machine Learning service. The notebooks and python modules for finetuning are available at [finetune](./finetune/) directory. We finetune and evaluate our pretrained checkpoints against the following:

###### GLUE benchmark
The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems. The [BERT_Eval_GLUE.ipynb](./finetune/PyTorch/notebooks/BERT_Eval_GLUE.ipynb) jupyter notebook allows the user to run one of the pretrained checkpoints against these tasks on Azure ML.

## Azure Machine Learning service
[Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) provides a cloud-based environment to prep data, train, test, deploy, manage, and track machine learning models. This service fully supports open-source technologies such as PyTorch, TensorFlow, and scikit-learn and can be used for any kind of machine learning, from classical ML to deep learning, supervised and unsupervised learning.

#### Notebooks
Jupyter notebooks can be used to use AzureML Python SDK and submit pretrain and finetune jobs. This repo contains the following notebooks for different activities.

###### PyTorch Notebooks
|Activity |Notebook |
|:---|:------|
|Pretrain | [BERT_Pretrain.ipynb](./pretrain/PyTorch/notebooks/BERT_Pretrain.ipynb) |
| [GLUE](https://www.nyu.edu/projects/bowman/glue.pdf) finetune/evaluate | [BERT_Eval_GLUE.ipynb](./finetune/PyTorch/notebooks/BERT_Eval_GLUE.ipynb) |

###### TensorFlow Notebooks
|Activity |Notebook |
|:---|:------|
| [GLUE](https://www.nyu.edu/projects/bowman/glue.pdf) finetune/evaluate | [Tensorflow-BERT-AzureML.ipynb](finetune/TensorFlow/notebooks/Tensorflow-BERT-AzureML.ipynb) |


## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

