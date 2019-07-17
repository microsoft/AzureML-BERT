## **Natural Language Processing**

In the natural language processing (NLP) domain, pre-trained language representations have traditionally been a key topic for a few important use cases, such as [named entity recognition](https://arxiv.org/pdf/cs/0306050.pdf) (Sang and Meulder, 2003), [question answering](https://arxiv.org/pdf/1606.05250.pdf) (Rajpurkar et al., 2016), and [syntactic parsing](https://nlp.stanford.edu/~mcclosky/papers/dmcc-naacl-2010.pdf) (McClosky et al., 2010).

The intuition for utilizing a pre-trained model is simple: A deep neural network that is trained on large corpus, say the entire Wikipedia dataset, should have enough knowledge about the underlying relationships between different words and sentences. One should then be able to adapt this DNN to be used on a different corpus, such as medical documents or financial documents, resulting in a model with better performance than one could obtain by training purely on the specialized corpus.

Recently, a paper called &quot;[BERT: Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805)&quot; was published by Devlin et al., which achieves new state-of-the-art results on 11 NLP tasks, using the pre-trained approach mentioned above. In this repo, we want to show how customers can efficiently and easily pretrain and then fine-tune BERT for their custom applications using Azure Machine Learning Services. We open sourced the code on [GitHub](https://github.com/Microsoft/AzureML-BERT).

## **Intuition behind BERT**

The intuition behind the new language model, BERT, is simple yet powerful. Researchers believe that a large enough deep neural network model, with large enough training corpus, should capture the contextual relations in the corpus. In NLP domain, it is hard to get a large annotated corpus, so researchers used a novel technique to get a lot of training data. Instead of having human beings label the corpus and feed it into neural networks, researchers use the large Internet available corpus such as English Wikipedia with 2,500M words. Two approaches, each for different language tasks, are used to generate the labels for the language model.

- **Masked language model:** To understand the relationship between words. The key idea is to mask some of the words in the sentence (around 15 percent) and use those masked words as labels to force the models to learn the relationship between words. For example, the original sentence would be:

```
The man went to the store. He bought a gallon of milk.
```

And the input/label pair to the language model is:

```
Input: The man went to the [MASK1]. He bought a [MASK2] of milk.

Labels: [MASK1] = store; [MASK2] = gallon
```

- **Sentence prediction task:**  To understand the relationship between sentences. This task helps the model predict whether sentence B is likely to be the next sentence following a given sentence A. Using the same example above, we can generate training data like:

```
Sentence A: The man went to the store.

Sentence B: He bought a gallon of milk.

Label: IsNextSentence
```

## **Applying BERT to customized dataset**

After BERT is trained on a large corpus (say all the available English Wikipedia) using the above steps, the assumption is that because the dataset is huge, the model can inherit a lot of knowledge about the English language. The next step is to fine-tune the model on different tasks, hoping the model can adapt to a new domain more quickly. The key idea is to use the large BERT model trained above and add different input/output layers for different types of tasks. For example, you might want to do sentiment analysis for a customer support department. This is a classification problem, so you might need to add an output classification layer (as shown on the left in the figure below) and structure your input. For a different task, say question answering, you might need to use a different input/output layer, where the input is the question and the corresponding paragraph, while the output is the start/end answer span for the question (see the figure on the right). In each case, the way BERT is designed, it can enable data scientists to plug in different layers easily so it can be adapted for different tasks.

![Adapting BERT for different tasks](https://azurecomcdn.azureedge.net/mediahandler/acomblog/media/Default/blog/39717ecf-8274-46c4-862d-21ca377b1957.png)

_Figure 1. Adapting BERT for different tasks (_[_Source_](https://arxiv.org/pdf/1810.04805.pdf)_)_

The image below shows the results on one of the most popular datasets in NLP field, the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/).

![Reported BERT performance on SQuAD 1.1 dataset](https://azurecomcdn.azureedge.net/mediahandler/acomblog/media/Default/blog/c37ee936-a5d2-4878-b8e2-ffc02a2797f2.png)

_Figure 2. Reported BERT performance on SQuAD 1.1 dataset (_[_Source_](https://arxiv.org/pdf/1810.04805.pdf)_)._

In the GitHub repository, we demonstrated the GLUE [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) (Wang et al., 2018) task.
