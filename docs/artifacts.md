# Artifacts for pretrain and finetune

The following artifacts are made available to make pretraining and finetuning of BERT models easier:
* Preprocessed data
* Pretrained BERT-base and BERT-large model checkpoints

## Preprocessed Data
The Wikipedia corpus used for BERT pretraining is preprocessed following the [data prep instructions](dataprep.md) and uploaded to  https://bertonazuremlwestus2.blob.core.windows.net/public/bert_data.tar.gz (70 GB). The data files have the sequence length of 512. The directory structure is as follows and this directory hierarchy is assumed in the implementation in [train.py](../pretrain/pytorch/train.py).
```
bert_data
│   bert-base.json
│   bert-large.json
│   bert-base-single-node.json
│   bert-large-single-node.json
│
└───512
│   │
│   └───wiki_pretrain
│       │   wikipedia_segmented_part_0.bin
│       │   wikipedia_segmented_part_1.bin
│       │   ...
│       │   wikipedia_segmented_part_98.bin
```

Individual data files from wiki_pretrain directory are available at the following urls:
* [wikipedia_segmented_part_0.bin](https://bertonazuremlwestus2.blob.core.windows.net/public/data/preprocessed/512/wiki_pretrain/wikipedia_segmented_part_0.bin)
* [wikipedia_segmented_part_1.bin](https://bertonazuremlwestus2.blob.core.windows.net/public/data/preprocessed/512/wiki_pretrain/wikipedia_segmented_part_1.bin)
* [wikipedia_segmented_part_2.bin](https://bertonazuremlwestus2.blob.core.windows.net/public/data/preprocessed/512/wiki_pretrain/wikipedia_segmented_part_2.bin)
* ...
* [wikipedia_segmented_part_98.bin](https://bertonazuremlwestus2.blob.core.windows.net/public/data/preprocessed/512/wiki_pretrain/wikipedia_segmented_part_98.bin)
    

## Pretrained BERT Model Checkpoints
The models pretrained in AzureML based on the original BERT implementation are available at the following locations:
* [BERT-Large, Uncased (original)](https://bertonazuremlwestus2.blob.core.windows.net/public/models/bert_large_uncased_original/bert_encoder_epoch_200.pt)
* [BERT-Base, Uncased (original)](https://bertonazuremlwestus2.blob.core.windows.net/public/models/bert_base_uncased_original/bert_encoder_epoch_0300.pt)
