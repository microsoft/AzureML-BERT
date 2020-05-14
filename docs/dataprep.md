# Data Preparation for BERT Pretraining
The following steps are to prepare Wikipedia corpus for pretraining. However, these steps can be used with little or no modification to preprocess other datasets as well:

1. Download wiki dump file from https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2.
  - This is a zip file and needs to be unzipped.
2. Clone [Wikiextractor](https://github.com/attardi/wikiextractor), and run
`python wikiextractor/WikiExtractor.py -o /out -b 1000M enwiki-latest-pages-articles.xml`.
3. Run `python single_line_doc_file_creation.py`. This script removes html tags and empty lines and outputs to one file where each line is a paragraph.
4. Run `python sentence_segmentation.py <input_file> <output_file>`. This script converts <input_file> to one file where each line is a sentence.
5. Split the above output file into ~100 files by line with `python split_data_into_files.py`.
6. From current folder (/pytorch/pretrian/dataprep), run `python create_pretraining.py --input_dir=<input_directory> --output_dir=<output_directory> --do_lower_case=true` which will convert each file into pickled .bin file.

---


# M$ leírás tesztje

[[-> talk page]](https://git.nlp.nytud.hu/BERT/bert_corpus_META/issues/7)

---

Megnéztem, kipróbáltam a M$ leírást.\
https://github.com/microsoft/AzureML-BERT/blob/master/docs/dataprep.md

A magyar wikipedia-t szedtem le (angol: 17G, magyar: 834M)\
A pontokon megyek végig:

1. meg kell szerezni az adatot :)
2. wikiextractor _(nekünk nyilván nem szükséges)_\
futási idő: 5 perc (a 834M-ra)\
itt vannak a scriptek: `AzureML-BERT/pretrain/PyTorch/dataprep`\
`/out` helyett `out` ajánlott...\
_kimenet:_ `out` könyvtár
3. single... = _1 fájlba, doc-ok üres sorral elválasztva = 1 dok 1 "bekezdés"_ \
install infó nincs, nekem kellett: `pip install tqdm`\
`$ ln -s out out2`\
`$ python3 AzureML-BERT/pretrain/PyTorch/dataprep/single_line_doc_file_creation.py`\
_kimenet:_ `wikipedia.txt`
4. sentseg = _to spl_\
install infó nincs, nekem kellett: `pip install nltk`\
`$ python3 AzureML-BERT/pretrain/PyTorch/dataprep/sentence_segmentation.py wikipedia.txt wikipedia.segmented.nltk.txt`\
'punkt'-ot használ... -> helyette nekünk: __emToken__\
_kimenet:_ `wikipedia.segmented.nltk.txt`
5. split = _egy file helyett sok kisebb_\
`$ mkdir data_shards`\
`$ python3 AzureML-BERT/pretrain/PyTorch/dataprep/split_data_into_files.py`\
_kimenet:_ `data_shards` könyvtár
6. convert to pickled bin = _nem csak convert, hanem BertTokenizer, meg ki tudja még mi_\
`$ python3 AzureML-BERT/pretrain/PyTorch/dataprep/create_pretraining.py --input_dir=data_shards --output_dir=pickled_pretrain_data --do_lower_case=true`\
__`--do_lower_case=true` kell nekünk?__\
_kimenet elvileg:_ `pickled_pretrain_data` könyvtár

__Az utsó lépésnél akadtam le, ennél a hibaüzenetnél:__
```bash
    from pytorch_pretrained_bert.tokenization import BertTokenizer
ModuleNotFoundError: No module named 'pytorch_pretrained_bert'
```

---

[<- főoldal](README.md)

