# Data Preparation for BERT Pretraining
The following steps are to prepare Wikipedia corpus for pretraining. However, these steps can be used with little or no modification to preprocess other datasets as well:

1. Download wiki dump file from https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2.  
   This is a zip file and needs to be unzipped.
2. Clone [Wikiextractor](https://github.com/attardi/wikiextractor), and run it:
   ```
   git clone https://github.com/attardi/wikiextractor
   python3 wikiextractor/WikiExtractor.py -o out -b 1000M enwiki-latest-pages-articles.xml
   ```
   Running time can be 5-10 minutes/GB.  
   _output:_ `out` directory
3. Run:
   ```
   ln -s out out2
   python3 AzureML-BERT/pretrain/PyTorch/dataprep/single_line_doc_file_creation.py
   ```
   This script removes html tags and empty lines and outputs to one file where each line is a paragraph.  
   (`pip install tqdm` if needed.)  
    _output:_ `wikipedia.txt`
4. Run:
   ```
   python3 AzureML-BERT/pretrain/PyTorch/dataprep/sentence_segmentation.py wikipedia.txt wikipedia.segmented.nltk.txt
   ```
   This script converts `wikipedia.txt` to one file where each line is a sentence.  
   (`pip install nltk` if needed.)  
    _output:_ `wikipedia.segmented.nltk.txt`
5. Split the above output file into ~100 files by line with:
   ```
   mkdir data_shards
   python3 AzureML-BERT/pretrain/PyTorch/dataprep/split_data_into_files.py
   ```
   _output:_ `data_shards` directory
6. Run:
   ```
   python3 AzureML-BERT/pretrain/PyTorch/dataprep/create_pretraining.py --input_dir=data_shards --output_dir=pickled_pretrain_data --do_lower_case=true
   ```
   This script will convert each file into pickled `.bin` file.  
   _output:_ `pickled_pretrain_data` directory

