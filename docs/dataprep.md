# Data Preparation for BERT Pretraining
The following steps are to prepare Wikipedia corpus for pretraining. However, these steps can be used with little or no modification to preprocess other datasets as well:

1. Download wiki dump file from https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2.
  - This is a zip file and needs to be unzipped.
2. Clone [Wikiextractor](https://github.com/attardi/wikiextractor), and run
`python wikiextractor/WikiExtractor.py -o /out -b 1000M enwiki-latest-pages-articles.xml`.
3. Run `python single_line_doc_file_creation.py`. This script removes html tags and empty lines and outputs to one file where each line is a paragraph.
4. Run `python sentence_segmentation.py <input_file> <output_file>`. This script converts <input_file> to a file where each line is a sentence.
5. Split the above output file into ~1000 files by line with `split <input_file> -l 1113000 -d <output_file_prefix>.
6. From current folder (/pytorch/pretrian/dataprep), run `python create_pretraining.py --input_dir=<input_directory> --output_dir=<output_directory> --do_lower_case=true` which will convert each file into pickled .bin file.
