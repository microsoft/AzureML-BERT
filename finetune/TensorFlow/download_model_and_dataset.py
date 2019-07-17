from __future__ import print_function
import argparse
import sys
import os
import shutil
import zipfile
import urllib

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--bert_model_name",
                    default = None,
                    type = str,
                    required = True,
                    help = "Name of pretrained BERT model. Possible values: "
                           "uncased_L-12_H-768_A-12,uncased_L-24_H-1024_A-16,cased_L-12_H-768_A-12,"
                           "multilingual_L-12_H-768_A-12,chinese_L-12_H-768_A-12")

parser.add_argument("--model_dump_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path to the output model.")

parser.add_argument("--glue_data_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path to store downloaded GLUE dataset")

args = parser.parse_args()

bert_model_url_map = {
    'uncased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip',
    'uncased_L-24_H-1024_A-16': 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip',
    'cased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip',
    'multilingual_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip',
    'chinese_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip'
}

if args.bert_model_name not in bert_model_url_map:
    sys.stderr.write('Unknown BERT model name ' + args.bert_model_name)
    sys.exit(1)

pretrained_model_url = bert_model_url_map.get(args.bert_model_name)

# make local directory for pretrained tensorflow BERT model
tensorflow_model_dir = './tensorflow_model'
if not os.path.exists(tensorflow_model_dir):
    os.makedirs(tensorflow_model_dir)

# download and extract pretrained tensorflow BERT model
download_file_name = 'tensorflow_model.zip'
urllib.request.urlretrieve(pretrained_model_url, filename=download_file_name)
print('Extracting pretrained model...')
with zipfile.ZipFile(download_file_name, 'r') as z:
    z.extractall(tensorflow_model_dir)

# make destination path
if not os.path.exists(args.model_dump_path):
    os.makedirs(args.model_dump_path)

files = ['bert_model.ckpt.meta', 'bert_model.ckpt.index', 'bert_model.ckpt.data-00000-of-00001', 'bert_config.json', 'vocab.txt']
for file in files:
    shutil.copy(os.path.join(tensorflow_model_dir, args.bert_model_name, file), os.path.join(args.model_dump_path, file))

print('Start to download GLUE dataset...\n')
urllib.request.urlretrieve(
    'https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py',
    filename='download_glue_data.py')
if os.system('python download_glue_data.py --data_dir {0} --tasks all'.format(args.glue_data_path)) != 0: sys.exit(1)