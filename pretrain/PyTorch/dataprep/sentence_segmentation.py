import nltk
import os
from tqdm import tqdm
import sys

nltk.download('punkt')

input_file = sys.argv[1]
output_file = sys.argv[2]

doc_seperator = "\n"

with open(input_file) as ifile:
    with open(output_file, "w") as ofile:
        for i, line in tqdm(enumerate(ifile)):
            if line != "\n":
                sent_list = nltk.tokenize.sent_tokenize(line)
                for sent in sent_list:
                    ofile.write(sent + "\n")
                ofile.write(doc_seperator)
