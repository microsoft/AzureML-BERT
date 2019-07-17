import glob
import os
from tqdm import tqdm

output_file = 'wikipedia.txt'

with open(output_file, "w") as ofile:
  for dirname in glob.glob('out2/*/', recursive=False):
    for filename in glob.glob(dirname + 'wiki_*', recursive=True):
      print(filename)
      article_lines = []
      article_open = False
      
      with open(filename, "r") as file:
        for i, line in tqdm(enumerate(file)):
          if "<doc id=" in line:
            article_open = True
          elif "</doc>" in line:
            article_open = False
            for oline in article_lines[1:]:
              if oline != "\n":
                ofile.write(oline.rstrip() + " ")
            ofile.write("\n\n")
            article_lines = []
          else:
            if article_open:
              article_lines.append(line)
