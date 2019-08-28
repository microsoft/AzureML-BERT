import os
from tqdm import tqdm

input_file = 'wikipedia.segmented.nltk.txt'
output_file = './data_shards/wikipedia.segmented.part.'

doc_seperator = "\n"

line_buffer = []
total_partitions = 100  # Mostly will create 1 extra partition
# shard_size = 396000 # Approximate, will split at next article break
line_counter = 0
shard_index = 0

ifile_lines = 0
with open(input_file) as ifile:
    for i, line in tqdm(enumerate(ifile)):
        ifile_lines += 1

print("Input file contains", ifile_lines, "lines.")

shard_size = ifile_lines // total_partitions

iline_counter = 1
with open(input_file) as ifile:
    for i, line in tqdm(enumerate(ifile)):
        if line_counter < shard_size and iline_counter < ifile_lines:
            line_buffer.append(line)
            line_counter += 1
            iline_counter += 1
        elif line_counter >= shard_size and line != "\n" and iline_counter < ifile_lines:
            line_buffer.append(line)
            line_counter += 1
            iline_counter += 1
        else:
            with open(output_file + str(shard_index) + ".txt", "w") as ofile:
                for oline in line_buffer:
                    ofile.write(oline)
                line_buffer = []
                line_counter = 0
                shard_index += 1
