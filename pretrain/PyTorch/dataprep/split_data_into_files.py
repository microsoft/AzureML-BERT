from tqdm import tqdm

input_file = "wikipedia.segmented.nltk.txt"
output_file = "./data_shards/wikipedia.segmented.part."

doc_seperator = "\n"
total_partitions = 100  # Mostly will create 1 extra partition
# shard_size = 396000 # Approximate, will split at next article break

with open(input_file, encoding="UTF-8") as ifile:
    ifile_lines = sum(1 for _ in tqdm(ifile))

print("Input file contains", ifile_lines, "lines.")

shard_size = ifile_lines // total_partitions

with open(input_file, encoding="UTF-8") as ifile:
    shard_line_counter = 0
    shard_index = 0
    ofile = open(f"{output_file}{shard_index}.txt", "w", encoding="UTF-8")  # Open the first file
    # Output files should not have doc_separator at the end of the file, but we accept input ending with doc_separator
    for iline_counter, line in tqdm(enumerate(ifile, start=1)):
        if line != doc_seperator or shard_line_counter < shard_size:
            shard_line_counter += 1
            ofile.write(line)
        # Prevent opening an empty output file or writing a doc_sep
        # when the iteration has reached the end of the input file (iline_counter == ifile_lines)
        elif iline_counter < ifile_lines:
            shard_line_counter = 0
            shard_index += 1
            ofile.close()
            ofile = open(f"{output_file}{shard_index}.txt", "w", encoding="UTF-8")
    ofile.close()  # Close the lastfile
