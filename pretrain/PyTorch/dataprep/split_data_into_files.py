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
    line_counter = 0
    shard_index = 0
    ofile = open(f"{output_file}{shard_index}.txt", "w", encoding="UTF-8")  # Open the first file
    for line in tqdm(ifile):
        # We take whole documents and write at least shard_size lines into a file
        if line_counter < shard_size or (line_counter >= shard_size and line != doc_seperator):
            ofile.write(line)
            line_counter += 1
        else:
            line_counter = 0
            shard_index += 1
            ofile.close()
            ofile = open(f"{output_file}{shard_index}.txt", "w", encoding="UTF-8")
            ofile.write(line)  # Do not forget to write this line too!
    ofile.close()  # Close the lastfile
