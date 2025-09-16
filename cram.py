import os

crams= []

def count_cram_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cram"):
                count += 1
                crams.append(file)
                # print(crams)
    return count

directory = "cram"
cram_count = count_cram_files(directory)
print(f"Found {cram_count} .cram files in the directory '{directory}'.")
with open("/home/jiangjiaqiao/ok.txt",'w') as f:
    for c in crams:
        f.write(c+'\n')
f.close()
