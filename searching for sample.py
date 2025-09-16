import pandas as pd
import os
from joblib import Parallel, delayed

sample = "HG00418.bed"

child_bed = "bed/child/" + sample
father_path = "bed/father/"
mother_path = "bed/mother/"

output_file = sample+"_positive_samples.txt"

#Read offspring data
child_data = pd.read_csv(child_bed, sep='\t', header=None, names=["chr", "start", "end"])

#Pre-load parental file data
father_dataframes = {}
father_list = os.listdir(father_path)
for father in father_list:
    father_dataframes[father] = pd.read_csv(
        os.path.join(father_path, father), sep='\t', header=None, names=["chr", "start", "end"]
    )
mother_dataframes = {}
mother_list = os.listdir(mother_path)
for mother in mother_list:
    mother_dataframes[mother] = pd.read_csv(
        os.path.join(mother_path, mother), sep='\t', header=None, names=["chr", "start", "end"]
    )

#Check for overlap
def has_overlap(parent_data, chr_query, start_query, end_query):
    #Filter data for the corresponding chromosome
    filtered = parent_data[parent_data['chr'] == chr_query]
    if filtered.empty:
        return False

    #Convert the range to an IntervalIndex
    intervals = pd.IntervalIndex.from_tuples(list(zip(filtered['start'], filtered['end'])), closed="both")
    query_interval = pd.Interval(start_query, end_query, closed="both")

    return intervals.overlaps(query_interval).any()


def process_parent_file(parent, parent_data, chr_query, start_query, end_query):
    if not has_overlap(parent_data, chr_query, start_query, end_query):
        return parent
    return None

results = []

for i in range(child_data.shape[0]):
    chr_query = child_data.iloc[i, 0]
    start_query = child_data.iloc[i, 1]
    end_query = child_data.iloc[i, 2]

    father_result = Parallel(n_jobs=-1)(delayed(process_parent_file)(
        father, father_dataframes[father], chr_query, start_query, end_query
    ) for father in father_list)
    father_result = [f for f in father_result if f]
    # print(father_result)

    mother_result = Parallel(n_jobs=-1)(delayed(process_parent_file)(
        mother, mother_dataframes[mother], chr_query, start_query, end_query
    ) for mother in mother_list)
    mother_result = [m for m in mother_result if m]

    results.append(f"{chr_query}\t{start_query}\t{end_query}\t{','.join(father_result)}\t{','.join(mother_result)}")

with open(output_file, "w") as f:
    f.write("chr\tstart\tend\tfather_result\tmother_result\n")  
    f.write("\n".join(results))  

print(f"The results have been saved to {output_file}.")



# from shutil import copyfile
# import os
# relationship = "mother"  
# source_path = "DEL_BED_BREAKPOINT/"
# tager_path = "bed/" + relationship + "/"
# os.makedirs(tager_path, exist_ok=True)
#
# ok_family_list_path = "ok_family_list.txt"
# with open(ok_family_list_path, "r") as f:
#     ok_family_list = f.readlines()
#     for line in ok_family_list:
#         line = line.replace("\t", " ").replace("\n", " ").split()
#         # print(line[1],relationship)
#         try:
#             if line[1] == relationship:
#                 copyfile(source_path + line[0]+".bed", tager_path + line[0]+".bed")
#         except:
#             print(line)
#
#
# f.close()