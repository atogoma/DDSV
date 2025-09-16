import pandas as pd
import os
import glob
import hashlib

def calculate_file_md5(file_path):
    """
    Calculate the MD5 hash of the specified file.

    :param file_path: Path to the file
    :return: MD5 hash as a hexadecimal string
    """
    import hashlib

    #Create an MD5 hash object
    md5_hash = hashlib.md5()

    #Open the file in binary mode
    with open(file_path, "rb") as f:
        #Read the file in chunks
        for chunk in iter(lambda: f.read(4096), b""):  #4096 bytes per chunk
            md5_hash.update(chunk)

    #Return the hexadecimal MD5 string
    return md5_hash.hexdigest()

path = "cram/"
rt_path = "rt/"
xls_path = "600trios.xlsx"

xls_df = pd.read_excel(xls_path, sheet_name="MD5") #SAMPLE_NAME #ENA_FILE_PATH MD5SUM
# family_df = pd.read_excel(xls_path, sheet_name="20130606_g1k_3202_samples_ped_p") #FamilyID	SampleID	FatherID	MotherID	Sex	Population	Superpopulation

cram_list = glob.glob(path+"*.cram")  
# sample_names = [os.path.basename(f).split('.')[0] for f in cram_list] 

error_list = []
ok_list = []
for cram in cram_list:
    md5 = calculate_file_md5(cram)
    sample_name = os.path.basename(cram).split('.')[0]
    row = xls_df[xls_df['SAMPLE_NAME'] == sample_name]
    if not row.empty:
        right_md5 = row.iloc[0]['MD5SUM']
        if right_md5 == md5:
            ok_list.append(sample_name+"\n")
            continue
        else:
            error_list.append(sample_name+"\n")

with open( rt_path + 'error.txt', 'w') as f:
    f.writelines(error_list)
f.close()

with open( rt_path + 'ok.txt', 'w') as f_ok:
    f_ok.writelines(ok_list)
f_ok.close()


# import os
# import re
# import pandas as pd
# #sample_name relationship sex
# #HG01234    child/father/mother   1/2(male/female)
#
# path = "data/data/star/ok.txt"
# cram_path = "data/data/star/ok_family_list.txt"
# xls_path = "data/data/star/600trios.xlsx"
# family_df = pd.read_excel(xls_path, sheet_name="20130606_g1k_3202_samples_ped_p") #FamilyID	SampleID	FatherID	MotherID	Sex	Population	Superpopulation
#
# out_list = []
# with open(path, "r") as f, open(cram_path, "w") as cram:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         row = family_df[family_df['SampleID'] == line]
#         if not row.empty:
#             sex = 'male' if row.iloc[0]['Sex'] == 1 else 'female'
#             if row.iloc[0]['FatherID']==0:
#                 str = line+"\t"+"child" + "\t" +sex + "\n"
#             else:
#                 if sex=='male':
#                     str = line+"\t"+"father" + "\t" +sex + "\n"
#                 else:
#                     str = line+"\t"+"mother" + "\t" +sex + "\n"
#
#             out_list.append(str)
#     cram.writelines(out_list)
#
# f.close()
# cram.close()

#Generate positive-set images  
#Create one image per sample at its corresponding locus, then combine them
