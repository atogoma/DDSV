import os
import re
import pandas as pd
#Classify .cram files named "ok" into child, father, and mother 
#Output to ok_family_list.txt
#sample_name relationship sex
#HG01234    child/father/mother   1/2(male/female)

path = "ok.txt"
cram_path = "ok_family_list.txt"
xls_path = "600trios.xlsx"
family_df = pd.read_excel(xls_path, sheet_name="20130606_g1k_3202_samples_ped_p") #FamilyID	SampleID	FatherID	MotherID	Sex	Population	Superpopulation

out_list = []
with open(path, "r") as f, open(cram_path, "w") as cram:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        row = family_df[family_df['SampleID'] == line]
        if not row.empty:
            sex = 'male' if row.iloc[0]['Sex'] == 1 else 'female'
            if row.iloc[0]['FatherID']==0:
                str = line+"\t"+"child" + "\t" +sex + "\n"
            else:
                if sex=='male':
                    str = line+"\t"+"father" + "\t" +sex + "\n"
                else:
                    str = line+"\t"+"mother" + "\t" +sex + "\n"

            out_list.append(str)
    cram.writelines(out_list)

f.close()
cram.close()
