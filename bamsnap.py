from os import mkdir
from tabnanny import NannyNag

import pandas as pd
import os
from joblib import Parallel, delayed
from pandas import isnull


#Sample name, edit manually
sample = "HG00418.bed"
sample_name = sample.split(".")[0]
child_bam = sample_name + ".final.cram"

path = sample_name+"_rawpic"
if not os.path.exists(path):
	os.makedirs(path)
	print(f"Directory '{path}' created successfully.")
else:
	print(f"Directory '{path}' already exists.")


def get_cram_files(directory):
	crams= []
	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith(".cram"):crams.append(file)
	return crams

directory = "cram"
crams = get_cram_files(directory)


positive_path = sample+"_positive_samples.txt"

output_file = sample+"_bamsnap.sh"


data = pd.read_csv(positive_path, sep='\t')
# print(data.columns) #['chr', 'start', 'end', 'father_result', 'mother_result']
#print(data.head())  #father_result HG02723.bed,HG02059.bed,HG01084.bed...


orders = []
for i in range(len(data)):
	
	if pd.isna(data.iloc[i, 3]) or pd.isna(data.iloc[i, 4]):continue
	
	try:
		chr = data.iloc[i,0]    #'chr1'
		start = data.iloc[i,1]
		end = data.iloc[i,2]
		father_result = data.iloc[i,3].split(',')
		mother_result = data.iloc[i,4].split(',')
		for f in father_result:
			f =f.split('.')[0] + '.final.cram'
			f_name = f.split('.')[0]
			for m in mother_result:
				m =m.split('.')[0] + '.final.cram'
				m_name = m.split('.')[0]
				if f in crams and m in crams:
					order_template = ("bamsnap "
					                  f"-bam {f} {m} {child_bam} "											
					                  "-title 'father' 'mother' 'child' "
					                  f"-pos {chr}:{start}-{end} "
					                  "-bamplot coverage read "
					                  "-margin 300 "
					                  "-no_target_line "
					                  "-show_soft_clipped "
					                  "-read_color_by interchrom "
					                  "-save_image_only "
					                  f"-out {path}/{sample_name}_{chr}_{start}_{end}_{f_name}_{m_name}.png "
					                  "-ref ref/GRCh38_full_analysis_set_plus_decoy_hla.fa")
					# print(order_template)
					orders.append(order_template)
	except:
		print(data.iloc[i,:])
# break

with open(output_file, 'w') as f:
	for order in orders:
		f.write(order + '\n')

f.close()
