import pandas as pd
import os
from joblib import Parallel, delayed

#Sample name – edit manually  
sample = "HG00418.bed"

#Path to positive sites  
positive_path = f"{sample}_positive_samples.txt"

#Output shell script  
output_file = f"{sample}_bamsnap.sh"

print("ok")
#Read data
data = pd.read_csv(positive_path, sep='\t')
print(data.columns) #['chr', 'start', 'end', 'father_result', 'mother_result']
#print(data.head())  #father_result HG02723.bed,HG02059.bed,HG01084.bed...

for i in range(len(data)):
	print(i,data.iloc[i,3])
	chr = data.iloc[i,0]    #'chr1'
	start = data.iloc[i,1]
	end = data.iloc[i,2]
	# father_result = data.iloc[i,3].split(',')
	# mother_result = data.iloc[i,4].split(',')
	# print(father_result)
	# print(mother_result)
	break
