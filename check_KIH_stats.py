#this custom python package is largely adapted from Weiyi Tang's original work, with modificaiton from Marco Mravic & Colleen Maillie   
import sys, numpy as np, time, collections
from prody import *
from helix_dimer_geometryV3 import *
from fitting_cylinder import *
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from KIH_packingScore_tools import * 


#python check_KIH_stats designs_dir 
#designs_dir :directory contianing all pdbs to chcek 
designs_dir = sys.argv[1] 

#list to store all design scores 
all_design_evals = []
all_design_ids = [] 
for f in os.listdir(designs_dir):
    if '.pdb' in f:
        design_id = f.split('__')[1].split('.')[0]
        all_design_ids.append(design_id)
        print("DESIGN ID: ", design_id)
        pdb = parsePDB(designs_dir+"/"+f)
        #chain C is rosetta membrane marker - not tolerated in packing score
        pdb = pdb.select('chain A or chain B')

        design_eval = packing_score_2helix(pdb)
        all_design_evals.append(np.array(design_eval))
all_design_evals = np.array(all_design_evals)
total_score_mean = np.mean(all_design_evals[:,1])
print('Mean of total score: ', total_score_mean) 
total_score_table = dict(zip(all_design_ids, all_design_evals[:,1] ))
for k,v in total_score_table.items():
    if v>total_score_mean:
        print("Check out: ", k,v)
print(total_score_table)
