import argparse
import pandas as pd
import glob

parser = argparse.ArgumentParser(description='Create participant file for only both modes')
parser.add_argument('-m','--missing', metavar='DIR', help='path to missing_modes dir')

args = parser.parse_args()

file_paths = [f for f in glob.glob(args.missing + '*.tsv')]


master_list = pd.DataFrame(columns=['participant_id','session_id'])

#print(file_paths)
#print(master_list)

for file in file_paths:
    
    session = file.replace(args.missing + 'missing_mods_', '').replace('.tsv', '')

    temp = pd.read_csv(file, sep='\t')
    temp = temp.loc[ (temp['pet'] == 1) & (temp['t1w'] == 1) ]
    temp['session_id'] = session
    temp = temp[['participant_id', 'session_id']]
    
    #print(temp)
    master_list = pd.concat([master_list, temp], ignore_index=True)

master_list = master_list.sort_values(by=['participant_id'])
#print(master_list)
master_list.to_csv('participants.tsv', sep='\t')
