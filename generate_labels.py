import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Generate a clean tsv file with numeric labels for a 4 class problem')
parser.add_argument('tsv', metavar='FILE', help='Tsv file generated from clinica iotools merge pipeline')
args = parser.parse_args()

master = pd.read_csv(args.tsv, sep='\t', usecols=['participant_id','session_id','diagnosis_bl','diagnosis'])
#new_master = pd.DataFrame(columns=['participant_id','session_id','diagnosis'])
#print(new_master)
for i, row in enumerate(master.itertuples()):
    
    if pd.isna(row.diagnosis):
        
        if row.diagnosis_bl == 'LMCI':
            temp = master.loc[master['participant_id'] == row.participant_id]
        
            if temp['diagnosis'].isin(['AD']).any():
                master.at[i, 'diagnosis'] = 'cMCI'
            else:
                master.at[i, 'diagnosis'] = 'nMCI'
        else:
            master.at[i, 'diagnosis'] = row.diagnosis_bl
    
 
    if row.diagnosis == 'MCI' or row.diagnosis == 'LMCI':
        temp = master.loc[master['participant_id'] == row.participant_id]
        #print(i, temp)
        if temp['diagnosis'].isin(['AD']).any():
            master.at[i, 'diagnosis'] = 'cMCI'
        else:
            master.at[i, 'diagnosis'] = 'nMCI'

if master.to_csv('./labels.tsv', sep='\t', index=False):
    print("File saved successfully!")
