import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Generate a clean tsv file with numeric labels for a 4 class problem')
parser.add_argument('tsv', metavar='FILE', help='Tsv file generated from clinica iotools merge pipeline')
args = parser.parse_args()

master = pd.read_csv(args.tsv, sep='\t', usecols=['participant_id','session_id','diagnosis_bl','diagnosis', 'age'])
master = master.sort_values(by=['participant_id','session_id'])
master.dropna(subset=['diagnosis'], inplace=True)
master.reset_index(drop=True, inplace=True)

threshold = 3.0 #36 months equals 3 years can adjust to something else
for i, row in enumerate(master.itertuples()):
    
    if row.diagnosis == 'MCI':
        temp = master.loc[master['participant_id'] == row.participant_id]
        temp = temp.loc[temp['age'] <= row.age + 3]

        print(i, temp)
        if temp['diagnosis'].isin(['AD']).any():
            master.at[i, 'diagnosis'] = 'cMCI'
        else:
            master.at[i, 'diagnosis'] = 'nMCI'

if master.to_csv('./labels.tsv', sep='\t', index=False):
    print("File saved successfully!")
