import pandas as pd

def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None, names=['slovak', 'english'], on_bad_lines='skip', quoting=3) 

def prepare_dataset(formal, informal):
    formal = read_tsv(formal)
    informal = read_tsv(informal)

    formal_shuffled = formal.sample(frac=1, random_state=42).reset_index(drop=True)
    informal_shuffled = informal.sample(frac=1, random_state=42).reset_index(drop=True)

    formal_sample = formal_shuffled.sample(n=2500, random_state=42).reset_index(drop=True)
    informal_sample = informal_shuffled.sample(n=2500, random_state=42).reset_index(drop=True)

    formal_sample['label'] = 'formal'
    informal_sample['label'] = 'informal'

    merged_df = pd.concat([formal_sample, informal_sample], ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged_df

if __name__ == "__main__":
    formal = 'datasets/sk-en.formal.tsv'
    informal = 'datasets/sk-en.informal.tsv'
    df = prepare_dataset(formal=formal, informal=informal)
    df.to_csv('datasets/sk-en.merged5000.tsv', sep='\t', header=False, index=False)