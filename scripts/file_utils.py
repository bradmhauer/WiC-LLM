import pandas as pd


def read_data_owic(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename,
                     sep='\t',
                     names=['lemma','POS','indices','sentence1','sentence2'])
    df[['index_sentence1','index_sentence2']] = df['indices'].str.split('-', expand=True)
    df = df[['lemma','POS','index_sentence1','index_sentence2','sentence1','sentence2']]

    return(df)


def read_gold_owic(filename: str) -> pd.DataFrame:
    # Read the file with 'T' or 'F' values
    with open(filename, 'r') as f:
        bool_values = [line.strip() == 'T' for line in f]

    # Add the boolean column to the dataframe
    return(pd.Series(bool_values))

    
if __name__ == '__main__':
    df = read_data_owic('owic/dev/dev.data.txt')
    df['gold'] = read_gold_owic('owic/dev/dev.gold.txt')
    print(df.head(10))
