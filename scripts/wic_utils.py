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


def get_prompt(lemma: str, sentence1: str, sentence2: str, no_think: bool = False) -> str:
    prompt_lines = [
        "You are an expert in semantics and NLP, especially in judging word meaning.",
        f"Below are two sentences containing the word '{lemma}'.",
        f"Sentence 1: {sentence1}",
        f"Sentence 2: {sentence2}",
        f"If the word '{lemma}' has the same meaning in both of those sentences, respond 'same'.",
        f"If the word '{lemma}' has a different meaning in each of those sentences, respond 'different'.",
        f"Provide only that single word as your response, nothing else.",
    ]
    if no_think:
        prompt_lines.insert(0, '/no_think')
    return '\n'.join(prompt_lines)


def wic_df_to_prompt_df(wic_df: pd.DataFrame, no_think: bool = False):
    columns = {
        'prompt':[],
        'response':[],
    }
    
    for _, row in wic_df.iterrows():
        prompt = get_prompt(row['lemma'], row['sentence1'], row['sentence2'], no_think)
        response = 'same' if row['gold'] else 'different'
        columns['prompt'].append(prompt)
        columns['response'].append(response)

    prompt_df = pd.DataFrame(columns)
    return(prompt_df)



    
if __name__ == '__main__':
    df = read_data_owic('owic/dev/dev.data.txt')
    df['gold'] = read_gold_owic('owic/dev/dev.gold.txt')
    print(df.head(10))
