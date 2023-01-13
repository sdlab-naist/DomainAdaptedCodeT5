from cmath import nan
import pandas as pd
from transformers import RobertaTokenizer

from nltk import bleu_score,edit_distance
import typer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

def zip_with_preprocess(masked_code,mask):
    masked_code = list(map(lambda x:x.replace("<x>","<extra_id_0>").replace("\n",""),masked_code))
    mask = list(map(lambda x:x.replace("<z>","").replace("\n",""),mask))
    return zip(masked_code,mask)

def load_dataset(path):
    masked_code_lines = []
    mask_lines = []
    with open(path) as masked_code:
        masked_code_lines = masked_code.readlines()
    with open(path.replace("masked_code","mask")) as mask:
        mask_lines = mask.readlines()
    df_list = [[masked_code,mask] for masked_code,mask in zip_with_preprocess(masked_code_lines,mask_lines)]
    df = pd.DataFrame(df_list,columns=["text","headlines"])
    return df
 
def count_to_complete(generated,actual):
    if type(actual) is not str:
        actual = " "
    tokens = tokenizer.encode(actual)
    
    return len(tokens)-2

callback_count_to_complete = lambda x: count_to_complete(x["Generated Text"], x["Actual Text"])


def bleu(generated,actual):
    weight = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    if type(generated) is not str or type(actual) is not str:
        return  pd.Series([None]*4)
    g_token = tokenizer.encode(generated)[1:-1]
    a_token = tokenizer.encode(actual)[1:-1]
    bleu_n = []
    for i,w in enumerate(weight):
        if i+1 > len(g_token)  :
            bleu_n+=[None]
            continue
        bleu_n += [bleu_score.sentence_bleu([a_token], g_token,w)]
    return pd.Series(bleu_n)

callback_bleu = lambda x: bleu(x["Generated Text"], x["Actual Text"])

def editing_distance(generated,actual):
    g_token ,a_token = [" "],[" "]
    if type(generated) is str:
        g_token = tokenizer.encode(generated)[1:-1]
    if type(actual) is str:
        a_token = tokenizer.encode(actual)[1:-1]
    return edit_distance(g_token,a_token)/max(len(g_token),len(a_token))

callback_editing_distance = lambda x: editing_distance(x["Generated Text"], x["Actual Text"])

def main (path:str):    
    csv_df = pd.read_csv(path)
    
    csv_df = pd.concat([csv_df ,csv_df.apply(callback_count_to_complete, axis=1).rename("#complete")],axis=1)
    bleu_df = csv_df.apply(callback_bleu, axis=1).rename(columns={0:"BLEU1",1:"BLEU2",2:"BLEU3",3:"BLEU4"})
    csv_df = pd.concat([csv_df ,bleu_df],axis=1)
    csv_df = pd.concat([csv_df ,csv_df.apply(callback_editing_distance, axis=1).rename("Editing distance")],axis=1)
    csv_df = csv_df.drop("Unnamed: 0",axis=1)

    csv_df.to_csv(path.replace(".csv","_analyzed.csv"))


if __name__ == "__main__":
    typer.run(main)

