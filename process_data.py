import pandas as pd
from sklearn.model_selection import train_test_split
import json
from fuzzywuzzy import fuzz
from sacremoses import MosesTokenizer
import logging

mt = MosesTokenizer()

def label_tokens(row, entity_types=["POI", "street"]):
    len_tokens = {x: len(row[f"{x}_tokens"]) for x in entity_types}
    all_tokens = len(row["tokens"])
    label = ["O"] * all_tokens
    
    for x in entity_types:
        max_score = 0
        max_i = 0
        
        if len_tokens[x] == 0:
            continue
            
        for i in range(all_tokens - len_tokens[x]):
            score = fuzz.ratio(row["tokens"][i:i+len_tokens[x]], row[f"{x}_tokens"])
            
            if score == 100:
                max_i = i
                break
            if score > max_score:
                max_score = score
                
        label[max_i:max_i+len_tokens[x]] = [x] * len_tokens[x]
        
    return label

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("Start preparing data")

    df = pd.read_csv("train.csv")
    logging.info("Tokenizing raw address")
    df["tokens"] = df["raw_address"].apply(mt.tokenize)
    logging.info("Finish tokenizing raw address")

    logging.info("Splitting POI and street")
    df[["POI", "street"]] = df["POI/street"].str.split("/").to_list()
    logging.info("Finish splitting POI and street")

    for ent_type in ["POI", "street"]:
        logging.info(f"Tokenizing {ent_type}")
        df[f"{ent_type}_tokens"] = df[ent_type].apply(mt.tokenize)
        logging.info(f"Finish tokenizing {ent_type}")

    logging.info("Labelling the tokens")
    df["label"] = df.apply(lambda x: label_tokens(x), axis=1)
    logging.info("Finish labelling the tokens")

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=0)
    df_train[["id", "tokens", "label"]].to_json("train_processed.json")
    logging.info("Exported train set")
    df_val[["id", "tokens", "label"]].to_json("val_processed.json")
    logging.info("Exported validation set")

if __name__ == "__main__":
    main()