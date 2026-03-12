import os, json
import numpy as np
import pandas as pd

from utils.utils import load_dataset


if __name__=="__main__":
    
    dataset_names = ["ip", "pu", "sa"]
    split_strategies = ["random", "split3"]
    model_names = [("dsformer", 11), ("ssrn", 7), ("hamidaetal", 5), ("vit",7), ("spectralformer", 7)]  
    
    
    for dataset_name in dataset_names:
        
        columns = pd.MultiIndex.from_product(
                [
                    model_names,
                    split_strategies, 
                ],
                names=["model", "split_strategy"]
            )

        _, _, labels, _ = load_dataset(dataset_name, "./datasets/", standardize_image=False)
        metrics = ["OA", "AA", "Kappa", "F1 Macro"]
        index = metrics + labels

        data = np.zeros((len(index), len(model_names)*len(split_strategies)))  
        df = pd.DataFrame(data, index=index, columns=columns)
        df = df.astype(object)
        
        
        for model_name, ps in model_names:
    
            for split_strategy in split_strategies:
     
                log_dir=os.path.join("runs", dataset_name, model_name, str(ps), split_strategy)
                
                # Get best model config
                path = os.path.join(log_dir, "aggregate_results.json")
                with open(path) as f:
                    data = json.load(f)
                    
                df.loc["OA", (model_name, split_strategy)] = data["OA"]
                df.loc["AA", (model_name, split_strategy)] = data["AA"]
                df.loc["F1 Macro", (model_name, split_strategy)] = data["F1 Macro"]
                df.loc["Kappa", (model_name, split_strategy)] = data["Kappa"]
                for i, classe in enumerate(labels):
                        df.loc[classe, (model_name, split_strategy)] = data["f1 scores"][i]
        
        
        # Save
        save_path = f"runs/{dataset_name}/{dataset_name}_models_perfs.csv"
        df.to_csv(save_path, header=True, index=True) 