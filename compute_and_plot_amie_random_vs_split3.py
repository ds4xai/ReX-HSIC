import numpy as np
import pandas as pd
import os, json, torch
import matplotlib.pyplot as plt

from models.get_model import get_model
from utils.utils import load_dataset, get_device
from utils.xai_utils import bootstrap_amie, compute_amie, get_test_loader, plot_amie_comparison





if __name__=="__main__":
    
    device = get_device(0)
    # config
    value_over_compute_amies=["proba"]#, "logit"]
    mask_sizes = [1, 3, 5]
    intervention_types = ["ablation"]#, "permutation"]
    dataset_names = ["ip", "pu", "sa"]
    split_stramiegies = ["random", "split3"]
    model_names = [("dsformer", 11), ("ssrn", 7), ("hamidaetal", 5), ("vit_patchwise",7)]#, "spectralformer_patchwise"] 
    
    
    for value_over_compute_amie in value_over_compute_amies:
        for dataset_name in dataset_names:
            
            for intervention_type in intervention_types:
            
                columns = pd.MultiIndex.from_product(
                                        [
                                            model_names,
                                            mask_sizes,
                                            split_stramiegies, 
                                        ],
                names=["model", "mask_size", "split_strategy"]
                )

                _, _, labels, _ = load_dataset(dataset_name,"./datasets/", standardize_image=False)
                metrics = ["AMIE"]
                index = metrics + labels

                data = np.zeros((len(index), len(model_names)*len(mask_sizes)*len(split_stramiegies)))  
                df = pd.DataFrame(data, index=index, columns=columns)
                df = df.astype(object)
                df.head(3)
                
                
                for (model_name, patch_size) in model_names:
                    data_model={}
                    for split_strategy in split_stramiegies:
                        data_model[split_strategy]={
                            "means":[], 
                            "ci": [] } #ci : confidence interval: 
                        
                        log_dir=os.path.join("runs", dataset_name, model_name, str(patch_size))
                
                        # Get best model config
                        path = os.path.join(log_dir, "best_overall_config.json")
                        with open(path) as f:
                            config = json.load(f)
                        
                        model, criterion, optimizer, scheduler, config = get_model(**config)
                        
                        # Load checkpoint
                        best_chkpt_file = config["best_checkpoint"]
                        state_dict = torch.load(best_chkpt_file, map_location=device, weights_only=False)['model_state_dict']
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove 'module.' prefix for DDP
                        print(config["model_name"], " : ", model.load_state_dict(state_dict))
                
                        # Get test loader 
                        config.update({"batch_size": 1024})
                
                        image, gt, labels, _ = load_dataset(config["dataset_name"], "./datasets/", standardize_image=True)
                        test_loader, test_set, baseline, config = get_test_loader(config["dataset_name"], config, "./datasets")        
                        
                        model.eval()
                        for mask_size in mask_sizes:
                            _, _, pb, pi, gt = compute_amie(model, test_loader, config["patch_size"], mask_size,
                                                    baseline, len(labels), intervention_type=intervention_type,
                                                    test_dataset=test_set, device=device, return_mode=value_over_compute_amie)
                            lower, upper, mean_amie = bootstrap_amie(pb, pi)
                            data_model[split_strategy]["means"].append(mean_amie)
                            data_model[split_strategy]["ci"].append((lower, upper))
                        
                    plot_amie_comparison(
                        dataset_name,
                        model_name,
                        random_amie=data_model["random"]["means"],
                        random_ci=data_model["random"]["ci"],
                        disjoint_amie=data_model["split3"]["means"],
                        disjoint_ci=data_model["split3"]["ci"],
                        levels=["TROI1", "TROI2", "TROI3"],
                        save_dir=f"images/{dataset_name}/amie/{patch_size}/{value_over_compute_amie}",
                        file_name=f"{value_over_compute_amie}_{dataset_name}_{model_name}_amie_comparison.png",
                        show_plot=False 
                        )