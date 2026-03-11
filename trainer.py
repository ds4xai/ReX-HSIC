import json
import torch, gc
from main import training
from models.get_model import get_model
import os, torch, argparse
from dataloader.dataset import PatchedDataset
from dataloader.preprocessing import get_gts, img_preprocessing
from utils.utils import aggregate_results, convert_to_serializable, load_dataset

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_dataloader(dst_name, config, parent_dir):
    
    assert dst_name in ("ip", "pu", "sa"), f"Dataset {dst_name} not supported"
    assert "patch_size" in config, "patch_size must be specified in config"
    assert "batch_size" in config, "batch_size must be specified in config"
    
    
    if config["split_strategy"] == "random_by_class":
        train_size = .3
    else: 
        train_size = .7 if dst_name=="ip" else .6
        
    # spliting random or split3
    train_gt, val_gt, test_gt = get_gts(dst_name, parent_dir, train_ratio=train_size, train_side="right", 
                                        split_strategy=config["split_strategy"], unlabeled_id=-1)

    img, pcs = img_preprocessing(dst_name, parent_dir, train_gt, use_pca=config["use_pca"], n_comps=config["n_comps"], unlabeled_id=-1)

    # exclude corner pixels
    if config["split_strategy"] == "split3":
        leakage_area_tr = ((test_gt+1) + (val_gt+1))-1
        leakage_area_val = ((test_gt+1) + (train_gt+1))-1
        leakage_area_ts =  (((val_gt+1) + (train_gt+1))-1)

    else: # random 
        leakage_area_tr, leakage_area_val, leakage_area_ts = None, None, None
        
    train_set = PatchedDataset(img, train_gt, leakage_area_tr, patch_size=config["patch_size"], use_pca=config["use_pca"],
                            pcs=pcs, unlabeled_id=-1, shuffle=True)
    
    val_set = PatchedDataset(img, val_gt, leakage_area_val, patch_size=config["patch_size"], use_pca=config["use_pca"],
                            pcs=pcs, unlabeled_id=-1, shuffle=True)
    
    test_set = PatchedDataset(img, test_gt, leakage_area_ts, patch_size=config["patch_size"], use_pca=config["use_pca"],
                            pcs=pcs, unlabeled_id=-1, shuffle=True)
    
   
    train_loader = torch.utils.data.DataLoader(train_set, config["batch_size"], shuffle=True, num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4)  
    val_loader = torch.utils.data.DataLoader(val_set, config["batch_size"], shuffle=True, num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4)  
    test_loader = torch.utils.data.DataLoader(test_set, config["batch_size"], shuffle=True, num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2)  
  
    return train_loader, val_loader, test_loader, config

    
if __name__ == "__main__":
    
    #  Clean GPU
    clean_gpu()
    
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--model_name", type=str, help="Specify the name of the model that you want to use.")
    parser.add_argument('--dataset_name', choices=['ip', 'pu', 'sa'], help='dataset to use')
    parser.add_argument('--patch_size', type=int, help='Patch size (must be odd)')
    parser.add_argument('--n_epochs', type=int, help='epoch number')
   
    
    parser.add_argument('--use_pca', type=str2bool, default=False, help='Tel if model use PCA')
    parser.add_argument('--n_comps', type=int, default=0, help='Number principal components')
    parser.add_argument('--optim_metric', default='accuracy', help='Metric that optimize')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--eval_step', type=int, default=3, help='number of evaluation')
    parser.add_argument('--weight_decay', type=float, default=3e-7, help='weight_decay')
    parser.add_argument("--n_runs", type=int, default=4, help="Number of run on cross validation")
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    
    ######################################### added for different samplers #########################################
    parser.add_argument("--split_strategy", type=str, choices=['random', 'split3']) # which sampler to use
    ##############################################################################################################

    config = parser.parse_args()
    config = dict(vars(config))

        
    iter_results = []
    best_metric_model = 0.0
    best_config_model = None
        
    for run in range(config["n_runs"]):
        train_side = ("left", "top", "right", "bottom")[run]
     
        img, _, labels, _ = load_dataset(config["dataset_name"], "datasets/", standardize_image=False)
        config["n_classes"]=len(labels)
        if config["use_pca"]:
            config["original_bands"] = img.shape[2]
            config["n_bands"]=config["n_comps"]
        else:
            config["n_bands"]=img.shape[2]
        iter_config = config.copy()
 
    
        # initialize model
        model, criterion, optimizer, scheduler, iter_config = get_model(**iter_config)
        train_loader, val_loader, test_loader, iter_config = get_dataloader(iter_config["dataset_name"], iter_config, "datasets/")
    

        model_log_dir = os.path.join("runs", config["dataset_name"], iter_config["model_name"], str(iter_config["patch_size"]), iter_config["split_strategy"])
        os.makedirs(model_log_dir, exist_ok=True)
        
        iter_log_dir = os.path.join(model_log_dir, f"run_{run}")
        iter_config["log_dir"] = iter_log_dir

        print("\n" + "="*80)
        print(f"Starting training Training")
    
        print(f"Configuration: {iter_config}")  
        iter_result = training(model, optimizer, criterion, train_loader, val_loader, iter_config, test_loader=test_loader, scheduler=scheduler)
        if iter_result is not None:
            iter_metric = iter_result[list(iter_result.keys())[0]]["metrics"][iter_config["optim_metric"]]
            if iter_metric > best_metric_model:
                best_metric_model = iter_metric
                iter_config["best_checkpoint"] = iter_result[list(iter_result.keys())[0]]["checkpoint"]
                best_config_model = iter_config.copy()
            iter_results.append(iter_result[list(iter_result.keys())[0]]["metrics"])

    if len(iter_results) > 0:
        aggregate_results(iter_results, labels, aggregated=len(iter_results)>1, config=best_config_model, save_path=model_log_dir)
        
    # save best overall config for this dataset
    serializable_results = {
            k: convert_to_serializable(v)
            for k, v in best_config_model.items()
        }
    filename = os.path.join(model_log_dir, "best_overall_config.json")
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)
        
    print("\n" + "#"*80)
    print(f"Best overall config for dataset {iter_config["dataset_name"]}: {best_config_model} with metric {best_metric_model}")
    print("#"*80 + "\n\n")  
