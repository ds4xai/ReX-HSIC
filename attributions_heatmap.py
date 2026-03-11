import numpy as np
import os, json, torch
from tqdm.auto import tqdm

from captum.attr import Saliency
from models.get_model import get_model
from utils.utils import get_device, load_dataset
from utils.xai_utils import get_test_loader, integrated_gradient, plot_spatiale_heatmap, plot_spectrale_heatmap

if __name__=="__main__":
    
    device = get_device(0)

    
    methods = ["ig"]
    dataset_names = ["ip", "pu", "sa"]
    split_strategies = ["random", "split3"]
    model_names = ["dsformer", "ssrn", "hamidaetal", "vit", "spectralformer"] 
    
    for method in methods:
        for dataset_name in dataset_names: 
            for model_name in model_names:
                strategy_attributions = {
                    strat: None for strat in split_strategies
                    }
                
                for split_strategy in split_strategies:
                    
                    log_dir=os.path.join("runs", dataset_name, model_name, split_strategy)
                    
                    # Get best model config
                    path = os.path.join(log_dir, "best_overall_config.json")
                    with open(path) as f:
                        config = json.load(f)
                
                    model, _, _, _, config = get_model(**config)
                    
                    # Load checkpoint
                    best_chkpt_file = config["best_checkpoint"]
                    state_dict = torch.load(best_chkpt_file, map_location=device, weights_only=False)['model_state_dict']
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove 'module.' prefix for DDP
                    print(config["model_name"], " : ", model.load_state_dict(state_dict))
                    
                    # Get test loader 
                    config.update({"batch_size": 10})

                    image, gt, labels, _ = load_dataset(config["dataset_name"], "./datasets/", standardize_image=True)
                    test_loader, test_set, baseline, config = get_test_loader(config["dataset_name"], config, "./datasets/")
                    
                    
                    model = model.to(device)
                    print("\n", model_name)

                    model.eval()
                    all_attributions = {"spatiale": [], "spectrale": []}
                    saliency = Saliency(model)
                    
                    with torch.no_grad():
                        pbar = tqdm(test_loader, colour="blue")
                        for i, (inputs, targets) in enumerate(pbar, start=1):
                            inputs, targets = inputs.to(device), targets.to(device)
                            output = model(inputs)
                            mask = output.argmax(dim=1).cpu().numpy() == targets.cpu().numpy()
                            if mask.sum() == 0:
                                continue
                            
                            inputs, targets = inputs[mask], targets[mask]
                            inputs.requires_grad = True
                            if baseline.shape==3:
                                    baseline = baseline.unsqueeze(0)
                
                            batch_size = inputs.shape[0]
                            
                            if method.lower() == "ig":
                                baseline_expanded = baseline.expand(batch_size, -1, -1, -1)
                                attributions, delta = integrated_gradient(inputs, targets, model, baseline=baseline_expanded, device=device)    
                                attributions_spa = attributions.mean(axis=(0, 1)).tolist()
                                #attributions_spe = attributions.mean(axis=(0, 2, 3)).tolist()
                                all_attributions["spatiale"].append(attributions_spa)
                                #all_attributions["spectrale"].append(attributions_spe)       
                                
                                # # sur quelques batch pour souci de complexité 
                                # if len(all_attributions["spatiale"])>100:
                                #     break
                            elif method.lower() == "saliency":
                                attributions = saliency.attribute(inputs, targets)
                                attributions_spa = attributions.mean(dim=(0, 1)).detach().cpu().numpy().tolist()
                                #attributions_spe = attributions.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()
                                all_attributions["spatiale"].append(attributions_spa)
                                #all_attributions["spectrale"].append(attributions_spe)
                                
                                # # sur quelques batch pour souci de complexité 
                                # if len(all_attributions)>1000:
                                #     break
                            else:
                                raise(f'{method} not implemented')
                            
                    strategy_attributions[split_strategy]={"spatiale" : np.array(all_attributions["spatiale"]).mean(axis=(0)),}
                                                            #"spectrale" : np.array(all_attributions["spectrale"]).mean(axis=(0)).reshape(1, -1)}
                    
                del all_attributions, attributions, inputs, targets, mask
                
                save_path = f"images/{dataset_name}/attribution_maps/{method}"
                os.makedirs(save_path, exist_ok=True)
                # aggregate on all sample
                plot_spatiale_heatmap(
                strategy_attributions["random"]["spatiale"],
                strategy_attributions["split3"]["spatiale"],
                method.lower(),
                model_name=model_name,
                save_path=save_path,
                normalization="per_map",
                show=False,
                cmap="YlOrRd"
            )           
       