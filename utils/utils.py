# Importations
import numpy as np
from typing import Union
from tqdm.auto import tqdm 
from scipy import io as sio
from matplotlib.patches import Patch
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
import os, time, json, torch, shutil, hsluv, joblib, random
from sklearn.metrics import classification_report, confusion_matrix





# Serialize object to JSON
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj
    
    
# Get device
def get_device(gpu_id):
    """
    Get the appropriate device for computation.

    This function checks if a CUDA GPU device is available. If it is, the function
    returns a CUDA device. If not, the function returns a CPU device.

    Returns:
    torch.device: The device for computation. It can be either 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        print(f"Computation on CUDA GPU device - N°{gpu_id}\n")
        device = torch.device(f'cuda:{gpu_id}')
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\ \n")
        device = torch.device('cpu')
    return device
    
    
# Sampling    
def split_gt(gt: np.ndarray,  train_size: Union[int, float] = 30,  train_side: str = "right", split_strategy: str = 'split3',
            unlabeled_id: int = -1, seed=42):
    
    if not seed:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)

    
    split_strategy = split_strategy.lower()
    train_side = train_side.lower()

    if unlabeled_id is None:
        unlabeled_id = -1

    xy_indices = np.argwhere(gt != unlabeled_id)
    y = gt[xy_indices[:, 0], xy_indices[:, 1]]
    unique_classes = np.unique(y)

    train_gt = np.copy(gt)
    test_gt = np.copy(gt)
    train_mask = np.zeros(gt.shape, dtype=bool)
    test_mask = np.zeros(gt.shape, dtype=bool)


    if split_strategy == 'random_stratify':
        train_size = train_size if isinstance(train_size, float)  else train_size/(gt!=unlabeled_id).sum().item()

        train_indices, test_indices = train_test_split(
            xy_indices, train_size=train_size, stratify=y, random_state=seed
        )
        train_mask[train_indices[:, 0], train_indices[:, 1]] = True
        test_mask = ~train_mask
        train_gt[test_mask] = unlabeled_id
        test_gt[train_mask] = unlabeled_id
        

    elif split_strategy == 'random':
        train_size = train_size if isinstance(train_size, int)  else int(train_size*(gt!=unlabeled_id).sum().item())
        
        indices = np.where(gt != unlabeled_id)
        class_indices = {}
        for idx in zip(*indices):
            label = gt[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        train_gt = np.full_like(gt, fill_value=-1)
        test_gt = np.full_like(gt, fill_value=-1)

        for label, idx_list in class_indices.items():
            if len(idx_list) < train_size:
                num_sample = len(idx_list)//2   
            else:
                num_sample = train_size
                
            selected_indices = random.sample(idx_list, num_sample)
            for idx in selected_indices:
                train_gt[idx] = gt[idx]
            for idx in idx_list:
                if idx not in selected_indices:
                    test_gt[idx] = gt[idx]
                    
    elif split_strategy == 'split3':
        H, W = gt.shape

        train_indices = []
        for class_label in unique_classes:
            # Mask for the current class
            class_mask = (gt == class_label)
            total_pixels = class_mask.sum()
            
            current_train_size = train_size if isinstance(train_size, float)  else (train_size/total_pixels)

            if train_side == 'left':            
                for col in range(1, W):
                    left_half_count = np.count_nonzero(class_mask[:, :col])
                    if (left_half_count / total_pixels) < current_train_size:
                        continue
                    elif left_half_count == total_pixels: 
                        # ensure at least element in test set
                        # polygon is too small vertically, we take a horizontal cut instead
                        done = False
                        for col in range(1, W):
                            for row in range(1, H):
                                left_half_count = np.count_nonzero(class_mask[:row, :col])
                                if (left_half_count / total_pixels) < current_train_size:
                                    continue
                                else:
                                    class_mask[:, col:] = False
                                    class_mask[row:, :] = False
                                    train_indices.extend(np.column_stack(np.where(class_mask)))
                                    done = True
                                    break
                            if done :
                                break
                            
                        break
                        
                    else:
                        class_mask[:, col:] = False
                        train_indices.extend(np.column_stack(np.where(class_mask)))
                        break
                            
            elif train_side == "right":
                for col in range(1, W):
                    right_half_count = np.count_nonzero(class_mask[:, -col:])
                    if (right_half_count / total_pixels) < current_train_size:
                        continue
                    elif right_half_count == total_pixels: 
                        # ensure at least element in test set
                        # polygon is too small vertically, we take a horizontal cut instead
                        done = False
                        for col in range(1, W):
                            for row in range(1, H):
                                right_half_count = np.count_nonzero(class_mask[:row, -col:])
                                if (right_half_count / total_pixels) < current_train_size:
                                    continue
                                else:
                                    class_mask[:, :-col] = False
                                    class_mask[row:, :] = False
                                    train_indices.extend(np.column_stack(np.where(class_mask)))
                                    done = True
                                    break
                            if done :
                                break
                    else:
                        class_mask[:, :-col] = False
                        train_indices.extend(np.column_stack(np.where(class_mask)))
                        break
                        
            elif train_side == 'bottom':
                for row in range(1, H):
                    bottom_half_count = np.count_nonzero(class_mask[-row:, :])
                    if (bottom_half_count / total_pixels) < current_train_size:
                        continue
                    elif bottom_half_count == total_pixels: # ensure at least element in test set
                        done = False
                        for row in range(1, H):
                            for col in range(1, W):
                                bottom_half_count = np.count_nonzero(class_mask[-row:, :col])
                                if (bottom_half_count / total_pixels) < current_train_size:
                                    continue
                                else:
                                    class_mask[:-row, :] = False
                                    class_mask[:, col:] = False
                                    train_indices.extend(np.column_stack(np.where(class_mask)))
                                    done = True
                                    break
                            if done :
                                break
                    else:
                        class_mask[:-row, :] = False
                        train_indices.extend(np.column_stack(np.where(class_mask)))
                        break
                            
            elif train_side == 'top':
                for row in range(1, H):
                    top_half_count = np.count_nonzero(class_mask[:row, :])
                    if (top_half_count / total_pixels) < current_train_size:
                        continue
                    elif top_half_count == total_pixels: # ensure at least element in test set
                        done = False
                        for row in range(1, H):
                            for col in range(1, W):
                                top_half_count = np.count_nonzero(class_mask[:row, :col])
                                if (top_half_count / total_pixels) < current_train_size:
                                    continue
                                else:
                                    class_mask[row:, :] = False
                                    class_mask[:, col:] = False
                                    train_indices.extend(np.column_stack(np.where(class_mask)))
                                    done = True
                                    break
                    else:
                        
                        class_mask[row:, :] = False
                        train_indices.extend(np.column_stack(np.where(class_mask)))
                        break
            else:
                raise ValueError("Unsupported train side: choose 'right', 'left', 'top' or 'bottom'")

        train_indices = np.array(train_indices)
        train_mask[train_indices[:, 0], train_indices[:, 1]] = True
        test_mask = ~train_mask
        
        train_gt[test_mask] = unlabeled_id
        test_gt[train_mask] = unlabeled_id
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    return train_gt, test_gt


# colors
def convert_2d_to_color(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for id, color in palette.items():
        mask_id = arr_2d == id
        arr_3d[mask_id] = color

    return arr_3d


def hex_to_rgb_tuple(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgb_tuple(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_palette_dict(n_valid_classes=20):
    palette = {0: (255, 255, 255)}  # background blanc

    if n_valid_classes <= 10:
        # Couleurs très distinctes, fortes
        base_colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999", "#00ffff"
        ]
        for i in range(1, n_valid_classes + 1):
            palette[i] = hex_to_rgb_tuple(base_colors[i % len(base_colors)])
    else:
        # Distribution contrastée sur le cercle HSLuv
        hues = [i * 360 / n_valid_classes for i in range(n_valid_classes)]
        for i, h in enumerate(hues, start=1):
            # On alterne saturation et luminance pour maximiser le contraste
            s = 90 if i % 2 == 0 else 70
            l = 55 if i % 3 == 0 else 65
            base_hex = hsluv.hsluv_to_hex([h, s, l])
            palette[i] = hex_to_rgb_tuple(base_hex)

        # Petit ajustement pour s'assurer que les couleurs ne soient pas trop proches
        # On applique une rotation sur le cercle si besoin
        if n_valid_classes > 30:
            palette = {
                k: hex_to_rgb_tuple(
                    hsluv.hsluv_to_hex([(i * 360 / n_valid_classes + 20 * (i % 2)) % 360, 80, 60])
                )
                for i, k in enumerate(palette.keys())
            }

    return palette


def plot_gt(gt, ids_to_crops, palette, title=None, save_path=None):
    """
    Plot a ground truth (GT) map with corresponding crop labels and a color legend.
    """
    unique_classes = np.unique(gt)
    if len(unique_classes) > len(list(ids_to_crops.keys())):
        raise("Warning: There are more unique classes than crop labels in the palette.")

    title = title if title else "Ground truth"

    img = convert_2d_to_color(gt, palette)

    legend_elements = [
        Patch(facecolor=np.array(color)/255, label=f'{ids_to_crops[i]}')
        for i, color in palette.items()
    ]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title.capitalize(), fontsize=20, fontweight='bold')
    plt.axis('off')

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()




def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:  # save the ckpt for each 10 epoch
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))
            
            
def compute_metrics(true_labels, predicted_labels, num_classes=None):
    """
    Compute and return various classification metrics, including accuracy, 
    confusion matrix, F1 scores, and kappa coefficient.

    Args:
        true_labels (list or np.ndarray): Ground truth labels.
        predicted_labels (list or np.ndarray): Predicted labels.
        num_classes (int, optional): Total number of classes. Defaults to max(true_labels) + 1.

    Returns:
        dict: A dictionary containing the following metrics:
            - "confusion_matrix": Confusion matrix of shape (num_classes, num_classes).
            - "overall_accuracy": Fraction of correctly classified samples.
            - "average_accuracy": Mean accuracy across all classes.
            - "accuracy_percentage": Global accuracy as a percentage.
            - "f1_scores": F1 score for each class.
            - "f1_macro": Macro-average F1 score across all classes.
            - "kappa": Cohen's kappa coefficient.
    """
    # Initialize results dictionary
    metrics = {}

    # Convert inputs to numpy arrays for consistency
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Determine the number of classes if not provided
    num_classes = np.max(true_labels) + 1 if num_classes is None else num_classes

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
    metrics["confusion_matrix"] = conf_matrix

    # Overall Accuracy (OA): Fraction of correctly classified samples
    total_correct = np.trace(conf_matrix)  # Sum of diagonal elements (true positives)
    total_samples = np.sum(conf_matrix)  # Total number of samples
    metrics["accuracy"] = total_correct / total_samples

    # Average Accuracy (AA): Mean accuracy per class
    class_totals = np.sum(conf_matrix, axis=1)  # Total samples per class
    class_accuracies = np.diag(conf_matrix) / np.maximum(class_totals, 1)  # Avoid division by zero
    metrics["average_accuracy"] = np.mean(class_accuracies)


    # F1 Scores: Compute per-class F1 scores
    f1_scores = []
    for i in range(num_classes):
        precision = conf_matrix[i, i] / np.maximum(np.sum(conf_matrix[:, i]), 1)  # Avoid division by zero
        recall = conf_matrix[i, i] / np.maximum(np.sum(conf_matrix[i, :]), 1)  # Avoid division by zero
        f1 = 2 * (precision * recall) / np.maximum((precision + recall), 1e-10)  # Avoid division by zero
        f1_scores.append(f1)
    metrics["f1_scores"] = np.array(f1_scores)
    metrics["f1_macro"] = np.mean(f1_scores)

    # Kappa Coefficient: Measure of inter-rater agreement
    observed_agreement = np.trace(conf_matrix) / total_samples
    expected_agreement = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_samples ** 2)
    metrics["kappa"] = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return metrics


def aggregate_results(results, labels, aggregated=False, config=None, save_path=None, print_final_result=True):
    """
    Agrège ou affiche les résultats d'une série d'expérimentations.

    Args:
        results (list[dict]): liste de dictionnaires contenant les métriques de chaque run.
        labels (list[str]): noms des classes.
        aggregated (bool): si True, agrège les résultats sur plusieurs runs.
        config (dict, optional): configuration de l’expérience.
        save_path (str, optional): chemin où sauvegarder les résultats.

    Returns:
        dict: dictionnaire des résultats agrégés ou du dernier run.
    """
    
    n_runs = len(results)
    txt = ""

    if config is not None:
        txt += f"\n{'='*20}\nConfiguration: Number of runs = {n_runs}\n"
        for k, v in config.items():
            txt += f"{k}: {v}\n"
        txt += f"{'-'*20}\n"

    if aggregated:
        agg_results = {
            "OA": str(round(np.mean([res["accuracy"] for res in results]), 2)) + " ± " + str(round(np.std([res["accuracy"] for res in results]), 4)),
            "Kappa": str(round(np.mean([res["kappa"] for res in results]), 2)) + " ± " + str(round(np.std([res["kappa"] for res in results]), 4)),
            "AA": str(round(np.mean([res["average_accuracy"] for res in results]), 2)) + " ± " + str(round(np.std([res["average_accuracy"] for res in results]), 4)),
            "F1 Macro": str(round(np.mean([res["f1_macro"] for res in results]), 2)) + " ± " + str(round(np.std([res["f1_macro"] for res in results]), 4)),
             "f1 scores": [
                 str(round(mean, 2)) + " ± " + str(round(std, 4))
                for (mean, std) in zip(
                    np.mean([res["f1_scores"] for res in results], axis=0),
                    np.std([res["f1_scores"] for res in results], axis=0)
                )
             ],
             "best_checkpoint": config.get("best_checkpoint", None) if config else None
            }

        for k, v in agg_results.items():
            if isinstance(v, np.ndarray): 
                continue
            txt += f"{k}: {v}\n"

        txt += "\nF1 per class\nLabel\tMean ± Std\n"
        for idx, label in enumerate(labels):
            f1 = agg_results["f1 scores"][idx]
            txt += f"{label}: {f1}\n"

        txt += f"{'='*20}\n"
        if print_final_result:
            print(txt)

        agg_results["resume_text"] = txt
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        
            serializable_results = {
                k: convert_to_serializable(v)
                for k, v in agg_results.items()
            }

            filename = os.path.join(save_path, "aggregate_results.json")
            with open(filename, "w") as f:
                json.dump(serializable_results, f, indent=4)

        return agg_results

    else:
        unique_run = results[-1]
        for k, v in unique_run.items():
            if not isinstance(v, float): 
                continue
            txt += f"{k}: {v:.02f}\n"

        txt += "\nF1 per class\nLabel\tF1 score\n"
        for idx, label in enumerate(labels):
            f1 = unique_run["f1_scores"][idx]
            txt += f"{label}: {f1:.02f}\n"

        txt += f"{'='*20}\n"
        if print_final_result:
            print(txt)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, "result.json")
            with open(filename, "w") as f:
                unique_run = convert_to_serializable(unique_run)
                json.dump(unique_run, f, indent=4)
        return unique_run



################################# modeling utils #################################

#### Trainer fonctions
def trainer(
    net,
    optimizer,
    criterion,
    train_dataloader,
    config, 
    save_path,
    device=None,
    scheduler=None,
    dev_dataloader=None,
    eval_step=1,
    pin_memory=False,
    optim_metric="accuracy",
    show_cls_report=False,
    display_iter=None,
    
):
    
    if display_iter is None:
        display_iter = len(train_dataloader)//2
        
    if criterion is None or optimizer is None: 
        raise Exception("Missing {}. You must specify a {}.".format("criterion" if criterion is None else "optimiser", "loss function" if criterion is None else "optimizer"))

    n_epochs = config["n_epochs"]
    device = device if device is not None else "cpu"
    optim_metric = optim_metric.lower()
    
    # save directory
    os.makedirs(save_path, exist_ok=True)
    
    net.to(device)
    
    temp_optim_metric = -1
    train_losses = []
    train_optim_metrics = []
    val_losses = []
    val_optim_metrics = []
    net_name = config["model_name"].lower() if "model_name" in config else str(net.__class__.__name__).lower()
    best_model_path = None

    # Dstart training
    start_time = time.time()
    

    """
            ###################################################
            pas d'optimisation par batch
            pour voir une estimimation du loss du model avec poids aléatoire
    """
    if dev_dataloader is not None:
        train_optim_metric, train_loss = validator(net, train_dataloader, criterion, optim_metric=optim_metric, \
                device=device, show_cls_report=False, pin_memory=pin_memory)
        
        val_optim_metric, val_loss = validator(net, dev_dataloader, criterion, optim_metric=optim_metric, \
                device=device, show_cls_report=show_cls_report, pin_memory=pin_memory)
        
        
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        val_optim_metrics.append(val_optim_metric)
        train_optim_metrics.append(train_optim_metric)
    #########################################################################

        
     # Training
    for e in range(1, n_epochs + 1): 
        
        # Training mode
        net.train()
        running_loss = 0.0
        all_targets, all_outputs = [], []
    
        pbar = tqdm(train_dataloader, desc=f"Training - Epoch [{e}/{n_epochs}]", colour="green", leave=True, total=len(train_dataloader))
        for i, batch in enumerate(train_dataloader, start=1):
            if pin_memory:
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                
            optimizer.zero_grad()
            outputs = net(inputs)
            
            # Optimization
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()
            
            # Check learning - compute metric(s)
            running_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_outputs.extend(outputs)
            all_targets.extend(targets.cpu().numpy())

            pbar.update(1) 

            # display log
            if i % display_iter == 0:
                pbar.set_postfix(Progress=f"{100.0 * i / len(train_dataloader):.0f}%", Loss=f"{running_loss / i:.6f}")
            
        # Update leaning rate
        if scheduler is not None:
            scheduler.step()
            
        if dev_dataloader is not None:
            if (e%eval_step == 0 or e==n_epochs):
                avg_loss = running_loss / len(train_dataloader)
                metrics = compute_metrics(all_targets, all_outputs)
                train_losses.append(avg_loss)
                train_optim_metrics.append(metrics[optim_metric])
                
                val_optim_metric, val_loss = validator(net, dev_dataloader, criterion, optim_metric=optim_metric, \
                    device=device, show_cls_report=show_cls_report, pin_memory=pin_memory)
                val_optim_metrics.append(val_optim_metric)
                val_losses.append(val_loss)
                print(f"Validation loss: {val_loss} ---- {optim_metric}: {val_optim_metric}")
                
                # save best model by metric of optimisation
                if val_optim_metric > temp_optim_metric: 
                    best_model_path = save_model(
                        net,
                        save_path,
                        epoch=e,
                        optim_metric={
                            optim_metric: val_optim_metric
                            },
                        )
                    temp_optim_metric = val_optim_metric
                print("\n") 
            
        else:
            avg_loss = running_loss / len(train_dataloader)
            metrics = compute_metrics(all_targets, all_outputs)
            train_losses.append(avg_loss)
            train_optim_metrics.append(metrics[optim_metric])

        pbar.close()  

    # empty culculator
    torch.cuda.empty_cache()
    end_time = time.time()
    training_time = np.round(end_time - start_time, 2)

    # logs path
    training_log_dir = os.path.join(save_path, "training_log")
    training_log_file = os.path.join(training_log_dir, "logs.json")
    
    logs = {}
        
    if "device" in config:
        config["device"] = str(config["device"])  # for serialisation
    if "weights" in config:
        config["weights"] = config["weights"].tolist()  # for serialisation

    # save logs
    if net_name not in logs: 
        logs[net_name] = {"losses": {"train": [], "val": []}, 
                          optim_metric: {"train": [], "val": []}, 
                          "hyperparameters": config,
                          "training_time": training_time
                         }
    
    logs[net_name]["losses"]["train"] = train_losses
    logs[net_name][optim_metric]["train"] = train_optim_metrics  
    if dev_dataloader is not None:
        logs[net_name]["losses"]["val"] = val_losses
        logs[net_name][optim_metric]["val"] = val_optim_metrics   

    # Save config 
    logs = convert_to_serializable(logs)
    os.makedirs(os.path.dirname(training_log_file), exist_ok=True)
    with open(training_log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    # generate learning curves
    generate_learning_curves(training_log_dir, optim_metric, eval_step)
    
    
    return best_model_path


# Validator function
def validator(
    net, 
    dev_dataloader, 
    criterion, 
    optim_metric="f1_macro",
    device="cpu", 
    show_cls_report=False, 
    pin_memory=False
    ):
    
    net.to(device)
    net.eval()
    all_targets, all_outputs = [], []
    running_loss = 0.0
    
    pbar = tqdm(dev_dataloader, desc=f"validation: ", colour="yellow", leave=True, total=len(dev_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader, start=1):
            if pin_memory == True:
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.squeeze())  
            running_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_outputs.extend(outputs)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({"Loss": running_loss / i})
            pbar.update(1)
       

    # compute metrics
    metric = compute_metrics(all_targets, all_outputs)[optim_metric]
    avg_loss = running_loss / len(dev_dataloader) 

    # display report
    if show_cls_report:
        print(classification_report(all_targets, all_outputs, zero_division=1))

    return metric, avg_loss


# Tester
def tester(net, 
           test_dataloader,
           checkpoint, 
           save_path, 
           device="cpu", 
           show_cls_report=False, 
           pin_memory=False):
    
    # Load best model weights from checkpoint
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove 'module.' prefix for DDP
    net.load_state_dict(state_dict)
    
    
    net.to(device)
    net.eval()
    
    all_targets, all_outputs = [], []
    test_log_dir = os.path.join(save_path, "test_log")

    pbar = tqdm(test_dataloader, desc=f"testing: ", colour="blue", leave=True, total=len(test_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader, start=1):
            if pin_memory == True:
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)

            outputs = net(inputs)
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_outputs.extend(outputs)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.update(1)
       

    # compute metrics
    results = compute_metrics(all_targets, all_outputs)
    
    if show_cls_report:
        print(classification_report(all_targets, all_outputs, zero_division=1))
        
    # logs file path
    net_name = str(net.__class__.__name__).lower()
    filename = os.path.join(test_log_dir, "logs.json")
    
    # load or initialize logs file
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            logs = json.load(f)
    else:
        logs = {}
        
    # for serialization
    results["confusion_matrix"] = results["confusion_matrix"].tolist()
    results["f1_scores"] = results["f1_scores"].tolist()
          
    # add results in logs
    if net_name not in logs:
        logs[net_name] = {
            "metrics": {},
            "checkpoint": checkpoint}
    logs[net_name]["metrics"] = results
    
    # Save log
    logs = convert_to_serializable(logs)
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    with open(filename, 'w') as f:
        json.dump(logs, f, indent=4)
        
    return logs



# Save checkpoint
def save_model(net, save_path, **kwargs):    
    if 'epoch' not in kwargs or 'optim_metric' not in kwargs:
        raise ValueError("Missing required kwargs: 'epoch' or 'optim_metric' must be provided for saving a torch model.")
    
    epoch = kwargs.get('epoch')
    net_name = kwargs["model_name"].lower() if "model_name" in kwargs else str(net.__class__.__name__).lower()
    metric, value = list(kwargs["optim_metric"].items())[0]
    
    chkpt_dir = os.path.join(save_path, "checkpoint")

    # Create model directory if it doesn't exist
    os.makedirs(chkpt_dir, exist_ok=True)
    
    # Clear the contents of the model directory
    try:
        for filename in os.listdir(chkpt_dir):
            file_path = os.path.join(chkpt_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        print(f"Error clearing the directory {chkpt_dir}: {e}")
    

    # Save PyTorch model weights
    if isinstance(net, torch.nn.Module):
        chkpt_name = f"{net_name}_epoch_{epoch}_{metric}_{value:.3f}.pth"
        chkpt_path = os.path.join(chkpt_dir, chkpt_name)
        print(f"Saving neural network weights in {chkpt_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': kwargs.get('optimizer_state_dict'),
            metric: value,
        }, chkpt_path)
        return chkpt_path

    
    # Save non-PyTorch model (e.g., scikit-learn model)
    else:
        print(f"Saving model params in {chkpt_name}.pkl")
        chkpt_name = f"{net_name}_epoch_{epoch}_{metric}_{value : .3f}.pkl"
        chkpt_path = os.path.join(chkpt_dir, chkpt_name)
        joblib.dump(net, chkpt_path)
        return chkpt_path




def compute_positions(gt, stride, absence_id=0):
    """Retourne les positions valides dans le masque gt selon le stride."""
    rows, cols = np.where(gt != absence_id)
    positions = set(zip(rows, cols))
    
    if stride > 1:
        R, C = gt.shape
        sampled_positions = [
            (r, c)
            for r in range(0, R, stride)
            for c in range(0, C, stride)
            if (r, c) in positions
        ]
        return sampled_positions
    
    return list(positions)


def count_examples(gt, cls, stride, absence_id=0):
    
    count = 0
    gt[gt != cls] = absence_id
    positions = compute_positions(gt, stride=stride, absence_id=absence_id)
    count = len(positions)
    return count


def find_stride(gt, cls, count, min_val, max_val, step=1, max_stride=100, absence_id=0):
    """Trouve un stride pour que le nombre d'exemples soit dans les bornes."""
  
    for stride in range(1, max_stride, step):
        count = count_examples(gt, cls, stride, absence_id=absence_id)

        if min_val <= count <= max_val:
            return stride, count
    count = count_examples(gt, cls, max_stride, absence_id=absence_id)
    return 64, count


def classes_strides(gt, dict_labels, min_val, max_val, absence_id=0):
    """Détermine le stride optimal pour chaque classe."""
    classes_stride = {}
    for cls, count in dict_labels.items():
        gt_cls = np.copy(gt)
        cls = int(cls)
        
        if cls == absence_id :
            continue 
        
        if min_val <= count <= max_val:
            classes_stride[cls] = (count, 1)
        else:
            stride, adjusted_count = find_stride(gt_cls, cls, count, min_val, max_val, absence_id=absence_id)
            classes_stride[cls] = (adjusted_count, stride)
           # print(f"  -> Nouveau count: {adjusted_count} | Stride: {stride}")
                
        #print(classes_stride)    
    return classes_stride



def update_positions(gt, class_stride, absence_id=-1):
    
    for cls, (count, stride) in class_stride.items():
        if cls == absence_id or stride == 1:
            continue
        
        cls_gt = np.copy(gt)
        cls_gt[cls_gt != cls] = absence_id

        x, y = np.where(cls_gt == cls)
        all_positions = np.array(list(zip(x, y)))
        retained_positions = np.array(compute_positions(cls_gt, stride=stride, absence_id=absence_id))
        excluded_positions = np.array(list(set(map(tuple, all_positions)) - set(map(tuple, retained_positions))))
        gt[excluded_positions[:,0], excluded_positions[:,1]] = absence_id
        
    return gt


# Standardization
def robust_minmax_cube(x, p_lo=1, p_hi=99, do_scale=True):
    # x: (B,C,H,W) or (B,L,C,H,W) time series
    lo = np.percentile(x, p_lo, axis=(-2, -1), keepdims=True)
    hi = np.percentile(x, p_hi, axis=(-2, -1), keepdims=True)
    # lo, hi : B,C) or (B,L,C)
    
    if do_scale:
        x_scale = x - lo /  (hi-lo)
    else:
        x_scale = None
    return  x_scale, lo, hi



def compute_patch_mask(x: int, y: int, patch_size: int, reference_points):
    half = patch_size // 2

   
    mask = np.ones((patch_size, patch_size), dtype=np.uint8)

    
    for (cx, cy) in reference_points:
        dx = cx - x
        dy = cy - y

        # get condidate
        if abs(dx) > patch_size or abs(dy) > patch_size:
            continue

        # compute corner 
        ix_min = max(-half, dx - half)
        ix_max = min(+half, dx + half)
        iy_min = max(-half, dy - half)
        iy_max = min(+half, dy + half)

        #  shit
        mask_x_min = ix_min + half
        mask_x_max = ix_max + half
        mask_y_min = iy_min + half
        mask_y_max = iy_max + half

        # Mark overlapping pixels as zero in the mask
        mask[mask_x_min:mask_x_max+1, mask_y_min:mask_y_max+1] = 0
    # keep center pixel
    mask[half, half]=1

    return mask


def build_mask_maker(reference_points, patch_size):
    half = patch_size // 2

    # Précompute meshgrid local de -(half)…+(half)
    gx = np.arange(-half, half+1)
    gy = np.arange(-half, half+1)
    DX, DY = np.meshgrid(gx, gy, indexing='ij')  # coord relative du patch

    # Convertir seulement une fois
    ref = np.array(list(reference_points))

    def compute_patch_mask_fast(x, y):

        # Coordonnées globales des pixels du patch
        PX = DX + x
        PY = DY + y

        # Condition EXACTE comme dans full_area:
        # centre du patch distant de <= patch_size
        close = (np.abs(ref[:,0] - x) <= patch_size) & \
                (np.abs(ref[:,1] - y) <= patch_size)

        mask = np.ones((patch_size, patch_size), dtype=np.uint8)

        if np.any(close):
            near_centers = ref[close]

            for cx, cy in near_centers:
                # Conditions EXACTES d'intersection :
                # pixel ∈ patch_region
                patch_region_x = np.abs(PX - x) <= half
                patch_region_y = np.abs(PY - y) <= half

                # pixel ∈ other_patch_region
                other_region_x = np.abs(PX - cx) <= half
                other_region_y = np.abs(PY - cy) <= half

                overlap = patch_region_x & patch_region_y & other_region_x & other_region_y
                mask[overlap] = 0

        # centre reste 1
        mask[half, half] = 1
        return mask

    return compute_patch_mask_fast



def generate_learning_curves(model_training_log_file, optim_metric, eval_step):
    """
    Plot training/validation loss curves and training/validation optimization metric curves.
    
    Args:
        model_training_log_file (str): directory containing log.dir
        optim_metric (str): key of the optimization metric in the log (e.g. "acc", "f1", etc.)
    """
    # Load log file
    file = os.path.join(model_training_log_file, "logs.json")
    with open(file) as f:
        training_log = json.load(f)

    # The log structure stores runs indexed by numbers, so we extract the first run
    run_key = list(training_log.keys())[0]
    log = training_log[run_key]

    # Extract values
    train_losses = log["losses"]["train"]
    val_losses = log["losses"]["val"]
    
    train_optim = log[optim_metric]["train"]
    val_optim = log[optim_metric]["val"]
    
    epochs = range(1, len(train_losses) + 1)
    epochs = np.array(epochs) * eval_step
    epochs = epochs.tolist()

    # Plot LOSS curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)

    loss_fig_path = os.path.join(model_training_log_file, "loss_curve.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()

    # Plot OPTIM METRIC curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_optim, label=f"Train {optim_metric}", linewidth=2)
    plt.plot(epochs, val_optim, label=f"Val {optim_metric}", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel(optim_metric.capitalize())
    plt.title(f"Training and Validation {optim_metric} Curves")
    plt.legend()
    plt.grid(True)

    optim_fig_path = os.path.join(model_training_log_file, f"{optim_metric}_curve.png")
    plt.savefig(optim_fig_path, dpi=300)
    plt.close()

    print(f"Saved loss figure to: {loss_fig_path}")
    print(f"Saved optim metric figure to: {optim_fig_path}")



def load_dataset(dataset_name, PARRENT_DIR, standardize_image=True):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'pu',
        'sa',
        'ip',
        'whulk',
        'whuhc',
        'whuhh',
        'houston13',
        'houston18',
        'QUH_TDW',
        'QUH_QY',
        'QUH_PA'
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if (dataset_name == 'sa'):
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "Salinas_corrected.mat"))
        image = image['salinas_corrected']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]
        

    elif (dataset_name == 'pu'):
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "PaviaU.mat"))
        image = image['paviaU']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
        

    elif (dataset_name == 'ip'):
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "Indian_pines_corrected.mat"))
        image = image['indian_pines_corrected']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "Indian_pines_gt.mat"))
        gt = gt['indian_pines_gt']
        labels = [
            'Undefined',
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]
        

    elif (dataset_name == 'whulk'):
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]
        

    elif dataset_name == 'whuhh':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "WHU_Hi_HongHu.mat"))
        image = image['WHU_Hi_HongHu']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "WHU_Hi_HongHu_gt.mat"))
        gt = gt['WHU_Hi_HongHu_gt']
        labels = [
            'Undefined',
            'Red roof',
            'Road',
            'Bare soil',
            'Cotton',
            'Cotton firewood',
            'Rape',
            'Chinese cabbage',
            'Pakchoi',
            'Cabbage',
            'Tuber mustard',
            'Brassica parachinensis',
            'Brassica chinensis',
            'Small Brassica chinensis',
            'Lactuca sativa',
            'Celtuce',
            'Film covered lettuce',
            'Romaine lettuce',
            'Carrot',
            'White radish',
            'Garlic sprout',
            'Broad bean',
            'Tree',
        ]
        

    elif dataset_name == 'whuhc':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "WHU_Hi_HanChuan.mat"))
        image = image['WHU_Hi_HanChuan']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "WHU_Hi_HanChuan_gt.mat"))
        gt = gt['WHU_Hi_HanChuan_gt']
        labels = [
            'Undefined',
            'Strawberry',
            'Cowpea',
            'Soybean',
            'Sorghum',
            'Water spinach',
            'Watermelon',
            'Greens',
            'Trees',
            'Grass',
            'Red roof',
            'Gray roof',
            'Plastic',
            'Bare soil',
            'Road',
            'Bright object',
            'Water',
        ]
        

    elif dataset_name == 'houston13':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "GRSS2013.mat"))
        image = image['GRSS2013']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "GRSS2013_gt.mat"))
        gt = gt['GRSS2013_gt']
        labels = [
            "Undefined",
            "Healthy grass",
            "Stressed grass",
            "Synthetic grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track",
        ]
        

    elif dataset_name == 'houston18':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "Houston2018.mat"))
        image = image['Houston2018']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "Houston2018_gt.mat"))
        gt = gt['Houston2018_gt']
        labels = [
            "Undefined",
            "Healthy grass",
            "Stressed grass",
            "Artificial turf",
            "Evergreen trees",
            "Deciduous trees",
            "Bare earth",
            "Water",
            "Residential buildings",
            "Non-residential buildings",
            "Roads",
            "Sidewalks",
            "Crosswalks",
            "Major thoroughfares",
            "Highways",
            "Railways",
            "Paved parking lots",
            "Unpaved parking lots",
            "Cars",
            "Trains",
            "Stadium seats",
        ]
        

    elif dataset_name == 'QUH_TDW':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "QUH-Tangdaowan.mat"))
        image = image['Tangdaowan']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "QUH-Tangdaowan_GT.mat"))
        gt = gt['TangdaowanGT']
        labels = [
            "Undefined",
            "Rubber track",
            "Flagging",
            "Sandy",
            "Asphalt",
            "Boardwalk",
            "Rocky shallows",
            "Grassland",
            "Bulrush",
            "Gravel road",
            "Ligustrum vicaryi",
            "Coniferous pine",
            "Spiraea",
            "Bare soil",
            "Buxus sinica",
            "Photinia serrulata",
            "Populus",
            "Ulmus pumila L",
            "Seawater",
        ]
        

    elif dataset_name == 'QUH_QY':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "QUH-Qingyun.mat"))
        image = image['Chengqu']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "QUH-Qingyun_GT.mat"))
        gt = gt['ChengquGT']
        labels = [
            "Undefined",
            "Trees",
            "Concrete building",
            "Car",
            "Ironhide building",
            "Plastic playground",
            "Asphalt road",
        ]
        

    elif dataset_name == 'QUH_PA':
        image = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "QUH-Pingan.mat"))
        image = image['Haigang']
        gt = sio.loadmat(os.path.join(PARRENT_DIR, dataset_name, "QUH-Pingan_GT.mat"))
        gt = gt['HaigangGT']
        labels = [
            "Undefined",
            "Ship",
            "Seawater",
            "Trees",
            "Concrete structure building",
            "Floating pier",
            "Brick houses",
            "Steel houses",
            "Wharf construction land",
            "Car",
            "Road",
        ]
        

    stats = {
    }
    
    if standardize_image:
        # from : https://github.com/YichuXu/DSFormer
        # after getting image and ground truth (gt), let us do data preprocessing!
        # step1 filter nan values out
        nan_mask = np.isnan(image.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
            image[nan_mask] = 0
            gt[nan_mask] = 0

        
        # step2 normalise the HSI data (method from SSAN, TGRS 2020)
        image = np.asarray(image, dtype=np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        stats['min'] = min_val
        stats['max'] = max_val
        mean_by_c = np.mean(image, axis=(0, 1))
        stats['means'] = mean_by_c.tolist()
        for c in range(image.shape[-1]):
            image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # step3 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step4 remove undefined label
    labels = labels[1:]

    return image, gt, labels, stats