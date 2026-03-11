import torch, os
import numpy as np
from einops import repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from dataloader.dataset import PatchedDataset
from utils.utils import compute_metrics, load_dataset
from captum.attr import IntegratedGradients, NoiseTunnel
from dataloader.preprocessing import get_gts, img_preprocessing



def build_baseline(dataset_name, parent_dir, patch_size, use_pca=False, pcs=None):
    
    # load original image
    _, _, _, stats =  load_dataset(dataset_name, parent_dir, standardize_image=True)
    
    # absence info for hsi is represented by 0
    baseline_vec = np.zeros((1, len(stats["means"])), dtype=np.float32)

    # apply preprocessing
    baseline_vec = baseline_vec - stats["min"] / (stats["max"] - stats["min"])
    for c in range(len(stats["means"])):
        baseline_vec[0, c] -= stats["means"][c] 
    
    if use_pca:
        baseline_vec = pcs.transform(baseline_vec).reshape(-1)
    
    # transform absence info vec to patches with respect to patch size
    baseline_patch = repeat(baseline_vec.reshape(-1), 'b -> b h w', h=patch_size, w=patch_size)
    baseline_patch = torch.tensor(baseline_patch, dtype=torch.float32)
    
    return baseline_patch

def integrated_gradient(input, target, model, baseline=None, device='cpu'):
    """
    Computes the integrated gradients for a given input, target, and model.
    """
    model.eval()
    model.to(device)
    input = input.to(device) if input.requires_grad else input.requires_grad_(True).to(device)
    if baseline is None:
        baseline = torch.zeros_like(input, device=device, requires_grad=True)
        
    else:
        baseline.requires_grad = True

    ig = IntegratedGradients(model)
    
    # Use NoiseTunnel to add noise and improve the stability of attributions.
    # NoiseTunnel applies noise to the input and averages the attributions over multiple noisy samples.
    # This reduces the variance of the attributions and makes them more robust.
    ig_with_nt = NoiseTunnel(ig)
    
    attributions, delta = ig_with_nt.attribute(
        input, 
        nt_type='smoothgrad_sq',  # Noise type: SmoothGrad with squared gradients.
        target=target, 
        nt_samples=3,           # Number of noisy samples to generate.
        stdevs=0.2,              # Standard deviation of the noise added.
        n_steps=10,#0,              # Number of steps for the integral approximation.
        return_convergence_delta=True
    )
    
    return attributions.detach().cpu().numpy(), delta.detach().cpu().numpy()

def feature_intervention(
    model,
    test_loader,
    patch_size,
    mask_size,
    baseline,
    intervention_type="ablation",   # "ablation" or "permutation"
    test_dataset=None,              # required only for permutation
    device="cpu"
):
    """
    Run feature-level intervention (ablation or permutation) on a model.
    """

    assert intervention_type in ["ablation", "permutation"], \
        "intervention_type must be: ablation | permutation "

    no_intervention = mask_size == 0

    # ---- Patch geometry ----
    center = patch_size // 2
    half = mask_size // 2
    top_left = max(0, center - half)
    bottom_right = center + half + 1
    
    # Function for patch insertion 
    def apply_patch(dest, src):
        """Replace the (mask_size × mask_size) subpatch inside dest with src."""
        dest[:, :, top_left:bottom_right, top_left:bottom_right] = \
            src[:, :, top_left:bottom_right, top_left:bottom_right]
        return dest

    model.to(device)
    model.eval()
    print(f"Intervention mode: {intervention_type} | mask size: {mask_size}")

    if baseline.dim() == 3:  # (C,H,W)
        baseline = baseline.unsqueeze(0)

    preds_baseline = []
    preds_intervention = []
    gts = []

    with torch.no_grad():
        pbar = tqdm(test_loader, colour="blue")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.size(0)
            gts.append(targets)

            # =========================
            # Case 1: NO INTERVENTION
            # =========================
            if no_intervention:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                preds_baseline.append(preds)
                continue

            # =========================
            # Case 2: INTERVENTION
            # =========================
            modified_inputs = inputs.clone()

            # ---- Apply the chosen intervention ----
            if intervention_type == "permutation":
                assert test_dataset is not None, \
                    "test_dataset must be provided for permutation."

                random_idx = torch.randint(0, len(test_dataset), (batch_size,))
                sampled_imgs = torch.stack(
                    [test_dataset[i][0] for i in random_idx], dim=0
                ).to(device)

                modified_inputs = apply_patch(modified_inputs, sampled_imgs)

            elif intervention_type == "ablation":
                baseline_expanded = baseline.expand(batch_size, -1, -1, -1)
                modified_inputs = apply_patch(modified_inputs, baseline_expanded)

            # ---- Model inference ----
            outputs_mod = model(modified_inputs)
            preds_mod = torch.argmax(outputs_mod, dim=1)
            preds_intervention.append(preds_mod)

    # =========================
    # METRICS
    # =========================
    gts = torch.cat(gts).cpu().numpy()

    if no_intervention:
        preds_b = torch.cat(preds_baseline).cpu().numpy()
        return compute_metrics(gts, preds_b)
    else:
        preds_i = torch.cat(preds_intervention).cpu().numpy()
        return compute_metrics(gts, preds_i)

def compute_amie(
    model,
    test_loader,
    patch_size,
    mask_size,
    baseline,
    n_classes,
    intervention_type="ablation",
    test_dataset=None,
    device="cpu",
    return_mode="proba"
):
    """
    Compute AMIE (Average Model Intervention Effect) under ablation
    or permutation, using either probabilities ("proba") or logits ("logit").
    """

    assert intervention_type in ["permutation", "ablation"], \
        "intervention_type must be: permutation | ablation"

    assert return_mode in ["proba", "logit"], \
        "return_mode must be: proba | logit"

    # ---- Patch geometry ----
    center = patch_size // 2
    half = mask_size // 2
    top_left = max(0, center - half)
    bottom_right = min(patch_size, center + half + 1)

    # apply intervention
    def apply_patch(dest, src):
        dest[:, :, top_left:bottom_right, top_left:bottom_right] = \
            src[:, :, top_left:bottom_right, top_left:bottom_right]
        return dest

    if baseline.dim() == 3:
        baseline = baseline.unsqueeze(0)

    model.to(device)
    model.eval()
    print(f"Intervention: {intervention_type} | mask={mask_size} | return={return_mode}")

    vals_base = []
    vals_do = []
    preds = []
    gts = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, colour="blue"):

            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            onehot = torch.nn.functional.one_hot(targets, num_classes=n_classes)

            # ===== BASELINE FORWARD =====
            out = model(inputs)

            if return_mode == "proba":
                out_eff = torch.softmax(out, dim=1)
            else:  # logits
                out_eff = out  # raw logits

            pred = torch.argmax(out_eff, dim=1)
            base_vals = out_eff[onehot.bool()].view(-1)

            vals_base.extend(base_vals.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            gts.extend(targets.cpu().numpy().tolist())

            # ===== INTERVENTION =====
            inputs_do = inputs.clone()

            if intervention_type == "ablation":
                b_exp = baseline.expand(batch_size, -1, -1, -1)
                inputs_do = apply_patch(inputs_do, b_exp)

            else:  # permutation
                idx = torch.randint(0, len(test_dataset), (batch_size,))
                sampled = torch.stack([test_dataset[i][0] for i in idx], dim=0).to(device)
                inputs_do = apply_patch(inputs_do, sampled)

            # ===== DO-FORWARD =====
            out_do = model(inputs_do)

            if return_mode == "proba":
                out_do_eff = torch.softmax(out_do, dim=1)
            else:
                out_do_eff = out_do

            do_vals = out_do_eff[onehot.bool()].view(-1)
            vals_do.extend(do_vals.cpu().numpy().tolist())

    # numpy
    vals_base = np.array(vals_base)
    vals_do = np.array(vals_do)
    preds = np.array(preds)
    gts = np.array(gts)

    correct_mask = preds == gts
    if correct_mask.sum() == 0:
        return np.nan, {cls: np.nan for cls in range(n_classes)}, None, None, None

    vals_base = vals_base[correct_mask]
    vals_do = vals_do[correct_mask]
    gts = gts[correct_mask]

    # ===== AMIE =====
    avg_amie = float(np.mean(vals_base - vals_do))

    amie_per_class = {}
    for cls in range(n_classes):
        m = gts == cls
        amie_per_class[cls] = (
            float(np.mean(vals_base[m] - vals_do[m])) if np.sum(m) > 0 else np.nan
        )

    return avg_amie, amie_per_class, vals_base, vals_do, gts

def bootstrap_amie(probs_baseline, probs_do, n_bootstrap=1000, alpha=0.05):
    """
    Calcule l'intervalle de confiance de l'AMIE par bootstrap.
    """
    amie_bootstraps = []
    n_samples = len(probs_baseline)
    for _ in range(n_bootstrap):
        # Rééchantillonnage avec remplacement
        indices = resample(np.arange(n_samples), replace=True)
        b_b = probs_baseline[indices]
        b_d = probs_do[indices]
        # Calcul de l'AMIE pour cet échantillon
        amie_bootstraps.append(np.mean(b_b - b_d))

    # Intervalle de confiance à (1-alpha)*100%
    lower = np.percentile(amie_bootstraps, 100 * (alpha / 2))
    upper = np.percentile(amie_bootstraps, 100 * (1 - alpha / 2))
    return lower, upper, np.mean(amie_bootstraps)

def plot_spectrale_heatmap(
    heatmap_random,
    heatmap_disjoint,
    method,
    model_name="model",
    save_path="attribution_maps",
    show=True,
    cmap="YlOrRd"
):

    # Shape string
    bd = f"{heatmap_random.shape[1]}"

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Output file
    file_path = os.path.join(save_path, f"{model_name}_{method.upper()}_spec_attributions.png")


    aspect_mode = "auto"

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(heatmap_random, cmap=cmap, aspect=aspect_mode)
    axes[0].set_title(f"{model_name.upper()} - Random (band={bd}) - {method.upper()}")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(heatmap_disjoint, cmap=cmap, aspect=aspect_mode)
    axes[1].set_title(f"{model_name.upper()} - split3 (band={bd}) - {method.upper()}")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    # Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)

def get_test_loader(dst_name, config, parent_dir):
    
    assert dst_name in ("ip", "pu", "sa"), f"Dataset {dst_name} not supported"
    assert "patch_size" in config, "patch_size must be specified in config"
    assert "batch_size" in config, "batch_size must be specified in config"
    
    
    if config["split_strategy"] == "random_by_class":
        train_size = .3
    else: 
        train_size = .5 if dst_name in ("pu", "sa") else .7
        
    # spliting random or split3
    train_gt, _, test_gt = get_gts(dst_name, parent_dir, train_ratio=train_size, train_side="right", 
                                        split_strategy=config["split_strategy"], unlabeled_id=-1)

    img, pcs = img_preprocessing(dst_name, parent_dir, train_gt, use_pca=config["use_pca"], n_comps=config["n_comps"], unlabeled_id=-1)
    

    test_set = PatchedDataset(img, test_gt, None, patch_size=config["patch_size"], use_pca=config["use_pca"],
                            pcs=pcs, unlabeled_id=-1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, config["batch_size"], shuffle=True, num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2)  
    
    baseline=build_baseline(dst_name, parent_dir, config["patch_size"], use_pca=config["use_pca"], pcs=pcs)
  
    return test_loader, test_set, baseline,  config

def normalize_map(x, mode="per_map"):
    """Normalize heatmap to [0,1]."""
    if mode is None:
        return x  # no normalization
    
    if mode == "per_map":
        return (x - x.min()) / (x.max() - x.min() + 1e-9)
    
    raise ValueError("Unknown normalization mode.")

def plot_spatiale_heatmap(
    heatmap_random,
    heatmap_disjoint,
    method,
    model_name="model",
    save_path="attribution_maps",
    normalization="global",   # None | "global" | "per_map"
    show=True,
    cmap="YlOrRd"
):
    os.makedirs(save_path, exist_ok=True)

    # Normalize heatmaps
    if normalization == "per_map":
        hr = normalize_map(heatmap_random, "per_map")
        hd = normalize_map(heatmap_disjoint, "per_map")
    elif normalization == "global":
        min_global = min(heatmap_random.min(), heatmap_disjoint.min())
        max_global = max(heatmap_random.max(), heatmap_disjoint.max())
        hr = (heatmap_random - min_global) / (max_global - min_global + 1e-9)
        hd = (heatmap_disjoint - min_global) / (max_global - min_global + 1e-9)
    else:
        hr = heatmap_random
        hd = heatmap_disjoint

    # Diff map (Random - split3)
    diff = hr - hd

    # Color scale shared for hr and hd
    vmin = min(hr.min(), hd.min())
    vmax = max(hr.max(), hd.max())

    # For difference map: symmetric scale around zero
    maxabs = np.max(np.abs(diff))

    ps = f"PS: {hr.shape[0]}x{hr.shape[0]}"

    file_path = os.path.join(
        save_path,
        f"{model_name}_{method.upper()}_spat_attributions_with_diff.png"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name.upper()} — {method.upper()} — {ps}", fontsize=14)

    # ---- Random ----
    im1 = axes[0].imshow(hr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Random")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # ---- split3 ----
    im2 = axes[1].imshow(hd, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("split3")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # ---- Diff map ----
    im3 = axes[2].imshow(
        diff,
        cmap="bwr",     # diverging map for positive/negative differences
        vmin=-maxabs,
        vmax=+maxabs
    )
    axes[2].set_title("Diff = Random - split3")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_amie_comparison(
    dataset_name,
    model_name,
    random_amie,
    random_ci,
    disjoint_amie,
    disjoint_ci,
    levels=["Level 1 (1x1)", "Level 2 (3x3)", "Level 3 (5x5)"],
    save_dir="./",
    file_name="",
    show_plot=True,
    h_offset=0.1,
    v_offset=0.01
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{file_name}"

    bar_width = 0.35
    x_pos_rand = np.arange(len(levels))
    x_pos_dis = x_pos_rand + bar_width

    plt.figure(figsize=(10, 6))

    # Barres pour "random" (en bleu)
    bars_rand = plt.bar(
        x_pos_rand,
        random_amie,
        width=bar_width,
        color='skyblue',
        label='Random'
    )

    # Barres pour "split3" (en orange)
    bars_dis = plt.bar(
        x_pos_dis,
        disjoint_amie,
        width=bar_width,
        color='orange',
        label='split3'
    )

    # Ajouter les IC manuellement pour "random"
    for i, (mean, (lower, upper)) in enumerate(zip(random_amie, random_ci)):
        plt.plot([x_pos_rand[i], x_pos_rand[i]], [lower, upper], color='black', linewidth=1.5)
        plt.plot([x_pos_rand[i] - bar_width/4, x_pos_rand[i] + bar_width/4], [lower, lower], color='black', linewidth=1.5)  # Barre horizontale inférieure
        plt.plot([x_pos_rand[i] - bar_width/4, x_pos_rand[i] + bar_width/4], [upper, upper], color='black', linewidth=1.5)  # Barre horizontale supérieure

    # Ajouter les IC manuellement pour "split3"
    for i, (mean, (lower, upper)) in enumerate(zip(disjoint_amie, disjoint_ci)):
        plt.plot([x_pos_dis[i], x_pos_dis[i]], [lower, upper], color='black', linewidth=1.5)
        plt.plot([x_pos_dis[i] - bar_width/4, x_pos_dis[i] + bar_width/4], [lower, lower], color='black', linewidth=1.5)
        plt.plot([x_pos_dis[i] - bar_width/4, x_pos_dis[i] + bar_width/4], [upper, upper], color='black', linewidth=1.5)

    plt.xlabel("Neighborhood Level", fontsize=12)
    plt.ylabel("Average Model Intervention Effect (AMIE)", fontsize=12)
    plt.title(f"{dataset_name}: Neighborhood Masking Impact on {model_name} Predictions (AMIE)", fontsize=14)
    plt.xticks(x_pos_rand + bar_width / 2, levels, fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.legend(fontsize=12)

    for i, v in enumerate(random_amie):
        plt.text(x_pos_rand[i] - h_offset, v + v_offset, f"{v:.3f}", ha='right', va='center', fontsize=10, color='black')
    for i, v in enumerate(disjoint_amie):
        plt.text(x_pos_dis[i] + h_offset, v + v_offset, f"{v:.3f}", ha='left', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.abspath(save_path)}")
    if show_plot:
        plt.show()
