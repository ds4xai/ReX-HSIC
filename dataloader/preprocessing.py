import numpy as np
from sklearn.decomposition import PCA
from utils.utils import load_dataset, split_gt



# Preprocessing function for ground truth data
def get_gts(dataset_name, parent_dir, train_ratio=.5, train_side="right", split_strategy="split3"
            , unlabeled_id=-1, standardize_image=True):

    dataset_name = dataset_name.lower()
    available_sets = ['pu', 'sa', 'ip',]
    assert dataset_name in available_sets, "This dataset in not available"
    
    _, gt, _, _ = load_dataset(dataset_name, parent_dir, standardize_image=standardize_image)
    gt = gt.astype(np.int8)
    
    # we fix test size and test side for all datasets and all folds
    ratio = .7 if dataset_name=='ip' else .6 
    train_val_gt, test_gt =  split_gt(gt, train_size=ratio, split_strategy=split_strategy,
            train_side=train_side, unlabeled_id=unlabeled_id)

    train_gt, dev_gt =  split_gt(train_val_gt, train_size=train_ratio, split_strategy=split_strategy, 
            train_side=train_side, unlabeled_id=unlabeled_id)
    
    return train_gt, dev_gt, test_gt

# Preprocessing function for images
def img_preprocessing(dataset_name, parent_dir, train_gt, use_pca=True, n_comps=20, unlabeled_id=-1):

    dataset_name = dataset_name.lower()
    available_sets = ['pu', 'sa', 'ip',]
    assert dataset_name in available_sets, "This dataset in not available"
    

    img, _, _, _= load_dataset(dataset_name, parent_dir, standardize_image=True)
    img = img.transpose(2, 0, 1)
    
    # handle invalid pixels 
    img_band_0 = img[0] 
    invalid_samples = np.all(img == img_band_0, axis=0)
    img[:, invalid_samples] = 0 
    img = img.astype(np.float32)


    # compute statistics on training set only
    train_mask = train_gt != unlabeled_id
    
    # exclude invalid pixels from training mask
    train_mask = train_mask & (~invalid_samples)
    
    # compute PCA on valid samples in training set only
    if use_pca:
        X = img[:, train_mask].transpose(1, 0)
        pcs = PCA(n_components=n_comps)
        pcs = pcs.fit(X)
    else:
        pcs = None
        
    return img, pcs


