####################### python importation #########################
import torch
import torch.nn as nn
from torch.nn import init

#################################### our importations ####################################
from utils.utils import get_device
from models.ssrn_2017 import get_ssrn
from models.dsformer_2025 import get_dsformer
from models.hamidaetal_2018 import get_hamidaetal
from models.spectralformer_2021 import get_vit_or_spectralformer





#################################### utils ####################################
def init_weights(model):
    """
    Initialize the weights of a neural network model using Kaiming normal initialization for convolutional layers,
    Xavier normal initialization for linear layers, and constant initialization for batch normalization layers.

    Parameters:
    model (nn.Module): The neural network module to initialize.

    Returns:
    None. The weights of the module are modified in-place.
    """
    if isinstance(model, nn.Conv3d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.Conv2d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.Conv1d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm3d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm1d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.Linear):
        init.xavier_normal_(model.weight)
        if model.bias is not None:
            init.constant_(model.bias, 0)
            
            
def get_model(**kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        model_name: string of the model model_name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    assert "model_name" in kwargs, "Model name is required"
    #  hyperparameters 
    device = get_device(kwargs.setdefault("gpu_id", 0))
    weights = torch.ones(kwargs.get("n_classes"))
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)  # Default: equal weight for all classes
    kwargs.setdefault("apply_weight_initialization", True)
    model_name = kwargs.get("model_name").lower()
            
    if model_name == "dsformer":
        model, optimizer, scheduler, kwargs = get_dsformer(kwargs)
        
    elif model_name in ("spectralformer", "vit"):
        model, optimizer, scheduler, kwargs = get_vit_or_spectralformer(kwargs)
        
    elif model_name == "ssrn":
        model, optimizer, scheduler, kwargs = get_ssrn(kwargs)
    
    elif model_name == "hamidaetal":
        model, optimizer, scheduler, kwargs = get_hamidaetal(kwargs)
    

    else:
        print("This architecture is not implemented")
  
    weights = weights if isinstance(weights, torch.Tensor) else torch.tensor(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Initialize weights
    if kwargs.get("apply_weight_initialization"):
        model.apply(init_weights)
        print("Layers Initialized \n")
            
    # Number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kwargs.setdefault("total_params", total_params)
    print(f"Number of trainable parameters: {total_params:,}".replace(",", " "))
    
    return model, criterion, optimizer, scheduler, kwargs

            
            
            
