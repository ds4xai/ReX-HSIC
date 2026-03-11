import os
import time
from utils.utils import trainer, tester, get_device


def training(model, optimizer, criterion, train_loader, val_loader, config, test_loader=None, scheduler=None):
     
     # device
     gpu_id = config.setdefault("gpu_id", 0)
     device = get_device(gpu_id)
     
     log_dir = config["log_dir"]+"_time_"+time.strftime("%Y%m%d-%H%M%S")  if "log_dir" in config else os.join.path("runs", model._get_name(), "time_"+time.strftime("%Y%m%d-%H%M%S")) 
     os.makedirs(log_dir, exist_ok=True)
     
     best_chkpt_file = trainer(
          model,
          optimizer,
          criterion,
          train_loader,
          config,
          log_dir,
          device=device,
          scheduler= scheduler,
          dev_dataloader=val_loader,
          eval_step=config["eval_step"],
          pin_memory=True,
          optim_metric=config["optim_metric"],
          show_cls_report=False,
          display_iter=len(train_loader)//2,
     )
     
     results = None
     if test_loader is not None:
          results = tester(
               model,
               test_loader,
               best_chkpt_file,
               log_dir,
               device=device,
               show_cls_report=True,
               pin_memory=True        
          )
     return results
     
          

def testing(model, test_loader, checkpoint, log_dir, device=None):
     
     if device is None:
          device = get_device(0)
          
     results = tester(
          model,
          test_loader,
          checkpoint,
          log_dir,
          device=device,
          show_cls_report=True,
          pin_memory=True        
     )
     