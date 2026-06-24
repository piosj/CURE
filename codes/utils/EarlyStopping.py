import torch
import os

import torch
import os  

class EarlyStopping: 
    """
    Using mean value of AUC, MRR, nDCG@5, nDCG@10

    """
    def __init__(self, emb_dim, patience=3, min_delta=1e-4, ckpt_dir='checkpoints', verbose=True, save_all=False):
        self.patience = patience
        self.min_delta = min_delta
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose
        self.save_all = save_all
        self.emb_dim = emb_dim

        self.counter = 0
        self.best_score = None  
        self.early_stop = False
        self.best_epoch = -1
        self.best_ckpt_path = None
        self.lr = None  

        if self.ckpt_dir is not None:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def __call__(self, val_score, model, epoch, lr):
        """
        val_score: (AUC + MRR + nDCG@5 + nDCG@10) / 4

        """
        self.lr = lr

        if self.save_all:
            ckpt_name = f"epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_avgscore_{val_score:.4f}.pth"
            save_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), save_path)
            if self.verbose:
                print(f"[EarlyStopping] Saved checkpoint (all epochs): {save_path}")

        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            if self.verbose:
                print(f"[EarlyStopping] Initialize best_score = {val_score:.6f}")
            self.save_best_model(val_score, model, epoch, lr)

        elif (val_score - self.best_score) >= self.min_delta:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] score improved to {val_score:.6f} at epoch={epoch}. Reset counter.")
            self.save_best_model(val_score, model, epoch, lr)

        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement. counter={self.counter}/{self.patience} (score={val_score:.6f}).")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Stop training.")

    def save_best_model(self, val_score, model, epoch, lr):
        """save the best model"""
        ckpt_name = f"best_epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_avgscore_{val_score:.4f}.pth"
        save_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save(model.state_dict(), save_path)
        self.best_ckpt_path = save_path
        if self.verbose:
            print(f"[EarlyStopping] Best model saved: {save_path}")

