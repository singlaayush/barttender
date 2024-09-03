import os
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl

from copy import deepcopy

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef
from mimic_constants import *

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner import Tuner
from sklearn.model_selection import KFold

image_size = (224, 224)
num_classes = 1

class MIMICDataset(Dataset):
    def __init__(self, csv_file_img, image_size, img_data_dir: Path, augmentation=False, pseudo_rgb=False):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        self.labels = ['Cardiomegaly']

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            og_path = self.data.loc[idx, 'path']
            img_path = construct_preproc_path(og_path, img_data_dir)
            img_label = np.array(self.data.loc[idx, self.labels[0].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image'])
        # unsqueeze adds an extra dimension to the labels to match the output of the model probs
        label = torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(0)

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        else:
            image = image.permute(2, 0, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label']}

class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, image_size, img_data_dir: Path, pseudo_rgb, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = MIMICDataset(self.csv_train_img, self.image_size, img_data_dir, augmentation=True, pseudo_rgb=pseudo_rgb)
        self.val_set = MIMICDataset(self.csv_val_img, self.image_size, img_data_dir, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = MIMICDataset(self.csv_test_img, self.image_size, img_data_dir,  augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)


class ResNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet34(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class DenseNetTrain(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        auc_scores = []
        for i in range(num_classes):
            auc = roc_auc_score(targets_np[:, i], preds_np[:, i])
            auc_scores.append(auc)
        
        for i, auc in enumerate(auc_scores):
            print(f"Cardiomegaly AUC: {auc:.4f}")

    return preds_np, targets_np, logits.cpu().numpy(), auc_scores

def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()

def compute_metrics(y_true, y_pred, y_logits):
    """Compute various classification metrics for binary classification."""
    # y_true: Actual binary labels (0 or 1)
    # y_pred: Predicted binary labels (0 or 1), thresholding applied
    # y_logits: Raw model outputs (logits) or probabilities for the positive class

    # Convert y_logits to probabilities if they are logits
    y_prob = y_logits if y_logits.shape[1] == 1 else y_logits[:, 1]

    auc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Convert y_pred to binary predictions if it's probabilities
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Metrics calculations
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0  # Sensitivity, Recall
    tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0  # Specificity
    ppv = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0  # Precision
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0  # Negative Predictive Value
    f1 = f1_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)

    return {
        'AUC': auc,
        'Average Precision': avg_precision,
        'TPR': tpr,
        'TNR': tnr,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
        'MCC': mcc
    }


def main(image_type: str = 'xray', order = None, batch_size: int = 192, epochs: int = 100, num_workers: int = 0, n_splits: int = 10, idp: bool = False, nan: bool = False, no_bars: bool = False):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    order, suffix = get_barcode_order_info(order=order, no_bars=no_bars, nan=nan)
    preproc_dir = get_preproc_subpath(image_type, suffix)
    root_dir = get_correct_root_dir(preproc_dir)
    img_data_dir = root_dir / preproc_dir

    # Create output directory for this cross-validation experiment
    exp_dir = home_out_dir / f"cross-val/densenet-{image_type}-{suffix}-{'idp' if idp else 'no_idp'}/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    df = get_cardiomegaly_df(idp=idp)
    if idp: df['split'] = 'train'
    train_df, val_df, test_df = df_train_test_split(df, test_size=0.2, val_size=0.1)
    train_df.to_csv(exp_dir / 'train.csv')    
    val_df.to_csv(exp_dir / 'val.csv')
    test_df.to_csv(exp_dir / 'test.csv')
    
    data = MIMICDataModule(csv_train_img=(exp_dir / 'train.csv').as_posix(),
                              csv_val_img=(exp_dir / 'val.csv').as_posix(),
                              csv_test_img=(exp_dir / 'test.csv').as_posix(),
                              img_data_dir=img_data_dir,
                              image_size=image_size,
                              pseudo_rgb=False,
                              batch_size=batch_size,
                              num_workers=num_workers)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Determine GPU usage and training strategy
    use_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if use_cuda else None
    device = "cuda" if use_cuda else "cpu"
    strategy = "ddp" if num_gpus > 1 else "auto"

    # Optimize batch size if only one GPU is available
    if num_gpus == 1 and False:
        model_type = DenseNetTrain(num_classes=num_classes)
        trainer = pl.Trainer(accelerator=accelerator, devices=1)
        tuner = Tuner(trainer)
        optimal_batch_size = tuner.scale_batch_size(model_type, datamodule=data)
        data.batch_size = optimal_batch_size
        print(f'Optimal batch size found: {optimal_batch_size}')
    else:
        optimal_batch_size = batch_size
        print(f'Optimal batch size found: {optimal_batch_size}')

    all_val_metrics = []
    all_test_metrics = []
    fold_metrics = []
    fold_data = deepcopy(data)
    for fold, (train_idx, val_idx) in enumerate(kf.split(data.train_set)):
        print(f'\n--- Fold {fold+1}/{n_splits} ---')

        # Split data for this fold
        train_subset = torch.utils.data.Subset(data.train_set, train_idx)
        val_subset = torch.utils.data.Subset(data.train_set, val_idx)
        
        fold_data.train_set = train_subset
        fold_data.val_set = val_subset

        if max(train_idx) >= len(data.train_set) or max(val_idx) >= len(data.train_set):
            raise IndexError("Indices out of bounds.")

        # Model
        model_type = DenseNetTrain
        model = model_type(num_classes=num_classes)

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')
        early_stopping_callback = EarlyStopping(monitor="val_loss", mode='min', patience=5, verbose=True)

        config = {}
        config["pseudo_rgb"] = False
        config["batch_size"] = optimal_batch_size
        config["image_size"] = image_size
        config["image_type"] = image_type
        config["barcode_order"] = suffix
        config["img_data_dir"] = img_data_dir
        config["idp"] = idp
        config['csv_train'] = (exp_dir / 'train.csv').as_posix()
        config['csv_val']   = (exp_dir / 'val.csv').as_posix()
        config['csv_test']  = (exp_dir / 'test.csv').as_posix()

        # Logger for this fold
        wandb_logger = WandbLogger(project='mimic', config=config, save_dir=home_out_dir)

        # Create output directory for this fold
        out_name = f"fold{fold+1}-{wandb_logger.experiment.id}/"
        out_dir = exp_dir / out_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        wandb_logger.experiment.config["out_dir"] = out_dir

        # Trainer
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=5,
            max_epochs=epochs,
            logger=wandb_logger,
            accelerator=accelerator,
            devices="auto" if use_cuda else 1,
            num_nodes=1,
            strategy=strategy
        )
        trainer.logger._default_hp_metric = False

        # Train the model
        trainer.fit(model, fold_data)

        # Load the best model from this fold
        model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes)
        model.to(device)

        # Validation
        print(f'\nVALIDATION for Fold {fold+1}')
        preds_val, targets_val, logits_val, aucs_val = test(model, fold_data.val_dataloader(), device)
        val_metrics = compute_metrics(targets_val, preds_val, logits_val)
        all_val_metrics.append(val_metrics)

        # Testing
        print(f'\nTESTING for Fold {fold+1}')
        preds_test, targets_test, logits_test, aucs_test = test(model, fold_data.test_dataloader(), device)
        test_metrics = compute_metrics(targets_test, preds_test, logits_test)
        all_test_metrics.append(test_metrics)

        # Log metrics for this fold
        for metric_name, metric_value in val_metrics.items():
            wandb_logger.log_metrics({f'fold_{fold+1}_val_{metric_name.lower()}': metric_value})
        for metric_name, metric_value in test_metrics.items():
            wandb_logger.log_metrics({f'fold_{fold+1}_test_{metric_name.lower()}': metric_value})

        cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
        cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
        cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

        # Save validation and test predictions for this fold
        df_val = pd.DataFrame(data=preds_val, columns=cols_names_classes)
        df_val_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
        df_val_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
        df_val = pd.concat([df_val, df_val_logits, df_val_targets], axis=1)
        df_val.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)

        df_test = pd.DataFrame(data=preds_test, columns=cols_names_classes)
        df_test_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
        df_test_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
        df_test = pd.concat([df_test, df_test_logits, df_test_targets], axis=1)
        df_test.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)

        # Store fold-wise metrics
        fold_metrics.append({
            'fold': fold+1,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        })

        wandb.finish()

    # Aggregate metrics over all folds
    val_metrics_mean = {key: np.mean([m[key] for m in all_val_metrics]) for key in all_val_metrics[0].keys()}
    test_metrics_mean = {key: np.mean([m[key] for m in all_test_metrics]) for key in all_test_metrics[0].keys()}
    val_metrics_std = {key: np.std([m[key] for m in all_val_metrics]) for key in all_val_metrics[0].keys()}
    test_metrics_std = {key: np.std([m[key] for m in all_test_metrics]) for key in all_test_metrics[0].keys()}
    best_folds = {key: max(enumerate(all_val_metrics), key=lambda x: x[1][key])[0]+1 for key in all_val_metrics[0].keys()}

    print(f'\nMean Validation Metrics across {n_splits} folds: {val_metrics_mean}')
    print(f'\nMean Test Metrics across {n_splits} folds: {test_metrics_mean}')

    # Save overall and fold-wise metrics to CSV
    metrics_summary = {
        'metric': list(val_metrics_mean.keys()),
        'val_mean': list(val_metrics_mean.values()),
        'val_std': list(val_metrics_std.values()),
        'test_mean': list(test_metrics_mean.values()),
        'test_std': list(test_metrics_std.values()),
        'best_fold': [best_folds[key] for key in val_metrics_mean.keys()]
    }
    df_metrics_summary = pd.DataFrame(metrics_summary)
    df_metrics_summary.to_csv(exp_dir / 'metrics_summary.csv', index=False)

    df_fold_metrics = pd.DataFrame(fold_metrics)
    df_fold_metrics.to_csv(exp_dir / 'fold_metrics.csv', index=False)

    # Log the fold that was best for each metric
    print(f'\nBest folds based on validation metrics: {best_folds}')

if __name__ == '__main__':
    fire.Fire(main)
