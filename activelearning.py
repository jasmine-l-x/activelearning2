
# Cell 1: Imports & Setup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm
import os 
import numpy as np
import random
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # For plotting

# FAISS for fast k-NN
import faiss

# Existing modules
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from utils.train_utils import simclr_train, selflabel_train, scan_train
from utils.config import create_config

# For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Cell 2: Configuration
config = {
    "pretext_checkpoint": "./checkpoints/simclr_cifar10.pth",
    "pretext_model": "./checkpoints/simclr_cifar10_model.pth",
    
    "batch_size": 256,
    "num_workers": 2,
    "simclr_epochs": 50,
    "simclr_lr": 1e-3,
    "temperature": 0.5,
    
    "clf_epochs": 10,
    "classifier_lr": 0.1,
    
    "semi_epochs": 10,
    "k_nn": 5,
    
    "budget": 10,
    "num_al_iterations": 2,
    
    "num_classes": 10,
    "num_heads": 1
}

print("Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")


# Cell 3: Data Prep
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=simclr_transform)
clf_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=classifier_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=classifier_transform)

trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=config["num_workers"])

print("Datasets loaded:")
print("SimCLR Training set size:", len(trainset))
print("Classifier Training set size:", len(clf_trainset))
print("Test set size:", len(testset))

import subprocess

# Check if pretrained checkpoint exists; if not, run simclr.py.
if not os.path.exists(config["pretext_checkpoint"]):
    print("Pretrained SimCLR model not found. Running simclr.py to pretrain the model...")
    subprocess.run([
        "python", "simclr.py",
        "--config_env", "configs/env.yml",
        "--config_exp", "


configs/pretext/simclr_cifar10.yml"
    ], check=True)
else:
    print("Pretrained SimCLR model found.")



# Cell 4: Model Definitions
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', projection_dim=128):
        super(SimCLR, self).__init__()
        if base_model == 'resnet18':
            backbone = models.resnet18(pretrained=False)
            num_ftrs = backbone.fc.in_features
            # Remove the final FC to get the encoder
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise NotImplementedError("Only resnet18 is supported.")
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, projection_dim)
        )
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        if return_features:
            return projections, features
        else:
            return projections

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10, pretrained_encoder=None):
        super(ResNet18Classifier, self).__init__()
        if pretrained_encoder is None:
            backbone = models.resnet18(pretrained=False)
            num_ftrs = backbone.fc.in_features
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.fc = nn.Linear(num_ftrs, num_classes)
        else:
            self.encoder = pretrained_encoder
            self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, return_features=False):
        feats = self.encoder(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.fc(feats)
        if return_features:
            return logits, feats
        else:
            return logits

simclr_model = SimCLR().to(device)
if torch.cuda.is_available():
    simclr_model = torch.nn.DataParallel(simclr_model)
    
if torch.cuda.is_available() and os.path.exists(config["pretext_checkpoint"]):
    print("Loading pretrained SimCLR model from checkpoint...")
    checkpoint = torch.load(config["pretext_checkpoint"], map_location=device)
    simclr_model.module.load_state_dict(checkpoint['model'])
else:
    print("No pretrained SimCLR checkpoint found; training from scratch is required.")
simclr_model.eval()

# Quick test
dummy_input = torch.randn(4, 3, 32, 32).to(device)
proj = simclr_model(dummy_input)
print("SimCLR model output shape:", proj.shape)


# Cell 5: Feature extraction, clustering, and sample selection
def extract_embeddings(encoder, dataloader):
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting Embeddings"):
            imgs = imgs.to(device)
            feats = encoder(imgs)
            feats = feats.view(feats.size(0), -1)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

def compute_typicality_faiss(embeddings, k_nn, eps=1e-6):
    embeddings = embeddings.astype('float32')
    N, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    distances, _ = index.search(embeddings, k_nn)
    avg_dists = np.mean(distances[:, 1:], axis=1)
    typicality = 1.0 / (avg_dists + eps)
    return typicality

def select_typical_samples(embeddings, cluster_labels, unlabeled_indices, labeled_indices, budget, k_nn):
    clusters = {}
    # Build dictionary: key = cluster label, value = list of original dataset indices
    for i, cl in enumerate(cluster_labels):
        orig_idx = unlabeled_indices[i]  # mapping: embeddings[i] corresponds to unlabeled_indices[i]
        clusters.setdefault(cl, []).append(orig_idx)
    
    # Sort clusters by number of elements,largest first
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    selected = []
    for cl, idxs in sorted_clusters:
        # Skip clusters that already have at least one labeled sample
        if any(i in labeled_indices for i in idxs):
            continue
        # Get the embeddings corresponding to these indices.
        rel_idxs = [int(np.where(unlabeled_indices == i)[0][0]) for i in idxs if np.where(unlabeled_indices == i)[0].size > 0]
        if not rel_idxs:
            continue
        cl_embeddings = embeddings[rel_idxs]
        typ_scores = compute_typicality_faiss(cl_embeddings, k_nn)
        best_local_idx = idxs[np.argmax(typ_scores)]
        selected.append(best_local_idx)
        if len(selected) >= budget:
            break
    
    if len(selected) < budget:
        remaining = list(set(unlabeled_indices) - set(selected) - set(labeled_indices))
        if remaining:
            # Find indices in embeddings corresponding to the remaining indices
            rem_idxs = [int(np.where(unlabeled_indices == i)[0][0]) for i in remaining if np.where(unlabeled_indices == i)[0].size > 0]
            rem_embeddings = embeddings[rem_idxs]
            rem_typ = compute_typicality_faiss(rem_embeddings, k_nn)
            sorted_remaining = [x for _, x in sorted(zip(rem_typ, remaining), key=lambda pair: pair[0], reverse=True)]
            additional = sorted_remaining[:(budget - len(selected))]
            selected.extend(additional)
    
    return np.array(selected)

def query_selection(encoder, dataloader, unlabeled_indices, labeled_indices, budget, config):
    # make sure unlabeled_indices is NumPy array.
    unlabeled_indices = np.array(unlabeled_indices)
    embeddings = extract_embeddings(encoder, dataloader)
    num_clusters = budget if len(labeled_indices) == 0 else len(labeled_indices) + budget
    cluster_labels = cluster_embeddings(embeddings, num_clusters)
    selected = select_typical_samples(
        embeddings,
        cluster_labels,
        unlabeled_indices,
        np.array(labeled_indices),
        budget,
        k_nn=config["k_nn"]
    )
    return selected


# Cell 6: Dataset wrappers
class LabeledDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.base_dataset[actual_idx]
        return image, label

class UnlabeledDatasetWeakStrong(Dataset):
    def __init__(self, base_dataset, indices, weak_transform, strong_transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, _ = self.base_dataset[actual_idx]
        img_weak = self.weak_transform(image) if self.weak_transform else image
        img_strong = self.strong_transform(image) if self.strong_transform else image
        return img_weak, img_strong


# Cell 7: Training Functions
def train_linear_on_embeddings(encoder, labeled_loader, test_loader, epochs=30, lr=0.1):
    encoder.eval()
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in labeled_loader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            feats = feats.view(feats.size(0), -1)
            embeddings.append(feats.cpu())
            labels_list.append(labels)
    embeddings = torch.cat(embeddings, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)
    
    input_dim = embeddings.size(1)
    linear_model = nn.Linear(input_dim, config["num_classes"]).to(device)
    optimizer = optim.SGD(linear_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    linear_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = linear_model(embeddings.to(device))
        loss = criterion(outputs, labels_tensor.to(device))
        loss.backward()
        optimizer.step()
        print(f"[Linear Train] Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
    
    linear_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            feats = feats.view(feats.size(0), -1)
            outputs = linear_model(feats)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels.to(device)).sum().item()
    test_acc = 100 * correct / total
    print(f"[Linear Evaluation] Test Accuracy: {test_acc:.2f}%")
    return linear_model, test_acc

def train_supervised_paper_style(model, train_loader, test_loader, epochs=30, lr=0.1):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = evaluate_classifier(model, test_loader)
        print(f"[Supervised Train] Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f} Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
    return model, best_acc

def evaluate_classifier(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100 * correct / total

def random_selection(unlabeled_indices, budget):
    return np.array(random.sample(list(unlabeled_indices), budget))

def train_selflabel(model, train_loader, test_loader, epochs=30, lr=0.1):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = evaluate_classifier(model, test_loader)
        print(f"[Self-label] Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f} Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
    return model, best_acc

def uncertainty_sampling(model, unlabeled_loader, budget):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for imgs, _ in unlabeled_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            uncertainty = 1 - probs.max(dim=1)[0]
            uncertainties.extend(uncertainty.cpu().numpy())
    sorted_indices = np.argsort(uncertainties)[::-1]
    return sorted_indices[:budget]


# Cell 8: Evaluation Metrics
def compute_tv_distance(dataset, indices, num_classes=10):
    labels = [dataset[i][1] for i in indices]
    counts = np.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    empirical = counts / (len(labels) + 1e-6)
    uniform = np.ones(num_classes) / num_classes
    tv = 0.5 * np.sum(np.abs(empirical - uniform))
    return tv

def compute_mean_std(metric_list):
    return np.mean(metric_list), np.std(metric_list)

def compute_confusion_matrix(model, dataloader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return confusion_matrix(all_labels, all_preds)

def active_learning_experiment(base_train_dataset, test_dataset, config, mode='supervised'):
    all_indices = np.arange(len(base_train_dataset))
    labeled_indices = []
    unlabeled_indices = all_indices.copy()
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=config["num_workers"])
    encoder = simclr_model.encoder
    encoder.eval()
    
    al_accuracies = []
    al_tv_distances = []
    clf_model = None  # For warm-starting
    
    for it in range(config["num_al_iterations"]):
        print(f"\n=== Active Learning Iteration {it+1}/{config['num_al_iterations']} ===")
        current_unlabeled_dataset = Subset(base_train_dataset, unlabeled_indices.tolist())
        unlabeled_loader = DataLoader(current_unlabeled_dataset, batch_size=config["batch_size"],
                                      shuffle=False, num_workers=config["num_workers"])
        
        # Always use K-means
        embeddings = extract_embeddings(encoder, unlabeled_loader)
        num_clusters = config["budget"] if len(labeled_indices)==0 else len(labeled_indices) + config["budget"]
        cluster_labels = cluster_embeddings(embeddings, num_clusters)
        
        selected = select_typical_samples(embeddings, cluster_labels,
                                          np.array(unlabeled_indices), np.array(labeled_indices),
                                          config["budget"], k_nn=config["k_nn"])
        print("Selected indices for query:", selected)
        
        labeled_indices.extend(selected.tolist())
        unlabeled_indices = np.setdiff1d(unlabeled_indices, selected)
        
        tv_distance = compute_tv_distance(base_train_dataset, labeled_indices, num_classes=config["num_classes"])
        al_tv_distances.append(tv_distance)
        print(f"TV Distance after iteration {it+1}: {tv_distance:.4f}")
        
        labeled_dataset = LabeledDataset(base_train_dataset, labeled_indices)
        train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=config["num_workers"])
        
        if mode == 'supervised':
            if clf_model is None:
                clf_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            clf_model, test_acc = train_supervised_paper_style(clf_model, train_loader, test_loader,
                                                               epochs=config["clf_epochs"],
                                                               lr=config["classifier_lr"])
        elif mode == 'unsupervised':
            clf_model, test_acc = train_linear_on_embeddings(encoder, train_loader, test_loader,
                                                             epochs=config["clf_epochs"],
                                                             lr=config["classifier_lr"])
        elif mode == 'semi-supervised':
            if clf_model is None:
                clf_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            clf_model, test_acc = train_selflabel(clf_model, train_loader, test_loader,
                                                  epochs=config["semi_epochs"],
                                                  lr=config["classifier_lr"])
        else:
            raise ValueError("Unknown mode")
        
        print(f"Iteration {it+1}: Test Accuracy = {test_acc:.2f}%")
        al_accuracies.append(test_acc)
    
    mean_acc, std_acc = compute_mean_std(al_accuracies)
    print(f"\nFinal {mode} AL Performance - Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")
    return al_accuracies, al_tv_distances


def baseline_random_experiment(base_train_dataset, test_dataset, config, mode='supervised'):
    all_indices = np.arange(len(base_train_dataset))
    labeled_indices = []
    unlabeled_indices = all_indices.copy()
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=config["num_workers"])
    encoder = simclr_model.encoder
    encoder.eval()
    
    al_accuracies = []
    for it in range(config["num_al_iterations"]):
        print(f"\n=== Random Baseline Iteration {it+1}/{config['num_al_iterations']} ===")
        selected = random_selection(unlabeled_indices, config["budget"])
        labeled_indices.extend(selected.tolist())
        unlabeled_indices = np.setdiff1d(unlabeled_indices, selected)
        
        labeled_dataset = LabeledDataset(base_train_dataset, labeled_indices)
        train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=config["num_workers"])
        
        if mode == 'supervised':
            clf_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            clf_model, test_acc = train_supervised_paper_style(clf_model, train_loader, test_loader,
                                                               epochs=config["clf_epochs"],
                                                               lr=config["classifier_lr"])
        elif mode == 'unsupervised':
            clf_model, test_acc = train_linear_on_embeddings(encoder, train_loader, test_loader,
                                                             epochs=config["clf_epochs"],
                                                             lr=config["classifier_lr"])
        elif mode == 'semi-supervised':
            clf_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            clf_model, test_acc = train_selflabel(clf_model, train_loader, test_loader,
                                                  epochs=config["semi_epochs"],
                                                  lr=config["classifier_lr"])
        else:
            raise ValueError("Unknown mode")
        print(f"Iteration {it+1}: Test Accuracy = {test_acc:.2f}%")
        al_accuracies.append(test_acc)
    
    mean_acc, std_acc = compute_mean_std(al_accuracies)
    print(f"\nRandom Baseline {mode} - Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")
    return al_accuracies


def baseline_uncertainty_experiment(base_train_dataset, test_dataset, config, mode='supervised'):
    all_indices = np.arange(len(base_train_dataset))
    labeled_indices = []
    unlabeled_indices = all_indices.copy()
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=config["num_workers"])
    encoder = simclr_model.encoder
    encoder.eval()
    
    al_accuracies = []
    for it in range(config["num_al_iterations"]):
        print(f"\n=== Uncertainty Baseline Iteration {it+1}/{config['num_al_iterations']} ===")
        current_unlabeled_dataset = Subset(base_train_dataset, unlabeled_indices.tolist())
        unlabeled_loader = DataLoader(current_unlabeled_dataset, batch_size=config["batch_size"],
                                      shuffle=False, num_workers=config["num_workers"])
        
        # Use uncertainty sampling :o
        if mode == 'supervised':
            temp_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            uncertainty_indices = uncertainty_sampling(temp_model, unlabeled_loader, config["budget"])
        else:
            uncertainty_indices = random_selection(unlabeled_indices, config["budget"])
        
        selected = uncertainty_indices
        labeled_indices.extend(selected.tolist())
        unlabeled_indices = np.setdiff1d(unlabeled_indices, selected)
        
        labeled_dataset = LabeledDataset(base_train_dataset, labeled_indices)
        train_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=config["num_workers"])
        
        if mode == 'supervised':
            clf_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            clf_model, test_acc = train_supervised_paper_style(clf_model, train_loader, test_loader,
                                                               epochs=config["clf_epochs"],
                                                               lr=config["classifier_lr"])
        elif mode == 'unsupervised':
            clf_model, test_acc = train_linear_on_embeddings(encoder, train_loader, test_loader,
                                                             epochs=config["clf_epochs"],
                                                             lr=config["classifier_lr"])
        elif mode == 'semi-supervised':
            clf_model = ResNet18Classifier(num_classes=config["num_classes"], pretrained_encoder=encoder).to(device)
            clf_model, test_acc = train_selflabel(clf_model, train_loader, test_loader,
                                                  epochs=config["semi_epochs"],
                                                  lr=config["classifier_lr"])
        else:
            raise ValueError("Unknown mode")
        print(f"Iteration {it+1}: Test Accuracy = {test_acc:.2f}%")
        al_accuracies.append(test_acc)
    
    mean_acc, std_acc = compute_mean_std(al_accuracies)
    print(f"\nUncertainty Baseline {mode} - Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")
    return al_accuracies


print("Starting Active Learning Experiment (Supervised Mode)...")
accs_supervised, tvs_supervised = active_learning_experiment(clf_trainset, testset, config, mode='supervised')

print("\nStarting Active Learning Experiment (Unsupervised Mode)...")
accs_unsupervised, tvs_unsupervised = active_learning_experiment(clf_trainset, testset, config, mode='unsupervised')

print("\nStarting Active Learning Experiment (Semi-Supervised Mode)...")
accs_semi, tvs_semi = active_learning_experiment(clf_trainset, testset, config, mode='semi-supervised')

print("\nStarting Random Baseline Experiment (Supervised Mode)...")
accs_random = baseline_random_experiment(clf_trainset, testset, config, mode='supervised')

print("\nStarting Uncertainty Sampling Baseline Experiment (Supervised Mode)...")
accs_uncertainty = baseline_uncertainty_experiment(clf_trainset, testset, config, mode='supervised')

print("\nDone with all experiments!")


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

summary_data = {
    "Method": ["AL - Supervised", "Random - Supervised", "Uncertainty - Supervised"],
    "Mean Accuracy (%)": [
        np.mean(accs_supervised),
        np.mean(accs_random),
        np.mean(accs_uncertainty)
    ],
    "Std Accuracy (%)": [
        np.std(accs_supervised),
        np.std(accs_random),
        np.std(accs_uncertainty)
    ],
    "Final Cycle Accuracy (%)": [
        accs_supervised[-1],
        accs_random[-1],
        accs_uncertainty[-1]
    ]
}
summary_df = pd.DataFrame(summary_data)
print("Summary Table of Results:")
print(summary_df)


# Cycle-wise Accuracy

plt.figure(figsize=(10, 6))
plt.plot(range(len(accs_supervised)), accs_supervised, marker='o', label='AL - Supervised')
plt.plot(range(len(accs_random)), accs_random, marker='o', label='Random - Supervised')
plt.plot(range(len(accs_uncertainty)), accs_uncertainty, marker='o', label='Uncertainty - Supervised')
plt.xlabel("Active Learning Cycle")
plt.ylabel("Test Accuracy (%)")
plt.title("Cycle-wise Test Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 3. Bar Chart of Final Cycle Accuracy

final_accuracies = [
    accs_supervised[-1],
    accs_random[-1],
    accs_uncertainty[-1]
]
methods = ["AL - Supervised", "Random - Supervised", "Uncertainty - Supervised"]

plt.figure(figsize=(8, 5))
plt.bar(methods, final_accuracies, color=['steelblue', 'orange', 'green'])
plt.ylabel("Accuracy (%)")
plt.title("Final Cycle Accuracy Comparison")
plt.grid(axis='y')
plt.show()

# Plot TV Distance 

if 'tvs_supervised' in globals() and len(tvs_supervised) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(tvs_supervised)), tvs_supervised, marker='o', color='purple')
    plt.xlabel("Active Learning Cycle")
    plt.ylabel("TV Distance")
    plt.title("TV Distance Over AL Cycles (Supervised AL)")
    plt.grid(True)
    plt.show()
else:
    print("TV Distance data (tvs_supervised) not available or empty.")

# a) T-test: AL - Supervised vs. Random - Supervised
t_stat_1, p_value_1 = ttest_ind(accs_supervised, accs_random)
print("\nStatistical Analysis (t-test) between AL - Supervised and Random - Supervised:")
print(f"t-statistic: {t_stat_1:.4f}, p-value: {p_value_1:.4f}")
alpha = 0.05
if p_value_1 < alpha:
    print(f"p < {alpha}. The difference is statistically significant.")
else:
    print(f"p >= {alpha}. The difference is not statistically significant at alpha={alpha}.")

# b) T-test: AL - Supervised vs. Uncertainty - Supervised (optional)
t_stat_2, p_value_2 = ttest_ind(accs_supervised, accs_uncertainty)
print("\nStatistical Analysis (t-test) between AL - Supervised and Uncertainty - Supervised:")
print(f"t-statistic: {t_stat_2:.4f}, p-value: {p_value_2:.4f}")
if p_value_2 < alpha:
    print(f"p < {alpha}. The difference is statistically significant.")
else:
    print(f"p >= {alpha}. The difference is not statistically significant at alpha={alpha}.")


