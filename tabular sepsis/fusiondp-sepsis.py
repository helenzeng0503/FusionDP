import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import argparse
# import wandb
import json
import random
from opacus import PrivacyEngine  # Importing PrivacyEngine from Opacus for differential privacy
from opacus.accountants import create_accountant
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from sklearn.preprocessing import StandardScaler

# ------------- imports ------------------------------------------
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_map, tree_flatten
# -----------------------------------------------------------------




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set a fixed random seed for reproducibility

MAX_SIGMA = 1e6

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(66)

def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    epochs: int = 10,
    alpha_range=None  # Add alpha_range parameter
):
    # Default alpha range if not provided
    if alpha_range is None:
        alpha_range = [1 + x / 10.0 for x in range(1, 1000)] + list(range(101, 10001, 100))  # Default: 1.1 to 100.0

    steps = int(epochs * (1 / sample_rate))  # Correct step calculation

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high *= 2
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=sigma_high,
            steps=steps,
            orders=alpha_range  # Use the alpha_range here
        )
        eps_high, _ = get_privacy_spent(orders=alpha_range, rdp=rdp, delta=target_delta)

        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=sigma,
            steps=steps,
            orders=alpha_range  # Use the alpha_range here
        )
        eps = get_privacy_spent(orders=alpha_range, rdp=rdp, delta=target_delta)[0]

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high



def prepare_data(data, batch_size):

    
    # train = data[acols]
    train = data.dropna()
    y_train = train['SepsisLabel'].astype('int').to_numpy()
    X_train = train.drop(columns=['SepsisLabel']).to_numpy()

    train_dataset = EicuDataset(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader


class EicuDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feats = self.x[idx].to(DEVICE)
        target = self.y[idx].to(DEVICE)
        return feats, target


class MLP(nn.Module):
    def __init__(self, l):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_features=l, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=1)

        self.gelu = nn.GELU()

        # Dropout layers
        self.dropout = nn.Dropout(p=0.15)

        # LayerNorm layers
        self.bn1 = nn.LayerNorm(64)
        self.bn2 = nn.LayerNorm(128)
        self.bn3 = nn.LayerNorm(64)

    def forward(self, x, *, return_rep: bool = False):
        hidden = self.gelu(self.bn1(self.linear1(x)))
        hidden = self.dropout(hidden)

        hidden = self.gelu(self.bn2(self.linear2(hidden)))
        hidden = self.dropout(hidden)

        hidden = self.gelu(self.bn3(self.linear3(hidden)))

        if return_rep:
            return hidden

        logits = self.linear4(hidden)

        return logits

    def get_rep(self, x):
        with torch.no_grad():
            return self.forward(x, return_rep=True)


def dp_sgd_update(model, optimizer,
                           batch, batch_labels,
                           loss_fn,
                           max_grad_norm,
                           noise_multiplier):
    
    # Get model parameters and buffers
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    B = batch.size(0)

    # per‑sample loss using functional_call
    def _loss(p, b, x, y):
        logits = functional_call(model, (p, b), x.unsqueeze(0))
        return loss_fn(logits.squeeze(), y)  # scalar

    # batched per‑sample grads
    per_ex_grads = vmap(
            torch.func.grad(_loss),                     # gradient fn
            in_dims=(None, None, 0, 0),      # map over batch dim of x, y
            randomness="different"           # <- allow Dropout per‑sample
    )(params, buffers, batch, batch_labels.float())

    # Convert to same structure as first version (list of per-sample gradients)
    # per_ex_grads is a dict with parameter names as keys, each containing gradients for all samples
    param_names = list(per_ex_grads.keys())
    per_sample_gradients = []
    
    for i in range(B):
        # Get gradients for sample i across all parameters
        sample_grads = [per_ex_grads[param_name][i] for param_name in param_names]
        per_sample_gradients.append(sample_grads)

    # Clip exactly like first version
    for i, grads in enumerate(per_sample_gradients):
        total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            per_sample_gradients[i] = [g.mul_(clip_coef) for g in grads]

    # Aggregate exactly like first version
    aggregated_grads = [
        torch.sum(torch.stack([grads[i] for grads in per_sample_gradients]), dim=0)
        for i in range(len(per_sample_gradients[0]))
    ]

    # Add noise exactly like first version
    for i, grad in enumerate(aggregated_grads):
        noise = torch.normal(0, noise_multiplier * max_grad_norm, size=grad.shape, device=grad.device)
        aggregated_grads[i] = grad / B + noise / B

    return aggregated_grads


def dp_sgd_dif(
        model, optimizer,
        batch, batch_labels,
        imp_batch, imp_labels,
        loss_fn,
        max_grad_norm,            # C
        noise_multiplier):        # σ

    # 1. equalise batch length
    B = min(batch.size(0), imp_batch.size(0))
    batch, batch_labels   = batch[:B], batch_labels[:B]
    imp_batch, imp_labels = imp_batch[:B], imp_labels[:B]

    # 2. functional model
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def _loss(p, b, x, y):
        logit = functional_call(model, (p, b), x.unsqueeze(0))
        return loss_fn(logit.squeeze(), y)          # scalar

    # batched per‑sample grads
    g_real = vmap(
            torch.func.grad(_loss),                     # gradient fn
            in_dims=(None, None, 0, 0),      # map over batch dim of x, y
            randomness="different"           # <- allow Dropout per‑sample
    )(params, buffers, batch, batch_labels.float())
    g_imp  = vmap(
            torch.func.grad(_loss),
            in_dims=(None,None,0,0), 
            randomness="different"
            )(params, buffers, imp_batch, imp_labels.float())

    # Convert to same structure as first version (list of per-sample gradients)
    param_names = list(g_real.keys())
    
    # Convert g_real and g_imp to list format
    per_sample_gradients = []
    imp_per_sample_gradients = []
    
    for i in range(B):
        # Real gradients for sample i
        real_sample_grads = [g_real[param_name][i] for param_name in param_names]
        per_sample_gradients.append(real_sample_grads)
        
        # Imputed gradients for sample i  
        imp_sample_grads = [g_imp[param_name][i] for param_name in param_names]
        imp_per_sample_gradients.append(imp_sample_grads)

    # Compute difference exactly like first version
    per_sample_grads_dif = []
    for i in range(len(per_sample_gradients)):
        grad_diff = [
            grad_a - grad_b
            for grad_a, grad_b in zip(per_sample_gradients[i], imp_per_sample_gradients[i])
        ]
        per_sample_grads_dif.append(grad_diff)

    # Clip exactly like first version
    for i, grads in enumerate(per_sample_grads_dif):
        total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            per_sample_grads_dif[i] = [g * clip_coef for g in grads]

    # Aggregate exactly like first version
    aggregated_grads = [
        torch.sum(torch.stack([grads[i] for grads in per_sample_grads_dif]), dim=0)
        for i in range(len(per_sample_grads_dif[0]))
    ]

    # Add noise exactly like first version
    for i, grad in enumerate(aggregated_grads):
        noise = torch.normal(0, noise_multiplier * max_grad_norm, size=grad.shape, device=grad.device)
        aggregated_grads[i] = grad / B + noise / B  # Note: using B (min batch size) like first version

    return aggregated_grads


# --- Representation Consistency Loss (normalised) ---
def rep_consistency_loss(h_real, h_imp, C_h, beta):
    """Per‑sample representation alignment loss.
    After clipping, the squared L2 gap is at most C_h**2.  We divide by
    C_h**2 so the maximum value is ≈ 1, which makes `beta` an intuitive
    weight (0 – 1) in the total private loss.
    Returns a tensor of shape (B,).
    """
    delta  = h_real - h_imp                              # (B, d)
    norms  = delta.norm(p=2, dim=1, keepdim=True)        # (B, 1)
    scale  = torch.clamp(C_h / (norms + 1e-12), max=1.0) # clip factor
    delta_clip = delta * scale                           # locally clipped Δh

    corr_i = delta_clip.pow(2).sum(dim=1)                # ≤ C_h**2
    corr_i = corr_i / (C_h ** 2)                        # ≤ 1 after norm

    return beta * corr_i



def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch, labels in dataloader:
            batch, labels = batch.cuda(), labels.cuda()
            logits = model(batch).squeeze()
            # Apply sigmoid to get probabilities for evaluation
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    auroc = roc_auc_score(all_labels, all_preds)
    aupr = average_precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return accuracy, auroc, aupr, f1


def project_to_W(params, M):
    total_norm = torch.norm(torch.cat([p.view(-1) for p in params]))
    if total_norm <= M:
        return
    scale = M / total_norm
    for p in params:
        p.data.mul_(scale)

def feature_dp_update_sample(model, X, X_pub, Y,
                      loss_fn,
                      max_grad_norm,
                      noise_multiplier,
                      alpha=1.0,
                      M=10.0,
                      batch_priv_size=128,
                      batch_pub_size=4096):

    N = X.shape[0]

    # --- Poisson sampling for B_priv ---
    p_poisson = batch_priv_size / N
    mask = (torch.rand(N, device=DEVICE) < p_poisson)

    B_priv = X[mask]
    B_priv_pub = X_pub[mask]
    B_priv_labels = Y[mask]

    # --- Uniform sampling for B_pub ---
    perm = torch.randperm(N, device=DEVICE)[:batch_pub_size]

    B_pub = X[perm]
    B_pub_pub = X_pub[perm]
    B_pub_labels = Y[perm]

    # --- Public gradient ---
    model.zero_grad()
    logits_pub = model(B_pub_pub).squeeze()
    loss_pub = loss_fn(logits_pub, B_pub_labels).mean()
    loss_pub.backward()

    grad_pub = [param.grad.detach().clone() for param in model.parameters()]
    model.zero_grad()

    # --- Private gradient (per-sample) ---
    def _loss_fn(params, buffers, x, x_pub, y):
        logits_total = functional_call(model, (params, buffers), x.unsqueeze(0)).squeeze()
        logits_pub = functional_call(model, (params, buffers), x_pub.unsqueeze(0)).squeeze()
        return loss_fn(logits_total, y) - loss_fn(logits_pub, y)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    grads = vmap(torch.func.grad(_loss_fn), in_dims=(None, None, 0, 0, 0), randomness="different")(
        params, buffers, B_priv, B_priv_pub, B_priv_labels
    )

    # Convert grads dict to list-of-lists per sample
    param_names = list(grads.keys())
    per_sample_grads = []
    for i in range(B_priv.size(0)):
        sample_grads = [grads[name][i] for name in param_names]
        per_sample_grads.append(sample_grads)

    # --- Clip per-sample grads ---
    for i, g in enumerate(per_sample_grads):
        total_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in g))
        scale = min(1.0, max_grad_norm / (total_norm + 1e-6))
        per_sample_grads[i] = [p * scale for p in g]

    # --- Aggregate and add noise ---
    num_params = len(per_sample_grads[0])
    aggregated_grads = []
    for i in range(num_params):
        stacked = torch.stack([grads[i] for grads in per_sample_grads])
        summed = stacked.sum(dim=0)
        noise = torch.normal(
            mean=0,
            std=noise_multiplier * max_grad_norm,
            size=summed.shape,
            device=summed.device
        )
        aggregated_grads.append(summed + noise)

    # Scale by number of samples to get average
    aggregated_grads = [g / len(per_sample_grads) for g in aggregated_grads]

    # --- Combine public + private gradient ---
    total_grads = [
        grad_pub[i] + alpha * aggregated_grads[i]
        for i in range(len(aggregated_grads))
    ]

    return total_grads  

def calibrate4(model, X, X_pub, X_noise, Y,
               loss_fn,
               max_grad_norm,
               noise_multiplier,
               C_h,
               beta,
               alpha,
               batch_priv_size=128,
               batch_pub_size=4096,
               device='cuda'
               ):

    N = X.shape[0]

    # --- Poisson sampling for B_priv ---
    p_poisson = batch_priv_size / N
    mask = (torch.rand(N, device=device) < p_poisson)

    B_priv = X[mask]
    B_priv_pub = X_pub[mask]
    B_priv_labels = Y[mask]


    # --- Uniform sampling for B_pub ---
    perm = torch.randperm(N, device=device)[:batch_pub_size]

    B_pub = X[perm]
    B_pub_pub = X_pub[perm]
    B_pub_labels = Y[perm]

    # --- Public gradient ---
    model.zero_grad()
    logits_pub = model(B_pub_pub).squeeze()
    loss_pub = loss_fn(logits_pub, B_pub_labels).mean()
    loss_pub.backward()

    grad_pub = [param.grad.detach().clone() for param in model.parameters()]
    model.zero_grad()

    # --- Private gradient (per-sample) ---
    def _loss_fn(params, buffers, x, x_pub, y):
        logits_total = functional_call(model, (params, buffers), x.unsqueeze(0)).squeeze()
        logits_pub = functional_call(model, (params, buffers), x_pub.unsqueeze(0)).squeeze()

        h_r = functional_call(model, (params, buffers), (x.unsqueeze(0),), {'return_rep': True}).squeeze(0)
        h_i = functional_call(model, (params, buffers), (x_pub.unsqueeze(0),), {'return_rep': True}).squeeze(0)
        
        corr_i = rep_consistency_loss(
            h_r.unsqueeze(0), h_i.unsqueeze(0), C_h, beta
        ).squeeze()

        return loss_fn(logits_total, y) - loss_fn(logits_pub, y) + corr_i

    # Get parameters and buffers as dictionaries
    params = {name: param for name, param in model.named_parameters()}
    buffers = {name: buffer for name, buffer in model.named_buffers()}
    
    # Handle case where B_priv might be empty
    if B_priv.size(0) == 0:
        # If no private samples, return only public gradient
        return grad_pub
    
    grads = vmap(torch.func.grad(_loss_fn), in_dims=(None, None, 0, 0, 0), randomness="different")(
    params, buffers, B_priv, B_priv_pub, B_priv_labels
)


    # Convert grads dict to list-of-lists per sample
    param_names = list(params.keys())  # Use params.keys() instead of grads.keys()
    per_sample_grads = []
    for i in range(B_priv.size(0)):
        sample_grads = [grads[name][i] for name in param_names]
        per_sample_grads.append(sample_grads)

    # --- Clip per-sample grads ---
    for i, g in enumerate(per_sample_grads):
        total_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in g))
        scale = min(1.0, max_grad_norm / (total_norm + 1e-6))
        per_sample_grads[i] = [p * scale for p in g]

    # --- Aggregate and add noise ---
    num_params = len(per_sample_grads[0])
    aggregated_grads = []
    for i in range(num_params):
        stacked = torch.stack([grads[i] for grads in per_sample_grads])
        summed = stacked.sum(dim=0)
        noise = torch.normal(
            mean=0,
            std=noise_multiplier * max_grad_norm,
            size=summed.shape,
            device=summed.device
        )
        aggregated_grads.append(summed + noise)

    # Scale by number of samples to get average
    aggregated_grads = [g / len(per_sample_grads) for g in aggregated_grads]

    # --- Combine public + private gradient ---
    total_grads = [
        grad_pub[i] + alpha * aggregated_grads[i]
        for i in range(len(aggregated_grads))
    ]

    return total_grads  


def main():
    # Parse command-line arguments within main
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    
    parser.add_argument("--mode", type=str, required=True,
                        help="Training mode.")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="epsilon for differential privacy.")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Total epochs")
    parser.add_argument("--max_grad_norm", type=float, required=True,
                        help="Maximum gradient norm for clipping.")
    parser.add_argument("--beta", type=float, required=False,
                        help="representation norm weight")
    parser.add_argument("--alpha", type=float, required=False,
                        help="loss difference weight")
    
    args = parser.parse_args()

    mode = args.mode
    epsilon = args.epsilon
    delta = 1e-5
    max_grad_norm = args.max_grad_norm
    beta = args.beta if args.beta is not None else 0
    alpha = args.alpha if args.alpha is not None else 0

    
    lr = 0.07
    epochs = args.epochs
    batch_size = 128

    dataa = pd.read_csv(filepath_or_buffer="train_15pos_ori718.csv")
    data_filled = pd.read_csv(filepath_or_buffer="train_15pos_imputed718.csv")
    atest = pd.read_csv(filepath_or_buffer="test_15pos718.csv")

    sens_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
    acols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
        'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
        'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Lactate', 'Magnesium',
        'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC',
        'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
        'SepsisLabel']
    
    input_dim = len(acols) - 1

    dataa = dataa[acols]
    atest = atest[acols]
    data_filled = data_filled[acols]
    data_filled0 = data_filled.copy()
    data_filled0[sens_cols] = np.random.normal(loc=0.0, scale=1.0, size=(len(data_filled0), len(sens_cols)))



    y_dataa = dataa["SepsisLabel"]
    y_atest = atest['SepsisLabel']
    y_fill = data_filled['SepsisLabel']
    y_fill0 = data_filled0['SepsisLabel']

    scaler_ori = StandardScaler()
    # Fill NaNs with column means before scaling
    dataa = dataa.fillna(dataa.mean(numeric_only=True))
    scaler_ori.fit(dataa)
    dataa = pd.DataFrame(scaler_ori.transform(dataa), columns=dataa.columns)

    scaler_imp = StandardScaler()
    data_filled = data_filled.fillna(data_filled.mean(numeric_only=True))
    scaler_imp.fit(data_filled)
    data_filled = pd.DataFrame(scaler_imp.transform(data_filled), columns=data_filled.columns)

    # if mode in {"dpsgd", "sgd_ori"}:
    atest = atest.fillna(atest.mean(numeric_only=True))
    atest = pd.DataFrame(scaler_ori.transform(atest), columns=atest.columns)
    # else:
    #     atest = pd.DataFrame(scaler_imp.transform(atest), columns=atest.columns)
    scaler_imp0 = StandardScaler()
    data_filled0 = data_filled0.fillna(data_filled0.mean(numeric_only=True))
    scaler_imp0.fit(data_filled0)
    data_filled0 = pd.DataFrame(scaler_imp0.transform(data_filled0), columns=data_filled0.columns)



    dataa["SepsisLabel"] = y_dataa
    atest['SepsisLabel'] = y_atest
    data_filled['SepsisLabel'] = y_fill
    data_filled0['SepsisLabel'] = y_fill0
    print(dataa.shape)
    print(atest.shape)


    

    dataloader = prepare_data(dataa, batch_size)
    filled_dataloader = prepare_data(data_filled, batch_size)
    test_dataloader = prepare_data(atest, batch_size)
    filled0_dataloader = prepare_data(data_filled0, batch_size)

    model = MLP(input_dim).cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')


    noise_multiplier = 0

    cur_result = {
        "mode": mode,
        "epsilon": epsilon,
        "delta": delta,
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": noise_multiplier,
        "final_aupr": 0.0,  # Initialize with 0
        "epoch": epochs
    }

    sample_rate = batch_size / len(dataa)

    best_aupr = 0

    C_h = None

    X_tensor = torch.tensor(dataa.drop(columns=["SepsisLabel"]).values, dtype=torch.float32, device=DEVICE)
    X_pub_tensor0 = torch.tensor(data_filled0.drop(columns=["SepsisLabel"]).values, dtype=torch.float32, device=DEVICE)
    
    X_pub_tensor = torch.tensor(data_filled.drop(columns=["SepsisLabel"]).values, dtype=torch.float32, device=DEVICE)
    Y_tensor = torch.tensor(dataa["SepsisLabel"].values, dtype=torch.float32, device=DEVICE)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch, labels), (filled_batch, filled_labels), (filled0_batch, filled0_labels) in zip(dataloader, filled_dataloader, filled0_dataloader):
            batch, labels = batch.cuda(), labels.cuda()
            filled_batch, filled_labels = filled_batch.cuda(), filled_labels.cuda()

            if mode == "dpsgd":
                # get_noise_multiplier
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)
                dp_gradients = dp_sgd_update(model, optimizer, batch, labels, loss_fn,  max_grad_norm, noise_multiplier)
                for i, param in enumerate(model.parameters()):
                    param.grad = dp_gradients[i]
            elif mode == "sgd_ori":
                optimizer.zero_grad()
                outputs = model(batch)
                loss = loss_fn(outputs.squeeze(), labels).mean()
                loss.backward()     
            elif mode == "sgd_hybrid":
                optimizer.zero_grad()
                outputs = model(filled_batch)
                loss = loss_fn(outputs.squeeze(), filled_labels).mean()
                loss.backward()
            elif mode == "sgd_pub":
                optimizer.zero_grad()
                outputs = model(filled0_batch)
                loss = loss_fn(outputs.squeeze(), filled0_labels).mean()
                loss.backward()
            elif mode == "naive_fusion":
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)

                N = X_tensor.shape[0]
                p_poisson = batch_size / N
                mask = (torch.rand(N, device=DEVICE) < p_poisson)

                B_priv = X_tensor[mask]
                # B_priv_pub = X_pub_tensor[mask]
                B_priv_labels = Y_tensor[mask]

                perm = torch.randperm(N, device=DEVICE)[:4096]

                # B_pub = X[perm]
                B_pub_pub = X_pub_tensor[perm]
                B_pub_labels = Y_tensor[perm]

                dp_gradients = dp_sgd_update(model, optimizer, B_priv, B_priv_labels, loss_fn,  max_grad_norm, noise_multiplier)
                optimizer.zero_grad()
                filled_outputs = model(B_pub_pub)
                filled_loss = loss_fn(filled_outputs.squeeze(), B_pub_labels).mean()
                filled_loss.backward()
                sgd_gradients = [p.grad.clone() for p in model.parameters()]
                for i, param in enumerate(model.parameters()):
                    param.grad = beta * dp_gradients[i] + (1-beta) * sgd_gradients[i]
            elif mode == "naive_fusion_pub":
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)

                N = X_tensor.shape[0]
                p_poisson = batch_size / N
                mask = (torch.rand(N, device=DEVICE) < p_poisson)

                B_priv = X_tensor[mask]
                # B_priv_pub = X_pub_tensor[mask]
                B_priv_labels = Y_tensor[mask]

                perm = torch.randperm(N, device=DEVICE)[:4096]

                # B_pub = X[perm]
                B_pub_pub = X_pub_tensor0[perm]
                B_pub_labels = Y_tensor[perm]

                dp_gradients = dp_sgd_update(model, optimizer, B_priv, B_priv_labels, loss_fn,  max_grad_norm, noise_multiplier)
                optimizer.zero_grad()
                filled_outputs = model(B_pub_pub)
                filled_loss = loss_fn(filled_outputs.squeeze(), B_pub_labels).mean()
                filled_loss.backward()
                sgd_gradients = [p.grad.clone() for p in model.parameters()]
                for i, param in enumerate(model.parameters()):
                    param.grad = beta * dp_gradients[i] + (1-beta) * sgd_gradients[i]
            elif mode == "feature-dp":
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)

                M = 10 #
                alpha0 = alpha # private loss weight

                dp_gradients = feature_dp_update_sample(model=model, X=X_tensor, X_pub=X_pub_tensor0, Y=Y_tensor,
                                         loss_fn=loss_fn,
                                         max_grad_norm=max_grad_norm,
                                         noise_multiplier=noise_multiplier,
                                         alpha=alpha0,
                                         M=M,
                                         batch_priv_size=128,
                                         batch_pub_size=4096)
        
                for i, param in enumerate(model.parameters()):
                    param.grad = dp_gradients[i]

            elif mode == "calibrated_fusion":
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)

                M = 10 # should be tuned, sec 6.2: M = maxw∈W | w |
                alpha0 = alpha # private loss weight


                dp_gradients = feature_dp_update_sample(model=model, X=X_tensor, X_pub=X_pub_tensor, Y=Y_tensor,
                                         loss_fn=loss_fn,
                                         max_grad_norm=max_grad_norm,
                                         noise_multiplier=noise_multiplier,
                                         alpha=alpha0,
                                         M=M,
                                         batch_priv_size=128,
                                         batch_pub_size=4096)
        
                for i, param in enumerate(model.parameters()):
                    param.grad = dp_gradients[i]

            elif mode == "fusiondp":

                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)

                if C_h is None:
                    with torch.no_grad():
                        norms = []
                        h_ori  = model.get_rep(batch.to(DEVICE))
                        h_imp  = model.get_rep(filled_batch.to(DEVICE))
                        norms.append((h_ori - h_imp).norm(dim=1).cpu())
                        norms = torch.cat(norms)
                        C_h   = norms.quantile(0.90).item()        # 90‑th percentile
                    print("C_h is ", C_h)


                alpha0 = alpha
                # print(alpha0)
                # print(beta)
                dp_gradients = calibrate4(model=model, X=X_tensor, X_pub=X_pub_tensor, X_noise=X_pub_tensor0, Y=Y_tensor,
                                        loss_fn=loss_fn,
                                        max_grad_norm=max_grad_norm,
                                        noise_multiplier=noise_multiplier,
                                        C_h=C_h,
                                        beta=beta,
                                        alpha=alpha0,
                                        batch_priv_size=128,
                                        batch_pub_size=4096)
                for i, param in enumerate(model.parameters()):
                    param.grad = dp_gradients[i]
                
            
            optimizer.step()
            # if mode in ("feature-dp", "feature-dp-imp"):
            M = 10.0
            project_to_W([p.data for p in model.parameters()], M)

        accuracy, auroc, aupr, f1 = evaluate(model, test_dataloader)

        if mode == "sgd_ori" and aupr > best_aupr:
            best_aupr = aupr
            torch.save(model.state_dict(), 'mlp_ori_model.pth')

        # Calculate privacy spent
        total_steps = (epoch + 1) * (1 / sample_rate) 
        if noise_multiplier > 0:
            rdp = compute_rdp(
                q=sample_rate,
                noise_multiplier=noise_multiplier,
                steps=total_steps,
                orders=[1 + x / 10.0 for x in range(1, 1000)]  # Alpha range
            )
            epsilon, _ = get_privacy_spent(
                orders=[1 + x / 10.0 for x in range(1, 1000)],
                rdp=rdp,
                delta=delta
            )
            print(f"Epoch {epoch + 1}/{epochs}, Privacy Spent: Epsilon = {epsilon:.2f}, Delta = {delta}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Privacy Spent: Noise multiplier not set.")

        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}, AUROC: {auroc}, AUPR: {aupr}, F1: {f1}")

         # Update latest result for the current combination
        cur_result["final_aupr"] = aupr
        cur_result["noise_multiplier"] = noise_multiplier
        cur_result["beta"] = beta
        cur_result["alpha"] = alpha

    results_file = "results.json"
    try:
        # Load existing results if file exists
        with open(results_file, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        # Initialize an empty list if file doesn't exist
        all_results = []

    # Append the current combination's final result
    all_results.append(cur_result)

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"final result for this combination saved: {cur_result}")



if __name__ == "__main__":
    main()