import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors, Word2Vec
from pathlib import Path
import pandas as pd
import json
from opacus.accountants import create_accountant
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from torch.func import functional_call, vmap
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import argparse
from opacus import PrivacyEngine
import random




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_SIGMA = 1e6

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# def build_embedding_matrix(vocab: dict, wv, embedding_dim: int = 100):
#     """
#     Builds an embedding matrix using provided KeyedVectors object.

#     Args:
#         vocab (dict): word -> index mapping
#         wv: KeyedVectors object (e.g., model.wv from Word2Vec)
#         embedding_dim (int): embedding size

#     Returns:
#         np.ndarray: embedding_matrix aligned with vocab indices
#     """
#     print("Building embedding matrix...")
#     embedding_matrix = np.zeros((len(vocab), embedding_dim))
#     not_found = []

#     for word, idx in vocab.items():
#         if word in wv:
#             embedding_matrix[idx] = wv[word]
#         elif word.lower() in wv:
#             embedding_matrix[idx] = wv[word.lower()]
#         else:
#             not_found.append(word)
#             embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

#     print(f"✅ Embedding matrix built: {len(vocab) - len(not_found)} found, {len(not_found)} not found.")
#     return embedding_matrix



def load_vocab(vocab_path: str) -> dict:
    """
    Loads vocab file into a word -> index dictionary.
    Supports files with one word per line (auto-indexed).
    """
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            word = line.strip()
            if word:  # avoid empty lines
                vocab[word] = i
    return vocab



# # --- Main Script ---

# # Load vocab
vocab = load_vocab("../mimicdata/mimic3/vocab.csv")
# # with open(r"C:\Users\Zack\Documents\vdp\629\mimic3\mimicdata\mimic3\vocab.csv", "r", encoding="utf-8") as f:
# #     for i, line in enumerate(f):
# #         print(line.strip())
# #         if i > 10:
# #             break

# model_path = "w2v_oa_all_100d.bin"
# wv = KeyedVectors.load(model_path, mmap='r') 

# # Build embedding matrix
# embedding_matrix = build_embedding_matrix(vocab=vocab, wv=wv.wv, embedding_dim=100)


# # Sanity check
# tokens_to_check = ["patient", "history", "unknown", "cardiology"]
# for token in tokens_to_check:
#     if token in vocab:
#         idx = vocab[token]
#         emb = embedding_matrix[idx]
#         print(f"{token} @ {idx} → mean: {emb.mean():.4f}, std: {emb.std():.4f}")
#     else:
#         print(f"{token} not in vocab")

# print("saving embedding matrix")
# np.save("embedding_matrix100.npy", embedding_matrix)



# # Load top 50 codes (with or without header)
# top_codes = pd.read_csv("../mimicdata/mimic3/top_50_codes.csv", header=None)[0].astype(str)

# # Build ICD9 → index mapping
# label_map = {code: idx for idx, code in enumerate(top_codes)}

# # Save to lookup.json
# with open("lookup.json", "w") as f:
#     json.dump(label_map, f, indent=2)

# print("✅ lookup.json created with", len(label_map), "codes.")


print("loading embedding matrix")
embedding_matrix = np.load("embedding_matrix.npy")
print("finish loading")


# To load into PyTorch model later:
# model.embed.weight.data.copy_(torch.tensor(embedding_matrix))
# model.embed.weight.requires_grad = False



label_map = json.load(open("lookup.json"))

class MimicNoteDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, vocab, label_map, max_len=1024):
        df = pd.read_csv(csv_path)
        self.text = df.TEXT.tolist()
        self.label_map = label_map

        self.labels = []
        for row in df.LABELS:
            filtered = [label_map[c] for c in row.split(';') if c in label_map]
            self.labels.append(filtered)

        self.tokeniser = lambda t: [vocab.get(tok, vocab.get('<unk>', 0)) for tok in t.split()][:max_len]

    def __getitem__(self, idx):
        x = torch.tensor(self.tokeniser(self.text[idx]), dtype=torch.long)
        y = torch.zeros(len(self.label_map))
        y[self.labels[idx]] = 1
        return x, y

    def __len__(self):
        return len(self.labels)
    
    




class DP_CAML(nn.Module):
    def __init__(self, vocab_size, num_labels,
                 d_model=300, filter_size=10, pretrained_embed=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embed is not None:
            self.embed.weight.data.copy_(pretrained_embed)
        self.conv = nn.Conv1d(d_model, d_model, filter_size, padding=filter_size-1)
        # self.dropout = nn.Dropout(0.3)
        self.U = nn.Parameter(torch.randn(num_labels, d_model))   # label queries
        self.fc_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, ids, return_hidden=False):
        """
        ids : (B, T) padded token IDs
        returns:
            logits: (B, L)
            hidden: (B, L, d_model) if return_hidden is True
        """
        e = self.embed(ids)                         # (B, T, d)
        h = F.relu(self.conv(e.transpose(1, 2)))    # (B, d, T)
        # h = self.dropout(F.relu(self.conv(e.transpose(1, 2))))

        attn = torch.einsum('ld,bdt->blt', self.U, h)  # (B, L, T)
        attn = attn.softmax(-1)

        z = torch.einsum('blt,bdt->bld', attn, h)      # (B, L, d)
        logits = (z * self.U).sum(-1) + self.fc_bias   # (B, L)

        return (logits, z) if return_hidden else logits

    def get_rep(self, ids):
        """Return attention-based document representation (used for rep alignment)."""
        _, z = self.forward(ids, return_hidden=True)
        return z.mean(dim=1)  # shape: (B, d)

    





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



def dp_sgd_update_microbatch(model, optimizer,
                              batch, batch_labels,
                              loss_fn,
                              max_grad_norm,
                              noise_multiplier):
    B = batch.size(0)
    param_names = list(dict(model.named_parameters()).keys())
    named_params = dict(model.named_parameters())
    device = batch.device

    # Storage for per-sample gradients
    per_sample_gradients = [
        torch.zeros((B, *p.shape), device=device)
        for p in model.parameters()
    ]

    # Compute gradients one sample at a time
    for i in range(B):
        model.zero_grad()
        x_i = batch[i].unsqueeze(0)        # shape: [1, T]
        y_i = batch_labels[i].unsqueeze(0) # shape: [1, L]
        logits = model(x_i)                # shape: [1, L]
        loss = loss_fn(logits, y_i)
        loss = loss.mean(dim=1)
        loss.backward()
        # print("Grad norm of embed:", model.embed.weight.grad.norm().item())


        for j, p in enumerate(model.parameters()):
            if p.grad is not None:
                per_sample_gradients[j][i] = p.grad.detach().clone()


    # Clip per-sample gradients
    for i in range(B):
        total_norm = torch.sqrt(sum(torch.sum(g[i] ** 2) for g in per_sample_gradients))
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in per_sample_gradients:
                g[i] *= clip_coef

    # Aggregate with noise
    aggregated_grads = []
    for g in per_sample_gradients:
        summed = g.sum(dim=0)
        noise = torch.normal(0, noise_multiplier * max_grad_norm, size=summed.shape, device=device)
        aggregated_grads.append((summed + noise) / B)

    return aggregated_grads


def project_to_W(params, M):
    total_norm = torch.norm(torch.cat([p.view(-1) for p in params]))
    if total_norm <= M:
        return
    scale = M / total_norm
    for p in params:
        p.data.mul_(scale)

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

def calibrate4(model, X, X_pub, Y,
               loss_fn,
               max_grad_norm,
               noise_multiplier,
               C_h,
               beta,
               alpha=1.0,
               batch_priv_size=128,
               batch_pub_size=4096,
               device='cuda'):
    
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
    logits_pub = model(B_pub_pub)  # (B_pub, L)
    loss_pub = loss_fn(logits_pub, B_pub_labels).mean()  # scalar
    loss_pub.backward()

    grad_pub = [
        param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
        for param in model.parameters()
    ]
    model.zero_grad()

    # --- Private gradient (microbatch style) ---
    B = B_priv.size(0)
    if B == 0:
        return grad_pub

    param_list = list(model.parameters())
    per_sample_gradients = [torch.zeros((B, *p.shape), device=device) for p in param_list]

    for i in range(B):
        model.zero_grad()

        x = B_priv[i].unsqueeze(0)         # (1, T)
        x_pub = B_priv_pub[i].unsqueeze(0) # (1, T)
        y = B_priv_labels[i].unsqueeze(0)  # (1, L)

        logits_total = model(x)            # (1, L)
        logits_pub_i = model(x_pub)        # (1, L)

        _, h_r = model(x, return_hidden=True)
        _, h_i = model(x_pub, return_hidden=True)

        h_r = h_r.mean(dim=1)  # (1, d)
        h_i = h_i.mean(dim=1)

        corr_i = rep_consistency_loss(h_r, h_i, C_h, beta).squeeze()

        loss_total = loss_fn(logits_total, y).mean()      # scalar
        loss_pub_sample = loss_fn(logits_pub_i, y).mean() # scalar

        loss = loss_total - loss_pub_sample + corr_i
        loss.backward()

        for j, p in enumerate(model.parameters()):
            if p.grad is not None:
                per_sample_gradients[j][i] = p.grad.detach().clone()

    # --- Clip per-sample gradients ---
    for i in range(B):
        total_norm = torch.sqrt(sum(torch.sum(g[i] ** 2) for g in per_sample_gradients))
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for g in per_sample_gradients:
                g[i] *= clip_coef

    # --- Aggregate and add noise ---
    aggregated_grads = []
    for g in per_sample_gradients:
        summed = g.sum(dim=0)
        noise = torch.normal(0, noise_multiplier * max_grad_norm, size=summed.shape, device=device)
        aggregated_grads.append((summed + noise) / B)

    # --- Combine public + private gradients ---
    total_grads = [
        grad_pub[i] + alpha * aggregated_grads[i]
        for i in range(len(aggregated_grads))
    ]

    return total_grads
  


@torch.no_grad()
def evaluate(model, dataloader, device="cpu", threshold=0.5):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for ids, labels in dataloader:
        ids = ids.to(device)
        labels = labels.to(device)

        logits = model(ids)  # shape: (B, L)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = {
        "micro/f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "macro/f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "micro/precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "macro/precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "micro/recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "macro/recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "micro/auc": roc_auc_score(y_true, y_prob, average='micro'),
        "macro/auc": roc_auc_score(y_true, y_prob, average='macro'),
    }

    return metrics



def pad_collate(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.stack(ys)

def flatten_batch(dataset):
    all_ids = []
    all_labels = []
    for x, y in dataset:
        all_ids.append(x)
        all_labels.append(y)
    return all_ids, all_labels


def main(epochs, batch_size, lr, epsilon, max_grad_norm, mode, alpha, beta):

    # mode = "calibrate4"
    # alpha = 1
    # beta = 0.15


    print("📦 Loading vocab and top-50 code label map...")
    label_map = json.load(open("lookup.json"))

    print("📄 Loading dataset...")
    # dataset = MimicNoteDataset("../mimicdata/mimic3/train_50.csv", vocab, label_map)
    dataset = MimicNoteDataset("../mimicdata/mimic3/hybrid/Data/1Real/train1.csv", vocab, label_map)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    hybrid_dataset = MimicNoteDataset("../mimicdata/mimic3/hybrid/Data/2Redacted/train1.csv", vocab, label_map)
    hybrid_loader = DataLoader(hybrid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_dataset = MimicNoteDataset("../mimicdata/mimic3/dev_50.csv", vocab, label_map)
    # val_dataset = MimicNoteDataset("../mimicdata/mimic3/hybrid/Data/1Real/valid1.csv", vocab, label_map)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, collate_fn=pad_collate)

    if Path("embedding_matrix.npy").exists():
        print("🔗 Loading pretrained embeddings...")
        emb = torch.tensor(np.load("embedding_matrix.npy"), dtype=torch.float)
        model = DP_CAML(vocab_size=len(vocab), num_labels=len(label_map), d_model=emb.shape[1], pretrained_embed=emb).to(device)
        model.embed.weight.data.copy_(emb)
        model.embed.weight.requires_grad = False
        print("using pretrained embeddings...")
    else:
        model = DP_CAML(vocab_size=len(vocab), num_labels=len(label_map)).to(device)

    # train_with_opacus(model, train_loader, val_loader, epochs, lr, epsilon, max_grad_norm)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    N = len(dataset)
    delta = 1 / N
    sample_rate = batch_size / N
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        accountant="rdp",
        epsilon_tolerance=0.01,
        epochs=epochs
    )
    print("noise_multiplier:", noise_multiplier)

    X_list, Y_list = flatten_batch(dataset)
    X_tensor = nn.utils.rnn.pad_sequence(X_list, batch_first=True).to(device)
    Y_tensor = torch.stack(Y_list).to(device)

    # Use same for X_pub_tensor, or load from hybrid_dataset
    hybrid_X_list, _ = flatten_batch(hybrid_dataset)
    X_pub_tensor = nn.utils.rnn.pad_sequence(hybrid_X_list, batch_first=True).to(device)


    C_h = None
    best_f1 = 0.0

    print("🚀 Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, filled_batch in zip(train_loader, hybrid_loader):
            ids, targets = batch
            ids, targets = ids.to(device), targets.to(device)

            ids_imp, targets_imp = filled_batch
            ids_imp, targets_imp = ids_imp.to(device), targets_imp.to(device)

            if mode == "sgd_ori":
                logits = model(ids)
                loss = loss_fn(logits, targets).mean()

                optimizer.zero_grad()
                loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
            elif mode == "sgd_hybrid":
                logits = model(ids_imp)
                loss = loss_fn(logits, targets_imp).mean()

                optimizer.zero_grad()
                loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
            elif mode == "dpsgd":

                grads = dp_sgd_update_microbatch(model, optimizer,
                                    ids, targets,
                                    loss_fn,
                                    max_grad_norm=max_grad_norm,
                                    noise_multiplier=noise_multiplier)

                for p, g in zip(model.parameters(), grads):
                    p.grad = g
                # optimizer.step()
                # optimizer.zero_grad()
            elif mode == "naive_fusion":
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)


                N = X_tensor.shape[0]
                p_poisson = batch_size / N
                mask = (torch.rand(N, device=device) < p_poisson)

                B_priv = X_tensor[mask]
                # B_priv_pub = X_pub_tensor[mask]
                B_priv_labels = Y_tensor[mask]

                perm = torch.randperm(N, device=device)[:batch_size * 2]

                # B_pub = X[perm]
                B_pub_pub = X_pub_tensor[perm]
                B_pub_labels = Y_tensor[perm]

                dp_gradients = dp_sgd_update_microbatch(model, optimizer,
                                    B_priv, B_priv_labels,
                                    loss_fn,
                                    max_grad_norm=max_grad_norm,
                                    noise_multiplier=noise_multiplier)
                optimizer.zero_grad()
                logits = model(B_pub_pub)
                filled_loss = loss_fn(logits, B_pub_labels).mean()
                filled_loss.backward()
                sgd_gradients = [
                    param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                    for param in model.parameters()
                ]
                for i, param in enumerate(model.parameters()):
                    param.grad = beta * dp_gradients[i] + (1-beta) * sgd_gradients[i]
            elif mode == "fusiondp":
                if noise_multiplier == 0:
                    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=sample_rate, accountant = "rdp", epsilon_tolerance = 0.01, epochs = epochs)
                    print("noise_multiplier: ", noise_multiplier)

                if C_h is None:
                    with torch.no_grad():
                        norms = []
                        h_ori  = model.get_rep(ids)
                        h_imp  = model.get_rep(ids_imp)
                        norms.append((h_ori - h_imp).norm(dim=1).cpu())
                        norms = torch.cat(norms)
                        C_h   = norms.quantile(0.90).item()        # 90‑th percentile
                    print("C_h is ", C_h)


                alpha0 = alpha
                dp_gradients = calibrate4(model=model, X=X_tensor, X_pub=X_pub_tensor, Y=Y_tensor,
                                        loss_fn=loss_fn,
                                        max_grad_norm=max_grad_norm,
                                        noise_multiplier=noise_multiplier,
                                        C_h=C_h,
                                        beta=beta,
                                        alpha=alpha0,
                                        batch_priv_size=batch_size,
                                        batch_pub_size=batch_size*2)
                for i, param in enumerate(model.parameters()):
                    param.grad = dp_gradients[i]
            

            optimizer.step()
            optimizer.zero_grad()

            # Optional: compute loss for logging
            with torch.no_grad():
                logits = model(ids)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                total_loss += loss.item()

        print(f"✅ Epoch {epoch+1}: Avg loss = {total_loss / len(train_loader):.4f}")
        metrics = evaluate(model, val_loader, device)
        print("Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        if metrics.get("micro/f1", 0.0) > best_f1:
            best_f1 = metrics["micro/f1"]
            best_results = {
                "mode": mode,
                "epoch": epoch + 1,  # make sure to use 1-based epoch index
                "learning_rate": lr,
                "epsilon": epsilon,
                "max_grad_norm": max_grad_norm,
                "batch_size": batch_size,
                "alpha": alpha,
                "beta": beta,
                "metrics": metrics
            }


    results_file = "test2.json"
    try:
        # Load existing results if file exists
        with open(results_file, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        # Initialize an empty list if file doesn't exist
        all_results = []
    all_results.append(best_results)

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"final result for this combination saved: {best_results}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=90)
    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)
    print("seed set ", seed)

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        mode=args.mode,
        alpha=args.alpha,
        beta=args.beta
    )



