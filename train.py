"""Train the FallLSTM on synthetic or real fall detection data.

Usage:
    python train.py --synthetic --epochs 50                    # Quick dev test
    python train.py --data data/training_data.npz --epochs 100 # Real training
"""
import argparse, os, numpy as np, math
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from modules.fall_detector import FallLSTM

class FallFeatureDataset(Dataset):
    """Sequences of 5 physics features with binary labels (0=fall, 1=normal)."""
    def __init__(self, seqs, labels, augment=False):
        self.seqs = torch.FloatTensor(seqs); self.labels = torch.LongTensor(labels); self.aug = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.seqs[i].clone()
        if self.aug: x += torch.randn_like(x) * 0.02
        return x, self.labels[i]

def generate_synthetic_fall_data(n_samples=2000, seq_len=36, seed=42):
    """Generate synthetic fall/normal sequences of 5 physics features."""
    np.random.seed(seed); n = n_samples // 2
    seqs, labels = [], []

    for _ in range(n):
        # Normal activity sequence
        seq = np.zeros((seq_len, 5), dtype=np.float32)
        for t in range(seq_len):
            seq[t, 0] = 0.3 + np.random.normal(0, 0.05)        # ratio_bbox ~ 0.3 (taller than wide)
            seq[t, 1] = np.log(1 + abs(np.random.normal(0, 0.15)))  # small body angle
            seq[t, 2] = abs(np.random.normal(0, 0.05))           # low rotational energy
            seq[t, 3] = np.random.normal(0, 0.3)                 # low ratio derivative
            seq[t, 4] = np.random.normal(0, 0.5)                 # low generalised force
        seqs.append(seq); labels.append(1)  # normal

    for _ in range(n):
        # Fall sequence: normal → transition → fallen
        seq = np.zeros((seq_len, 5), dtype=np.float32)
        fall_start = np.random.randint(seq_len // 4, seq_len // 2)
        fall_end = min(fall_start + np.random.randint(4, 10), seq_len - 1)

        for t in range(seq_len):
            if t < fall_start:
                # Normal standing
                seq[t, 0] = 0.3 + np.random.normal(0, 0.05)
                seq[t, 1] = np.log(1 + abs(np.random.normal(0, 0.15)))
                seq[t, 2] = abs(np.random.normal(0, 0.05))
                seq[t, 3] = np.random.normal(0, 0.3)
                seq[t, 4] = np.random.normal(0, 0.5)
            elif t < fall_end:
                # Falling transition
                progress = (t - fall_start) / max(fall_end - fall_start, 1)
                seq[t, 0] = 0.3 + progress * 0.8 + np.random.normal(0, 0.1)  # ratio increases
                seq[t, 1] = np.log(1 + progress * math.pi/2)                  # angle increases
                seq[t, 2] = 0.5 + progress * 2.0 + abs(np.random.normal(0, 0.3))  # high RE
                seq[t, 3] = 2.0 + progress * 3.0 + np.random.normal(0, 0.5)  # ratio spike
                seq[t, 4] = 5.0 + progress * 10.0 + np.random.normal(0, 1.0)  # high GF
            else:
                # Lying on ground
                seq[t, 0] = 1.2 + np.random.normal(0, 0.15)     # wider than tall
                seq[t, 1] = np.log(1 + math.pi/2 + abs(np.random.normal(0, 0.1)))  # large angle
                seq[t, 2] = abs(np.random.normal(0, 0.1))        # low RE (still)
                seq[t, 3] = np.random.normal(0, 0.3)             # stable ratio
                seq[t, 4] = np.random.normal(0, 0.5)             # low GF
        seqs.append(seq); labels.append(0)  # fall

    seqs, labels = np.array(seqs), np.array(labels)
    perm = np.random.permutation(len(labels))
    return seqs[perm], labels[perm]

def train(model, train_dl, val_dl, epochs=50, lr=0.001, patience=10, device=None, save_path=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, 15, 0.5)
    best_f1 = 0; wait = 0
    for ep in range(epochs):
        model.train(); tl=tc=tt=0
        for x,y in train_dl:
            x,y = x.to(device),y.to(device); opt.zero_grad()
            o,_ = model(x); loss = crit(o,y); loss.backward(); opt.step()
            tl+=loss.item()*x.size(0); tc+=(o.argmax(1)==y).sum().item(); tt+=y.size(0)
        model.eval(); ap,al=[],[]
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device); o,_=model(x)
                ap.extend(o.argmax(1).cpu().numpy()); al.extend(y.cpu().numpy())
        sched.step(); vf1=f1_score(al,ap,average="weighted")
        print(f"Epoch {ep+1}/{epochs} | Loss:{tl/tt:.4f} Acc:{tc/tt:.4f} | F1:{vf1:.4f}")
        if vf1>best_f1:
            best_f1=vf1; wait=0
            if save_path: os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True); torch.save(model.state_dict(),save_path); print(f"  -> Saved (F1:{vf1:.4f})")
        else:
            wait+=1
            if wait>=patience: print(f"  Early stop"); break

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--batch_size",type=int,default=32)
    pa.add_argument("--output",default="models/fall_lstm.pth")
    pa.add_argument("--synthetic",action="store_true",help="Use synthetic data for quick dev testing")
    pa.add_argument("--data",default=None,help="Path to .npz from prepare_data.py")
    pa.add_argument("--lr",type=float,default=0.001)
    pa.add_argument("--eval_dir",default="outputs/training/",help="Directory for evaluation plots")
    a=pa.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    if a.data is not None:
        print(f"Loading real data from {a.data}...")
        d = np.load(a.data)
        seqs, labels = d["seqs"], d["labels"]
        print(f"  Loaded: {len(labels)} sequences ({sum(labels==0)} fall, {sum(labels==1)} normal)")
    elif a.synthetic:
        print("Generating synthetic fall data...")
        seqs, labels = generate_synthetic_fall_data(4000)
    else:
        print("ERROR: Provide --data path/to/training_data.npz or use --synthetic")
        print("  To create training data: python prepare_data.py --data_dir data/ --output data/training_data.npz")
        return

    # --- Class weighting for imbalanced data ---
    n_fall = sum(labels == 0)
    n_normal = sum(labels == 1)
    total = len(labels)
    weight_fall = total / (2 * max(n_fall, 1))
    weight_normal = total / (2 * max(n_normal, 1))
    class_weights = torch.FloatTensor([weight_fall, weight_normal]).to(device)
    print(f"  Class weights: fall={weight_fall:.2f}, normal={weight_normal:.2f}")

    # --- Split dataset ---
    ds = FallFeatureDataset(seqs, labels, augment=True)
    n = len(ds); tn = int(.7*n); vn = int(.15*n); ten = n-tn-vn
    tr, va, te = random_split(ds, [tn, vn, ten])
    print(f"  Split: {tn} train / {vn} val / {ten} test")

    # --- Train ---
    model = FallLSTM(input_dim=5, hidden_size=48, num_layers=2, dropout=0.1, num_classes=2)
    train(model, DataLoader(tr,a.batch_size,True), DataLoader(va,a.batch_size),
          a.epochs, a.lr, device=device, save_path=a.output)

    # --- Evaluate on test set ---
    print("\n" + "="*50 + "\nEvaluating on test set\n" + "="*50)
    model.load_state_dict(torch.load(a.output, map_location=device, weights_only=True))
    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(te, a.batch_size):
            x = x.to(device)
            o, _ = model(x)
            all_preds.extend(o.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    # Classification report
    print("\n" + classification_report(all_labels, all_preds, target_names=["fall","normal"]))

    # Save evaluation plots
    os.makedirs(a.eval_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["fall","normal"], yticklabels=["fall","normal"])
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(a.eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Summary
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision_fall = cm[0,0] / max(cm[0,0]+cm[1,0], 1)
    recall_fall = cm[0,0] / max(cm[0,0]+cm[0,1], 1)
    print(f"\n  Test F1 (weighted): {f1:.4f}")
    print(f"  Fall precision:     {precision_fall:.4f}  (low = too many false alarms)")
    print(f"  Fall recall:        {recall_fall:.4f}  (low = missing real falls)")
    print(f"\n  Model saved to: {a.output}")
    print(f"\n  Next step: python main.py --input video.mp4 --model_weights {a.output} --show")

if __name__=="__main__": main()
