"""Ablation: CE-only / CE+severity / CE+margin / CE+severity+margin.

Tests the project's main contribution: margin loss (penalizing specific
confusable pairs, based on a neighbor list of easily-confused pairs inferred
via the BI-SIM algorithm) + severity-weighted CE (adapted from
Risk-Calibrated Learning, arXiv: 2604.12693) - whether it reduces the
confusable_wrong_drug rate without reducing correct / increasing
hallucination, compared to CE-only (the baseline phenomenon already
confirmed in kernel 02_lora_ceonly).

Runs on a Kaggle kernel (T4 GPU). Fine-tunes SEPARATELY for each (dataset,
loss config) pair - 4 configs x 2 datasets = 8 fine-tuning runs, each one
restarting from the original pretrained checkpoint (no continuation between
configs).

Output (/kaggle/working/):
  - predictions_<dataset>_<config>.csv
  - summary_ablation.csv (updated incrementally after each (dataset,config) -
    so that if the kernel errors out partway through, results already
    computed are preserved)
"""

import glob
import os
import random
import re
import subprocess
import sys
import time
import traceback

print("Installing dependencies (peft pinned == 0.13.2, see notes in kernel 02)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft==0.13.2", "jellyfish", "rapidfuzz"],
    check=False,
)

import jellyfish  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working"

EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-4
MAX_LABEL_LEN = 32
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
CONFUSABLE_THRESHOLD = 0.65  # taxonomy classification threshold (same as kernel 01/02)
NEIGHBOR_MIN_SCORE = 0.5  # minimum score to count as a "confusable neighbor" when building N(w)
NEIGHBOR_TOP_K = 5
BETA = 1.0           # severity-weighted CE coefficient: weight = 1 + BETA * max_similarity
MARGIN = 1.0          # margin (in NLL/nat units) for the margin-ranking loss
LAMBDA_MARGIN = 1.0   # coefficient combining margin_loss into total_loss

random.seed(42)

# ---------------------------------------------------------------------------
# Vocabulary + taxonomy (same as kernel 01/02)
# ---------------------------------------------------------------------------
_TRAILING_DOSAGE_RE = re.compile(r"\s*\d+(\.\d+)?\s*(mg|ml|gm|g|mcg|iu)?\s*$", re.IGNORECASE)
_DOSAGE_FORM_PREFIX_RE = re.compile(r"^(tab\.?|cap\.?|inj\.?|syp\.?|susp\.?)\s*", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.\-]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_label(raw: str) -> str:
    s = raw.strip()
    return _TRAILING_DOSAGE_RE.sub("", s).strip()


def canonicalize_for_dedup(norm_label: str) -> str:
    s = norm_label.lower()
    s = _DOSAGE_FORM_PREFIX_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    return _MULTI_SPACE_RE.sub(" ", s).strip()


def normalized_edit_similarity(a: str, b: str) -> float:
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def phonetic_similarity(a: str, b: str) -> float:
    ma, mb = jellyfish.metaphone(a), jellyfish.metaphone(b)
    if not ma or not mb:
        return 0.0
    return jellyfish.jaro_winkler_similarity(ma, mb)


def bi_sim_score(a: str, b: str) -> float:
    a_l, b_l = a.lower(), b.lower()
    return 0.5 * normalized_edit_similarity(a_l, b_l) + 0.5 * phonetic_similarity(a_l, b_l)


def classify_error(true_label: str, pred_label: str, vocab_canon_set: set) -> str:
    true_c = canonicalize_for_dedup(true_label)
    pred_c = canonicalize_for_dedup(pred_label)
    if not pred_c:
        return "hallucination_far_off"
    if pred_c == true_c:
        return "correct"
    if pred_c in vocab_canon_set:
        score = bi_sim_score(true_label, pred_label)
        return "confusable_wrong_drug" if score >= CONFUSABLE_THRESHOLD else "other_wrong_real_word"
    edit_sim = normalized_edit_similarity(true_label.lower(), pred_label.lower())
    return "minor_ocr_noise" if edit_sim >= 0.7 else "hallucination_far_off"


def build_neighbor_map(vocab_canon_list, top_k=NEIGHBOR_TOP_K, min_score=NEIGHBOR_MIN_SCORE):
    """Returns a dict: canon_word -> [(canon_neighbor, score), ...], sorted."""
    vocab = sorted(set(vocab_canon_list))
    neighbor_map = {}
    t0 = time.time()
    for i, w in enumerate(vocab):
        scored = [(o, bi_sim_score(w, o)) for o in vocab if o != w]
        scored = [p for p in scored if p[1] >= min_score]
        scored.sort(key=lambda p: -p[1])
        neighbor_map[w] = scored[:top_k]
    print(f"  build_neighbor_map: {len(vocab)} words, {time.time()-t0:.1f}s")
    return neighbor_map


# ---------------------------------------------------------------------------
# Locate dataset directory
# ---------------------------------------------------------------------------
def find_dir_containing(root: str, filename_pattern: str) -> str:
    matches = glob.glob(os.path.join(root, "**", filename_pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find '{filename_pattern}' under {root}")
    return os.path.dirname(matches[0])


KAGGLE_BD_TEST_DIR = find_dir_containing(INPUT_ROOT, "testing_labels.csv")
KAGGLE_BD_ROOT = os.path.dirname(KAGGLE_BD_TEST_DIR)
_rxhandbd_matches = glob.glob(os.path.join(INPUT_ROOT, "**", "RxHandBDMain", "Test.csv"), recursive=True)
RXHANDBD_MAIN_DIR = os.path.dirname(_rxhandbd_matches[0])


def load_kaggle_bd():
    train_df = pd.read_csv(os.path.join(KAGGLE_BD_ROOT, "Training", "training_labels.csv"))
    val_df = pd.read_csv(os.path.join(KAGGLE_BD_ROOT, "Validation", "validation_labels.csv"))
    test_df = pd.read_csv(os.path.join(KAGGLE_BD_ROOT, "Testing", "testing_labels.csv"))

    train_pairs = list(zip(train_df["IMAGE"], train_df["MEDICINE_NAME"].astype(str).str.strip()))
    test_pairs = list(zip(test_df["IMAGE"], test_df["MEDICINE_NAME"].astype(str).str.strip()))
    full_vocab = set(
        pd.concat([train_df["MEDICINE_NAME"], val_df["MEDICINE_NAME"], test_df["MEDICINE_NAME"]]).astype(str).str.strip()
    )
    return {
        "name": "kaggle_bd",
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_dir": os.path.join(KAGGLE_BD_ROOT, "Training", "training_words"),
        "test_dir": os.path.join(KAGGLE_BD_ROOT, "Testing", "testing_words"),
        "vocab_canon_set": {canonicalize_for_dedup(v) for v in full_vocab},
    }


def load_rxhandbd():
    train_df = pd.read_csv(os.path.join(RXHANDBD_MAIN_DIR, "Train.csv"))
    test_df = pd.read_csv(os.path.join(RXHANDBD_MAIN_DIR, "Test.csv"))
    train_df["norm_label"] = train_df["Labels"].astype(str).apply(normalize_label)
    test_df["norm_label"] = test_df["Labels"].astype(str).apply(normalize_label)

    train_pairs = list(zip(train_df["Images"], train_df["norm_label"]))
    test_pairs = list(zip(test_df["Images"], test_df["norm_label"]))
    full_vocab = set(train_df["norm_label"]) | set(test_df["norm_label"])
    return {
        "name": "rxhandbd",
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_dir": os.path.join(RXHANDBD_MAIN_DIR, "Train"),
        "test_dir": os.path.join(RXHANDBD_MAIN_DIR, "Test"),
        "vocab_canon_set": {canonicalize_for_dedup(v) for v in full_vocab},
    }


# ---------------------------------------------------------------------------
# Dataset + loss + train + eval
# ---------------------------------------------------------------------------
class OCRDataset(Dataset):
    def __init__(self, pairs, image_dir, processor, max_length=MAX_LABEL_LEN):
        self.pairs = [(fn, lbl) for fn, lbl in pairs if os.path.exists(os.path.join(image_dir, fn))]
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fn, label = self.pairs[idx]
        image = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        ids = self.processor.tokenizer(
            label, padding="max_length", max_length=self.max_length, truncation=True
        ).input_ids
        ids = [i if i != self.processor.tokenizer.pad_token_id else -100 for i in ids]
        return {"pixel_values": pixel_values, "labels": torch.tensor(ids), "label_text": label}


def tokenize_batch(tokenizer, texts, max_length=MAX_LABEL_LEN):
    enc = tokenizer(texts, padding="max_length", max_length=max_length, truncation=True).input_ids
    enc = [[t if t != tokenizer.pad_token_id else -100 for t in seq] for seq in enc]
    return torch.tensor(enc)


def per_example_nll(model, pixel_values, label_ids):
    outputs = model(pixel_values=pixel_values, labels=label_ids)
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = label_ids.view(-1)
    per_token = loss_fct(flat_logits, flat_labels).view(label_ids.size(0), -1)
    mask = (label_ids != -100).float()
    return (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def train_lora(model, processor, train_pairs, image_dir, device, use_severity, use_margin, neighbor_map, tag):
    ds = OCRDataset(train_pairs, image_dir, processor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_ce, total_margin, n_batches = 0.0, 0.0, 0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            true_ids = batch["labels"].to(device)
            label_texts = batch["label_text"]

            true_nll = per_example_nll(model, pixel_values, true_ids)

            if use_severity:
                weights = torch.tensor(
                    [
                        1.0 + BETA * (neighbor_map.get(canonicalize_for_dedup(t), [(None, 0.0)])[0][1]
                                      if neighbor_map.get(canonicalize_for_dedup(t)) else 0.0)
                        for t in label_texts
                    ],
                    device=device,
                )
            else:
                weights = torch.ones(len(label_texts), device=device)
            ce_loss = (weights * true_nll).mean()

            margin_loss = torch.tensor(0.0, device=device)
            if use_margin:
                neg_texts, has_neighbor = [], []
                for t in label_texts:
                    neighbors = neighbor_map.get(canonicalize_for_dedup(t), [])
                    if neighbors:
                        neg_texts.append(random.choice(neighbors)[0])
                        has_neighbor.append(1.0)
                    else:
                        neg_texts.append(t)
                        has_neighbor.append(0.0)
                has_neighbor_t = torch.tensor(has_neighbor, device=device)
                if has_neighbor_t.sum() > 0:
                    neg_ids = tokenize_batch(processor.tokenizer, neg_texts).to(device)
                    neg_nll = per_example_nll(model, pixel_values, neg_ids)
                    margin_term = torch.relu(MARGIN - (neg_nll - true_nll)) * has_neighbor_t
                    margin_loss = margin_term.sum() / has_neighbor_t.sum()

            total_loss = ce_loss + (LAMBDA_MARGIN * margin_loss if use_margin else 0.0)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_ce += ce_loss.item()
            total_margin += float(margin_loss)
            n_batches += 1
            if step % 50 == 0:
                print(
                    f"  [{tag}] epoch {epoch} step {step}/{len(loader)} "
                    f"ce={ce_loss.item():.4f} margin={float(margin_loss):.4f} elapsed={time.time()-t0:.0f}s"
                )
        print(f"  [{tag}] epoch {epoch} done avg_ce={total_ce/max(n_batches,1):.4f} avg_margin={total_margin/max(n_batches,1):.4f}")
    model.eval()
    return model


def generate_text(model, pixel_values, max_length=MAX_LABEL_LEN):
    try:
        return model.generate(pixel_values=pixel_values, max_length=max_length)
    except AttributeError:
        return model.get_base_model().generate(pixel_values=pixel_values, max_length=max_length)


def evaluate(model, processor, test_pairs, image_dir, vocab_canon_set, dataset_name, config_name, device):
    rows = []
    batch_size = 16
    valid_pairs = [(fn, lbl) for fn, lbl in test_pairs if os.path.exists(os.path.join(image_dir, fn))]
    for start in range(0, len(valid_pairs), batch_size):
        chunk = valid_pairs[start : start + batch_size]
        images = [Image.open(os.path.join(image_dir, fn)).convert("RGB") for fn, _ in chunk]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen_ids = generate_text(model, pixel_values)
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
        for (fn, true_label), pred in zip(chunk, preds):
            rows.append(
                {
                    "dataset": dataset_name,
                    "config": config_name,
                    "image": fn,
                    "true_label": true_label,
                    "pred": pred,
                    "error_category": classify_error(true_label, pred, vocab_canon_set),
                }
            )
    return pd.DataFrame(rows)


def load_fresh_model_and_lora():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(DEVICE)
    base_model.config.decoder_start_token_id = base_model.decoder.config.decoder_start_token_id
    base_model.config.pad_token_id = base_model.decoder.config.pad_token_id
    base_model.config.eos_token_id = base_model.decoder.config.eos_token_id
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    return processor, lora_model


CONFIGS = [
    ("ce_only", False, False),
    ("severity", True, False),
    ("margin", False, True),
    ("severity_margin", True, True),
]

# ---------------------------------------------------------------------------
# Main: run the ablation grid, saving results incrementally after each
# (dataset, config)
# ---------------------------------------------------------------------------
summary_rows = []


def save_summary():
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_ROOT, "summary_ablation.csv"), index=False)


for loader_fn in [load_rxhandbd, load_kaggle_bd]:
    data = loader_fn()
    name = data["name"]
    print(f"\n=== Dataset: {name} | train={len(data['train_pairs'])} test={len(data['test_pairs'])} ===")

    print("Building confusable-neighbor map from train vocab...")
    train_vocab_canon = {canonicalize_for_dedup(lbl) for _, lbl in data["train_pairs"]}
    neighbor_map = build_neighbor_map(train_vocab_canon)
    n_with_neighbor = sum(1 for v in neighbor_map.values() if v)
    print(f"  {n_with_neighbor}/{len(neighbor_map)} words have at least 1 neighbor meeting the {NEIGHBOR_MIN_SCORE} threshold")

    for config_name, use_severity, use_margin in CONFIGS:
        tag = f"{name}/{config_name}"
        print(f"\n--- Running {tag} ---")
        try:
            processor, lora_model = load_fresh_model_and_lora()
            lora_model.print_trainable_parameters()
            lora_model = train_lora(
                lora_model, processor, data["train_pairs"], data["train_dir"], DEVICE,
                use_severity, use_margin, neighbor_map, tag,
            )
            ft_df = evaluate(lora_model, processor, data["test_pairs"], data["test_dir"], data["vocab_canon_set"], name, config_name, DEVICE)
            ft_df.to_csv(os.path.join(OUTPUT_ROOT, f"predictions_{name}_{config_name}.csv"), index=False)

            n = len(ft_df)
            counts = ft_df["error_category"].value_counts()
            for cat, cnt in counts.items():
                summary_rows.append(
                    {"dataset": name, "config": config_name, "error_category": cat, "count": cnt, "rate": cnt / n if n else 0.0, "n_total": n}
                )
            save_summary()
            print(f"--- Done {tag} ---")

            del processor, lora_model
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            print(f"!!! ERROR in {tag}, skipping and continuing to the next config !!!")
            traceback.print_exc()
            summary_rows.append({"dataset": name, "config": config_name, "error_category": "KERNEL_ERROR", "count": -1, "rate": -1, "n_total": -1})
            save_summary()
            torch.cuda.empty_cache()
            continue

print("\n=== SUMMARY ABLATION (all dataset x config combinations) ===")
print(pd.DataFrame(summary_rows).to_string(index=False))
print("DONE.")
