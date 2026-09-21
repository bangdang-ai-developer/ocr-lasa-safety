"""CONFIRMATORY RERUNS of the "winning" config with full seeding.

The adversarial review round (4 independent agents) found that the winning
config (margin lambda=2.0 + severity beta=0.3) was picked AFTER looking at
the results of 7 other configs on the SAME RxHandBD test set, with no
separate validation set - after Holm-Bonferroni correction
(scripts/multiple_comparisons.py), the corrected p is 0.057, which is NO
LONGER significant. On top of that, every earlier kernel only seeded
Python's random (which only affects negative sampling for the margin loss)
- it did NOT seed torch/numpy, so LoRA init and the DataLoader shuffle
order were uncontrolled, and each config was only run once.

This kernel runs 5 independent seeds (1-5). FOR EACH SEED: set_all_seeds(seed)
is called RIGHT BEFORE building each model, so that LoRA init + DataLoader
order are IDENTICAL between the (ce_only, winning) pair within the same
seed - isolating the init/shuffle noise variable so the only remaining
difference is the loss function - then a paired comparison (McNemar) is run
SEPARATELY FOR EACH SEED. If the majority of independent seeds agree on
p<0.05 one-sided (confusable_wrong_drug decreases, correct doesn't
decrease), that's solid evidence of a real effect from the loss function,
not a lucky single run.

Output (/kaggle/working/):
  - predictions_rxhandbd_seed<k>_<config>.csv (config = ce_only|winning)
  - summary_confirmatory_per_seed.csv (McNemar per seed, updated incrementally)
"""

import glob
import os
import random
import re
import subprocess
import sys
import time
import traceback

print("Installing dependencies (peft pinned == 0.13.2, see kernel 02 notes)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft==0.13.2", "jellyfish", "rapidfuzz"],
    check=False,
)

import jellyfish  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working"

BATCH_SIZE = 8
LR = 1e-4
MAX_LABEL_LEN = 32
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
CONFUSABLE_THRESHOLD = 0.65  # taxonomy classification threshold (same as kernel 01/02/03)
NEIGHBOR_MIN_SCORE = 0.5  # minimum score to count as a "confusable neighbor" when building N(w)
NEIGHBOR_TOP_K = 5
MARGIN = 1.0  # margin (in NLL/nat units) for the margin-ranking loss
SEEDS = [1, 2, 3, 4, 5]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    """Returns a dict: canon_word -> [(canon_neighbor, score), ...], ranked."""
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
# Find dataset directory
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


def train_lora(model, processor, train_pairs, image_dir, device, neighbor_map, tag,
                use_severity=False, use_margin=False, beta=1.0, margin=MARGIN,
                lambda_margin=1.0, epochs=3):
    ds = OCRDataset(train_pairs, image_dir, processor)
    # num_workers=0: avoids extra randomness from worker processes, keeping
    # the shuffle order deterministic across runs with the same seed.
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        total_ce, total_margin, n_batches = 0.0, 0.0, 0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            true_ids = batch["labels"].to(device)
            label_texts = batch["label_text"]

            true_nll = per_example_nll(model, pixel_values, true_ids)

            if use_severity:
                weights = torch.tensor(
                    [
                        1.0 + beta * (neighbor_map.get(canonicalize_for_dedup(t), [(None, 0.0)])[0][1]
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
                    margin_term = torch.relu(margin - (neg_nll - true_nll)) * has_neighbor_t
                    margin_loss = margin_term.sum() / has_neighbor_t.sum()

            total_loss = ce_loss + (lambda_margin * margin_loss if use_margin else 0.0)
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


# "winning" config to be confirmed (from kernel 04_margin_sweep)
WINNING_CONFIG = dict(use_severity=True, use_margin=True, beta=0.3, margin=MARGIN, lambda_margin=2.0, epochs=3)
CE_ONLY_CONFIG = dict(use_severity=False, use_margin=False, beta=0.0, margin=MARGIN, lambda_margin=0.0, epochs=3)


def mcnemar_for_seed(base_df, other_df, category="confusable_wrong_drug"):
    common = base_df.set_index("image").index.intersection(other_df.set_index("image").index)
    b = base_df.set_index("image").loc[common]
    o = other_df.set_index("image").loc[common]
    b_flag = b["error_category"] == category
    o_flag = o["error_category"] == category
    fixed = int((b_flag & ~o_flag).sum())
    newly = int((~b_flag & o_flag).sum())
    n = fixed + newly
    p = binomtest(fixed, n, 0.5).pvalue if n else float("nan")
    return {"fixed": fixed, "newly": newly, "p": p, "base_count": int(b_flag.sum()), "other_count": int(o_flag.sum())}


# ---------------------------------------------------------------------------
# Main: run 5 independent seeds, each seed trains the (ce_only, winning)
# PAIR with the SAME seed (isolating the LoRA init / shuffle order noise),
# and runs a paired comparison separately for each seed. Results are saved
# incrementally after every seed.
# ---------------------------------------------------------------------------
per_seed_rows = []


def save_summary():
    pd.DataFrame(per_seed_rows).to_csv(os.path.join(OUTPUT_ROOT, "summary_confirmatory_per_seed.csv"), index=False)


data = load_rxhandbd()
name = data["name"]
print(f"\n=== Dataset: {name} | train={len(data['train_pairs'])} test={len(data['test_pairs'])} ===")

print("Building confusable-neighbor map from train vocab...")
train_vocab_canon = {canonicalize_for_dedup(lbl) for _, lbl in data["train_pairs"]}
neighbor_map = build_neighbor_map(train_vocab_canon)
n_with_neighbor = sum(1 for v in neighbor_map.values() if v)
print(f"  {n_with_neighbor}/{len(neighbor_map)} words have at least 1 neighbor meeting the {NEIGHBOR_MIN_SCORE} threshold")

for seed in SEEDS:
    print(f"\n=========== SEED {seed} ===========")
    try:
        # --- ce_only, seed fixed before creating the model ---
        set_all_seeds(seed)
        processor, model_ce = load_fresh_model_and_lora()
        tag_ce = f"{name}/seed{seed}/ce_only"
        model_ce = train_lora(model_ce, processor, data["train_pairs"], data["train_dir"], DEVICE, neighbor_map, tag_ce, **CE_ONLY_CONFIG)
        df_ce = evaluate(model_ce, processor, data["test_pairs"], data["test_dir"], data["vocab_canon_set"], name, f"seed{seed}_ce_only", DEVICE)
        df_ce.to_csv(os.path.join(OUTPUT_ROOT, f"predictions_rxhandbd_seed{seed}_ce_only.csv"), index=False)
        del model_ce
        torch.cuda.empty_cache()

        # --- winning config, RESET to the SAME seed before creating the model ---
        set_all_seeds(seed)
        processor, model_win = load_fresh_model_and_lora()
        tag_win = f"{name}/seed{seed}/winning"
        model_win = train_lora(model_win, processor, data["train_pairs"], data["train_dir"], DEVICE, neighbor_map, tag_win, **WINNING_CONFIG)
        df_win = evaluate(model_win, processor, data["test_pairs"], data["test_dir"], data["vocab_canon_set"], name, f"seed{seed}_winning", DEVICE)
        df_win.to_csv(os.path.join(OUTPUT_ROOT, f"predictions_rxhandbd_seed{seed}_winning.csv"), index=False)
        del model_win
        torch.cuda.empty_cache()

        # --- Paired McNemar comparison FOR THIS SEED ---
        result = mcnemar_for_seed(df_ce, df_win, "confusable_wrong_drug")
        result_correct = mcnemar_for_seed(df_ce, df_win, "correct")
        result_halluc = mcnemar_for_seed(df_ce, df_win, "hallucination_far_off")
        per_seed_rows.append(
            {
                "seed": seed,
                "ce_only_confusable": result["base_count"], "winning_confusable": result["other_count"],
                "confusable_fixed": result["fixed"], "confusable_newly": result["newly"], "confusable_p": result["p"],
                "ce_only_correct": result_correct["base_count"], "winning_correct": result_correct["other_count"],
                "correct_p": result_correct["p"],
                "ce_only_halluc": result_halluc["base_count"], "winning_halluc": result_halluc["other_count"],
                "halluc_p": result_halluc["p"],
            }
        )
        save_summary()
        print(f"--- Seed {seed}: confusable {result['base_count']}->{result['other_count']} (p={result['p']:.4f}), "
              f"correct p={result_correct['p']:.4f}, halluc p={result_halluc['p']:.4f} ---")
    except Exception:  # noqa: BLE001
        print(f"!!! ERROR at seed {seed}, skipping and continuing to next seed !!!")
        traceback.print_exc()
        per_seed_rows.append({"seed": seed, "confusable_p": None, "error": "KERNEL_ERROR"})
        save_summary()
        torch.cuda.empty_cache()
        continue

print("\n=== SUMMARY: McNemar (ce_only vs winning) PER SEED ===")
print(pd.DataFrame(per_seed_rows).to_string(index=False))
n_sig_seeds = sum(1 for r in per_seed_rows if r.get("confusable_p") is not None and r["confusable_p"] < 0.05)
print(f"\nNumber of seeds (out of {len(SEEDS)}) with uncorrected p<0.05 for confusable_wrong_drug: {n_sig_seeds}")
print("DONE.")
