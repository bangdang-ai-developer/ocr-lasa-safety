"""Independent verification of the original phenomenon: does LoRA fine-tuning of
TrOCR-base-handwritten (CE-only, without margin loss / severity-weighted loss)
INCREASE the confusable_wrong_drug error rate while DECREASING
hallucination_far_off?

Runs on a Kaggle kernel (T4 GPU). Fine-tunes SEPARATELY for each dataset
(different vocabulary), evaluating both zero-shot and post-fine-tune on the
Test set so they can be compared directly within the same results file.

Output (/kaggle/working/):
  - predictions_<dataset>_stage.csv (per dataset, both stages: zeroshot / lora_ceonly)
  - summary_before_after.csv
  - <dataset>_lora_adapter/ (fine-tuned LoRA weights, for reuse later)
"""

import glob
import os
import subprocess
import sys
import time

print("Installing peft...")
# Pin peft==0.13.2: newer versions on Kaggle fail at import time with the
# preinstalled torchao (0.10.0 < the 0.16.0 required) in LoRA's
# dispatch_torchao - confirmed by one failed run (KernelWorkerStatus.ERROR)
# plus a local smoke test showing 0.13.2 works correctly with this script's logic.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft==0.13.2", "jellyfish", "rapidfuzz"],
    check=False,
)

import jellyfish  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
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
CONFUSABLE_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Vocabulary + taxonomy (same as ocr_benchmark_kernel.py, rewritten independently in this kernel)
# ---------------------------------------------------------------------------
import re  # noqa: E402

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


# ---------------------------------------------------------------------------
# Dataset abstraction: (train_pairs, test_pairs, full_vocab, train_dir, test_dir)
# ---------------------------------------------------------------------------
def load_kaggle_bd():
    train_df = pd.read_csv(os.path.join(KAGGLE_BD_ROOT, "Training", "training_labels.csv"))
    val_df = pd.read_csv(os.path.join(KAGGLE_BD_ROOT, "Validation", "validation_labels.csv"))
    test_df = pd.read_csv(os.path.join(KAGGLE_BD_ROOT, "Testing", "testing_labels.csv"))

    train_pairs = list(zip(train_df["IMAGE"], train_df["MEDICINE_NAME"].astype(str).str.strip()))
    test_pairs = list(zip(test_df["IMAGE"], test_df["MEDICINE_NAME"].astype(str).str.strip()))
    full_vocab = set(
        pd.concat([train_df["MEDICINE_NAME"], val_df["MEDICINE_NAME"], test_df["MEDICINE_NAME"]])
        .astype(str)
        .str.strip()
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
# Torch Dataset + training + eval
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
        return {"pixel_values": pixel_values, "labels": torch.tensor(ids)}


def train_lora_ceonly(model, processor, train_pairs, image_dir, device):
    ds = OCRDataset(train_pairs, image_dir, processor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s")
        print(f"  epoch {epoch} done, avg_loss={total_loss/max(len(loader),1):.4f}")
    model.eval()
    return model


def generate_text(model, pixel_values, max_length=MAX_LABEL_LEN):
    try:
        return model.generate(pixel_values=pixel_values, max_length=max_length)
    except AttributeError:
        return model.get_base_model().generate(pixel_values=pixel_values, max_length=max_length)


def evaluate(model, processor, test_pairs, image_dir, vocab_canon_set, dataset_name, stage, device):
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
                    "stage": stage,
                    "image": fn,
                    "true_label": true_label,
                    "pred": pred,
                    "error_category": classify_error(true_label, pred, vocab_canon_set),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main: run for each dataset
# ---------------------------------------------------------------------------
all_predictions = []
all_summary = []

for loader_fn in [load_kaggle_bd, load_rxhandbd]:
    data = loader_fn()
    name = data["name"]
    print(f"\n=== Dataset: {name} | train={len(data['train_pairs'])} test={len(data['test_pairs'])} ===")

    print("Loading fresh TrOCR-base-handwritten...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(DEVICE)
    # REQUIRED: without setting this manually, computing CE loss (labels=...) raises
    # "Make sure to set the decoder_start_token_id attribute" on some
    # transformers versions - confirmed via a local smoke test before
    # running on Kaggle.
    base_model.config.decoder_start_token_id = base_model.decoder.config.decoder_start_token_id
    base_model.config.pad_token_id = base_model.decoder.config.pad_token_id
    base_model.config.eos_token_id = base_model.decoder.config.eos_token_id

    print("Evaluating ZERO-SHOT (before fine-tune)...")
    zs_df = evaluate(base_model, processor, data["test_pairs"], data["test_dir"], data["vocab_canon_set"], name, "zeroshot", DEVICE)
    all_predictions.append(zs_df)

    print("Wrapping with LoRA (q_proj, v_proj) and fine-tuning CE-only...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model = train_lora_ceonly(lora_model, processor, data["train_pairs"], data["train_dir"], DEVICE)

    adapter_dir = os.path.join(OUTPUT_ROOT, f"{name}_lora_adapter")
    lora_model.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    print("Evaluating AFTER LoRA CE-only fine-tune...")
    ft_df = evaluate(lora_model, processor, data["test_pairs"], data["test_dir"], data["vocab_canon_set"], name, "lora_ceonly", DEVICE)
    all_predictions.append(ft_df)

    combined = pd.concat([zs_df, ft_df], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_ROOT, f"predictions_{name}_stage.csv"), index=False)

    for stage, df in [("zeroshot", zs_df), ("lora_ceonly", ft_df)]:
        n = len(df)
        counts = df["error_category"].value_counts()
        for cat, cnt in counts.items():
            all_summary.append(
                {"dataset": name, "stage": stage, "error_category": cat, "count": cnt, "rate": cnt / n if n else 0.0, "n_total": n}
            )

    del base_model, lora_model
    torch.cuda.empty_cache()

summary_df = pd.DataFrame(all_summary)
summary_df.to_csv(os.path.join(OUTPUT_ROOT, "summary_before_after.csv"), index=False)
print("\n=== SUMMARY (zeroshot vs lora_ceonly) ===")
print(summary_df.to_string(index=False))
print("DONE.")
