"""OCR benchmark thu gon (zero-shot): Tesseract + TrOCR-base-handwritten,
tren tap Test cua Kaggle-BD (78 lop) va RxHandBD.

Chay tren Kaggle kernel (T4 GPU). Day la buoc DO GOC lap doc lap, KHONG dung
lai nhan/ket qua cua du an ocr-research truoc do.

Output (luu vao /kaggle/working/):
  - predictions_kaggle_bd.csv, predictions_rxhandbd.csv
  - summary_error_taxonomy.csv
"""

import glob
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 0. Cai dat phu thuoc runtime (Kaggle image khong co san tesseract-ocr binary)
# ---------------------------------------------------------------------------
print("Installing system + python dependencies...")
subprocess.run(["apt-get", "-qq", "update"], check=False)
subprocess.run(["apt-get", "-qq", "install", "-y", "tesseract-ocr"], check=False)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pytesseract", "jellyfish", "rapidfuzz"],
    check=False,
)

import jellyfish  # noqa: E402
import pandas as pd  # noqa: E402
import pytesseract  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working"


# ---------------------------------------------------------------------------
# 1. Tim thu muc dataset thuc te (ten thu muc goc co the chua ky tu dac biet)
# ---------------------------------------------------------------------------
def find_dir_containing(root: str, filename_pattern: str) -> str:
    """Tim thu muc con (bat ky do sau) co chua file khop filename_pattern."""
    matches = glob.glob(os.path.join(root, "**", filename_pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Khong tim thay '{filename_pattern}' duoi {root}")
    return os.path.dirname(matches[0])


KAGGLE_BD_TEST_DIR = find_dir_containing(INPUT_ROOT, "testing_labels.csv")
_rxhandbd_matches = glob.glob(
    os.path.join(INPUT_ROOT, "**", "RxHandBDMain", "Test.csv"), recursive=True
)
if not _rxhandbd_matches:
    raise FileNotFoundError("Khong tim thay RxHandBDMain/Test.csv duoi /kaggle/input")
RXHANDBD_MAIN_DIR = os.path.dirname(_rxhandbd_matches[0])
print("Kaggle-BD test dir:", KAGGLE_BD_TEST_DIR)
print("RxHandBD main dir:", RXHANDBD_MAIN_DIR)


# ---------------------------------------------------------------------------
# 2. Tu vung + suy luan diem tuong dong (giong src/vocab.py, src/confusable_pairs.py
#    cua repo - viet lai doc lap trong kernel de khong phu thuoc upload them file)
# ---------------------------------------------------------------------------
_TRAILING_DOSAGE_RE = re.compile(r"\s*\d+(\.\d+)?\s*(mg|ml|gm|g|mcg|iu)?\s*$", re.IGNORECASE)
_DOSAGE_FORM_PREFIX_RE = re.compile(r"^(tab\.?|cap\.?|inj\.?|syp\.?|susp\.?)\s*", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.\-]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_label(raw: str) -> str:
    s = raw.strip()
    s = _TRAILING_DOSAGE_RE.sub("", s).strip()
    return s


def canonicalize_for_dedup(norm_label: str) -> str:
    s = norm_label.lower()
    s = _DOSAGE_FORM_PREFIX_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


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


CONFUSABLE_THRESHOLD = 0.65  # chon dua tren phan bo diem da quan sat o buoc suy luan cap truoc


def classify_error(true_label: str, pred_label: str, vocab_canon_set: set[str]) -> str:
    true_c = canonicalize_for_dedup(true_label)
    pred_c = canonicalize_for_dedup(pred_label)
    if not pred_c:
        return "hallucination_far_off"
    if pred_c == true_c:
        return "correct"
    if pred_c in vocab_canon_set:
        score = bi_sim_score(true_label, pred_label)
        if score >= CONFUSABLE_THRESHOLD:
            return "confusable_wrong_drug"
        return "other_wrong_real_word"
    edit_sim = normalized_edit_similarity(true_label.lower(), pred_label.lower())
    return "minor_ocr_noise" if edit_sim >= 0.7 else "hallucination_far_off"


# ---------------------------------------------------------------------------
# 3. OCR engines
# ---------------------------------------------------------------------------
print("Loading TrOCR-base-handwritten (zero-shot)...")
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(DEVICE)
trocr_model.eval()


def run_tesseract(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 7")
        return text.strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR:{e}]"


def run_trocr_batch(image_paths: list[str]) -> list[str]:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = trocr_processor(images=images, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values, max_length=32)
    return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 4. Chay benchmark tren 1 dataset
# ---------------------------------------------------------------------------
def benchmark_dataset(name: str, image_dir: str, labels: list[tuple], full_vocab_norm: list[str]) -> pd.DataFrame:
    """labels: list of (image_filename, true_label_norm)"""
    vocab_canon_set = {canonicalize_for_dedup(v) for v in full_vocab_norm}

    rows = []
    t0 = time.time()
    batch_size = 16
    for start in range(0, len(labels), batch_size):
        chunk = labels[start : start + batch_size]
        paths = [os.path.join(image_dir, fn) for fn, _ in chunk]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        trocr_preds = run_trocr_batch(paths)
        for (fn, true_label), path, trocr_pred in zip(chunk, paths, trocr_preds):
            tess_pred = run_tesseract(path)
            rows.append(
                {
                    "dataset": name,
                    "image": fn,
                    "true_label": true_label,
                    "tesseract_pred": tess_pred,
                    "tesseract_error_category": classify_error(true_label, tess_pred, vocab_canon_set),
                    "trocr_zeroshot_pred": trocr_pred,
                    "trocr_zeroshot_error_category": classify_error(true_label, trocr_pred, vocab_canon_set),
                }
            )
        if start % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            print(f"[{name}] {start}/{len(labels)} images, {elapsed:.1f}s elapsed")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Nap nhan cho tung dataset
# ---------------------------------------------------------------------------
print("Loading Kaggle-BD test labels...")
kb_test_csv = os.path.join(KAGGLE_BD_TEST_DIR, "testing_labels.csv")
kb_df = pd.read_csv(kb_test_csv)
kb_labels = list(zip(kb_df["IMAGE"], kb_df["MEDICINE_NAME"].astype(str).str.strip()))
kb_image_dir = os.path.join(KAGGLE_BD_TEST_DIR, "testing_words")

# tu vung day du Kaggle-BD (78 lop) tu ca 3 split de dam bao khong thieu lop nao
kb_all_labels = set()
for split_csv, col in [
    (os.path.join(os.path.dirname(KAGGLE_BD_TEST_DIR), "Training", "training_labels.csv"), "MEDICINE_NAME"),
    (os.path.join(os.path.dirname(KAGGLE_BD_TEST_DIR), "Validation", "validation_labels.csv"), "MEDICINE_NAME"),
    (kb_test_csv, "MEDICINE_NAME"),
]:
    if os.path.exists(split_csv):
        kb_all_labels.update(pd.read_csv(split_csv)[col].astype(str).str.strip().tolist())

print("Loading RxHandBD test labels...")
rx_test_csv = os.path.join(RXHANDBD_MAIN_DIR, "Test.csv")
rx_train_csv = os.path.join(RXHANDBD_MAIN_DIR, "Train.csv")
rx_test_df = pd.read_csv(rx_test_csv)
rx_test_df["norm_label"] = rx_test_df["Labels"].astype(str).apply(normalize_label)
rx_labels = list(zip(rx_test_df["Images"], rx_test_df["norm_label"]))
rx_image_dir = os.path.join(RXHANDBD_MAIN_DIR, "Test")
if not os.path.isdir(rx_image_dir):
    raise FileNotFoundError(f"Khong tim thay thu muc anh: {rx_image_dir}")

rx_all_labels = set()
for split_csv in [rx_train_csv, rx_test_csv]:
    df = pd.read_csv(split_csv)
    rx_all_labels.update(df["Labels"].astype(str).apply(normalize_label).tolist())

# ---------------------------------------------------------------------------
# 6. Chay va luu ket qua
# ---------------------------------------------------------------------------
print(f"Kaggle-BD: {len(kb_labels)} test images, vocab={len(kb_all_labels)}")
kb_results = benchmark_dataset("kaggle_bd", kb_image_dir, kb_labels, list(kb_all_labels))
kb_results.to_csv(os.path.join(OUTPUT_ROOT, "predictions_kaggle_bd.csv"), index=False)

print(f"RxHandBD: {len(rx_labels)} test images, vocab={len(rx_all_labels)}")
rx_results = benchmark_dataset("rxhandbd", rx_image_dir, rx_labels, list(rx_all_labels))
rx_results.to_csv(os.path.join(OUTPUT_ROOT, "predictions_rxhandbd.csv"), index=False)

# ---------------------------------------------------------------------------
# 7. Tom tat ty le loi theo taxonomy, theo model va dataset
# ---------------------------------------------------------------------------
summary_rows = []
for df, dname in [(kb_results, "kaggle_bd"), (rx_results, "rxhandbd")]:
    n = len(df)
    for model in ["tesseract", "trocr_zeroshot"]:
        col = f"{model}_error_category"
        counts = df[col].value_counts()
        for cat, cnt in counts.items():
            summary_rows.append(
                {
                    "dataset": dname,
                    "model": model,
                    "error_category": cat,
                    "count": cnt,
                    "rate": cnt / n if n else 0.0,
                    "n_total": n,
                }
            )
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_ROOT, "summary_error_taxonomy.csv"), index=False)
print(summary_df.to_string(index=False))
print("DONE.")
