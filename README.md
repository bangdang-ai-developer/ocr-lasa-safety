# ocr-lasa-safety

Nghiên cứu độc lập: **Risk-Calibrated Fine-tuning** để giảm lỗi "đọc nhầm thành
một tên thuốc thật khác" (look-alike/sound-alike, LASA) khi fine-tune model
OCR chữ viết tay cho tên thuốc, mà không cần lexicon quy mô RxNorm và không
cần sinh dữ liệu tổng hợp.

Xem đề cương đầy đủ: [`docs/00-de-cuong-nghien-cuu.md`](docs/00-de-cuong-nghien-cuu.md).

Dự án này **độc lập hoàn toàn** với dự án `ocr-research` (đơn thuốc viết tay +
LoRA TrOCR 7-model benchmark) — không tái sử dụng code/dữ liệu/nhãn của dự án
đó, chỉ dùng làm bối cảnh động lực nghiên cứu.

## Cấu trúc

- `docs/` — đề cương, ghi chú thiết kế
- `src/` — module dùng chung (nạp dữ liệu, suy luận cặp nhầm lẫn, loss function...)
- `scripts/` — script chạy từng bước của pipeline
- `data/raw/` — dataset thô tải về (không commit, xem `.gitignore`)
- `results/` — output trung gian (CSV cặp nhầm lẫn, log huấn luyện, số liệu)
- `notebooks/` — notebook chạy trên Kaggle T4

## Hạ tầng

Kaggle T4 GPU miễn phí, tài khoản riêng `bangdang007112`. Trước MỌI lệnh
`kaggle`, chạy:

```bash
source scripts/kaggle_env.sh
```

## Trạng thái hiện tại

- [x] Tải 2 dataset công khai: Kaggle "Doctor's Handwritten Prescription BD"
      (78 lớp) và RxHandBD (~1.430 mục sau chuẩn hoá, mirror trên Kaggle của
      `abrahametry/rxhandbd-handwritten-word-image-dataset`).
- [x] Suy luận cặp tên thuốc dễ nhầm lẫn bằng thuật toán (BI-SIM: edit distance
      + phonetic similarity), xuất CSV top-100/top-5000 để kiểm tra thủ công.
- [ ] Kiểm tra thủ công (spot-check) danh sách cặp nhầm lẫn.
- [ ] Chạy OCR benchmark thu gọn (Tesseract + TrOCR zero-shot) để tự đo hiện
      tượng gốc một cách độc lập.
- [ ] Cài margin loss + severity-weighted loss vào pipeline LoRA fine-tune.
- [ ] Đánh giá ablation (CE-only / +severity / +margin / +cả hai).
