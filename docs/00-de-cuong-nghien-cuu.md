# Đề cương nghiên cứu: Risk-Calibrated Fine-tuning chống nhầm lẫn tên thuốc (LASA) trong OCR chữ viết tay

Dự án nghiên cứu **độc lập**, không liên quan CCI Australia, không phải OCR tiếng Việt, dữ liệu công khai, không cần IRB. Khác hoàn toàn với dự án `ocr-research` (đơn thuốc viết tay + LoRA TrOCR + 7 model benchmark) — dự án đó chỉ dùng làm bối cảnh động lực nghiên cứu, KHÔNG tái sử dụng code/dữ liệu/nhãn của dự án đó.

## 1. Động lực

Quan sát gốc (từ dự án trước, sẽ được **tự kiểm chứng độc lập** trên chính dataset công khai của dự án này, không copy số liệu cũ): sau khi fine-tune một model OCR chữ viết tay, lỗi "output vô nghĩa" (hallucination) giảm mạnh, nhưng lỗi "đọc nhầm thành MỘT TÊN THUỐC THẬT KHÁC" (confusable_wrong_drug / look-alike-sound-alike, LASA) lại tăng — loại lỗi nguy hiểm hơn vì trông giống đơn thuốc hợp lệ.

## 2. Đóng góp chính

Không xây một lớp hậu xử lý (post-hoc) hay bộ phân loại gắn cờ riêng biệt. Thay vào đó: **sửa ngay hàm loss dùng để fine-tune model OCR**, kết hợp 2 thành phần bổ sung, để bản thân quá trình huấn luyện học cách tránh loại lỗi LASA thay vì phải vá sau:

- **(a) Margin loss chống nhầm lẫn cụ thể**: với mỗi ảnh có nhãn đúng `w`, lấy tập "hàng xóm dễ nhầm" `N(w)` từ danh sách cặp nhầm lẫn suy luận thuật toán (mục 4). Decoder chạy thêm 1 lần teacher-forcing với nhãn giả `w' ∈ N(w)`, tính `logP(w'|x)`, phạt `max(0, margin − (logP(w|x) − logP(w'|x)))`. Không cần ảnh ghép cặp đặc biệt — bất kỳ ảnh nào của lớp `N(w)` sẵn có là đủ.
- **(b) Severity-weighted cross-entropy** (thích ứng từ Risk-Calibrated Learning, Mohammadi-Seif & Baeza-Yates, IJCNN 2026, arXiv:2604.12693 — bài gốc cho phân loại ảnh 1 nhãn, ở đây thích ứng cho sinh chuỗi): nhân hệ số cho CE loss mỗi mẫu theo độ tương đồng-nguy-hiểm của lớp đó: `weight = 1 + β·max_similarity(w, other classes)`.
- **Tổng loss**: `L = Σᵢ wᵢ·CEᵢ + λ·margin_loss`.

Cơ chế giả thuyết: LoRA fine-tune "sắc nét hoá" prior ngôn ngữ có sẵn trong decoder (TrOCR decoder khởi tạo từ RoBERTa) theo hướng ưu tiên chuỗi phổ biến/hợp lý — với cặp LASA, "chuỗi hợp lý khác" chính là tên thuốc thật khác. Margin loss can thiệp trực tiếp vào cơ chế này; severity-weighting phân bổ nhiều "công sửa" hơn vào lớp rủi ro cao.

## 3. Dataset (tự chạy toàn bộ, độc lập, không dùng lại kết quả dự án trước)

- **RxHandBD** — tập chính. Cần chốt 1 bản canonical trước khi dùng (2 bản mâu thuẫn metadata):
  - Zenodo DOI [10.5281/zenodo.18478741](https://zenodo.org/records/18478741) — v1, 128×128px, MIT license.
  - Mendeley DOI [10.17632/dsb5r6vskg.3](https://data.mendeley.com/datasets/dsb5r6vskg/3) — v3, 512×512px, CC BY 4.0.
  - 5.578 ảnh, ~1.559 mục duy nhất (gồm cả token liều lượng/hướng dẫn, cần lọc ra tên thuốc thật).
- **Kaggle "Doctor's Handwritten Prescription BD"** — tập phụ/kiểm tra chéo, 78 lớp, 4.680 ảnh. Dùng tài khoản Kaggle riêng `bangdang007112` (xem `scripts/kaggle_env.sh`).
- Cả hai là tên biệt dược Bangladesh — **không khớp RxNorm/ISMP** (đã xác nhận ở vòng nghiên cứu trước: "Sergel" → 0 kết quả RxNav, "Napa" → nhầm ra "Napabucasin"). Vì vậy dùng cặp nhầm lẫn tự suy luận (mục 4), không dùng ISMP.

## 4. Suy luận cặp nhầm lẫn (thuật toán, không phải ISMP)

Kết hợp edit-distance có trọng số + độ tương đồng ngữ âm (double metaphone / soft bigram similarity), theo phương pháp Kondrak & Dorr (BI-SIM/ALINE) — thuật toán tổng quát, đã được kiểm chứng bằng cách so khớp với danh sách nhầm lẫn thực tế (USP/ISMP) trước khi công bố, và đã có tiền lệ áp dụng cho thị trường thuốc ngoài Mỹ thật (Úc-TGA "LASA v2", Canada, Iran). Sau khi thuật toán xếp hạng, **kiểm tra thủ công một mẫu (top 30-50 cặp điểm cao nhất)** để giảm rủi ro circularity (không có ground-truth Bangladesh độc lập).

## 5. Pipeline thí nghiệm

1. Chốt bản RxHandBD, lọc từ vựng thật (loại token liều lượng/hướng dẫn).
2. Dựng + kiểm tra thủ công danh sách cặp nhầm lẫn.
3. OCR benchmark thu gọn (không cần lặp lại 8-model benchmark của dự án trước): Tesseract (baseline rẻ) + TrOCR-base-handwritten zero-shot — để có số liệu gốc **tự đo, độc lập**.
4. Fine-tune LoRA TrOCR với 4 cấu hình loss để so sánh (ablation): CE-only / CE+severity / CE+margin / CE+severity+margin.
5. Đánh giá dưới **free-running decoding thật** (không chỉ teacher-forcing likelihood).

## 6. Đánh giá

CER/WER tổng thể + tỉ lệ lỗi confusable_wrong_drug riêng (trước/sau mỗi cấu hình loss), kiểm định thống kê ghép cặp (Wilcoxon/bootstrap CI).

## 7. Rủi ro chính

- **Overfitting mẫu nhỏ**: mỗi lớp chỉ có vài chục ảnh — cần tách biệt nghiêm ngặt pool negative với tập test, dò `β`/`λ` thận trọng, early-stopping theo đúng metric cặp-nhầm-lẫn.
- Cỡ mẫu dương tính (confusable_wrong_drug) có thể ít — kiểm tra tỉ lệ thật ngay từ tuần đầu trước khi cam kết phân tích thống kê chi tiết.
- License RxHandBD (Zenodo vs Mendeley) cần làm rõ trước khi công bố công khai.

## 8. Tính mới

Chưa có nghiên cứu nào kết hợp: (a) model sinh chuỗi tự hồi quy (encoder-decoder OCR/HTR) + (b) loss định hướng rõ theo danh sách cặp nhầm lẫn cụ thể + (c) miền tên thuốc y tế. Các nghiên cứu gần nhất (BioSyn/CODER, SeqCLR/RCLSTR/CLAPS, Risk-Calibrated Learning) chỉ chiếm từng góc riêng lẻ.

## 9. Hạ tầng
- Kaggle T4 GPU miễn phí, tài khoản riêng `bangdang007112` cho dự án này (xem `scripts/kaggle_env.sh` trước khi chạy bất kỳ lệnh `kaggle` nào).
- Git author: `bangdang007112 <bangdang007112@gmail.com>` (đã set local, không set global).
