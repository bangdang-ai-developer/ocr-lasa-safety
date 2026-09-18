# Tài liệu tham khảo đã xác minh

Tất cả trích dẫn dưới đây đã được xác minh thật (không phải AI bịa) qua tra
cứu trực tiếp arXiv/ACL Anthology/CrossRef/PubMed/SpringerLink — không trích
từ trí nhớ. Format: APA-giản lược.

## Cơ chế loss chính (severity-weighted CE)

- Mohammadi-Seif, A., & Baeza-Yates, R. (2026). *Risk-Calibrated Learning:
  Minimizing Fatal Errors in Medical AI*. arXiv:2604.12693 [cs.CV]. Chấp nhận
  tại IJCNN 2026. https://arxiv.org/abs/2604.12693
  — Xác nhận đúng như mô tả: nhúng ma trận mức-độ-nguy-hiểm lâm sàng vào loss
  cho phân loại ảnh y tế (Brain Tumor MRI, ISIC 2018, BreaKHis, SICAPv2), giảm
  20.0-92.4% Critical Error Rate so với Focal Loss. **Đây là bài cho bài toán
  phân loại 1 nhãn** — dự án này thích ứng sang bối cảnh sinh chuỗi (OCR), cần
  nêu rõ đây là "thích ứng", không phải tái tạo trực tiếp.

## Bài liên quan gần (cần trích dẫn để phân biệt, tránh bị reviewer coi là bỏ sót)

- Alansary, A., Mohamed, M., & Hamdi, A. (2026). *Severity-Aware Weighted Loss
  for Arabic Medical Text Generation*. arXiv:2604.06346 [cs.CL]. Chấp nhận tại
  ICTIS 2026. https://arxiv.org/abs/2604.06346
  — Severity-weighting cấp token cho sinh văn bản hội thoại y tế tiếng Ả Rập
  (LLM), KHÔNG phải OCR/thị giác. ⚠️ Cẩn thận: có 1 bài khác rất gần ID
  (arXiv:2604.06365, curriculum learning, khác phương pháp) — không nhầm.
- Tupakula, M. (2025). *Thin Bridges for Drug Text Alignment: Lightweight
  Contrastive Learning for Target Specific Drug Retrieval*. arXiv:2510.03309.
  https://arxiv.org/abs/2510.03309
  — ⚠️ **Mô tả trước đây SAI**: bài này KHÔNG phải về tên thuốc dễ nhầm lẫn
  (LASA). Đây là contrastive learning căn chỉnh embedding phân tử (ECFP4) với
  văn bản y sinh (ChEMBL) để truy hồi thuốc theo ĐÍCH TÁC ĐỘNG (target), có
  dùng margin loss + hard-negative nhưng cho mục đích khác hẳn. Nếu trích dẫn,
  phải sửa lại mô tả cho đúng — hoặc bỏ khỏi related work vì không thực sự liên
  quan đến LASA.

## Chuẩn hoá thực thể y sinh bằng contrastive/hard-negative (text-only, không OCR)

- Sung, M., Jeon, H., Lee, J., & Kang, J. (2020). *Biomedical Entity
  Representations with Synonym Marginalization*. ACL 2020, pp. 3641–3650.
  DOI: 10.18653/v1/2020.acl-main.335. https://aclanthology.org/2020.acl-main.335/
- Yuan, Z., Zhao, Z., Sun, H., Li, J., Wang, F., & Yu, S. (2022). *CODER:
  Knowledge-infused cross-lingual medical term embedding for term
  normalization*. Journal of Biomedical Informatics, 126, 103983.
  DOI: 10.1016/j.jbi.2021.103983. (arXiv:2011.02947)

## Contrastive learning cho sinh chuỗi/nhận diện văn bản (general, không nhắm cặp nhầm lẫn cụ thể)

- Aberdam, A., Litman, R., Tsiper, S., Anschel, O., Slossberg, R., Mazor, S.,
  Manmatha, R., & Perona, P. (2021). *Sequence-to-Sequence Contrastive
  Learning for Text Recognition*. CVPR 2021. arXiv:2012.10873.
- Zhang, J., Lin, T., Xu, Y., Chen, K., & Zhang, R. (2023). *Relational
  Contrastive Learning for Scene Text Recognition*. ACM MM 2023.
  arXiv:2308.00508.
- Lee, S., Lee, D. B., & Hwang, S. J. (2021). *Contrastive Learning with
  Adversarial Perturbations for Conditional Text Generation*. ICLR 2021.
  arXiv:2012.07280. (Lưu ý: bài về sinh văn bản MT/tóm tắt, không phải OCR.)

## Thuật toán suy luận cặp nhầm lẫn tên thuốc (nền tảng cho src/confusable_pairs.py)

- Kondrak, G., & Dorr, B. (2004). *Identification of Confusable Drug Names: A
  New Approach and Evaluation Methodology*. COLING 2004, pp. 952–958.
  DOI: 10.3115/1220355.1220492.
- Kondrak, G., & Dorr, B. (2006). *Automatic identification of confusable drug
  names*. Artificial Intelligence in Medicine, 36(1), 29–42.
  **DOI đúng: 10.1016/j.artmed.2005.07.005** (DOI đoán trước đó
  10.1016/j.artmed.2005.06.001 là SAI — trỏ nhầm sang 1 bài khác).
- Millán-Hernández, C. E., García-Hernández, R. A., Ledeneva, Y., &
  Hernández-Castañeda, Á. (2019). *Soft Bigram Similarity to Identify
  Confusable Drug Names*. MCPR 2019, LNCS vol. 11524, pp. 433–442. Springer.
  DOI: 10.1007/978-3-030-21077-9_40.
  (Lưu ý: 1 đồng tác giả trước đây bị thiếu trong ghi chú nội bộ — đã bổ sung.)

## Việc cần làm thêm trước khi nộp bài

- Kiểm tra tay abstract đầy đủ của Millán-Hernández et al. 2019 (SpringerLink
  yêu cầu đăng nhập khi tra cứu tự động) trước khi trích số liệu cụ thể
  ("vượt 17 thước đo khác trên 396.900 cặp") trong bài viết.
- Sửa mọi chỗ đã lỡ mô tả arXiv:2510.03309 là về "tên thuốc dễ nhầm lẫn".
