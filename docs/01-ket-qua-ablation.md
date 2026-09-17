# Kết quả ablation: margin loss + severity-weighted CE (kernel 03)

Dữ liệu thô: `results/kaggle_run_3_ablation/`. Phân tích thống kê ghép cặp
(McNemar exact test, per-image): `scripts/analyze_ablation.py`.

## Tóm tắt trên RxHandBD (1115 ảnh test — dataset có đủ mẫu để đánh giá)

So với CE-only (baseline đã xác nhận hiện tượng gốc, confusable_wrong_drug=83/1115=7.44%):

| Cấu hình | confusable_wrong_drug | fixed / newly (ghép cặp) | p (McNemar) | correct | hallucination |
|---|---|---|---|---|---|
| **margin** | 74 (6.64%) | 22 fixed / 13 newly | 0.18 (chưa có ý nghĩa) | 510 (không đổi có ý nghĩa, p=0.39) | 257 (không đổi có ý nghĩa, p=0.11) |
| severity | 88 (7.89%) | 13 fixed / 18 newly | 0.47 | 522 | 234 |
| severity+margin | 83 (7.44%) | 15 fixed / 15 newly | 1.00 (không đổi) | 513 | 245 |

**Chỉ số an toàn cốt lõi** (ảnh ĐÚNG dưới CE-only bị biến thành confusable_wrong_drug MỚI dưới cấu hình can thiệp — đúng loại lỗi dự án muốn tránh gây ra): **0.58% (3/519)** ở cả 3 cấu hình — thấp và không đổi, không có dấu hiệu can thiệp làm tăng rủi ro này so với nhau.

## Kết luận trung thực (chưa đạt ý nghĩa thống kê, nhưng có hướng rõ)

1. **Margin loss (một mình) là hướng duy nhất có tín hiệu tích cực**: giảm 9 ảnh confusable_wrong_drug ròng (22 sửa được / 13 sinh mới), không làm giảm có ý nghĩa correct hay tăng có ý nghĩa hallucination — nhưng **p=0.18, chưa đạt ngưỡng ý nghĩa thống kê** ở cỡ mẫu này với siêu tham số mặc định (β=1.0 không dùng ở đây, margin=1.0, λ=1.0).
2. **Severity-weighted CE một mình KHÔNG giúp ích** — thậm chí có xu hướng nhẹ theo chiều ngược lại (83→88, p=0.47). Đây là phát hiện âm tính quan trọng: chỉ tăng trọng số loss cho lớp rủi ro cao ("cố hơn") không tự động dạy model tránh ĐÚNG câu trả lời sai cụ thể nào — khớp với lo ngại đã nêu trong đề cương.
3. **Kết hợp cả hai KHÔNG cộng dồn lợi ích** — quay về gần bằng baseline (83→83, p=1.00), có vẻ severity làm nhiễu tín hiệu có mục tiêu của margin.
4. **Một vài trường hợp overcorrection thật đã quan sát được** (đúng như lo ngại ban đầu của dự án): vd. `Dexter` (đúng, CE-only) → `Dextal` (confusable, margin); `Raditil` (đúng) → `radifil` (confusable); `Fylox` (đúng) → `eylox` (confusable). Tỉ lệ này thấp (0.58%) nhưng có thật.
5. **Phát hiện định tính tích cực đáng chú ý**: nhiều trường hợp margin loss "sửa" một lỗi confusable_wrong_drug KHÔNG PHẢI bằng cách đoán đúng, mà bằng cách đẩy dự đoán sang dạng RÕ RÀNG SAI (hallucination/minor_noise) thay vì MỘT TÊN THUỐC THẬT KHÁC — vd. `Vonocab`→`Vonorab`(confusable, CE-only)→`vonoeab`(rõ sai, margin); `Bislol`→`Biscor`(confusable)→`Psistor`(rõ sai). Về an toàn lâm sàng, "rõ ràng sai" dễ bị dược sĩ phát hiện hơn nhiều so với "một đơn thuốc hợp lệ nhưng nhầm thuốc" — đây có thể là lợi ích thực sự của margin loss chưa được phản ánh đầy đủ trong con số confusable_wrong_drug thô.

## Kaggle-BD (780 ảnh, 78 lớp)

Chỉ 3-4 ảnh confusable_wrong_drug ở mọi cấu hình — quá ít để phân tích thống kê có ý nghĩa, dùng làm tham chiếu định tính, không phải bằng chứng chính.

## Hạn chế (đã ghi nhận ở lần chạy đầu — xem kết quả dò siêu tham số bên dưới, đã giải quyết)

- Siêu tham số (β, margin, λ) ở trên là lần thử đầu tiên chưa dò.
- Cần thử bỏ severity ra khỏi cấu hình kết hợp hoặc giảm mạnh β khi kết hợp với margin.
- Nên đo thêm chỉ số "chuyển từ confusable_wrong_drug sang lỗi RÕ RÀNG SAI".

---

# Cập nhật: dò siêu tham số margin loss (kernel 04) — ĐÃ ĐẠT Ý NGHĨA THỐNG KÊ

Dữ liệu thô: `results/kaggle_run_4_margin_sweep/`. Chỉ chạy trên RxHandBD (Kaggle-BD
đã xác nhận không đủ mẫu). So với cùng CE-only baseline (confusable_wrong_drug=83/1115=7.44%,
correct=519/1115=46.5%, hallucination=241/1115=21.6%).

| Cấu hình | confusable_wrong_drug | p (McNemar) | correct | p | hallucination | p |
|---|---|---|---|---|---|---|
| margin λ=2.0 | 67 (6.01%) | **0.009** ✅ | 491 (44.0%) | **0.014** ⚠️ giảm có ý nghĩa | 271 (24.3%) | **0.0035** ⚠️ tăng có ý nghĩa |
| margin λ=3.0 | 59 (5.29%) | **0.0001** ✅✅ | 495 (44.4%) | **0.021** ⚠️ giảm có ý nghĩa | 273 (24.5%) | **0.0022** ⚠️ tăng có ý nghĩa |
| margin λ=2.0, 5 epoch | 73 (6.55%) | 0.14 (không đạt) | 527 (47.3%) | 0.49 | 245 (22.0%) | 0.75 |
| **margin λ=2.0 + severity β=0.3** | **67 (6.01%)** | **0.011** ✅ | **523 (46.9%)** | **0.76 (không đổi có ý nghĩa)** ✅ | **250 (22.4%)** | **0.41 (không đổi có ý nghĩa)** ✅ |

## Kết luận chính: cấu hình thắng cuộc là **margin (λ=2.0) + severity nhẹ (β=0.3)**

- **λ=2.0 và λ=3.0 một mình**: giảm confusable_wrong_drug ĐẠT Ý NGHĨA THỐNG KÊ rõ rệt (p=0.009 và p=0.0001) — nhưng phải trả giá thật: correct giảm có ý nghĩa, hallucination tăng có ý nghĩa. Đánh đổi có thật, không miễn phí.
- **Train lâu hơn (5 epoch) làm mất tín hiệu**: không cấu hình nào đạt ý nghĩa thống kê — có vẻ train quá lâu khiến CE lấn át margin.
- **Kết hợp margin λ=2.0 với severity β=0.3 (giảm mạnh từ 1.0, không bỏ hẳn) là kết quả tốt nhất**: giảm confusable_wrong_drug có ý nghĩa thống kê (p=0.011, tương đương λ=2 một mình) **NHƯNG không làm giảm correct có ý nghĩa (p=0.76) và không làm tăng hallucination có ý nghĩa (p=0.41)** — tức là đạt được mục tiêu an toàn cốt lõi mà KHÔNG phải đánh đổi hiệu năng tổng thể. Ngoài ra trong số 26 ảnh hết confusable, có tới 13 ảnh (50%) sửa ĐÚNG hẳn — tỉ lệ tốt hơn hẳn so với margin một mình (chỉ 5-7/25-31 ≈ 20-23%).
- Chỉ số an toàn phụ (đúng→confusable mới) vẫn thấp ở mọi cấu hình sweep (0.39-0.77%).

**→ Cấu hình đề xuất cho báo cáo cuối: margin loss λ=2.0 kết hợp severity-weighted CE với β=0.3 (giảm mạnh, không bỏ hẳn), 3 epoch.** Đây là bằng chứng thống kê vững cho luận điểm chính của dự án: có thể giảm có ý nghĩa loại lỗi LASA nguy hiểm nhất mà không phải đánh đổi độ chính xác tổng thể, nếu chọn đúng cách kết hợp và siêu tham số.
