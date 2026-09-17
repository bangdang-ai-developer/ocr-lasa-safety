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

## Hạn chế & bước tiếp theo

- Siêu tham số (β, margin, λ) là **lần thử đầu tiên chưa dò**, không phải giá trị tối ưu — margin loss cho tín hiệu đúng hướng nhưng cần tăng λ (vd. thử 2.0-3.0) hoặc train nhiều epoch hơn để tín hiệu đủ mạnh đạt ý nghĩa thống kê.
- Cần thử bỏ severity ra khỏi cấu hình kết hợp hoặc giảm mạnh β khi kết hợp với margin, vì severity một mình không giúp và có thể đang "nhiễu" margin.
- Nên đo thêm chỉ số "chuyển từ confusable_wrong_drug sang lỗi RÕ RÀNG SAI" (không chỉ "sửa đúng") như một chỉ số an toàn phụ, vì phát hiện định tính cho thấy đây có thể là lợi ích thực sự bị bỏ sót trong số liệu thô.
