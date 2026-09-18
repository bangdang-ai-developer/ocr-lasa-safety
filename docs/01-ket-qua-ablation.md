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

---

# Cập nhật: kiểm tra chéo cấu hình thắng cuộc trên Kaggle-BD (kernel 05)

Dữ liệu thô: `results/kaggle_run_5_winning_kaggle_bd/`. So với CE-only baseline Kaggle-BD (confusable_wrong_drug=3/780=0.38%, correct=684/780=87.7%, hallucination=44/780=5.6%).

| Chỉ số | CE-only | Cấu hình thắng cuộc | Ghép cặp | p (McNemar) |
|---|---|---|---|---|
| confusable_wrong_drug | 3 (0.38%) | 4 (0.51%) | 1 sửa / 2 sinh mới | 1.00 (hoàn toàn không có thông tin — n quá nhỏ) |
| correct | 684 (87.7%) | 671 (86.0%) | 24 mất / 11 được | **0.041 (giảm có ý nghĩa)** ⚠️ |
| hallucination | 44 (5.6%) | 51 (6.5%) | 11 sửa / 18 sinh mới | 0.26 (không có ý nghĩa) |

**Kết luận trung thực: kết quả KHÔNG lặp lại được trên Kaggle-BD.** Đúng như dự đoán, cỡ mẫu confusable_wrong_drug (chỉ 3-4 ảnh) hoàn toàn không đủ để nói gì về chỉ số mục tiêu (p=1.00). Đáng chú ý hơn: trên dataset này, cấu hình thắng cuộc lại làm **giảm correct rate có ý nghĩa thống kê** (87.7%→86.0%, p=0.041) mà không có lợi ích bù lại nào đo được. Diễn giải hợp lý nhất: lợi ích của margin+severity nhẹ có thể phụ thuộc vào **quy mô/độ đa dạng từ vựng** — trên RxHandBD (~1430 từ, nhiều cặp nhầm lẫn thật) can thiệp có không gian để phát huy; trên Kaggle-BD (từ điển đóng chỉ 78 lớp, chỉ 3-4 ảnh thuộc đúng loại lỗi mục tiêu) can thiệp chủ yếu chỉ thêm nhiễu vào một bài toán đã gần bão hoà, không có đủ "chỗ" để mang lại lợi ích rõ ràng.

---

# Cập nhật QUAN TRỌNG NHẤT: xác nhận lại với 5 seed đầy đủ (kernel 06) — SỬA LẠI kết luận "không đánh đổi"

Dữ liệu thô: `results/kaggle_run_6_confirmatory/`. Phân tích: `scripts/analyze_confirmatory.py`.
Bối cảnh: vòng phản biện phát hiện "cấu hình thắng cuộc" (p=0.011) không qua được
hiệu chỉnh Holm-Bonferroni (p_holm=0.057, xem `scripts/multiple_comparisons.py`)
vì được chọn sau khi xem 7 cấu hình trên cùng 1 tập test, và mỗi cấu hình trước
đó chỉ chạy **1 lần, không seed đầy đủ** (chỉ `random.seed(42)` cho Python, chưa
seed torch/numpy). Kernel 06 chạy lại **5 seed độc lập** (torch+numpy+random đều
được set), mỗi seed train CẶP (CE-only, thắng cuộc) với CÙNG seed để cô lập biến
nhiễu do khởi tạo/thứ tự dữ liệu, so sánh ghép cặp riêng cho từng seed.

## Kết quả từng seed (CE-only vs cấu hình thắng cuộc)

| Seed | confusable_wrong_drug | p | correct | p | hallucination | p |
|---|---|---|---|---|---|---|
| 1 | 79→65 (giảm) | 0.049 ✅ | 522→495 (giảm) | 0.011 ⚠️ | 242→268 (tăng) | 0.018 ⚠️ |
| 2 | 74→68 (giảm) | 0.362 | 524→508 (giảm) | 0.101 | 238→275 (tăng) | 0.0002 ⚠️ |
| 3 | 90→79 (giảm) | 0.043 ✅ | 513→503 (giảm) | 0.368 | 238→254 (tăng) | 0.109 |
| 4 | 78→66 (giảm) | 0.036 ✅ | 526→498 (giảm) | 0.010 ⚠️ | 235→255 (tăng) | 0.045 ⚠️ |
| 5 | 78→71 (giảm) | 0.265 | 529→498 (giảm) | 0.005 ⚠️ | 243→264 (tăng) | 0.033 ⚠️ |

**Điểm mấu chốt: cả 3 chỉ số đều đổi CÙNG MỘT HƯỚNG ở TẤT CẢ 5/5 SEED** —
confusable_wrong_drug luôn giảm, correct luôn giảm, hallucination luôn tăng —
không có ngoại lệ. Đây là bằng chứng nhất quán rất mạnh, không phải ngẫu nhiên.

## Gộp 5 seed (pooled McNemar, tăng lực thống kê ~5 lần)

| Chỉ số | Sửa được / Sinh mới (gộp) | p gộp | Số seed riêng lẻ đạt p<0.05 |
|---|---|---|---|
| confusable_wrong_drug | 103 / 53 | **p = 0.000077** | 3/5 |
| correct | 312 mất / 200 được | **p = 0.000001** | 3/5 |
| hallucination | 178 sửa / 298 sinh mới | **p < 0.000001** | 4/5 |

Quy đổi tỉ lệ trung bình 5 seed: confusable_wrong_drug **7.16%→6.26%** (giảm
~12.6% tương đối), correct **46.9%→44.9%** (giảm ~2.0 điểm %), hallucination
**21.5%→23.6%** (tăng ~2.2 điểm %).

## PHẢI SỬA LẠI kết luận trước đó

Kết luận cũ ("giảm confusable_wrong_drug có ý nghĩa MÀ KHÔNG đánh đổi correct/
hallucination có ý nghĩa", dựa trên 1 lần chạy p=0.76/p=0.41) **SAI, hoặc ít nhất
không đại diện** — lần chạy đơn lẻ đó là một **outlier thuận lợi tình cờ**, đúng
như lo ngại "chỉ 1 lần chạy không seed" của vòng phản biện. Với 5 seed độc lập,
correct rate GIẢM và hallucination TĂNG **nhất quán và có ý nghĩa thống kê rất
mạnh** (p gộp < 0.00001 cho cả 2) — đây là **một đánh đổi thật, không phải miễn
phí**.

**Tin tốt**: bản thân lợi ích cốt lõi (giảm confusable_wrong_drug — loại lỗi
nguy hiểm nhất) được xác nhận **VỮNG CHẮC HƠN nhiều** so với trước (p gộp
=0.000077, dựa trên 5 lần lặp độc lập, thay vì 1 lần chạy với p=0.011 dễ bị nghi
ngờ là may mắn).

## Tuyên bố đúng cho bài báo (thay thế hoàn toàn tuyên bố "không đánh đổi" cũ)

> Margin loss (λ=2.0) kết hợp severity-weighting nhẹ (β=0.3) làm giảm có ý nghĩa
> thống kê, nhất quán qua nhiều lần chạy độc lập, tỉ lệ lỗi "nhầm sang thuốc
> thật khác" (confusable_wrong_drug, ~12.6% tương đối) — nhưng đây là MỘT ĐÁNH
> ĐỔI THẬT: đi kèm giảm ~2 điểm % correct rate và tăng ~2 điểm % hallucination,
> cả hai đều có ý nghĩa thống kê mạnh qua 5 seed độc lập. Đây là một điểm trên
> đường cong đánh đổi an toàn-hiệu năng, không phải một cải tiến miễn phí.

Điều này thực ra làm bài báo **trung thực và đáng tin hơn**, đúng tinh thần dự
án: không phải "chúng tôi giải quyết được vấn đề mà không mất gì", mà là "chúng
tôi định lượng chính xác đánh đổi giữa an toàn và hiệu năng, và cho công cụ để
người dùng chọn điểm vận hành phù hợp với mức độ chấp nhận rủi ro của họ" (đúng
như đề xuất "trình bày cả đường cong đánh đổi" từ vòng phản biện trước).

**Hệ quả cho báo cáo cuối cùng**: phải nêu rõ đây là giới hạn thật của phương pháp — hiệu quả được chứng minh có ý nghĩa thống kê trên RxHandBD (từ vựng lớn, thực tế hơn), nhưng KHÔNG khẳng định được (và có dấu hiệu ngược lại về correct rate) trên dataset từ điển đóng nhỏ. Không nên trình bày cấu hình thắng cuộc như một giải pháp tổng quát cho mọi quy mô từ vựng.

---

# Phân tích mở rộng trên dữ liệu kernel 06 (effect size + CI, không cần chạy Kaggle thêm)

Script: `scripts/analyze_confirmatory_extended.py`. Bổ sung cho phần trên (vốn
chỉ có p-value) bằng effect size (risk difference) + khoảng tin cậy 95%, và
đào sâu breakdown "confusable được sửa thì đi về đâu" gộp trên cả 5 seed
(n lớn hơn nhiều so với kernel 03 đơn lẻ).

## Effect size + 95% CI (gộp 5 seed, n=5575 cặp ảnh)

| Chỉ số | Risk difference (thắng cuộc − CE-only) | 95% CI |
|---|---|---|
| confusable_wrong_drug | **−0.90 điểm %** | [−1.34, −0.46] |
| correct | **−2.01 điểm %** | [−2.80, −1.22] |
| hallucination_far_off | **+2.15 điểm %** | [+1.39, +2.92] |

Cả 3 khoảng tin cậy đều **không chứa 0** — nhất quán với p-value đã báo cáo,
nhưng giờ có độ lớn hiệu ứng + độ bất định cụ thể, đúng chuẩn báo cáo mà
reviewer tạp chí Q2 thường yêu cầu (không chỉ p-value).

## Lỗi confusable_wrong_drug "được sửa" thì đi về đâu? (gộp 5 seed, n=399 lỗi confusable ở baseline)

| | Số lượng | % trong số "hết confusable" |
|---|---|---|
| Tổng số lỗi confusable ở CE-only (gộp 5 seed) | 399 | — |
| Hết confusable dưới cấu hình thắng cuộc | 103 (25.8% của 399) | 100% |
| — trong đó sửa ĐÚNG hẳn (correct) | 36 | 35.0% |
| — trong đó thành RÕ RÀNG SAI (minor_ocr_noise 37 + hallucination 30) | 67 | **65.0%** |

**Ý nghĩa lâm sàng**: khi can thiệp "sửa" được một lỗi nguy hiểm (nhầm sang
tên thuốc thật khác), phần lớn (65%, n=103, đáng tin hơn nhiều so với quan sát
định tính nhỏ lẻ ở kernel 03) không phải vì model đoán đúng, mà vì nó đẩy dự
đoán sang một lỗi RÕ RÀNG SAI — dễ bị dược sĩ/bác sĩ phát hiện hơn nhiều so
với "một đơn thuốc hợp lệ nhưng nhầm thuốc". Đây là lợi ích an toàn thực sự
mà con số confusable_wrong_drug thô không phản ánh hết, và nay đã được lượng
hoá vững chắc trên n đủ lớn.

## Tỉ lệ "quá tay" (đúng ở CE-only → confusable MỚI ở cấu hình thắng cuộc), gộp 5 seed

**16/2614 = 0.61%** (Wilson 95% CI: [0.38%, 0.99%]).

So với mức giảm correct rate tổng thể (~2 điểm %, xem trên), phần bị "quá tay"
thành đúng LOẠI LỖI NGUY HIỂM MÀ can thiệp đang cố tránh chỉ chiếm một phần rất
nhỏ (0.61% trên tổng số ảnh vốn đúng) — phần lớn "mất correct" còn lại chuyển
thành minor_ocr_noise/hallucination (ít nguy hiểm hơn), không phải confusable
mới. Điểm này nên đưa vào phần Discussion để làm rõ bản chất đánh đổi: cái giá
phải trả (mất ~2pp correct) chủ yếu KHÔNG phải ở dạng nguy hiểm nhất.

**Kết luận**: 2 phân tích bổ sung này không thay đổi kết luận cốt lõi (đánh đổi
thật, đã xác nhận multi-seed) nhưng làm bài báo giàu thông tin định lượng hơn
nhiều — có effect size + CI cho phần Results, và một luận điểm Discussion mới
("cái giá phải trả chủ yếu không phải loại nguy hiểm nhất, và khi sửa được lỗi
nguy hiểm thì phần lớn là đẩy sang lỗi dễ phát hiện hơn chứ không phải đoán
đúng") giúp câu chuyện đánh đổi cân bằng, thuyết phục hơn thay vì chỉ nêu số
tăng/giảm trần trụi.
