# Spot-check thủ công danh sách cặp nhầm lẫn (bắt buộc theo docs/00 §4/§7)

**Lưu ý minh bạch quan trọng**: đây là spot-check hỗ trợ bởi AI (Claude), dựa trên
kiến thức chung về chính tả/ngữ âm/dược lý phổ thông — **KHÔNG PHẢI** đánh giá đã
được xác minh bởi dược sĩ người Bangladesh hoặc đối chiếu với danh mục sản phẩm
thực tế tại Bangladesh. Đây là bước kiểm tra "face validity" đầu tiên để phá vỡ
một phần rủi ro circularity (thuật toán tự chấm điểm rồi tự đánh giá), không phải
thay thế hoàn toàn cho việc này. Khuyến nghị: nếu có thể, nhờ một người quen thuộc
với tên biệt dược Bangladesh xem lại trước khi nộp bài.

## Giao thức

Với mỗi cặp (name_a, name_b) trong top-100 điểm cao nhất (đã xuất trước đó), đánh
giá theo câu hỏi: *"một người đọc vội chữ viết tay có thể nhầm name_a thành
name_b (hoặc ngược lại) không?"* Phát hiện thêm 1 loại cần tách riêng trong lúc
review: **"biến thể cùng nhãn hiệu"** — cặp chỉ khác nhau bởi hậu tố dạng bào chế/
phối hợp ngắn (vd. `Calboral` vs `Calboral D`, `Fenix` vs `Fenix-V`, `Spilac` vs
`Spilac F`) — đây KHÔNG phải 2 sản phẩm độc lập trông giống nhau tình cờ, mà là
biến thể có chủ đích của CÙNG một nhãn hiệu gốc. Thuật toán suy luận không phân
biệt được loại này với cặp nhầm lẫn LASA thật sự.

3 nhãn dùng: `co` (có, hợp lý), `khong` (không/đáng ngờ — nhiều khả năng là nhiễu
thuật toán hoặc lỗi ghi nhãn trong dataset), `bien_the_cung_nhan_hieu` (biến thể
cùng nhãn hiệu, phát hiện tự động bằng heuristic hậu tố ngắn ≤3 ký tự).

## Kết quả

| Dataset | n | co (hợp lý) | khong (đáng ngờ) | biến thể cùng nhãn hiệu |
|---|---|---|---|---|
| Kaggle-BD (top 100) | 100 | 38 | 61 | 1 |
| RxHandBD (top 100) | 100 | 42 | 40 | 18 |

**Phát hiện quan trọng — kiểm tra ngưỡng `CONFUSABLE_THRESHOLD=0.65` đang dùng
trong pipeline**:

- **Kaggle-BD**: ở đúng ngưỡng sản xuất (score ≥ 0.65, n=32): tỉ lệ hợp lý
  **22/32 = 68.75%**. Dưới ngưỡng (score < 0.65, n=68): tỉ lệ hợp lý rơi xuống
  **16/68 = 23.5%**. → **Ngưỡng 0.65 hoạt động hợp lý cho Kaggle-BD**, lọc được
  phần lớn nhiễu.
- **RxHandBD**: toàn bộ top-100 đã xuất đều có score ≥ 0.65 (chưa lấy mẫu quanh
  ranh giới ngưỡng), nên chưa đánh giá được chất lượng ngưỡng từ dữ liệu này.
  Trong top-100 (toàn bộ đều ≥ ngưỡng): chỉ **42% thật sự hợp lý là 2 nhãn độc
  lập**, **18% là biến thể cùng nhãn hiệu** (rủi ro thật nhưng khác bản chất),
  và **40% có vẻ là nhiễu thuật toán/lỗi ghi nhãn dataset**.

## Hệ quả cần nêu trong bài báo (giới hạn thật, không che giấu)

1. Với RxHandBD — dataset chính dùng cho kết quả có ý nghĩa thống kê — khoảng
   **4/10 cặp nhầm lẫn điểm cao nhất là nhiễu**, không phải cặp LASA thật. Danh
   sách hàng xóm N(w) dùng cho margin loss (xây từ cùng thuật toán) do đó cũng
   chứa một tỉ lệ nhiễu tương tự — cần nêu rõ đây là giới hạn của phương pháp,
   không phải giả định đã được xác minh hoàn toàn.
2. **Cải tiến cụ thể cho vòng tiếp theo** (chưa áp dụng vào các kernel 03-06 đã
   chạy, vì các kernel confirmatory đã chạy trước khi phát hiện này rõ ràng):
   loại bỏ tự động các cặp "biến thể cùng nhãn hiệu" (hậu tố ngắn ≤3 ký tự sau
   khi bỏ khoảng trắng/dấu câu) khỏi N(w) và khỏi nhãn taxonomy — heuristic đã
   viết sẵn, xem hàm `is_family_variant` trong spot-check script.
3. Tỉ lệ đồng thuận 42-60% (tuỳ có tính biến thể cùng nhãn hiệu là "hợp lý" hay
   không) nên được báo cáo minh bạch trong phần Methods, kèm chú thích rõ đây là
   review AI-assisted, chưa qua dược sĩ bản địa xác minh.

File CSV đã điền cột `manual_review_ok`: `results/confusable_pairs_kaggle_bd_top100.csv`,
`results/confusable_pairs_rxhandbd_top100.csv`.
