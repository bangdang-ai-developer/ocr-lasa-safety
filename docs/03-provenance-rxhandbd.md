# Truy nguồn gốc/license RxHandBD

**Cập nhật quan trọng (đã tự tải + đối chiếu trực tiếp file gốc từ Zenodo)**:
tình hình **tốt hơn nhiều** so với đánh giá ban đầu. Vấn đề license vẫn còn,
nhưng KHÔNG còn nghi ngờ về việc dữ liệu có bị lẫn/giả hay không.

## Đã tự kiểm chứng trực tiếp (không chỉ dựa trên metadata trang web)

Tự tải trực tiếp cả 2 file trên Zenodo (`RxHand Original.zip`, `RxHandBD.zip`),
xác nhận MD5 checksum khớp 100% với Zenodo công bố, rồi đối chiếu tay với dữ
liệu đang dùng trong dự án (`data/raw/rxhandbd/RxHandBDMain/`):

1. **Đối chiếu nhãn văn bản theo đúng ID ảnh — khớp 13/13 mẫu ngẫu nhiên**
   (P0001→"Nexcital", P0100→"Etorix", P1000→"celebrex", P0050→"CORTAN",
   P2000→"Napa extra", P4200→"Ceevit", v.v. — giống hệt giữa bản Zenodo chính
   thức và bản Kaggle đang dùng). **→ Xác nhận chắc chắn: đây đúng là dataset
   RxHandBD thật của Md. Masudul Islam (BUBT), không phải dữ liệu giả hay dữ
   liệu của người khác.**
2. **Xác nhận code CHƯA BAO GIỜ đụng vào thư mục bị "gộp nhầm"**: toàn bộ
   kernel (01-06) dùng glob pattern bắt buộc đúng tên thư mục `RxHandBDMain`
   (`glob.glob(.../ "RxHandBDMain" / "Test.csv")`) — nghĩa là dù bản tải Kaggle
   có bundle thêm thư mục `BD Handwritten Prescription Dataset` (dataset khác,
   license khác) vào chung 1 lượt tải, **code không bao giờ đọc từ thư mục đó
   cho các thí nghiệm gắn nhãn "rxhandbd"**. Dataset "kaggle_bd" trong dự án
   được tải RIÊNG, trực tiếp từ chính trang Kaggle gốc của nó (mamun1113), có
   license rõ ràng (Open Database License) từ đầu. → **Không có lẫn dữ liệu
   giữa 2 "dataset" trong kết quả thí nghiệm đã chạy.**
3. **Khác biệt còn lại chỉ ở xử lý ảnh, không phải nội dung**: file ảnh trong
   bản Kaggle đang dùng là 128×64 RGB (đã chuẩn hoá, không phải "raw" kích
   thước tuỳ ý như báo cáo trước — số đo "130-230×55-105px" trước đó là SAI,
   đã tự đo lại 20 mẫu cho kết quả nhất quán 128×64). Cả 2 file Zenodo chính
   thức đều là 128×128 grayscale (khác cả về kích thước lẫn màu). → Người
   chuẩn bị bản Kaggle đã tự resize/xử lý ảnh theo cách riêng, KHÔNG dùng
   nguyên file Zenodo/Mendeley — nhưng nội dung (ảnh chữ viết tay + nhãn) vẫn
   đúng là cùng bộ dữ liệu gốc, chỉ khác bước xử lý ảnh.

## Vấn đề license — vẫn cần xử lý, nhưng đơn giản hơn nhiều

Zenodo (MIT) và Mendeley (CC BY 4.0) vẫn mâu thuẫn nhau về license cho cùng
dữ liệu của cùng tác giả — đây là sự thiếu nhất quán từ chính tác giả gốc,
không phải lỗi của bản Kaggle. Vì nội dung đã xác nhận là thật, **giải pháp
đơn giản**: trích dẫn thẳng bản gốc (khuyến nghị Mendeley DOI
10.17632/dsb5r6vskg.3, bản mới nhất v3), ghi công đầy đủ tác giả, và tuân theo
điều khoản CC BY 4.0 (điều khoản có ràng buộc chặt hơn — tuân theo điều này
thì dù license thật là MIT hay CC-BY cũng đều hợp lệ).

## Khuyến nghị cập nhật (thay thế khuyến nghị cũ)

1. **Không cần đổi dataset, không cần chạy lại thí nghiệm** — nội dung đã xác
   minh là thật và đúng.
2. Trong **Data Availability statement**: ghi rõ "ảnh sử dụng lấy qua bản
   mirror trên Kaggle (URL cụ thể); đã đối chiếu độc lập tương ứng ID-ảnh↔nhãn
   với bản gốc do Md. Masudul Islam (BUBT) công bố trên Mendeley Data (DOI
   10.17632/dsb5r6vskg.3, CC BY 4.0) và Zenodo (DOI 10.5281/zenodo.18478741,
   MIT) — khớp 100% trên mẫu kiểm tra; bước xử lý/resize ảnh cụ thể của bản
   Kaggle không được ghi lại độc lập với bản gốc." Trích dẫn tác giả gốc, tuân
   theo CC BY 4.0.
3. **Vẫn KHÔNG tái phân phối trực tiếp file ảnh** trong bản công bố code của
   dự án (dù đã rõ nguồn, vẫn nên trỏ người đọc về nguồn gốc thay vì tự host
   lại, theo đúng tinh thần CC BY 4.0 là ghi công + để nguyên nguồn).
4. Không cần hạ vai trò RxHandBD trong bài báo nữa — dataset đã xác minh là
   thật, chỉ cần trích dẫn đúng và minh bạch về bước xử lý ảnh không độc lập.
5. (Không bắt buộc) Vẫn có thể liên hệ tác giả để hỏi rõ license nào là chính
   thức, nhưng không còn là việc chặn tiến độ.

## Nguồn đã tự kiểm chứng
- Zenodo: https://zenodo.org/records/18478741 (đã tải, MD5 khớp)
- Mendeley: https://data.mendeley.com/datasets/dsb5r6vskg/3
- Bản Kaggle đang dùng: https://www.kaggle.com/datasets/abrahametry/rxhandbd-handwritten-word-image-dataset
