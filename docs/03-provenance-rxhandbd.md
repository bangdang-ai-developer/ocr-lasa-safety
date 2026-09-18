# Truy nguồn gốc/license RxHandBD — phát hiện nghiêm trọng, cần xử lý trước khi công bố

Điều tra độc lập (đọc trực tiếp trang Kaggle qua trình duyệt thật, đối chiếu
Zenodo/Mendeley, tìm bài báo học thuật, kiểm tra hồ sơ người upload). Kết luận
ngắn gọn: **nguồn gốc và license của bản dữ liệu đang dùng KHÔNG THỂ xác nhận
chắc chắn** — nghiêm trọng hơn nghi ngờ ban đầu.

## Phát hiện chính

1. **Zenodo và Mendeley là 2 bản tự nộp (không qua bình duyệt)** bởi cùng một
   tác giả (Md. Masudul Islam, Bangladesh University of Business and Technology)
   nhưng **MÂU THUẪN NHAU**: Zenodo v1 ghi 128×128px + license MIT; Mendeley v3
   ghi 512×512px + license **CC BY 4.0** — cho cùng một dataset 5.578 ảnh. Không
   có cách nào từ bên ngoài biết license nào là "chính thức".
2. **Không tồn tại bài báo học thuật (peer-reviewed) nào mô tả "RxHandBD"** —
   đã tìm cả Google Scholar của chính tác giả (có ~15 bài thật, về da liễu/gạo/
   nén ảnh..., không có bài nào về OCR/chữ viết tay/đơn thuốc). Dataset chưa
   từng được trích dẫn.
3. **Bản Kaggle đang dùng (`abrahametry/rxhandbd-handwritten-word-image-dataset`)
   có dấu hiệu là một bản re-upload không có nguồn gốc rõ ràng**: mục Author và
   DOI-Citation trên trang Kaggle đều để TRỐNG; người upload (`abrahametry`) là
   một giảng viên CSE tại CÙNG trường BUBT (không phải chính tác giả), đã từng
   re-upload một dataset khác của cùng nhóm nghiên cứu lên Kaggle theo đúng kiểu
   này trước đây (dataset giống lúa "Aruzz22.5K") — đây có vẻ là thói quen
   "mirror nội bộ", không phải hành vi đánh cắp từ người lạ, nhưng **vẫn hoàn
   toàn không có trích dẫn/giấy phép rõ ràng trên chính trang Kaggle**.
4. **Phát hiện nghiêm trọng nhất**: thư mục con `BD Handwritten Prescription
   Dataset` bên trong bản Kaggle "RxHandBD" chứa **chính xác 4.680 file** —
   trùng khớp hoàn toàn với một dataset HOÀN TOÀN KHÁC, có bài báo bình duyệt
   riêng (Mia et al. 2024, iCACCESS) và license khác (Open Database License,
   không phải MIT/CC-BY) — đây là dataset Kaggle-BD (78 lớp) mà dự án này ĐANG
   DÙNG LÀM DATASET RIÊNG BIỆT! → bản Kaggle "RxHandBD" nhiều khả năng đã **gộp
   nhầm/không ghi công 2 dataset khác nhau** vào 1 lượt tải, dưới 1 nhãn license
   duy nhất.
5. Kích thước ảnh thực tế đo được (~130-230×55-105px, không vuông) khớp hợp lý
   với biến thể **"Raw/gốc"** mà cả Zenodo lẫn Mendeley đều có công bố riêng
   (khác với biến thể "AI-compatible" đã resize vuông 128×128 hay 512×512) — nên
   đây RẤT CÓ THỂ vẫn là ảnh RxHandBD thật, chỉ là biến thể chưa resize, không
   phải bằng chứng đây là dataset giả — nhưng vẫn không giải quyết được vấn đề
   license nào áp dụng.
6. Có ít nhất 1 bản mirror Kaggle KHÁC của cùng "RxHandBD" bởi người dùng thứ 3
   (`fareswaleed`) — tên/mô tả này đang lan truyền qua nhiều bản re-upload không
   rõ nguồn.

## Khuyến nghị bắt buộc trước khi công bố

1. **Không tái phân phối ảnh gốc** (không host lại, không đính kèm vào bất kỳ
   bản công bố code/data nào của dự án này) — vì có 3 tuyên bố license mâu thuẫn
   nhau (MIT / CC BY 4.0 / Open Database License) chồng lên nhau trong cùng 1
   lượt tải, không thể khẳng định chắc chắn quyền tái phân phối.
2. Trong **Data Availability statement**: nêu rõ ảnh huấn luyện lấy từ URL
   Kaggle cụ thể (kèm ngày truy cập), rằng trang Kaggle này để trống mục
   Author/Citation, rằng nó liên quan nhưng KHÔNG xác minh được là giống hệt
   bản Zenodo/Mendeley (2 bản này tự mâu thuẫn nhau), và rằng license chính xác
   áp dụng cho các file đã dùng KHÔNG xác định được chắc chắn.
3. Code, script, số liệu tổng hợp, output dự đoán trên tập test — vẫn công bố
   bình thường dưới license riêng của dự án (không phải nội dung có bản quyền
   của ảnh).
4. Cân nhắc **hạ vai trò của RxHandBD** trong bài báo (từ "benchmark chính"
   xuống "tập dữ liệu bổ sung/kiểm tra độ mạnh") và làm nổi bật Kaggle-BD (78
   lớp) — dataset có bài báo bình duyệt thật, license rõ ràng (Open Database
   License) — như nguồn dữ liệu chính, đặc biệt vì kết quả có ý nghĩa thống kê
   chính lại nằm trên chính RxHandBD, nên đây là một đánh đổi cần cân nhắc kỹ,
   không thể giải quyết đơn giản.
5. Nếu còn thời gian: liên hệ trực tiếp tác giả Zenodo/Mendeley (Md. Masudul
   Islam) hoặc người upload Kaggle (`abrahametry`) để hỏi license nào là chính
   thức và bản Kaggle có được phép hay không.

Nguồn đầy đủ: xem log workflow, các URL Kaggle/Zenodo/Mendeley/Scholar đã kiểm
tra trực tiếp qua trình duyệt (không chỉ qua tóm tắt tự động).
