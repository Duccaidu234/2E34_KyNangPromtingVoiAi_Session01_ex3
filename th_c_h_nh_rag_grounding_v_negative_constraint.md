# Tài liệu đào tạo Thu mua MartX: Chống "Ảo giác AI" bằng RAG Grounding & Negative Constraint

## Phần 1: Phân biệt "Hỏi LLM không kèm tài liệu" và "RAG Grounding"

Để hiểu rõ tại sao ChatGPT lại trả lời sai quy định của công ty một cách đầy tự tin, chúng ta hãy hình dung cách AI làm việc giống như **hai hình thức thi cử**:

| Tiêu chí so sánh | Hỏi LLM không kèm tài liệu (Đóng sách) | RAG Grounding - Dán kèm tài liệu nguồn (Mở sách) |
| :--- | :--- | :--- |
| **Hình tượng ẩn dụ** | **Bài thi học thuộc lòng:** Thí sinh phải tự lục lại ký ức về những gì đã đọc trên Internet trước đây để đoán câu trả lời. | **Bài thi mở tài liệu:** Thí sinh được phát tận tay cuốn hợp đồng của MartX và chỉ được phép chép đúng thông tin trong đó. |
| **Nguồn dữ liệu AI sử dụng** | Dữ liệu chung khổng lồ trên Internet (các hợp đồng mẫu ngẫu nhiên trên mạng). AI **không hề biết** văn bản nội bộ của MartX. | Dữ liệu thực tế, chính xác tuyệt đối từ văn bản/hợp đồng nội bộ mà người dùng cung cấp trực tiếp trong câu lệnh. |
| **Cơ chế trả lời** | **Đoán xác suất từ ngữ:** AI ghép các từ thường đi cùng nhau trên mạng (ví dụ: *"giao hàng lỗi" + "phạt" $\rightarrow$ thường đi với "20%", "đình chỉ"*). | **Trích xuất & Đối chiếu:** AI đọc hiểu đoạn văn bản được cung cấp và trích xuất đúng con số, điều khoản có mặt trong văn bản đó. |
| **Rủi ro nghiệp vụ** | **Cực kỳ cao:** Dễ mắc bẫy **Ảo giác (Hallucination)** — AI nói dối một cách trôi chảy, logic nhưng sai hoàn toàn thực tế doanh nghiệp. | **Rất thấp:** Gần như loại bỏ hoàn toàn việc AI tự bịa số liệu nếu đi kèm các ràng buộc chặt chẽ. |

### Phân tích sự cố tại bộ phận Thu mua MartX:
Khi nhân viên hỏi ChatGPT mà không đính kèm hợp đồng, AI đã rơi vào **Bẫy "Bài thi học thuộc"**. Vì không có dữ liệu nội bộ của MartX, AI đã tự động lấy một mức phạt phổ biến thường thấy trong các hợp đồng thương mại trên Internet (*"phạt 20%, đình chỉ 30 ngày"*) để điền vào chỗ trống. Việc nhân viên tin tưởng tuyệt đối vào câu trả lời này mà không kiểm chứng đã dẫn đến áp dụng sai chế tài ("5% giá trị lô hàng và đổi trong 24h"), gây tranh chấp nghiêm trọng với nhà cung cấp.

---

## Phần 2: Thực hành thiết kế Prompt với RAG Grounding & Negative Constraint

Để ngăn chặn triệt để tình trạng AI tự suy đoán, nhân viên Thu mua cần áp dụng công thức kết hợp giữa **RAG Grounding** (cung cấp văn bản gốc làm neo kiến thức) và **Negative Constraint** (khóa an toàn cấm AI dùng kiến thức bên ngoài).

### Cấu trúc Prompt chuẩn mực áp dụng cho MartX:

```markdown
[VAI TRÒ]
Bạn là Trợ lý Pháp lý & Thu mua nội bộ của công ty MartX. Nhiệm vụ của bạn là tra cứu chính xác các điều khoản trong hợp đồng được cung cấp dưới đây.

[RÀNG BUỘC AN TOÀN - NEGATIVE CONSTRAINT]
1. CHỈ ĐƯỢC PHÉP sử dụng thông tin nằm trong phần [TÀI LIỆU NGUỒN] bên dưới để trả lời.
2. TUYỆT ĐỐI KHÔNG tự suy đoán, không dùng kiến thức chung từ Internet và không tự ý thêm thắt bất kỳ con số hay chế tài nào không có trong văn bản.
3. Nếu thông tin người dùng hỏi không được đề cập trong tài liệu nguồn, bắt buộc phải trả lời chính xác câu sau: "Tài liệu hiện tại không quy định vấn đề này, vui lòng kiểm tra lại phụ lục hợp đồng."

[TÀI LIỆU NGUỒN - RAG GROUNDING]
"""
Điều 5. Quy định về chất lượng hàng hóa nông sản (Trích Hợp đồng Nguyên tắc MartX - 2026):
5.1. Tỷ lệ hao hụt cho phép: Hàng hóa rau củ quả tươi sống được phép có tỷ lệ dập nát tự nhiên trong quá trình vận chuyển không vượt quá 3% tổng trọng lượng lô hàng.
5.2. Xử lý vi phạm chất lượng: Trường hợp nhà cung cấp giao hàng có tỷ lệ rau củ bị dập nát vượt quá 10% tổng trọng lượng lô hàng, MartX áp dụng mức phạt hành chính bằng 5% giá trị của riêng lô hàng bị dập nát đó, đồng thời yêu cầu nhà cung cấp phải thực hiện đổi bù đủ số lượng hàng đạt chuẩn trong vòng 24 giờ kể từ lúc lập biên bản.
"""

[CÂU HỎI CỦA NHÂN VIÊN]
Theo quy định MartX, nhà cung cấp giao rau bị dập nát trên 10% thì bị phạt bao nhiêu?
```

### Kết quả đầu ra kỳ vọng từ AI:
> Dựa trên Điều 5.2 của Hợp đồng Nguyên tắc MartX được cung cấp, nếu nhà cung cấp giao rau bị dập nát vượt quá 10% tổng trọng lượng lô hàng, mức xử lý áp dụng như sau:
> * **Mức phạt tiền:** Phạt 5% giá trị của riêng lô hàng bị dập nát đó.
> * **Biện pháp khắc phục:** Yêu cầu nhà cung cấp đổi bù đủ số lượng hàng đạt chuẩn trong vòng 24 giờ kể từ lúc lập biên bản.