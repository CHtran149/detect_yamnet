-Đầu tiên vào trang chủ Respeaker rồi làm theo hướng dẫn, link dùng link bên dưới do link gốc ko được(đã thử cài và sửa nhưng link gốc vẫn ko chạy):


GitHub - respeaker/seeed-voicecard: 2 Mic Hat, 4 Mic Array, 6-Mic Circular Array Kit, and 4-Mic Linear Array Kit for Raspberry Pi. file cài respeaker.

Kiểm tra xem là cài được chưa. Nếu thấy có dòng seeed-4mic-voicecard là được.


--> Tiếp cài phần mềm cho nó: GitHub - respeaker/mic_array: DOA, VAD and KWS for ReSpeaker Microphone Array.
Sau đo chui vò thuc muc chua file vad_doa là thấy file, rồi quay lại ra cmd chạy file vad-doa trên đó rồi nó yêu cầu cài gì cài theo có ảnh( cài bằng cmp luôn), nhớ dùng pip3 vì pip 0 là cài cho python 2.
-Cài sudo_ gì đó để nó chạy được pip3 install trước rồi cho chạy file vad_doa nó yêu cầu gì làm theo(Lỗi queue thì đổi thành q thường ở các file lỗi).

Link chữa lỗi sai hướng: https://github.com/respeaker/seeed-voicecard/issues/309
Sai khi copy file sửa vào thì lại phải chạy lại file install của file seeed-voicard. Dùng leenh cd vào file seedvoicecard rồi lại sudo ./install.sh để cài lại
Rồi lạy vào file mic_array rồi chạy lênh python3 vad_doa.py để chạy lại xem hoạt động ko.

--> Tiếp là cài yamnet để nó nhận diện được tiếng link cài:
GitHub - x1001000/sed-yamnet-raspberrypi: Sound Event Detection with YAMNet+tw on Raspberry Pi
Bên dưới nó có hướng dẫn trong git thì làm theo.

Các Project mà Project này phụ thuộc:
https://github.com/nhatthanh12/seeed-voicecard-thanhdonhat
https://github.com/nhatthanh12/mic_array_thanhdonhat
https://github.com/nhatthanh12/sed-yamnet-raspberrypi_thanhdonhat
