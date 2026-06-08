# Hướng dẫn chạy OmniVoice Web UI trên Google Colab, dùng cache model trên Google Drive, và chạy từ fork riêng

Tài liệu này hướng dẫn 2 workflow:

1. **Workflow tối ưu cho Web UI trên Google Colab**: cài từ PyPI, lưu cache model vào Google Drive để lần đầu tải model, các lần sau dùng lại, chạy với `--no-asr` để nhẹ hơn.
2. **Workflow chạy Google Colab từ fork hoặc branch của bạn**: clone đúng branch từ fork, cài editable, kiểm tra Colab đang dùng code của bạn thay vì bản PyPI.

Mục tiêu là:
- Code tạm thời nằm trong `/content`
- Model nằm bền vững trong Google Drive
- Lần chạy đầu tải model về Drive
- Những lần chạy sau không phải tải lại nếu cache và thư mục model còn nguyên

> Ghi chú: luồng tối ưu bên dưới **không bật ASR mặc định**. Bạn sẽ tự nhập `ref_text` trong Web UI. ASR, ví dụ Whisper, chỉ là lựa chọn thêm và không nằm trong flow tối ưu này.

---

## 1. Điều kiện cần

### Phần cứng khuyên dùng

- Google Colab có **GPU**
- Nên chọn runtime có CUDA
- Có tài khoản Google Drive đủ chỗ để lưu model cache

### Cách bật GPU trong Colab

Vào:
- **Runtime**
- **Change runtime type**
- Chọn **GPU**

### Cell kiểm tra GPU

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

Nếu `cuda available: False` thì đừng chạy Web UI ngay, hãy xem phần troubleshooting ở cuối tài liệu.

---

## 2. Workflow A, Web UI tối ưu trên Colab với cache model trong Drive và `--no-asr`

## Ý tưởng của workflow này

- Cài `omnivoice` từ PyPI cho nhanh
- Mount Google Drive
- Chỉ định cache Hugging Face vào Drive
- Predownload model vào Drive một lần
- Chạy `omnivoice-demo` với `--model <Drive path> --device cuda:0 --no-asr --share`
- Trong Web UI, bạn **tự nhập `ref_text`** thay vì để ASR tự chép lời

### Khi nào nên dùng workflow này

Dùng khi bạn chỉ muốn:
- mở Web UI nhanh
- giữ model qua nhiều lần chạy Colab
- không cần sửa code trong repo
- không muốn gánh thêm phần ASR trong flow mặc định

---

## 3. Cell 1, mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sau khi mount, bạn có thể dùng một thư mục riêng để chứa cache và model, ví dụ:

- `/content/drive/MyDrive/omnivoice-cache`
- `/content/drive/MyDrive/omnivoice-models`

---

## 4. Cell 2, khai báo biến môi trường cache

Cell này giúp Hugging Face và Transformers lưu dữ liệu vào Drive.

```python
import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
OMNIVOICE_CACHE = DRIVE_ROOT / "omnivoice-cache"
OMNIVOICE_MODELS = DRIVE_ROOT / "omnivoice-models"

OMNIVOICE_CACHE.mkdir(parents=True, exist_ok=True)
OMNIVOICE_MODELS.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(OMNIVOICE_CACHE / "hf-home")
os.environ["HF_HUB_CACHE"] = str(OMNIVOICE_CACHE / "hf-hub")
os.environ["TRANSFORMERS_CACHE"] = str(OMNIVOICE_CACHE / "transformers-cache")

print("HF_HOME =", os.environ["HF_HOME"])
print("HF_HUB_CACHE =", os.environ["HF_HUB_CACHE"])
print("TRANSFORMERS_CACHE =", os.environ["TRANSFORMERS_CACHE"])
print("OMNIVOICE_MODELS =", OMNIVOICE_MODELS)
```

### Ghi chú tương thích

- `HF_HOME` và `HF_HUB_CACHE` là biến chính cho hệ sinh thái Hugging Face hiện nay.
- `TRANSFORMERS_CACHE` vẫn nên giữ để tương thích với một số luồng cũ hoặc dependency cũ.
- Nếu bạn đã mount Drive và giữ nguyên các thư mục này, những lần chạy sau thường sẽ không cần tải lại model.

---

## 5. Cell 3, cài OmniVoice từ PyPI

```bash
!pip install -U pip
!pip install omnivoice
```

Nếu mạng tới Hugging Face chập chờn, bạn có thể thêm mirror trước khi tải model:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("HF_ENDPOINT =", os.environ["HF_ENDPOINT"])
```

> Chỉ dùng mirror khi bạn thực sự cần. Nếu kết nối Hugging Face bình thường thì có thể bỏ qua.

---

## 6. Cell 4, predownload model vào Google Drive

Cell này tải trước các model cần dùng cho flow `--no-asr`.

```python
import os
from huggingface_hub import snapshot_download

base_dir = "/content/drive/MyDrive/omnivoice-models"
omnivoice_dir = os.path.join(base_dir, "OmniVoice")
tokenizer_dir = os.path.join(base_dir, "higgs-audio-v2-tokenizer")

snapshot_download(
    repo_id="k2-fsa/OmniVoice",
    local_dir=omnivoice_dir,
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="eustlb/higgs-audio-v2-tokenizer",
    local_dir=tokenizer_dir,
    local_dir_use_symlinks=False,
)

print("Done")
print("OmniVoice model:", omnivoice_dir)
print("Tokenizer:", tokenizer_dir)
```

### Lần đầu chạy sẽ xảy ra gì

- Colab sẽ tải model từ Hugging Face về cache trong Drive
- Đồng thời tạo thư mục model mà bạn chỉ định trong Drive
- Bước này có thể mất thời gian tùy mạng và trạng thái máy chủ

### Những lần chạy sau sẽ ra sao

Nếu các thư mục sau vẫn còn trong Drive:
- `omnivoice-cache/...`
- `omnivoice-models/OmniVoice`
- `omnivoice-models/higgs-audio-v2-tokenizer`

thì thông thường Colab sẽ dùng lại dữ liệu đã có và nhanh hơn nhiều.

> Không nêu kích thước model cố định vì nội dung repo hoặc artifact có thể thay đổi theo thời gian.

---

## 7. Cell 5, chạy Web UI với `--no-asr`

Đây là lệnh tối ưu cho Web UI khi bạn muốn tránh phần ASR mặc định.

```bash
!omnivoice-demo --model /content/drive/MyDrive/omnivoice-models/OmniVoice --device cuda:0 --no-asr --share
```

### Ý nghĩa các tham số

- `--model /content/drive/MyDrive/omnivoice-models/OmniVoice`: dùng model đã lưu trong Drive
- `--device cuda:0`: chạy bằng GPU đầu tiên
- `--no-asr`: tắt nhận dạng lời nói tự động
- `--share`: tạo link public của Gradio để mở giao diện

Sau khi chạy, Colab sẽ in ra một link Gradio. Mở link đó để vào Web UI.

---

## 8. Cách dùng Web UI trong flow `--no-asr`

Vì bạn chạy với `--no-asr`, Web UI sẽ không tự chép lời từ audio tham chiếu. Bạn cần tự nhập phần `ref_text`.

### Cách dùng cơ bản

1. Upload hoặc ghi âm `ref_audio`
2. Nhập **đúng nội dung lời nói** của đoạn `ref_audio` vào ô `ref_text`
3. Nhập câu muốn tổng hợp ở ô text đầu ra
4. Bấm generate

### Mẹo để kết quả ổn định hơn

- Dùng audio tham chiếu ngắn, rõ tiếng
- `ref_text` nên khớp sát với lời thật trong audio tham chiếu
- Nếu phát âm lạ, kiểm tra lại dấu câu, số, tên riêng
- Nếu clone giọng khác ngôn ngữ, vẫn nên nhập `ref_text` thật chính xác

### Vì sao flow này nhanh hơn

- Không cần tải và chạy thành phần ASR trong flow mặc định
- Ít phụ thuộc hơn, ít điểm lỗi hơn
- Phù hợp khi bạn đã biết nội dung của audio tham chiếu

---

## 9. Lối chạy nhanh cho các lần sau

Khi bạn đã cài xong và model đã nằm trong Drive, những lần mở Colab sau thường chỉ cần chạy lại các cell sau:

### Cell A, mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell B, set lại env cache

```python
import os

os.environ["HF_HOME"] = "/content/drive/MyDrive/omnivoice-cache/hf-home"
os.environ["HF_HUB_CACHE"] = "/content/drive/MyDrive/omnivoice-cache/hf-hub"
os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/omnivoice-cache/transformers-cache"

print("Cache env restored")
```

### Cell C, cài package

```bash
!pip install omnivoice
```

### Cell D, chạy UI

```bash
!omnivoice-demo --model /content/drive/MyDrive/omnivoice-models/OmniVoice --device cuda:0 --no-asr --share
```

Nếu Colab instance mới hoàn toàn, bạn vẫn phải cài lại package trong môi trường hiện tại. Nhưng model không cần tải lại nếu đã có trong Drive.

---

## 10. Workflow B, chạy Google Colab từ fork hoặc branch của bạn

Workflow này dành cho trường hợp bạn muốn:
- thử code mới trong fork của mình
- chạy một branch cụ thể
- sửa giao diện, sửa logic, hoặc test patch chưa có trên PyPI

### Nguyên tắc nên giữ

- **Code repo** để trong `/content/OmniVoice`
- **Model và cache** để trong Google Drive
- Không nhét cả repo vào Drive nếu bạn chỉ cần giữ model bền vững

---

## 11. Cell 1, mount Drive và set cache như cũ

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
OMNIVOICE_CACHE = DRIVE_ROOT / "omnivoice-cache"
OMNIVOICE_MODELS = DRIVE_ROOT / "omnivoice-models"

OMNIVOICE_CACHE.mkdir(parents=True, exist_ok=True)
OMNIVOICE_MODELS.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(OMNIVOICE_CACHE / "hf-home")
os.environ["HF_HUB_CACHE"] = str(OMNIVOICE_CACHE / "hf-hub")
os.environ["TRANSFORMERS_CACHE"] = str(OMNIVOICE_CACHE / "transformers-cache")
```

---

## 12. Cell 2, clone đúng branch từ fork của bạn

Thay `<YOUR_USERNAME>` và `<branch>` bằng giá trị thật.

```bash
!git clone -b <branch> https://github.com/<YOUR_USERNAME>/OmniVoice.git /content/OmniVoice
```

Ví dụ:

```bash
!git clone -b my-feature https://github.com/yourname/OmniVoice.git /content/OmniVoice
```

Nếu branch đó chưa tồn tại trên remote, hãy push branch lên fork của bạn trước.

---

## 13. Cell 3, cài editable từ source trong fork

```bash
%cd /content/OmniVoice
!pip install -U pip
!pip install -e .
```

Lệnh này giúp Python import trực tiếp từ source code trong `/content/OmniVoice`.

---

## 14. Cell 4, kiểm tra Colab đang dùng đúng code của bạn

```python
import omnivoice
print(omnivoice.__file__)
```

Bạn nên thấy đường dẫn trỏ về `/content/OmniVoice/...`

Nếu nó trỏ về thư mục `site-packages` của môi trường Python, nghĩa là Colab chưa dùng bản source bạn vừa clone như mong muốn.

---

## 15. Cell 5, nếu cần thì predownload model vào Drive

Nếu bạn đã từng tải model trước đó rồi thì thường có thể bỏ qua cell này.

```python
import os
from huggingface_hub import snapshot_download

base_dir = "/content/drive/MyDrive/omnivoice-models"
omnivoice_dir = os.path.join(base_dir, "OmniVoice")
tokenizer_dir = os.path.join(base_dir, "higgs-audio-v2-tokenizer")

snapshot_download(
    repo_id="k2-fsa/OmniVoice",
    local_dir=omnivoice_dir,
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="eustlb/higgs-audio-v2-tokenizer",
    local_dir=tokenizer_dir,
    local_dir_use_symlinks=False,
)

print("Done")
```

---

## 16. Cell 6, chạy Web UI từ code trong fork

```bash
%cd /content/OmniVoice
!omnivoice-demo --model /content/drive/MyDrive/omnivoice-models/OmniVoice --device cuda:0 --no-asr --share
```

Nếu branch của bạn có sửa CLI hoặc Web UI, Colab sẽ dùng đúng code trong fork, miễn là `pip install -e .` đã chạy thành công.

---

## 17. Cập nhật code fork trong Colab

Nếu bạn mở lại cùng notebook hoặc muốn kéo update mới từ fork:

```bash
%cd /content/OmniVoice
!git pull
```

Nếu bạn vừa đổi branch:

```bash
%cd /content/OmniVoice
!git fetch --all
!git checkout <branch>
!git pull
```

Sau khi update code, nếu cần chắc ăn, bạn có thể chạy lại:

```bash
%cd /content/OmniVoice
!pip install -e .
```

Rồi kiểm tra lại:

```python
import omnivoice
print(omnivoice.__file__)
```

---

## 18. Tổ chức thư mục khuyên dùng

### Nên làm

- Giữ **code** ở `/content/OmniVoice`
- Giữ **cache** ở `/content/drive/MyDrive/omnivoice-cache`
- Giữ **model** ở `/content/drive/MyDrive/omnivoice-models`

### Không nên làm

- Không phụ thuộc vào code trong `/content` để lưu lâu dài, vì Colab reset là mất
- Không trộn lung tung giữa thư mục code và thư mục model
- Không coi ASR là mặc định trong flow tối ưu này

---

## 19. Troubleshooting

## 19.1. Không thấy GPU hoặc `cuda:0` lỗi

Triệu chứng:
- `torch.cuda.is_available()` trả về `False`
- Chạy `--device cuda:0` bị lỗi

Cách xử lý:
- Vào lại **Runtime > Change runtime type** và chọn GPU
- Restart runtime rồi chạy lại từ đầu
- Kiểm tra lại bằng cell GPU ở đầu tài liệu
- Nếu Colab không cấp GPU cho phiên hiện tại, hãy chờ rồi thử lại

Nếu vẫn không có GPU, bạn có thể thử chạy bằng CPU, nhưng sẽ chậm hơn nhiều. Tài liệu này không tối ưu cho CPU.

---

## 19.2. Web UI không lên link share hoặc bị treo khi start

Cách xử lý:
- Chạy lại cell launch
- Kiểm tra runtime còn GPU và RAM không
- Nếu vừa sửa code fork, chạy lại `pip install -e .`
- Đảm bảo đường dẫn `--model` là đúng và còn tồn tại trong Drive

---

## 19.3. Lỗi liên quan đến `ffmpeg`

Một số flow audio có thể cần `ffmpeg`. Nếu môi trường Colab báo thiếu, cài bằng:

```bash
!apt-get update
!apt-get install -y ffmpeg
```

Sau đó chạy lại cell liên quan.

---

## 19.4. Hugging Face tải chậm hoặc không tải được

Bạn có thể thử mirror:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

Rồi chạy lại bước download hoặc launch.

Nếu mạng bình thường trở lại, bạn có thể bỏ biến này đi.

---

## 19.5. Repo private hoặc model private

Nếu bạn cần truy cập repo hoặc artifact private trên Hugging Face, hãy login trước:

```python
from huggingface_hub import login
login()
```

Hoặc dùng token môi trường nếu bạn đã có cách quản lý token riêng.

Sau đó chạy lại bước `snapshot_download(...)`.

---

## 19.6. Colab đang import nhầm bản PyPI thay vì code từ fork

Triệu chứng:
- `print(omnivoice.__file__)` không trỏ về `/content/OmniVoice`

Cách xử lý:

```bash
%cd /content/OmniVoice
!pip install -e .
```

Sau đó kiểm tra lại:

```python
import omnivoice
print(omnivoice.__file__)
```

Nếu notebook đã import package từ trước, bạn có thể cần **Runtime restart** rồi chạy lại các cell theo thứ tự.

---

## 19.7. `ref_text` nhập sai nên giọng clone bị kém

Vì flow này dùng `--no-asr`, chất lượng sẽ phụ thuộc một phần vào độ chính xác của `ref_text`.

Cách xử lý:
- Nghe lại audio tham chiếu
- Sửa `ref_text` cho khớp sát câu nói thật
- Dùng audio ngắn, rõ, ít tạp âm

---

## 20. Mẫu notebook ngắn gọn nhất cho flow tối ưu

Nếu bạn chỉ cần bản tối giản để copy nhanh, dùng chuỗi cell sau.

### Cell 1

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2

```python
import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
OMNIVOICE_CACHE = DRIVE_ROOT / "omnivoice-cache"
OMNIVOICE_MODELS = DRIVE_ROOT / "omnivoice-models"

OMNIVOICE_CACHE.mkdir(parents=True, exist_ok=True)
OMNIVOICE_MODELS.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(OMNIVOICE_CACHE / "hf-home")
os.environ["HF_HUB_CACHE"] = str(OMNIVOICE_CACHE / "hf-hub")
os.environ["TRANSFORMERS_CACHE"] = str(OMNIVOICE_CACHE / "transformers-cache")
```

### Cell 3

```bash
!pip install -U pip
!pip install omnivoice
```

### Cell 4

```python
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="k2-fsa/OmniVoice",
    local_dir="/content/drive/MyDrive/omnivoice-models/OmniVoice",
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="eustlb/higgs-audio-v2-tokenizer",
    local_dir="/content/drive/MyDrive/omnivoice-models/higgs-audio-v2-tokenizer",
    local_dir_use_symlinks=False,
)
```

### Cell 5

```bash
!omnivoice-demo --model /content/drive/MyDrive/omnivoice-models/OmniVoice --device cuda:0 --no-asr --share
```

---

## 21. Tóm tắt ngắn

- Muốn chạy nhanh và ổn định trên Colab, hãy cài từ PyPI, mount Drive, lưu cache model vào Drive, rồi chạy `omnivoice-demo` với `--no-asr`.
- Trong flow `--no-asr`, bạn phải tự nhập `ref_text` trong Web UI.
- Lần đầu sẽ tải model. Những lần sau thường chỉ cần mount Drive, set env, cài package, rồi launch lại UI.
- Nếu bạn cần test code riêng, clone fork bằng `git clone -b <branch> https://github.com/<YOUR_USERNAME>/OmniVoice.git`, chạy `pip install -e .`, rồi kiểm tra `omnivoice.__file__` để chắc là Colab đang dùng đúng source của bạn.
- Hãy giữ code trong `/content` và giữ model trong Drive.
