# Crack Detector: DeepCrack + ConvNeXt-Small

ตรวจจับรอยร้าวในภาพทีละ pixel (segmentation) ใช้ได้กับถนน คอนกรีต ผนัง และพื้นผิวทั่วไป
โมเดลถูกเทรนด้วย augmentation หนัก จึงทนต่อภาพเบลอ, noise, ภาพ JPEG คุณภาพต่ำ, แสงไม่ดี, เงา และภาพขาวดำ

## ไฟล์

| ไฟล์ | คืออะไร |
|---|---|
| `crack_detector.py` | โค้ดทั้งหมดอยู่ในไฟล์เดียว: โมเดล, ตัวช่วยใช้งาน (`CrackDetector`) และ command line |
| `deepcrack_convnext_small_aug.pt` | weight ของโมเดล (208 MB) ต้องอยู่โฟลเดอร์เดียวกับ `crack_detector.py` หรือระบุ path เอง |
| `requirements.txt` | library ที่ต้องใช้ ได้แก่ torch, torchvision, opencv-python และ numpy |

คัดลอกทั้งโฟลเดอร์ไปไว้ในโปรเจ็คอื่นได้เลย ไม่ต้องใช้โค้ดเทรน (ซึ่งไม่ได้รวมอยู่ใน repository นี้)

## ใช้ใน Python

```python
from crack_detector import CrackDetector

det = CrackDetector()                    # โหลด weight ข้างไฟล์, ใช้ GPU ถ้ามี (หรือ device="cpu")

mask = det.predict_mask("road.jpg")      # numpy uint8 HxW: 255 = รอยร้าว, 0 = ไม่ใช่
prob = det.predict("road.jpg")           # numpy float32 HxW: ความน่าจะเป็น 0-1

# รับภาพได้หลายแบบ
mask = det.predict_mask(frame)                 # numpy BGR จาก cv2.imread / cv2.VideoCapture (ค่าเริ่มต้น)
mask = det.predict_mask(rgb_array, color="rgb")  # numpy RGB
mask = det.predict_mask(pil_image)             # PIL.Image
# ภาพขาวดำ (HxW), BGRA/RGBA, uint16 และ float 0-1 ก็รับได้

# ภาพ 16-bit จากกล้อง line-scan (Mono12)
det = CrackDetector()                    # PNG 16-bit ที่บันทึกแบบเลื่อนบิตแล้ว (ค่าสูงสุด ~65520) ใช้ค่าเริ่มต้นได้เลย
det12 = CrackDetector(input_bits=12)     # buffer ดิบจากกล้องที่มีค่า 0-4095

# หลายภาพพร้อมกัน (ภาพขนาดเท่ากันจะรันเป็น batch บน GPU ซึ่งเร็วกว่า)
probs = det.predict_batch([img1, img2, img3])
masks = [(p > det.threshold).astype("uint8") * 255 for p in probs]

# วาดผลทับภาพ (สีแดง)
vis = CrackDetector.overlay(bgr_image, mask)

# ร้อยละพื้นที่รอยร้าว
crack_percent = (mask > 0).mean() * 100
```

### ตัวเลือกของ `CrackDetector(...)`

| ตัวเลือก | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `weights` | ไฟล์ .pt ข้างโค้ด | path ของ weight |
| `device` | `"cuda"` ถ้ามี, ไม่งั้น `"cpu"` | อุปกรณ์ที่ใช้รัน |
| `threshold` | `0.16` | ค่าความน่าจะเป็นขั้นต่ำที่นับเป็นรอยร้าว ลดลงจะเจอรอยร้าวมากขึ้น (เส้นหนาขึ้น และเจอเส้นจางได้มากขึ้น) แต่ก็ทำนายผิดมากขึ้นด้วย |
| `amp` | `True` | ใช้ bf16 บน GPU ซึ่งเป็นค่าที่ใช้วัดความแม่นในตาราง และเร็วกว่า |
| `input_bits` | `16` | จำนวนบิตของภาพ uint16: 16 = ใช้ช่วง 0–65535 เต็ม, 12 = Mono12 ดิบ (0–4095) |
| `tile` | `1024` | ภาพที่ใหญ่กว่า 1024×1024 จะถูกแบ่งเป็นชิ้นที่ซ้อนทับกัน ทำให้ใช้หน่วยความจำ GPU ไม่เกินราว 2 GB ใส่ `0` เพื่อรันทั้งภาพในครั้งเดียว |

### ตัวเลือก `scale` (ขนาดของรอยร้าวในภาพ)

โมเดลเทรนจากภาพ 448×448 ที่ถ่ายใกล้รอยร้าว
ถ้าภาพของคุณใหญ่มาก เช่นภาพมือถือ 4000×3000 ที่ถ่ายถนนทั้งเลน รอยร้าวอาจดูกว้างหรือแคบต่างจากตอนเทรนมาก
ให้ลองย่อภาพก่อน ผลที่ได้จะถูกขยายกลับเป็นขนาดเดิมให้อัตโนมัติ

```python
mask = det.predict_mask(big_photo, scale=0.5)    # ลองค่า 0.25-0.5 กับภาพความละเอียดสูง
```

วิธีเลือกค่า: ลองกับภาพจริงสักไม่กี่ภาพแล้วดูภาพ overlay ว่าค่าไหนให้เส้นรอยร้าวตรงและครบที่สุด

## ใช้ผ่าน command line

```bash
python crack_detector.py photos/ --out results/ --overlay          # ทั้งโฟลเดอร์
python crack_detector.py a.jpg b.png --out results/ --prob         # ระบุไฟล์ และบันทึกภาพความน่าจะเป็นด้วย
python crack_detector.py photos/ --out results/ --scale 0.5 --thr 0.3 --cpu
```

ไฟล์ที่ได้ในโฟลเดอร์ผลลัพธ์:
- `<ชื่อ>_mask.png`: mask ขาวดำ ขาว = รอยร้าว
- `<ชื่อ>_overlay.jpg`: ภาพเดิมที่ระบายรอยร้าวเป็นสีแดง (เมื่อใส่ `--overlay`)
- `<ชื่อ>_prob.png`: ภาพความน่าจะเป็น ×255 (เมื่อใส่ `--prob`)

ชื่อไฟล์ภาษาไทยใช้ได้

## ความแม่นและความเร็ว

วัดบน test ของ crack_merged (1,695 ภาพ, 448×448) ที่ threshold 0.16

| วัดอะไร | ค่า |
|---|---|
| F-score (ยอมให้คลาดได้ 2 pixel) | 0.877 |
| IoU (ตรงทุก pixel, threshold 0.5) | 0.618 |
| AP | 0.918 |
| F-score เฉลี่ยบนภาพเสีย 10 แบบ (มืด, สว่างเกิน, เงา, noise, เบลอ, JPEG ฯลฯ) | 0.870 |
| F-score ในกรณีที่แย่ที่สุด (ภาพเบลอจากการเคลื่อนไหว) | 0.856 |

ความเร็วบน RTX PRO 4000 Blackwell สำหรับภาพ 448×448 (bf16):
- ส่งทีละภาพ: ~100 ภาพ/วินาที
- ส่งทีละ 8 ภาพ: ~155 ภาพ/วินาที

ภาพ 1920×1080 ใช้หน่วยความจำ GPU ราว 1.7 GB บน CPU ทำงานได้แต่ช้ากว่ามาก

## ข้อควรรู้

- **ข้อมูลที่ใช้เทรน:** crack_merged (CRACK500, Rissbilder, Volker, DeepCrack, GAPS384, CFD และอื่นๆ) ชุด train+val รวม 11,503 ภาพ ภาพที่ต่างจากชุดนี้มาก เช่นพื้นผิวที่ไม่เคยเห็น ควรทดสอบก่อนใช้งานจริง
- **ความกว้างของรอยร้าวที่ทำนาย:** แต่ละแหล่งข้อมูลวาด label ไม่เหมือนกัน บางแหล่งวาดแค่เส้นกลางกว้าง 1 pixel บางแหล่งวาดเต็มความกว้างของรอยร้าว โมเดลเรียนรู้ค่ากลางๆ ถ้าต้องวัดความกว้างรอยร้าวให้แม่นยำ ควรปรับ `threshold` โดยเทียบกับภาพของคุณเอง
- **รูปแบบไฟล์ weight:** `torch.load(..., weights_only=True)` ให้ dict ที่มี `state_dict`, `threshold`, `test_metrics` และข้อมูลการเทรน ดูได้จาก `det.info`
- **ที่มา:** weight มาจาก checkpoint สุดท้าย (`last.pt`) ของ run `deepcrack_convnext_small_final_aug` ใช้ EMA weights เทรนด้วย preset `convnext_small_final_aug` ของโค้ดเทรน ซึ่งไม่ได้รวมอยู่ใน repository นี้
