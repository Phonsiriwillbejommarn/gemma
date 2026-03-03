import os
from huggingface_hub import snapshot_download

# --- ตั้งค่าตรงนี้ ---
REPO_ID = "Phonsiri/gemma-2-2b-Distillation-gemma-3-27b-it"
CHECKPOINT_NAME = "last-checkpoint"
# --------------------

print(f"📥 กำลังดาวน์โหลด {CHECKPOINT_NAME} จาก Hugging Face...")

# 1. โหลดข้อมูล Checkpoint เฉพาะโฟลเดอร์ที่ต้องการลงมาไว้ที่ ./distill_output
try:
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{CHECKPOINT_NAME}/*"],
        local_dir="./distill_output",
        local_dir_use_symlinks=False
    )
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการดาวน์โหลด: {e}")
    exit(1)

print("\n✅ ดาวน์โหลดเสร็จสมบูรณ์! เตรียมตัวรัน Distillation ต่อ...")

# 2. สั่งรันสคริปต์หลักพร้อมโหมด Auto-resume ทันที!
# มันจะเจอโฟลเดอร์ที่เพิ่งโหลดมา แล้วรันต่อยอดทันที
exit_code = os.system("python distill_gemma.py --config distill_config.yaml --resume_from_checkpoint auto")

if exit_code == 0:
    print("✅ การฝึกฝนเสร็จสมบูรณ์!")
else:
    print("❌ เกิดข้อผิดพลาดระหว่างการทำ Distillation (Exit Code: {})".format(exit_code))
