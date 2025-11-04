import kagglehub
import shutil
from pathlib import Path

# Veri setini indir
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Dataset indirildi:", path)

# Hedef klasör: data/raw
target_dir = Path("data/raw")
target_dir.mkdir(parents=True, exist_ok=True)

# İndirilen dosyayı data/raw içine taşı
for file in Path(path).glob("*.csv"):
    shutil.copy(file, target_dir / file.name)
    print(f"Taşındı: {file.name}")

print("✅ Veri dosyası hazır: data/raw klasöründe")
