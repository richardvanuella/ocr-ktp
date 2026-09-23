import cv2
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================================================
# 1. BACA GAMBAR
# =========================================================

gambar_asli = cv2.imread("contoh.jpg")

if gambar_asli is None:
    print("Gambar tidak ditemukan!")
    exit()

# =========================================================
# 2. GRAYSCALE
# =========================================================

gambar_gray = cv2.cvtColor(
    gambar_asli,
    cv2.COLOR_BGR2GRAY
)

# =========================================================
# 3. THRESHOLDING
# =========================================================

nilai_threshold, gambar_biner = cv2.threshold(
    gambar_gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# =========================================================
# 4. THINNING / SKELETONIZATION
# =========================================================

# Balik gambar
gambar_inv = cv2.bitwise_not(gambar_biner)

# Matriks kosong
skeleton = np.zeros(
    gambar_inv.shape,
    dtype=np.uint8
)

# Kernel
kernel = cv2.getStructuringElement(
    cv2.MORPH_CROSS,
    (3, 3)
)

temp_img = gambar_inv.copy()

while cv2.countNonZero(temp_img) > 0:

    # Erosi
    eroded = cv2.erode(
        temp_img,
        kernel
    )

    # Dilasi
    temp = cv2.dilate(
        eroded,
        kernel
    )

    # Ambil bagian yang hilang
    temp = cv2.subtract(
        temp_img,
        temp
    )

    # Gabungkan ke skeleton
    skeleton = cv2.bitwise_or(
        skeleton,
        temp
    )

    # Iterasi berikutnya
    temp_img = eroded.copy()

# Balik kembali
gambar_thinned = cv2.bitwise_not(skeleton)


# =========================================================
# 5. OCR
# =========================================================

# OCR menggunakan hasil thresholding
# Untuk awal kita gunakan thresholding,
# karena thinning belum tentu lebih bagus untuk OCR.

teks = pytesseract.image_to_string(
    gambar_biner,
    config="--psm 6"
)

print("\n==============================")
print("HASIL OCR")
print("==============================")
print(teks)


# =========================================================
# 6. EKSTRAKSI DATA KTP
# =========================================================

data_ktp = {
    "NIK": "",
    "Nama": "",
    "Tempat/Tanggal Lahir": "",
    "Jenis Kelamin": "",
    "Alamat": "",
    "RT/RW": "",
    "Kel/Desa": "",
    "Kecamatan": "",
    "Agama": "",
    "Status Perkawinan": "",
    "Pekerjaan": "",
    "Kewarganegaraan": ""
}


# ---------------------------------------------------------
# NIK
# ---------------------------------------------------------

match = re.search(
    r'NIK\s*[:.]?\s*(\d{16})',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["NIK"] = match.group(1)


# ---------------------------------------------------------
# NAMA
# ---------------------------------------------------------

match = re.search(
    r'Nama\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Nama"] = match.group(1).strip()


# ---------------------------------------------------------
# TEMPAT / TANGGAL LAHIR
# ---------------------------------------------------------

match = re.search(
    r'(?:Tempat/Tgl Lahir|Tempat/Tanggal Lahir|Tempat.*Lahir)\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Tempat/Tanggal Lahir"] = match.group(1).strip()


# ---------------------------------------------------------
# JENIS KELAMIN
# ---------------------------------------------------------

match = re.search(
    r'Jenis Kelamin\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Jenis Kelamin"] = match.group(1).strip()


# ---------------------------------------------------------
# ALAMAT
# ---------------------------------------------------------

match = re.search(
    r'Alamat\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Alamat"] = match.group(1).strip()


# ---------------------------------------------------------
# RT/RW
# ---------------------------------------------------------

match = re.search(
    r'RT/RW\s*[:.]?\s*(\d{1,3}\s*/\s*\d{1,3})',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["RT/RW"] = match.group(1)


# ---------------------------------------------------------
# KELURAHAN / DESA
# ---------------------------------------------------------

match = re.search(
    r'(?:Kel/Desa|Kelurahan|Desa)\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Kel/Desa"] = match.group(1).strip()


# ---------------------------------------------------------
# KECAMATAN
# ---------------------------------------------------------

match = re.search(
    r'Kecamatan\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Kecamatan"] = match.group(1).strip()


# ---------------------------------------------------------
# AGAMA
# ---------------------------------------------------------

match = re.search(
    r'Agama\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Agama"] = match.group(1).strip()


# ---------------------------------------------------------
# STATUS PERKAWINAN
# ---------------------------------------------------------

match = re.search(
    r'Status Perkawinan\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Status Perkawinan"] = match.group(1).strip()


# ---------------------------------------------------------
# PEKERJAAN
# ---------------------------------------------------------

match = re.search(
    r'Pekerjaan\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Pekerjaan"] = match.group(1).strip()


# ---------------------------------------------------------
# KEWARGANEGARAAN
# ---------------------------------------------------------

match = re.search(
    r'(?:Kewarganegaraan|Kewarganegaraan)\s*[:.]?\s*(.+)',
    teks,
    re.IGNORECASE
)

if match:
    data_ktp["Kewarganegaraan"] = match.group(1).strip()


# =========================================================
# 7. TAMPILKAN HASIL
# =========================================================

print("\n==============================")
print("DATA KTP")
print("==============================")

for key, value in data_ktp.items():
    print(f"{key:<25}: {value}")


# =========================================================
# 8. TAMPILKAN GAMBAR
# =========================================================

cv2.imshow(
    "1. Grayscale",
    gambar_gray
)

cv2.imshow(
    "2. Thresholding",
    gambar_biner
)

cv2.imshow(
    "3. Thinning",
    gambar_thinned
)

cv2.waitKey(0)
cv2.destroyAllWindows()