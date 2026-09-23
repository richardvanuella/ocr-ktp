import cv2
import numpy as np
import pytesseract
import re

# Lokasi Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# 1. BACA GAMBAR
# =========================================================

gambar_asli = cv2.imread("contoh.jpg")

if gambar_asli is None:
    print("ERROR: Gambar 'contoh.jpg' tidak ditemukan!")
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
# 4. THINNING
# =========================================================

# Balik gambar
gambar_inv = cv2.bitwise_not(gambar_biner)

# Buat gambar kosong untuk skeleton
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

    # Cari bagian yang hilang
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


# Balik kembali supaya teks hitam
# dan background putih
gambar_thinned = cv2.bitwise_not(skeleton)


# =========================================================
# 5. OCR MENGGUNAKAN HASIL THINNING
# =========================================================

teks = pytesseract.image_to_string(
    gambar_thinned,
    config="--psm 6"
)


# =========================================================
# 6. TAMPILKAN HASIL OCR
# =========================================================

print("\n==============================")
print("HASIL OCR DENGAN THINNING")
print("==============================")
print(teks)


# =========================================================
# 7. TAMPILKAN GAMBAR
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