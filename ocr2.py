import cv2
import numpy as np
import pytesseract
import re
import sqlite3


# =========================================================
# 1. KONFIGURASI TESSERACT
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# 2. BACA GAMBAR
# =========================================================

gambar_asli = cv2.imread("contoh.jpg")

if gambar_asli is None:
    print("ERROR: Gambar 'contoh.jpg' tidak ditemukan!")
    exit()



# =========================================================
# 3. GRAYSCALE
# =========================================================

gambar_gray = cv2.cvtColor(
    gambar_asli,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# 4. THRESHOLDING
# =========================================================

nilai_threshold, gambar_biner = cv2.threshold(
    gambar_gray,
    127,
    255,
    cv2.THRESH_BINARY
)


# =========================================================
# 5. OCR
# =========================================================
# IMPORTANT:
# OCR menggunakan gambar_biner, BUKAN thinning.

teks_ocr = pytesseract.image_to_string(
    gambar_biner,
    config="--psm 6"
)

print("\n========================================")
print("HASIL OCR MENTAH")
print("========================================")
print(teks_ocr)


# =========================================================
# 6. BERSIHKAN HASIL OCR
# =========================================================

def bersihkan_teks(teks):
    """
    Membersihkan hasil OCR tanpa mengubah
    isi data secara berlebihan.
    """

    # Hilangkan karakter yang tidak diperlukan
    teks = teks.replace("|", " ")
    teks = teks.replace("—", ":")
    teks = teks.replace("–", ":")
    teks = teks.replace("»", ":")
    teks = teks.replace(">", ":")
    teks = teks.replace("‘", "")
    teks = teks.replace("’", "")
    teks = teks.replace("“", "")
    teks = teks.replace("”", "")

    # Hilangkan spasi berlebihan
    teks = re.sub(r'[ \t]+', ' ', teks)

    # Hilangkan baris kosong
    baris = []

    for line in teks.splitlines():
        line = line.strip()

        if line:
            baris.append(line)

    return baris


baris_ocr = bersihkan_teks(teks_ocr)


# =========================================================
# 7. NORMALISASI LABEL KTP
# =========================================================

def normalisasi_label(baris):
    """
    Mengubah typo OCR pada LABEL menjadi
    label KTP yang benar.
    """

    # Daftar label dan kemungkinan typo OCR
    pola_label = {
        "NIK": [
            r"^MIK\b",
            r"^NIK\b",
            r"^NIK\s*[:.]"
        ],

        "Nama": [
            r"^fama\b",
            r"^Nama\b"
        ],

        "Tempat/Tanggal Lahir": [
            r"^Tempautg\)?\s*Lahir\b",
            r"^Tempat.*Lahir\b",
            r"^Tempat/Tgl Lahir\b",
            r"^Tempat/Tanggal Lahir\b"
        ],

        "Jenis Kelamin": [
            r"^Janis\s*Kolamin\b",
            r"^Jenis\s*Kelamin\b"
        ],

        "Alamat": [
            r"^Afamat\b",
            r"^Alamat\b"
        ],

        "RT/RW": [
            r"^RTIRYE\b",
            r"^RT/RW\b",
            r"^RT.*RW\b"
        ],

        "Kel/Desa": [
            r"^KoliBosa\b",
            r"^Kel.*Desa\b",
            r"^Kelurahan\b",
            r"^Desa\b"
        ],

        "Kecamatan": [
            r"^Kecamaian\b",
            r"^Kecamatan\b"
        ],

        "Agama": [
            r"^Agama\b"
        ],

        "Status Perkawinan": [
            r"^Stains\s*Porkawinan\b",
            r"^Status\s*Perkawinan\b"
        ],

        "Pekerjaan": [
            r"^Poksrjaan\b",
            r"^Pekerjaan\b"
        ],

        "Kewarganegaraan": [
            r"^Kewarganegaraan\b"
        ],

        "Berlaku Hingga": [
            r"^Berlaky\s*Hingga\b",
            r"^Berlaku\s*Hingga\b"
        ]
    }

    hasil = []

    for line in baris:

        label_ditemukan = None
        isi = line

        for label, pola_list in pola_label.items():

            for pola in pola_list:

                match = re.search(
                    pola,
                    line,
                    re.IGNORECASE
                )

                if match:
                    label_ditemukan = label

                    # Ambil teks setelah label
                    isi = line[match.end():].strip()

                    # Bersihkan tanda pemisah
                    isi = re.sub(
                        r"^[\s:.\-]+",
                        "",
                        isi
                    )

                    break

            if label_ditemukan:
                break

        if label_ditemukan:
            hasil.append(
                (label_ditemukan, isi)
            )
        else:
            hasil.append(
                (None, line)
            )

    return hasil


baris_normal = normalisasi_label(baris_ocr)


# =========================================================
# 8. EKSTRAKSI DATA
# =========================================================

data_ktp = {
    "NIK": "",
    "Nama": "",
    "Tempat/Tanggal Lahir": "",
    "Jenis Kelamin": "",
    "Golongan Darah": "",
    "Alamat": "",
    "RT/RW": "",
    "Kel/Desa": "",
    "Kecamatan": "",
    "Agama": "",
    "Status Perkawinan": "",
    "Pekerjaan": "",
    "Kewarganegaraan": "",
    "Berlaku Hingga": ""
}


# ---------------------------------------------------------
# Masukkan hasil berdasarkan label
# ---------------------------------------------------------

for label, isi in baris_normal:

    if label is None:
        continue

    isi = isi.strip()

    if not isi:
        continue

    # Kalau field belum memiliki data
    if data_ktp[label] == "":
        data_ktp[label] = isi

    # Kalau ternyata OCR memecah field menjadi
    # beberapa baris, gabungkan
    else:
        data_ktp[label] += " " + isi


# =========================================================
# 9. CARI NIK KALAU LABEL NIK TIDAK TERBACA
# =========================================================

if data_ktp["NIK"] == "":

    # Cari angka 16 digit di seluruh hasil OCR
    match_nik = re.search(
        r"\b\d{16}\b",
        teks_ocr
    )

    if match_nik:
        data_ktp["NIK"] = match_nik.group()


# =========================================================
# 10. BERSIHKAN NIK
# =========================================================

if data_ktp["NIK"]:

    # Ambil hanya angka
    nik = re.sub(
        r"\D",
        "",
        data_ktp["NIK"]
    )

    # NIK Indonesia seharusnya 16 digit
    if len(nik) == 16:
        data_ktp["NIK"] = nik


# =========================================================
# 11. EKSTRAKSI GOLONGAN DARAH
# =========================================================

if data_ktp["Golongan Darah"] == "":

    match_darah = re.search(
        r"(?:Darah|Gol\s*[,.:]?\s*Darah)\s*[:.]?\s*(AB|A|B|O)\b",
        teks_ocr,
        re.IGNORECASE
    )

    if match_darah:
        data_ktp["Golongan Darah"] = (
            match_darah.group(1).upper()
        )


# =========================================================
# 12. BERSIHKAN BEBERAPA FIELD
# =========================================================

for key in data_ktp:

    data_ktp[key] = data_ktp[key].strip()

    # Hilangkan karakter aneh di awal/akhir
    data_ktp[key] = re.sub(
        r"^[^A-Za-z0-9]+",
        "",
        data_ktp[key]
    )

    data_ktp[key] = re.sub(
        r"[^A-Za-z0-9/.,\- ]+$",
        "",
        data_ktp[key]
    )

# =========================================================
# 12. PISAHKAN GOLONGAN DARAH DARI JENIS KELAMIN
# =========================================================

if data_ktp["Jenis Kelamin"]:

    # Hapus bagian "Gol, Darah : AB" dari Jenis Kelamin
    data_ktp["Jenis Kelamin"] = re.sub(
        r"\s*Gol\s*[,.:]?\s*Darah\s*[:.]?\s*(AB|A|B|O)\b",
        "",
        data_ktp["Jenis Kelamin"],
        flags=re.IGNORECASE
    ).strip()

# =========================================================
# 13. TAMPILKAN HASIL AKHIR
# =========================================================

print("\n========================================")
print("DATA KTP HASIL EKSTRAKSI")
print("========================================")

for key, value in data_ktp.items():

    if value == "":
        value = "Tidak terbaca"

    print(f"{key:<25}: {value}")

# =========================================================
# 14. SIMPAN DATA KE SQLITE
# =========================================================

# Membuka / membuat database
koneksi = sqlite3.connect("ktp.db")

cursor = koneksi.cursor()


# Membuat tabel jika belum ada
cursor.execute("""
CREATE TABLE IF NOT EXISTS data_ktp (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nik TEXT,
    nama TEXT,
    tempat_tanggal_lahir TEXT,
    jenis_kelamin TEXT,
    golongan_darah TEXT,
    alamat TEXT,
    rt_rw TEXT,
    kel_desa TEXT,
    kecamatan TEXT,
    agama TEXT,
    status_perkawinan TEXT,
    pekerjaan TEXT,
    kewarganegaraan TEXT,
    berlaku_hingga TEXT
)
""")


# Memasukkan data hasil OCR
cursor.execute("""
INSERT INTO data_ktp (
    nik,
    nama,
    tempat_tanggal_lahir,
    jenis_kelamin,
    golongan_darah,
    alamat,
    rt_rw,
    kel_desa,
    kecamatan,
    agama,
    status_perkawinan,
    pekerjaan,
    kewarganegaraan,
    berlaku_hingga
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    data_ktp["NIK"],
    data_ktp["Nama"],
    data_ktp["Tempat/Tanggal Lahir"],
    data_ktp["Jenis Kelamin"],
    data_ktp["Golongan Darah"],
    data_ktp["Alamat"],
    data_ktp["RT/RW"],
    data_ktp["Kel/Desa"],
    data_ktp["Kecamatan"],
    data_ktp["Agama"],
    data_ktp["Status Perkawinan"],
    data_ktp["Pekerjaan"],
    data_ktp["Kewarganegaraan"],
    data_ktp["Berlaku Hingga"]
))


# Simpan perubahan
koneksi.commit()

print("\nData berhasil disimpan ke database SQLite!")


# Tutup koneksi
koneksi.close()

# =========================================================
# 14. TAMPILKAN GAMBAR
# =========================================================

cv2.imshow(
    "1. Grayscale",
    gambar_gray
)

cv2.imshow(
    "2. Thresholding",
    gambar_biner
)

cv2.waitKey(0)
cv2.destroyAllWindows()