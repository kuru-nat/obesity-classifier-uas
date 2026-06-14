# Obesity Health Category Classifier
**AIB02 UAS | Josua Nathanael Dharmawan | 38250005**

## Deskripsi
Sistem klasifikasi kategori berat badan menggunakan **Artificial Neural Network (MLP)** dengan dua mode prediksi:
- **Input Manual** — Form 16 fitur gaya hidup
- **Face Scan** — Estimasi BMI dari analisis geometri wajah via OpenCV

**Akurasi model: 94.09%** | Dataset: UCI Obesity Levels (2.111 data, 7 kelas, 16 fitur)

🔗 **Demo Online:** [isi link Streamlit Cloud di sini setelah deploy]

---

## Struktur File
```
project/
├── Project_UAS_Josua_Nathanael_Dharmawan_38250005.py   ← Script training
├── app.py                                               ← Aplikasi Streamlit
├── ObesityDataSet_raw_and_data_sinthetic.csv           ← Dataset
├── Laporan_UAS_Josua_Nathanael_Dharmawan_38250005.docx ← Laporan
├── requirements.txt                                     ← Daftar dependencies
├── README.md
└── model/   ← Berisi artefak hasil training (wajib ada untuk app.py)
    ├── mlp_obesity.pkl
    ├── scaler.pkl
    ├── label_encoder_target.pkl
    ├── label_encoders_features.pkl
    └── feature_cols.pkl
```

---

## Cara Menjalankan Secara Lokal

### 1. Install Library
```bash
pip install -r requirements.txt
```

### 2. Jalankan Script Training (wajib dilakukan pertama kali)
```bash
python Project_UAS_Josua_Nathanael_Dharmawan_38250005.py
```
> Proses ini akan melatih model dan menyimpan file hasil training ke folder `model/`.
> Tunggu hingga muncul pesan **[SELESAI] Project UAS berhasil dijalankan!**

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```
> Buka browser di: **http://localhost:8501**

### Catatan
- Langkah **2 wajib dijalankan terlebih dahulu** sebelum langkah 3. Jika `app.py` dijalankan tanpa training, aplikasi akan error karena file model belum tersedia.
- Untuk mode **Face Scan**, gunakan foto wajah frontal dengan pencahayaan yang cukup.

---

## Deploy Online (Streamlit Community Cloud)

Aplikasi ini sudah siap untuk dideploy secara online menggunakan **Streamlit Community Cloud** (gratis).

### 1. Persiapan Repository GitHub
Buat repository baru di GitHub, lalu upload seluruh isi folder `project/` di atas — **termasuk folder `model/` beserta 5 file `.pkl` di dalamnya**. Folder `model/` harus sudah dibuat dengan menjalankan script training di lokal terlebih dahulu, karena Streamlit Cloud tidak menjalankan script training secara otomatis.

### 2. Deploy ke Streamlit Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io) dan login menggunakan akun GitHub.
2. Klik **"New app"**.
3. Pilih repository, branch (`main`), dan file utama: `app.py`.
4. Klik **Deploy**.
5. Tunggu proses build (sekitar 1–3 menit). Setelah selesai, aplikasi dapat diakses melalui URL publik berformat `https://nama-app.streamlit.app`.

### 3. Hal yang Perlu Diperhatikan
- Pastikan `requirements.txt` menggunakan `opencv-python-headless` (bukan `opencv-python`), karena server cloud tidak memiliki library GUI yang dibutuhkan oleh versi biasa.
- Pastikan folder `model/` (file `.pkl`) ikut di-commit ke GitHub. Tanpa folder ini, aplikasi akan menampilkan peringatan "Model belum ditemukan".
- Mode **Face Scan** tetap berfungsi di cloud karena file Haarcascade sudah tersedia bawaan dari package OpenCV.

### 4. Update Aplikasi
Setiap kali ada perubahan kode yang di-push ke branch yang terhubung, Streamlit Cloud akan otomatis melakukan re-deploy.
