# Risk Profiler: Alternative Credit Scoring Engine

## Deskripsi Proyek
Risk Profiler adalah infrastruktur Back-End B2B SaaS yang dirancang untuk memitigasi risiko kredit pada segmen UMKM yang belum terjangkau perbankan (unbanked) di Indonesia. Sistem ini berfungsi sebagai mesin analitik yang mengonversi jejak digital alternatif—seperti transaksi QRIS, riwayat pembayaran utilitas, dan aktivitas e-commerce—menjadi metrik profil risiko yang terukur dan transparan.

Proyek ini dikembangkan dalam rangka PIDI DIGDAYA X HACKATHON 2026 untuk mendukung inklusi ekonomi nasional dengan menyediakan alat ukur risiko yang objektif bagi lembaga keuangan seperti BPR, Bank Syariah, dan Koperasi.

## Fitur Utama
* **Cashflow DNA Analysis**: Mengevaluasi stabilitas pendapatan dan kedisiplinan operasional melalui data transaksi QRIS.
* **Alternative Data Integration**: Menarik data pembayaran listrik (PLN), air (PDAM), dan riwayat toko e-commerce sebagai indikator karakter nasabah.
* **Explainable AI (XAI)**: Menggunakan metode SHAP untuk memberikan transparansi pada setiap keputusan skor, sehingga dapat diaudit oleh Risk Officer.
* **Agnostic Scoring Model**: Menghasilkan Probability of Default (PD) yang dapat diadaptasi baik untuk sistem perbankan konvensional maupun sistem bagi hasil syariah.
* **Hybrid Validation**: Mendukung integrasi SLIK OJK sebagai variabel konfirmasi tambahan tanpa menjadikannya ketergantungan utama.

## Arsitektur Teknologi
* **Bahasa Pemrograman**: Python
* **Kerangka Kerja API**: FastAPI (RESTful API dengan format JSON)
* **Machine Learning**: Scikit-learn (Random Forest dan Logistic Regression)
* **Analisis Transparansi**: SHAP (Explainable AI)
* **Basis Data**: PostgreSQL / Supabase
* **Keamanan**: OAuth 2.0 dan Enkripsi AES-256

## Struktur Direktori
    risk-profiler/
    ├── api/                # Logika Backend dan Endpoint FastAPI
    ├── dashboard/          # Antarmuka Visualisasi (Streamlit)
    ├── data/               # Dataset (Raw dan Processed)
    ├── models/             # Model Machine Learning (Pickle files)
    ├── notebooks/          # Eksperimen Data Cleaning dan Training
    ├── .gitignore          # File dan folder yang diabaikan oleh Git
    ├── README.md           # Dokumentasi utama proyek
    └── requirements.txt    # Daftar dependensi library

## Instalasi dan Penggunaan
1. **Kloning repositori**:
   git clone https://github.com/username/risk-profiler.git
   cd risk-profiler

2. **Instalasi dependensi**:
   pip install -r requirements.txt

3. **Menjalankan API**:
   uvicorn api.main:app --reload

4. **Menjalankan Dashboard**:
   streamlit run dashboard/app.py

## Identitas Tim
* **Aditya Cakti Chandrasa**: Project Manager
* **Nabil Muhammad Hilmi**: Lead Engineer
* **Zahran Muhammad Syahbana Fardiaz**: Machine Learning & Data Pipeline
* **Muhammad Ghazi Ali Asy'ary**: Risk & Actuary

## Lisensi
Proyek ini dikembangkan secara khusus untuk kompetisi Digdaya x Hackathon 2026. Seluruh hak kekayaan intelektual mengikuti ketentuan panitia penyelenggara.
