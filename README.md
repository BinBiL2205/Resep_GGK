# Sistem Rekomendasi Menu Diet Gagal Ginjal Kronik (GGK) dengan IndoT5

Aplikasi ini merupakan bagian dari penelitian skripsi yang bertujuan untuk membantu penderita Gagal Ginjal Kronik (GGK) dalam menentukan menu diet yang aman dan sesuai dengan kebutuhan nutrisi, menggunakan model arsitektur **IndoT5**.

## 🚀 Demo Aplikasi
Aplikasi ini telah dideploy dan dapat diakses secara langsung melalui Hugging Face Spaces:
👉 **(https://huggingface.co/spaces/Binbil/resep_GGK)** 

## 🛠️ Fitur Utama
*   **Rekomendasi Menu Personal:** Menghasilkan saran menu berdasarkan input kondisi pasien.
*   **Validasi Nutrisi:** Memastikan bahan makanan sesuai dengan standar diet GGK (rendah protein, kalium, dan fosfor).
*   **Model IndoT5:** Menggunakan model Transformer bahasa Indonesia yang telah di-finetuning untuk domain nutrisi.

## 📁 Struktur Repositori
*   `app.py`: File utama aplikasi menggunakan Streamlit.
*   `requirements.txt`: Daftar library Python yang diperlukan.
*   `DATASET_INDOT5_MASTER_FINAL.csv`: Dataset pendukung untuk referensi bahan makanan.
*   `.gitignore`: Mengonfigurasi file besar (seperti model) agar tidak diunggah ke GitHub.

> **Catatan Penting:** 
> Folder model `indot5_ggk_memorization/` tidak diunggah ke repositori GitHub ini karena ukurannya yang sangat besar (>1GB). Jika Anda ingin menjalankan aplikasi ini secara lokal, Anda harus memiliki file bobot model tersebut di folder yang sama.

## 💻 Cara Menjalankan Secara Lokal
1. **Clone repositori ini:**
   ```bash
   git clone (https://github.com/BinBiL2205/Resep_GGK)
