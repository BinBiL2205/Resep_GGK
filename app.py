import streamlit as st
import pandas as pd
import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Diet GGK - Hybrid RAG Pro", layout="centered")

# --- DAFTAR BAHAN BERBAHAYA (SAFETY FIRST) ---
BLACKLIST_BAHAN = [
    'sarden', 'kornet', 'sosis', 'nugget', 'ikan asin', 'terasi',
    'pete', 'petai', 'jengkol', 'jeroan', 'bayam', 'belimbing', 
    'lamb leg', 'udang', 'cumi', 'kepiting', 'kerang', 'tongkol', 
    'msg', 'saori', 'masako', 'royco'
]

# --- 1. LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    # Sesuaikan path model Anda
    path_model = "C:/Users/BINTANG_NABIL/Desktop/pakardiabetes/resep_ggk/indot5_ggk_memorization"
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_data
def load_dataset():
    # Pastikan pandas tidak memotong teks panjang
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv("DATASET_INDOT5_MASTER_FINAL.csv")
    # Memastikan kolom dibaca sebagai string utuh
    df['steps_cleaned'] = df['steps_cleaned'].astype(str)
    df['ingredients_final'] = df['ingredients_final'].astype(str)
    return df

tokenizer, model, device = load_model()
df_ref = load_dataset()

# --- 2. FUNGSI VALIDASI & KEAMANAN ---
def cek_bahan_berbahaya(user_input):
    """Mendeteksi bahan terlarang dalam input user."""
    input_clean = user_input.lower()
    # Membersihkan karakter non-alfabet untuk pengecekan kata yang akurat
    words = re.findall(r'\b\w+\b', input_clean)
    bahan_terdeteksi = [b for b in BLACKLIST_BAHAN if b in words]
    return (True, list(set(bahan_terdeteksi))) if bahan_terdeteksi else (False, [])

def validasi_output_generator(resep_teks):
    """Mendeteksi halusinasi/degenerasi teks pada output IndoT5."""
    if len(resep_teks) < 50: # Menaikkan batas minimal agar resep lebih lengkap
        return False
    
    words = resep_teks.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35: # Ambang batas repetisi
            return False
            
    kata_kunci = ['masak', 'campur', 'sajikan', 'panaskan', 'rebus', 'iris', 'siapkan', 'aduk']
    if not any(kata in resep_teks.lower() for kata in kata_kunci):
        return False
    return True

# --- 3. FUNGSI GENERATOR (IndoT5) ---
def hasilkan_resep_ggk(user_input):
    input_text = "resep ggk: " + user_input
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    outputs = model.generate(
        inputs, 
        max_new_tokens=1024,     # Menjamin output panjang (sampai 1024 token baru)
        min_length=100,          # Memaksa model memberikan instruksi detail
        do_sample=True, 
        top_k=50, 
        top_p=0.92, 
        temperature=0.6, 
        no_repeat_ngram_size=3, 
        repetition_penalty=2.5,  # Penalti repetisi tinggi untuk cegah halusinasi
        early_stopping=False
    )
    
    resep_raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if validasi_output_generator(resep_raw):
        return resep_raw
    else:
        return ("MAAF: Sistem tidak dapat menghasilkan instruksi masak yang cukup akurat untuk kombinasi bahan tersebut. "
                "Silakan gunakan bahan utama lain yang lebih umum.")

# --- 4. LOGIKA HYBRID RAG PRO ---
def hybrid_rag_system_pro(user_input, threshold=0.5):
    # A. Filter Keamanan Medis (Prioritas 1)
    is_bahaya, daftar_bahaya = cek_bahan_berbahaya(user_input)
    if is_bahaya:
        detail = ", ".join(daftar_bahaya)
        pesan = (f"⚠️ **PERINGATAN KEAMANAN**\n\nInput Anda mengandung bahan berisiko tinggi bagi pasien GGK: **{detail}**.\n\n"
                 "Sistem menolak memberikan resep demi keselamatan kesehatan Anda. Mohon pilih bahan yang lebih aman.")
        return pesan, "Keamanan Medis", 0.0

    # B. Persiapan Retrieval
    bahan_utama_user = user_input.split(',')[0].strip().lower()
    vectorizer = TfidfVectorizer()
    all_docs = df_ref['ingredients_final'].fillna('').tolist() + [user_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    max_score = cosine_sim.max()
    best_index = cosine_sim.argmax()
    best_recipe_ingredients = df_ref.iloc[best_index]['ingredients_final'].lower()
    
    # C. Jalur Keputusan (RAG vs Generator)
    if max_score >= threshold and bahan_utama_user in best_recipe_ingredients:
        source = "RAG (Data Asli Terverifikasi)"
        # Menggunakan .at untuk menjamin string utuh tanpa truncation pandas
        resep = df_ref.at[best_index, 'steps_cleaned']
        return resep, source, max_score
    else:
        source = "AI Generator (IndoT5)"
        resep = hasilkan_resep_ggk(user_input)
        return resep, source, max_score

# --- 5. TAMPILAN ANTARMUKA (UI) ---
st.title("🍳 Generator Resep Diet GGK")
st.markdown("Sistem Pakar Diet berbasis **Hybrid RAG** untuk pasien Ginjal Kronis.")

with st.expander("ℹ️ Petunjuk Penggunaan"):
    st.write("1. Masukkan minimal 7 bahan makanan dipisahkan koma.")
    st.write("2. Bahan pertama dianggap sebagai bahan utama.")
    st.write("3. Hindari bahan olahan, tinggi natrium, dan tinggi kalium.")

user_input = st.text_area("Masukkan Bahan Makanan:", placeholder="Contoh: ayam, jeruk nipis, bawang putih, ...", height=150)

if st.button("Hasilkan Instruksi Masak"):
    if user_input:
        with st.spinner('Memverifikasi keamanan dan menganalisis resep...'):
            resep, sumber, skor = hybrid_rag_system_pro(user_input)
            
            if sumber == "Keamanan Medis":
                st.error(resep)
            else:
                st.info(f"Metode: **{sumber}** | Confidence Score: **{skor:.2f}**")
                st.subheader("Langkah Memasak:")
                
                if "MAAF:" in resep:
                    st.warning(resep)
                else:
                    # Menampilkan resep utuh (berapapun panjangnya)
                    st.success(resep)
            
            st.warning("⚠️ *Selalu konsultasikan porsi dan takaran bumbu seperti garam dan kaldu bubuk dengan ahli gizi Anda.*")
    else:
        st.error("Silakan masukkan daftar bahan terlebih dahulu.")