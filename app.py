"""
Yerel RAG (Retrieval-Augmented Generation) Sistemi Ana Arayüz Dosyası.

Bu uygulama, Streamlit kullanarak kullanıcı dostu bir sohbet arayüzü sunar. 
Kullanıcıdan alınan sorular, Qwen embedding modeli ile vektörleştirilir, 
SQLite veritabanında anlamsal (semantic) arama yapılır ve bulunan bağlam (context)
katı bir sistem istemiyle (strict system prompt) Qwen sohbet modeline iletilerek yanıt üretilir.
"""

import math
import sqlite3
import json
from typing import List, Tuple, Any

import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager

# --- SAYFA YAPILANDIRMASI (PAGE CONFIGURATION) ---
st.set_page_config(page_title="Local RAG Assistant", page_icon="✨", layout="wide")
st.title("✨ Local RAG AI Assistant")
st.caption("Microsoft Foundry Local SDK & SQLite Tabanlı Çevrimdışı Soru-Cevap Sistemi")

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.header("🎯 Doğrudan Bilgi Testi (Direct Fact Checking)")
    st.caption("Sistemimizin doğruluğunu test etmek için aşağıdaki soruları kullanabilirsiniz:")
    
    st.markdown("""
    * What programming languages does the SDK support?
    * How does Foundry Local run models?
    
    """)

    st.header("🧠 Kavramsal Anlayış Testi (Conceptual Understanding)")
    st.markdown("""
        * What do embedding models convert text into?
        * What is vector similarity search?
        """)
    
    st.header("🛡️ Güvenlik ve Halüsinasyon Testi (Strict QA Test - Edge Cases)")
    st.markdown("""
            * How much does a Microsoft Foundry subscription cost?
            * Who is the CEO of OpenAI?
            """)


    
    st.info("⚠️ Sistemimiz sıfır halüsinasyon prensibiyle çalışır. Kapsam dışı bir soru sorarsanız model uydurma yapmaz, bilgi eksikliği uyarısı verir.")
    st.divider() # Araya şık bir çizgi çeker
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun() # Sayfayı yenileyerek ekranı tertemiz yapar

# --- YARDIMCI FONKSİYONLAR (UTILITY FUNCTIONS) ---
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """İki vektör arasındaki kosinüs benzerliğini (cosine similarity) hesaplar."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def find_relevant(query_embedding: List[float], db_records: List[Tuple], top_k: int = 2) -> List[Tuple]:
    """Kullanıcı sorgusuna en yakın anlamsal eşleşmeyi sağlayan veritabanı kayıtlarını bulur."""
    scores = []
    for record in db_records:
        doc_id, doc_content, doc_emb = record
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((doc_id, doc_content, score))
    
    # Skorlara göre büyükten küçüğe sırala ve en iyi 'top_k' sonucu döndür
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]

# --- MODEL VE VERİTABANI YÜKLEME (ÖNBELLEKLENMİŞ - CACHED) ---
@st.cache_resource
def load_rag_pipeline() -> Tuple[Any, Any]:
    """Foundry Local mimarisini başlatır ve embedding ile sohbet modellerini RAM'e yükler."""
    config = Configuration(app_name="foundry_local_rag")
    
    # Singleton hatasını önlemek için koruma bloğu
    try:
        FoundryLocalManager.initialize(config)
    except Exception as e:
        if "already been initialized" not in str(e):
            raise e # Eğer başka bir hataysa ekrana yansıt
            
    manager = FoundryLocalManager.instance

    # Embedding Modeli
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Chat (Sohbet) Modeli
    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embedding_client, chat_client

@st.cache_data
def load_database() -> List[Tuple]:
    """SQLite veritabanına bağlanarak tüm kaynak metinleri ve vektörleri belleğe alır."""
    db_records = []
    conn = sqlite3.connect("rag_database.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, content, embedding FROM documents")
        for row in cursor.fetchall():
            db_records.append((row[0], row[1], json.loads(row[2])))
    finally:
        conn.close()
    return db_records

# --- ARKA PLAN (BACKEND) BAŞLATMA ---
with st.spinner("Modeller ve Veritabanı Yükleniyor..."):
    try:
        embedding_client, chat_client = load_rag_pipeline()
        db_records = load_database()
        st.success("Sistem Hazır!", icon="✅")
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()

# --- SOHBET ARAYÜZÜ (CHAT INTERFACE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana yazdır
for message in st.session_state.messages:
    # Mesajı yazanın rolüne göre doğru avatarı seç
    secilen_avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    
    with st.chat_message(message["role"], avatar=secilen_avatar):
        st.markdown(message["content"])

# Kullanıcı Girişi
# Kullanıcı Girişi
if prompt := st.chat_input("Bir soru sorun..."):
    # Kullanıcı mesajını arayüze ve oturum geçmişine ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):  # Yazılımcı emojisi ekledik
        st.markdown(prompt)

    # RAG İşlemi & Model Çıkarımı (Inference)
    with st.chat_message("assistant", avatar="🤖"): # Robot emojisi ekledik
        # ... (altındaki kodlar aynı kalacak)
        # 1. Sorguyu Vektörleştir (Embed Query)
        query_response = embedding_client.generate_embedding(prompt)
        query_embedding = query_response.data[0].embedding

        # 2. Vektör Araması (Vector Search)
        results = find_relevant(query_embedding, db_records, top_k=2)
        top_score = results[0][2] if results else 0.0 # En yüksek benzerlik skoru

        # 3. YAZILIM SEVİYESİNDE GÜVENLİK FİLTRESİ (ALTIN DENGE)
        SIMILARITY_THRESHOLD = 0.68

        if top_score < SIMILARITY_THRESHOLD:
            full_response = "Bu konuda veritabanımda yeterli bilgi bulunmamaktadır."
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Benzerlik skoru yeterli ise bağlamı oluştur
            context = "\n".join(f"- {record[1]}" for record in results)

            # Bulunan bağlamı arayüzde göster
            with st.expander("🔍 Veritabanından Getirilen Bağlam (Context)"):
                st.write(f"**Eşleşme Skoru:** %{top_score*100:.1f}")
                st.write(context)

            # 4. LLM ÇIKARIMI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise QA assistant. Answer the user's question strictly using ONLY the provided Context. "
                        "If the answer is NOT explicitly stated in the Context, respond EXACTLY with: "
                        "'Bu konuda veritabanımda yeterli bilgi bulunmamaktadır.'"
                    ),
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}"
                }
            ]

            # Yanıtı ekrana akıt (Streaming)
            response_placeholder = st.empty()
            full_response = ""

           
            for chunk in chat_client.complete_streaming_chat(messages):
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                        
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
