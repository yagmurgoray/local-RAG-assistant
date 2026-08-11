"""
Yerel RAG (Retrieval-Augmented Generation) Sistemi Ana Arayuz Dosyasi.

Bu uygulama, Streamlit kullanarak kullanici dostu bir sohbet arayuzu sunar.
Kullanicidan alinan sorular, Qwen embedding modeli ile vektorlestirilir,
SQLite veritabaninda anlamsal (semantic) arama yapilir ve bulunan baglam (context)
kati bir sistem istemiyle (strict system prompt) Qwen sohbet modeline iletilerek yanit uretilir.
"""

import sqlite3
import json
from typing import List, Tuple, Any

import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager

from config import (
    APP_NAME,
    DB_NAME,
    EMBEDDING_MODEL_NAME,
    CHAT_MODEL_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    FALLBACK_MESSAGE,
)
from rag_utils import find_relevant

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
    st.divider()  # Araya şık bir çizgi çeker
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()  # Sayfayı yenileyerek ekranı tertemiz yapar

# --- MODEL VE VERİTABANI YÜKLEME (ÖNBELLEKLENMİŞ - CACHED) ---
@st.cache_resource
def load_rag_pipeline() -> Tuple[Any, Any]:
    """Foundry Local mimarisini başlatır ve embedding ile sohbet modellerini RAM'e yükler."""
    config = Configuration(app_name=APP_NAME)

    # Singleton hatasını önlemek için koruma bloğu
    try:
        FoundryLocalManager.initialize(config)
    except Exception as e:
        if "already been initialized" not in str(e):
            raise e  # Eğer başka bir hataysa ekrana yansıt

    manager = FoundryLocalManager.instance

    # Embedding Modeli
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Chat (Sohbet) Modeli
    chat_model = manager.catalog.get_model(CHAT_MODEL_NAME)
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embedding_client, chat_client


@st.cache_data
def load_database() -> List[Tuple]:
    """SQLite veritabanına bağlanarak tüm kaynak metinleri ve vektörleri belleğe alır."""
    db_records = []
    conn = sqlite3.connect(DB_NAME)
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
if prompt := st.chat_input("Bir soru sorun..."):
    # Kullanıcı mesajını arayüze ve oturum geçmişine ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # RAG İşlemi & Model Çıkarımı (Inference)
    with st.chat_message("assistant", avatar="🤖"):
        # 1. Sorguyu Vektörleştir (Embed Query)
        query_response = embedding_client.generate_embedding(prompt)
        query_embedding = query_response.data[0].embedding

        # 2. Vektör Araması (Vector Search)
        results = find_relevant(query_embedding, db_records, top_k=TOP_K)
        top_score = results[0][2] if results else 0.0  # En yüksek benzerlik skoru

        # 3. YAZILIM SEVİYESİNDE GÜVENLİK FİLTRESİ (ALTIN DENGE)
        if top_score < SIMILARITY_THRESHOLD:
            full_response = FALLBACK_MESSAGE
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
                        f"If the answer is NOT explicitly stated in the Context, respond EXACTLY with: '{FALLBACK_MESSAGE}'"
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