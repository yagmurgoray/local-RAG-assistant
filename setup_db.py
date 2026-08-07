"""
Yerel RAG (Retrieval-Augmented Generation) Sistemi Veritabanı Kurulum Dosyası.

Bu betik (script), Microsoft Foundry Local SDK kullanarak belirlenen teknik metinleri 
vektörleştirir (embedding) ve benzerlik araması yapılabilmesi için SQLite veritabanına kaydeder.
"""

import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

# --- SABİT DEĞİŞKENLER (CONSTANTS) ---
DB_NAME = "rag_database.db"
EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"

def main() -> None:
    """
    Ana kurulum fonksiyonu. 
    1. Foundry Local mimarisini ve yerel gömme (embedding) modelini başlatır.
    2. SQLite veritabanı bağlantısını kurar ve gerekli 'documents' tablosunu oluşturur.
    3. Kaynak metinleri vektörlere dönüştürerek JSON formatında veritabanına yazar.
    """
    # --- BÖLÜM 1: FOUNDRY LOCAL BAŞLATMA ---
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Qwen embedding modelini yüklüyoruz (SDK en iyi donanımı otomatik seçer)
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)
    
    embedding_model.download(lambda p: print(f"\rDownloading model: {p:.1f}%", end="", flush=True))
    print()
    
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # RAG sisteminin temelini oluşturacak olan bilgi havuzu (Knowledge Base)
    documents = [
        "Foundry Local runs AI models directly on your device without cloud connectivity.",
        "The Foundry Local SDK supports Python, C#, JavaScript, and Rust.",
        "Embedding models convert text into numerical vectors for similarity search.",
        "Foundry Local uses ONNX Runtime for efficient model inference on CPUs and GPUs.",
        "The model catalog provides pre-optimized models that you can download and run locally.",
        "Retrieval-augmented generation grounds model responses in your own data.",
        "Vector similarity search finds documents that are semantically close to a query.",
        "Chat completions generate natural language responses from a prompt and context.",
    ]

    # --- BÖLÜM 2: SQLITE BAĞLANTISI VE TABLO KURULUMU ---
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Mevcut tablo yoksa oluştur, varsa içindeki eski verileri temizle
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding TEXT
        )
    ''')
    cursor.execute("DELETE FROM documents")

    # --- BÖLÜM 3: VEKTÖRE ÇEVİRME VE KAYDETME ---
    print("Embedding documents and saving to database...")
    response = embedding_client.generate_embeddings(documents)
    
    for i, item in enumerate(response.data):
        doc_text = documents[i]
        embedding_json = json.dumps(item.embedding) # Vektör dizisini JSON metnine çevirir
        
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)", 
            (doc_text, embedding_json)
        )

    # Değişiklikleri kaydet ve kaynakları serbest bırak
    conn.commit()
    conn.close()
    embedding_model.unload()
    print("Success! Documents embedded and saved to SQLite.")

if __name__ == "__main__":
    main()