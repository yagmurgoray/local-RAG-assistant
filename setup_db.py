"""
Yerel RAG (Retrieval-Augmented Generation) Sistemi Veritabani Kurulum Dosyasi.

Bu betik (script), Microsoft Foundry Local SDK kullanarak belirlenen teknik metinleri
vektorlestirir (embedding) ve benzerlik aramasi yapilabilmesi icin SQLite veritabanina kaydeder.
"""

import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

from config import APP_NAME, DB_NAME, EMBEDDING_MODEL_NAME


def main() -> None:
    """
    Ana kurulum fonksiyonu.
    1. Foundry Local mimarisini ve yerel gomme (embedding) modelini baslatir.
    2. SQLite veritabani baglantisini kurar ve gerekli 'documents' tablosunu olusturur.
    3. Kaynak metinleri vektorlere donusturerek JSON formatinda veritabanina yazar.
    """
    # --- BOLUM 1: FOUNDRY LOCAL BASLATMA ---
    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Qwen embedding modelini yukluyoruz (SDK en iyi donanimi otomatik secer)
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)

    embedding_model.download(lambda p: print(f"\rDownloading model: {p:.1f}%", end="", flush=True))
    print()

    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # RAG sisteminin temelini olusturacak olan bilgi havuzu (Knowledge Base)
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

    # --- BOLUM 2: SQLITE BAGLANTISI VE TABLO KURULUMU ---
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Mevcut tablo yoksa olustur, varsa icindeki eski verileri temizle
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding TEXT
        )
    ''')
    cursor.execute("DELETE FROM documents")

    # --- BOLUM 3: VEKTORE CEVIRME VE KAYDETME ---
    print("Embedding documents and saving to database...")
    response = embedding_client.generate_embeddings(documents)

    for i, item in enumerate(response.data):
        doc_text = documents[i]
        embedding_json = json.dumps(item.embedding)  # Vektor dizisini JSON metnine cevirir

        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)",
            (doc_text, embedding_json)
        )

    # Degisiklikleri kaydet ve kaynaklari serbest birak
    conn.commit()
    conn.close()
    embedding_model.unload()
    print("Success! Documents embedded and saved to SQLite.")


if __name__ == "__main__":
    main()
