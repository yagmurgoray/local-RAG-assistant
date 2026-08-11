import sqlite3
import json
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


def main():
    # 1. Initialize the SDK
    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 2. Load the embedding model
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # 3. VERITABANINDAN BELGELERI CEKME
    print("Loading documents from SQLite database...")
    db_records = []
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id, content, embedding FROM documents")
        for row in cursor.fetchall():
            db_records.append((row[0], row[1], json.loads(row[2])))
        print(f"Loaded {len(db_records)} documents from database.")
    except sqlite3.OperationalError:
        print("HATA: rag_database.db bulunamadi! Once setup_db.py dosyasini calistirin.")
        return
    finally:
        conn.close()

    # 4. Load the chat model
    chat_model = manager.catalog.get_model(CHAT_MODEL_NAME)
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded successfully. Ready for questions.")
    print('Type "quit" to exit.\n')

    # 5. Interaktif Sorgu Dongusu
    while True:
        query = input("Question: ").strip()
        if not query or query.lower() == "quit":
            break

        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding
        results = find_relevant(query_embedding, db_records, top_k=TOP_K)

        top_score = results[0][2] if results else 0.0

        # KORUMA FILTRESI: Eger en yuksek benzerlik skoru barajin altindaysa LLM'i hic yorma
        if top_score < SIMILARITY_THRESHOLD:
            print(f"Answer: {FALLBACK_MESSAGE}\n")
            continue

        context = "\n".join(f"- {record[1]}" for record in results)

        # NOT: Bu sistem promptu bilinçli olarak degistirilmedi (app.py'deki
        # "strict" versiyondan farkli). Bu ayri bir guncelleme konusu.
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "If the context doesn't contain enough information, say so.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ]

        print("Answer: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
        print("\n")

    # Clean up
    embedding_model.unload()
    chat_model.unload()
    print("Models unloaded. Done!")


if __name__ == "__main__":
    main()
