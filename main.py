import math
import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def find_relevant(query_embedding, db_records, top_k=2):
    scores = []
    for record in db_records:
        doc_id, doc_content, doc_emb = record
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((doc_id, doc_content, score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]

def main():
    # 1. Initialize the SDK
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 2. Load the embedding model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # 3. VERİTABANINDAN BELGELERİ ÇEKME
    print("Loading documents from SQLite database...")
    db_records = []
    conn = sqlite3.connect("rag_database.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, content, embedding FROM documents")
        for row in cursor.fetchall():
            db_records.append((row[0], row[1], json.loads(row[2])))
        print(f"Loaded {len(db_records)} documents from database.")
    except sqlite3.OperationalError:
        print("HATA: rag_database.db bulunamadı! Önce setup_db.py dosyasını çalıştırın.")
        return
    finally:
        conn.close()

    # 4. Load the chat model
    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded successfully. Ready for questions.")
    print('Type "quit" to exit.\n')

    # 5. İnteraktif Sorgu Döngüsü
    SIMILARITY_THRESHOLD = 0.68  # Güvenlik eşik değeri

    while True:
        query = input("Question: ").strip()
        if not query or query.lower() == "quit":
            break

        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding
        results = find_relevant(query_embedding, db_records, top_k=2)
        
        top_score = results[0][2] if results else 0.0

        # KORUMA FİLTRESİ: Eğer en yüksek benzerlik skoru barajın altındaysa LLM'i hiç yorma
        if top_score < SIMILARITY_THRESHOLD:
            print("Answer: Bu konuda veritabanımda yeterli bilgi bulunmamaktadır.\n")
            continue

        context = "\n".join(f"- {record[1]}" for record in results)

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