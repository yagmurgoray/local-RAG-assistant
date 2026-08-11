"""
RAG Projesi - Ortak Vektor Karsilastirma Fonksiyonlari

main.py ve app.py bu moduldeki fonksiyonlari kullanir; boylece ayni mantik
iki ayri dosyada birbirinden bagimsiz olarak (ve potansiyel olarak
tutarsiz sekilde) tekrar yazilmamis olur.
"""

import math
from typing import List, Tuple


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Iki vektor arasindaki kosinus benzerligini hesaplar.

    Farkli boyuttaki (dimension) vektorler zip() ile sessizce kisa olanina
    gore kirpilip yanlis bir skor uretebilir -- ornegin embedding modeli
    ileride degistirildiginde. Bu durumu sessizce gecmek yerine burada
    acik bir hata olarak firlatiyoruz, boylece sorun hemen fark edilir.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vektor boyutlari uyusmuyor: {len(a)} != {len(b)}. "
            "Embedding modeli degismis olabilir; veritabanini yeniden "
            "olusturmaniz (setup_db.py) gerekebilir."
        )

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(
    query_embedding: List[float],
    db_records: List[Tuple],
    top_k: int = 2,
) -> List[Tuple]:
    """Kullanici sorgusuna en yakin anlamsal eslesmeyi saglayan kayitlari bulur."""
    scores = []
    for record in db_records:
        doc_id, doc_content, doc_emb = record
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((doc_id, doc_content, score))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]