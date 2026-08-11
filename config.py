"""
RAG Projesi - Ortak Yapilandirma Sabitleri (Single Source of Truth)

Bu dosya, projede birden fazla dosyada (main.py, app.py, setup_db.py) tekrar
tanimlanan sabitleri tek bir yerde toplar. Bir degeri degistirmek istediginizde
(ornegin benzerlik esigini veya model adini) sadece burayi guncellemeniz yeterlidir.
"""

# Foundry Local uygulama kimligi
APP_NAME = "foundry_local_rag"

# SQLite veritabani dosya adi
DB_NAME = "rag_database.db"

# Kullanilan modeller
EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
CHAT_MODEL_NAME = "qwen2.5-0.5b"

# Vektor aramasinda dondurulecek en alakali belge sayisi
TOP_K = 2

# Guvenlik esigi: Bu skorun altindaki eslesmeler "yetersiz baglam" kabul edilir
# ve LLM'e hic istek gonderilmeden standart uyari mesaji dondurulur.
SIMILARITY_THRESHOLD = 0.68

# Baglam yetersiz oldugunda kullaniciya gosterilecek standart yanit
FALLBACK_MESSAGE = "Bu konuda veritabanımda yeterli bilgi bulunmamaktadır."