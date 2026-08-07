# 🚀 Microsoft Foundry Local - Local RAG AI Assistant

Bu proje, **Microsoft Foundry Local SDK** ve **SQLite** veritabanı kullanılarak geliştirilmiş, internet bağlantısı gerektirmeyen tamamen çevrimdışı (offline) ve sıfır-halüsinasyon odaklı bir **Yerel RAG (Retrieval-Augmented Generation)** sistemidir.

---

## 🌟 Öne Çıkan Özellikler

* 🔒 **%100 Yerel ve Güvenli:** Tüm vektörleştirme ve LLM çıkarım işlemleri cihaz üzerinde (on-device) gerçekleşir, veri dışarı çıkmaz.
* 🛡️ **Sıfır Halüsinasyon (Zero Hallucination):** Yazılım seviyesinde eklenen **Kosinüs Benzerliği Eşik Filtresi (Threshold = 0.68)** sayesinde veritabanında karşılığı olmayan sorular LLM'e ulaşmadan engellenir.
* ⚡ **Gelişmiş Vektör Arama:** `qwen3-embedding-0.6b` modeli ile metinler vektörleştirilir ve SQLite veritabanı üzerinde anlamsal arama yapılır.
* 💻 **Kullanıcı Dostu Arayüz:** Streamlit tabanlı sohbet arayüzü, canlı yanıt akışı (streaming), bağlam gösterici (context expander) ve test senaryoları içerir.

---

## 🏗️ Mimari ve Akış

```text
[ Kullanıcı Sorgusu ] 
         │
         ▼
[ Qwen Embedding Modeli ] ──► Sorgu Vektöre Çevrilir
         │
         ▼
[ SQLite Veritabanı ] ──────► Kosinüs Benzerlik Araması Yapılır
         │
         ▼
[ Güvenlik Filtresi (Threshold >= 0.68) ]
   ├── (Skor Düşükse) ────► "Yeterli bilgi bulunmamaktadır."
   └── (Skor Yüksekse) ───► [ Qwen 2.5 0.5b LLM ] ──► Yanıt Üretilir (Streaming)