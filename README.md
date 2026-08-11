# 🚀 Microsoft Foundry Local - Local RAG AI Assistant

Bu proje, **Microsoft Foundry Local SDK** ve **SQLite** veritabanı kullanılarak geliştirilmiş, internet bağlantısı gerektirmeyen tamamen çevrimdışı (offline) ve sıfır-halüsinasyon odaklı bir **Yerel RAG (Retrieval-Augmented Generation)** sistemidir.

---

## 🌟 Öne Çıkan Özellikler

* 🔒 **%100 Yerel ve Güvenli:** Tüm vektörleştirme ve LLM çıkarım işlemleri cihaz üzerinde (on-device) gerçekleşir, veri dışarı çıkmaz.
* 🛡️ **Sıfır Halüsinasyon (Zero Hallucination):** Yazılım seviyesinde eklenen **Kosinüs Benzerliği Eşik Filtresi (Threshold = 0.68)** sayesinde veritabanında karşılığı olmayan sorular LLM'e ulaşmadan engellenir.
* ⚡ **Gelişmiş Vektör Arama:** `qwen3-embedding-0.6b` modeli ile metinler vektörleştirilir ve SQLite veritabanı üzerinde anlamsal arama yapılır.
* 💻 **Kullanıcı Dostu Arayüz:** Streamlit tabanlı sohbet arayüzü, canlı yanıt akışı (streaming), bağlam gösterici (context expander) ve test senaryoları içerir.
* 🧩 **Modüler Kod Yapısı:** Ortak sabitler (`config.py`) ve ortak vektör karşılaştırma mantığı (`rag_utils.py`) tek bir kaynaktan yönetilir; hem CLI hem web arayüzü aynı, tek doğrulanmış çekirdeği kullanır.

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
```

---

## 📁 Proje Yapısı

```text
local-RAG-assistant/
├── config.py          # Ortak sabitler: model adları, eşik değeri, DB adı, fallback mesajı
├── rag_utils.py        # Ortak fonksiyonlar: cosine_similarity, find_relevant (boyut kontrolü dahil)
├── setup_db.py          # Bilgi tabanını vektörleştirip SQLite'a yazan tek seferlik kurulum betiği
├── main.py             # Komut satırı (CLI) sürümü
├── app.py               # Streamlit tabanlı web arayüzü
├── requirements.txt      # Proje bağımlılıkları
├── .gitignore            # Git'e dahil edilmeyecek dosya/klasörler
└── README.md              # Bu dosya
```

> `rag_database.db` deponun bir parçası değildir — `.gitignore` tarafından hariç tutulur ve her kurulumda `setup_db.py` çalıştırılarak yeniden oluşturulur.

---

## ⚙️ Kurulum

1. Depoyu klonlayın ve proje klasörüne girin:
   ```bash
   git clone https://github.com/yagmurgoray/local-RAG-assistant.git
   cd local-RAG-assistant
   ```

2. Sanal ortam oluşturun ve etkinleştirin (önerilir):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

4. Bilgi tabanını oluşturun (yalnızca ilk kurulumda gerekli):
   ```bash
   python setup_db.py
   ```
   Bu komut, `qwen3-embedding-0.6b` modelini indirir, örnek belgeleri vektörleştirir ve `rag_database.db` dosyasını oluşturur.

---

## ▶️ Kullanım

Sistemin iki farklı çalıştırma modu vardır — ikisi de aynı `config.py` / `rag_utils.py` çekirdeğini kullanır:

**Komut satırı (CLI) sürümü:**
```bash
python main.py
```
Terminalden interaktif olarak soru sorabilir, çıkmak için `quit` yazabilirsiniz.

**Web arayüzü (Streamlit) sürümü:**
```bash
streamlit run app.py
```
Tarayıcıda açılan sohbet arayüzünden soru sorabilir, sidebar'daki örnek sorularla sistemi test edebilirsiniz.

> ⚠️ Not: `app.py`'nin sistem promptu "Strict QA" politikasıyla katılaştırılmıştır; `main.py`'nin komut satırı sürümü şu an için daha esnek bir sistem promptu kullanır — bu güncelleme ayrı bir adımda planlanmıştır.

---

## 🔐 .gitignore Hakkında

Proje kökündeki `.gitignore` şu klasör/dosyaları depo dışında tutar: `__pycache__/`, derlenmiş `.pyc` dosyaları, sanal ortam klasörleri (`venv/`, `env/`, `.venv/`), SQLite veritabanı (`*.db`) ve Streamlit'in yerel yapılandırma klasörü (`.streamlit/`). Bu dosya **silinmemeli** — özellikle `.db` dosyasının ve sanal ortamın depoya karışmasını önler.
