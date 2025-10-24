# Kurumsal SSS Botu – RAG

Bu repo, Markdown/TXT SSS dokümanlarından **RAG tabanlı** kısa ve atıflı cevap üretir.

## 1) Proje Yapısı (şablon)
```
project-template/
├─ data/
│ ├─ externa/
│ ├─ processed/
│ └─ raw/
├─ notebooks/
│ └─ eda.ipynb
├─ src/
│ ├─ data_processing/
│ │ └─ make_dataset.py
│ ├─ models/
│ │ ├─ predict_model.py
│ │ └─ train_model.py # (Kolay projede kullanılmıyor)
│ └─ init.py
├─ outputs/
│ └─ models/
├─ configs/
│ └─ config.yaml
├─ tests/
│ └─ test_model.py
├─ README.md
├─ requirements.txt
├─ Dockerfile
└─ main.py
```


## 2) Hızlı Başlangıç (Yerel)
```bash
pip install -r requirements.txt
# Ham veriyi koy:
#   data/raw/*.md, *.txt
python main.py build-index --config configs/config.yaml
python main.py query "Üyelik ücretli mi?"
```

## 3) Config 

- data.raw_dir, data.processed_dir → girdi/çıktı dizinleri
- indexing.chunk_size / chunk_overlap → parçalama kontrolü
- indexing.embedding_model / normalize_embeddings → gömme/benzerlik uzayı
- retrieval.top_k, min_score, use_mmr → geri getirme kalitesi
- answer.mode (extractive/llm), max_sentences → kısa ve atıflı cevap

## 4) Testler
```bash
pytest -q
```

## 5) Docker ile Çalıştırma
Aşağıdaki komutlarda, mevcut klasörü konteynere bağlayıp aynı kaynakları kullanıyoruz.

### 5.1 İmajı oluştur
```bash
docker build -t rag-sss:latest .
```

### 5.2 İndeks üret
Linux/Mac:
```bash
docker run --rm -it -v "$(pwd)":/app -w /app rag-sss:latest \
  python main.py build-index --config configs/config.yaml
```

Windows (PowerShell):
```bash
docker run --rm -it -v "${PWD}:/app" -w /app rag-sss:latest `
  python main.py build-index --config configs/config.yaml
```

### 5.3 Sorgu çalıştır
Linux/Mac:
```bash
docker run --rm -it -v "$(pwd)":/app -w /app rag-sss:latest \
  python main.py query "Üyelik ücretli mi?"
```

Windows (PowerShell):
```bash
docker run --rm -it -v "${PWD}:/app" -w /app rag-sss:latest `
  python main.py query "Üyelik ücretli mi?"
```

LLM modu istersen:
- "```pip install openai```, ```OPENAI_API_KEY``` ortam değişkeni"
- "```configs/config.yaml``` içinde ```answer.mode: llm``` yap"

## 6) Sorun Giderme
- Yanlış/uzun cevaplar → ```answer.max_sentences```, ```retrieval.min_score``` ayarlarını yükseltin, ```chunk_size``` 400–500 deneyin.
- Model uyumsuzluğu → ```index_meta.json``` içindeki ```embedding_model``` ile ```config.yaml``` aynı olmalı; değilse indeksi yeniden kurun.
- FAISS/pyyaml bulunamadı → doğru ortamda ```ipip install -r requirements.txt```.
- Unicode hata/çıktı bozuk → Docker imajı UTF-8; yerelde ```PYTHONIOENCODING=utf-8``` ayarlayın.

## 7) Küçük sağlamlaştırmalar (opsiyonel ama faydalı)
- ```requirements.txt``` içine zaten eklediklerimiz yeterli. LLM kullanacaksan ```openai``` eklersin.
- ```main.py``` içinde daha önce yaptığımız UTF-8 stdout ve lazy import kalmalı.
- CI/CD’ye taşımayı düşünürsen Docker imajını registry’ye push edip "```docker run ...```" ile her yerde aynı sonucu alırsın.