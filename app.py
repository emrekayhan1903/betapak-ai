import os
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI(title="BetaPak AI Teknik Asistan")

# API Anahtarı ve Gemini Client (Cevap üretmek için)
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# ChromaDB Bağlantısı
CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "makine_kilavuzlari"

try:
  chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
  collection = chroma_client.get_or_create_collection(
      name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
  )
  print(
      f"--- CHROMA DB BAĞLANTI BAŞARILI. Toplam Belge:"
      f" {collection.count()} ---"
  )
except Exception as e:
  print(f"--- CHROMA DB HATA: {str(e)} ---")
  collection = None


def gemini_embedding_al(metin):
  """Colab'de birebir kullandığın ve çalışan HTTP tabanlı embedding fonksiyonu"""
  try:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": metin}]},
    }
    res = requests.post(url, json=payload, headers=headers, timeout=15)

    if res.status_code == 200:
      return res.json()["embedding"]["values"]
    else:
      print(f"❌ API Embedding Hatası ({res.status_code}): {res.text}")
      return None
  except Exception as e:
    print(f"❌ Bağlantı Hatası: {e}")
    return None


class SoruIstegi(BaseModel):
  soru: str


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
  return FileResponse("static/index.html")


@app.post("/soru-sor")
def soru_sor(istek: SoruIstegi):
  try:
    context_text = ""
    if collection:
      # 1. Kullanıcının sorusunu Colab ile birebir aynı yöntemle vektöre çeviriyoruz
      query_vector = gemini_embedding_al(istek.soru)

      if query_vector:
        # 2. ChromaDB'de vektör ile arama yapıyoruz
        results = collection.query(
            query_embeddings=[query_vector], n_results=3
        )
        print("Sorgu Sonucu:", results)

        if results and "documents" in results and results["documents"]:
          documents = results["documents"][0]
          if documents:
            context_text = "\n\n".join(documents)

    system_prompt = (
        "Sen BetaPak AI Teknik Destek Asistanısın. Omron ve Weintek sistemleri,"
        " servo sürücüler ve paketleme makineleri konusunda uzmansın.\n"
        "Aşağıda sana sağlanan kılavuz bağlamını kullanarak kullanıcının"
        " sorusunu yanıtla.\n"
        "KESİN KURAL: Eğer sorunun yanıtı verilen kılavuz metinlerinde kesin"
        " olarak geçmiyorsa, asla kendi genel bilgini kullanma. 'Bu konuda"
        " yüklenen kılavuzlarda herhangi bir bilgi bulunamadı' de.\n\n"
        f"İlgili Kılavuz Metinleri:\n{context_text}"
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=istek.soru,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    )

    return {"cevap": response.text}

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)
