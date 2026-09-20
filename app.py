import os
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="BetaPak AI Teknik Asistan")

# Gemini API ve ChromaDB Bağlantıları
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

db_path = "./chroma_db"
try:
  chroma_client = chromadb.PersistentClient(path=db_path)
  collection = chroma_client.get_or_create_collection("betapak_kilavuzlar")
except Exception as e:
  collection = None


class SoruIstegi(BaseModel):
  soru: str


# Statik dosyaları (HTML/CSS) sunmak için
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
  return FileResponse("static/index.html")


@app.post("/soru-sor")
def soru_sor(istek: SoruIstegi):
  try:
    context_text = ""
    if collection:
      results = collection.query(query_texts=[istek.soru], n_results=3)
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
  uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
