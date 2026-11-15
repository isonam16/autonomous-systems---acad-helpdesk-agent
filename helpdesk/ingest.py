"""Build the persistent Chroma index using Gemini embeddings."""
import hashlib,json,re
import chromadb
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
from loguru import logger
try: import pymupdf as fitz
except ImportError: import fitz
from . import config as C
client=genai.Client(api_key=C.GEMINI_API_KEY)
WEB_NOISE="###147852369$$$_RedirectToLoginPage_%%%963258741!!!"
WEB_FOOTER=re.compile(r"Important links\\s+CCCD @ IITJ.*",re.S)

def embed(texts:list[str],task_type:str="RETRIEVAL_QUERY")->list[list[float]]:
 vectors=[]
 for start in range(0,len(texts),32):
  response=client.models.embed_content(model=C.EMBED_MODEL,contents=texts[start:start+32],config=types.EmbedContentConfig(task_type=task_type,output_dimensionality=768))
  vectors.extend(item.values for item in response.embeddings)
 return vectors

def split_text(text:str)->list[str]:
 paras=[re.sub(r"\\s+"," ",p).strip() for p in re.split(r"\\n\\s*\\n",text)]
 paras=[p for p in paras if len(p)>20]
 out=[]; current=""
 for p in paras:
  while len(p)>C.CHUNK_CHARS:
   out.append(p[:C.CHUNK_CHARS]);p=p[C.CHUNK_CHARS-C.CHUNK_OVERLAP:]
  if current and len(current)+len(p)>C.CHUNK_CHARS:
   out.append(current);current=current[-C.CHUNK_OVERLAP:]+" "+p
  else: current=(current+" "+p).strip()
 if current: out.append(current)
 return out

def parse_file(path):
 items=[];suffix=path.suffix.lower()
 if suffix==".pdf":
  for i,page in enumerate(fitz.open(path),1):
   items.extend({"text":t,"source":path.name,"loc":f"page {i}"} for t in split_text(page.get_text()))
 elif suffix in {".html",".htm"}:
  soup=BeautifulSoup(path.read_text(encoding="utf-8",errors="ignore"),"html.parser")
  items.extend({"text":t,"source":path.name,"loc":"html"} for t in split_text(soup.get_text("\\n")))
 elif suffix in {".txt",".md"}:
  raw=path.read_text(encoding="utf-8",errors="ignore")
  m=re.search(r"SOURCE_URL:\\s*(\\S+)",raw);loc=m.group(1) if m else "text"
  raw=WEB_FOOTER.sub("",raw.replace(WEB_NOISE," "))
  items.extend({"text":t,"source":path.name,"loc":loc} for t in split_text(raw))
 return items

def add_context(chunk,doc_title):
 prompt=f"Document: {doc_title} ({chunk['loc']})\\nChunk:\\n{chunk['text']}\\n\\nIn one short sentence, say what part of the document this chunk is about."
 return (client.models.generate_content(model=C.GEMINI_MODEL,contents=prompt).text or "").strip()

def main():
 C.CHROMA_DIR.mkdir(parents=True,exist_ok=True);C.CHUNKS_JSON.parent.mkdir(parents=True,exist_ok=True)
 chunks=[];seen=set();files=[p for p in sorted(C.DOCS_DIR.iterdir()) if p.is_file() and not p.name.startswith(".")]
 for path in files:
  for item in parse_file(path):
   key=hashlib.md5(re.sub(r"\\W+","",item["text"].lower()).encode()).hexdigest()
   if key in seen: continue
   seen.add(key);item["id"]=f"c{len(chunks)}";item["embed_text"]=item["text"]
   if C.USE_CONTEXTUAL_RETRIEVAL:item["embed_text"]=add_context(item,path.name)+"\\n"+item["text"]
   chunks.append(item)
 if not chunks: raise SystemExit("No chunks produced. Add documents to data/docs/.")
 db=chromadb.PersistentClient(path=str(C.CHROMA_DIR))
 try: db.delete_collection("helpdesk")
 except Exception: pass
 col=db.create_collection("helpdesk",metadata={"hnsw:space":"cosine"})
 for start in range(0,len(chunks),32):
  batch=chunks[start:start+32]
  col.add(ids=[x["id"] for x in batch],embeddings=embed([x["embed_text"] for x in batch],"RETRIEVAL_DOCUMENT"),documents=[x["text"] for x in batch],metadatas=[{"source":x["source"],"loc":x["loc"]} for x in batch])
  logger.info(f"Indexed {min(start+len(batch),len(chunks))}/{len(chunks)} chunks")
 C.CHUNKS_JSON.write_text(json.dumps(chunks,ensure_ascii=False),encoding="utf-8")
 logger.info("Gemini embedding ingestion complete")
if __name__=="__main__":main()
