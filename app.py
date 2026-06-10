import os
import json
import time
import requests
import pickle
import numpy as np
import faiss
import gradio as gr
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

load_dotenv()

# ── Constants ─────────────────────────────────────────────────
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DATA_DIR = "data"
CHUNKS_FILE = f"{DATA_DIR}/chunks_with_meta.pkl"
INDEX_FILE = f"{DATA_DIR}/faiss_index.bin"

os.makedirs(DATA_DIR, exist_ok=True)

# ── PubMed Fetch ──────────────────────────────────────────────
def fetch_pubmed(query, max_results=200):
    url = f"{BASE_URL}esearch.fcgi"
    params = {"db": "pubmed", "term": query, 
              "retmax": max_results, "retmode": "json"}
    r = requests.get(url, params=params)
    return r.json()["esearchresult"]["idlist"]

def fetch_abstracts(pmids, batch_size=100):
    import xml.etree.ElementTree as ET
    abstracts = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        url = f"{BASE_URL}efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(batch),
                  "rettype": "abstract", "retmode": "xml"}
        r = requests.get(url, params=params)
        try:
            root = ET.fromstring(r.text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    title = article.findtext(".//ArticleTitle", "No title")
                    texts = article.findall(".//AbstractText")
                    abstract = " ".join([a.text for a in texts if a.text])
                    if not abstract:
                        continue
                    pmid = article.findtext(".//PMID", "Unknown")
                    year = article.findtext(".//PubDate/Year", "Unknown")
                    abstracts.append({
                        "pmid": pmid, "title": title,
                        "abstract": abstract, "year": year,
                        "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
                except:
                    continue
        except:
            continue
        time.sleep(0.5)
    return abstracts

# ── Build Index ───────────────────────────────────────────────
def build_index(abstracts, model):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    
    chunks = []
    for paper in abstracts:
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        splits = splitter.split_text(full_text)
        for i, chunk in enumerate(splits):
            chunks.append({
                "chunk_id": f"{paper['pmid']}_{i}",
                "pmid": paper["pmid"],
                "text": chunk,
                "title": paper["title"],
                "year": paper["year"],
                "source": paper["source"]
            })
    
    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Index built with {index.ntotal} vectors")
    return index, chunks

# ── Load or Build ─────────────────────────────────────────────
def initialize():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing index...")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded index with {index.ntotal} vectors")
    else:
        print("Building index from PubMed...")
        queries = [
            "glioma brain tumor MRI diagnosis",
            "glioblastoma treatment",
            "brain tumor segmentation",
        ]
        all_pmids = []
        for q in queries:
            pmids = fetch_pubmed(q, max_results=50)  # smaller batch
            print(f"Got {len(pmids)} PMIDs for: {q}")
            all_pmids.extend(pmids)
            time.sleep(1)  # wait between requests
        
        all_pmids = list(set(all_pmids))
        print(f"Total unique PMIDs: {len(all_pmids)}")
        
        if len(all_pmids) == 0:
            raise Exception("PubMed returned 0 results - check network")
        
        abstracts = fetch_abstracts(all_pmids)
        print(f"Fetched {len(abstracts)} abstracts")
        
        if len(abstracts) == 0:
            raise Exception("0 abstracts parsed - check XML parsing")
        
        index, chunks = build_index(abstracts, model)
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    
    return model, index, chunks, llm

# ── RAG ───────────────────────────────────────────────────────
def retrieve(query, model, index, chunks, top_k=5):
    qe = model.encode([query])
    qe = np.array(qe, dtype="float32")
    faiss.normalize_L2(qe)
    scores, indices = index.search(qe, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "title": chunk["title"],
            "source": chunk["source"],
            "year": chunk["year"],
            "score": float(score)
        })
    return results

def ask(question, model, index, chunks, llm):
    retrieved = retrieve(question, model, index, chunks)
    context = ""
    for i, chunk in enumerate(retrieved):
        context += f"""
[Source {i+1}]
Title: {chunk['title']} ({chunk['year']})
Text: {chunk['text']}
URL: {chunk['source']}
---"""
    
    prompt = f"""You are a medical research assistant.
Answer using ONLY the provided research context.
Cite sources by mentioning paper titles.
If context is insufficient, say so honestly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, retrieved

# ── Gradio UI ─────────────────────────────────────────────────
print("Initializing MedRAG...")
model, index, chunks, llm = initialize()
print("Ready!")

def query_medrag(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    answer, sources = ask(question, model, index, chunks, llm)
    
    sources_md = "\n\n---\n### 📚 Sources\n"
    for i, s in enumerate(sources):
        sources_md += f"\n**{i+1}. {s['title']}** ({s['year']})"
        sources_md += f"\n🔗 [{s['source']}]({s['source']})"
        sources_md += f"\n📊 Score: {s['score']:.4f}\n"
    
    return answer, sources_md

with gr.Blocks(theme=gr.themes.Soft(), title="MedRAG") as demo:
    gr.Markdown("""
    # 🧠 MedRAG — Medical Research Assistant
    ### Ask questions about brain tumors · Powered by PubMed + Llama 3
    """)
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What are treatment options for glioblastoma?",
            lines=2
        )
    
    ask_btn = gr.Button("🔍 Search Research", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer")
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="Sources")
    
    gr.Examples(
        examples=[
            ["What are treatment options for glioblastoma?"],
            ["How is MRI used in brain tumor diagnosis?"],
            ["What is the survival rate for glioma patients?"],
            ["What role does temozolomide play in glioma treatment?"],
        ],
        inputs=question_input
    )
    
    ask_btn.click(fn=query_medrag,
                  inputs=question_input,
                  outputs=[answer_output, sources_output])
    question_input.submit(fn=query_medrag,
                          inputs=question_input,
                          outputs=[answer_output, sources_output])

demo.launch()