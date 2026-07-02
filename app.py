import streamlit as st
import requests
import json
import math
import re
import os
from collections import Counter

# TF-IDF Simple RAG Search Engine
def tokenize(text):
    return re.findall(r'\w+', text.lower())

class SimpleSearch:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [tokenize(doc.get('instruction', '')) for doc in documents]
        self.doc_freqs = Counter()
        for doc in self.tokenized_docs:
            self.doc_freqs.update(set(doc))
        self.num_docs = len(documents)

    def search(self, query, top_k=2):
        query_tokens = tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.tokenized_docs):
            score = 0
            doc_len = len(doc_tokens)
            if doc_len == 0:
                continue
            for token in query_tokens:
                if token in doc_tokens:
                    tf = doc_tokens.count(token) / doc_len
                    df = self.doc_freqs.get(token, 0)
                    idf = math.log((1 + self.num_docs) / (1 + df)) + 1
                    score += tf * idf
            scores.append((score, self.documents[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0]

@st.cache_resource
def load_search_index():
    dataset_path = "nityacare_culture_dataset.jsonl"
    
    # Check if the file exists in the current working directory
    if not os.path.exists(dataset_path):
        # Fallback to the exact directory where app.py is hosted
        project_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(project_dir, dataset_path)
    
    documents = []
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        documents.append(json.loads(line.strip()))
                    except Exception:
                        pass
    return SimpleSearch(documents)

st.set_page_config(page_title="NityaCare Culture Coach", page_icon="🛡️")
st.title("🛡️ NityaCare Culture Coach")
st.caption("A 100% private, anonymous workplace alignment assistant.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Describe a workplace conflict or scenario..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # RAG - Search for matching examples
        search_index = load_search_index()
        matches = search_index.search(user_input, top_k=2)
        
        # Build prompt incorporating retrieved context
        examples_text = ""
        for idx, match in enumerate(matches):
            examples_text += f"EXAMPLE {idx + 1}:\n### Workplace Situation:\n{match['instruction']}\n\n### Culture Coach Response:\n{match['output']}\n\n"
            
        # Formulate an adaptive prompt that tells the model to scale its response based on the input
        formatted_prompt = (
            f"<start_of_turn>user\n"
            f"You are the NityaCare Culture Coach. Analyze the workplace situation and provide clear guidance based on the Team Culture Charter using 'Our Way' vs 'Not Our Way' parameters.\n"
            f"DYNAMIC LENGTH CONTROL RULE:\n"
            f"- If the user input is just a short greeting, casual statement, or simple sentence (under 10 words), reply with a warm, brief, direct response (1-2 sentences max).\n"
            f"- If the user input describes a detailed scenario or conflict, provide a comprehensive, deep structural breakdown using the charter guidelines.\n\n"
        )
        if examples_text:
            formatted_prompt += f"Here are examples of how we handle similar situations:\n\n{examples_text}"
        
        formatted_prompt += (
            f"Now, analyze the following situation:\n"
            f"### Workplace Situation:\n{user_input}\n\n"
            f"### Culture Coach Response:<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma2:2b", # 1. Pivoted away from the corrupted file to the pure base model
            "prompt": formatted_prompt, 
            "stream": True,
            "options": {
                "temperature": 0.2, # 2. Lowered temperature slightly to keep answers factual and grounded
                "stop": ["<end_of_turn>", "### Workplace Situation:", "### Culture Coach Response:"]
            }
        }
        
        response = requests.post(url, json=payload, stream=True)
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                text_chunk = chunk.get("response", "")
                
                # Double-check guardrail to cut off text string loop immediately
                if "### Workplace" in text_chunk or "### Culture" in text_chunk:
                    break
                    
                full_response += text_chunk
                response_placeholder.markdown(full_response + "▌")
                
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
