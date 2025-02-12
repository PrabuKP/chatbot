# chatbot.py
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Konfigurasi Model
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Ganti dengan model yang sesuai
HF_TOKEN = "your_hf_token"  # Dapatkan dari https://huggingface.co/settings/tokens

# Inisialisasi Model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True  # Kuantisasi untuk menghemat memori
    )
    return model, tokenizer

model, tokenizer = load_model()

# Fungsi Generate Response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI Streamlit
st.title("🦙 LLaMA Chatbot 3.2 1B")
st.caption("Powered by Meta LLaMA and Streamlit")

# Inisialisasi chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pengguna
if prompt := st.chat_input("Apa pertanyaan Anda?"):
    # Tambahkan user message ke history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Tampilkan user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Anda adalah asisten AI yang membantu. Berikan jawaban yang jelas dan singkat.<|eot_id|>
        {"".join([f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>" 
                 for msg in st.session_state.messages])}
        <|start_header_id|>assistant<|end_header_id|>"""
        
        response = generate_response(full_prompt)
        response_clean = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        st.markdown(response_clean)
    
    # Tambahkan assistant response ke history
    st.session_state.messages.append({"role": "assistant", "content": response_clean})