import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# Ambil API token dari Streamlit secrets (pastikan Anda sudah mengatur file .streamlit/secrets.toml atau melalui dashboard Streamlit Cloud)
hf_api_key = st.secrets["HF_API_KEY"]

# Inisialisasi client untuk remote inference menggunakan token tersebut
client = InferenceClient(api_key=hf_api_key)

# Judul aplikasi
st.title("Chatbot dengan Remote Inference")

# Inisialisasi riwayat pesan (session state) jika belum ada
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat percakapan yang telah disimpan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input dari pengguna
user_input = st.chat_input("Ketik pesan Anda...")

if user_input:
    # Tambahkan pesan pengguna ke riwayat dan tampilkan
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Buat prompt dengan instruksi agar jawaban dihasilkan dalam bahasa Indonesia
    prompt = "Tolong jawab pertanyaan berikut dalam bahasa Indonesia:\n" + user_input

    try:
        # Panggil endpoint text generation (remote inference) untuk menghasilkan respons chatbot
        response = client.text_generation(
            prompt=prompt,
            model="meta-llama/Llama-3.2-3B",  # Ganti dengan model yang Anda inginkan
            max_new_tokens=150
        )
        # Misalnya, respons berupa list dict dengan key "generated_text"
        if isinstance(response, list) and "generated_text" in response[0]:
            bot_response = response[0]["generated_text"].strip()
        else:
            bot_response = "No response"
    except HfHubHTTPError as e:
        # Jika terjadi error, tangkap error dan konversikan ke string
        bot_response = str(e)

    # Tambahkan respons bot ke riwayat dan tampilkan
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
