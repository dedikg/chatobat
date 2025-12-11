# File: quick_test.py
import streamlit as st

st.write("🔑 API Key dari Secrets:", st.secrets.get("OPENROUTER_API_KEY", "TIDAK DITEMUKAN"))

if st.button("Test Sederhana"):
    import requests
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": [{"role": "user", "content": "Test dari Streamlit"}]
        }
    )
    
    st.write("Status:", response.status_code)
    if response.status_code == 200:
        st.success("✅ API Ready!")
