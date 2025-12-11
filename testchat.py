import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="Test OpenRouter API",
    page_icon="🔑",
    layout="centered"
)

st.title("🔑 Test OpenRouter API Key")
st.markdown("Testing API key dari Streamlit Secrets")

# ============================================
# FUNGSI TESTING
# ============================================

def test_basic_connection(api_key: str):
    """Test koneksi dasar ke OpenRouter"""
    with st.spinner("🔄 Menguji koneksi ke OpenRouter..."):
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": st.secrets.get("APP_URL", "https://streamlit.io"),
            "X-Title": "Drug Assistant Test"
        }
        
        # Request sederhana
        payload = {
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": [
                {"role": "user", "content": "Jawab singkat: 'API berhasil terhubung!'"}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                reply = result['choices'][0]['message']['content']
                
                st.success("✅ **KONEKSI BERHASIL!**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status Code", response.status_code)
                with col2:
                    st.metric("Model", result.get('model', 'Unknown'))
                
                st.info(f"**Response:** {reply}")
                
                # Tampilkan usage info
                if 'usage' in result:
                    usage = result['usage']
                    st.caption(f"📊 Usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion tokens")
                
                return True, result
                
            else:
                st.error(f"❌ **ERROR {response.status_code}**")
                
                if response.status_code == 401:
                    st.warning("API Key tidak valid atau expired. Cek kembali key di Secrets.")
                elif response.status_code == 402:
                    st.warning("Kredit tidak mencukupi atau key sudah expired.")
                elif response.status_code == 429:
                    st.warning("Rate limit tercapai. Tunggu beberapa saat.")
                
                st.code(response.text[:300], language="json")
                return False, None
                
        except requests.exceptions.Timeout:
            st.error("⏱️ **TIMEOUT:** Request lebih dari 10 detik")
            return False, None
        except Exception as e:
            st.error(f"⚠️ **ERROR:** {type(e).__name__}: {str(e)[:100]}")
            return False, None

def test_medical_translation(api_key: str):
    """Test terjemahan medis"""
    with st.spinner("🧪 Testing terjemahan medis..."):
        medical_text = "Take 2 tablets every 6 hours for pain. Do not exceed 4000 mg per day."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Terjemahkan teks medis ke Bahasa Indonesia:

{medical_text}

ATURAN:
1. Pertahankan angka (2, 6, 4000) dan satuan (mg, tablets, hours, day)
2. Hasil natural untuk pasien Indonesia
3. Jangan ubah makna medis"""
        
        payload = {
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": [
                {"role": "system", "content": "Anda penerjemah medis profesional."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                translation = result['choices'][0]['message']['content']
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("📝 **Original (English)**", medical_text, height=100)
                with col2:
                    st.text_area("🇮🇩 **Hasil Terjemahan**", translation, height=100)
                
                # Cek kualitas terjemahan
                st.subheader("🔍 Quality Check")
                
                checks = []
                if any(char.isdigit() for char in translation):
                    checks.append("✅ Angka dipertahankan")
                if 'mg' in translation.lower():
                    checks.append("✅ Satuan 'mg' dipertahankan")
                if 'tablet' in translation.lower():
                    checks.append("✅ 'tablet' dipertahankan")
                
                for check in checks:
                    st.write(check)
                
                return True
            else:
                st.warning(f"Translation test failed: {response.status_code}")
                return False
                
        except Exception as e:
            st.warning(f"Translation error: {e}")
            return False

def get_available_models(api_key: str):
    """Ambil daftar model yang tersedia"""
    with st.spinner("🔍 Mengecek model yang tersedia..."):
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('data', [])
                
                # Filter model yang relevan
                relevant_models = []
                for model in models:
                    model_id = model.get('id', '')
                    
                    # Filter model Gemini/Google atau yang gratis
                    if any(x in model_id.lower() for x in ['gemini', 'google', 'gpt-4o-mini', 'claude-3-haiku']):
                        pricing = model.get('pricing', {})
                        is_free = (pricing.get('prompt') == '0' and 
                                  pricing.get('completion') == '0')
                        
                        relevant_models.append({
                            'id': model_id,
                            'free': is_free,
                            'description': model.get('description', '')[:100]
                        })
                
                # Tampilkan dalam tabel
                if relevant_models:
                    st.subheader("📦 Model yang Tersedia")
                    
                    for model in relevant_models[:8]:  # Tampilkan 8 pertama
                        free_badge = " 🆓" if model['free'] else " 💰"
                        st.write(f"• **{model['id']}**{free_badge}")
                        if model['description']:
                            st.caption(f"  _{model['description']}_")
                    
                    return relevant_models
            
        except Exception as e:
            st.warning(f"Cannot fetch models: {e}")
    
    return []

# ============================================
# UI STREAMLIT
# ============================================

# Sidebar untuk manual input (optional)
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Option untuk manual override
    use_manual_key = st.checkbox("Gunakan API Key manual (override Secrets)")
    
    if use_manual_key:
        manual_key = st.text_input("Manual API Key:", type="password")
        api_key = manual_key
    else:
        # Ambil dari secrets
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    
    st.divider()
    st.caption("Key dari Secrets:")
    if api_key:
        masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        st.code(masked_key)
    else:
        st.warning("Key tidak ditemukan di Secrets")

# Main content
if not api_key:
    st.error("""
    ## ❌ OPENROUTER_API_KEY tidak ditemukan!
    
    Tambahkan ke `.streamlit/secrets.toml`:
    
    ```toml
    OPENROUTER_API_KEY = "sk-or-v1-2e4...fdf"
    ```
    
    Atau masukkan manual di sidebar.
    """)
    
    with st.expander("📁 Contoh file secrets.toml"):
        st.code("""# .streamlit/secrets.toml
OPENROUTER_API_KEY = "sk-or-v1-2e4...your_key...fdf"

# Optional: App URL untuk referer header
APP_URL = "https://your-app-name.streamlit.app"
""")
    
    st.stop()

# Display key info
st.subheader("📋 API Key Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Key Length", f"{len(api_key)} chars")
with col2:
    st.metric("Starts with", api_key[:8] if api_key else "N/A")
with col3:
    # Cek jika key valid format
    if api_key.startswith("sk-or-"):
        st.success("Format Valid")
    else:
        st.warning("Format Tidak Biasa")

st.divider()

# Tombol untuk test
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 **Test Basic Connection**", use_container_width=True):
        success, result = test_basic_connection(api_key)

with col2:
    if st.button("🧪 **Test Translation**", use_container_width=True):
        test_medical_translation(api_key)

with col3:
    if st.button("📦 **List Models**", use_container_width=True):
        get_available_models(api_key)

st.divider()

# Advanced testing
with st.expander("🔬 Advanced Testing"):
    st.write("Test dengan custom prompt:")
    
    custom_prompt = st.text_area(
        "Custom Prompt:",
        value="Buat ringkasan singkat tentang paracetamol dalam 50 kata",
        height=100
    )
    
    if st.button("🚀 Run Custom Test"):
        with st.spinner("Running..."):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "google/gemini-2.0-flash-exp:free",
                "messages": [
                    {"role": "user", "content": custom_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    reply = result['choices'][0]['message']['content']
                    
                    st.success("✅ Custom test berhasil!")
                    st.info(reply)
                    
                    # Tampilkan metadata
                    with st.expander("📊 Response Details"):
                        st.json(result)
                else:
                    st.error(f"Error: {response.status_code}")
                    st.code(response.text[:500])
                    
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.divider()
st.caption("""
🔧 **Tips:**
- Pastikan API key dimulai dengan `sk-or-`
- Key expired dalam 2 bulan (cek dashboard OpenRouter)
- Monitor usage di [OpenRouter Dashboard](https://openrouter.ai/activity)
""")

# ============================================
# BAGIAN UNTUK INTEGRASI KE APLIKASI OBAT
# ============================================
with st.expander("🩺 **Kode untuk Integrasi ke Aplikasi Obat**", expanded=False):
    st.write("Setelah test berhasil, gunakan kode ini di aplikasi obat Anda:")
    
    st.code("""# Di file testchat.py, tambahkan:
import requests

class OpenRouterTranslation:
    def __init__(self):
        self.api_key = st.secrets.get("OPENROUTER_API_KEY")
    
    def translate_to_indonesian(self, text):
        if not self.api_key or not text:
            return text
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "google/gemini-2.0-flash-exp:free",
                    "messages": [
                        {"role": "system", "content": "Anda penerjemah medis."},
                        {"role": "user", "content": f"Terjemahkan: {text}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        
        except:
            pass
        
        return text  # Fallback
""", language="python")
    
    st.write("""
    **Lalu di kelas `SimpleDrugAssistant`, ganti:**
    ```python
    class SimpleDrugAssistant:
        def __init__(self):
            self.fda_api = FDADrugAPI()
            self.translator = OpenRouterTranslation()  # GANTI INI
            # ... lanjutan kode
    ```
    """)

# Auto-run basic test on load
if 'auto_tested' not in st.session_state:
    st.session_state.auto_tested = True
    
    with st.spinner("🔄 Running initial test..."):
        time.sleep(1)  # Delay kecil untuk UX
        success, _ = test_basic_connection(api_key)
        
        if success:
            st.balloons()
