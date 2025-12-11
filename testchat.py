# ============================================
# FILE: testchat.py - BAGIAN YANG PERLU DIUBAH
# ============================================

import streamlit as st
import requests

# ============================================
# 1. OPENROUTER TRANSLATION SERVICE (SIMPLE)
# ============================================
class OpenRouterTranslator:
    """Service terjemahan menggunakan OpenRouter"""
    def __init__(self):
        # Ambil API key dari secrets
        self.api_key = st.secrets.get("OPENROUTER_API_KEY", "")
        
        if not self.api_key:
            st.warning("OpenRouter API key tidak ditemukan di secrets")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Cache sederhana
        self.cache = {}
    
    def translate_medical_text(self, text: str):
        """Terjemahkan teks medis ke Bahasa Indonesia"""
        if not self.api_key or not text:
            return text
        
        # Skip jika teks pendek
        if len(text) < 20:
            return text
        
        # Cek cache dulu
        cache_key = hash(text[:100])
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Prompt sederhana untuk terjemahan medis
            prompt = f"""Terjemahkan ke Bahasa Indonesia (pertahankan angka dan satuan):
            
"{text}"

Hasil:"""
            
            payload = {
                "model": "google/gemini-2.0-flash-exp:free",  # Model gratis Anda
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                translated = result['choices'][0]['message']['content'].strip()
                
                # Simpan ke cache
                self.cache[cache_key] = translated
                return translated
            
        except Exception as e:
            print(f"OpenRouter error: {e}")
        
        # Fallback: return text asli
        return text

# ============================================
# 2. UPDATE SIMPLE DRUG ASSISTANT
# ============================================
class SimpleDrugAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()  # Asumsi kelas FDA sudah ada
        self.translator = OpenRouterTranslator()  # GANTI KE OPENROUTER
        
        # Mapping obat (sama seperti sebelumnya)
        self.drug_mapping = {
            'paracetamol': 'acetaminophen',
            'amoxicillin': 'amoxicillin',
            'ibuprofen': 'ibuprofen',
            'omeprazole': 'omeprazole',
            'metformin': 'metformin',
            'aspirin': 'aspirin'
        }
    
    def ask_question(self, question: str):
        """Fungsi utama tanya-jawab"""
        # 1. Cari obat dalam pertanyaan
        drug_name = None
        for drug in self.drug_mapping:
            if drug in question.lower():
                drug_name = drug
                fda_name = self.drug_mapping[drug]
                break
        
        if not drug_name:
            return "❌ Obat tidak dikenali", None
        
        # 2. Ambil data FDA
        drug_info = self.fda_api.get_drug_info(fda_name)
        if not drug_info:
            return f"❌ Data {drug_name} tidak ditemukan", None
        
        # 3. Buat jawaban
        answer = self._generate_answer(drug_info)
        return answer, drug_info
    
    def _generate_answer(self, drug_info: dict):
        """Generate jawaban singkat"""
        # Terjemahkan field penting
        translated_fields = {}
        for field in ['indikasi', 'dosis', 'efek_samping']:
            if field in drug_info:
                translated_fields[field] = self.translator.translate_medical_text(
                    drug_info[field]
                )
        
        # Format jawaban sederhana
        answer = f"💊 **{drug_info.get('nama', 'Obat')}**\n\n"
        
        if 'indikasi' in translated_fields:
            answer += f"**Kegunaan:** {translated_fields['indikasi'][:150]}...\n\n"
        
        if 'dosis' in translated_fields:
            answer += f"**Dosis:** {translated_fields['dosis'][:100]}...\n\n"
        
        if 'efek_samping' in translated_fields:
            answer += f"**Efek Samping:** {translated_fields['efek_samping'][:100]}...\n\n"
        
        answer += "---\n"
        answer += "📋 **Sumber:** FDA Database\n"
        answer += "⚠️ **Konsultasikan ke dokter sebelum menggunakan.**"
        
        return answer

# ============================================
# 3. MAIN APP - MODIFIKASI MINIMAL
# ============================================
def main():
    # Initialize assistant
    assistant = SimpleDrugAssistant()
    
    # UI Streamlit (sama seperti sebelumnya)
    st.title("💊 Sistem Informasi Obat")
    
    # Input user
    question = st.text_input("Tanyakan tentang obat:", 
                           placeholder="contoh: paracetamol dosis berapa?")
    
    if question:
        with st.spinner("🔍 Mencari informasi..."):
            answer, details = assistant.ask_question(question)
            
            # Tampilkan jawaban
            st.markdown(answer)
            
            # Tampilkan detail jika ada
            if details:
                with st.expander("📄 Detail Informasi FDA"):
                    st.write(f"**Nama Generik:** {details.get('nama', '')}")
                    st.write(f"**Merek Dagang:** {details.get('merek_dagang', '')}")
                    
                    # Tampilkan info lengkap (opsional)
                    if 'indikasi' in details:
                        st.write("**Kegunaan Lengkap:**")
                        st.info(details['indikasi'][:300] + "...")
    
    # Footer
    st.caption("🔧 Powered by FDA API + OpenRouter AI")

if __name__ == "__main__":
    main()
