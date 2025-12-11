import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime
import time
import re
import json
import random

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Informasi Obat FDA",
    page_icon="💊",
    layout="wide"
)

# Setup Gemini API
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_available = True
except Exception as e:
    st.error(f"❌ Error konfigurasi Gemini API: {str(e)}")
    gemini_available = False

# ===========================================
# TRANSLATION SERVICE - SEDERHANA
# ===========================================
class TranslationService:
    def __init__(self):
        self.available = gemini_available
        self.translation_cache = {}
    
    def translate_to_indonesian(self, text: str):
        """Translate text ke Bahasa Indonesia - SEDERHANA"""
        if not self.available or not text or text == "Tidak tersedia":
            return text
        
        # Cache untuk hemat kuota
        cache_key = hash(text[:200])
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Skip teks pendek atau sudah Indonesia
            if len(text.strip()) < 20:
                return text
            
            # Terjemahan sederhana untuk bagian penting saja
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            # Prompt yang lebih singkat
            prompt = f"""
            Terjemahkan ke Bahasa Indonesia, pertahankan angka dan satuan:
            "{text}"
            
            Hasil:
            """
            
            response = model.generate_content(prompt)
            translated = response.text.strip()
            
            # Simpan ke cache
            self.translation_cache[cache_key] = translated
            return translated
            
        except Exception as e:
            print(f"Translation skipped: {e}")
            return text  # Return original jika error

# ===========================================
# FDA API - OPTIMIZED
# ===========================================
class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        self.cache = {}
    
    def get_drug_info(self, generic_name: str):
        """Ambil data obat dengan caching"""
        cache_key = generic_name.lower()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    drug_info = self._parse_fda_data(data['results'][0], generic_name)
                    self.cache[cache_key] = drug_info
                    return drug_info
        
        except Exception as e:
            print(f"FDA API error for {generic_name}: {e}")
        
        return None
    
    def _parse_fda_data(self, fda_data: dict, generic_name: str):
        """Parse data FDA - hanya ambil yang penting"""
        openfda = fda_data.get('openfda', {})
        
        def extract_field(field_name):
            value = fda_data.get(field_name, '')
            if isinstance(value, list) and value:
                return value[0][:300]  # Potong jadi 300 karakter
            elif value:
                return str(value)[:300]
            return None
        
        # Ambil hanya field yang diperlukan
        indications = extract_field('indications_and_usage') or extract_field('purpose')
        dosage = extract_field('dosage_and_administration') or extract_field('directions')
        side_effects = extract_field('adverse_reactions')
        warnings = extract_field('warnings')
        
        # Nama obat Indonesia
        drug_name = generic_name.title()
        if 'acetaminophen' in generic_name.lower():
            drug_name = 'Paracetamol'
        
        drug_info = {
            "nama": drug_name,
            "nama_generik": drug_name,
            "indikasi": indications or "Tidak tersedia",
            "dosis": dosage or "Tidak tersedia",
            "efek_samping": side_effects or "Tidak tersedia",
            "peringatan": warnings or "Tidak tersedia",
            "merek_dagang": ", ".join(openfda.get('brand_name', ['Tidak tersedia']))[:100],
            "bentuk": ", ".join(openfda.get('dosage_form', ['Tidak tersedia']))[:100]
        }
        
        return drug_info

# ===========================================
# SIMPLE DRUG ASSISTANT - SINGKAT & PADAT
# ===========================================
class SimpleDrugAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        
        # Mapping nama obat
        self.drug_mapping = {
            'paracetamol': 'acetaminophen',
            'panadol': 'acetaminophen',
            'sanmol': 'acetaminophen',
            'tempra': 'acetaminophen',
            
            'omeprazole': 'omeprazole',
            'prilosec': 'omeprazole',
            'losec': 'omeprazole',
            
            'amoxicillin': 'amoxicillin',
            'amoxilin': 'amoxicillin',
            
            'ibuprofen': 'ibuprofen',
            'proris': 'ibuprofen',
            
            'metformin': 'metformin',
            'glucophage': 'metformin',
            
            'aspirin': 'aspirin',
            'aspro': 'aspirin',
            
            'vitamin c': 'ascorbic acid',
            'redoxon': 'ascorbic acid',
            
            'salbutamol': 'albuterol',
            'ventolin': 'albuterol'
        }
    
    def extract_drug_name(self, query: str):
        """Ekstrak nama obat dari query"""
        query_lower = query.lower()
        
        for indo_name, fda_name in self.drug_mapping.items():
            if indo_name in query_lower:
                return indo_name, fda_name
        
        # Coba cari kata yang mirip nama obat
        common_drugs = ['paracetamol', 'amoxicillin', 'ibuprofen', 'omeprazole', 
                       'metformin', 'aspirin', 'vitamin', 'salbutamol']
        
        for drug in common_drugs:
            if drug in query_lower:
                fda_name = self.drug_mapping.get(drug, drug)
                return drug, fda_name
        
        return None, None
    
    def get_concise_answer(self, drug_info: dict):
        """Buat jawaban singkat dari data obat"""
        if not drug_info:
            return "❌ Maaf, informasi obat tidak ditemukan dalam database FDA."
        
        # Terjemahkan jika perlu
        translated_info = {}
        for key, value in drug_info.items():
            if value != "Tidak tersedia":
                translated_info[key] = self.translator.translate_to_indonesian(value)
            else:
                translated_info[key] = value
        
        # Format jawaban singkat
        answer = f"**💊 {translated_info['nama']}**\n\n"
        
        if translated_info['indikasi'] != "Tidak tersedia":
            # Ambil 2 kalimat pertama saja
            sentences = translated_info['indikasi'].split('. ')
            short_indications = '. '.join(sentences[:2])
            if len(sentences) > 2:
                short_indications += "..."
            answer += f"**Kegunaan:** {short_indications}\n\n"
        
        if translated_info['dosis'] != "Tidak tersedia":
            # Ambil informasi dosis penting
            dose_text = translated_info['dosis']
            # Cari angka dengan mg/tablet
            import re
            dose_match = re.search(r'(\d+\s*-\s*\d+\s*mg|\d+\s*mg.*per.*day|\d+\s*tablet)', dose_text, re.IGNORECASE)
            if dose_match:
                dose_info = dose_match.group(0)
                answer += f"**Dosis Umum:** {dose_info}\n\n"
            else:
                # Ambil 150 karakter pertama
                short_dose = dose_text[:150]
                if len(dose_text) > 150:
                    short_dose += "..."
                answer += f"**Dosis:** {short_dose}\n\n"
        
        if translated_info['efek_samping'] != "Tidak tersedia":
            # Sebutkan beberapa efek samping umum
            side_text = translated_info['efek_samping'].lower()
            common_effects = []
            
            if 'nausea' in side_text or 'mual' in side_text:
                common_effects.append("mual")
            if 'dizziness' in side_text or 'pusing' in side_text:
                common_effects.append("pusing")
            if 'headache' in side_text or 'sakit kepala' in side_text:
                common_effects.append("sakit kepala")
            if 'drowsiness' in side_text or 'kantuk' in side_text:
                common_effects.append("kantuk")
            if 'rash' in side_text or 'ruam' in side_text:
                common_effects.append("ruam kulit")
            
            if common_effects:
                answer += f"**Efek Samping Umum:** {', '.join(common_effects[:3])}\n\n"
        
        if translated_info['peringatan'] != "Tidak tersedia":
            # Ambil peringatan penting
            warning_text = translated_info['peringatan'].lower()
            if 'pregnancy' in warning_text or 'hamil' in warning_text:
                answer += "⚠️ **Peringatan:** Konsultasi dokter jika hamil atau menyusui.\n\n"
            elif 'liver' in warning_text or 'hati' in warning_text:
                answer += "⚠️ **Peringatan:** Hati-hati jika memiliki gangguan hati.\n"
            elif 'alcohol' in warning_text or 'alkohol' in warning_text:
                answer += "⚠️ **Peringatan:** Hindari konsumsi alkohol.\n"
        
        # Tambahkan disclaimer
        answer += "---\n"
        answer += "**ℹ️ Sumber:** Data dari U.S. Food and Drug Administration (FDA)\n"
        answer += "**⚠️ Peringatan:** Konsultasikan dengan dokter sebelum menggunakan obat."
        
        return answer, translated_info
    
    def ask_question(self, question: str):
        """Main function untuk menjawab pertanyaan"""
        # Ekstrak nama obat
        drug_name, fda_name = self.extract_drug_name(question)
        
        if not drug_name:
            available_drugs = ", ".join(['paracetamol', 'amoxicillin', 'ibuprofen', 
                                        'omeprazole', 'metformin', 'aspirin'])
            return f"❌ Obat tidak dikenali. Coba tanyakan tentang: {available_drugs}", None
        
        # Ambil data dari FDA
        drug_info = self.fda_api.get_drug_info(fda_name)
        
        if not drug_info:
            return f"❌ Data {drug_name} tidak ditemukan dalam database FDA.", None
        
        # Generate jawaban singkat
        answer, detailed_info = self.get_concise_answer(drug_info)
        
        return answer, detailed_info

# ===========================================
# FUNGSI UTAMA STREAMLIT
# ===========================================
def main():
    # Initialize assistant
    assistant = SimpleDrugAssistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Custom CSS minimal
    st.markdown("""
    <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 15px;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
        }
        .bot-message {
            background-color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 4px solid #4CAF50;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .drug-info-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            font-size: 0.9em;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("💊 Informasi Obat FDA")
    st.markdown("**Dapatkan informasi singkat tentang penggunaan, dosis, dan efek samping obat**")
    
    st.markdown("""
    <div style="background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0;">
    <strong>📋 Obat yang dikenali:</strong> Paracetamol, Amoxicillin, Ibuprofen, Omeprazole, Metformin, Aspirin, Vitamin C, Salbutamol
    </div>
    """, unsafe_allow_html=True)
    
    # Chat interface
    st.markdown("### 💬 Tanya tentang Obat")
    
    # Tampilkan chat history
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>Anda:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan info detail jika ada
                if "details" in message:
                    with st.expander("📄 Detail Informasi FDA"):
                        details = message["details"]
                        st.markdown(f"""
                        <div class="drug-info-box">
                            <strong>Nama Generik:</strong> {details.get('nama_generik', 'N/A')}<br>
                            <strong>Merek Dagang:</strong> {details.get('merek_dagang', 'N/A')}<br>
                            <strong>Bentuk Sediaan:</strong> {details.get('bentuk', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Info lengkap (tersembunyi)
                        col1, col2 = st.columns(2)
                        with col1:
                            if details.get('indikasi') != "Tidak tersedia":
                                st.markdown("**Kegunaan Lengkap:**")
                                st.info(details['indikasi'][:500])
                        
                        with col2:
                            if details.get('efek_samping') != "Tidak tersedia":
                                st.markdown("**Efek Samping Lengkap:**")
                                st.warning(details['efek_samping'][:500])
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Welcome message
        st.info("""
        **Contoh pertanyaan:**
        - "Paracetamol untuk apa?"
        - "Dosis amoxicillin berapa?"
        - "Efek samping ibuprofen?"
        - "Kegunaan omeprazole?"
        - "Peringatan penggunaan aspirin?"
        
        **Cukup tanyakan dalam bahasa Indonesia atau Inggris.**
        """)
    
    # Input form
    with st.form("drug_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Tanyakan tentang obat:",
                placeholder="Contoh: paracetamol dosis berapa?",
                key="user_input"
            )
        
        with col2:
            submit_btn = st.form_submit_button(
                "🔍 Cari", 
                use_container_width=True,
                type="primary"
            )
    
    if submit_btn and user_input:
        # Simpan pertanyaan user
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Proses pertanyaan
        with st.spinner("🔍 Mencari data dari FDA..."):
            answer, details = assistant.ask_question(user_input)
            
            # Simpan jawaban
            message_data = {
                "role": "bot",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            
            if details:
                message_data["details"] = details
            
            st.session_state.messages.append(message_data)
            
            # Simpan history
            st.session_state.conversation_history.append({
                'timestamp': datetime.now(),
                'question': user_input,
                'answer': answer[:200] + "..." if len(answer) > 200 else answer
            })
        
        st.rerun()
    
    # Tombol clear chat
    if st.session_state.messages:
        if st.button("🗑️ Hapus Percakapan", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
    
    # Footer dengan disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ PERINGATAN MEDIS:</strong> Informasi ini berasal dari database FDA AS dan untuk tujuan edukasi saja. 
    Selalu konsultasikan dengan dokter atau apoteker sebelum menggunakan obat apa pun. 
    Obat mungkin memiliki nama merek yang berbeda di Indonesia.
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("💡 **Tips:** Sistem ini memberikan informasi singkat. Untuk detail lengkap, konsultasikan dengan profesional kesehatan.")

if __name__ == "__main__":
    main()
