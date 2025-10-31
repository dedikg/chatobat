import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime
import time
import re
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Tanya Jawab Informasi Obat - FDA API",
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

class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
    
    def get_drug_info(self, generic_name: str):
        """Ambil data obat langsung dari FDA API"""
        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    return self._parse_fda_data(data['results'][0], generic_name)
                else:
                    st.warning(f"⚠️ Data FDA tidak ditemukan untuk: {generic_name}")
            return None
                
        except Exception as e:
            st.error(f"Error FDA API: {e}")
            return None
    
    def _parse_fda_data(self, fda_data: dict, generic_name: str):
        """Parse data FDA menjadi format yang kita butuhkan"""
        openfda = fda_data.get('openfda', {})
        
        def get_field(field_name):
            value = fda_data.get(field_name, '')
            if isinstance(value, list) and value:
                return value[0]
            return value
        
        drug_info = {
            "nama": generic_name.title(),
            "nama_generik": generic_name.title(),
            "merek_dagang": ", ".join(openfda.get('brand_name', ['Tidak tersedia'])),
            "golongan": get_field('drug_class') or "Tidak tersedia",
            "indikasi": get_field('indications_and_usage') or "Tidak tersedia",
            "dosis_dewasa": get_field('dosage_and_administration') or "Tidak tersedia",
            "efek_samping": get_field('adverse_reactions') or "Tidak tersedia",
            "kontraindikasi": get_field('contraindications') or "Tidak tersedia",
            "interaksi": get_field('drug_interactions') or "Tidak tersedia",
            "peringatan": get_field('warnings') or "Tidak tersedia",
            "bentuk_sediaan": ", ".join(openfda.get('dosage_form', ['Tidak tersedia'])),
            "route_pemberian": ", ".join(openfda.get('route', ['Tidak tersedia'])),
            "sumber": "FDA API"
        }
        
        return drug_info

class TranslationService:
    def __init__(self):
        self.available = gemini_available
    
    def translate_to_indonesian(self, text: str):
        """Translate text ke Bahasa Indonesia menggunakan Gemini"""
        if not self.available or not text or text == "Tidak tersedia":
            return text
        
        try:
            # Skip translation jika teks sudah pendek atau mengandung banyak angka/dosis
            if len(text) < 50 or any(word in text.lower() for word in ['mg', 'ml', 'tablet', 'capsule', 'day']):
                return text
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Terjemahkan teks medis berikut ke Bahasa Indonesia dengan tetap mempertahankan makna medis yang akurat:
            
            {text}
            
            Hasil terjemahan:
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            st.error(f"Error translation: {e}")
            return text

class EnhancedDrugDetector:
    def __init__(self):
        # Expanded drug dictionary dengan berbagai nama dan sinonim
        self.drug_dictionary = {
            'paracetamol': ['paracetamol', 'acetaminophen', 'panadol', 'sanmol', 'tempra'],
            'omeprazole': ['omeprazole', 'prilosec', 'losec', 'omepron'],
            'amoxicillin': ['amoxicillin', 'amoxilin', 'amoxan', 'moxigra'],
            'ibuprofen': ['ibuprofen', 'proris', 'arthrifen', 'ibufar'],
            'metformin': ['metformin', 'glucophage', 'metfor', 'diabex'],
            'atorvastatin': ['atorvastatin', 'lipitor', 'atorva', 'tovast'],
            'simvastatin': ['simvastatin', 'zocor', 'simvor', 'lipostat'],
            'loratadine': ['loratadine', 'clarityne', 'loramine', 'allertine'],
            'aspirin': ['aspirin', 'aspro', 'aspilet', 'cardiprin'],
            'vitamin c': ['vitamin c', 'ascorbic acid', 'redoxon', 'enervon c'],
            'lansoprazole': ['lansoprazole', 'prevacid', 'lanzol', 'gastracid'],
            'esomeprazole': ['esomeprazole', 'nexium', 'esotrax', 'esomep'],
            'cefixime': ['cefixime', 'suprax', 'cefix', 'fixcef'],
            'cetirizine': ['cetirizine', 'zyrtec', 'cetrizin', 'allertec'],
            'dextromethorphan': ['dextromethorphan', 'dmp', 'dextro', 'valtus'],
            'ambroxol': ['ambroxol', 'mucosolvan', 'ambrox', 'broxol'],
            'salbutamol': ['salbutamol', 'ventolin', 'salbu', 'asmasolon']
        }
    
    def detect_drug_from_query(self, query: str):
        """Detect drug name from user query dengan matching yang lebih baik"""
        query_lower = query.lower()
        detected_drugs = []
        
        for drug_name, aliases in self.drug_dictionary.items():
            # Check semua alias
            for alias in aliases:
                if alias in query_lower:
                    detected_drugs.append({
                        'drug_name': drug_name,
                        'alias_found': alias,
                        'confidence': 'high' if alias == drug_name else 'medium'
                    })
                    break  # Stop setelah menemukan satu match
        
        return detected_drugs
    
    def get_all_available_drugs(self):
        """Get list of all available drugs"""
        return list(self.drug_dictionary.keys())

class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        self.drug_detector = EnhancedDrugDetector()
        self.drugs_cache = {}
        self.current_context = {}
        
    def _get_or_fetch_drug_info(self, drug_name: str):
        """Dapatkan data dari cache atau fetch dari FDA API"""
        drug_key = drug_name.lower()
        
        if drug_key in self.drugs_cache:
            return self.drugs_cache[drug_key]
        
        # Fetch dari FDA API
        drug_info = self.fda_api.get_drug_info(drug_name)
        
        if drug_info:
            # Translate fields yang penting
            drug_info = self._translate_drug_info(drug_info)
            self.drugs_cache[drug_key] = drug_info
        
        return drug_info
    
    def _translate_drug_info(self, drug_info: dict):
        """Translate field-field penting ke Bahasa Indonesia"""
        fields_to_translate = ['indikasi', 'dosis_dewasa', 'efek_samping', 'kontraindikasi', 'interaksi', 'peringatan']
        
        for field in fields_to_translate:
            if field in drug_info and drug_info[field] != "Tidak tersedia":
                translated = self.translator.translate_to_indonesian(drug_info[field])
                if translated != drug_info[field]:
                    drug_info[field] = translated
        
        return drug_info
    
    def _rag_retrieve(self, query, top_k=3):
        """Retrieve relevant information menggunakan FDA API"""
        query_lower = query.lower()
        results = []
        
        # Step 1: Detect drugs from query
        detected_drugs = self.drug_detector.detect_drug_from_query(query)
        
        if not detected_drugs:
            # Jika tidak detect, coba obat-obat umum
            common_drugs = self.drug_detector.get_all_available_drugs()
        else:
            # Prioritize detected drugs
            common_drugs = [drug['drug_name'] for drug in detected_drugs]
        
        # Step 2: Cari data untuk setiap drug yang relevan
        for drug_name in common_drugs[:top_k]:
            score = 0
            
            # Scoring berdasarkan relevance dengan query
            if drug_name in query_lower:
                score += 10
            
            # Check aliases
            aliases = self.drug_detector.drug_dictionary.get(drug_name, [])
            for alias in aliases:
                if alias in query_lower:
                    score += 8
                    break
            
            # Question type matching
            question_keywords = {
                'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai', 'dosis untuk', 'berapa mg'],
                'efek': ['efek samping', 'side effect', 'bahaya', 'efeknya', 'akibat'],
                'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan', 'kontraindikasi'],
                'interaksi': ['interaksi', 'bereaksi dengan', 'makanan', 'minuman', 'interaksinya'],
                'indikasi': ['untuk apa', 'kegunaan', 'manfaat', 'indikasi', 'guna', 'fungsi']
            }
            
            for key, keywords in question_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    score += 3
            
            if score > 0:
                # Fetch data dari FDA API
                drug_info = self._get_or_fetch_drug_info(drug_name)
                if drug_info and drug_info.get('indikasi') != "Tidak tersedia":
                    results.append({
                        'score': score,
                        'drug_info': drug_info,
                        'drug_id': drug_name
                    })
        
        # Sort by score dan ambil top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _build_rag_context(self, retrieved_results):
        """Build context untuk RAG generator dari data FDA"""
        if not retrieved_results:
            return "Tidak ada informasi yang relevan ditemukan dalam database FDA."
        
        context = "🔍 **INFORMASI OBAT DARI FDA:**\n\n"
        
        for i, result in enumerate(retrieved_results, 1):
            drug_info = result['drug_info']
            context += f"**OBAT {i}: {drug_info['nama']}**\n"
            context += f"- Golongan: {drug_info['golongan']}\n"
            context += f"- Indikasi: {drug_info['indikasi']}\n"
            context += f"- Dosis Dewasa: {drug_info['dosis_dewasa']}\n"
            context += f"- Efek Samping: {drug_info['efek_samping']}\n"
            context += f"- Kontraindikasi: {drug_info['kontraindikasi']}\n"
            context += f"- Interaksi: {drug_info['interaksi']}\n"
            if drug_info['peringatan'] != "Tidak tersedia":
                context += f"- Peringatan: {drug_info['peringatan']}\n"
            context += f"- Bentuk Sediaan: {drug_info['bentuk_sediaan']}\n"
            context += "\n"
        
        return context
    
    def ask_question(self, question):
        """Main RAG interface dengan FDA API"""
        try:
            # Step 1: Retrieve relevant information dari FDA API
            retrieved_results = self._rag_retrieve(question)
            
            if not retrieved_results:
                available_drugs = ", ".join(self.drug_detector.get_all_available_drugs()[:10])
                return f"❌ Tidak ditemukan informasi yang relevan dalam database FDA untuk pertanyaan Anda.\n\n💡 **Coba tanyakan tentang:** {available_drugs}", []
            
            # Step 2: Build context dari data FDA
            rag_context = self._build_rag_context(retrieved_results)
            
            # Step 3: Generate response dengan RAG
            answer = self._generate_rag_response(question, rag_context)
            
            # Step 4: Get sources
            sources = []
            seen_drug_names = set()
            
            for result in retrieved_results:
                drug_name = result['drug_info']['nama']
                if drug_name not in seen_drug_names:
                    sources.append(result['drug_info'])
                    seen_drug_names.add(drug_name)
            
            # Update context
            self._update_conversation_context(question, answer, sources)
            
            return answer, sources
            
        except Exception as e:
            st.error(f"Error dalam RAG system: {e}")
            return "Maaf, terjadi error dalam sistem. Silakan coba lagi.", []
    
    def _generate_rag_response(self, question, context):
        """Generate response menggunakan RAG pattern dengan Gemini"""
        if not gemini_available:
            return f"**Informasi dari FDA:**\n\n{context}"
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            # PERAN: Asisten Farmasi Profesional
            # TUGAS: Jawab pertanyaan tentang obat menggunakan informasi FDA yang disediakan
            # BAHASA: Bahasa Indonesia yang jelas dan mudah dipahami

            ## INFORMASI RESMI DARI FDA:
            {context}

            ## PERTANYAAN PENGGUNA:
            {question}

            ## INSTRUKSI:
            1. JAWAB BERDASARKAN INFORMASI FDA DI ATAS - jangan membuat informasi baru
            2. Fokus pada obat yang paling relevan dengan pertanyaan
            3. Jika informasi tidak lengkap, jelaskan apa yang tersedia dari FDA
            4. Sertakan peringatan penting dari data FDA
            5. Gunakan bahasa yang mudah dipahami pasien
            6. Jelaskan dalam Bahasa Indonesia
            7. Berikan jawaban yang langsung menjawab pertanyaan

            ## JAWABAN:
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            st.error(f"⚠️ Error AI: {e}")
            return f"**Informasi dari FDA:**\n\n{context}"
    
    def _update_conversation_context(self, question, answer, sources):
        """Update conversation context"""
        if sources:
            self.current_context = {
                'current_drug': sources[0]['nama'],
                'timestamp': datetime.now()
            }

# Initialize RAG assistant dengan FDA API
@st.cache_resource
def load_rag_assistant():
    return SimpleRAGPharmaAssistant()

def main():
    assistant = load_rag_assistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Custom CSS
    st.markdown("""
    <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            background-color: #fafafa;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #0078D4;
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 70%;
            margin-left: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .bot-message {
            background-color: white;
            color: #333;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 70%;
            margin-right: auto;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 0.75em;
            opacity: 0.7;
            margin-top: 5px;
            text-align: right;
        }
        .bot-message .message-time {
            text-align: left;
        }
        .fda-indicator {
            background-color: #e8f5e8;
            border: 1px solid #4caf50;
            border-radius: 8px;
            padding: 8px 12px;
            margin: 5px 0;
            font-size: 0.8em;
            color: #2e7d32;
        }
        .welcome-message {
            text-align: center;
            padding: 40px;
            color: #666;
            background: white;
            border-radius: 10px;
            border: 2px dashed #e0e0e0;
        }
        .drug-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("💊 Sistem Tanya Jawab Obat - FDA API")
    st.markdown("Sistem informasi obat dengan data langsung dari **FDA API** dan terjemahan menggunakan **Gemini AI**")

    # FDA API Indicator
    st.markdown("""
    <div class="fda-indicator">
        🏥 <strong>DATA RESMI FDA</strong> - Informasi obat langsung dari U.S. Food and Drug Administration
    </div>
    """, unsafe_allow_html=True)

    # Chat container
    st.markdown("### 💬 Percakapan")

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-message">
            <h3>👋 Selamat Datang di Asisten Obat FDA</h3>
            <p>Dapatkan informasi obat <strong>langsung dari database resmi FDA</strong> dengan terjemahan otomatis ke Bahasa Indonesia</p>
            <p><strong>💡 Contoh pertanyaan:</strong></p>
            <p>"Dosis paracetamol?" | "Efek samping amoxicillin?" | "Interaksi obat omeprazole?"</p>
            <p>"Untuk apa metformin digunakan?" | "Peringatan penggunaan ibuprofen?"</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div>{message["content"]}</div>
                    <div class="message-time">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div>{message["content"]}</div>
                    <div class="message-time">{message["timestamp"]} • Sumber: FDA API</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan sources jika ada
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Informasi Obat dari FDA"):
                        for drug in message["sources"]:
                            st.markdown(f"""
                            <div class="drug-card">
                                <h4>💊 {drug['nama']}</h4>
                                <p><strong>Golongan:</strong> {drug['golongan']}</p>
                                <p><strong>Merek Dagang:</strong> {drug['merek_dagang']}</p>
                                <p><strong>Indikasi:</strong> {drug['indikasi'][:150]}...</p>
                                <p><strong>Bentuk:</strong> {drug['bentuk_sediaan']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Tulis pertanyaan Anda tentang obat:",
            placeholder="Contoh: Apa dosis paracetamol? Efek samping amoxicillin? Interaksi obat?",
            key="user_input"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            submit_btn = st.form_submit_button(
                "🚀 Tanya FDA API", 
                use_container_width=True
            )
        
        with col_btn2:
            clear_btn = st.form_submit_button(
                "🗑️ Hapus Chat", 
                use_container_width=True
            )

    if submit_btn and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Get RAG response dari FDA API
        with st.spinner("🔍 Mengakses FDA API..."):
            answer, sources = assistant.ask_question(user_input)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'timestamp': datetime.now(),
                'question': user_input,
                'answer': answer,
                'sources': [drug['nama'] for drug in sources],
                'source': 'FDA API'
            })
            
            # Add bot message
            st.session_state.messages.append({
                "role": "bot", 
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        st.rerun()

    if clear_btn:
        st.session_state.messages = []
        st.session_state.conversation_history = []
        assistant.current_context = {}
        assistant.drugs_cache = {}
        st.rerun()

    # Informasi obat yang tersedia - DIPINDAH ke dalam main()
    st.sidebar.markdown("### 💊 Obat yang Tersedia")
    available_drugs = assistant.drug_detector.get_all_available_drugs()
    st.sidebar.info(f"""
    Sistem dapat mencari informasi tentang:
    {', '.join(available_drugs[:12])}
    ...dan banyak lagi
    """)

    # Medical disclaimer
    st.warning("""
    **⚠️ Peringatan Medis:** Informasi ini berasal dari database FDA AS dan untuk edukasi saja. 
    Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat. 
    Obat mungkin memiliki nama merek berbeda di Indonesia.
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Sistem Tanya Jawab Obat - Data dari FDA API dengan terjemahan Gemini AI"
        "</div>", 
        unsafe_allow_html=True
    )

# Panggil main function
if __name__ == "__main__":
    main()
