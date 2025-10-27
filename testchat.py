import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime
import time
import re

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

class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        self.drugs_cache = {}  # Cache untuk menyimpan data obat yang sudah diambil
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
                drug_info[field] = self.translator.translate_to_indonesian(drug_info[field])
        
        return drug_info
    
    def _rag_retrieve(self, query, top_k=3):
        """Retrieve relevant information menggunakan FDA API"""
        query_lower = query.lower()
        results = []
        
        # Daftar obat umum yang akan dicari
        common_drugs = [
            "omeprazole", "amoxicillin", "paracetamol", "ibuprofen", 
            "metformin", "atorvastatin", "simvastatin", "loratadine",
            "vitamin c", "aspirin", "lansoprazole", "esomeprazole"
        ]
        
        # Cari obat yang relevan dengan query
        for drug_name in common_drugs:
            score = 0
            
            # Exact name matching
            if drug_name in query_lower:
                score += 10
            
            # Partial name matching
            if drug_name.split()[0] in query_lower:  # untuk "vitamin c" -> "vitamin"
                score += 5
            
            # Question type matching
            question_keywords = {
                'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai'],
                'efek': ['efek samping', 'side effect', 'bahaya'],
                'kontraindikasi': ['kontra', 'tidak boleh', 'hindari'],
                'interaksi': ['interaksi', 'bereaksi dengan'],
                'indikasi': ['untuk apa', 'kegunaan', 'manfaat']
            }
            
            for key, keywords in question_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    score += 3
            
            if score > 0:
                # Fetch data dari FDA API
                drug_info = self._get_or_fetch_drug_info(drug_name)
                if drug_info:
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
                return "❌ Tidak ditemukan informasi yang relevan dalam database FDA. Coba tanyakan tentang obat-obat umum seperti: omeprazole, amoxicillin, paracetamol, dll.", []
            
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
            # Fallback ke response sederhana
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
        <p>"Dosis omeprazole?" | "Efek samping amoxicillin?" | "Interaksi obat paracetamol?"</p>
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
        placeholder="Contoh: Apa dosis omeprazole? Efek samping amoxicillin? Interaksi obat?",
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
    assistant.drugs_cache = {}  # Clear cache juga
    st.rerun()

# Informasi obat yang tersedia
st.sidebar.markdown("### 💊 Obat yang Tersedia")
st.sidebar.info("""
Sistem dapat mencari informasi tentang:
- Omeprazole
- Amoxicillin  
- Paracetamol
- Ibuprofen
- Metformin
- Atorvastatin
- Simvastatin
- Loratadine
- Vitamin C
- Aspirin
- Lansoprazole
- Esomeprazole
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

# ==================== EVALUATION SECTION ====================

def show_enhanced_evaluation():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Evaluasi FDA API")
    
    if st.sidebar.button("Tes Akurasi FDA API"):
        with st.spinner("Testing FDA API Accuracy..."):
            
            class EnhancedEvaluator:
                def __init__(self, assistant):
                    self.assistant = assistant
                    self.test_cases = [
                        {
                            'question': 'Apa dosis omeprazole untuk dewasa?',
                            'expected_keywords': ['mg', 'hari', 'sebelum', 'makan'],
                            'expected_drug': 'Omeprazole',
                            'category': 'dosis'
                        },
                        {
                            'question': 'Efek samping amoxicillin?',
                            'expected_keywords': ['diare', 'mual', 'ruam', 'alergi'],
                            'expected_drug': 'Amoxicillin', 
                            'category': 'keamanan'
                        },
                        {
                            'question': 'Untuk apa paracetamol digunakan?',
                            'expected_keywords': ['nyeri', 'demam', 'sakit', 'kepala'],
                            'expected_drug': 'Paracetamol',
                            'category': 'indikasi'
                        }
                    ]
                
                def evaluate(self):
                    results = []
                    for test in self.test_cases:
                        answer, sources = self.assistant.ask_question(test['question'])
                        
                        # Clean answer untuk analysis
                        clean_answer = self._clean_answer(answer)
                        
                        # Scoring
                        keyword_score = self._keyword_score(clean_answer, test['expected_keywords'])
                        drug_score = self._drug_score(sources, test['expected_drug'])
                        
                        # Found keywords detail
                        found_keywords = self._get_found_keywords(clean_answer, test['expected_keywords'])
                        missing_keywords = self._get_missing_keywords(clean_answer, test['expected_keywords'])
                        
                        results.append({
                            'Question': test['question'],
                            'Category': test['category'],
                            'Expected Drug': test['expected_drug'],
                            'Keyword Score': f"{keyword_score:.0%}",
                            'Drug Match': "✅" if drug_score == 1.0 else "❌",
                            'Found Keywords': ", ".join(found_keywords) if found_keywords else "None",
                            'Missing Keywords': ", ".join(missing_keywords) if missing_keywords else "None",
                            'Answer Preview': clean_answer[:100] + "..." if len(clean_answer) > 100 else clean_answer,
                            'Source': sources[0]['sumber'] if sources else 'No source'
                        })
                    
                    return results
                
                def _clean_answer(self, answer):
                    """Remove markdown formatting"""
                    import re
                    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
                    clean = re.sub(r'\*(.*?)\*', r'\1', clean)
                    clean = re.sub(r'`(.*?)`', r'\1', clean)
                    clean = re.sub(r'#+\s*', '', clean)
                    return clean.strip()
                
                def _keyword_score(self, answer, keywords):
                    """Keyword matching"""
                    answer_lower = answer.lower()
                    found_keywords = 0
                    
                    for keyword in keywords:
                        if keyword.lower() in answer_lower:
                            found_keywords += 1
                    
                    return found_keywords / len(keywords) if keywords else 0
                
                def _get_found_keywords(self, answer, keywords):
                    """Return list of found keywords"""
                    answer_lower = answer.lower()
                    found = []
                    for keyword in keywords:
                        if keyword.lower() in answer_lower:
                            found.append(keyword)
                    return found
                
                def _get_missing_keywords(self, answer, keywords):
                    """Return list of missing keywords"""
                    answer_lower = answer.lower()
                    missing = []
                    for keyword in keywords:
                        if keyword.lower() not in answer_lower:
                            missing.append(keyword)
                    return missing
                
                def _drug_score(self, sources, expected_drug):
                    """Check if correct drug is retrieved"""
                    if not sources:
                        return 0.0
                    return 1.0 if any(drug['nama'].lower() == expected_drug.lower() for drug in sources) else 0.0
            
            # Run evaluation
            evaluator = EnhancedEvaluator(assistant)
            results = evaluator.evaluate()
            
            # Display results
            st.subheader("📊 FDA API Accuracy Evaluation")
            
            # Summary metrics
            total_tests = len(results)
            keyword_scores = [float(r['Keyword Score'].strip('%'))/100 for r in results]
            avg_keyword_score = sum(keyword_scores) / total_tests
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Keyword Score", f"{avg_keyword_score:.1%}")
            with col2:
                drug_matches = sum(1 for r in results if r['Drug Match'] == "✅")
                st.metric("Drug Match Rate", f"{(drug_matches/total_tests):.1%}")
            with col3:
                st.metric("Data Source", "FDA API")
            
            # Detailed results
            for result in results:
                with st.expander(f"🧪 {result['Question']} ({result['Category']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Expected Drug:** {result['Expected Drug']}")
                        st.write(f"**Keyword Score:** {result['Keyword Score']}")
                        st.write(f"**Drug Match:** {result['Drug Match']}")
                        st.write(f"**Source:** {result['Source']}")
                    
                    with col2:
                        st.write(f"**✅ Found:** {result['Found Keywords']}")
                        st.write(f"**❌ Missing:** {result['Missing Keywords']}")
                    
                    st.write("**Answer:**")
                    st.info(result['Answer Preview'])

# Panggil evaluasi
show_enhanced_evaluation()
