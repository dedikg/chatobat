import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem RAG Tanya Jawab Informasi Obat",
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

class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.drugs_db = self._initialize_drug_database()
        self.current_context = {}
        
    def _initialize_drug_database(self):
        """Initialize comprehensive drug database"""
        drugs_db = {
            "paracetamol": {
                "nama": "Paracetamol",
                "golongan": "Analgesik dan Antipiretik",
                "indikasi": "Demam, nyeri ringan hingga sedang, sakit kepala, sakit gigi, nyeri otot, nyeri haid, migrain",
                "dosis_dewasa": "500-1000 mg setiap 4-6 jam, maksimal 4000 mg/hari",
                "dosis_anak": "10-15 mg/kgBB setiap 4-6 jam",
                "efek_samping": "Gangguan pencernaan, ruam kulit (jarang), hepatotoksik pada overdosis",
                "kontraindikasi": "Gangguan hati berat, hipersensitif, penyakit ginjal berat",
                "interaksi": "Alcohol meningkatkan risiko kerusakan hati, warfarin meningkatkan risiko perdarahan",
                "merek_dagang": "Panadol, Sanmol, Tempra, Biogesic",
                "kategori": "analgesik, antipiretik, nyeri, demam",
                "gejala": "sakit kepala, demam, nyeri, sakit gigi, nyeri haid, pusing, panas, migrain",
                "peringatan": "Jangan melebihi dosis maksimal, hindari alkohol, hati-hati pada pasien gangguan hati"
            },
            "amoxicillin": {
                "nama": "Amoxicillin",
                "golongan": "Antibiotik Beta-Laktam", 
                "indikasi": "Infeksi bakteri saluran napas, telinga, kulit, saluran kemih, radang tenggorokan, sinusitis, pneumonia",
                "dosis_dewasa": "250-500 mg setiap 8 jam atau 875 mg setiap 12 jam",
                "dosis_anak": "20-50 mg/kgBB/hari dibagi 3 dosis",
                "efek_samping": "Diare, mual, ruam kulit, reaksi alergi, kandidiasis",
                "kontraindikasi": "Alergi penisilin, mononukleosis infeksiosa, riwayat penyakit hati",
                "interaksi": "Mengurangi efektivitas kontrasepsi oral, probenesid meningkatkan kadar amoxicillin",
                "merek_dagang": "Amoxan, Kalmoxillin, Moxigra, Amoxiclav",
                "kategori": "antibiotik, infeksi, bakteri",
                "gejala": "infeksi, radang, demam karena infeksi, batuk berdahak, radang tenggorokan, sinusitis",
                "peringatan": "Lengkapi seluruh regimen antibiotik, perhatikan reaksi alergi"
            },
            "omeprazole": {
                "nama": "Omeprazole", 
                "golongan": "Penghambat Pompa Proton (PPI)",
                "indikasi": "Tukak lambung, GERD, dispepsia, sindrom Zollinger-Ellison, maag, asam lambung, heartburn",
                "dosis_dewasa": "20-40 mg sekali sehari sebelum makan",
                "dosis_anak": "Tidak dianjurkan untuk anak <1 tahun",
                "efek_samping": "Sakit kepala, diare, mual, pusing, defisiensi magnesium",
                "kontraindikasi": "Hipersensitif, hamil trimester pertama",
                "interaksi": "Mengurangi absorpsi ketoconazole, itraconazole, clopidogrel", 
                "merek_dagang": "Losec, Omepron, Gastruz",
                "kategori": "lambung, maag, gerd, asam",
                "gejala": "maag, asam lambung, nyeri ulu hati, heartburn, perut kembung, mual",
                "peringatan": "Gunakan sebelum makan, tidak untuk penggunaan jangka panjang tanpa pengawasan dokter"
            }
        }
        return drugs_db
    
    def _rag_retrieve(self, query, top_k=3):
        """Retrieve relevant information menggunakan semantic search sederhana"""
        query_lower = query.lower()
        results = []
        
        for drug_id, drug_info in self.drugs_db.items():
            score = 0
            
            # Drug name matching (high priority)
            if drug_info['nama'].lower() in query_lower:
                score += 10
            
            # Brand name matching
            for merek in drug_info['merek_dagang'].lower().split(','):
                merek_clean = merek.strip()
                if merek_clean and merek_clean in query_lower:
                    score += 8
            
            # Symptom matching
            if 'gejala' in drug_info and drug_info['gejala']:
                symptoms = drug_info['gejala'].lower().split(',')
                for symptom in symptoms:
                    symptom_clean = symptom.strip()
                    if symptom_clean and symptom_clean in query_lower:
                        score += 5
            
            # Indication keyword matching
            indication_lower = drug_info['indikasi'].lower()
            indication_keywords = [kw.strip() for kw in indication_lower.split(',')]
            for keyword in indication_keywords:
                if keyword and keyword in query_lower:
                    score += 3
            
            # Category matching
            if 'kategori' in drug_info and drug_info['kategori']:
                categories = drug_info['kategori'].lower().split(',')
                for category in categories:
                    category_clean = category.strip()
                    if category_clean and category_clean in query_lower:
                        score += 2
            
            # Question type matching
            follow_up_keywords = {
                'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai', 'dosis untuk'],
                'efek': ['efek samping', 'side effect', 'bahaya', 'efeknya'],
                'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan', 'kontraindikasi'],
                'interaksi': ['interaksi', 'bereaksi dengan', 'makanan', 'minuman', 'interaksinya'],
                'indikasi': ['untuk apa', 'kegunaan', 'manfaat', 'indikasi', 'guna']
            }
            
            for key, keywords in follow_up_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    if key == 'dosis' and drug_info.get('dosis_dewasa'):
                        score += 4
                    elif key in drug_info and drug_info[key]:
                        score += 4
            
            if score > 0:
                results.append({
                    'score': score,
                    'drug_info': drug_info,
                    'drug_id': drug_id
                })
        
        # Sort by score dan ambil top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _build_rag_context(self, retrieved_results):
        """Build context untuk RAG generator"""
        if not retrieved_results:
            return "Tidak ada informasi yang relevan ditemukan dalam database."
        
        context = "🔍 **INFORMASI OBAT YANG RELEVAN:**\n\n"
        
        for i, result in enumerate(retrieved_results, 1):
            drug_info = result['drug_info']
            context += f"**OBAT {i}: {drug_info['nama']}**\n"
            context += f"- Golongan: {drug_info['golongan']}\n"
            context += f"- Indikasi: {drug_info['indikasi']}\n"
            context += f"- Dosis Dewasa: {drug_info['dosis_dewasa']}\n"
            context += f"- Dosis Anak: {drug_info['dosis_anak']}\n"
            context += f"- Efek Samping: {drug_info['efek_samping']}\n"
            context += f"- Kontraindikasi: {drug_info['kontraindikasi']}\n"
            context += f"- Interaksi: {drug_info['interaksi']}\n"
            if 'peringatan' in drug_info:
                context += f"- Peringatan: {drug_info['peringatan']}\n"
            context += "\n"
        
        return context
    
    def ask_question(self, question):
        """Main RAG interface - FIXED VERSION"""
        try:
            # Step 1: Retrieve relevant information
            retrieved_results = self._rag_retrieve(question)
            
            if not retrieved_results:
                available_drugs = ", ".join([drug['nama'] for drug in self.drugs_db.values()])
                return f"❌ Tidak ditemukan informasi yang relevan. Coba tanyakan tentang: {available_drugs}", []
            
            # Step 2: Build context
            rag_context = self._build_rag_context(retrieved_results)
            
            # Step 3: Generate response dengan RAG
            answer = self._generate_rag_response(question, rag_context)
            
            # Step 4: Get sources - SIMPLE AND SAFE APPROACH
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
        """Generate response menggunakan RAG pattern"""
        if not gemini_available:
            # Fallback ke response sederhana
            return f"Sistem RAG menemukan informasi berikut:\n\n{context}"
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            # PERAN: Asisten Farmasi Profesional
            # TUGAS: Jawab pertanyaan tentang obat menggunakan informasi yang disediakan
            # STYLE: Bahasa Indonesia yang jelas, profesional, dan mudah dipahami

            ## INFORMASI OBAT YANG RELEVAN:
            {context}

            ## PERTANYAAN PENGGUNA:
            {question}

            ## INSTRUKSI:
            1. JAWAB BERDASARKAN INFORMASI DI ATAS - jangan membuat informasi baru
            2. Fokus pada obat yang paling relevan dengan pertanyaan
            3. Jika informasi tidak lengkap, jelaskan apa yang tersedia
            4. Sertakan peringatan penting jika ada
            5. Gunakan format yang mudah dibaca
            6. Jelaskan dalam bahasa yang pasien mudah pahami

            ## JAWABAN:
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            st.error(f"⚠️ Error AI: {e}")
            return f"Sistem RAG menemukan informasi berikut:\n\n{context}"
    
    def _update_conversation_context(self, question, answer, sources):
        """Update conversation context"""
        if sources:
            self.current_context = {
                'current_drug': sources[0]['nama'],
                'timestamp': datetime.now()
            }

# Initialize RAG assistant
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
    .rag-indicator {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 5px 0;
        font-size: 0.8em;
        color: #1976d2;
    }
    .welcome-message {
        text-align: center;
        padding: 40px;
        color: #666;
        background: white;
        border-radius: 10px;
        border: 2px dashed #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="centered-title">💊 Sistem Tanya Jawab Informasi Obat</h1>', unsafe_allow_html=True)
st.markdown('<p class="centered-subtitle">Implementasi RAG untuk Sistem Tanya Jawab Informasi Obat Berbasis AI</p>', unsafe_allow_html=True)

# RAG Indicator
# st.markdown("""
# <div class="rag-indicator">
#     🚀 <strong>SISTEM RAG AKTIF</strong> - Menggunakan Retrieval-Augmented Generation untuk jawaban yang lebih akurat
# </div>
# """, unsafe_allow_html=True)

# Chat container
# st.markdown("### 💬 Percakapan")
# st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-message">
        <h3>👋 Selamat Datang di Asisten Obat AI</h3>
        <p>Silahkan tanyakan terkait informasi Obat-obatan</p>
        <p><strong>Contoh pertanyaan:</strong></p>
        <p>"Dosis paracetamol untuk dewasa?" | "Efek samping amoxicillin?" | "Interaksi obat omeprazole?"</p>
    </div>
    """, unsafe_allow_html=True)
else:
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
                <div class="message-time">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan sources jika ada
            if "sources" in message and message["sources"]:
                with st.expander("📚 Informasi Obat"):
                    for drug in message["sources"]:
                        st.write(f"• **{drug['nama']}** - {drug['golongan']}")

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
            "🚀 Tanya dengan RAG", 
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
    
    # Get RAG response
    with st.spinner("🔍 RAG System: Retrieving information..."):
        answer, sources = assistant.ask_question(user_input)
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            'timestamp': datetime.now(),
            'question': user_input,
            'answer': answer,
            'sources': [drug['nama'] for drug in sources],
            'rag_used': True
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
    st.rerun()

# Footer dengan penjelasan RAG
st.markdown("---")
# st.markdown("""
# ### 🔍 Tentang Sistem RAG
# **Retrieval-Augmented Generation (RAG)** adalah teknologi AI yang:
# 1. **Retrieve** - Mencari informasi relevan dari database obat
# 2. **Augment** - Memperkaya konteks dengan informasi yang ditemukan  
# 3. **Generate** - Menghasilkan jawaban yang akurat berdasarkan informasi terpercaya

# ✅ **Keunggulan:** Jawaban lebih akurat, terkini, dan dapat dipertanggungjawabkan
# """)

# Medical disclaimer
st.warning("""
**⚠️ Peringatan Medis:** Informasi ini untuk edukasi dan referensi saja. 
Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat. 
Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
""")
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Tugas Kuliah Sistem Biomedis - Implementasi RAG untuk Sistem Tanya Jawab Informasi Obat"
    "</div>", 
    unsafe_allow_html=True)
