import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime
import hashlib

# Konfigurasi halaman
st.set_page_config(
    page_title="AI-PharmaAssist BPJS - Chatbot",
    page_icon="💊",
    layout="wide"
)

# Setup Gemini API - DENGAN KEAMANAN
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_available = True
except Exception as e:
    st.error(f"❌ Error konfigurasi Gemini API: {str(e)}")
    gemini_available = False

class EnhancedPharmaAssistant:
    def __init__(self):
        self.drugs_db = self._initialize_drug_database()
        
    def _initialize_drug_database(self):
        """Initialize expanded drug database dengan gejala yang benar"""
        drugs_db = {
            "paracetamol": {
                "nama": "Paracetamol",
                "golongan": "Analgesik dan Antipiretik",
                "indikasi": "Demam, nyeri ringan hingga sedang, sakit kepala, sakit gigi, nyeri otot, nyeri haid, migrain",
                "dosis_dewasa": "500-1000 mg setiap 4-6 jam, maksimal 4000 mg/hari",
                "dosis_anak": "10-15 mg/kgBB setiap 4-6 jam",
                "efek_samping": "Gangguan pencernaan, ruam kulit (jarang)",
                "kontraindikasi": "Gangguan hati berat, hipersensitif",
                "interaksi": "Alcohol meningkatkan risiko kerusakan hati",
                "merek_dagang": "Panadol, Sanmol, Tempra, Biogesic",
                "kategori": "analgesik, antipiretik, nyeri, demam",
                "gejala": "sakit kepala, demam, nyeri, sakit gigi, nyeri haid, pusing, panas, migrain"
            },
            "amoxicillin": {
                "nama": "Amoxicillin",
                "golongan": "Antibiotik Beta-Laktam", 
                "indikasi": "Infeksi bakteri saluran napas, telinga, kulit, saluran kemih, radang tenggorokan, sinusitis",
                "dosis_dewasa": "250-500 mg setiap 8 jam",
                "dosis_anak": "20-50 mg/kgBB/hari dibagi 3 dosis",
                "efek_samping": "Diare, mual, ruam kulit, reaksi alergi",
                "kontraindikasi": "Alergi penisilin, mononukleosis infeksiosa",
                "interaksi": "Mengurangi efektivitas kontrasepsi oral",
                "merek_dagang": "Amoxan, Kalmoxillin, Moxigra",
                "kategori": "antibiotik, infeksi, bakteri",
                "gejala": "infeksi, radang, demam karena infeksi, batuk berdahak, radang tenggorokan"
            },
            "omeprazole": {
                "nama": "Omeprazole", 
                "golongan": "Penghambat Pompa Proton (PPI)",
                "indikasi": "Tukak lambung, GERD, dispepsia, sindrom Zollinger-Ellison, maag, asam lambung, heartburn",
                "dosis_dewasa": "20-40 mg sekali sehari sebelum makan",
                "dosis_anak": "Tidak dianjurkan untuk anak <1 tahun",
                "efek_samping": "Sakit kepala, diare, mual, pusing",
                "kontraindikasi": "Hipersensitif, hamil trimester pertama",
                "interaksi": "Mengurangi absorpsi ketoconazole, itraconazole", 
                "merek_dagang": "Losec, Omepron, Gastruz",
                "kategori": "lambung, maag, gerd, asam",
                "gejala": "maag, asam lambung, nyeri ulu hati, heartburn, perut kembung, mual"
            },
            "ibuprofen": {
                "nama": "Ibuprofen",
                "golongan": "Anti-inflamasi nonsteroid (NSAID)",
                "indikasi": "Nyeri, inflamasi, demam, arthritis, dismenore, sakit kepala, migrain, nyeri otot, nyeri sendi",
                "dosis_dewasa": "200-400 mg setiap 4-6 jam, maksimal 1200 mg/hari",
                "dosis_anak": "5-10 mg/kgBB setiap 6-8 jam",
                "efek_samping": "Gangguan lambung, pusing, ruam kulit, tinitus",
                "kontraindikasi": "Ulkus peptikum, gangguan ginjal, hamil trimester ketiga",
                "interaksi": "Meningkatkan risiko perdarahan dengan antikoagulan",
                "merek_dagang": "Proris, Arthrifen, Ibufar",
                "kategori": "antiinflamasi, nyeri, demam, radang",
                "gejala": "sakit kepala, migrain, nyeri, demam, radang, kram haid, nyeri sendi, pegal"
            },
            "vitamin_c": {
                "nama": "Vitamin C",
                "golongan": "Vitamin dan Suplemen",
                "indikasi": "Suplementasi vitamin C, meningkatkan daya tahan tubuh, penyembuhan luka, sariawan, flu",
                "dosis_dewasa": "500-1000 mg per hari",
                "dosis_anak": "sesuai kebutuhan, konsultasi dokter",
                "efek_samping": "Diare pada dosis tinggi, gangguan pencernaan",
                "kontraindikasi": "Hipersensitif",
                "interaksi": "Dapat mempengaruhi efektivitas beberapa obat kemoterapi",
                "merek_dagang": "Redoxon, Enervon C, Holisticare Ester C",
                "kategori": "vitamin, suplemen, imunitas",
                "gejala": "daya tahan tubuh lemah, sariawan, pemulihan sakit, lelah, flu"
            },
            "loratadine": {
                "nama": "Loratadine",
                "golongan": "Antihistamin Generasi Kedua",
                "indikasi": "Rinitis alergi, urtikaria, alergi kulit, biduran, gatal-gatal, bersin-bersin, rhinitis",
                "dosis_dewasa": "10 mg sekali sehari",
                "dosis_anak": "5 mg sekali sehari (usia 6-12 tahun)",
                "efek_samping": "Mengantuk (jarang), sakit kepala, mulut kering",
                "kontraindikasi": "Hipersensitif, anak <6 tahun",
                "interaksi": "Erythromycin, ketoconazole dapat meningkatkan kadar loratadine",
                "merek_dagang": "Clarityne, Loramine, Allertine",
                "kategori": "alergi, antihistamin, gatal",
                "gejala": "alergi, gatal, bersin, pilek alergi, biduran, ruam kulit, hidung tersumbat"
            },
            "simvastatin": {
                "nama": "Simvastatin",
                "golongan": "Statin (Penurun Kolesterol)",
                "indikasi": "Hiperkolesterolemia, pencegahan penyakit kardiovaskular, kolesterol tinggi, trigliserida tinggi",
                "dosis_dewasa": "10-40 mg sekali sehari malam hari",
                "dosis_anak": "Tidak dianjurkan untuk anak",
                "efek_samping": "Nyeri otot, gangguan hati, sakit kepala",
                "kontraindikasi": "Penyakit hati aktif, hamil, menyusui",
                "interaksi": "Eritromisin, antijamur, grapefruit juice",
                "merek_dagang": "Zocor, Simvor, Lipostat",
                "kategori": "kolesterol, statin, jantung",
                "gejala": "kolesterol tinggi, lemak darah tinggi, risiko jantung"
            }
        }
        return drugs_db
    
    def _calculate_similarity_score(self, query, drug_info):
        """Enhanced semantic similarity scoring dengan symptom mapping"""
        query = query.lower()
        score = 0
        
        # 1. Direct symptom matching (HIGHEST PRIORITY)
        if 'gejala' in drug_info and drug_info['gejala']:
            symptoms = drug_info['gejala'].lower().split(',')
            for symptom in symptoms:
                symptom_clean = symptom.strip()
                if symptom_clean and symptom_clean in query:
                    score += 5
        
        # 2. Direct drug name match
        if drug_info['nama'].lower() in query:
            score += 5
            
        # 3. Indication keyword matching
        indication_lower = drug_info['indikasi'].lower()
        indication_keywords = [kw.strip() for kw in indication_lower.split(',')]
        for keyword in indication_keywords:
            if keyword and keyword in query:
                score += 3
        
        # 4. Brand name match
        for merek in drug_info['merek_dagang'].lower().split(','):
            merek_clean = merek.strip()
            if merek_clean and merek_clean in query:
                score += 3
        
        # 5. Category matching
        if 'kategori' in drug_info and drug_info['kategori']:
            categories = drug_info['kategori'].lower().split(',')
            for category in categories:
                category_clean = category.strip()
                if category_clean and category_clean in query:
                    score += 2
        
        return score
    
    def _fallback_symptom_search(self, query, top_k=2):
        """Fallback search untuk query gejala"""
        symptom_drug_mapping = {
            'sakit kepala': ['paracetamol', 'ibuprofen'],
            'pusing': ['paracetamol'],
            'demam': ['paracetamol', 'ibuprofen'],
            'panas': ['paracetamol'],
            'pilek': ['loratadine'],
            'alergi': ['loratadine'],
            'maag': ['omeprazole'],
            'kolesterol': ['simvastatin'],
            'nyeri': ['paracetamol', 'ibuprofen'],
            'sakit': ['paracetamol', 'ibuprofen'],
            'radang': ['ibuprofen'],
            'infeksi': ['amoxicillin'],
            'bakteri': ['amoxicillin'],
            'vitamin': ['vitamin_c'],
            'imun': ['vitamin_c'],
            'daya tahan': ['vitamin_c'],
            'flu': ['vitamin_c'],
            'migrain': ['paracetamol', 'ibuprofen'],
            'pegal': ['paracetamol', 'ibuprofen']
        }
        
        results = []
        query_lower = query.lower()
        
        for symptom, drug_ids in symptom_drug_mapping.items():
            if symptom in query_lower:
                for drug_id in drug_ids:
                    if drug_id in self.drugs_db:
                        # Cek agar tidak duplicate
                        if not any(r['drug_id'] == drug_id for r in results):
                            results.append({
                                'score': 4,
                                'drug_info': self.drugs_db[drug_id],
                                'drug_id': drug_id
                            })
        
        return results[:top_k]
    
    def semantic_search(self, query, top_k=3):
        """Enhanced semantic search dengan symptom understanding"""
        results = []
        
        for drug_id, drug_info in self.drugs_db.items():
            score = self._calculate_similarity_score(query, drug_info)
            
            if score > 0:
                results.append({
                    'score': score,
                    'drug_info': drug_info,
                    'drug_id': drug_id
                })
        
        # FALLBACK: Jika tidak ada hasil, cari berdasarkan gejala
        if not results:
            results = self._fallback_symptom_search(query, top_k)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        final_drugs = [result['drug_info'] for result in results[:top_k]]
        
        return final_drugs
    
    def ask_question(self, question):
        """Enhanced RAG dengan Gemini"""
        # Semantic search for relevant drugs
        relevant_drugs = self.semantic_search(question)
        
        if not relevant_drugs:
            available_drugs = ", ".join([drug['nama'] for drug in self.drugs_db.values()])
            return f"Maaf, tidak ditemukan informasi tentang obat tersebut dalam database kami. Coba tanyakan tentang: {available_drugs}.", []
        
        # Prepare enhanced context for Gemini
        context = "INFORMASI OBAT YANG RELEVAN:\n"
        for i, drug in enumerate(relevant_drugs, 1):
            context += f"""
            OBAT {i}:
            - NAMA: {drug['nama']}
            - MEREK: {drug['merek_dagang']}
            - GOLONGAN: {drug['golongan']}
            - INDIKASI: {drug['indikasi']}
            - DOSIS DEWASA: {drug['dosis_dewasa']}
            - DOSIS ANAK: {drug['dosis_anak']}
            - EFEK SAMPING: {drug['efek_samping']}
            - KONTRAINDIKASI: {drug['kontraindikasi']}
            - INTERAKSI: {drug['interaksi']}
            - GEJALA: {drug.get('gejala', 'Tidak tersedia')}
            """
        
        try:
            if gemini_available:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = f"""
                Anda adalah asisten farmasi BPJS Kesehatan yang profesional.
                
                INFORMASI OBAT YANG TERSEDIA:
                {context}
                
                PERTANYAAN PASIEN: {question}
                
                INSTRUKSI:
                1. Jawab pertanyaan dengan AKURAT berdasarkan informasi obat di atas
                2. Gunakan bahasa Indonesia yang JELAS dan mudah dipahami
                3. Jika informasi tidak tersedia, jangan membuat-buat jawaban
                4. Sertakan nama obat yang relevan dalam jawaban
                5. Tetap singkat namun informatif
                6. Format jawaban dengan rapi dan mudah dibaca
                
                JAWABAN:
                """
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=800,
                        top_p=0.8
                    )
                )
                return response.text, relevant_drugs
            else:
                # Fallback to manual answer
                return self._generate_manual_answer(question, relevant_drugs), relevant_drugs
            
        except Exception as e:
            st.error(f"⚠️ Gemini API Error: {e}")
            # Fallback to manual answer
            return self._generate_manual_answer(question, relevant_drugs), relevant_drugs
    
    def _generate_manual_answer(self, question, drugs):
        """Manual answer fallback"""
        answer_parts = [f"**Untuk: '{question}'**"]
        
        for drug in drugs:
            answer_parts.append(f"💊 **{drug['nama']}**")
            answer_parts.append(f"• Indikasi: {drug['indikasi']}")
            answer_parts.append(f"• Dosis: {drug['dosis_dewasa']}")
            if 'gejala' in drug and drug['gejala']:
                answer_parts.append(f"• Gejala Terkait: {drug['gejala']}")
        
        return "\n".join(answer_parts)

# Initialize enhanced assistant
@st.cache_resource
def load_assistant():
    return EnhancedPharmaAssistant()

assistant = load_assistant()

# Initialize session state untuk chat
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Custom CSS untuk tampilan chatbot
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
    .quick-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 15px 0;
    }
    .quick-question-btn {
        background-color: #f0f2f6;
        border: 1px solid #d0d0d0;
        border-radius: 16px;
        padding: 6px 12px;
        font-size: 0.85em;
        cursor: pointer;
        transition: all 0.3s;
    }
    .quick-question-btn:hover {
        background-color: #0078D4;
        color: white;
    }
    .sources-badge {
        background-color: #e8f4fd;
        border: 1px solid #b3e0ff;
        border-radius: 8px;
        padding: 6px 10px;
        margin: 8px 0;
        font-size: 0.8em;
    }
    .welcome-message {
        text-align: center;
        padding: 30px;
        color: #666;
        background: white;
        border-radius: 10px;
        border: 2px dashed #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header dengan logo yang diperbaiki
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='text-align: center; font-size: 48px;'>💊</h1>", unsafe_allow_html=True)
with col2:
    st.title("AI-PharmaAssist BPJS")
    st.markdown("**Chatbot Informasi Obat BPJS Kesehatan**")

st.markdown("---")

# Layout utama
col_chat, col_info = st.columns([2, 1])

with col_chat:
    # Container chat
    st.subheader("💬 Chat dengan Asisten Farmasi")
    
    # Quick questions
    st.markdown("**🎯 Pertanyaan Cepat:**")
    quick_questions = [
        "Obat untuk sakit kepala?",
        "Apa dosis amoxicillin?",
        "Obat untuk maag?",
        "Vitamin untuk daya tahan tubuh?",
        "Efek samping simvastatin?",
        "Interaksi obat alergi?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(question, use_container_width=True, key=f"quick_{i}"):
                # Add user message
                st.session_state.messages.append({
                    "role": "user", 
                    "content": question,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Get bot response
                with st.spinner("🔄 Mencari informasi..."):
                    answer, sources = assistant.ask_question(question)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now(),
                        'question': question,
                        'answer': answer,
                        'sources': [drug['nama'] for drug in sources]
                    })
                    
                    # Add bot message
                    st.session_state.messages.append({
                        "role": "bot", 
                        "content": answer,
                        "sources": sources,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                
                st.rerun()
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-message">
            <h3>👋 Selamat Datang di AI-PharmaAssist!</h3>
            <p>Silakan tanyakan informasi tentang obat-obatan BPJS Kesehatan</p>
            <p><small>Contoh: "Obat untuk sakit kepala?", "Dosis paracetamol?", "Efek samping amoxicillin?"</small></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
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
                    with st.expander("📚 Lihat Sumber Informasi"):
                        for drug in message["sources"]:
                            st.markdown(f"""
                            <div class="sources-badge">
                                <strong>💊 {drug['nama']}</strong><br>
                                <small>{drug['golongan']}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Tulis pertanyaan Anda:",
            placeholder="Contoh: Obat untuk sakit kepala? Apa dosis amoxicillin? Bolehkah ibu hamil minum obat alergi?",
            height=80,
            key="user_input"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            submit_btn = st.form_submit_button("🚀 Kirim Pertanyaan", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.form_submit_button("🗑️ Hapus Chat", use_container_width=True)
    
    if submit_btn and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Get bot response
        with st.spinner("🔍 Mencari informasi obat..."):
            answer, sources = assistant.ask_question(user_input)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'timestamp': datetime.now(),
                'question': user_input,
                'answer': answer,
                'sources': [drug['nama'] for drug in sources]
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
        st.rerun()

with col_info:
    st.subheader("ℹ️ Informasi Sistem")
    
    # Status sistem
    st.markdown("### 🔧 Status Sistem")
    if gemini_available:
        st.success("✅ Gemini AI: Terhubung")
    else:
        st.warning("⚠️ Gemini AI: Mode Fallback")
    
    st.metric("💊 Obat dalam Database", len(assistant.drugs_db))
    st.metric("💬 Pesan dalam Chat", len(st.session_state.messages))
    
    # Database info
    with st.expander("📊 Database Obat"):
        for drug_id, drug_info in assistant.drugs_db.items():
            st.write(f"• **{drug_info['nama']}** - {drug_info['golongan']}")
    
    # Medical disclaimer
    st.markdown("""
    ### ⚠️ Peringatan Medis
    
    **Informasi ini untuk edukasi dan referensi saja.**
    
    Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat. Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
    
    ---
    **💊 AI-PharmaAssist BPJS**  
    *Powered by Gemini AI & Enhanced RAG*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "💊 AI-PharmaAssist BPJS Kesehatan - Healthkathon 2025 | "
    "Chatbot Informasi Obat dengan Enhanced RAG"
    "</div>", 
    unsafe_allow_html=True
)
