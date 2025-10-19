import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime
import hashlib
import re

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
    
    def _extract_drug_from_context(self, query):
        """Extract drug name from conversation context"""
        if not st.session_state.get('conversation_history'):
            return None
            
        # Ambil percakapan terakhir
        last_conversations = st.session_state.conversation_history[-3:]  # 3 pesan terakhir
        
        # Cari referensi obat dalam percakapan sebelumnya
        drug_keywords = list(self.drugs_db.keys()) + [drug['nama'].lower() for drug in self.drugs_db.values()]
        
        for conv in reversed(last_conversations):
            # Cek di pertanyaan
            question_lower = conv['question'].lower()
            for drug_key in drug_keywords:
                if drug_key in question_lower:
                    return drug_key
            
            # Cek di jawaban (sources)
            if 'sources' in conv and conv['sources']:
                return conv['sources'][0].lower()
        
        return None
    
    def _enhance_query_with_context(self, query):
        """Enhance query dengan konteks percakapan sebelumnya"""
        enhanced_query = query
        
        # Deteksi pertanyaan lanjutan
        follow_up_indicators = [
            'berapa', 'bagaimana', 'apa', 'bolehkah', 'bisa', 'untuk', 'dewas', 'anak',
            'dosis', 'efek', 'samping', 'kontra', 'interaksi', 'indikasi'
        ]
        
        is_follow_up = any(indicator in query.lower() for indicator in follow_up_indicators)
        
        if is_follow_up and st.session_state.get('conversation_history'):
            # Cari obat dari konteks sebelumnya
            context_drug = self._extract_drug_from_context(query)
            
            if context_drug:
                # Jika ditemukan obat dari konteks, tambahkan ke query
                enhanced_query = f"{query} {context_drug}"
                st.sidebar.write(f"🔍 CONTEXT: Enhanced query '{query}' -> '{enhanced_query}'")
        
        return enhanced_query
    
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
        
        # 6. Contextual matching untuk pertanyaan lanjutan
        follow_up_keywords = {
            'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai'],
            'efek': ['efek samping', 'side effect', 'bahaya'],
            'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan'],
            'interaksi': ['interaksi', 'bereaksi dengan', 'makanan', 'minuman'],
            'indikasi': ['untuk apa', 'kegunaan', 'manfaat']
        }
        
        for key, keywords in follow_up_keywords.items():
            if any(kw in query for kw in keywords):
                # Beri bonus score jika drug_info memiliki field yang relevan
                if key == 'dosis' and drug_info.get('dosis_dewasa'):
                    score += 2
                elif key in drug_info and drug_info[key]:
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
        """Enhanced semantic search dengan context awareness"""
        # Enhance query dengan konteks percakapan
        enhanced_query = self._enhance_query_with_context(query)
        
        results = []
        
        st.sidebar.write(f"🎯 SEARCH: '{query}' -> '{enhanced_query}'")
        
        for drug_id, drug_info in self.drugs_db.items():
            score = self._calculate_similarity_score(enhanced_query, drug_info)
            
            if score > 0:
                results.append({
                    'score': score,
                    'drug_info': drug_info,
                    'drug_id': drug_id
                })
                st.sidebar.write(f"✅ {drug_info['nama']}: score {score}")
        
        # FALLBACK: Jika tidak ada hasil, cari berdasarkan gejala
        if not results:
            st.sidebar.write("🔄 No direct matches, trying fallback...")
            results = self._fallback_symptom_search(enhanced_query, top_k)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        final_drugs = [result['drug_info'] for result in results[:top_k]]
        
        st.sidebar.write(f"📊 FINAL: {[drug['nama'] for drug in final_drugs]}")
        return final_drugs
    
    def ask_question(self, question):
        """Enhanced RAG dengan context awareness"""
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
        
        # Tambahkan konteks percakapan sebelumnya
        conversation_context = self._get_conversation_context()
        
        try:
            if gemini_available:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = f"""
                Anda adalah asisten farmasi BPJS Kesehatan yang profesional.
                
                {conversation_context}
                
                INFORMASI OBAT YANG TERSEDIA:
                {context}
                
                PERTANYAAN PASIEN: {question}
                
                INSTRUKSI:
                1. Jawab pertanyaan dengan AKURAT berdasarkan informasi obat di atas
                2. Gunakan bahasa Indonesia yang JELAS dan mudah dipahami
                3. Jika informasi tidak tersedia, jangan membuat-buat jawaban
                4. Sertakan nama obat yang relevan dalam jawaban
                5. Tetap singkat namun informatif
                6. Perhatikan konteks percakapan sebelumnya
                
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
    
    def _get_conversation_context(self):
        """Get recent conversation context"""
        if not st.session_state.get('conversation_history'):
            return ""
        
        recent_conv = st.session_state.conversation_history[-2:]  # 2 pesan terakhir
        
        context = "KONTEKS PERCAKAPAN SEBELUMNYA:\n"
        for conv in recent_conv:
            context += f"Pasien: {conv['question']}\n"
            context += f"Asisten: {conv['answer'][:150]}...\n"
        
        return context
    
    def _generate_manual_answer(self, question, drugs):
        """Manual answer fallback dengan context awareness"""
        answer_parts = []
        
        # Deteksi tipe pertanyaan
        question_lower = question.lower()
        
        for drug in drugs:
            if 'dosis' in question_lower and 'dewasa' in question_lower:
                answer_parts.append(f"💊 **{drug['nama']}**")
                answer_parts.append(f"**Dosis Dewasa:** {drug['dosis_dewasa']}")
            elif 'dosis' in question_lower and 'anak' in question_lower:
                answer_parts.append(f"💊 **{drug['nama']}**")
                answer_parts.append(f"**Dosis Anak:** {drug['dosis_anak']}")
            elif 'efek samping' in question_lower:
                answer_parts.append(f"💊 **{drug['nama']}**")
                answer_parts.append(f"**Efek Samping:** {drug['efek_samping']}")
            elif 'kontra' in question_lower:
                answer_parts.append(f"💊 **{drug['nama']}**")
                answer_parts.append(f"**Kontraindikasi:** {drug['kontraindikasi']}")
            elif 'interaksi' in question_lower:
                answer_parts.append(f"💊 **{drug['nama']}**")
                answer_parts.append(f"**Interaksi:** {drug['interaksi']}")
            else:
                answer_parts.append(f"💊 **{drug['nama']}**")
                answer_parts.append(f"• Indikasi: {drug['indikasi']}")
                answer_parts.append(f"• Dosis Dewasa: {drug['dosis_dewasa']}")
        
        return "\n\n".join(answer_parts)

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
    .context-indicator {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 5px 0;
        font-size: 0.8em;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("💊 AI-PharmaAssist BPJS - Enhanced Chatbot")
st.markdown("**Chatbot Informasi Obat dengan Context Awareness**")

# Sidebar untuk debug info
with st.sidebar:
    st.header("🔍 Debug Info")
    if st.session_state.get('conversation_history'):
        st.write("**Last Context:**")
        last_conv = st.session_state.conversation_history[-1]
        st.write(f"Q: {last_conv['question']}")
        st.write(f"A: {last_conv['answer'][:100]}...")

# Layout utama
col_chat, col_info = st.columns([2, 1])

with col_chat:
    # Container chat
    st.subheader("💬 Chat dengan Context Awareness")
    
    # Quick questions
    st.markdown("**🎯 Contoh Percakapan Berantai:**")
    
    demo_scenarios = [
        "Apa dosis amoxicillin?",
        "Dosis untuk dewasa berapa?",
        "Efek sampingnya apa?"
    ]
    
    cols = st.columns(3)
    for i, scenario in enumerate(demo_scenarios):
        with cols[i]:
            if st.button(scenario, use_container_width=True, key=f"scenario_{i}"):
                # Add user message
                st.session_state.messages.append({
                    "role": "user", 
                    "content": scenario,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Get bot response
                with st.spinner("🔄 Memproses..."):
                    answer, sources = assistant.ask_question(scenario)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now(),
                        'question': scenario,
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
        <div style='text-align: center; padding: 40px; color: #666;'>
            <h3>👋 Selamat Datang!</h3>
            <p>Chatbot ini memiliki <strong>context awareness</strong> - bisa memahami pertanyaan lanjutan</p>
            <p><small>Contoh: Tanya "dosis amoxicillin?" lalu "untuk dewasa?" - sistem akan paham konteksnya</small></p>
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
                    with st.expander("📚 Sumber Informasi"):
                        for drug in message["sources"]:
                            st.write(f"• **{drug['nama']}** - {drug['golongan']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Tulis pertanyaan Anda:",
            placeholder="Contoh: Dosis untuk dewasa? (setelah menanyakan obat tertentu)",
            key="user_input"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            submit_btn = st.form_submit_button("🚀 Kirim", use_container_width=True)
        
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
        with st.spinner("🔍 Menganalisis konteks..."):
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
    st.subheader("ℹ️ Fitur Context Awareness")
    
    st.info("""
    **🎯 Fitur Baru:**
    
    • **Pemahaman Konteks**: Sistem mengingat percakapan sebelumnya
    • **Pertanyaan Lanjutan**: Bisa tanya "dosis untuk dewasa?" setelah sebut obat
    • **Enhanced Search**: Query otomatis diperkaya dengan konteks
    • **Conversation Memory**: Menyimpan riwayat percakapan
    """)
    
    st.metric("💊 Obat dalam Database", len(assistant.drugs_db))
    st.metric("💬 Riwayat Percakapan", len(st.session_state.conversation_history))
    
    # Medical disclaimer
    st.warning("""
    **⚠️ Peringatan Medis**
    
    Informasi untuk edukasi saja. Selalu konsultasi dengan dokter sebelum menggunakan obat.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "💊 AI-PharmaAssist BPJS - Chatbot dengan Context Awareness"
    "</div>", 
    unsafe_allow_html=True
)
