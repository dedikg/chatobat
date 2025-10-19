import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime
import hashlib

# Konfigurasi halaman
st.set_page_config(
    page_title="AI-PharmaAssist BPJS - Enhanced RAG",
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
        self.conversation_history = []
        
    def _initialize_drug_database(self):
        """Initialize expanded drug database dengan gejala yang benar"""
        return {
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
    
    def _calculate_similarity_score(self, query, drug_info):
        """Enhanced semantic similarity scoring dengan symptom mapping"""
        query = query.lower()
        score = 0
        
        # 1. Direct symptom matching (HIGHEST PRIORITY)
        if 'gejala' in drug_info:
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
        categories = drug_info.get('kategori', '').lower().split(',')
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
            'nyeri': ['paracetamol', 'ibuprofen']
        }
        
        results = []
        query_lower = query.lower()
        
        for symptom, drug_ids in symptom_drug_mapping.items():
            if symptom in query_lower:
                for drug_id in drug_ids:
                    if drug_id in self.drugs_db:
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
        return [result['drug_info'] for result in results[:top_k]]
    
    def add_to_conversation_history(self, question, answer, sources):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'question': question,
            'answer': answer,
            'sources': [drug['nama'] for drug in sources],
            'session_id': hashlib.md5(str(datetime.now().date()).encode()).hexdigest()[:8]
        })
        
        # Keep only last 20 conversations
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self):
        """Get recent conversation context for better continuity"""
        if len(self.conversation_history) < 2:
            return ""
        
        recent_conv = self.conversation_history[-2:]  # Get last 2 exchanges
        context = "Percakapan sebelumnya:\n"
        for conv in recent_conv:
            context += f"Q: {conv['question']}\nA: {conv['answer'][:100]}...\n"
        return context
    
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
        
        return "\n".join(answer_parts)

# Initialize enhanced assistant
@st.cache_resource
def load_assistant():
    return EnhancedPharmaAssistant()

assistant = load_assistant()

# Enhanced UI Components
st.title("💊 AI-PharmaAssist BPJS Kesehatan - Enhanced RAG")
st.markdown("**Sistem Tanya Jawab Informasi Obat Berbasis Conversational AI dengan RAG**")
st.markdown("---")

# Sidebar dengan informasi enhanced
with st.sidebar:
    st.header("⚙️ Tentang Enhanced RAG")
    st.info("""
    **🤖 Enhanced RAG Features:**
    • Semantic Search dengan Symptom Mapping
    • Conversation Memory
    • Expanded Drug Database (7+ obat)
    • Context-Aware Responses
    • Fallback Mechanisms
    • Symptom-to-Drug Matching
    
    **💊 Database:** 7+ obat umum dengan gejala
    """)
    
    st.markdown("---")
    st.subheader("🔧 System Status")
    if gemini_available:
        st.success("✅ Gemini 2.0 Flash: Connected")
    else:
        st.warning("⚠️ Gemini: Using Fallback Mode")
    st.metric("Jumlah Obat", len(assistant.drugs_db))
    st.metric("Percakapan Tersimpan", len(assistant.conversation_history))
    
    st.markdown("---")
    st.subheader("💊 Daftar Obat Tersedia")
    for drug_id, drug_info in assistant.drugs_db.items():
        with st.expander(f"📦 {drug_info['nama']}"):
            st.caption(f"Golongan: {drug_info['golongan']}")
            st.caption(f"Indikasi: {drug_info['indikasi'][:50]}...")
            if 'gejala' in drug_info:
                st.caption(f"Gejala: {drug_info['gejala'][:50]}...")

# Main Interface dengan tabs enhanced
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Tanya Obat", "📊 Data Obat", "💬 Riwayat", "🎯 Demo Cepat"])

with tab1:
    st.subheader("Tanya Informasi Obat - Enhanced RAG")
    
    # Conversation starter
    if not assistant.conversation_history:
        st.info("""
        💡 **Tips:** Anda bisa menanyakan:
        • **Gejala:** "obat sakit kepala", "obat demam", "obat maag"
        • **Informasi obat:** "dosis paracetamol", "efek samping amoxicillin"
        • **Kondisi khusus:** "bolehkah ibu hamil minum obat alergi?"
        """)
    
    question = st.text_area(
        "Masukkan pertanyaan tentang obat:",
        placeholder="Contoh: Obat untuk sakit kepala? Apa dosis amoxicillin? Bolehkah ibu hamil minum obat alergi? Interaksi simvastatin dengan apa saja?",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_btn = st.button("🎯 Tanya AI", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            assistant.conversation_history.clear()
            st.rerun()
    
    if ask_btn and question:
        with st.spinner("🔍 Mencari informasi dengan Enhanced RAG..."):
            answer, sources = assistant.ask_question(question)
            
            # Display results
            st.success("💡 **Informasi Obat:**")
            st.write(answer)
            
            if sources:
                with st.expander("📚 Sumber Informasi & Relevansi"):
                    for drug in sources:
                        st.write(f"• **{drug['nama']}** - {drug['golongan']}")
                        st.caption(f"Indikasi: {drug['indikasi']}")
                        if 'gejala' in drug:
                            st.caption(f"Gejala terkait: {drug['gejala']}")
            
            # Medical disclaimer
            st.error("""
            ⚠️ **PERINGATAN MEDIS:** 
            Informasi ini untuk edukasi dan referensi saja. 
            **Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat.**
            Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
            """)

with tab2:
    st.subheader("📊 Enhanced Drug Database")
    
    # Search and filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Cari obat:", placeholder="Nama obat, golongan, gejala, atau indikasi...")
    
    with col2:
        filter_category = st.selectbox("Filter Kategori:", ["Semua", "Analgesik", "Antibiotik", "Vitamin", "Antihistamin", "Statin", "PPI", "NSAID"])
    
    # Display filtered drugs
    filtered_drugs = []
    for drug_id, drug_info in assistant.drugs_db.items():
        match = False
        if search_term:
            search_lower = search_term.lower()
            if (search_lower in drug_info['nama'].lower() or 
                search_lower in drug_info['golongan'].lower() or
                search_lower in drug_info['indikasi'].lower() or
                ('gejala' in drug_info and search_lower in drug_info['gejala'].lower())):
                match = True
        elif filter_category != "Semua":
            if filter_category.lower() in drug_info['golongan'].lower():
                match = True
        else:
            match = True
        
        if match:
            filtered_drugs.append(drug_info)
    
    # Display as dataframe
    if filtered_drugs:
        drugs_df = pd.DataFrame([{
            "Nama Obat": drug["nama"],
            "Golongan": drug["golongan"],
            "Indikasi": drug["indikasi"][:80] + "..." if len(drug["indikasi"]) > 80 else drug["indikasi"],
            "Merek Dagang": drug["merek_dagang"],
            "Gejala Terkait": drug.get("gejala", "Tidak tersedia")[:60] + "..." if drug.get("gejala") and len(drug.get("gejala", "")) > 60 else drug.get("gejala", "Tidak tersedia")
        } for drug in filtered_drugs])
        
        st.dataframe(drugs_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada obat yang sesuai dengan filter.")
    
    # Medical disclaimer untuk tab Data Obat
    st.error("""
    ⚠️ **PERINGATAN MEDIS:** 
    Informasi ini untuk edukasi dan referensi saja. 
    **Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat.**
    Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
    """)

with tab3:
    st.subheader("💬 Riwayat Percakapan")
    
    if assistant.conversation_history:
        for i, conv in enumerate(reversed(assistant.conversation_history)):
            with st.expander(f"🕒 {conv['timestamp'].strftime('%H:%M:%S')} - {conv['question'][:50]}..."):
                st.write(f"**Q:** {conv['question']}")
                st.write(f"**A:** {conv['answer']}")
                st.caption(f"Sumber: {', '.join(conv['sources'])}")
    else:
        st.info("Belum ada riwayat percakapan. Mulai tanya obat di tab 'Tanya Obat'.")
    
    # Medical disclaimer untuk tab Riwayat
    st.error("""
    ⚠️ **PERINGATAN MEDIS:** 
    Informasi ini untuk edukasi dan referensi saja. 
    **Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat.**
    Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
    """)

with tab4:
    st.subheader("🎯 Demo Cepat - Enhanced")
    st.markdown("Coba pertanyaan-pertanyaan berikut untuk testing Enhanced RAG:")
    
    demo_questions = [
        "Obat untuk sakit kepala?",
        "Apa dosis amoxicillin untuk infeksi telinga?",
        "Efek samping simvastatin yang serius?",
        "Bolehkah ibu hamil minum loratadine untuk alergi?",
        "Interaksi omeprazole dengan obat lain?",
        "Apa perbedaan paracetamol dan ibuprofen?",
        "Dosis vitamin C untuk daya tahan tubuh?",
        "Obat apa yang cocok untuk kolesterol tinggi?",
        "Obat untuk demam dan pilek?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(demo_questions):
        with cols[i % 2]:
            if st.button(q, key=f"demo_{i}", use_container_width=True):
                st.session_state.demo_question = q
                st.rerun()

# Handle demo questions
if 'demo_question' in st.session_state:
    question = st.session_state.demo_question
    with st.spinner(f"🔍 Memproses dengan Enhanced RAG: {question}"):
        answer, sources = assistant.ask_question(question)
        st.success("💡 **Informasi Obat:**")
        st.write(answer)
        
        if sources:
            with st.expander("📚 Sumber Informasi"):
                for drug in sources:
                    st.write(f"• **{drug['nama']}** - {drug['golongan']}")
        
        # Medical disclaimer untuk demo questions
        st.error("""
        ⚠️ **PERINGATAN MEDIS:** 
        Informasi ini untuk edukasi dan referensi saja. 
        **Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat.**
        Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
        """)

# Enhanced Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "💊 <b>AI-PharmaAssist Enhanced RAG</b> - Healthkathon 2025 | "
    "Powered by <b>Gemini 2.0 Flash</b> | "
    "Enhanced Semantic Search & Symptom Mapping"
    "</div>", 
    unsafe_allow_html=True
)
