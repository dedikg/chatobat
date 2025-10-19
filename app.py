import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime
import hashlib

# Konfigurasi halaman
st.set_page_config(
    page_title="AI-asdasdasdasd BPJS - Enhanced RAG",
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
        """Initialize expanded drug database"""
        return {
            "paracetamol": {
                "nama": "Paracetamol",
                "golongan": "Analgesik dan Antipiretik",
                "indikasi": "Demam, nyeri ringan hingga sedang",
                "dosis_dewasa": "500-1000 mg setiap 4-6 jam, maksimal 4000 mg/hari",
                "dosis_anak": "10-15 mg/kgBB setiap 4-6 jam",
                "efek_samping": "Gangguan pencernaan, ruam kulit (jarang)",
                "kontraindikasi": "Gangguan hati berat, hipersensitif",
                "interaksi": "Alcohol meningkatkan risiko kerusakan hati",
                "merek_dagang": "Panadol, Sanmol, Tempra, Biogesic",
                "kategori": "analgesik, antipiretik, nyeri, demam"
            },
            "amoxicillin": {
                "nama": "Amoxicillin",
                "golongan": "Antibiotik Beta-Laktam", 
                "indikasi": "Infeksi bakteri saluran napas, telinga, kulit, saluran kemih",
                "dosis_dewasa": "250-500 mg setiap 8 jam",
                "dosis_anak": "20-50 mg/kgBB/hari dibagi 3 dosis",
                "efek_samping": "Diare, mual, ruam kulit, reaksi alergi",
                "kontraindikasi": "Alergi penisilin, mononukleosis infeksiosa",
                "interaksi": "Mengurangi efektivitas kontrasepsi oral",
                "merek_dagang": "Amoxan, Kalmoxillin, Moxigra",
                "kategori": "antibiotik, infeksi, bakteri"
            },
            "omeprazole": {
                "nama": "Omeprazole", 
                "golongan": "Penghambat Pompa Proton (PPI)",
                "indikasi": "Tukak lambung, GERD, dispepsia, sindrom Zollinger-Ellison",
                "dosis_dewasa": "20-40 mg sekali sehari sebelum makan",
                "dosis_anak": "Tidak dianjurkan untuk anak <1 tahun",
                "efek_samping": "Sakit kepala, diare, mual, pusing",
                "kontraindikasi": "Hipersensitif, hamil trimester pertama",
                "interaksi": "Mengurangi absorpsi ketoconazole, itraconazole", 
                "merek_dagang": "Losec, Omepron, Gastruz",
                "kategori": "lambung, maag, gerd, asam"
            },
            "ibuprofen": {
                "nama": "Ibuprofen",
                "golongan": "Anti-inflamasi nonsteroid (NSAID)",
                "indikasi": "Nyeri, inflamasi, demam, arthritis, dismenore",
                "dosis_dewasa": "200-400 mg setiap 4-6 jam, maksimal 1200 mg/hari",
                "dosis_anak": "5-10 mg/kgBB setiap 6-8 jam",
                "efek_samping": "Gangguan lambung, pusing, ruam kulit, tinitus",
                "kontraindikasi": "Ulkus peptikum, gangguan ginjal, hamil trimester ketiga",
                "interaksi": "Meningkatkan risiko perdarahan dengan antikoagulan",
                "merek_dagang": "Proris, Arthrifen, Ibufar",
                "kategori": "antiinflamasi, nyeri, demam, radang"
            },
            "vitamin_c": {
                "nama": "Vitamin C",
                "golongan": "Vitamin dan Suplemen",
                "indikasi": "Suplementasi vitamin C, meningkatkan daya tahan tubuh, penyembuhan luka",
                "dosis_dewasa": "500-1000 mg per hari",
                "dosis_anak": "sesuai kebutuhan, konsultasi dokter",
                "efek_samping": "Diare pada dosis tinggi, gangguan pencernaan",
                "kontraindikasi": "Hipersensitif",
                "interaksi": "Dapat mempengaruhi efektivitas beberapa obat kemoterapi",
                "merek_dagang": "Redoxon, Enervon C, Holisticare Ester C",
                "kategori": "vitamin, suplemen, imunitas"
            },
            "loratadine": {
                "nama": "Loratadine",
                "golongan": "Antihistamin Generasi Kedua",
                "indikasi": "Rinitis alergi, urtikaria, alergi kulit",
                "dosis_dewasa": "10 mg sekali sehari",
                "dosis_anak": "5 mg sekali sehari (usia 6-12 tahun)",
                "efek_samping": "Mengantuk (jarang), sakit kepala, mulut kering",
                "kontraindikasi": "Hipersensitif, anak <6 tahun",
                "interaksi": "Erythromycin, ketoconazole dapat meningkatkan kadar loratadine",
                "merek_dagang": "Clarityne, Loramine, Allertine",
                "kategori": "alergi, antihistamin, gatal"
            },
            "simvastatin": {
                "nama": "Simvastatin",
                "golongan": "Statin (Penurun Kolesterol)",
                "indikasi": "Hiperkolesterolemia, pencegahan penyakit kardiovaskular",
                "dosis_dewasa": "10-40 mg sekali sehari malam hari",
                "dosis_anak": "Tidak dianjurkan untuk anak",
                "efek_samping": "Nyeri otot, gangguan hati, sakit kepala",
                "kontraindikasi": "Penyakit hati aktif, hamil, menyusui",
                "interaksi": "Eritromisin, antijamur, grapefruit juice",
                "merek_dagang": "Zocor, Simvor, Lipostat",
                "kategori": "kolesterol, statin, jantung"
            }
        }
    
    def _calculate_similarity_score(self, query, drug_info):
        """Enhanced semantic similarity scoring"""
        query = query.lower()
        score = 0
        
        # Exact name match (highest priority)
        if drug_info['nama'].lower() in query:
            score += 5
            
        # Brand name match
        for merek in drug_info['merek_dagang'].lower().split(', '):
            if merek in query:
                score += 3
        
        # Category/keyword matching
        categories = drug_info.get('kategori', '').lower().split(', ')
        for category in categories:
            if category in query:
                score += 2
        
        # Symptom/indication matching
        if any(keyword in query for keyword in drug_info['indikasi'].lower().split(', ')):
            score += 2
        
        # Specific question type matching
        question_types = {
            'dosis': ['dosis', 'berapa', 'takaran', 'dosis'],
            'efek_samping': ['efek samping', 'efek', 'samping', 'bahaya'],
            'interaksi': ['interaksi', 'bereaksi', 'campur', 'bersama'],
            'kontraindikasi': ['kontraindikasi', 'larangan', 'tidak boleh', 'hamil', 'menyusui'],
            'indikasi': ['indikasi', 'untuk apa', 'kegunaan', 'manfaat']
        }
        
        for q_type, keywords in question_types.items():
            if any(keyword in query for keyword in keywords):
                score += 1
                # Bonus if the drug info has relevant content
                if drug_info.get(q_type):
                    score += 1
        
        return score
    
    def semantic_search(self, query, top_k=3):
        """Enhanced semantic search with better scoring"""
        results = []
        
        for drug_id, drug_info in self.drugs_db.items():
            score = self._calculate_similarity_score(query, drug_info)
            
            if score > 0:
                results.append({
                    'score': score,
                    'drug_info': drug_info,
                    'drug_id': drug_id
                })
        
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
        """Enhanced RAG with conversation context"""
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
            """
        
        # Add conversation context
        conversation_context = self.get_conversation_context()
        
        try:
            if gemini_available:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = f"""
                Anda adalah asisten farmasi BPJS Kesehatan yang profesional dan berpengalaman.
                
                {conversation_context}
                
                INFORMASI OBAT YANG TERSEDIA:
                {context}
                
                PERTANYAAN PASIEN: {question}
                
                INSTRUKSI:
                1. Jawab pertanyaan dengan AKURAT berdasarkan informasi obat di atas
                2. Gunakan bahasa Indonesia yang JELAS dan mudah dipahami
                3. Jika informasi tidak tersedia, jangan membuat-buat jawaban
                4. Sertakan nama obat yang relevan dalam jawaban
                5. Berikan peringatan jika ada informasi penting (efek samping serius, kontraindikasi)
                6. Tetap singkat namun informatif
                
                FORMAT JAWABAN:
                - Jawaban langsung dan informatif
                - Gunakan poin-poin untuk informasi penting
                - Sertakan saran konsultasi ke dokter/apoteker
                
                JAWABAN:
                """
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Lebih deterministic
                        max_output_tokens=800,
                        top_p=0.8
                    )
                )
                answer = response.text
            else:
                # Fallback to manual answer
                answer = self._generate_manual_answer(question, relevant_drugs)
            
            # Add to conversation history
            self.add_to_conversation_history(question, answer, relevant_drugs)
            
            return answer, relevant_drugs
            
        except Exception as e:
            st.error(f"⚠️ Error Gemini API: {str(e)}")
            answer = self._generate_manual_answer(question, relevant_drugs)
            self.add_to_conversation_history(question, answer, relevant_drugs)
            return answer, relevant_drugs
    
    def _generate_manual_answer(self, question, drugs):
        """Enhanced manual answer generation"""
        question_lower = question.lower()
        answer_parts = []
        
        for drug in drugs:
            drug_answer = [f"**{drug['nama']}** ({drug['merek_dagang']})"]
            
            if any(keyword in question_lower for keyword in ['dosis', 'berapa', 'takaran']):
                drug_answer.append(f"• **Dosis Dewasa:** {drug['dosis_dewasa']}")
                if drug['dosis_anak'] and 'tidak dianjurkan' not in drug['dosis_anak'].lower():
                    drug_answer.append(f"• **Dosis Anak:** {drug['dosis_anak']}")
                    
            elif any(keyword in question_lower for keyword in ['efek samping', 'efek', 'samping', 'bahaya']):
                drug_answer.append(f"• **Efek Samping:** {drug['efek_samping']}")
                
            elif any(keyword in question_lower for keyword in ['interaksi', 'bereaksi', 'campur']):
                drug_answer.append(f"• **Interaksi Obat:** {drug['interaksi']}")
                
            elif any(keyword in question_lower for keyword in ['kontraindikasi', 'larangan', 'tidak boleh', 'hamil', 'menyusui']):
                drug_answer.append(f"• **Kontraindikasi:** {drug['kontraindikasi']}")
                
            elif any(keyword in question_lower for keyword in ['indikasi', 'untuk apa', 'kegunaan', 'manfaat']):
                drug_answer.append(f"• **Indikasi:** {drug['indikasi']}")
                drug_answer.append(f"• **Golongan:** {drug['golongan']}")
                
            else:
                # General comprehensive info
                drug_answer.extend([
                    f"• **Golongan:** {drug['golongan']}",
                    f"• **Indikasi:** {drug['indikasi']}",
                    f"• **Dosis Dewasa:** {drug['dosis_dewasa']}",
                    f"• **Dosis Anak:** {drug['dosis_anak']}",
                    f"• **Efek Samping:** {drug['efek_samping']}",
                    f"• **Kontraindikasi:** {drug['kontraindikasi']}",
                    f"• **Interaksi:** {drug['interaksi']}"
                ])
            
            answer_parts.append("\n".join(drug_answer))
        
        # Add medical disclaimer
        answer_parts.append("\n\n⚠️ **PERINGATAN MEDIS:** Informasi ini untuk edukasi. Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat.")
        
        return "\n\n".join(answer_parts)

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
    • Semantic Search Scoring
    • Conversation Memory
    • Expanded Drug Database (7+ obat)
    • Context-Aware Responses
    • Fallback Mechanisms
    
    **💊 Database:** 7+ obat umum dengan kategori
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

# Main Interface dengan tabs enhanced
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Tanya Obat", "📊 Data Obat", "💬 Riwayat", "🎯 Demo Cepat"])

with tab1:
    st.subheader("Tanya Informasi Obat - Enhanced RAG")
    
    # Conversation starter
    if not assistant.conversation_history:
        st.info("💡 **Tips:** Tanyakan tentang dosis, efek samping, interaksi, atau kontraindikasi obat.")
    
    question = st.text_area(
        "Masukkan pertanyaan tentang obat:",
        placeholder="Contoh: Apa dosis amoxicillin untuk infeksi tenggorokan? Bolehkah ibu hamil minum obat alergi? Interaksi simvastatin dengan apa saja?",
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
            
            # RAG Process Visualization
            with st.expander("🔍 Proses RAG Detail"):
                st.write("**1. Retrieval:** Mencari obat relevan dari database")
                st.write("**2. Augmentation:** Menyiapkan context untuk LLM")
                st.write("**3. Generation:** Generate jawaban dengan konteks")
                st.write(f"**Obat yang ditemukan:** {len(sources)} obat relevan")

with tab2:
    st.subheader("📊 Enhanced Drug Database")
    
    # Search and filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Cari obat:", placeholder="Nama obat, golongan, atau indikasi...")
    
    with col2:
        filter_category = st.selectbox("Filter Kategori:", ["Semua", "Analgesik", "Antibiotik", "Vitamin", "Antihistamin", "Statin", "PPI", "NSAID"])
    
    # Display filtered drugs
    filtered_drugs = []
    for drug_id, drug_info in assistant.drugs_db.items():
        if search_term:
            if (search_term.lower() in drug_info['nama'].lower() or 
                search_term.lower() in drug_info['golongan'].lower() or
                search_term.lower() in drug_info['indikasi'].lower()):
                filtered_drugs.append(drug_info)
        elif filter_category != "Semua":
            if filter_category.lower() in drug_info['golongan'].lower():
                filtered_drugs.append(drug_info)
        else:
            filtered_drugs.append(drug_info)
    
    # Display as dataframe
    if filtered_drugs:
        drugs_df = pd.DataFrame([{
            "Nama Obat": drug["nama"],
            "Golongan": drug["golongan"],
            "Indikasi": drug["indikasi"][:80] + "..." if len(drug["indikasi"]) > 80 else drug["indikasi"],
            "Merek Dagang": drug["merek_dagang"]
        } for drug in filtered_drugs])
        
        st.dataframe(drugs_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada obat yang sesuai dengan filter.")
    
    # Detailed drug view
    st.subheader("📖 Detail Informasi Obat")
    selected_drug = st.selectbox(
        "Pilih obat untuk detail lengkap:", 
        list(assistant.drugs_db.keys()), 
        format_func=lambda x: assistant.drugs_db[x]['nama']
    )
    
    if selected_drug:
        drug = assistant.drugs_db[selected_drug]
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**💊 Nama Obat:** {drug['nama']}")
            st.write(f"**📋 Golongan:** {drug['golongan']}")
            st.write(f"**🏷️ Merek Dagang:** {drug['merek_dagang']}")
            st.write(f"**🎯 Indikasi:** {drug['indikasi']}")
            
        with col2:
            st.write(f"**📏 Dosis Dewasa:** {drug['dosis_dewasa']}")
            st.write(f"**👶 Dosis Anak:** {drug['dosis_anak']}")
            st.write(f"**⚠️ Efek Samping:** {drug['efek_samping']}")
            st.write(f"**🚫 Kontraindikasi:** {drug['kontraindikasi']}")
            st.write(f"**🔄 Interaksi:** {drug['interaksi']}")

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

with tab4:
    st.subheader("🎯 Demo Cepat - Enhanced")
    st.markdown("Coba pertanyaan-pertanyaan berikut untuk testing Enhanced RAG:")
    
    demo_questions = [
        "Apa dosis amoxicillin untuk infeksi telinga?",
        "Efek samping simvastatin yang serius?",
        "Bolehkah ibu hamil minum loratadine untuk alergi?",
        "Interaksi omeprazole dengan obat lain?",
        "Apa perbedaan paracetamol dan ibuprofen?",
        "Dosis vitamin C untuk daya tahan tubuh?",
        "Obat apa yang cocok untuk kolesterol tinggi?"
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

# Enhanced Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "💊 <b>AI-PharmaAssist Enhanced RAG</b> - Healthkathon 2025 | "
    "Powered by <b>Gemini 2.0 Flash</b> | "
    "Enhanced Semantic Search & Conversation Memory"
    "</div>", 
    unsafe_allow_html=True
)
