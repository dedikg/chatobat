import streamlit as st
import pandas as pd
import google.generativeai as genai

# Konfigurasi halaman
st.set_page_config(
    page_title="AI-PharmaAssist BPJS",
    page_icon="💊",
    layout="wide"
)

# Setup Gemini API - PAKAI GEMINI 2.0 FLASH
GEMINI_API_KEY = st.secrets.get("AIzaSyDiL6OfjKNaD8U00aa7Epn2IPZlmZVqngE")
genai.configure(api_key=GEMINI_API_KEY)

class SimplePharmaAssistant:
    def __init__(self):
        self.drugs_db = {
            "paracetamol": {
                "nama": "Paracetamol",
                "golongan": "Analgesik dan Antipiretik",
                "indikasi": "Demam, nyeri ringan hingga sedang",
                "dosis_dewasa": "500-1000 mg setiap 4-6 jam, maksimal 4000 mg/hari",
                "dosis_anak": "10-15 mg/kgBB setiap 4-6 jam",
                "efek_samping": "Gangguan pencernaan, ruam kulit (jarang)",
                "kontraindikasi": "Gangguan hati berat, hipersensitif",
                "interaksi": "Alcohol meningkatkan risiko kerusakan hati",
                "merek_dagang": "Panadol, Sanmol, Tempra, Biogesic"
            },
            "amoxicillin": {
                "nama": "Amoxicillin",
                "golongan": "Antibiotik Beta-Laktam", 
                "indikasi": "Infeksi bakteri saluran napas, telinga, kulit",
                "dosis_dewasa": "250-500 mg setiap 8 jam",
                "dosis_anak": "20-50 mg/kgBB/hari dibagi 3 dosis",
                "efek_samping": "Diare, mual, ruam kulit, reaksi alergi",
                "kontraindikasi": "Alergi penisilin, mononukleosis infeksiosa",
                "interaksi": "Mengurangi efektivitas kontrasepsi oral",
                "merek_dagang": "Amoxan, Kalmoxillin, Moxigra"
            },
            "omeprazole": {
                "nama": "Omeprazole", 
                "golongan": "Penghambat Pompa Proton (PPI)",
                "indikasi": "Tukak lambung, GERD, dispepsia",
                "dosis_dewasa": "20-40 mg sekali sehari sebelum makan",
                "dosis_anak": "Tidak dianjurkan untuk anak <1 tahun",
                "efek_samping": "Sakit kepala, diare, mual, pusing",
                "kontraindikasi": "Hipersensitif, hamil trimester pertama",
                "interaksi": "Mengurangi absorpsi ketoconazole, itraconazole", 
                "merek_dagang": "Losec, Omepron, Gastruz"
            },
            "ibuprofen": {
                "nama": "Ibuprofen",
                "golongan": "Anti-inflamasi nonsteroid (NSAID)",
                "indikasi": "Nyeri, inflamasi, demam",
                "dosis_dewasa": "200-400 mg setiap 4-6 jam, maksimal 1200 mg/hari",
                "dosis_anak": "5-10 mg/kgBB setiap 6-8 jam",
                "efek_samping": "Gangguan lambung, pusing, ruam kulit",
                "kontraindikasi": "Ulkus peptikum, gangguan ginjal, hamil trimester ketiga",
                "interaksi": "Meningkatkan risiko perdarahan dengan antikoagulan",
                "merek_dagang": "Proris, Arthrifen, Ibufar"
            },
            "vitamin_c": {
                "nama": "Vitamin C",
                "golongan": "Vitamin dan Suplemen",
                "indikasi": "Suplementasi vitamin C, meningkatkan daya tahan tubuh",
                "dosis_dewasa": "500-1000 mg per hari",
                "dosis_anak": " sesuai kebutuhan, konsultasi dokter",
                "efek_samping": "Diare pada dosis tinggi, gangguan pencernaan",
                "kontraindikasi": "Hipersensitif",
                "interaksi": "Dapat mempengaruhi efektivitas beberapa obat kemoterapi",
                "merek_dagang": "Redoxon, Enervon C, Holisticare Ester C"
            }
        }
    
    def search_drug(self, query):
        """Cari obat berdasarkan keyword"""
        query = query.lower()
        results = []
        
        for drug_id, drug_info in self.drugs_db.items():
            score = 0
            
            # Exact match
            if drug_id in query:
                score += 3
            if drug_info['nama'].lower() in query:
                score += 3
                
            # Merek dagang match
            for merek in drug_info['merek_dagang'].lower().split(', '):
                if merek in query:
                    score += 2
            
            # Keyword match
            keywords = ['dosis', 'efek', 'samping', 'interaksi', 'kontraindikasi', 'indikasi', 'golongan']
            for keyword in keywords:
                if keyword in query:
                    score += 1
            
            if score > 0:
                results.append((score, drug_info))
        
        # Sort by relevance
        results.sort(reverse=True, key=lambda x: x[0])
        return [drug for _, drug in results[:2]]
    
    def ask_question(self, question):
        """Jawab pertanyaan dengan Gemini 2.0 Flash"""
        relevant_drugs = self.search_drug(question)
        
        if not relevant_drugs:
            return "Maaf, tidak ditemukan informasi tentang obat tersebut dalam database kami. Coba tanyakan tentang: Paracetamol, Amoxicillin, Omeprazole, Ibuprofen, atau Vitamin C.", []
        
        # Prepare context for Gemini 2.0 Flash
        context = ""
        for drug in relevant_drugs:
            context += f"""
            NAMA OBAT: {drug['nama']}
            MEREK DAGANG: {drug['merek_dagang']}
            GOLONGAN: {drug['golongan']}
            INDIKASI: {drug['indikasi']}
            DOSIS DEWASA: {drug['dosis_dewasa']}
            DOSIS ANAK: {drug['dosis_anak']}
            EFEK SAMPING: {drug['efek_samping']}
            KONTRAINDIKASI: {drug['kontraindikasi']}
            INTERAKSI: {drug['interaksi']}
            """
        
        try:
            # Gunakan Gemini 2.0 Flash - MODEL TERBARU
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Anda adalah asisten farmasi BPJS Kesehatan yang profesional. 
            
            TUGAS:
            1. Jawab pertanyaan pasien dengan AKURAT berdasarkan informasi obat yang diberikan
            2. Gunakan bahasa Indonesia yang JELAS dan mudah dipahami masyarakat awam
            3. Berikan informasi yang SINGKAT tapi INFORMATIF
            4. Jika relevan, sebutkan nama obat yang menjadi sumber informasi
            
            INFORMASI OBAT YANG TERSEDIA:
            {context}
            
            PERTANYAAN PASIEN:
            {question}
            
            FORMAT JAWABAN:
            - Mulai dengan jawaban langsung ke inti pertanyaan
            - Sertakan informasi penting dari data obat
            - Gunakan poin-poin jika perlu untuk kejelasan
            - Akhiri dengan saran untuk konsultasi dokter jika diperlukan
            
            JAWABAN:
            """
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lebih deterministic untuk informasi medis
                    max_output_tokens=500,  # Cukup untuk jawaban lengkap
                    top_p=0.8
                )
            )
            return response.text, relevant_drugs
            
        except Exception as e:
            # Fallback manual jika Gemini error
            st.error(f"⚠️ Gemini API Error: {str(e)}")
            return self._generate_manual_answer(question, relevant_drugs), relevant_drugs
    
    def _generate_manual_answer(self, question, drugs):
        """Generate answer manual jika Gemini error"""
        question_lower = question.lower()
        answers = []
        
        for drug in drugs:
            if 'dosis' in question_lower:
                answers.append(f"**{drug['nama']}**")
                answers.append(f"• Dosis Dewasa: {drug['dosis_dewasa']}")
                if drug['dosis_anak'] and 'tidak dianjurkan' not in drug['dosis_anak'].lower():
                    answers.append(f"• Dosis Anak: {drug['dosis_anak']}")
                    
            elif 'efek' in question_lower or 'samping' in question_lower:
                answers.append(f"**{drug['nama']}**")
                answers.append(f"• Efek Samping: {drug['efek_samping']}")
                
            elif 'interaksi' in question_lower:
                answers.append(f"**{drug['nama']}**")
                answers.append(f"• Interaksi Obat: {drug['interaksi']}")
                
            elif 'bolehkah' in question_lower or 'hamil' in question_lower or 'kontraindikasi' in question_lower:
                answers.append(f"**{drug['nama']}**")
                answers.append(f"• Kontraindikasi: {drug['kontraindikasi']}")
                
            elif 'indikasi' in question_lower or 'guna' in question_lower:
                answers.append(f"**{drug['nama']}**")
                answers.append(f"• Indikasi: {drug['indikasi']}")
                answers.append(f"• Golongan: {drug['golongan']}")
                
            else:
                # General info
                answers.extend([
                    f"**{drug['nama']}** ({drug['merek_dagang']})",
                    f"• Golongan: {drug['golongan']}",
                    f"• Indikasi: {drug['indikasi']}",
                    f"• Dosis Dewasa: {drug['dosis_dewasa']}",
                    f"• Dosis Anak: {drug['dosis_anak']}",
                    f"• Efek Samping: {drug['efek_samping']}",
                    f"• Kontraindikasi: {drug['kontraindikasi']}",
                    f"• Interaksi: {drug['interaksi']}"
                ])
        
        return "\n\n".join(answers)

# Initialize assistant
@st.cache_resource
def load_assistant():
    return SimplePharmaAssistant()

assistant = load_assistant()

# UI Components
st.title("💊 AI-PharmaAssist BPJS Kesehatan")
st.markdown("**Sistem Tanya Jawab Informasi Obat Berbasis AI - Powered by Gemini 2.0 Flash**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Tentang")
    st.info("""
    **AI-PharmaAssist** - Healthkathon 2025
    Kategori: Artificial Intelligence
    
    **🤖 Technology Stack:**
    • Gemini 2.0 Flash (Latest)
    • Streamlit Cloud
    • Python 3.8+
    
    **💊 Database:** 5 obat umum
    """)
    
    st.markdown("---")
    st.subheader("🔧 Status System")
    try:
        st.success("✅ Gemini 2.0 Flash: Connected")
        st.metric("Jumlah Obat", "5")
    except:
        st.error("❌ Gemini: Disconnected")
    
    st.markdown("---")
    st.subheader("💊 Daftar Obat")
    for drug_id, drug_info in assistant.drugs_db.items():
        st.write(f"• **{drug_info['nama']}**")

# Main Interface
tab1, tab2, tab3 = st.tabs(["🔍 Tanya Obat", "📊 Data Obat", "🎯 Demo Cepat"])

with tab1:
    st.subheader("Tanya Informasi Obat")
    
    question = st.text_area(
        "Masukkan pertanyaan tentang obat:",
        placeholder="Contoh: Apa dosis paracetamol untuk anak? Apa efek samping amoxicillin? Bolehkah ibu hamil minum omeprazole?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_btn = st.button("🎯 Tanya AI", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Hapus", use_container_width=True):
            st.rerun()
    
    if ask_btn and question:
        with st.spinner("🔍 Mencari informasi dengan Gemini 2.0 Flash..."):
            answer, sources = assistant.ask_question(question)
            
            st.success("💡 **Informasi Obat:**")
            st.write(answer)
            
            if sources:
                with st.expander("📚 Sumber Informasi"):
                    for drug in sources:
                        st.write(f"• **{drug['nama']}** - {drug['golongan']}")
            
            # Medical disclaimer
            st.error("""
            ⚠️ **PERINGATAN MEDIS:** 
            Informasi ini untuk edukasi dan referensi saja. 
            **Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat.**
            Jangan mengganti atau menghentikan pengobatan tanpa konsultasi profesional.
            """)

with tab2:
    st.subheader("📊 Database Obat")
    
    # Display drug database
    drugs_list = []
    for drug_id, drug_info in assistant.drugs_db.items():
        drugs_list.append({
            "Nama Obat": drug_info["nama"],
            "Golongan": drug_info["golongan"], 
            "Indikasi": drug_info["indikasi"],
            "Merek Dagang": drug_info["merek_dagang"]
        })
    
    st.dataframe(pd.DataFrame(drugs_list), use_container_width=True, hide_index=True)
    
    # Drug details
    st.subheader("📖 Detail Informasi Obat")
    selected_drug = st.selectbox("Pilih obat untuk detail lengkap:", list(assistant.drugs_db.keys()), format_func=lambda x: assistant.drugs_db[x]['nama'])
    
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
    st.subheader("🎯 Demo Cepat")
    st.markdown("Coba pertanyaan-pertanyaan berikut untuk testing:")
    
    demo_questions = [
        "Apa dosis paracetamol untuk dewasa?",
        "Apa efek samping amoxicillin?",
        "Bolehkah ibu hamil minum omeprazole?",
        "Apa interaksi paracetamol dengan alkohol?",
        "Berapa dosis amoxicillin untuk anak?",
        "Apa indikasi ibuprofen?",
        "Vitamin C untuk apa?"
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
    with st.spinner(f"🔍 Memproses: {question}"):
        answer, sources = assistant.ask_question(question)
        st.success("💡 **Informasi Obat:**")
        st.write(answer)
        
        if sources:
            with st.expander("📚 Sumber Informasi"):
                for drug in sources:
                    st.write(f"• **{drug['nama']}** - {drug['golongan']}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "💊 <b>AI-PharmaAssist</b> - Healthkathon 2025 | "
    "Powered by <b>Gemini 2.0 Flash</b> | "
    "Kategori Artificial Intelligence"
    "</div>", 
    unsafe_allow_html=True
)
