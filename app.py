# app.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime
import hashlib
import traceback

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI-PharmaAssist BPJS - Chatbot RAG",
    page_icon="💊",
    layout="wide"
)

# =========================
# Gemini (genai) setup
# =========================
gemini_available = False
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_available = True
except Exception as e:
    # Jangan crash di UI, cukup tampilkan peringatan
    st.sidebar.error("❌ Gemini API key tidak ditemukan di secrets. App akan menggunakan fallback manual.")
    st.sidebar.write(f"Detail: {e}")
    gemini_available = False

# =========================
# Assistant class (RAG + DB)
# =========================
class EnhancedPharmaAssistant:
    def __init__(self):
        self.drugs_db = self._initialize_drug_database()
        
    def _initialize_drug_database(self):
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

        # Sidebar debug kecil
        st.sidebar.write("🔍 DEBUG: Database Structure")
        for drug_id, drug_info in drugs_db.items():
            st.sidebar.write(f"• {drug_info['nama']} ({drug_id})")

        return drugs_db

    # similarity, fallback, semantic_search same as original (trimmed here for brevity)
    def _calculate_similarity_score(self, query, drug_info):
        query = query.lower()
        score = 0
        # symptom match
        if 'gejala' in drug_info and drug_info['gejala']:
            symptoms = drug_info['gejala'].lower().split(',')
            for symptom in symptoms:
                if symptom.strip() and symptom.strip() in query:
                    score += 5
        # drug name
        if drug_info['nama'].lower() in query:
            score += 5
        # indikasi keywords
        for kw in [k.strip() for k in drug_info['indikasi'].lower().split(',')]:
            if kw and kw in query:
                score += 3
        # merek
        for merek in drug_info['merek_dagang'].lower().split(','):
            if merek.strip() and merek.strip() in query:
                score += 3
        # kategori
        if 'kategori' in drug_info:
            for cat in drug_info['kategori'].lower().split(','):
                if cat.strip() and cat.strip() in query:
                    score += 2
        return score

    def _fallback_symptom_search(self, query, top_k=2):
        symptom_drug_mapping = {
            'sakit kepala': ['paracetamol', 'ibuprofen'],
            'pusing': ['paracetamol'],
            'demam': ['paracetamol', 'ibuprofen'],
            'maag': ['omeprazole'],
            'infeksi': ['amoxicillin'],
            'kolesterol': ['simvastatin'],
            'vitamin': ['vitamin_c'],
            'alergi': ['loratadine'],
            'flu': ['vitamin_c'],
            'migrain': ['paracetamol', 'ibuprofen'],
        }
        results = []
        q = query.lower()
        for symptom, drug_ids in symptom_drug_mapping.items():
            if symptom in q:
                for did in drug_ids:
                    if did in self.drugs_db and not any(r['drug_id']==did for r in results):
                        results.append({'score':4, 'drug_info':self.drugs_db[did], 'drug_id':did})
        return results[:top_k]

    def semantic_search(self, query, top_k=3):
        results = []
        for drug_id, drug_info in self.drugs_db.items():
            score = self._calculate_similarity_score(query, drug_info)
            if score > 0:
                results.append({'score':score,'drug_info':drug_info,'drug_id':drug_id})
        if not results:
            results = self._fallback_symptom_search(query, top_k)
        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['drug_info'] for r in results[:top_k]]

    def add_to_conversation_history(self, question, answer, sources):
        conversation_history = st.session_state.get('conversation_history', [])
        conversation_history.append({
            'timestamp': datetime.now(),
            'question': question,
            'answer': answer,
            'sources': [drug['nama'] for drug in sources] if sources else [],
            'session_id': hashlib.md5(str(datetime.now().date()).encode()).hexdigest()[:8]
        })
        # keep last 50
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        st.session_state.conversation_history = conversation_history

    def get_conversation_context(self, max_items=3):
        """Return last N exchanges formatted for prompt context."""
        history = st.session_state.get('conversation_history', [])
        if not history:
            return ""
        recent = history[-max_items:]
        context = ""
        for conv in recent:
            # keep it concise
            q = conv.get('question','').replace('\n',' ')
            a = conv.get('answer','').replace('\n',' ')
            context += f"User: {q}\nAssistant: {a}\n"
        return context

    def ask_question(self, question):
        """Use semantic search then call Gemini (or fallback manual)"""
        relevant_drugs = self.semantic_search(question)
        if not relevant_drugs:
            available = ", ".join([d['nama'] for d in self.drugs_db.values()])
            return f"Maaf, data obat tidak ditemukan. Coba tanyakan: {available}", []

        # build context for Gemini
        drugs_context = "INFORMASI OBAT YANG RELEVAN:\n"
        for i, drug in enumerate(relevant_drugs, 1):
            drugs_context += (
                f"OBAT {i}:\n"
                f"- NAMA: {drug['nama']}\n"
                f"- MEREK: {drug['merek_dagang']}\n"
                f"- GOLONGAN: {drug['golongan']}\n"
                f"- INDIKASI: {drug['indikasi']}\n"
                f"- DOSIS DEWASA: {drug['dosis_dewasa']}\n"
                f"- DOSIS ANAK: {drug['dosis_anak']}\n"
                f"- EFEK SAMPING: {drug['efek_samping']}\n"
                f"- KONTRAINDIKASI: {drug['kontraindikasi']}\n"
                f"- INTERAKSI: {drug['interaksi']}\n"
                f"- GEJALA: {drug.get('gejala','Tidak tersedia')}\n\n"
            )

        # include recent conversation as context
        context_history = self.get_conversation_context(max_items=3)

        prompt = f"""
Anda adalah asisten farmasi BPJS Kesehatan yang profesional.

{('PERCAPAKAN TERAKHIR:\n' + context_history) if context_history else ''}

{drugs_context}

PERTANYAAN PENGGUNA:
{question}

INSTRUKSI:
1) Jawab singkat dan jelas dalam Bahasa Indonesia.
2) Gunakan hanya informasi yang tercantum di atas.
3) Jika informasi tidak tersedia, nyatakan bahwa informasi tidak ada.
4) Sebutkan nama obat yang relevan.
"""

        # Try Gemini
        try:
            if gemini_available:
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=800,
                        top_p=0.8
                    )
                )
                return response.text.strip(), relevant_drugs
            else:
                return self._generate_manual_answer(question, relevant_drugs), relevant_drugs
        except Exception as e:
            # show error in sidebar but return fallback
            st.sidebar.error("⚠️ Gemini API Error (see sidebar for details)")
            st.sidebar.write(traceback.format_exc())
            return self._generate_manual_answer(question, relevant_drugs), relevant_drugs

    def _generate_manual_answer(self, question, drugs):
        parts = [f"**Pertanyaan:** {question}"]
        for d in drugs:
            parts.append(f"💊 **{d['nama']}**")
            parts.append(f"• Indikasi: {d['indikasi']}")
            parts.append(f"• Dosis dewasa: {d['dosis_dewasa']}")
            if d.get('gejala'):
                parts.append(f"• Gejala terkait: {d['gejala']}")
        return "\n".join(parts)

# =========================
# Load assistant (cached)
# =========================
@st.cache_resource
def load_assistant():
    return EnhancedPharmaAssistant()

assistant = load_assistant()

# =========================
# Session state init
# =========================
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = True

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("**System status & debug**")
    if gemini_available:
        st.success("✅ Gemini configured")
    else:
        st.warning("⚠️ Gemini not configured — fallback manual used")

    st.metric("Jumlah Obat", len(assistant.drugs_db))
    st.metric("Riwayat tersimpan", len(st.session_state.conversation_history))

    if st.button("🗑️ Reset Chat"):
        st.session_state.conversation_history = []
        st.experimental_rerun()

    st.markdown("---")
    st.caption("Deploy: Streamlit Cloud • Simpan GEMINI API KEY di Secrets (STREAMLIT)")

# =========================
# Main UI (Chatbot)
# =========================
st.title("💊 AI-PharmaAssist BPJS - Chatbot RAG")
st.markdown("Sistem tanya-jawab obat dengan RAG + Gemini. **Informasi hanya referensi, bukan pengganti tenaga medis.**")
st.markdown("---")

# Chat box area (left column wide)
chat_col, info_col = st.columns([3, 1])

with chat_col:
    st.subheader("Obrolan")
    # render chat history
    if st.session_state.conversation_history:
        for entry in st.session_state.conversation_history:
            # show user message
            with st.chat_message("user"):
                st.markdown(entry['question'])
            # show assistant message
            with st.chat_message("assistant"):
                st.markdown(entry['answer'])
    else:
        st.info("Mulai tanya tentang obat: misal 'Obat sakit kepala', 'Dosis amoxicillin untuk dewasa'.")

    # chat input (Streamlit >= v1.28)
    user_input = st.chat_input("Tanyakan sesuatu tentang obat...")

    if user_input:
        # show user's message immediately
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("🔍 Mencari informasi..."):
            answer, sources = assistant.ask_question(user_input)
            assistant.add_to_conversation_history(user_input, answer, sources)

        with st.chat_message("assistant"):
            st.write(answer)

        # small medical disclaimer
        st.warning("⚠️ Informasi ini untuk edukasi. Konsultasikan dengan tenaga medis sebelum mengambil tindakan.")
        # rerun to re-render the full history (keeps messages ordered)
        st.experimental_rerun()

with info_col:
    st.subheader("Sumber & Ringkasan")
    if st.session_state.conversation_history:
        last = st.session_state.conversation_history[-1]
        st.write("Pertanyaan terakhir:")
        st.write(last['question'])
        st.write("Sumber obat terkait:")
        if last['sources']:
            for s in last['sources']:
                st.write(f"- {s}")
        else:
            st.write("— Tidak ada sumber spesifik")
    st.markdown("---")
    st.write("Demo cepat:")
    demo_qs = [
        "Obat untuk sakit kepala?",
        "Apa dosis amoxicillin untuk dewasa?",
        "Bolehkah ibu hamil minum obat alergi?",
    ]
    for q in demo_qs:
        if st.button(q):
            # use the assistant pipeline
            with st.spinner("Memproses..."):
                ans, src = assistant.ask_question(q)
                assistant.add_to_conversation_history(q, ans, src)
            st.experimental_rerun()

# footer
st.markdown("---")
st.caption("AI-PharmaAssist — Powered by Gemini (jika tersedia) • Developed for Streamlit Cloud")
