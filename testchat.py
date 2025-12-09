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

# ===========================================
# KELAS EVALUASI RAG - 4 METRIK
# ===========================================
class RAGEvaluator:
    def __init__(self, assistant):
        self.assistant = assistant
        
        # Dataset evaluasi (10 pertanyaan representatif)
        self.test_set = [
            {
                "question": "Apa dosis paracetamol untuk dewasa?",
                "expected_drug": "paracetamol",
                "key_facts": ["500", "1000", "mg", "4-6 jam", "4000", "hari"],
                "expected_answer": "Dosis paracetamol untuk dewasa adalah 500-1000mg setiap 4-6 jam, maksimal 4000mg per hari."
            },
            {
                "question": "Efek samping amoxicillin apa saja?",
                "expected_drug": "amoxicillin",
                "key_facts": ["diare", "mual", "ruam", "alergi", "efek samping"],
                "expected_answer": "Efek samping amoxicillin termasuk diare, mual, ruam kulit, dan reaksi alergi."
            },
            {
                "question": "Untuk apa omeprazole digunakan?",
                "expected_drug": "omeprazole",
                "key_facts": ["maag", "asam lambung", "ulkus", "GERD", "indikasi"],
                "expected_answer": "Omeprazole digunakan untuk mengobati maag, asam lambung berlebih, ulkus, dan GERD."
            },
            {
                "question": "Apa kontraindikasi ibuprofen?",
                "expected_drug": "ibuprofen",
                "key_facts": ["alergi", "hamil", "ginjal", "hati", "kontraindikasi"],
                "expected_answer": "Kontraindikasi ibuprofen termasuk alergi, kehamilan trimester ketiga, gangguan ginjal dan hati berat."
            },
            {
                "question": "Interaksi obat metformin dengan apa?",
                "expected_drug": "metformin",
                "key_facts": ["alkohol", "obat jantung", "kontras", "interaksi"],
                "expected_answer": "Metformin berinteraksi dengan alkohol, beberapa obat jantung, dan zat kontras radiografi."
            },
            {
                "question": "Dosis atorvastatin berapa?",
                "expected_drug": "atorvastatin",
                "key_facts": ["10", "20", "40", "80", "mg", "dosis"],
                "expected_answer": "Dosis atorvastatin biasanya 10-80mg per hari tergantung kondisi."
            },
            {
                "question": "Apa efek samping simvastatin?",
                "expected_drug": "simvastatin",
                "key_facts": ["nyeri otot", "lever", "pusing", "efek samping"],
                "expected_answer": "Efek samping simvastatin termasuk nyeri otot, peningkatan enzim hati, dan pusing."
            },
            {
                "question": "Untuk apa lansoprazole?",
                "expected_drug": "lansoprazole",
                "key_facts": ["asam lambung", "ulkus", "GERD", "indikasi"],
                "expected_answer": "Lansoprazole digunakan untuk mengobati ulkus lambung dan GERD."
            },
            {
                "question": "Apa peringatan penggunaan aspirin?",
                "expected_drug": "aspirin",
                "key_facts": ["perdarahan", "lambung", "anak", "peringatan"],
                "expected_answer": "Peringatan aspirin termasuk risiko perdarahan lambung dan sindrom Reye pada anak."
            },
            {
                "question": "Bagaimana cara pakai cetirizine?",
                "expected_drug": "cetirizine",
                "key_facts": ["10", "mg", "sekali", "hari", "dosis"],
                "expected_answer": "Cetirizine biasanya diberikan 10mg sekali sehari."
            }
        ]
    
    def calculate_mrr(self):
        """1. Hitung Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []
        
        for test in self.test_set:
            detected_drugs = self.assistant.drug_detector.detect_drug_from_query(test["question"])
            
            # Cari rank dari expected drug
            rank = None
            for i, drug_info in enumerate(detected_drugs, 1):
                if drug_info['drug_name'] == test["expected_drug"]:
                    rank = i
                    break
            
            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    def calculate_faithfulness(self):
        """2. Hitung Faithfulness (kesetiaan ke sumber FDA)"""
        faithful_count = 0
        
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            
            # Check sederhana: jika ada sumber dari FDA, anggap faithful
            if sources and len(sources) > 0:
                faithful_count += 1
                
                # Check tambahan: jawaban tidak mengandung "tidak ada informasi"
                if "tidak ditemukan" not in answer.lower() and "tidak tersedia" not in answer.lower():
                    faithful_count += 0.5  # Bonus untuk jawaban yang lengkap
        
        return faithful_count / (len(self.test_set) * 1.5)  # Normalize ke 0-1
    
    def calculate_answer_relevancy(self):
        """3. Hitung Answer Relevancy"""
        relevancy_scores = []
        
        for test in self.test_set:
            answer, _ = self.assistant.ask_question(test["question"])
            
            # Hitung berapa banyak key facts yang muncul di jawaban
            found_facts = 0
            for fact in test["key_facts"]:
                if fact.lower() in answer.lower():
                    found_facts += 1
            
            # Normalize score
            score = found_facts / len(test["key_facts"]) if test["key_facts"] else 0
            relevancy_scores.append(score)
        
        return np.mean(relevancy_scores) if relevancy_scores else 0
    
    def calculate_semantic_similarity(self):
        """4. Hitung Semantic Similarity (sederhana dengan Jaccard)"""
        similarity_scores = []
        
        for test in self.test_set:
            answer, _ = self.assistant.ask_question(test["question"])
            expected = test.get("expected_answer", "")
            
            if expected:
                # Simple Jaccard similarity
                score = self._jaccard_similarity(answer, expected)
                similarity_scores.append(score)
        
        return np.mean(similarity_scores) if similarity_scores else 0
    
    def _jaccard_similarity(self, text1, text2):
        """Menghitung Jaccard similarity antara dua teks"""
        # Preprocess: lowercase, remove punctuation, split words
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        # Jaccard similarity = intersection / union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def run_complete_evaluation(self):
        """Jalankan semua evaluasi dan return hasil"""
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(self.test_set),
            "MRR": self.calculate_mrr(),
            "Faithfulness": self.calculate_faithfulness(),
            "Answer_Relevancy": self.calculate_answer_relevancy(),
            "Semantic_Similarity": self.calculate_semantic_similarity()
        }
        
        # Hitung rata-rata keseluruhan
        scores = [results["MRR"], results["Faithfulness"], 
                  results["Answer_Relevancy"], results["Semantic_Similarity"]]
        results["Overall_Score"] = np.mean(scores)
        
        return results

# ===========================================
# KELAS-KELAS EXISTING (SAMA)
# ===========================================
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
            return text

class EnhancedDrugDetector:
    def __init__(self):
        # PERBAIKAN: Mapping yang benar antara nama Indonesia dan nama FDA
        # Format: 'nama_yang_dikenal': ['nama_fda_actual', 'alias1', 'alias2']
        self.drug_dictionary = {
            'paracetamol': ['acetaminophen', 'paracetamol', 'panadol', 'sanmol', 'tempra'],
            'omeprazole': ['omeprazole', 'prilosec', 'losec', 'omepron'],
            'amoxicillin': ['amoxicillin', 'amoxilin', 'amoxan', 'moxigra'],
            'ibuprofen': ['ibuprofen', 'proris', 'arthrifen', 'ibufar'],
            'metformin': ['metformin', 'glucophage', 'metfor', 'diabex'],
            'atorvastatin': ['atorvastatin', 'lipitor', 'atorva', 'tovast'],
            'simvastatin': ['simvastatin', 'zocor', 'simvor', 'lipostat'],
            'loratadine': ['loratadine', 'clarityne', 'loramine', 'allertine'],
            'aspirin': ['aspirin', 'aspro', 'aspilet', 'cardiprin'],
            'vitamin c': ['ascorbic acid', 'vitamin c', 'redoxon', 'enervon c'],
            'lansoprazole': ['lansoprazole', 'prevacid', 'lanzol', 'gastracid'],
            'esomeprazole': ['esomeprazole', 'nexium', 'esotrax', 'esomep'],
            'cefixime': ['cefixime', 'suprax', 'cefix', 'fixcef'],
            'cetirizine': ['cetirizine', 'zyrtec', 'cetrizin', 'allertec'],
            'dextromethorphan': ['dextromethorphan', 'dmp', 'dextro', 'valtus'],
            'ambroxol': ['ambroxol', 'mucosolvan', 'ambrox', 'broxol'],
            'salbutamol': ['albuterol', 'salbutamol', 'ventolin', 'salbu', 'asmasolon']  # Salbutamol = Albuterol di FDA
        }
        
        # Mapping khusus untuk nama FDA
        self.fda_name_mapping = {
            'paracetamol': 'acetaminophen',
            'vitamin c': 'ascorbic acid', 
            'salbutamol': 'albuterol'
        }
    
    def detect_drug_from_query(self, query: str):
        """Detect drug name from user query dengan mapping ke nama FDA"""
        query_lower = query.lower()
        detected_drugs = []
        
        for drug_name, aliases in self.drug_dictionary.items():
            # Check semua alias
            for alias in aliases:
                if alias in query_lower:
                    # Dapatkan nama FDA yang sebenarnya
                    fda_name = self.fda_name_mapping.get(drug_name, drug_name)
                    
                    detected_drugs.append({
                        'drug_name': drug_name,
                        'fda_name': fda_name,  # Nama yang akan dicari di FDA API
                        'alias_found': alias,
                        'confidence': 'high' if alias == drug_name else 'medium'
                    })
                    break
        
        return detected_drugs
    
    def get_all_available_drugs(self):
        """Get list of all available drugs (nama yang dikenali user)"""
        return list(self.drug_dictionary.keys())
    
    def get_fda_name(self, drug_name: str):
        """Get FDA name untuk drug tertentu"""
        return self.fda_name_mapping.get(drug_name, drug_name)

class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        self.drug_detector = EnhancedDrugDetector()
        self.drugs_cache = {}
        self.current_context = {}
        
    def _get_or_fetch_drug_info(self, drug_name: str):
        """Dapatkan data dari cache atau fetch dari FDA API dengan nama FDA yang benar"""
        drug_key = drug_name.lower()
        
        if drug_key in self.drugs_cache:
            return self.drugs_cache[drug_key]
        
        # Dapatkan nama FDA yang sebenarnya
        fda_name = self.drug_detector.get_fda_name(drug_name)
        
        # Fetch dari FDA API dengan nama FDA
        drug_info = self.fda_api.get_drug_info(fda_name)
        
        if drug_info:
            # Update nama ke nama yang familiar untuk user
            if drug_name != fda_name:
                drug_info['nama'] = drug_name.title()
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"
            
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
        """Retrieve relevant information menggunakan FDA API dengan drug detection yang lebih baik"""
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
                # Fetch data dari FDA API dengan nama yang benar
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
            
            # Tambahkan catatan jika ada nama FDA yang berbeda
            if 'catatan' in drug_info:
                context += f"- Catatan: {drug_info['catatan']}\n"
                
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
            return f"**Informasi dari FDA:**\n\n{context}"
    
    def _update_conversation_context(self, question, answer, sources):
        """Update conversation context"""
        if sources:
            self.current_context = {
                'current_drug': sources[0]['nama'],
                'timestamp': datetime.now()
            }

# ===========================================
# FUNGSI UTAMA DENGAN TAB EVALUASI
# ===========================================
def main():
    # Initialize assistant
    assistant = SimpleRAGPharmaAssistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

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
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .good-score { color: #4CAF50; }
        .medium-score { color: #FF9800; }
        .poor-score { color: #F44336; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("💊 Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["🏠 Chatbot Obat", "📊 Evaluasi RAG System"]
    )
    
    # Informasi obat yang tersedia di sidebar
    st.sidebar.markdown("### 💊 Obat yang Tersedia")
    
    drug_detector = EnhancedDrugDetector()
    available_drugs = drug_detector.get_all_available_drugs()
    
    st.sidebar.info(f"""
    Sistem dapat mencari informasi tentang:
    {', '.join(available_drugs[:10])}
    ...dan {len(available_drugs) - 10} obat lainnya
    
    *Beberapa obat memiliki nama berbeda di FDA
    """)
    
    # ===========================================
    # HALAMAN CHATBOT (EXISTING)
    # ===========================================
    if page == "🏠 Chatbot Obat":
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
                <p><em>Catatan: Beberapa obat memiliki nama berbeda di FDA (contoh: Paracetamol = Acetaminophen)</em></p>
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
                                card_content = f"""
                                <div class="drug-card">
                                    <h4>💊 {drug['nama']}</h4>
                                    <p><strong>Golongan:</strong> {drug['golongan']}</p>
                                    <p><strong>Merek Dagang:</strong> {drug['merek_dagang']}</p>
                                    <p><strong>Indikasi:</strong> {drug['indikasi'][:150]}...</p>
                                    <p><strong>Bentuk:</strong> {drug['bentuk_sediaan']}</p>
                                """
                                if 'catatan' in drug:
                                    card_content += f"<p><em>{drug['catatan']}</em></p>"
                                card_content += "</div>"
                                st.markdown(card_content, unsafe_allow_html=True)
            
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
            st.rerun()

        # Medical disclaimer
        st.warning("""
        **⚠️ Peringatan Medis:** Informasi ini berasal dari database FDA AS dan untuk edukasi saja. 
        Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat. 
        Obat mungkin memiliki nama merek berbeda di Indonesia.
        """)
    
    # ===========================================
    # HALAMAN EVALUASI RAG
    # ===========================================
    elif page == "📊 Evaluasi RAG System":
        st.title("📊 Evaluasi RAG System - 4 Metrik")
        st.markdown("Evaluasi performa sistem menggunakan 4 metrik standar RAG")
        
        # Deskripsi metrik
        with st.expander("ℹ️ Tentang 4 Metrik Evaluasi"):
            st.markdown("""
            ### **1. Mean Reciprocal Rank (MRR)**
            - **Apa**: Mengukur akurasi sistem dalam menemukan obat yang benar dari query
            - **Target**: > 0.8 (80%)
            - **Rumus**: MRR = (1/rank₁ + 1/rank₂ + ...) / n
            
            ### **2. Faithfulness**
            - **Apa**: Mengukur kesetiaan jawaban terhadap data sumber (FDA)
            - **Target**: > 0.85 (85%)
            - **Indikator**: Jawaban berdasarkan data FDA, tidak ada informasi fiktif
            
            ### **3. Answer Relevancy**
            - **Apa**: Mengukur relevansi jawaban terhadap pertanyaan spesifik
            - **Target**: > 0.7 (70%)
            - **Metode**: Keyword matching dengan expected key facts
            
            ### **4. Semantic Similarity**
            - **Apa**: Mengukur kesamaan makna dengan jawaban referensi
            - **Target**: > 0.75 (75%)
            - **Metode**: Jaccard similarity pada kata kunci
            """)
        
        # Tombol untuk menjalankan evaluasi
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Jalankan Evaluasi Komprehensif", use_container_width=True):
                with st.spinner("Menjalankan evaluasi 10 test cases..."):
                    evaluator = RAGEvaluator(assistant)
                    results = evaluator.run_complete_evaluation()
                    st.session_state.evaluation_results = results
                    st.rerun()
        
        with col2:
            if st.button("📥 Simpan Hasil", use_container_width=True):
                if st.session_state.evaluation_results:
                    # Simpan ke file JSON
                    filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(st.session_state.evaluation_results, f, indent=2)
                    st.success(f"✅ Hasil disimpan ke {filename}")
                else:
                    st.warning("⚠️ Jalankan evaluasi terlebih dahulu!")
        
        with col3:
            if st.button("🔄 Reset Hasil", use_container_width=True):
                st.session_state.evaluation_results = None
                st.rerun()
        
        # Tampilkan hasil evaluasi jika ada
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            st.markdown("---")
            st.markdown(f"### 📈 Hasil Evaluasi ({results['timestamp']})")
            st.markdown(f"**Test Cases:** {results['total_test_cases']} pertanyaan")
            
            # Tampilkan metrik dalam 4 kolom
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mrr = results["MRR"]
                color_class = "good-score" if mrr >= 0.8 else "medium-score" if mrr >= 0.6 else "poor-score"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MRR</div>
                    <div class="metric-value {color_class}">{mrr:.3f}</div>
                    <div>Mean Reciprocal Rank</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Target: >0.800 | Baseline: 0.930 (Samudra dkk.)")
            
            with col2:
                faithfulness = results["Faithfulness"]
                color_class = "good-score" if faithfulness >= 0.85 else "medium-score" if faithfulness >= 0.7 else "poor-score"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Faithfulness</div>
                    <div class="metric-value {color_class}">{faithfulness:.3f}</div>
                    <div>Kesetiaan ke Sumber</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Target: >0.850 | Baseline: 0.620 (Samudra dkk.)")
            
            with col3:
                relevancy = results["Answer_Relevancy"]
                color_class = "good-score" if relevancy >= 0.7 else "medium-score" if relevancy >= 0.5 else "poor-score"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Answer Relevancy</div>
                    <div class="metric-value {color_class}">{relevancy:.3f}</div>
                    <div>Relevansi Jawaban</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Target: >0.700 | Baseline: 0.570 (Samudra dkk.)")
            
            with col4:
                similarity = results["Semantic_Similarity"]
                color_class = "good-score" if similarity >= 0.75 else "medium-score" if similarity >= 0.6 else "poor-score"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Semantic Similarity</div>
                    <div class="metric-value {color_class}">{similarity:.3f}</div>
                    <div>Kesamaan Semantik</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Target: >0.750 | Baseline: 0.810 (Samudra dkk.)")
            
            # Overall Score
            st.markdown("---")
            overall = results["Overall_Score"]
            col_overall1, col_overall2 = st.columns([1, 3])
            
            with col_overall1:
                color_class = "good-score" if overall >= 0.8 else "medium-score" if overall >= 0.65 else "poor-score"
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background: #f5f5f5;">
                    <div style="font-size: 0.9em; color: #666;">Overall Score</div>
                    <div style="font-size: 2.5em; font-weight: bold; {f'color: {color_class}' if isinstance(color_class, str) else ''}">
                        {overall:.3f}
                    </div>
                    <div style="font-size: 0.8em; color: #666;">Rata-rata 4 Metrik</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_overall2:
                # Progress bar untuk setiap metrik
                st.markdown("**Detail Skor:**")
                
                for metric, value in [
                    ("MRR", results["MRR"]),
                    ("Faithfulness", results["Faithfulness"]),
                    ("Answer Relevancy", results["Answer_Relevancy"]),
                    ("Semantic Similarity", results["Semantic_Similarity"])
                ]:
                    st.progress(value, text=f"{metric}: {value:.3f}")
            
            # Tampilkan detail dalam expander
            with st.expander("📋 Detail Hasil Evaluasi"):
                st.json(results)
                
                # Tampilkan test cases
                st.markdown("### 🧪 Test Cases yang Digunakan")
                test_df = pd.DataFrame([
                    {
                        "No": i+1,
                        "Pertanyaan": test["question"],
                        "Obat yang Diharapkan": test["expected_drug"],
                        "Key Facts": ", ".join(test["key_facts"])
                    }
                    for i, test in enumerate(evaluator.test_set)
                ])
                st.dataframe(test_df, use_container_width=True)
        
        else:
            st.info("👈 Klik tombol 'Jalankan Evaluasi Komprehensif' untuk memulai evaluasi")
            
            # Preview test cases
            evaluator = RAGEvaluator(assistant)
            st.markdown("### 🧪 Preview Test Cases")
            preview_df = pd.DataFrame([
                {"No": i+1, "Pertanyaan": test["question"], "Obat": test["expected_drug"]}
                for i, test in enumerate(evaluator.test_set[:3])
            ])
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"Total: {len(evaluator.test_set)} test cases")

    # Footer (tampil di semua halaman)
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
