import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime
import time
import re
import json
import random

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Tanya Jawab Obat dengan RAG",
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
# PERBAIKAN TRANSLATION SERVICE - HANYA TERJEMAHKAN JIKA DIPERLUKAN
# ===========================================
class TranslationService:
    def __init__(self):
        self.available = gemini_available
    
    def translate_to_indonesian(self, text: str):
        """Translate text ke Bahasa Indonesia menggunakan Gemini"""
        if not self.available or not text or text == "Tidak tersedia":
            return text
        
        try:
            # Skip jika sudah mengandung banyak kata Bahasa Indonesia
            indonesian_indicators = ['untuk', 'dengan', 'yang', 'dari', 'dalam', 'pada', 'adalah', 'sebagai']
            indonesian_count = sum(1 for word in indonesian_indicators if word in text.lower())
            
            if indonesian_count > 3:
                return text
            
            # Skip teks yang sangat teknis atau pendek
            if len(text.strip()) < 15 or text.replace('.', '').replace('mg', '').replace('ml', '').replace(' ', '').isalnum():
                return text
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Anda adalah penerjemah medis profesional. Terjemahkan teks medis berikut ke Bahasa Indonesia:
            
            TEKS ASLI: {text}
            
            ATURAN PENERJEMAHAN:
            1. Pertahankan SEMUA angka, dosis, satuan (mg, ml, tablet, etc.)
            2. Pertahankan nama obat asli (acetaminophen, ibuprofen, etc.)
            3. Pertahankan istilah medis baku yang sudah dikenal di Indonesia
            4. Gunakan bahasa Indonesia formal yang mudah dipahami pasien
            5. Jangan ubah makna atau informasi medis
            6. Terjemahkan seluruh teks kecuali yang disebutkan di poin 1-3
            
            CONTOH:
            - "Take 2 tablets every 6 hours" → "Minum 2 tablet setiap 6 jam"
            - "Do not exceed 4000 mg per day" → "Jangan melebihi 4000 mg per hari"
            - "May cause drowsiness" → "Dapat menyebabkan kantuk"
            - "For the management of pain" → "Untuk penanganan nyeri"
            
            HASIL TERJEMAHAN:
            """
            
            response = model.generate_content(prompt)
            translated = response.text.strip()
            
            # Clean up
            translated = translated.replace('"', '').replace("'", "").strip()
            
            # Pastikan terjemahan tidak kosong
            if not translated or len(translated) < 5:
                return text
            
            return translated
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text

# ===========================================
# FDA API - KELAS LENGKAP
# ===========================================
class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
    
    def get_drug_info(self, generic_name: str):
        """Ambil data obat langsung dari FDA API"""
        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 5
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    best_result = None
                    max_field_count = 0
                    
                    for result in data['results']:
                        field_count = self._count_complete_fields(result)
                        if field_count > max_field_count:
                            max_field_count = field_count
                            best_result = result
                    
                    if best_result:
                        return self._parse_fda_data(best_result, generic_name)
                    
                    return self._parse_fda_data(data['results'][0], generic_name)
            
            return self._try_alternative_search(generic_name)
                
        except Exception as e:
            st.error(f"Error FDA API: {e}")
            return None
    
    def _count_complete_fields(self, fda_data: dict):
        """Hitung jumlah field yang memiliki data"""
        important_fields = [
            'indications_and_usage',
            'dosage_and_administration', 
            'adverse_reactions',
            'contraindications',
            'drug_interactions',
            'warnings'
        ]
        
        count = 0
        for field in important_fields:
            if field in fda_data and fda_data[field]:
                value = fda_data[field]
                if isinstance(value, list) and value:
                    if value[0] and value[0].strip():
                        count += 1
                elif value and value.strip():
                    count += 1
        
        return count
    
    def _try_alternative_search(self, generic_name: str):
        """Coba pencarian alternatif jika data tidak ditemukan"""
        params = {
            'search': f'_exists_:openfda.generic_name AND {generic_name}',
            'limit': 3
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    return self._parse_fda_data(data['results'][0], generic_name)
        except:
            pass
        
        return None
    
    def _parse_fda_data(self, fda_data: dict, generic_name: str):
        """Parse data FDA menjadi format yang kita butuhkan"""
        openfda = fda_data.get('openfda', {})
        
        def get_field(field_name, default="Tidak tersedia"):
            value = fda_data.get(field_name, '')
            
            if isinstance(value, list):
                if value:
                    text = ' '.join([str(v) for v in value if v])
                    return text[:500]
                return default
            
            if value and str(value).strip():
                return str(value).strip()[:500]
            
            return default
        
        # Gunakan nama Indonesia untuk obat tertentu
        display_name = generic_name.title()
        if 'acetaminophen' in generic_name.lower():
            display_name = 'Paracetamol'
        elif 'albuterol' in generic_name.lower():
            display_name = 'Salbutamol'
        elif 'ascorbic acid' in generic_name.lower():
            display_name = 'Vitamin C'
        
        # Ekstrak informasi
        indications = self._extract_indications(fda_data)
        dosage = self._extract_dosage(fda_data)
        side_effects = self._extract_side_effects(fda_data)
        contraindications = self._extract_contraindications(fda_data)
        interactions = self._extract_interactions(fda_data)
        warnings = self._extract_warnings(fda_data)
        
        drug_info = {
            "nama": display_name,
            "nama_generik": display_name,
            "merek_dagang": ", ".join(openfda.get('brand_name', ['Tidak tersedia']))[:200],
            "golongan": get_field('drug_class', "Tidak tersedia")[:100],
            "indikasi": indications[:500] if indications != "Tidak tersedia" else "Tidak tersedia",
            "dosis_dewasa": dosage[:500] if dosage != "Tidak tersedia" else "Tidak tersedia",
            "efek_samping": side_effects[:500] if side_effects != "Tidak tersedia" else "Tidak tersedia",
            "kontraindikasi": contraindications[:500] if contraindications != "Tidak tersedia" else "Tidak tersedia",
            "interaksi": interactions[:500] if interactions != "Tidak tersedia" else "Tidak tersedia",
            "peringatan": warnings[:500] if warnings != "Tidak tersedia" else "Tidak tersedia",
            "bentuk_sediaan": ", ".join(openfda.get('dosage_form', ['Tidak tersedia']))[:100],
            "route_pemberian": ", ".join(openfda.get('route', ['Tidak tersedia']))[:100],
            "sumber": "FDA API"
        }
        
        # Tambahkan catatan untuk nama FDA yang berbeda
        if generic_name.lower() in ['acetaminophen', 'albuterol', 'ascorbic acid']:
            drug_info['catatan'] = f"Di FDA dikenal sebagai {generic_name}"
        
        return drug_info
    
    def _extract_indications(self, fda_data: dict):
        """Ekstrak informasi indikasi"""
        fields_to_check = [
            'indications_and_usage',
            'purpose',
            'description'
        ]
        
        for field in fields_to_check:
            if field in fda_data and fda_data[field]:
                value = fda_data[field]
                if isinstance(value, list) and value:
                    return value[0][:500]
                elif value:
                    return str(value)[:500]
        
        return "Tidak tersedia"
    
    def _extract_dosage(self, fda_data: dict):
        """Ekstrak informasi dosis"""
        fields_to_check = [
            'dosage_and_administration',
            'directions',
            'dosage'
        ]
        
        for field in fields_to_check:
            if field in fda_data and fda_data[field]:
                value = fda_data[field]
                if isinstance(value, list) and value:
                    return value[0][:500]
                elif value:
                    return str(value)[:500]
        
        return "Tidak tersedia"
    
    def _extract_side_effects(self, fda_data: dict):
        """Ekstrak informasi efek samping"""
        if 'adverse_reactions' in fda_data and fda_data['adverse_reactions']:
            value = fda_data['adverse_reactions']
            if isinstance(value, list) and value:
                return value[0][:500]
            elif value:
                return str(value)[:500]
        
        return "Tidak tersedia"
    
    def _extract_contraindications(self, fda_data: dict):
        """Ekstrak informasi kontraindikasi"""
        if 'contraindications' in fda_data and fda_data['contraindications']:
            value = fda_data['contraindications']
            if isinstance(value, list) and value:
                return value[0][:500]
            elif value:
                return str(value)[:500]
        
        return "Tidak tersedia"
    
    def _extract_interactions(self, fda_data: dict):
        """Ekstrak informasi interaksi"""
        if 'drug_interactions' in fda_data and fda_data['drug_interactions']:
            value = fda_data['drug_interactions']
            if isinstance(value, list) and value:
                return value[0][:500]
            elif value:
                return str(value)[:500]
        
        return "Tidak tersedia"
    
    def _extract_warnings(self, fda_data: dict):
        """Ekstrak informasi peringatan"""
        if 'warnings' in fda_data and fda_data['warnings']:
            value = fda_data['warnings']
            if isinstance(value, list) and value:
                return value[0][:500]
            elif value:
                return str(value)[:500]
        
        return "Tidak tersedia"

# ===========================================
# SIMPLE RAG ASSISTANT - KELAS LENGKAP
# ===========================================
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
        
        fda_name = self.drug_detector.get_fda_name(drug_name)
        drug_info = self.fda_api.get_drug_info(fda_name)
        
        if drug_info:
            if drug_name != fda_name:
                drug_info['nama'] = drug_name.title()
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"
            
            drug_info = self._translate_all_fields(drug_info)
            self.drugs_cache[drug_key] = drug_info
        
        return drug_info
    
    def _translate_all_fields(self, drug_info: dict):
        """Translate SEMUA field yang penting ke Bahasa Indonesia"""
        fields_to_translate = [
            'indikasi', 'dosis_dewasa', 'efek_samping', 
            'kontraindikasi', 'interaksi', 'peringatan',
            'golongan', 'bentuk_sediaan', 'route_pemberian',
            'nama', 'merek_dagang'
        ]
        
        for field in fields_to_translate:
            if field in drug_info and drug_info[field] != "Tidak tersedia":
                text = drug_info[field]
                drug_info[field] = self.translator.translate_to_indonesian(text)
        
        return drug_info
    
    def _rag_retrieve(self, query, top_k=3):
        """Retrieve relevant information dari FDA API"""
        query_lower = query.lower()
        results = []
        
        detected_drugs = self.drug_detector.detect_drug_from_query(query)
        
        if not detected_drugs:
            common_drugs = self.drug_detector.get_all_available_drugs()
        else:
            common_drugs = [drug['drug_name'] for drug in detected_drugs]
        
        for drug_name in common_drugs[:top_k]:
            score = 0
            
            if drug_name in query_lower:
                score += 10
            
            aliases = self.drug_detector.drug_dictionary.get(drug_name, [])
            for alias in aliases:
                if alias in query_lower:
                    score += 8
                    break
            
            question_keywords = {
                'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai'],
                'efek': ['efek samping', 'side effect', 'bahaya', 'efeknya'],
                'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan'],
                'interaksi': ['interaksi', 'bereaksi dengan', 'makanan', 'minuman'],
                'indikasi': ['untuk apa', 'kegunaan', 'manfaat', 'indikasi']
            }
            
            for key, keywords in question_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    score += 3
            
            if score > 0:
                drug_info = self._get_or_fetch_drug_info(drug_name)
                if drug_info:
                    results.append({
                        'score': score,
                        'drug_info': drug_info,
                        'drug_id': drug_name
                    })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _build_rag_context(self, retrieved_results):
        """Build context untuk RAG generator dari data FDA"""
        if not retrieved_results:
            return "Tidak ada informasi yang relevan ditemukan dalam database FDA."
        
        context = "## INFORMASI OBAT DARI FDA:\n\n"
        
        for i, result in enumerate(retrieved_results, 1):
            drug_info = result['drug_info']
            context += f"### OBAT {i}: {drug_info['nama']}\n"
            
            if 'catatan' in drug_info:
                context += f"- **Catatan:** {drug_info['catatan']}\n"
            
            fields_to_display = [
                ('Nama Generik', 'nama_generik'),
                ('Merek Dagang', 'merek_dagang'),
                ('Golongan', 'golongan'),
                ('Indikasi', 'indikasi'),
                ('Dosis Dewasa', 'dosis_dewasa'),
                ('Efek Samping', 'efek_samping'),
                ('Kontraindikasi', 'kontraindikasi'),
                ('Interaksi', 'interaksi'),
                ('Peringatan', 'peringatan')
            ]
            
            for label, field in fields_to_display:
                if field in drug_info and drug_info[field] != "Tidak tersedia":
                    text = drug_info[field]
                    if len(text) > 300:
                        text = text[:300] + "..."
                    context += f"- **{label}:** {text}\n"
            
            context += "\n"
        
        return context
    
    def ask_question(self, question):
        """Main RAG interface dengan FDA API"""
        try:
            retrieved_results = self._rag_retrieve(question)
            
            if not retrieved_results:
                available_drugs = ", ".join(self.drug_detector.get_all_available_drugs()[:10])
                return f"❌ Tidak ditemukan informasi yang relevan dalam database FDA untuk pertanyaan Anda.\n\n💡 **Coba tanyakan tentang:** {available_drugs}", []
            
            rag_context = self._build_rag_context(retrieved_results)
            answer = self._generate_rag_response(question, rag_context)
            
            sources = []
            seen_drug_names = set()
            
            for result in retrieved_results:
                drug_name = result['drug_info']['nama']
                if drug_name not in seen_drug_names:
                    sources.append(result['drug_info'])
                    seen_drug_names.add(drug_name)
            
            self._update_conversation_context(question, answer, sources)
            
            return answer, sources
            
        except Exception as e:
            st.error(f"Error dalam proses RAG: {e}")
            return "Maaf, terjadi error dalam sistem. Silakan coba lagi.", []
    
    def _generate_rag_response(self, question, context):
        """Generate response menggunakan RAG pattern dengan Gemini"""
        if not gemini_available:
            return f"**Informasi dari FDA:**\n\n{context}"
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            ANDA HARUS MENGGUNAKAN BAHASA INDONESIA SELURUHNYA.
            
            Anda adalah asisten farmasi profesional di Indonesia. Jawab pertanyaan tentang obat HANYA berdasarkan informasi dari FDA yang disediakan.
            
            ## INFORMASI RESMI DARI FDA:
            {context}
            
            ## PERTANYAAN PENGGUNA:
            {question}
            
            ## ATURAN JAWABAN:
            1. JAWAB LANGSUNG dalam BAHASA INDONESIA yang mudah dipahami pasien Indonesia
            2. Gunakan informasi dari FDA di atas sebagai SATU-SATUNYA sumber
            3. Sebutkan bahwa informasi berasal dari database resmi FDA (U.S. Food and Drug Administration)
            4. Tambahkan peringatan: "HARAP KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT INI"
            5. Jika ada informasi yang tidak lengkap dalam data FDA, katakan: "Informasi ini tidak lengkap dalam database FDA"
            6. Berikan jawaban yang SINGKAT, JELAS, dan RELEVAN dengan pertanyaan
            7. JANGAN membuat informasi baru di luar yang ada di data FDA
            8. Fokus pada informasi yang diminta dalam pertanyaan
            9. Untuk nama obat: gunakan nama Indonesia jika ada (contoh: acetaminophen = paracetamol)
            
            ## FORMAT JAWABAN:
            - Mulai dengan jawaban langsung
            - Berikan informasi spesifik dari FDA
            - Tambahkan disclaimer medis
            - Akhiri dengan saran konsultasi
            
            ## JAWABAN (DALAM BAHASA INDONESIA):
            """
            
            response = model.generate_content(prompt)
            answer = response.text
            
            # Pastikan jawaban dalam Bahasa Indonesia
            if self._is_mostly_english(answer):
                answer = self.translator.translate_to_indonesian(answer)
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"**Informasi dari FDA:**\n\n{context}\n\n**Peringatan:** Konsultasikan dengan dokter atau apoteker sebelum menggunakan obat ini."
    
    def _is_mostly_english(self, text):
        """Cek apakah teks masih banyak bahasa Inggrisnya"""
        indonesian_words = ['yang', 'untuk', 'dengan', 'dari', 'pada', 'adalah', 'sebagai', 'dapat', 'harus', 'jangan']
        english_words = ['the', 'and', 'for', 'with', 'from', 'to', 'is', 'are', 'should', 'not']
        
        text_lower = text.lower()
        id_count = sum(1 for word in indonesian_words if word in text_lower)
        en_count = sum(1 for word in english_words if word in text_lower)
        
        return en_count > id_count
    
    def _update_conversation_context(self, question, answer, sources):
        """Update conversation context"""
        if sources:
            self.current_context = {
                'current_drug': sources[0]['nama'],
                'timestamp': datetime.now()
            }

# ===========================================
# KELAS DRUG DETECTOR (TIDAK BERUBAH)
# ===========================================
class EnhancedDrugDetector:
    def __init__(self):
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
            'salbutamol': ['albuterol', 'salbutamol', 'ventolin', 'salbu', 'asmasolon']
        }
        
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
            for alias in aliases:
                if alias in query_lower:
                    fda_name = self.fda_name_mapping.get(drug_name, drug_name)
                    detected_drugs.append({
                        'drug_name': drug_name,
                        'fda_name': fda_name,
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

# ===========================================
# KELAS EVALUASI (TIDAK BERUBAH - tetap sama)
# ===========================================
class FocusedRAGEvaluator:
    def __init__(self, assistant):
        self.assistant = assistant
        
        # Test set fokus pada 2 metrik: MRR & Faithfulness
        self.test_set = [
            {
                "id": 1,
                "question": "Apa dosis paracetamol?",
                "expected_drug": "paracetamol",
                "question_type": "dosis",
                "key_info_expected": ["dosis", "mg", "paracetamol"]
            },
            {
                "id": 2,
                "question": "Efek samping amoxicillin?",
                "expected_drug": "amoxicillin",
                "question_type": "efek_samping",
                "key_info_expected": ["efek", "samping", "amoxicillin"]
            },
            {
                "id": 3,
                "question": "Untuk apa omeprazole digunakan?",
                "expected_drug": "omeprazole",
                "question_type": "indikasi",
                "key_info_expected": ["indikasi", "kegunaan", "omeprazole"]
            },
            {
                "id": 4,
                "question": "Apa kontraindikasi ibuprofen?",
                "expected_drug": "ibuprofen",
                "question_type": "kontraindikasi",
                "key_info_expected": ["kontraindikasi", "ibuprofen"]
            },
            {
                "id": 5,
                "question": "Interaksi obat metformin?",
                "expected_drug": "metformin",
                "question_type": "interaksi",
                "key_info_expected": ["interaksi", "metformin"]
            },
            {
                "id": 6,
                "question": "Berapa dosis atorvastatin?",
                "expected_drug": "atorvastatin",
                "question_type": "dosis",
                "key_info_expected": ["dosis", "atorvastatin"]
            },
            {
                "id": 7,
                "question": "Efek samping simvastatin?",
                "expected_drug": "simvastatin",
                "question_type": "efek_samping",
                "key_info_expected": ["efek", "samping", "simvastatin"]
            },
            {
                "id": 8,
                "question": "Kegunaan lansoprazole?",
                "expected_drug": "lansoprazole",
                "question_type": "indikasi",
                "key_info_expected": ["kegunaan", "lansoprazole"]
            },
            {
                "id": 9,
                "question": "Peringatan penggunaan aspirin?",
                "expected_drug": "aspirin",
                "question_type": "peringatan",
                "key_info_expected": ["peringatan", "aspirin"]
            },
            {
                "id": 10,
                "question": "Dosis cetirizine untuk dewasa?",
                "expected_drug": "cetirizine",
                "question_type": "dosis",
                "key_info_expected": ["dosis", "cetirizine", "dewasa"]
            }
        ]
    
    def calculate_mrr(self):
        """Hitung MRR untuk evaluasi komponen RETRIEVAL RAG"""
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
        """Hitung Faithfulness untuk evaluasi komponen GENERATION RAG"""
        faithful_scores = []
        
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            answer_lower = answer.lower()
            
            # Kriteria Faithfulness untuk aplikasi medis
            criteria_scores = []
            
            # 1. Sumber Data (40%)
            if sources and len(sources) > 0:
                criteria_scores.append(0.4)
            else:
                criteria_scores.append(0)
            
            # 2. Referensi FDA dalam jawaban (25%)
            fda_indicators = ["fda", "food and drug administration", "data resmi fda", "sumber fda"]
            has_fda_ref = any(indicator in answer_lower for indicator in fda_indicators)
            criteria_scores.append(0.25 if has_fda_ref else 0)
            
            # 3. Tidak ada informasi fiktif (20%)
            fictional_indicators = [
                "menurut saya", "biasanya", "umumnya", "seharusnya", 
                "kemungkinan besar", "menurut pengetahuan saya"
            ]
            has_fictional = any(indicator in answer_lower for indicator in fictional_indicators)
            criteria_scores.append(0.20 if not has_fictional else 0)
            
            # 4. Disclaimer medis (15%)
            disclaimer_indicators = ["konsultasi", "dokter", "apoteker", "sebelum menggunakan"]
            has_disclaimer = any(indicator in answer_lower for indicator in disclaimer_indicators)
            criteria_scores.append(0.15 if has_disclaimer else 0)
            
            # Total score untuk test case ini
            total_score = sum(criteria_scores)
            faithful_scores.append(min(total_score, 1.0))
        
        return np.mean(faithful_scores) if faithful_scores else 0
    
    def run_evaluation(self):
        """Jalankan evaluasi 2 metrik utama RAG"""
        try:
            # Hitung kedua metrik
            mrr_score = self.calculate_mrr()
            faithfulness_score = self.calculate_faithfulness()
            
            # Kompilasi hasil
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_set),
                "MRR": float(mrr_score),
                "Faithfulness": float(faithfulness_score),
                "RAG_Score": float((mrr_score + faithfulness_score) / 2)
            }
            
            # Simpan detail test case untuk analisis
            results["test_case_details"] = self._get_test_case_details()
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error dalam evaluasi: {str(e)}")
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "MRR": 0,
                "Faithfulness": 0,
                "RAG_Score": 0
            }
    
    def _get_test_case_details(self):
        """Ambil detail hasil untuk setiap test case"""
        details = []
        
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            
            # Deteksi drug untuk MRR
            detected_drugs = self.assistant.drug_detector.detect_drug_from_query(test["question"])
            
            # Analisis faithfulness
            answer_lower = answer.lower()
            has_source = bool(sources)
            has_fda_ref = any(indicator in answer_lower for indicator in ["fda", "food and drug administration"])
            has_disclaimer = any(indicator in answer_lower for indicator in ["dokter", "apoteker", "konsultasi"])
            
            detail = {
                "test_id": test["id"],
                "question": test["question"],
                "expected_drug": test["expected_drug"],
                "detected_drugs": [drug['drug_name'] for drug in detected_drugs],
                "detection_correct": test["expected_drug"] in [drug['drug_name'] for drug in detected_drugs],
                "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
                "has_sources": has_source,
                "source_count": len(sources) if sources else 0,
                "has_fda_reference": has_fda_ref,
                "has_medical_disclaimer": has_disclaimer
            }
            
            details.append(detail)
        
        return details

# ===========================================
# FUNGSI UTAMA (sama seperti sebelumnya)
# ===========================================
def main():
    # Initialize assistant dengan versi yang diperbaiki
    assistant = SimpleRAGPharmaAssistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None

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
        .evaluation-info {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .rag-score-card {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("💊 Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["🏠 Chatbot Obat", "📊 Evaluasi RAG"]
    )
    
    # HALAMAN CHATBOT
    if page == "🏠 Chatbot Obat":
        st.title("💊 Sistem Tanya Jawab Obat")
        st.markdown("Sistem informasi obat dengan data langsung dari **FDA API** dan terjemahan menggunakan **Gemini AI**")

        st.markdown("""
        <div class="fda-indicator">
            🏥 <strong>DATA RESMI FDA</strong> - Informasi obat langsung dari U.S. Food and Drug Administration
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💬 Percakapan")

        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-message">
                <h3>👋 Selamat Datang di Asisten Obat</h3>
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
                    "🚀 Tanya", 
                    use_container_width=True
                )
            
            with col_btn2:
                clear_btn = st.form_submit_button(
                    "🗑️ Hapus Chat", 
                    use_container_width=True
                )

        if submit_btn and user_input:
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🔍 Mengakses FDA API..."):
                answer, sources = assistant.ask_question(user_input)
                
                st.session_state.conversation_history.append({
                    'timestamp': datetime.now(),
                    'question': user_input,
                    'answer': answer,
                    'sources': [drug['nama'] for drug in sources],
                    'source': 'FDA API'
                })
                
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

        st.warning("""
        **⚠️ Peringatan Medis:** Informasi ini berasal dari database FDA AS dan untuk edukasi saja. 
        Selalu konsultasi dengan dokter atau apoteker sebelum menggunakan obat. 
        Obat mungkin memiliki nama merek berbeda di Indonesia.
        """)
    
    # HALAMAN EVALUASI RAG
    elif page == "📊 Evaluasi RAG":
        st.title("📊 Evaluasi Sistem RAG")
        st.markdown("**Fokus pada evaluasi komponen RETRIEVAL dan GENERATION dari RAG**")
        
        with st.expander("ℹ️ Tentang 2 Metrik Evaluasi RAG", expanded=True):
            st.markdown("""
            <div class="evaluation-info">
            ### **🎯 FOKUS EVALUASI RAG**
            
            **1. Mean Reciprocal Rank (MRR) - Evaluasi RETRIEVAL**
            - **Fungsi**: Mengukur akurasi sistem dalam menemukan obat yang benar dari query
            - **Target**: > 0.8 (80%)
            - **Baseline**: Samudra dkk. (2024): 0.930
            
            **2. Faithfulness - Evaluasi GENERATION**
            - **Fungsi**: Mengukur kesetiaan jawaban terhadap sumber data FDA
            - **Target**: > 0.85 (85%)
            - **Baseline**: Samudra dkk. (2024): 0.620
            </div>
            """, unsafe_allow_html=True)
        
        # Tombol evaluasi
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Jalankan Evaluasi RAG", use_container_width=True, type="primary"):
                with st.spinner("Menjalankan evaluasi pada 10 test cases..."):
                    st.session_state.evaluator = FocusedRAGEvaluator(assistant)
                    results = st.session_state.evaluator.run_evaluation()
                    st.session_state.evaluation_results = results
                    st.success("✅ Evaluasi RAG selesai!")
                    st.rerun()
        
        with col2:
            if st.button("📥 Simpan Hasil", use_container_width=True):
                if st.session_state.evaluation_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rag_evaluation_{timestamp}.json"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.evaluation_results, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"✅ Hasil disimpan ke `{filename}`")
                    
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = f.read()
                    
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=data,
                        file_name=filename,
                        mime="application/json"
                    )
                else:
                    st.warning("⚠️ Jalankan evaluasi terlebih dahulu!")
        
        with col3:
            if st.button("🔄 Reset Hasil", use_container_width=True):
                st.session_state.evaluation_results = None
                st.session_state.evaluator = None
                st.rerun()
        
        st.markdown("---")
        
        # Tampilkan hasil evaluasi
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            st.markdown(f"### 📈 Hasil Evaluasi RAG ({results['timestamp']})")
            st.markdown(f"**Test Cases:** {results['total_test_cases']} pertanyaan")
            
            # Tampilkan metrik
            col1, col2, col3 = st.columns([1, 1, 1])
            
            def get_score_color(score, target):
                if score >= target:
                    return "good-score"
                elif score >= target * 0.8:
                    return "medium-score"
                else:
                    return "poor-score"
            
            with col1:
                mrr = results["MRR"]
                color_class = get_score_color(mrr, 0.8)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MRR</div>
                    <div class="metric-value {color_class}">{mrr:.3f}</div>
                    <div>Retrieval Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"**Target:** >0.800 | **Baseline:** 0.930")
            
            with col2:
                faithfulness = results["Faithfulness"]
                color_class = get_score_color(faithfulness, 0.85)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Faithfulness</div>
                    <div class="metric-value {color_class}">{faithfulness:.3f}</div>
                    <div>Generation Reliability</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"**Target:** >0.850 | **Baseline:** 0.620")
            
            with col3:
                rag_score = results["RAG_Score"]
                if rag_score >= 0.8:
                    rag_color = "#4CAF50"
                    rag_status = "Excellent"
                elif rag_score >= 0.7:
                    rag_color = "#FF9800"
                    rag_status = "Good"
                else:
                    rag_color = "#F44336"
                    rag_status = "Needs Improvement"
                
                st.markdown(f"""
                <div class="rag-score-card">
                    <div style="font-size: 1em; opacity: 0.9;">RAG Score</div>
                    <div style="font-size: 2.5em; font-weight: bold;">{rag_score:.3f}</div>
                    <div style="font-size: 0.9em; font-weight: bold;">{rag_status}</div>
                    <div style="font-size: 0.8em; opacity: 0.8;">Average of 2 Metrics</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detail hasil
            with st.expander("📋 Detail Hasil Evaluasi"):
                st.json(results)
                
                if st.session_state.evaluator:
                    evaluator = st.session_state.evaluator
                    
                    # Contoh jawaban
                    st.markdown("### 🤖 Contoh Jawaban")
                    
                    sample_indices = random.sample(range(len(evaluator.test_set)), min(2, len(evaluator.test_set)))
                    
                    for idx in sample_indices:
                        test = evaluator.test_set[idx]
                        with st.spinner(f"Mengambil jawaban: '{test['question']}'..."):
                            answer, sources = assistant.ask_question(test["question"])
                            
                            with st.container():
                                st.markdown(f"**Test {test['id']}:** `{test['question']}`")
                                st.markdown("**Jawaban:**")
                                st.info(answer)
                                
                                if sources:
                                    st.success(f"✅ Sumber FDA ditemukan: {sources[0]['nama']}")
                                else:
                                    st.warning("⚠️ Tidak ada sumber FDA ditemukan")
                                
                                st.markdown("---")
        else:
            st.info("""
            **📝 Informasi Evaluasi RAG:**
            
            Sistem akan dievaluasi menggunakan **2 metrik inti RAG**:
            
            1. **MRR (Retrieval)** - Mengukur akurasi dalam menemukan obat yang relevan
            2. **Faithfulness (Generation)** - Mengukur kesetiaan jawaban ke sumber FDA
            
            **Test Cases:** 10 pertanyaan representatif tentang obat
            
            **Klik tombol 'Jalankan Evaluasi RAG' untuk memulai.**
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "💊 **Sistem Tanya Jawab Obat dengan RAG** • Evaluasi 2 Metrik Inti (MRR & Faithfulness)"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
