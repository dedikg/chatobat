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
# PERBAIKAN 1: TRANSLATION SERVICE
# ===========================================
class TranslationService:
    def __init__(self):
        self.available = gemini_available
    
    def translate_to_indonesian(self, text: str):
        """Translate text ke Bahasa Indonesia menggunakan Gemini - LEBIH AGRESIF"""
        if not self.available or not text or text == "Tidak tersedia":
            return text
        
        try:
            # Skip jika sudah mengandung kata-kata Indonesia atau angka saja
            indonesian_indicators = ['untuk', 'dengan', 'dalam', 'adalah', 'yang', 'dapat', 'akan', 'tidak']
            has_indonesian = any(word in text.lower() for word in indonesian_indicators)
            
            if has_indonesian and len(text.split()) < 20:
                # Kemungkinan sudah dalam Bahasa Indonesia
                return text
            
            # Kata-kata yang TIDAK BOLEH diterjemahkan
            preserve_words = [
                'mg', 'ml', 'g', 'kg', 'mg/kg', 'mg/mL', 'mg/tablet',
                'FDA', 'USP', 'BP', 'mL', 'hr', 'day', 'week', 'month',
                'acetaminophen', 'ibuprofen', 'paracetamol', 'omeprazole',
                'amoxicillin', 'hydrocodone', 'bitartrate'
            ]
            
            # Buat prompt yang lebih ketat
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Terjemahkan teks medis berikut ke Bahasa Indonesia dengan KETAT:
            
            TEKS ASLI: "{text}"
            
            PERATURAN:
            1. Hasil HARUS 100% Bahasa Indonesia
            2. PERTAHANKAN semua: angka, satuan (mg, ml, dll), nama obat asli
            3. JANGAN tambahkan informasi baru
            4. JANGAN ubah makna medis
            5. Gunakan bahasa formal tapi mudah dipahami pasien Indonesia
            6. Jika teks panjang, buat menjadi poin-poin
            7. JANGAN gunakan bahasa Inggris sama sekali dalam terjemahan
            
            CONTOH TERJEMAHAN YANG BENAR:
            - "Take 2 tablets every 6 hours" → "Minum 2 tablet setiap 6 jam"
            - "Do not exceed 4000 mg per day" → "Jangan melebihi 4000 mg per hari"
            - "May cause drowsiness" → "Dapat menyebabkan kantuk"
            - "Consult your doctor" → "Konsultasikan dengan dokter Anda"
            
            TERJEMAHAN BAHASA INDONESIA:
            """
            
            response = model.generate_content(prompt)
            translated = response.text.strip()
            
            # Bersihkan output
            translated = translated.replace('"', '').replace('Terjemahan:', '').strip()
            
            # Jika terjemahan terlalu pendek atau sama, return original
            if len(translated) < len(text) * 0.3 or translated.lower() == text.lower():
                return text
            
            # Pastikan tidak ada kata kunci Inggris yang seharusnya dipertahankan
            for word in preserve_words:
                if word.lower() in text.lower() and word.lower() not in translated.lower():
                    # Tempatkan kembali kata yang penting
                    import re
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    translated = pattern.sub(word, translated)
            
            return translated
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def is_mostly_english(self, text: str) -> bool:
        """Cek apakah teks sebagian besar berbahasa Inggris"""
        english_words = ['the', 'and', 'for', 'with', 'that', 'this', 'are', 'you', 'have', 'from']
        indonesian_words = ['yang', 'dengan', 'untuk', 'dalam', 'adalah', 'dari', 'pada', 'atau']
        
        text_lower = text.lower()
        eng_count = sum(1 for word in english_words if word in text_lower)
        ind_count = sum(1 for word in indonesian_words if word in text_lower)
        
        return eng_count > ind_count * 2  # Jika lebih banyak kata Inggris

# ===========================================
# PERBAIKAN 2: FDA API - CARI DATA LEBIH LENGKAP
# ===========================================
class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
    
    def get_drug_info(self, generic_name: str):
        """Ambil data obat langsung dari FDA API dengan pencarian lebih baik"""
        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 5  # Ambil lebih banyak hasil untuk dipilih yang terbaik
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    # Cari data yang paling lengkap
                    best_result = None
                    max_field_count = 0
                    
                    for result in data['results']:
                        field_count = self._count_complete_fields(result)
                        if field_count > max_field_count:
                            max_field_count = field_count
                            best_result = result
                    
                    if best_result:
                        return self._parse_fda_data(best_result, generic_name)
                    
                    # Jika tidak ada yang lengkap, ambil yang pertama
                    return self._parse_fda_data(data['results'][0], generic_name)
            
            # Coba dengan pencarian alternatif
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
            'warnings',
            'description',
            'purpose'
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
        # Coba dengan pencarian lebih umum
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
    """Parse data FDA dan langsung terjemahkan ke Bahasa Indonesia"""
    # ... [kode sebelumnya sama] ...
    
    drug_info = {
        "nama": generic_name.title(),
        "nama_generik": generic_name.title(),
        "merek_dagang": self._translate_if_needed(", ".join(openfda.get('brand_name', ['Tidak tersedia']))[:200]),
        "golongan": self._translate_if_needed(get_field('drug_class', "Tidak tersedia")[:100]),
        "indikasi": self._translate_if_needed(indications[:500] if indications != "Tidak tersedia" else "Tidak tersedia"),
        "dosis_dewasa": self._translate_if_needed(dosage[:500] if dosage != "Tidak tersedia" else "Tidak tersedia"),
        "efek_samping": self._translate_if_needed(side_effects[:500] if side_effects != "Tidak tersedia" else "Tidak tersedia"),
        "kontraindikasi": self._translate_if_needed(contraindications[:500] if contraindications != "Tidak tersedia" else "Tidak tersedia"),
        "interaksi": self._translate_if_needed(interactions[:500] if interactions != "Tidak tersedia" else "Tidak tersedia"),
        "peringatan": self._translate_if_needed(warnings[:500] if warnings != "Tidak tersedia" else "Tidak tersedia"),
        "bentuk_sediaan": self._translate_if_needed(", ".join(openfda.get('dosage_form', ['Tidak tersedia']))[:100]),
        "route_pemberian": self._translate_if_needed(", ".join(openfda.get('route', ['Tidak tersedia']))[:100]),
        "sumber": "FDA API",
        "bahasa": "Indonesia"  # Tambahkan flag bahasa
    }
    
    return drug_info

def _translate_if_needed(self, text: str) -> str:
    """Cek dan terjemahkan jika teks berbahasa Inggris"""
    if text == "Tidak tersedia" or not text:
        return text
    
    # Import translator di sini untuk menghindari circular import
    translator = TranslationService()
    
    # Cek apakah teks berbahasa Inggris
    english_indicators = ['the', 'and', 'for', 'with', 'should', 'may', 'can', 'will']
    indonesian_indicators = ['untuk', 'dengan', 'yang', 'adalah', 'dari', 'pada']
    
    text_lower = text.lower()
    eng_count = sum(1 for word in english_indicators if word in text_lower)
    ind_count = sum(1 for word in indonesian_indicators if word in text_lower)
    
    # Jika lebih banyak indikator Inggris, terjemahkan
    if eng_count > ind_count and len(text.split()) > 3:
        return translator.translate_to_indonesian(text)
    
    return text

# ===========================================
# PERBAIKAN 3: SIMPLE RAG ASSISTANT - IMPROVE PROMPT
# ===========================================
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
        
        fda_name = self.drug_detector.get_fda_name(drug_name)
        drug_info = self.fda_api.get_drug_info(fda_name)
        
        if drug_info:
            if drug_name != fda_name:
                drug_info['nama'] = drug_name.title()
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"
            
            # TERJEMAHKAN SEMUA FIELD YANG PERLU
            drug_info = self._translate_all_fields(drug_info)
            self.drugs_cache[drug_key] = drug_info
        
        return drug_info
    
    def _translate_all_fields(self, drug_info: dict):
        """Translate SEMUA field yang penting ke Bahasa Indonesia"""
        fields_to_translate = [
            'indikasi', 'dosis_dewasa', 'efek_samping', 
            'kontraindikasi', 'interaksi', 'peringatan',
            'golongan', 'bentuk_sediaan', 'route_pemberian'
        ]
        
        for field in fields_to_translate:
            if field in drug_info and drug_info[field] != "Tidak tersedia":
                # Untuk field yang panjang, translate bagian per bagian
                text = drug_info[field]
                if len(text) > 300:
                    # Split untuk teks panjang
                    parts = text.split('. ')
                    translated_parts = []
                    for part in parts[:10]:  # Batasi agar tidak terlalu panjang
                        if len(part) > 20:  # Hanya translate bagian yang cukup panjang
                            translated = self.translator.translate_to_indonesian(part)
                            translated_parts.append(translated)
                        else:
                            translated_parts.append(part)
                    
                    drug_info[field] = '. '.join(translated_parts)
                else:
                    translated = self.translator.translate_to_indonesian(text)
                    if translated != text:  # Hanya update jika berbeda
                        drug_info[field] = translated
        
        return drug_info
    
    def _rag_retrieve(self, query, top_k=3):
        """Retrieve relevant information menggunakan FDA API dengan drug detection yang lebih baik"""
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
            
            # Hanya tampilkan field yang ada informasinya
            fields_to_display = [
                ('Golongan', 'golongan'),
                ('Indikasi', 'indikasi'),
                ('Dosis Dewasa', 'dosis_dewasa'),
                ('Efek Samping', 'efek_samping'),
                ('Kontraindikasi', 'kontraindikasi'),
                ('Interaksi', 'interaksi'),
                ('Peringatan', 'peringatan'),
                ('Bentuk Sediaan', 'bentuk_sediaan')
            ]
            
            for label, field in fields_to_display:
                if field in drug_info and drug_info[field] != "Tidak tersedia":
                    # Potong teks jika terlalu panjang
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
            return "Maaf, terjadi error dalam sistem. Silakan coba lagi.", []
    
    def _generate_rag_response(self, question, context):
    """Generate response menggunakan RAG pattern dengan Gemini - PASTIKAN BAHASA INDONESIA"""
    if not gemini_available:
        # Jika Gemini tidak tersedia, format context dalam Bahasa Indonesia
        return self._format_context_for_display(context)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # PROMPT YANG LEBIH KETAT - PASTIKAN BAHASA INDONESIA
        prompt = f"""
        Anda adalah asisten farmasi profesional yang HANYA menggunakan BAHASA INDONESIA.
        
        ## INFORMASI RESMI DARI FDA:
        {context}
        
        ## PERTANYAAN PENGGUNA:
        {question}
        
        ## ATURAN KETAT:
        1. JAWAB DENGAN BAHASA INDONESIA 100% - jangan gunakan bahasa Inggris sama sekali
        2. Jawaban HARUS berdasarkan informasi FDA di atas
        3. Jangan tambahkan informasi yang tidak ada dalam data FDA
        4. Sederhanakan bahasa teknis medis menjadi bahasa yang mudah dipahami pasien Indonesia
        5. Format jawaban: ringkas, jelas, dan langsung ke poin
        6. Selalu tambahkan: "Konsultasikan dengan dokter atau apoteker sebelum menggunakan obat"
        
        ## CONTOH JAWABAN YANG BENAR:
        Pertanyaan: "Apa dosis paracetamol?"
        Jawaban: "Berdasarkan data FDA, dosis paracetamol biasanya ... [jelaskan berdasarkan data FDA]. Konsultasikan dengan dokter atau apoteker sebelum menggunakan obat."
        
        ## JAWABAN (HANYA BAHASA INDONESIA):
        """
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # VALIDASI: Pastikan jawaban dalam Bahasa Indonesia
        # Cek apakah ada banyak kata Inggris yang tidak perlu
        english_words = ['the', 'and', 'for', 'with', 'use', 'dosage', 'tablet', 'mg', 'ml']
        english_count = sum(1 for word in english_words if word.lower() in answer.lower())
        
        if english_count > 5:  # Terlalu banyak kata Inggris
            # Terjemahkan paksa ke Bahasa Indonesia
            answer = self.translator.translate_to_indonesian(answer)
        
        return answer
        
    except Exception as e:
        # Fallback: format context dengan Bahasa Indonesia
        return self._format_context_for_display(context)

def _format_context_for_display(self, context):
    """Format context untuk display jika Gemini gagal - PASTIKAN BAHASA INDONESIA"""
    lines = context.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.startswith('### OBAT'):
            formatted_lines.append(line)
        elif line.startswith('- **'):
            # Terjemahkan field label ke Bahasa Indonesia jika perlu
            translations = {
                'Indications and Usage': 'Indikasi',
                'Dosage and Administration': 'Dosis Dewasa',
                'Adverse Reactions': 'Efek Samping',
                'Contraindications': 'Kontraindikasi',
                'Drug Interactions': 'Interaksi Obat',
                'Warnings': 'Peringatan',
                'Description': 'Deskripsi',
                'Purpose': 'Tujuan'
            }
            
            for eng, ind in translations.items():
                if eng in line:
                    line = line.replace(eng, ind)
            
            # Terjemahkan konten jika masih dalam Inggris
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    label, content = parts
                    # Terjemahkan content jika perlu
                    content_ind = self.translator.translate_to_indonesian(content)
                    line = f"{label}: {content_ind}"
            
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    formatted_context = '\n'.join(formatted_lines)
    
    return f"**Informasi dari FDA (diterjemahkan):**\n\n{formatted_context}\n\n⚠️ **Peringatan:** Informasi ini untuk edukasi. Konsultasikan dengan dokter sebelum menggunakan obat."

# ===========================================
# KELAS LAINNYA (TIDAK BERUBAH)
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
# KELAS EVALUASI (TIDAK BERUBAH)
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
                "RAG_Score": float((mrr_score + faithfulness_score) / 2)  # Simple average
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
# FUNGSI UTAMA (TIDAK BERUBAH)
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

    # Custom CSS (sama seperti sebelumnya)
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
    
    # ===========================================
    # HALAMAN CHATBOT
    # ===========================================
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
    
    # ===========================================
    # HALAMAN EVALUASI RAG
    # ===========================================
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
