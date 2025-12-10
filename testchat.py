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
# CLEAN TRANSLATION SERVICE
# ===========================================
class CleanTranslationService:
    def __init__(self):
        self.available = gemini_available
        self.translation_cache = {}
    
    def translate_to_indonesian(self, text: str):
        """Terjemahkan teks ke Bahasa Indonesia dengan validasi ketat"""
        if not self.available or not text or text == "Tidak tersedia":
            return text
        
        # Skip jika sudah pendek atau angka saja
        if len(text.strip()) < 25:
            return text
        
        # Cek cache
        text_hash = hash(text)
        if text_hash in self.translation_cache:
            return self.translation_cache[text_hash]
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Terjemahkan teks medis berikut ke Bahasa Indonesia dengan AKURAT:
            
            "{text}"
            
            PERATURAN PENTING:
            1. Hasil HARUS dalam Bahasa Indonesia yang benar
            2. PERTAHANKAN: semua angka, satuan (mg, ml, g, tablet, kapsul), nama obat
            3. JANGAN ubah struktur kalimat asli
            4. Terjemahkan kata-kata Inggris ke Indonesia dengan tepat
            5. Hasil harus natural dan mudah dipahami
            6. JANGAN tambahkan informasi baru
            7. JANGAN membuat teks menjadi berantakan
            
            CONTOH TERJEMAHAN YANG BENAR:
            "INDICATIONS AND USAGE: Amoxicillin is a penicillin antibiotic"
            → "INDIKASI DAN PENGGUNAAN: Amoxicillin adalah antibiotik penisilin"
            
            "DOSAGE AND ADMINISTRATION: 500 mg every 8 hours"
            → "DOSIS DAN CARA PEMAKAIAN: 500 mg setiap 8 jam"
            
            "ADVERSE REACTIONS: nausea, vomiting, diarrhea"
            → "EFEK SAMPING: mual, muntah, diare"
            
            TERJEMAHAN BAHASA INDONESIA:
            """
            
            response = model.generate_content(prompt)
            translated = response.text.strip()
            
            # Bersihkan hasil
            translated = translated.replace('"', '').strip()
            
            # Validasi kualitas terjemahan
            if self._is_translation_quality_good(translated, text):
                self.translation_cache[text_hash] = translated
                return translated
            else:
                # Jika terjemahan buruk, gunakan fallback
                return self._simple_safe_translation(text)
                
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return self._simple_safe_translation(text)
    
    def _is_translation_quality_good(self, translated: str, original: str) -> bool:
        """Validasi kualitas terjemahan"""
        if not translated or len(translated) < 10:
            return False
        
        # Cek jika terjemahan terlalu pendek
        if len(translated) < len(original) * 0.3:
            return False
        
        # Cek karakter aneh atau pola rusak
        bad_patterns = [
            r'\b\d+\s+[a-z]+\s+\d+\s+[a-z]+\b',  # Pola "1 dalam 2 DOS"
            r'[a-z][A-Z][a-z]',  # Pola camelCase rusak
            r'\b[a-z]{10,}\b',  # Kata sangat panjang
            r'\b\d+[a-zA-Z]+\d+\b'  # Angka+teks+angka
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, translated):
                return False
        
        # Cek jika masih banyak kata Inggris yang seharusnya diterjemahkan
        english_medical_terms = [
            'indications', 'usage', 'dosage', 'administration',
            'adverse', 'reactions', 'contraindications', 'warnings'
        ]
        
        translated_lower = translated.lower()
        eng_count = sum(1 for term in english_medical_terms if term in translated_lower)
        
        # Jika masih banyak istilah Inggris, terjemahan kurang baik
        return eng_count < 3
    
    def _simple_safe_translation(self, text: str) -> str:
        """Terjemahan sederhana dan aman"""
        # Hanya terjemahkan bagian-bagian kunci
        translations = {
            # Header FDA
            'INDICATIONS AND USAGE': 'INDIKASI DAN PENGGUNAAN:',
            'DOSAGE AND ADMINISTRATION': 'DOSIS DAN CARA PEMAKAIAN:',
            'ADVERSE REACTIONS': 'EFEK SAMPING:',
            'CONTRAINDICATIONS': 'KONTRAINDIKASI:',
            'WARNINGS': 'PERINGATAN:',
            'DRUG INTERACTIONS': 'INTERAKSI OBAT:',
            'PRECAUTIONS': 'TINDAKAN PENCEGAHAN:',
            
            # Kata umum
            ' is indicated for ': ' diindikasikan untuk ',
            ' should be ': ' sebaiknya ',
            ' may cause ': ' dapat menyebabkan ',
            ' must not ': ' tidak boleh ',
            ' do not ': ' jangan ',
            ' take ': ' minum ',
            ' use ': ' gunakan ',
            ' every ': ' setiap ',
            ' hours ': ' jam ',
            ' days ': ' hari ',
            ' weeks ': ' minggu ',
            ' months ': ' bulan ',
            ' mg ': ' mg ',
            ' ml ': ' ml ',
            ' tablet': ' tablet',
            ' tablets': ' tablet',
            ' capsule': ' kapsul',
            ' capsules': ' kapsul'
        }
        
        result = text
        for eng, indo in translations.items():
            result = result.replace(eng, indo)
            result = result.replace(eng.lower(), indo.lower())
            result = result.replace(eng.title(), indo.title())
        
        return result
    
    def translate_fda_field(self, field_name: str, content: str) -> str:
        """Terjemahkan field FDA tertentu dengan penanganan khusus"""
        if not content or content == "Tidak tersedia":
            return "Tidak tersedia"
        
        # Map nama field ke Bahasa Indonesia
        field_translations = {
            'indications_and_usage': 'INDIKASI',
            'dosage_and_administration': 'DOSIS',
            'adverse_reactions': 'EFEK SAMPING',
            'contraindications': 'KONTRAINDIKASI',
            'drug_interactions': 'INTERAKSI OBAT',
            'warnings': 'PERINGATAN',
            'description': 'DESKRIPSI',
            'purpose': 'TUJUAN'
        }
        
        field_label = field_translations.get(field_name.lower(), field_name.upper())
        
        # Terjemahkan konten
        translated_content = self.translate_to_indonesian(content)
        
        return f"{field_label}: {translated_content}"

# ===========================================
# FDA API DENGAN PARSING BERSIH
# ===========================================
class CleanFDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        self.translator = CleanTranslationService()
        self.cache = {}
        self.request_timeout = 15
    
    def get_drug_info(self, generic_name: str):
        """Ambil data obat dari FDA API dengan parsing bersih"""
        cache_key = generic_name.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print(f"🔍 Mencari data FDA untuk: {generic_name}")
        
        # Coba beberapa strategi pencarian
        search_strategies = [
            f'openfda.generic_name:"{generic_name}"',
            f'openfda.generic_name:{generic_name}',
            f'_exists_:openfda.generic_name AND {generic_name}',
            f'brand_name:{generic_name}',
            f'substance_name:{generic_name}'
        ]
        
        for search_query in search_strategies:
            params = {'search': search_query, 'limit': 3}
            
            try:
                response = requests.get(self.base_url, params=params, timeout=self.request_timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        print(f"✅ Data ditemukan dengan query: {search_query}")
                        drug_info = self._parse_fda_data_clean(data['results'][0], generic_name)
                        if drug_info:
                            self.cache[cache_key] = drug_info
                            return drug_info
            
            except Exception as e:
                print(f"⚠️ Search error: {e}")
                continue
        
        print(f"❌ Tidak ada data ditemukan untuk: {generic_name}")
        return None
    
    def _parse_fda_data_clean(self, fda_data: dict, generic_name: str):
        """Parse data FDA dengan approach yang bersih"""
        print(f"📝 Parsing data untuk: {generic_name}")
        
        openfda = fda_data.get('openfda', {})
        
        # Ekstrak data dasar
        drug_info = {
            "nama": self._capitalize_name(generic_name),
            "nama_generik": self._capitalize_name(generic_name),
            "merek_dagang": self._extract_brand_names(openfda),
            "golongan": self._extract_drug_class(fda_data),
            "bentuk_sediaan": self._extract_dosage_forms(openfda),
            "route_pemberian": self._extract_routes(openfda),
            "sumber": "FDA API",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Ekstrak dan terjemahkan field medis penting
        medical_fields = self._extract_medical_fields(fda_data)
        drug_info.update(medical_fields)
        
        return drug_info
    
    def _capitalize_name(self, name: str):
        """Kapitalisasi nama obat dengan benar"""
        if not name:
            return "Tidak tersedia"
        
        # Kapitalisasi setiap kata
        words = name.split()
        capitalized_words = []
        
        for word in words:
            if word.lower() in ['and', 'or', 'the', 'of', 'for']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        
        return ' '.join(capitalized_words)
    
    def _extract_brand_names(self, openfda: dict):
        """Ekstrak nama merek"""
        if 'brand_name' in openfda and openfda['brand_name']:
            brands = openfda['brand_name']
            if isinstance(brands, list):
                # Ambil maksimal 3 merek, kapitalisasi
                brand_list = [str(b).title() for b in brands[:3]]
                return ', '.join(brand_list)
        return "Tidak tersedia"
    
    def _extract_drug_class(self, fda_data: dict):
        """Ekstrak golongan obat"""
        if 'drug_class' in fda_data and fda_data['drug_class']:
            drug_class = fda_data['drug_class']
            if isinstance(drug_class, list):
                return drug_class[0]
            return str(drug_class)
        return "Tidak tersedia"
    
    def _extract_dosage_forms(self, openfda: dict):
        """Ekstrak bentuk sediaan"""
        if 'dosage_form' in openfda and openfda['dosage_form']:
            forms = openfda['dosage_form']
            if isinstance(forms, list):
                form_list = [str(f).title() for f in forms[:2]]
                return ', '.join(form_list)
        return "Tidak tersedia"
    
    def _extract_routes(self, openfda: dict):
        """Ekstrak route pemberian"""
        if 'route' in openfda and openfda['route']:
            routes = openfda['route']
            if isinstance(routes, list):
                route_list = [str(r).title() for r in routes[:2]]
                return ', '.join(route_list)
        return "Tidak tersedia"
    
    def _extract_medical_fields(self, fda_data: dict):
        """Ekstrak dan terjemahkan field medis penting"""
        fields = {
            'indikasi': self._extract_and_translate_field(fda_data, 'indications_and_usage'),
            'dosis_dewasa': self._extract_and_translate_field(fda_data, 'dosage_and_administration'),
            'efek_samping': self._extract_and_translate_field(fda_data, 'adverse_reactions'),
            'kontraindikasi': self._extract_and_translate_field(fda_data, 'contraindications'),
            'interaksi': self._extract_and_translate_field(fda_data, 'drug_interactions'),
            'peringatan': self._extract_and_translate_field(fda_data, 'warnings')
        }
        
        return fields
    
    def _extract_and_translate_field(self, fda_data: dict, field_name: str):
        """Ekstrak dan terjemahkan field tertentu"""
        if field_name not in fda_data:
            return "Tidak tersedia"
        
        field_value = fda_data[field_name]
        
        # Ekstrak teks
        extracted_text = self._extract_text_from_field(field_value)
        if not extracted_text or extracted_text == "Tidak tersedia":
            return "Tidak tersedia"
        
        # Terjemahkan
        translated = self.translator.translate_to_indonesian(extracted_text)
        
        # Validasi hasil terjemahan
        if self._is_translation_valid(translated):
            return translated
        else:
            # Gunakan terjemahan sederhana
            return self._get_simple_field_summary(field_name, extracted_text)
    
    def _extract_text_from_field(self, field_value):
        """Ekstrak teks dari field FDA"""
        if isinstance(field_value, list):
            if field_value:
                # Gabungkan jika ada multiple entries
                text = ' '.join([str(v) for v in field_value if v])
                return self._clean_text(text[:400])  # Batasi panjang
            return "Tidak tersedia"
        elif field_value:
            text = str(field_value)
            return self._clean_text(text[:400])
        
        return "Tidak tersedia"
    
    def _clean_text(self, text: str):
        """Bersihkan teks dari karakter aneh"""
        if not text:
            return text
        
        # Hapus karakter kontrol
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Hapus multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_translation_valid(self, text: str) -> bool:
        """Validasi apakah terjemahan valid"""
        if not text or text == "Tidak tersedia":
            return False
        
        # Cek pola teks rusak
        bad_patterns = [
            r'\b\d+\s+[a-z]+\s+\d+',  # "1 dalam 2"
            r'[a-z][A-Z]{2,}[a-z]',  # camelCase rusak
            r'\b[a-z]{12,}\b',  # kata sangat panjang
            r'\b\w+\d+\w+\b'  # teks+angka+teks
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, text):
                return False
        
        return True
    
    def _get_simple_field_summary(self, field_name: str, original_text: str):
        """Dapatkan ringkasan sederhana untuk field"""
        # Ambil hanya kalimat pertama yang bermakna
        sentences = re.split(r'[.!?]', original_text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:
                # Hapus angka di awal
                sentence = re.sub(r'^\d+\s*', '', sentence)
                meaningful_sentences.append(sentence)
                if len(meaningful_sentences) >= 1:  # Maksimal 1 kalimat untuk ringkasan
                    break
        
        if not meaningful_sentences:
            return "Tidak tersedia"
        
        summary = meaningful_sentences[0]
        
        # Terjemahkan header jika ada
        header_map = {
            'indications_and_usage': 'Digunakan untuk: ',
            'dosage_and_administration': 'Dosis: ',
            'adverse_reactions': 'Efek samping: ',
            'contraindications': 'Kontraindikasi: ',
            'drug_interactions': 'Interaksi obat: ',
            'warnings': 'Peringatan: '
        }
        
        prefix = header_map.get(field_name, '')
        
        # Terjemahkan kata kunci sederhana
        simple_translations = {
            'take': 'minum',
            'use': 'gunakan',
            'every': 'setiap',
            'hours': 'jam',
            'days': 'hari',
            'mg': 'mg',
            'ml': 'ml',
            'tablet': 'tablet',
            'capsule': 'kapsul'
        }
        
        result = summary
        for eng, indo in simple_translations.items():
            result = re.sub(rf'\b{eng}\b', indo, result, flags=re.IGNORECASE)
        
        return f"{prefix}{result}"

# ===========================================
# SIMPLE RAG ASSISTANT - OUTPUT BERSIH
# ===========================================
class CleanRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = CleanFDADrugAPI()
        self.translator = CleanTranslationService()
        self.drug_detector = EnhancedDrugDetector()
        self.drugs_cache = {}
    
    def ask_question(self, question: str):
        """Jawab pertanyaan tentang obat dengan output bersih"""
        print(f"\n{'='*60}")
        print(f"🤔 PERTANYAAN: {question}")
        
        try:
            # 1. Deteksi obat dari pertanyaan
            detected_drugs = self.drug_detector.detect_drug_from_query(question)
            
            if not detected_drugs:
                print("❌ Tidak ada obat terdeteksi")
                return self._get_no_drug_found_response(), []
            
            # 2. Ambil informasi obat
            drug_info_list = []
            for detected in detected_drugs[:2]:  # Maksimal 2 obat
                drug_name = detected['drug_name']
                print(f"💊 Memproses: {drug_name}")
                
                drug_info = self._get_drug_info_clean(drug_name)
                if drug_info:
                    drug_info_list.append(drug_info)
            
            if not drug_info_list:
                print("❌ Tidak ada informasi obat ditemukan")
                return self._get_no_info_found_response(), []
            
            print(f"✅ Ditemukan {len(drug_info_list)} obat")
            
            # 3. Generate jawaban berdasarkan pertanyaan
            answer = self._generate_clean_answer(question, drug_info_list)
            
            return answer, drug_info_list
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return "Maaf, terjadi kesalahan dalam sistem. Silakan coba lagi.", []
    
    def _get_drug_info_clean(self, drug_name: str):
        """Dapatkan informasi obat dengan validasi"""
        cache_key = drug_name.lower()
        
        if cache_key in self.drugs_cache:
            return self.drugs_cache[cache_key]
        
        # Dapatkan nama FDA yang benar
        fda_name = self.drug_detector.get_fda_name(drug_name)
        
        # Ambil data dari FDA API
        drug_info = self.fda_api.get_drug_info(fda_name)
        
        if drug_info:
            # Tambahkan catatan jika nama berbeda
            if drug_name.lower() != fda_name.lower():
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"
            
            # Validasi dan bersihkan data
            drug_info = self._validate_drug_info(drug_info)
            
            # Simpan ke cache
            self.drugs_cache[cache_key] = drug_info
        
        return drug_info
    
    def _validate_drug_info(self, drug_info: dict):
        """Validasi dan bersihkan informasi obat"""
        # Field yang perlu divalidasi
        fields_to_validate = [
            'indikasi', 'dosis_dewasa', 'efek_samping',
            'kontraindikasi', 'interaksi', 'peringatan'
        ]
        
        for field in fields_to_validate:
            if field in drug_info and drug_info[field] != "Tidak tersedia":
                text = drug_info[field]
                
                # Cek dan perbaiki jika ada masalah
                if self._has_text_issues(text):
                    drug_info[field] = self._fix_text_issues(text)
        
        return drug_info
    
    def _has_text_issues(self, text: str) -> bool:
        """Cek apakah teks memiliki masalah"""
        issues = [
            r'\b\d+\s+[a-z]+\s+\d+\s+[a-z]+',  # "1 dalam 2 DOS"
            r'[a-z][A-Z]{2,}[a-z]',  # camelCase rusak
            r'\b[a-z]{15,}\b',  # kata sangat panjang
            r'\b\w+\d{2,}\w+\b'  # teks+angka+teks
        ]
        
        for pattern in issues:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _fix_text_issues(self, text: str) -> str:
        """Perbaiki masalah pada teks"""
        # Hapus angka di awal kalimat
        text = re.sub(r'^\d+\s*', '', text)
        
        # Hapus pola "1 dalam 2" dll
        text = re.sub(r'\b\d+\s+[a-z]+\s+\d+\s+', '', text)
        
        # Hapus kata yang sangat panjang
        words = text.split()
        clean_words = []
        
        for word in words:
            if len(word) < 20:  # Hanya ambil kata yang wajar panjangnya
                clean_words.append(word)
        
        return ' '.join(clean_words)
    
    def _generate_clean_answer(self, question: str, drug_info_list: list):
        """Generate jawaban yang bersih dan mudah dipahami"""
        question_lower = question.lower()
        
        # Deteksi tipe pertanyaan
        question_type = self._detect_question_type(question_lower)
        
        # Bangun jawaban berdasarkan tipe pertanyaan
        answer_parts = []
        
        for drug_info in drug_info_list:
            drug_name = drug_info['nama']
            
            # Header untuk obat ini
            answer_parts.append(f"**{drug_name}**")
            
            if 'catatan' in drug_info:
                answer_parts.append(f"*{drug_info['catatan']}*")
            
            # Tambahkan informasi berdasarkan tipe pertanyaan
            if question_type == 'dosis':
                if drug_info['dosis_dewasa'] != "Tidak tersedia":
                    answer_parts.append(f"**Dosis:** {drug_info['dosis_dewasa']}")
                else:
                    answer_parts.append("**Dosis:** Informasi tidak tersedia")
            
            elif question_type == 'efek_samping':
                if drug_info['efek_samping'] != "Tidak tersedia":
                    answer_parts.append(f"**Efek Samping:** {drug_info['efek_samping']}")
                else:
                    answer_parts.append("**Efek Samping:** Informasi tidak tersedia")
            
            elif question_type == 'indikasi':
                if drug_info['indikasi'] != "Tidak tersedia":
                    answer_parts.append(f"**Indikasi:** {drug_info['indikasi']}")
                else:
                    answer_parts.append("**Indikasi:** Informasi tidak tersedia")
            
            elif question_type == 'kontraindikasi':
                if drug_info['kontraindikasi'] != "Tidak tersedia":
                    answer_parts.append(f"**Kontraindikasi:** {drug_info['kontraindikasi']}")
                else:
                    answer_parts.append("**Kontraindikasi:** Informasi tidak tersedia")
            
            elif question_type == 'interaksi':
                if drug_info['interaksi'] != "Tidak tersedia":
                    answer_parts.append(f"**Interaksi Obat:** {drug_info['interaksi']}")
                else:
                    answer_parts.append("**Interaksi Obat:** Informasi tidak tersedia")
            
            else:
                # Jawaban umum
                if drug_info['indikasi'] != "Tidak tersedia":
                    answer_parts.append(f"**Kegunaan:** {drug_info['indikasi'][:150]}...")
                
                if drug_info['dosis_dewasa'] != "Tidak tersedia":
                    answer_parts.append(f"**Dosis Umum:** {drug_info['dosis_dewasa'][:100]}...")
            
            answer_parts.append("")  # Spasi antara obat
        
        # Tambahkan disclaimer
        answer_parts.append("---")
        answer_parts.append("**Sumber:** Data resmi dari U.S. Food and Drug Administration (FDA)")
        answer_parts.append("**Penting:** Informasi ini untuk edukasi. Konsultasikan dengan dokter sebelum menggunakan obat.")
        
        return '\n'.join(answer_parts)
    
    def _detect_question_type(self, question: str) -> str:
        """Deteksi tipe pertanyaan"""
        question_patterns = {
            'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai', 'berapa mg', 'berapa ml'],
            'efek_samping': ['efek samping', 'side effect', 'bahaya', 'efeknya', 'akibat'],
            'indikasi': ['untuk apa', 'kegunaan', 'manfaat', 'indikasi', 'fungsi', 'guna'],
            'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan', 'kontraindikasi'],
            'interaksi': ['interaksi', 'bereaksi', 'makanan', 'minuman', 'interaksinya'],
            'peringatan': ['peringatan', 'warning', 'hati-hati', 'waspada']
        }
        
        for qtype, keywords in question_patterns.items():
            for keyword in keywords:
                if keyword in question:
                    return qtype
        
        return 'umum'
    
    def _get_no_drug_found_response(self):
        """Response ketika tidak ada obat terdeteksi"""
        available_drugs = self.drug_detector.get_all_available_drugs()[:8]
        
        # Konversi ke nama Indonesia yang familiar
        indo_names = []
        for drug in available_drugs:
            if drug == 'paracetamol':
                indo_names.append('parasetamol')
            elif drug == 'amoxicillin':
                indo_names.append('amoksisilin')
            elif drug == 'omeprazole':
                indo_names.append('omeprazol')
            else:
                indo_names.append(drug)
        
        drug_list = ', '.join(indo_names)
        
        return f"""
Tidak ada obat yang terdeteksi dalam pertanyaan Anda.

**Obat yang tersedia dalam sistem:**
{drug_list}

**Contoh pertanyaan:**
- "Apa dosis parasetamol?"
- "Efek samping amoksisilin?"
- "Untuk apa omeprazol digunakan?"
"""
    
    def _get_no_info_found_response(self):
        """Response ketika tidak ada informasi ditemukan"""
        return """
Maaf, informasi tentang obat tersebut tidak ditemukan dalam database FDA.

Mungkin karena:
1. Nama obat ditulis berbeda
2. Obat tidak terdaftar di FDA Amerika
3. Data tidak tersedia dalam format yang dapat diakses

Coba gunakan nama generik obat atau tanyakan tentang obat lain.
"""

# ===========================================
# ENHANCED DRUG DETECTOR
# ===========================================
class EnhancedDrugDetector:
    def __init__(self):
        self.drug_dictionary = {
            'paracetamol': ['acetaminophen', 'paracetamol', 'panadol', 'sanmol', 'tempra', 'parasetamol'],
            'omeprazole': ['omeprazole', 'prilosec', 'losec', 'omepron', 'omeprazol'],
            'amoxicillin': ['amoxicillin', 'amoxilin', 'amoxan', 'moxigra', 'amoksisilin'],
            'ibuprofen': ['ibuprofen', 'proris', 'arthrifen', 'ibufar', 'ibuprom'],
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
            'salbutamol': 'albuterol',
            'parasetamol': 'acetaminophen',
            'amoksisilin': 'amoxicillin',
            'omeprazol': 'omeprazole'
        }
    
    def detect_drug_from_query(self, query: str):
        """Deteksi nama obat dari query"""
        query_lower = query.lower()
        detected_drugs = []
        
        for drug_name, aliases in self.drug_dictionary.items():
            for alias in aliases:
                if alias.lower() in query_lower:
                    detected_drugs.append({
                        'drug_name': drug_name,
                        'fda_name': self.fda_name_mapping.get(drug_name, drug_name),
                        'alias_found': alias,
                        'confidence': 'high' if alias == drug_name else 'medium'
                    })
                    break
        
        return detected_drugs
    
    def get_all_available_drugs(self):
        """Dapatkan semua obat yang tersedia"""
        return list(self.drug_dictionary.keys())
    
    def get_fda_name(self, drug_name: str):
        """Dapatkan nama FDA untuk obat"""
        return self.fda_name_mapping.get(drug_name, drug_name)

# ===========================================
# EVALUATOR
# ===========================================
class FocusedRAGEvaluator:
    def __init__(self, assistant):
        self.assistant = assistant
        self.test_set = [
            {"id": 1, "question": "Apa dosis paracetamol?", "expected_drug": "paracetamol"},
            {"id": 2, "question": "Efek samping amoxicillin?", "expected_drug": "amoxicillin"},
            {"id": 3, "question": "Untuk apa omeprazole digunakan?", "expected_drug": "omeprazole"},
            {"id": 4, "question": "Apa kontraindikasi ibuprofen?", "expected_drug": "ibuprofen"},
            {"id": 5, "question": "Interaksi obat metformin?", "expected_drug": "metformin"},
            {"id": 6, "question": "Berapa dosis atorvastatin?", "expected_drug": "atorvastatin"},
            {"id": 7, "question": "Efek samping simvastatin?", "expected_drug": "simvastatin"},
            {"id": 8, "question": "Kegunaan lansoprazole?", "expected_drug": "lansoprazole"},
            {"id": 9, "question": "Peringatan penggunaan aspirin?", "expected_drug": "aspirin"},
            {"id": 10, "question": "Dosis cetirizine untuk dewasa?", "expected_drug": "cetirizine"}
        ]
    
    def run_evaluation(self):
        """Jalankan evaluasi"""
        try:
            mrr = self.calculate_mrr()
            faithfulness = self.calculate_faithfulness()
            
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_set),
                "MRR": float(mrr),
                "Faithfulness": float(faithfulness),
                "RAG_Score": float((mrr + faithfulness) / 2)
            }
            
        except Exception as e:
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "MRR": 0,
                "Faithfulness": 0,
                "RAG_Score": 0
            }
    
    def calculate_mrr(self):
        """Hitung MRR"""
        reciprocal_ranks = []
        
        for test in self.test_set:
            detected_drugs = self.assistant.drug_detector.detect_drug_from_query(test["question"])
            
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
        """Hitung Faithfulness"""
        faithful_scores = []
        
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            answer_lower = answer.lower()
            
            criteria_scores = []
            
            # Sumber Data
            if sources and len(sources) > 0:
                criteria_scores.append(0.4)
            else:
                criteria_scores.append(0)
            
            # Referensi FDA
            if any(indicator in answer_lower for indicator in ["fda", "food and drug"]):
                criteria_scores.append(0.25)
            else:
                criteria_scores.append(0)
            
            # Tidak ada informasi fiktif
            if not any(indicator in answer_lower for indicator in ["menurut saya", "biasanya", "umumnya"]):
                criteria_scores.append(0.20)
            else:
                criteria_scores.append(0)
            
            # Disclaimer medis
            if any(indicator in answer_lower for indicator in ["dokter", "apoteker", "konsultasi"]):
                criteria_scores.append(0.15)
            else:
                criteria_scores.append(0)
            
            total_score = sum(criteria_scores)
            faithful_scores.append(min(total_score, 1.0))
        
        return np.mean(faithful_scores) if faithful_scores else 0

# ===========================================
# FUNGSI UTAMA STREAMLIT
# ===========================================
def main():
    # Initialize assistant
    assistant = CleanRAGPharmaAssistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
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
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: #856404;
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
        st.markdown("Sistem informasi obat dengan data langsung dari **FDA API**")
        
        st.markdown("""
        <div class="fda-indicator">
            🏥 <strong>DATA RESMI FDA</strong> - Informasi obat langsung dari U.S. Food and Drug Administration
            <br>🇮🇩 <strong>100% BAHASA INDONESIA</strong> - Semua informasi dalam Bahasa Indonesia
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💬 Percakapan")
        
        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-message">
                <h3>👋 Selamat Datang di Asisten Obat</h3>
                <p>Dapatkan informasi obat <strong>langsung dari database resmi FDA</strong></p>
                <p><strong>💡 Contoh pertanyaan:</strong></p>
                <p>"Apa dosis parasetamol?" | "Efek samping amoksisilin?" | "Interaksi obat omeprazol?"</p>
                <p>"Untuk apa metformin digunakan?" | "Peringatan penggunaan ibuprofen?"</p>
                <p><em>📝 Catatan: Semua jawaban dalam Bahasa Indonesia</em></p>
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
                        <div class="message-time">{message["timestamp"]} • Sumber: FDA API 🇮🇩</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Tulis pertanyaan Anda tentang obat:",
                placeholder="Contoh: Apa dosis paracetamol? Efek samping amoxicillin?",
                key="user_input"
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                submit_btn = st.form_submit_button("🚀 Tanya", use_container_width=True, type="primary")
            
            with col2:
                clear_btn = st.form_submit_button("🗑️ Hapus Chat", use_container_width=True)
        
        if submit_btn and user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🔍 Mencari informasi dari FDA..."):
                answer, sources = assistant.ask_question(user_input)
                
                st.session_state.messages.append({
                    "role": "bot",
                    "content": answer,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            
            st.rerun()
        
        if clear_btn:
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ PERINGATAN MEDIS:</strong>
            <ul>
                <li>Informasi ini berasal dari database FDA Amerika Serikat</li>
                <li>Informasi untuk tujuan edukasi dan referensi</li>
                <li><strong>SELALU KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT</strong></li>
                <li>Dosis dan indikasi dapat berbeda untuk setiap pasien</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # HALAMAN EVALUASI
    else:
        st.title("📊 Evaluasi Sistem")
        
        if st.button("🚀 Jalankan Evaluasi", type="primary"):
            with st.spinner("Menjalankan evaluasi..."):
                evaluator = FocusedRAGEvaluator(assistant)
                results = evaluator.run_evaluation()
                st.session_state.evaluation_results = results
            
            st.success("✅ Evaluasi selesai!")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            st.subheader("📈 Hasil Evaluasi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MRR", f"{results['MRR']:.3f}")
            
            with col2:
                st.metric("Faithfulness", f"{results['Faithfulness']:.3f}")
            
            with col3:
                st.metric("RAG Score", f"{results['RAG_Score']:.3f}")
            
            st.info(f"Test Cases: {results['total_test_cases']}")
        
        else:
            st.info("Klik 'Jalankan Evaluasi' untuk melihat performa sistem")

if __name__ == "__main__":
    main()
