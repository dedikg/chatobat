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

# ===========================================
# KONFIGURASI APLIKASI
# ===========================================
st.set_page_config(
    page_title="Asisten Informasi Obat FDA",
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
# STRICT TRANSLATION SERVICE - 100% INDONESIA
# ===========================================
class StrictTranslationService:
    def __init__(self):
        self.available = gemini_available
        self.cache = {}
        self.english_words = self._load_english_dictionary()
    
    def _load_english_dictionary(self):
        """Load dictionary kata Inggris yang harus diterjemahkan"""
        return {
            # Kata kerja
            'take': 'minum', 'use': 'gunakan', 'administer': 'berikan',
            'consult': 'konsultasikan', 'see': 'lihat', 'avoid': 'hindari',
            'stop': 'hentikan', 'start': 'mulai', 'continue': 'lanjutkan',
            'discontinue': 'hentikan', 'follow': 'ikuti', 'read': 'baca',
            
            # Kata sifat dan adverbia
            'recommended': 'dianjurkan', 'required': 'diperlukan',
            'necessary': 'perlu', 'important': 'penting', 'safe': 'aman',
            'unsafe': 'tidak aman', 'effective': 'efektif', 'daily': 'harian',
            'weekly': 'mingguan', 'monthly': 'bulanan', 'regular': 'teratur',
            
            # Kata benda umum
            'adults': 'dewasa', 'children': 'anak-anak', 'patients': 'pasien',
            'doctor': 'dokter', 'physician': 'dokter', 'pharmacist': 'apoteker',
            'nurse': 'perawat', 'healthcare': 'kesehatan', 'treatment': 'pengobatan',
            'therapy': 'terapi', 'medication': 'obat', 'drug': 'obat',
            'tablet': 'tablet', 'tablets': 'tablet', 'capsule': 'kapsul',
            'capsules': 'kapsul', 'gelcap': 'kapsul gel', 'gelcaps': 'kapsul gel',
            'pill': 'pil', 'pills': 'pil', 'dose': 'dosis', 'dosage': 'dosis',
            'strength': 'kekuatan', 'mg': 'mg', 'ml': 'ml', 'g': 'g',
            'kg': 'kg', 'hour': 'jam', 'hours': 'jam', 'day': 'hari',
            'days': 'hari', 'week': 'minggu', 'weeks': 'minggu',
            'month': 'bulan', 'months': 'bulan', 'year': 'tahun', 'years': 'tahun',
            
            # Frase umum
            'do not': 'jangan', 'should not': 'sebaiknya tidak',
            'must not': 'tidak boleh', 'may not': 'mungkin tidak',
            'can not': 'tidak dapat', 'need to': 'perlu',
            'have to': 'harus', 'ought to': 'seharusnya',
            'more than': 'lebih dari', 'less than': 'kurang dari',
            'up to': 'hingga', 'at least': 'setidaknya',
            'no more than': 'tidak lebih dari',
            'as directed': 'sesuai petunjuk', 'as needed': 'sesuai kebutuhan',
            'with food': 'dengan makanan', 'without food': 'tanpa makanan',
            'with water': 'dengan air', 'on empty stomach': 'dengan perut kosong',
            'every day': 'setiap hari', 'every other day': 'setiap dua hari sekali',
            'twice daily': 'dua kali sehari', 'three times daily': 'tiga kali sehari',
            'four times daily': 'empat kali sehari',
            
            # Header FDA
            'indications and usage': 'INDIKASI DAN PENGGUNAAN',
            'dosage and administration': 'DOSIS DAN CARA PEMAKAIAN',
            'adverse reactions': 'EFEK SAMPING',
            'contraindications': 'KONTRAINDIKASI',
            'warnings': 'PERINGATAN',
            'precautions': 'TINDAKAN PENCEGAHAN',
            'drug interactions': 'INTERAKSI OBAT',
            'clinical pharmacology': 'FARMAKOLOGI KLINIS',
            'how supplied': 'CARA DISEDIAKAN',
            'patient counseling information': 'INFORMASI KONSELING PASIEN'
        }
    
    def translate_to_indonesian(self, text: str) -> str:
        """Terjemahkan teks ke Bahasa Indonesia 100%"""
        if not text or text.strip() == "" or text == "Tidak tersedia":
            return text
        
        # Teks pendek tidak perlu diterjemahkan
        if len(text.strip()) < 25:
            return text
        
        # Cek cache
        text_hash = hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Jika Gemini tidak tersedia, gunakan metode manual
        if not self.available:
            result = self._manual_translation(text)
            self.cache[text_hash] = result
            return result
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            TERJEMAHKAN TEKS MEDIS BERIKUT KE BAHASA INDONESIA 100%:
            
            TEKS ASLI: "{text}"
            
            PERATURAN MUTLAK:
            1. HASIL HARUS 100% BAHASA INDONESIA - tidak ada kata Inggris sama sekali
            2. PERTAHANKAN: semua angka (500, 1000), satuan (mg, ml, tablet), nama obat (paracetamol, amoxicillin)
            3. TERJEMAHKAN SEMUA kata kerja, kata sifat, kata keterangan, dan penjelasan ke Bahasa Indonesia
            4. Gunakan istilah medis yang umum di Indonesia
            5. JANGAN ubah struktur informasi medis
            6. JANGAN tambahkan informasi baru
            7. Format harus natural untuk pembaca Indonesia
            8. Jika ada singkatan medis standar (mg, ml, FDA), biarkan dalam bentuk aslinya
            
            CONTOH TERJEMAHAN YANG BENAR:
            "Take 2 tablets every 6 hours" → "Minum 2 tablet setiap 6 jam"
            "Do not exceed 4000 mg per day" → "Jangan melebihi 4000 mg per hari"
            "For adults and children 12 years and over" → "Untuk dewasa dan anak usia 12 tahun ke atas"
            "Consult your doctor if symptoms persist" → "Konsultasikan dokter jika gejala berlanjut"
            
            TERJEMAHAN BAHASA INDONESIA:
            """
            
            response = model.generate_content(prompt)
            translated = response.text.strip()
            
            # Bersihkan hasil
            translated = translated.replace('"', '').replace('Terjemahan:', '').strip()
            
            # Validasi terjemahan
            if self._is_translation_valid(translated, text):
                self.cache[text_hash] = translated
                return translated
            else:
                # Jika terjemahan buruk, gunakan manual
                result = self._manual_translation(text)
                self.cache[text_hash] = result
                return result
                
        except Exception as e:
            print(f"❌ Translation error: {e}")
            result = self._manual_translation(text)
            self.cache[text_hash] = result
            return result
    
    def _is_translation_valid(self, translated: str, original: str) -> bool:
        """Validasi kualitas terjemahan"""
        if not translated or len(translated) < 10:
            return False
        
        # Cek jika terjemahan terlalu pendek
        if len(translated) < len(original) * 0.3:
            return False
        
        # Cek jika masih banyak kata Inggris yang harusnya diterjemahkan
        english_count = 0
        words = translated.lower().split()
        
        for word in words:
            # Kata Inggris umum yang harusnya sudah diterjemahkan
            if word in ['take', 'use', 'do', 'not', 'more', 'than', 'adults', 'children']:
                english_count += 1
        
        # Jika masih banyak kata Inggris, terjemahan buruk
        if english_count > 3:
            return False
        
        # Cek karakter aneh
        bad_patterns = [
            r'\b\d+\s+[a-z]+\s+\d+',  # "1 dalam 2"
            r'[a-z][A-Z]{2,}[a-z]',    # camelCase rusak
            r'\b[a-z]{15,}\b'          # kata sangat panjang
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, translated):
                return False
        
        return True
    
    def _manual_translation(self, text: str) -> str:
        """Terjemahan manual jika Gemini gagal"""
        result = text
        
        # Langkah 1: Terjemahkan header FDA
        header_patterns = {
            r'INDICATIONS AND USAGE': 'INDIKASI DAN PENGGUNAAN:',
            r'DOSAGE AND ADMINISTRATION': 'DOSIS DAN CARA PEMAKAIAN:',
            r'ADVERSE REACTIONS': 'EFEK SAMPING:',
            r'CONTRAINDICATIONS': 'KONTRAINDIKASI:',
            r'WARNINGS': 'PERINGATAN:',
            r'DRUG INTERACTIONS': 'INTERAKSI OBAT:',
            r'PRECAUTIONS': 'TINDAKAN PENCEGAHAN:'
        }
        
        for pattern, replacement in header_patterns.items():
            result = re.sub(pattern, replacement, result)
            result = re.sub(pattern.lower(), replacement.lower(), result)
            result = re.sub(pattern.title(), replacement.title(), result)
        
        # Langkah 2: Terjemahkan kata-kata umum
        for eng_word, indo_word in self.english_words.items():
            # Untuk kata tunggal
            result = re.sub(rf'\b{eng_word}\b', indo_word, result, flags=re.IGNORECASE)
            
            # Untuk bentuk plural (tambahkan 's')
            if eng_word.endswith('y'):
                plural = eng_word[:-1] + 'ies'
            else:
                plural = eng_word + 's'
                
            if plural in self.english_words:
                result = re.sub(rf'\b{plural}\b', self.english_words[plural], result, flags=re.IGNORECASE)
        
        # Langkah 3: Terjemahkan pola umum
        common_patterns = {
            r'Take (\d+) (tablets?|capsules?) every (\d+) hours?': r'Minum \1 \2 setiap \3 jam',
            r'(\d+) (mg|ml) every (\d+) hours?': r'\1 \2 setiap \3 jam',
            r'(\d+) (mg|ml) per day': r'\1 \2 per hari',
            r'(\d+) times (daily|a day)': r'\1 kali sehari',
            r'For adults and children (\d+) years and (over|older)': r'Untuk dewasa dan anak usia \1 tahun ke atas',
            r'Children under (\d+) years?': r'Anak di bawah \1 tahun',
            r'Do not take (more than|exceed) (\d+)': r'Jangan minum lebih dari \2',
            r'Consult your (doctor|physician)': r'Konsultasikan dokter',
            r'Unless directed by a (doctor|physician)': r'Kecuali diarahkan oleh dokter',
            r'(\d+)-(\d+) (mg|ml)': r'\1-\2 \3',
            r'every (\d+)-(\d+) hours?': r'setiap \1-\2 jam',
            r'(\d+) to (\d+) (mg|ml)': r'\1 sampai \2 \3'
        }
        
        for pattern, replacement in common_patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Langkah 4: Kapitalisasi kalimat
        sentences = re.split(r'(?<=[.!?])\s+', result)
        capitalized_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Kapitalisasi huruf pertama
                sentence = sentence.strip()
                if len(sentence) > 0:
                    sentence = sentence[0].upper() + sentence[1:]
                capitalized_sentences.append(sentence)
        
        result = ' '.join(capitalized_sentences)
        
        # Langkah 5: Bersihkan spasi ganda
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

# ===========================================
# FDA DRUG API - PARSING BERSIH
# ===========================================
class CleanDrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        self.translator = StrictTranslationService()
        self.cache = {}
    
    def get_drug_info(self, drug_name: str):
        """Dapatkan informasi obat dari FDA API"""
        cache_key = drug_name.lower()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print(f"🔍 Mencari data FDA untuk: {drug_name}")
        
        try:
            # Cari data obat
            params = {
                'search': f'openfda.generic_name:"{drug_name}"',
                'limit': 1
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    print(f"✅ Data ditemukan untuk: {drug_name}")
                    drug_info = self._parse_drug_data(data['results'][0], drug_name)
                    self.cache[cache_key] = drug_info
                    return drug_info
                else:
                    print(f"⚠️ Tidak ada hasil untuk: {drug_name}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error FDA API: {e}")
            return None
    
    def _parse_drug_data(self, fda_data: dict, drug_name: str):
        """Parse data obat dari respons FDA"""
        openfda = fda_data.get('openfda', {})
        
        # Data dasar
        drug_info = {
            "nama": self._get_indonesian_name(drug_name),
            "nama_fda": drug_name,
            "merek_dagang": self._extract_brand_names(openfda),
            "golongan": self._extract_field(fda_data, 'drug_class'),
            "bentuk_sediaan": self._extract_dosage_forms(openfda),
            "sumber": "FDA Amerika Serikat",
            "diakses": datetime.now().strftime("%d-%m-%Y %H:%M")
        }
        
        # Data medis penting - langsung diterjemahkan
        medical_fields = {
            'indikasi': self._extract_and_translate(fda_data, 'indications_and_usage', 'KEGUNAAN'),
            'dosis': self._extract_and_translate(fda_data, 'dosage_and_administration', 'DOSIS'),
            'efek_samping': self._extract_and_translate(fda_data, 'adverse_reactions', 'EFEK SAMPING'),
            'kontraindikasi': self._extract_and_translate(fda_data, 'contraindications', 'KONTRAINDIKASI'),
            'interaksi_obat': self._extract_and_translate(fda_data, 'drug_interactions', 'INTERAKSI OBAT'),
            'peringatan': self._extract_and_translate(fda_data, 'warnings', 'PERINGATAN')
        }
        
        drug_info.update(medical_fields)
        
        return drug_info
    
    def _get_indonesian_name(self, drug_name: str):
        """Dapatkan nama obat dalam Bahasa Indonesia"""
        indonesian_names = {
            'paracetamol': 'Parasetamol',
            'acetaminophen': 'Parasetamol',
            'amoxicillin': 'Amoksisilin',
            'omeprazole': 'Omeprazol',
            'ibuprofen': 'Ibuprofen',
            'metformin': 'Metformin',
            'atorvastatin': 'Atorvastatin',
            'simvastatin': 'Simvastatin',
            'loratadine': 'Loratadin',
            'aspirin': 'Aspirin',
            'vitamin c': 'Vitamin C',
            'lansoprazole': 'Lansoprazol',
            'esomeprazole': 'Esomeprazol',
            'cefixime': 'Sefiksim',
            'cetirizine': 'Setirizin',
            'dextromethorphan': 'Dextrometorfan',
            'ambroxol': 'Ambroksol',
            'salbutamol': 'Salbutamol',
            'albuterol': 'Salbutamol'
        }
        
        return indonesian_names.get(drug_name.lower(), drug_name.capitalize())
    
    def _extract_brand_names(self, openfda: dict):
        """Ekstrak nama merek dagang"""
        if 'brand_name' in openfda and openfda['brand_name']:
            brands = openfda['brand_name']
            if isinstance(brands, list):
                # Ambil maksimal 3 merek, terjemahkan ke Indonesia
                brand_list = []
                for brand in brands[:3]:
                    brand_str = str(brand)
                    # Terjemahkan jika ada kata Inggris umum
                    if 'plus' in brand_str.lower():
                        brand_str = brand_str.replace('Plus', 'Plus').replace('plus', 'plus')
                    brand_list.append(brand_str)
                
                return ', '.join(brand_list)
        
        return "Tidak tersedia"
    
    def _extract_dosage_forms(self, openfda: dict):
        """Ekstrak bentuk sediaan"""
        if 'dosage_form' in openfda and openfda['dosage_form']:
            forms = openfda['dosage_form']
            if isinstance(forms, list):
                form_list = []
                for form in forms[:2]:
                    form_str = str(form)
                    # Terjemahkan bentuk sediaan
                    translations = {
                        'tablet': 'tablet',
                        'capsule': 'kapsul',
                        'gelcap': 'kapsul gel',
                        'solution': 'larutan',
                        'suspension': 'suspensi',
                        'injection': 'suntikan',
                        'cream': 'krim',
                        'ointment': 'salep'
                    }
                    
                    for eng, indo in translations.items():
                        if eng in form_str.lower():
                            form_str = form_str.lower().replace(eng, indo)
                    
                    form_list.append(form_str.capitalize())
                
                return ', '.join(form_list)
        
        return "Tidak tersedia"
    
    def _extract_field(self, fda_data: dict, field_name: str):
        """Ekstrak field sederhana"""
        if field_name in fda_data and fda_data[field_name]:
            value = fda_data[field_name]
            if isinstance(value, list):
                return value[0][:100]
            return str(value)[:100]
        
        return "Tidak tersedia"
    
    def _extract_and_translate(self, fda_data: dict, field_name: str, label: str):
        """Ekstrak dan terjemahkan field medis"""
        if field_name not in fda_data or not fda_data[field_name]:
            return "Tidak tersedia"
        
        value = fda_data[field_name]
        
        # Ekstrak teks
        if isinstance(value, list):
            text = ' '.join([str(v) for v in value if v])
        else:
            text = str(value)
        
        # Batasi panjang
        if len(text) > 500:
            text = text[:497] + "..."
        
        # Bersihkan teks
        text = self._clean_medical_text(text)
        
        # Terjemahkan
        translated = self.translator.translate_to_indonesian(text)
        
        # Format output
        return f"{label}: {translated}"
    
    def _clean_medical_text(self, text: str):
        """Bersihkan teks medis"""
        # Hapus karakter kontrol
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Hapus multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Hapus referensi dalam kurung seperti [see ...]
        text = re.sub(r'\[see[^\]]*\]', '', text)
        
        return text.strip()

# ===========================================
# SIMPLE ANSWER GENERATOR
# ===========================================
class SimpleAnswerGenerator:
    def __init__(self):
        self.translator = StrictTranslationService()
    
    def generate_answer(self, question: str, drug_info: dict):
        """Generate jawaban dalam Bahasa Indonesia"""
        question_lower = question.lower()
        drug_name = drug_info['nama']
        
        # Deteksi jenis pertanyaan
        if any(word in question_lower for word in ['dosis', 'berapa', 'takaran', 'aturan minum']):
            return self._generate_dosage_answer(drug_info)
        elif any(word in question_lower for word in ['efek samping', 'efeknya', 'bahaya']):
            return self._generate_side_effects_answer(drug_info)
        elif any(word in question_lower for word in ['untuk apa', 'kegunaan', 'manfaat', 'indikasi']):
            return self._generate_indications_answer(drug_info)
        elif any(word in question_lower for word in ['kontra', 'larangan', 'tidak boleh', 'hindari']):
            return self._generate_contraindications_answer(drug_info)
        elif any(word in question_lower for word in ['interaksi', 'bereaksi']):
            return self._generate_interactions_answer(drug_info)
        elif any(word in question_lower for word in ['peringatan', 'hati-hati', 'waspada']):
            return self._generate_warnings_answer(drug_info)
        else:
            return self._generate_general_answer(drug_info)
    
    def _generate_dosage_answer(self, drug_info: dict):
        """Generate jawaban tentang dosis"""
        drug_name = drug_info['nama']
        dosage = drug_info.get('dosis', 'Tidak tersedia')
        
        # Format khusus untuk dosis
        if dosage != "Tidak tersedia":
            # Bersihkan label berulang
            dosage = re.sub(r'^DOSIS\s*:\s*', '', dosage, flags=re.IGNORECASE)
            
            answer = f"""**💊 {drug_name}**

**INFORMASI DOSIS:**
{dosage}

**PETUNJUM UMUM:**
• Ikuti petunjuk dokter atau apoteker
• Jangan melebihi dosis yang dianjurkan
• Gunakan sesuai kebutuhan saja
• Simpan pada suhu ruangan yang sejuk

**UNTUK ANAK-ANAK:**
• Dosis disesuaikan dengan berat badan
• Konsultasikan dokter untuk dosis tepat
• Jangan berikan obat dewasa pada anak"""
        else:
            answer = f"""**💊 {drug_name}**

Informasi dosis tidak tersedia dalam database FDA.

**SARAN:**
• Konsultasikan dengan dokter untuk dosis yang tepat
• Baca petunjuk pada kemasan obat
• Jangan mengubah dosis tanpa arahan dokter"""
        
        answer += self._get_footer()
        return answer
    
    def _generate_side_effects_answer(self, drug_info: dict):
        """Generate jawaban tentang efek samping"""
        drug_name = drug_info['nama']
        side_effects = drug_info.get('efek_samping', 'Tidak tersedia')
        
        answer = f"""**💊 {drug_name}**

**EFEK SAMPING:**"""
        
        if side_effects != "Tidak tersedia":
            # Bersihkan label
            side_effects = re.sub(r'^EFEK SAMPING\s*:\s*', '', side_effects, flags=re.IGNORECASE)
            answer += f"\n{side_effects}"
        else:
            answer += "\nInformasi efek samping tidak tersedia."
        
        answer += """

**YANG HARUS DILAKUKAN:**
• Hentikan penggunaan jika terjadi reaksi alergi
• Hubungi dokter jika efek samping berat
• Laporkan efek samping ke fasilitas kesehatan"""
        
        answer += self._get_footer()
        return answer
    
    def _generate_indications_answer(self, drug_info: dict):
        """Generate jawaban tentang indikasi"""
        drug_name = drug_info['nama']
        indications = drug_info.get('indikasi', 'Tidak tersedia')
        
        answer = f"""**💊 {drug_name}**

**KEGUNAAN UTAMA:**"""
        
        if indications != "Tidak tersedia":
            # Bersihkan label
            indications = re.sub(r'^KEGUNAAN\s*:\s*', '', indications, flags=re.IGNORECASE)
            answer += f"\n{indications}"
        else:
            answer += "\nInformasi kegunaan tidak tersedia."
        
        answer += self._get_footer()
        return answer
    
    def _generate_general_answer(self, drug_info: dict):
        """Generate jawaban umum"""
        drug_name = drug_info['nama']
        
        answer = f"""**💊 {drug_name}**

"""
        
        # Tambahkan informasi yang tersedia
        info_added = False
        
        if drug_info.get('indikasi', 'Tidak tersedia') != "Tidak tersedia":
            indications = drug_info['indikasi']
            indications = re.sub(r'^KEGUNAAN\s*:\s*', '', indications, flags=re.IGNORECASE)
            if len(indications) > 150:
                indications = indications[:147] + "..."
            answer += f"**Kegunaan:** {indications}\n\n"
            info_added = True
        
        if drug_info.get('dosis', 'Tidak tersedia') != "Tidak tersedia":
            dosage = drug_info['dosis']
            dosage = re.sub(r'^DOSIS\s*:\s*', '', dosage, flags=re.IGNORECASE)
            if len(dosage) > 100:
                dosage = dosage[:97] + "..."
            answer += f"**Dosis Umum:** {dosage}\n\n"
            info_added = True
        
        if drug_info.get('efek_samping', 'Tidak tersedia') != "Tidak tersedia":
            side_effects = drug_info['efek_samping']
            side_effects = re.sub(r'^EFEK SAMPING\s*:\s*', '', side_effects, flags=re.IGNORECASE)
            if len(side_effects) > 100:
                side_effects = side_effects[:97] + "..."
            answer += f"**Efek Samping:** {side_effects}\n\n"
            info_added = True
        
        if not info_added:
            answer += "Informasi terbatas tersedia untuk obat ini.\n\n"
        
        answer += self._get_footer()
        return answer
    
    def _get_footer(self):
        """Dapatkan footer standar"""
        return """

---
**📋 INFORMASI PENTING:**
• Data dari U.S. Food and Drug Administration (FDA)
• Informasi untuk edukasi dan referensi
• Dosis dan indikasi dapat berbeda untuk setiap individu
• Obat mungkin memiliki nama merek berbeda di Indonesia

**⚠️ PERINGATAN:**
SELALU KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT APAPUN.
"""

# ===========================================
# DRUG DETECTOR
# ===========================================
class DrugDetector:
    def __init__(self):
        self.drugs = {
            'paracetamol': ['parasetamol', 'acetaminophen', 'panadol', 'sanmol', 'tempra', 'biogesic'],
            'amoxicillin': ['amoksisilin', 'amoxan', 'moxigra', 'moxypen'],
            'omeprazole': ['omeprazol', 'prilosec', 'losec', 'omepron'],
            'ibuprofen': ['ibuprom', 'proris', 'arthrifen', 'ibufar'],
            'metformin': ['glucophage', 'diabex', 'metfor'],
            'atorvastatin': ['lipitor', 'atorva', 'tovast'],
            'simvastatin': ['zocor', 'simvor', 'lipostat'],
            'loratadine': ['klaritin', 'loramine', 'allertine'],
            'aspirin': ['aspro', 'aspilet', 'cardiprin'],
            'vitamin c': ['redoxon', 'enervon c', 'vitacimin'],
            'lansoprazole': ['lansoprazol', 'prevacid', 'lanzol'],
            'esomeprazole': ['esomeprazol', 'nexium', 'esotrax'],
            'cefixime': ['sefiksim', 'suprax', 'cefix'],
            'cetirizine': ['setirizin', 'zyrtec', 'cetrizin'],
            'dextromethorphan': ['dextrometorfan', 'dmp', 'valtus'],
            'ambroxol': ['mucosolvan', 'broxol', 'mucos'],
            'salbutamol': ['ventolin', 'asmasolon', 'salbumol']
        }
    
    def detect_drug(self, question: str):
        """Deteksi obat dalam pertanyaan"""
        question_lower = question.lower()
        detected = []
        
        for drug_name, aliases in self.drugs.items():
            for alias in aliases:
                if alias in question_lower:
                    detected.append({
                        'nama_indonesia': self._get_indonesian_name(drug_name),
                        'nama_fda': drug_name,
                        'alias': alias
                    })
                    break
        
        return detected
    
    def _get_indonesian_name(self, drug_name: str):
        """Dapatkan nama Indonesia"""
        indonesian_names = {
            'paracetamol': 'Parasetamol',
            'amoxicillin': 'Amoksisilin',
            'omeprazole': 'Omeprazol',
            'ibuprofen': 'Ibuprofen',
            'metformin': 'Metformin',
            'atorvastatin': 'Atorvastatin',
            'simvastatin': 'Simvastatin',
            'loratadine': 'Loratadin',
            'aspirin': 'Aspirin',
            'vitamin c': 'Vitamin C',
            'lansoprazole': 'Lansoprazol',
            'esomeprazole': 'Esomeprazol',
            'cefixime': 'Sefiksim',
            'cetirizine': 'Setirizin',
            'dextromethorphan': 'Dextrometorfan',
            'ambroxol': 'Ambroksol',
            'salbutamol': 'Salbutamol'
        }
        
        return indonesian_names.get(drug_name, drug_name.capitalize())

# ===========================================
# MAIN ASSISTANT
# ===========================================
class MedicalAssistant:
    def __init__(self):
        self.api = CleanDrugAPI()
        self.detector = DrugDetector()
        self.generator = SimpleAnswerGenerator()
        self.cache = {}
    
    def ask_question(self, question: str):
        """Jawab pertanyaan tentang obat"""
        print(f"\n{'='*60}")
        print(f"PERTANYAAN: {question}")
        
        # Deteksi obat
        detected_drugs = self.detector.detect_drug(question)
        
        if not detected_drugs:
            return self._get_no_drug_response(), []
        
        # Ambil obat pertama yang terdeteksi
        main_drug = detected_drugs[0]
        drug_name_fda = main_drug['nama_fda']
        
        print(f"💊 Terdeteksi: {main_drug['nama_indonesia']} ({drug_name_fda})")
        
        # Ambil data dari API
        drug_info = self.api.get_drug_info(drug_name_fda)
        
        if not drug_info:
            return self._get_no_data_response(main_drug['nama_indonesia']), []
        
        # Generate jawaban
        answer = self.generator.generate_answer(question, drug_info)
        
        print(f"✅ Jawaban siap")
        print(f"{'='*60}")
        
        return answer, [drug_info]
    
    def _get_no_drug_response(self):
        """Response ketika tidak ada obat terdeteksi"""
        available_drugs = list(self.detector.drugs.keys())[:10]
        
        # Konversi ke nama Indonesia
        indo_drugs = []
        for drug in available_drugs:
            if drug == 'paracetamol':
                indo_drugs.append('parasetamol')
            elif drug == 'amoxicillin':
                indo_drugs.append('amoksisilin')
            elif drug == 'omeprazole':
                indo_drugs.append('omeprazol')
            else:
                indo_drugs.append(drug)
        
        drug_list = ', '.join(indo_drugs)
        
        return f"""
**❌ Obat tidak terdeteksi**

Saya tidak dapat mengidentifikasi obat dalam pertanyaan Anda.

**💊 Obat yang tersedia dalam sistem:**
{drug_list}

**💡 Contoh pertanyaan:**
• "Apa dosis parasetamol?"
• "Efek samping amoksisilin?"
• "Untuk apa omeprazol digunakan?"
• "Interaksi obat ibuprofen?"

**📝 Tips:**
• Gunakan nama generik obat
• Tanyakan satu obat sekaligus
• Gunakan Bahasa Indonesia
"""
    
    def _get_no_data_response(self, drug_name: str):
        """Response ketika tidak ada data ditemukan"""
        return f"""
**❌ Data tidak ditemukan**

Informasi tentang **{drug_name}** tidak tersedia dalam database FDA.

**🔍 Kemungkinan penyebab:**
• Nama obat mungkin berbeda di FDA
• Data tidak tersedia untuk publik
• Obat tidak terdaftar di FDA Amerika

**💡 Saran:**
• Coba gunakan nama lain
• Konsultasikan langsung ke dokter
• Periksa kemasan obat

**🏥 Konsultasikan dengan dokter atau apoteker untuk informasi yang akurat.**
"""

# ===========================================
# STREAMLIT APP
# ===========================================
def main():
    # Initialize assistant
    assistant = MedicalAssistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.title("💊 Asisten Obat")
        st.markdown("""
        **Informasi Obat dari FDA Amerika Serikat**
        
        Semua data berasal dari database resmi FDA.
        
        **Dukung oleh:**
        • Google Gemini AI
        • FDA OpenAPI
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Hapus Percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        **⚠️ PERINGATAN MEDIS:**
        
        Informasi ini untuk edukasi.
        Selalu konsultasi dengan dokter.
        
        **📞 Darurat: 119**
        """)
    
    # Main content
    st.title("💊 Asisten Informasi Obat")
    
    # Header
    st.markdown("""
    <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #2e7d32;">🏥 DATA RESMI FDA AMERIKA SERIKAT</h4>
        <p style="margin: 5px 0 0 0; color: #666;">
        Semua informasi diterjemahkan ke <strong>Bahasa Indonesia 100%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Tampilkan history chat
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
        
        # Input chat
        if prompt := st.chat_input("Tanyakan tentang obat..."):
            # Tambahkan pesan user
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Tampilkan pesan user
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Mencari informasi dari FDA..."):
                    answer, _ = assistant.ask_question(prompt)
                    st.markdown(answer)
            
            # Tambahkan pesan assistant
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Contoh pertanyaan
    if not st.session_state.messages:
        st.markdown("---")
        st.subheader("💡 Contoh Pertanyaan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Apa dosis parasetamol?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Apa dosis parasetamol?"})
                st.rerun()
            
            if st.button("Efek samping amoksisilin?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Efek samping amoksisilin?"})
                st.rerun()
            
            if st.button("Untuk apa omeprazol?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Untuk apa omeprazol?"})
                st.rerun()
        
        with col2:
            if st.button("Interaksi obat ibuprofen?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Interaksi obat ibuprofen?"})
                st.rerun()
            
            if st.button("Peringatan metformin?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Peringatan metformin?"})
                st.rerun()
            
            if st.button("Kontraindikasi aspirin?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Kontraindikasi aspirin?"})
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>💊 <strong>Asisten Informasi Obat</strong> • Data dari U.S. Food and Drug Administration</p>
        <p>🇮🇩 100% Bahasa Indonesia • Untuk edukasi medis</p>
    </div>
    """, unsafe_allow_html=True)

# ===========================================
# RUN APPLICATION
# ===========================================
if __name__ == "__main__":
    main()
