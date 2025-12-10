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
import hashlib

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
# TRANSLATION SERVICE - MODE AGGRESIF (DIPERBAIKI)
# ===========================================
class TranslationService:
    def __init__(self):
        self.available = gemini_available
        self.translation_cache = {}  # Cache untuk menghindari terjemahan berulang

    def _stable_hash(self, text: str) -> str:
        """Stable hash untuk cache (cross-process safe)."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def translate_to_indonesian(self, text: str, max_attempts=3):
        """
        Terjemahkan teks ke Bahasa Indonesia:
        - Prioritas: gunakan Gemini (jika tersedia) dengan prompt terstruktur.
        - Jika hasil masih mengandung potongan Bahasa Inggris, coba lagi per-kalimat.
        - Jika tetap gagal, gunakan fallback berbasis replacement yang aman (whole-word).
        """
        if not text or text == "Tidak tersedia":
            return text

        text_hash = self._stable_hash(text)
        if text_hash in self.translation_cache:
            return self.translation_cache[text_hash]

        # Skip jika sudah jelas Bahasa Indonesia
        if self._is_definitely_indonesian(text):
            self.translation_cache[text_hash] = text
            return text

        # 1) Jika Gemini tersedia, coba translate full text dengan prompt ketat
        translated = None
        if self.available:
            try:
                translated = self._translate_with_gemini(text)
            except Exception as e:
                print(f"⚠️ Gemini translate error (full): {e}")
                translated = None

        # 2) Jika hasil Gemini kosong atau masih berisi potongan Inggris, coba per-kalimat
        if not translated or self._contains_english_heavy(translated):
            if self.available:
                try:
                    translated = self._translate_sentence_by_sentence(text, max_attempts=max_attempts)
                except Exception as e:
                    print(f"⚠️ Gemini translate error (per-sentence): {e}")
                    translated = None

        # 3) Jika masih gagal, gunakan fallback replacement yang aman
        if not translated or self._contains_english_heavy(translated):
            translated = self._translate_fallback(text)

        # Final clean-up: normalisasi spasi dan tanda baca
        translated = self._final_cleanup(translated)

        # Simpan ke cache
        self.translation_cache[text_hash] = translated
        return translated

    def _translate_with_gemini(self, text: str):
        """Panggil Gemini dengan prompt yang ketat untuk menerjemahkan seluruh teks."""
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = f"""
Terjemahkan teks berikut ke BAHASA INDONESIA 100%. Hanya keluarkan TERJEMAHAN (tanpa penjelasan tambahan).
- Pertahankan angka, satuan (mg, ml, g, tablet), dan nama obat/paten sebagaimana adanya.
- Jangan memecah kata atau menambahkan kata 'dalam' di tengah kata.
- Jika ada istilah teknis, terjemahkan menggunakan istilah medis yang umum di Indonesia; bila perlu jelaskan singkat dalam 1 frasa.
- Jangan menambahkan informasi baru, jangan menghapus informasi penting.
- Keluaran harus teks polos (plain text) dan sebaiknya mempertahankan struktur paragraf aslinya.

TEKS ASLI:
\"\"\"{text}\"\"\"

TERJEMAHAN:
"""
        response = model.generate_content(prompt)
        result = response.text.strip()
        return result

    def _translate_sentence_by_sentence(self, text: str, max_attempts=3):
        """Bagi teks menjadi kalimat, terjemahkan masing-masing kalimat (mengurangi risiko mixing)."""
        # Simple sentence splitter (preserve parentheses/newlines)
        sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        translated_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            # jika kalimat sudah pendek dan tampak Indonesian, lewati
            if self._is_definitely_indonesian(sentence):
                translated_sentences.append(sentence.strip())
                continue

            attempt = 0
            translated_sentence = None
            while attempt < max_attempts and self.available:
                try:
                    translated_sentence = self._translate_with_gemini(sentence)
                    if translated_sentence and not self._contains_english_heavy(translated_sentence):
                        break
                except Exception as e:
                    print(f"⚠️ Error terjemahkan kalimat: {e}")
                attempt += 1

            if not translated_sentence:
                # fallback per-kalimat: gunakan replacement aman
                translated_sentence = self._translate_fallback(sentence)

            translated_sentences.append(translated_sentence.strip())

        return " ".join(translated_sentences)

    def _contains_english_heavy(self, text: str) -> bool:
        """Deteksi apakah masih banyak potongan Bahasa Inggris di teks keluaran."""
        if not text:
            return False
        text_lower = text.lower()

        # Cari frasa/fragments bahasa Inggris yang umum muncul ketika terjemahan gagal
        english_indicators = [
            r'\bthe\b', r'\band\b', r'\bfor\b', r'\bwith\b', r'\buse\b', r'\btake\b',
            r'\bdosage\b', r'\badministration\b', r'\badverse\b', r'\breactions\b',
            r'\bcontraindications\b', r'\bwarnings\b', r'\bmg\b', r'\bml\b'
        ]
        count = 0
        for patt in english_indicators:
            if re.search(patt, text_lower):
                count += 1

        # Jika lebih dari 1 indikator Inggris terdeteksi, anggap masih heavy
        return count > 0

    def _is_definitely_indonesian(self, text: str) -> bool:
        """Cek apakah teks sudah pasti Bahasa Indonesia (lebih konservatif)."""
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        if not words:
            return True

        indokeys = {'untuk','dengan','dalam','adalah','yang','dari','pada','atau','dapat','tidak','juga','dokter','dosis','efek','samping'}
        indo_count = sum(1 for w in words if w in indokeys)
        return indo_count / max(1, len(words)) > 0.15

    def _translate_fallback(self, text: str):
        """Fallback translation jika semua gagal — aman: replace whole-word saja."""
        replacements = {
            # Header/phrases
            'indications and usage': 'indikasi dan penggunaan',
            'dosage and administration': 'dosis dan cara pemakaian',
            'adverse reactions': 'efek samping',
            'contraindications': 'kontraindikasi',
            'warnings': 'peringatan',
            'drug interactions': 'interaksi obat',
            'precautions': 'tindakan pencegahan',
            'clinical pharmacology': 'farmakologi klinis',
            # kata-kata umum
            'take': 'minum', 'use': 'gunakan', 'tablet': 'tablet', 'tablets': 'tablet',
            'capsule': 'kapsul', 'capsules': 'kapsul', 'every': 'setiap', 'hours': 'jam',
            'hour': 'jam', 'days': 'hari', 'day': 'hari', 'weeks': 'minggu', 'week': 'minggu',
            'months': 'bulan', 'month': 'bulan', 'should': 'sebaiknya', 'may': 'dapat',
            'can': 'bisa', 'must': 'harus', 'do not': 'jangan', 'consult': 'konsultasikan',
            'doctor': 'dokter', 'physician': 'dokter', 'pharmacist': 'apoteker', 'before': 'sebelum',
            'after': 'setelah', 'food': 'makanan', 'water': 'air', 'patient': 'pasien',
            'patients': 'pasien', 'cause': 'menyebabkan', 'causes': 'menyebabkan',
            'side effect': 'efek samping', 'side effects': 'efek samping', 'dosage': 'dosis',
            'dose': 'dosis', 'administration': 'pemberian', 'indicated': 'diindikasikan',
            'as': 'sebagai', 'is': 'adalah', 'are': 'adalah', 'by': 'oleh', 'without': 'tanpa',
            'not': 'tidak', 'no': 'tidak', 'yes': 'ya', 'if': 'jika', 'when': 'ketika',
            'while': 'sementara', 'during': 'selama', 'between': 'antara', 'among': 'di antara',
            'about': 'tentang', 'against': 'terhadap', 'under': 'di bawah', 'over': 'di atas',
            'inside': 'di dalam', 'like': 'seperti', 'near': 'dekat', 'off': 'lepas',
            'on': 'pada', 'out': 'keluar', 'outside': 'di luar', 'past': 'melewati',
            'since': 'sejak', 'throughout': 'sepanjang', 'till': 'sampai', 'toward': 'menuju',
            'until': 'sampai', 'up': 'naik', 'upon': 'pada', 'within': 'dalam'
        }

        result = text
        # Urutkan key berdasarkan panjang descending agar frasa diproses dulu
        for eng, indo in sorted(replacements.items(), key=lambda x: -len(x[0])):
            if not eng.strip():
                continue
            pattern = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
            result = pattern.sub(indo, result)

        # Bersihkan spasi ganda dan koreksi spasi sebelum tanda baca
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([,\.;:])', r'\1', result)
        return result

    def _final_cleanup(self, text: str):
        """Normalisasi terakhir: pastikan tidak ada potongan kata tercampur dan spasi aneh."""
        if not text:
            return text
        # Hapus pengulangan kata 'dalam' jika muncul berulang kali tanpa spasi yang wajar
        text = re.sub(r'(dalam){2,}', 'dalam ', text, flags=re.IGNORECASE)
        # Pastikan kapitalisasi awal paragraf; minimal cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ===========================================
# (Sisanya kode tidak diubah secara signifikan — tetap menggunakan TranslationService baru)
# ===========================================
# FDA API DENGAN TERJEMAHAN OTOMATIS
class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        self.translator = TranslationService()
        self.cache = {}  # Cache untuk menghindari API call berulang

    def get_drug_info(self, generic_name: str):
        cache_key = generic_name.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 5
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    # pilih result paling lengkap
                    best_result = None
                    max_field_count = 0
                    for result in data['results']:
                        field_count = self._count_complete_fields(result)
                        if field_count > max_field_count:
                            max_field_count = field_count
                            best_result = result
                    if not best_result:
                        best_result = data['results'][0]
                    drug_info = self._parse_fda_data(best_result, generic_name)
                    self.cache[cache_key] = drug_info
                    return drug_info
            # alternatif search
            drug_info = self._try_alternative_search(generic_name)
            if drug_info:
                self.cache[cache_key] = drug_info
                return drug_info
            return None
        except Exception as e:
            print(f"❌ Error FDA API: {e}")
            return None

    def _count_complete_fields(self, fda_data: dict):
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
                    if value[0] and isinstance(value[0], str) and value[0].strip():
                        count += 1
                elif isinstance(value, str) and value.strip():
                    count += 1
        return count

    def _try_alternative_search(self, generic_name: str):
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
        except Exception as e:
            print(f"⚠️ Alternatif search error: {e}")
        return None

    def _parse_fda_data(self, fda_data: dict, generic_name: str):
        openfda = fda_data.get('openfda', {})
        raw_data = {
            'brand_name': self._extract_list(openfda.get('brand_name')),
            'drug_class': self._extract_value(fda_data.get('drug_class')),
            'indications': self._extract_indications_raw(fda_data),
            'dosage': self._extract_dosage_raw(fda_data),
            'side_effects': self._extract_side_effects_raw(fda_data),
            'contraindications': self._extract_contraindications_raw(fda_data),
            'interactions': self._extract_interactions_raw(fda_data),
            'warnings': self._extract_warnings_raw(fda_data),
            'dosage_form': self._extract_list(openfda.get('dosage_form')),
            'route': self._extract_list(openfda.get('route'))
        }

        # TERJEMAHKAN setiap field dengan translator yang diperbarui (menggunakan Gemini bila memungkinkan)
        drug_info = {
            "nama": generic_name.title(),
            "nama_generik": generic_name.title(),
            "merek_dagang": self._translate_and_format(raw_data['brand_name']),
            "golongan": self._translate_and_format(raw_data['drug_class']),
            "indikasi": self._translate_and_format(raw_data['indications']),
            "dosis_dewasa": self._translate_and_format(raw_data['dosage']),
            "efek_samping": self._translate_and_format(raw_data['side_effects']),
            "kontraindikasi": self._translate_and_format(raw_data['contraindications']),
            "interaksi": self._translate_and_format(raw_data['interactions']),
            "peringatan": self._translate_and_format(raw_data['warnings']),
            "bentuk_sediaan": self._translate_and_format(raw_data['dosage_form']),
            "route_pemberian": self._translate_and_format(raw_data['route']),
            "sumber": "FDA API",
            "bahasa": "Indonesia",
            "terjemahan_otomatis": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return drug_info

    def _extract_list(self, value):
        if isinstance(value, list) and value:
            return ', '.join([str(v) for v in value if v])
        return "Tidak tersedia"

    def _extract_value(self, value):
        if isinstance(value, list) and value:
            return value[0]
        elif value:
            return str(value)
        return "Tidak tersedia"

    def _extract_indications_raw(self, fda_data: dict):
        fields_to_check = [
            'indications_and_usage',
            'purpose',
            'description',
            'clinical_pharmacology'
        ]
        for field in fields_to_check:
            if field in fda_data and fda_data[field]:
                value = fda_data[field]
                if isinstance(value, list) and value:
                    return value[0][:2000]
                elif value:
                    return str(value)[:2000]
        return "Tidak tersedia"

    def _extract_dosage_raw(self, fda_data: dict):
        fields_to_check = [
            'dosage_and_administration',
            'directions',
            'dosage',
            'how_supplied'
        ]
        for field in fields_to_check:
            if field in fda_data and fda_data[field]:
                value = fda_data[field]
                if isinstance(value, list) and value:
                    return value[0][:2000]
                elif value:
                    return str(value)[:2000]
        return "Tidak tersedia"

    def _extract_side_effects_raw(self, fda_data: dict):
        if 'adverse_reactions' in fda_data and fda_data['adverse_reactions']:
            value = fda_data['adverse_reactions']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]
        if 'warnings' in fda_data and fda_data['warnings']:
            value = fda_data['warnings']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]
        return "Tidak tersedia"

    def _extract_contraindications_raw(self, fda_data: dict):
        if 'contraindications' in fda_data and fda_data['contraindications']:
            value = fda_data['contraindications']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]
        return "Tidak tersedia"

    def _extract_interactions_raw(self, fda_data: dict):
        if 'drug_interactions' in fda_data and fda_data['drug_interactions']:
            value = fda_data['drug_interactions']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]
        return "Tidak tersedia"

    def _extract_warnings_raw(self, fda_data: dict):
        if 'warnings' in fda_data and fda_data['warnings']:
            value = fda_data['warnings']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]
        if 'precautions' in fda_data and fda_data['precautions']:
            value = fda_data['precautions']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]
        return "Tidak tersedia"

    def _translate_and_format(self, text):
        if not text or text == "Tidak tersedia":
            return "Tidak tersedia"
        translated = self.translator.translate_to_indonesian(text)
        if len(translated) > 1000:
            translated = translated[:997] + "..."
        return translated

# (Bagian SimpleRAGPharmaAssistant, EnhancedDrugDetector, FocusedRAGEvaluator dan UI Streamlit
#  tetap sama seperti sebelumnya dan menggunakan FDADrugAPI / TranslationService yang diperbarui.)
# Untuk ringkasnya, saya tidak menulis ulang seluruh file UI di sini — jika Anda ingin,
# saya bisa kirim file lengkap dengan semua bagian (sekarang hanya menonjolkan perubahan penting).
# Namun jika Anda lebih suka file lengkap (seluruh aplikasi), beri tahu saya dan saya kirimkan.

# Jika file ini dimaksudkan menggantikan file lama sepenuhnya, beri tahu saya maka saya kirim versi penuh.
