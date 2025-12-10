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
    # Jangan gunakan st.error di top-level jika app mungkin dipanggil dari luar UI
    print(f"❌ Error konfigurasi Gemini API: {e}")
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
            r'\bcontraindications\b', r'\bwarnings\b', r'\bdrug\b', r'\bmg\b', r'\bml\b'
        ]
        count = 0
        for patt in english_indicators:
            if re.search(patt, text_lower):
                count += 1

        # Jika ada indikator Inggris, anggap masih heavy (konservatif)
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
        # Pastikan spasi dan tanda baca rapi
        text = re.sub(r'\s+', ' ', text).strip()
        # Perbaikan sederhana: jangan gabungkan huruf kecil kapital secara tidak wajar
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text

# ===========================================
# FDA API DENGAN TERJEMAHAN OTOMATIS
# ===========================================
class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        self.translator = TranslationService()
        self.cache = {}  # Cache untuk menghindari API call berulang

    def get_drug_info(self, generic_name: str):
        """Ambil data obat langsung dari FDA API dengan pencarian lebih baik"""
        # Cek cache dulu
        cache_key = generic_name.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 5
        }

        try:
            print(f"🔍 Mencari data FDA untuk: {generic_name}")
            response = requests.get(self.base_url, params=params, timeout=20)

            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    print(f"✅ Ditemukan {len(data['results'])} hasil untuk {generic_name}")

                    # Cari data yang paling lengkap
                    best_result = None
                    max_field_count = 0

                    for result in data['results']:
                        field_count = self._count_complete_fields(result)
                        if field_count > max_field_count:
                            max_field_count = field_count
                            best_result = result

                    if best_result:
                        drug_info = self._parse_fda_data(best_result, generic_name)
                        self.cache[cache_key] = drug_info
                        return drug_info

                    # Jika tidak ada yang lengkap, ambil yang pertama
                    drug_info = self._parse_fda_data(data['results'][0], generic_name)
                    self.cache[cache_key] = drug_info
                    return drug_info

            # Coba dengan pencarian alternatif
            print(f"⚠️ Pencarian utama gagal, mencoba alternatif untuk: {generic_name}")
            drug_info = self._try_alternative_search(generic_name)
            if drug_info:
                self.cache[cache_key] = drug_info
                return drug_info

            print(f"❌ Tidak ada data ditemukan untuk: {generic_name}")
            return None

        except Exception as e:
            print(f"❌ Error FDA API: {e}")
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
                    if value[0] and isinstance(value[0], str) and value[0].strip():
                        count += 1
                elif isinstance(value, str) and value.strip():
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
                    print(f"✅ Alternatif ditemukan untuk: {generic_name}")
                    return self._parse_fda_data(data['results'][0], generic_name)
        except Exception as e:
            print(f"⚠️ Alternatif search error: {e}")

        return None

    def _parse_fda_data(self, fda_data: dict, generic_name: str):
        """Parse data FDA - OTOMATIS TERJEMAH KE BAHASA INDONESIA"""
        print(f"📝 Parsing data FDA untuk: {generic_name}")

        openfda = fda_data.get('openfda', {})

        # Ekstrak data mentah
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

        # TERJEMAHKAN SEMUA FIELD
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

        print(f"✅ Data berhasil diparsing dan diterjemahkan untuk: {generic_name}")
        return drug_info

    def _extract_list(self, value):
        """Ekstrak nilai dari list"""
        if isinstance(value, list) and value:
            return ', '.join([str(v) for v in value if v])
        return "Tidak tersedia"

    def _extract_value(self, value):
        """Ekstrak nilai tunggal"""
        if isinstance(value, list) and value:
            return value[0]
        elif value:
            return str(value)
        return "Tidak tersedia"

    def _extract_indications_raw(self, fda_data: dict):
        """Ekstrak informasi indikasi dari berbagai field"""
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
        """Ekstrak informasi dosis dari berbagai field"""
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
        """Ekstrak informasi efek samping"""
        if 'adverse_reactions' in fda_data and fda_data['adverse_reactions']:
            value = fda_data['adverse_reactions']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]

        # Coba di field lain
        if 'warnings' in fda_data and fda_data['warnings']:
            value = fda_data['warnings']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]

        return "Tidak tersedia"

    def _extract_contraindications_raw(self, fda_data: dict):
        """Ekstrak informasi kontraindikasi"""
        if 'contraindications' in fda_data and fda_data['contraindications']:
            value = fda_data['contraindications']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]

        return "Tidak tersedia"

    def _extract_interactions_raw(self, fda_data: dict):
        """Ekstrak informasi interaksi"""
        if 'drug_interactions' in fda_data and fda_data['drug_interactions']:
            value = fda_data['drug_interactions']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]

        return "Tidak tersedia"

    def _extract_warnings_raw(self, fda_data: dict):
        """Ekstrak informasi peringatan"""
        if 'warnings' in fda_data and fda_data['warnings']:
            value = fda_data['warnings']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]

        # Coba di field precautions
        if 'precautions' in fda_data and fda_data['precautions']:
            value = fda_data['precautions']
            if isinstance(value, list) and value:
                return value[0][:2000]
            elif value:
                return str(value)[:2000]

        return "Tidak tersedia"

    def _translate_and_format(self, text):
        """Terjemahkan dan format teks"""
        if not text or text == "Tidak tersedia":
            return "Tidak tersedia"

        # Terjemahkan
        translated = self.translator.translate_to_indonesian(text)

        # Format: pastikan tidak terlalu panjang
        if len(translated) > 1000:
            translated = translated[:997] + "..."

        return translated

# ===========================================
# SIMPLE RAG ASSISTANT - 100% BAHASA INDONESIA
# (menggunakan FDADrugAPI dan TranslationService yang diperbarui)
# ===========================================
class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        self.drug_detector = EnhancedDrugDetector()
        self.drugs_cache = {}
        self.current_context = {}
        self.query_history = []

    def _get_or_fetch_drug_info(self, drug_name: str):
        """Dapatkan data dari cache atau fetch dari FDA API"""
        drug_key = drug_name.lower()

        if drug_key in self.drugs_cache:
            print(f"📦 Menggunakan cache untuk: {drug_name}")
            return self.drugs_cache[drug_key]

        print(f"🌐 Fetch data FDA untuk: {drug_name}")
        fda_name = self.drug_detector.get_fda_name(drug_name)
        drug_info = self.fda_api.get_drug_info(fda_name)

        if drug_info:
            if drug_name != fda_name:
                drug_info['nama'] = drug_name.title()
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"

            # Validasi dan perbaiki terjemahan jika perlu
            drug_info = self._validate_and_fix_translation(drug_info)
            self.drugs_cache[drug_key] = drug_info
            print(f"✅ Data berhasil diambil untuk: {drug_name}")
        else:
            print(f"❌ Gagal mendapatkan data untuk: {drug_name}")

        return drug_info

    def _validate_and_fix_translation(self, drug_info: dict):
        """Validasi dan perbaiki terjemahan jika masih ada bahasa Inggris"""
        fields_to_check = [
            'indikasi', 'dosis_dewasa', 'efek_samping',
            'kontraindikasi', 'interaksi', 'peringatan',
            'golongan', 'bentuk_sediaan', 'route_pemberian',
            'merek_dagang'
        ]

        for field in fields_to_check:
            if field in drug_info and drug_info[field] != "Tidak tersedia":
                text = drug_info[field]

                # Cek jika masih banyak bahasa Inggris
                if self._contains_too_much_english(text):
                    print(f"⚠️  Masih ada Inggris di {field}, menerjemahkan ulang...")
                    translated = self.translator.translate_to_indonesian(text)
                    if translated and translated != text:
                        drug_info[field] = translated
                        drug_info['diperbaiki'] = True

        return drug_info

    def _contains_too_much_english(self, text: str) -> bool:
        """Cek apakah teks mengandung terlalu banyak bahasa Inggris"""
        if not text or len(text) < 20:
            return False

        text_lower = text.lower()

        # Kata-kata Inggris yang umum dalam teks medis (yang harus diterjemahkan)
        common_english_medical = [
            'indications', 'usage', 'dosage', 'administration', 'adverse',
            'reactions', 'contraindications', 'warnings', 'interactions',
            'tablets', 'capsules', 'should', 'may', 'can', 'must', 'do not',
            'consult', 'doctor', 'physician', 'pharmacist', 'patient',
            'use', 'take', 'cause', 'causes', 'side effect', 'side effects'
        ]

        # Kata-kata Indonesia yang seharusnya ada
        common_indonesian = [
            'untuk', 'dengan', 'dalam', 'adalah', 'yang', 'dari', 'pada',
            'atau', 'dapat', 'akan', 'tidak', 'juga', 'oleh', 'lebih',
            'sama', 'obat', 'minum', 'gunakan', 'setiap', 'jam', 'hari',
            'efek', 'samping', 'konsultasikan', 'dokter', 'apoteker',
            'sebelum', 'gunakan', 'dosis', 'takaran', 'aturan', 'pakai'
        ]

        words = text_lower.split()
        if len(words) < 5:
            return False

        eng_count = sum(1 for word in words if any(eng_word in word for eng_word in common_english_medical))
        indo_count = sum(1 for word in words if word in common_indonesian)

        # Jika ada kata Inggris medis dan sedikit kata Indonesia
        return eng_count > 1 and indo_count < 3

    def _rag_retrieve(self, query, top_k=3):
        """Retrieve relevant information - query sudah dalam Bahasa Indonesia"""
        query_lower = query.lower()
        self.query_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'language': 'indonesia'
        })

        results = []

        # 1. Deteksi obat dari query
        detected_drugs = self.drug_detector.detect_drug_from_query(query)

        if not detected_drugs:
            # Jika tidak terdeteksi, coba dengan sinonim Bahasa Indonesia
            detected_drugs = self._detect_with_indonesian_synonyms(query)

        if not detected_drugs:
            common_drugs = self.drug_detector.get_all_available_drugs()
        else:
            common_drugs = [drug['drug_name'] for drug in detected_drugs]

        print(f"🔍 Obat terdeteksi: {common_drugs[:top_k]}")

        # 2. Hitung skor relevansi untuk setiap obat
        for drug_name in common_drugs[:top_k]:
            score = 0

            # Cek apakah nama obat atau aliasnya ada dalam query
            if drug_name in query_lower:
                score += 10

            # Cek alias dari drug dictionary
            aliases = self.drug_detector.drug_dictionary.get(drug_name, [])
            for alias in aliases:
                if alias in query_lower:
                    score += 8
                    break

            # Deteksi tipe pertanyaan
            question_keywords = {
                'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai', 'dosis untuk', 'berapa mg', 'aturan minum'],
                'efek': ['efek samping', 'side effect', 'bahaya', 'efeknya', 'akibat', 'resiko'],
                'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan', 'kontraindikasi', 'pantangan'],
                'interaksi': ['interaksi', 'bereaksi dengan', 'makanan', 'minuman', 'interaksinya', 'reaksi'],
                'indikasi': ['untuk apa', 'kegunaan', 'manfaat', 'indikasi', 'guna', 'fungsi', 'khasiat'],
                'peringatan': ['peringatan', 'warning', 'hati-hati', 'waspada', 'catatan']
            }

            for key, keywords in question_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    score += 5  # Tambah skor lebih besar untuk tipe pertanyaan

            if score > 0:
                drug_info = self._get_or_fetch_drug_info(drug_name)
                if drug_info:
                    results.append({
                        'score': score,
                        'drug_info': drug_info,
                        'drug_id': drug_name,
                        'match_type': 'exact' if drug_name in query_lower else 'alias'
                    })

        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"📊 Hasil retrieval: {[(r['drug_id'], r['score']) for r in results[:top_k]]}")
        return results[:top_k]

    def _detect_with_indonesian_synonyms(self, query: str):
        """Deteksi obat dengan sinonim Bahasa Indonesia"""
        indonesian_synonyms = {
            'parasetamol': 'paracetamol',
            'panadol': 'paracetamol',
            'sanmol': 'paracetamol',
            'tempra': 'paracetamol',
            'omeprasol': 'omeprazole',
            'losec': 'omeprazole',
            'omepron': 'omeprazole',
            'amoksisilin': 'amoxicillin',
            'amoxan': 'amoxicillin',
            'moxigra': 'amoxicillin',
            'ibuprom': 'ibuprofen',
            'proris': 'ibuprofen',
            'arthrifen': 'ibuprofen',
            'ibufar': 'ibuprofen',
            'metformin': 'metformin',
            'glucophage': 'metformin',
            'diabex': 'metformin',
            'simvastatin': 'simvastatin',
            'zocor': 'simvastatin',
            'klaritine': 'loratadine',
            'loramine': 'loratadine',
            'aspirin': 'aspirin',
            'aspro': 'aspirin',
            'cardiprin': 'aspirin',
            'vitamin c': 'vitamin c',
            'redoxon': 'vitamin c',
            'enervon c': 'vitamin c',
            'lansoprasol': 'lansoprazole',
            'lanzol': 'lansoprazole',
            'gastracid': 'lansoprazole',
            'sefiksim': 'cefixime',
            'suprax': 'cefixime',
            'setirizin': 'cetirizine',
            'zyrtec': 'cetirizine',
            'dextromethorphan': 'dextromethorphan',
            'dmp': 'dextromethorphan',
            'valtus': 'dextromethorphan',
            'ambroxol': 'ambroxol',
            'mucosolvan': 'ambroxol',
            'broxol': 'ambroxol',
            'salbutamol': 'salbutamol',
            'ventolin': 'salbutamol',
            'asmasolon': 'salbutamol'
        }

        detected_drugs = []
        query_lower = query.lower()

        for indo_name, eng_name in indonesian_synonyms.items():
            if indo_name in query_lower:
                detected_drugs.append({
                    'drug_name': eng_name,
                    'fda_name': self.drug_detector.get_fda_name(eng_name),
                    'alias_found': indo_name,
                    'confidence': 'high'
                })

        return detected_drugs

    def _build_rag_context(self, retrieved_results):
        """Build context untuk RAG generator - 100% BAHASA INDONESIA"""
        if not retrieved_results:
            return "Tidak ada informasi yang relevan ditemukan dalam database FDA."

        context = "## INFORMASI OBAT DARI FDA:\n\n"

        for i, result in enumerate(retrieved_results, 1):
            drug_info = result['drug_info']
            context += f"### OBAT {i}: {drug_info['nama']}\n"

            if 'catatan' in drug_info:
                context += f"- **Catatan:** {drug_info['catatan']}\n"

            # Field-field yang akan ditampilkan - DALAM BAHASA INDONESIA
            fields_to_display = [
                ('Golongan', 'golongan'),
                ('Indikasi', 'indikasi'),
                ('Dosis Dewasa', 'dosis_dewasa'),
                ('Efek Samping', 'efek_samping'),
                ('Kontraindikasi', 'kontraindikasi'),
                ('Interaksi Obat', 'interaksi'),
                ('Peringatan', 'peringatan'),
                ('Bentuk Sediaan', 'bentuk_sediaan')
            ]

            for label, field in fields_to_display:
                if field in drug_info and drug_info[field] != "Tidak tersedia":
                    text = drug_info[field]

                    # Validasi akhir: pastikan tidak ada Inggris
                    if self._has_english_medical_terms(text):
                        text = self.translator.translate_to_indonesian(text)

                    # Potong teks jika terlalu panjang
                    if len(text) > 300:
                        text = text[:297] + "..."

                    context += f"- **{label}:** {text}\n"

            context += "\n"

        # Tambahkan disclaimer yang jelas
        context += """
⚠️ **INFORMASI PENTING:**
- Data ini berasal dari U.S. Food and Drug Administration (FDA) Amerika Serikat
- Semua informasi telah diterjemahkan ke Bahasa Indonesia
- Informasi ini untuk tujuan edukasi dan referensi saja
- **SELALU KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT APAPUN**
- Dosis dan indikasi dapat berbeda untuk setiap pasien
- Obat mungkin memiliki nama merek berbeda di Indonesia
"""

        return context

    def _has_english_medical_terms(self, text: str) -> bool:
        """Cek apakah teks masih mengandung istilah medis Inggris"""
        if not text:
            return False

        english_medical_terms = [
            'indications', 'usage', 'dosage', 'administration',
            'adverse reactions', 'contraindications', 'warnings',
            'drug interactions', 'precautions', 'clinical pharmacology'
        ]

        text_lower = text.lower()
        return any(term in text_lower for term in english_medical_terms)

    def ask_question(self, question):
        """Main RAG interface - OUTPUT 100% BAHASA INDONESIA"""
        try:
            print(f"\n{'='*60}")
            print(f"🤔 PERTANYAAN: {question}")
            print(f"{'='*60}")

            retrieved_results = self._rag_retrieve(question)

            if not retrieved_results:
                print("❌ Tidak ada hasil ditemukan")
                available_drugs = self.drug_detector.get_all_available_drugs()
                indo_drugs = []
                for drug in available_drugs[:10]:
                    if drug == 'paracetamol':
                        indo_drugs.append('parasetamol')
                    elif drug == 'omeprazole':
                        indo_drugs.append('omeprazol')
                    elif drug == 'amoxicillin':
                        indo_drugs.append('amoksisilin')
                    else:
                        indo_drugs.append(drug)

                return f"❌ Tidak ditemukan informasi obat yang relevan dalam database FDA.\n\n💡 **Coba tanyakan tentang:** {', '.join(indo_drugs)}", []

            print(f"✅ Ditemukan {len(retrieved_results)} obat relevan")

            rag_context = self._build_rag_context(retrieved_results)
            answer = self._generate_indonesian_response(question, rag_context)

            sources = []
            seen_drug_names = set()

            for result in retrieved_results:
                drug_name = result['drug_info']['nama']
                if drug_name not in seen_drug_names:
                    sources.append(result['drug_info'])
                    seen_drug_names.add(drug_name)

            print(f"📚 Sumber: {[s['nama'] for s in sources]}")
            print(f"{'='*60}\n")

            return answer, sources

        except Exception as e:
            print(f"❌ Error dalam ask_question: {e}")
            import traceback
            traceback.print_exc()
            return "Maaf, terjadi error dalam sistem. Silakan coba lagi.", []

    def _generate_indonesian_response(self, question, context):
        """Generate response - 100% BAHASA INDONESIA"""
        if not gemini_available:
            # Fallback dengan teks Indonesia yang jelas
            return f"""
**JAWABAN BERDASARKAN DATA FDA:**

{context}

**PERINGATAN PENTING:** Informasi ini berasal dari database FDA Amerika Serikat dan telah diterjemahkan. Informasi ini hanya untuk tujuan edukasi. **SELALU KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER ANDA SEBELUM MENGGUNAKAN OBAT APAPUN.**
            """

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')

            prompt = f"""
            ANDA ADALAH ASISTEN FARMASI DI INDONESIA.
            ANDA HARUS MENJAWAB DENGAN BAHASA INDONESIA 100%.
            JANGAN GUNAKAN SATU KATA BAHASA INGGRIS PUN.

            ## DATA RESMI DARI FDA (SUDAH DITERJEMAHKAN KE BAHASA INDONESIA):
            {context}

            ## PERTANYAAN PASIEN:
            {question}

            ## INSTRUKSI KETAT:
            1. JAWAB PERTANYAAN DI ATAS DENGAN BAHASA INDONESIA 100%
            2. HANYA gunakan informasi dari DATA FDA di atas
            3. JANGAN tambahkan informasi dari pengetahuan Anda sendiri
            4. Berikan jawaban yang SINGKAT, JELAS, dan MUDAH DIPAHAMI
            5. Gunakan format poin-poin jika informasi banyak
            6. SELALU sertakan peringatan: "KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT"
            7. Sebutkan bahwa informasi berasal dari FDA

            ## JAWABAN ANDA (100% BAHASA INDONESIA):
            """

            response = model.generate_content(prompt)
            answer = response.text.strip()

            # VALIDASI AKHIR: Pastikan tidak ada bahasa Inggris yang tidak perlu
            answer_lower = answer.lower()

            # Daftar kata Inggris yang TIDAK BOLEH ada (kecuali dalam konteks numerik atau "FDA")
            forbidden_english = [
                ' the ', ' and ', ' for ', ' with ', ' that ', ' this ',
                ' are ', ' you ', ' have ', ' from ', ' should ', ' may ',
                ' can ', ' will ', ' must ', ' take ', ' use ', ' dosage ',
                ' administration', ' adverse', ' reactions', ' contraindications'
            ]

            # Cek jika ada kata Inggris yang bukan bagian dari "500 mg" atau "FDA"
            needs_retranslation = False
            for forbidden in forbidden_english:
                if forbidden in answer_lower:
                    # Cek konteks - jika ini bagian dari "FDA" atau angka+satuan, boleh
                    if not (forbidden.strip() == 'fda' or self._is_numeric_context(answer_lower, forbidden)):
                        needs_retranslation = True
                        break

            if needs_retranslation:
                print("⚠️  Masih ada bahasa Inggris, menerjemahkan ulang jawaban...")
                answer = self.translator.translate_to_indonesian(answer)

            # Pastikan ada disclaimer
            if 'konsultasikan' not in answer_lower or 'dokter' not in answer_lower:
                answer += "\n\n⚠️ **KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT.**"

            # Pastikan ada referensi FDA
            if 'fda' not in answer_lower and 'food and drug' not in answer_lower:
                answer = "**Berdasarkan data resmi dari U.S. Food and Drug Administration (FDA):**\n\n" + answer

            return answer

        except Exception as e:
            print(f"❌ Error generate response: {e}")
            # Fallback ke format sederhana dalam Bahasa Indonesia
            return f"""
**BERDASARKAN DATA FDA:**

{context}

**PERINGATAN:** Informasi ini untuk edukasi. Selalu konsultasikan dengan dokter sebelum menggunakan obat.
            """

    def _is_numeric_context(self, text: str, word: str) -> bool:
        """Cek apakah kata muncul dalam konteks numerik (seperti "500 mg")"""
        import re

        # Pola untuk angka diikuti satuan
        patterns = [
            r'\d+\s*mg', r'\d+\s*ml', r'\d+\s*g', r'\d+\s*kg',
            r'\d+\s*tablet', r'\d+\s*capsule', r'\d+\s*jam',
            r'\d+\s*hari', r'\d+\s*minggu', r'\d+\s*bulan'
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True

        return False

    def _update_conversation_context(self, question, answer, sources):
        """Update conversation context"""
        if sources:
            self.current_context = {
                'current_drug': sources[0]['nama'],
                'bahasa': 'Indonesia',
                'timestamp': datetime.now()
            }

# ===========================================
# ENHANCED DRUG DETECTOR
# ===========================================
class EnhancedDrugDetector:
    def __init__(self):
        # Dictionary dengan nama obat dan aliasnya (termasuk nama Indonesia)
        self.drug_dictionary = {
            'paracetamol': ['acetaminophen', 'paracetamol', 'panadol', 'sanmol', 'tempra', 'parasetamol', 'biogesic'],
            'omeprazole': ['omeprazole', 'prilosec', 'losec', 'omepron', 'omeprazol', 'gastrul'],
            'amoxicillin': ['amoxicillin', 'amoxilin', 'amoxan', 'moxigra', 'amoksisilin', 'moxypen'],
            'ibuprofen': ['ibuprofen', 'proris', 'arthrifen', 'ibufar', 'ibuprom', 'profen'],
            'metformin': ['metformin', 'glucophage', 'metfor', 'diabex', 'glucofage'],
            'atorvastatin': ['atorvastatin', 'lipitor', 'atorva', 'tovast', 'ator', 'lipistat'],
            'simvastatin': ['simvastatin', 'zocor', 'simvor', 'lipostat', 'simva'],
            'loratadine': ['loratadine', 'clarityne', 'loramine', 'allertine', 'klaritin', 'loradin'],
            'aspirin': ['aspirin', 'aspro', 'aspilet', 'cardiprin', 'aspilets'],
            'vitamin c': ['ascorbic acid', 'vitamin c', 'redoxon', 'enervon c', 'vitacimin'],
            'lansoprazole': ['lansoprazole', 'prevacid', 'lanzol', 'gastracid', 'lansoprasol'],
            'esomeprazole': ['esomeprazole', 'nexium', 'esotrax', 'esomep', 'esomeprazol'],
            'cefixime': ['cefixime', 'suprax', 'cefix', 'fixcef', 'sefiksim'],
            'cetirizine': ['cetirizine', 'zyrtec', 'cetrizin', 'allertec', 'setirizin'],
            'dextromethorphan': ['dextromethorphan', 'dmp', 'dextro', 'valtus', 'dexteem'],
            'ambroxol': ['ambroxol', 'mucosolvan', 'ambrox', 'broxol', 'mucos', 'vicks'],
            'salbutamol': ['albuterol', 'salbutamol', 'ventolin', 'salbu', 'asmasolon', 'salbumol']
        }

        # Mapping nama ke nama FDA resmi
        self.fda_name_mapping = {
            'paracetamol': 'acetaminophen',
            'vitamin c': 'ascorbic acid',
            'salbutamol': 'albuterol',
            'parasetamol': 'acetaminophen',
            'amoksisilin': 'amoxicillin',
            'omeprazol': 'omeprazole',
            'lansoprasol': 'lansoprazole',
            'esomeprazol': 'esomeprazole',
            'setirizin': 'cetirizine',
            'sefiksim': 'cefixime'
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
# KELAS EVALUASI
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
# FUNGSI UTAMA STREAMLIT
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

    # ===========================================
    # HALAMAN CHATBOT
    # ===========================================
    if page == "🏠 Chatbot Obat":
        st.title("💊 Sistem Tanya Jawab Obat")
        st.markdown("Sistem informasi obat dengan data langsung dari **FDA API** dan terjemahan otomatis ke Bahasa Indonesia")

        st.markdown("""
        <div class="fda-indicator">
            🏥 <strong>DATA RESMI FDA</strong> - Informasi obat langsung dari U.S. Food and Drug Administration
            <br>📚 <strong>100% BAHASA INDONESIA</strong> - Semua informasi telah diterjemahkan
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💬 Percakapan")

        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-message">
                <h3>👋 Selamat Datang di Asisten Obat</h3>
                <p>Dapatkan informasi obat <strong>langsung dari database resmi FDA</strong> dengan terjemahan otomatis ke Bahasa Indonesia</p>
                <p><strong>💡 Contoh pertanyaan:</strong></p>
                <p>"Apa dosis paracetamol?" | "Efek samping amoxicillin?" | "Interaksi obat omeprazole?"</p>
                <p>"Untuk apa metformin digunakan?" | "Peringatan penggunaan ibuprofen?"</p>
                <p><em>📝 Catatan: Semua jawaban dalam Bahasa Indonesia. Beberapa obat memiliki nama berbeda di FDA.</em></p>
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

                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Informasi Obat dari FDA"):
                            for drug in message["sources"]:
                                card_content = f"""
                                <div class="drug-card">
                                    <h4>💊 {drug['nama']}</h4>
                                    <p><strong>Golongan:</strong> {drug['golongan']}</p>
                                    <p><strong>Merek Dagang:</strong> {drug['merek_dagang']}</p>
                                    <p><strong>Indikasi:</strong> {drug['indikasi'][:150]}...</p>
                                    <p><strong>Bentuk Sediaan:</strong> {drug['bentuk_sediaan']}</p>
                                """
                                if 'catatan' in drug:
                                    card_content += f"<p><em>📝 {drug['catatan']}</em></p>"
                                if 'terjemahan_otomatis' in drug and drug['terjemahan_otomatis']:
                                    card_content += "<p><small>🌐 <em>Informasi telah diterjemahkan dari data FDA</em></small></p>"
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
                    use_container_width=True,
                    type="primary"
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

            with st.spinner("🔍 Mengakses FDA API dan menerjemahkan..."):
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

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ PERINGATAN MEDIS PENTING:</strong>
            <ul>
                <li>Informasi ini berasal dari database FDA Amerika Serikat dan telah diterjemahkan ke Bahasa Indonesia</li>
                <li>Informasi ini untuk tujuan edukasi dan referensi saja</li>
                <li><strong>SELALU KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT</strong></li>
                <li>Dosis dan indikasi dapat berbeda untuk setiap pasien</li>
                <li>Obat mungkin memiliki nama merek berbeda di Indonesia</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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

            **3. Bahasa Indonesia 100%**
            - **Fungsi**: Memastikan semua output dalam Bahasa Indonesia
            - **Target**: 100% teks dalam Bahasa Indonesia
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
                                    # Cek bahasa
                                    answer_lower = answer.lower()
                                    english_words = ['the', 'and', 'for', 'with', 'should', 'may']
                                    has_english = any(word in answer_lower for word in english_words)
                                    if has_english:
                                        st.warning("⚠️ Masih ada kata Inggris dalam jawaban")
                                    else:
                                        st.success("✅ 100% Bahasa Indonesia")
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
        "💊 **Sistem Tanya Jawab Obat dengan RAG** • 100% Bahasa Indonesia • Evaluasi 2 Metrik Inti (MRR & Faithfulness)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
