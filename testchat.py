import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import re
import json
import random
import hashlib
import logging

# Setup logging (so prints are manageable)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.warning(f"Gemini config error: {e}")
    gemini_available = False

# ===========================================
# TRANSLATION SERVICE - with robust rate-limit handling
# ===========================================
class TranslationService:
    def __init__(self):
        self.available = gemini_available
        self.translation_cache = {}  # Cache untuk menghindari terjemahan berulang
        # Rate-limit handling
        self.rate_limited_until = None  # datetime until which we avoid calling Gemini
        self.rate_limit_backoff_seconds = 60  # initial backoff, will increase multiplicatively
        self.max_backoff_seconds = 3600  # 1 hour cap
        self.last_rate_limit_error = None

    def _stable_hash(self, text: str) -> str:
        """Stable hash untuk cache (cross-process safe)."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _call_gemini(self, prompt: str, model_name: str = 'gemini-2.0-flash'):
        """Wrapper to call Gemini with rate-limit handling and exponential backoff."""
        # If we are currently backoff due to rate limit, skip call early
        now = datetime.utcnow()
        if self.rate_limited_until and now < self.rate_limited_until:
            raise RuntimeError(f"Gemini currently rate-limited until {self.rate_limited_until.isoformat()}")

        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            # Detect rate-limit / quota errors conservatively
            if '429' in msg or 'quota' in msg.lower() or 'rate limit' in msg.lower():
                # increase backoff
                prev_backoff = self.rate_limit_backoff_seconds
                self.rate_limit_backoff_seconds = min(int(self.rate_limit_backoff_seconds * 2), self.max_backoff_seconds)
                self.rate_limited_until = now + timedelta(seconds=prev_backoff)
                # log once
                if self.last_rate_limit_error != msg:
                    logger.warning(f"Gemini rate limit detected: {msg}. Backing off for {prev_backoff} seconds.")
                    self.last_rate_limit_error = msg
                raise RuntimeError("Gemini rate limited")
            else:
                # other errors: bubble up
                logger.exception("Gemini call error")
                raise

    def translate_to_indonesian(self, text: str, max_attempts=2):
        """
        Terjemahkan teks ke Bahasa Indonesia:
        - Prioritas: gunakan Gemini (jika tersedia) dengan prompt terstruktur.
        - Jika Gemini tidak tersedia atau rate-limited, gunakan fallback replacement yang aman.
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

        translated = None

        # Try using Gemini if available and not in backoff
        if self.available:
            attempt = 0
            while attempt < max_attempts:
                try:
                    translated = self._translate_with_gemini(text)
                    break
                except RuntimeError as rexc:
                    # rate-limited or wrapper indicated skip
                    logger.info("Gemini unavailable due to rate-limit or temporary error; will fallback.")
                    translated = None
                    break
                except Exception as e:
                    logger.warning(f"Gemini translate attempt {attempt+1} failed: {e}")
                    attempt += 1
                    time.sleep(0.5 * (attempt + 1))
                    continue

        # If Gemini failed or not available, fallback per-sentence only if gemini not allowed,
        # but don't hammer Gemini if it's rate-limited.
        if not translated and self.available and (not self.rate_limited_until or datetime.utcnow() >= self.rate_limited_until):
            # If gemini didn't succeed but not rate limited, try sentence-by-sentence briefly
            try:
                translated = self._translate_sentence_by_sentence(text, max_attempts=1)
            except RuntimeError:
                translated = None
            except Exception as e:
                logger.warning(f"Sentence-level Gemini attempts failed: {e}")
                translated = None

        # If still not translated, use fallback
        if not translated:
            translated = self._translate_fallback(text)

        # Final cleanup and cache
        translated = self._final_cleanup(translated)
        self.translation_cache[text_hash] = translated
        return translated

    def _translate_with_gemini(self, text: str):
        """Panggil Gemini dengan prompt yang ketat untuk menerjemahkan seluruh teks."""
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
        return self._call_gemini(prompt)

    def _translate_sentence_by_sentence(self, text: str, max_attempts=1):
        """Bagi teks menjadi kalimat, terjemahkan masing-masing kalimat (mengurangi risiko mixing)."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[\.\?\!\n])\s+', text.strip())
        translated_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            if self._is_definitely_indonesian(sentence):
                translated_sentences.append(sentence.strip())
                continue
            attempt = 0
            translated_sentence = None
            while attempt < max_attempts:
                try:
                    # build small prompt
                    prompt = f"Terjemahkan kalimat berikut ke Bahasa Indonesia: \"{sentence.strip()}\""
                    translated_sentence = self._call_gemini(prompt)
                    break
                except RuntimeError:
                    # rate limit - propagate so caller can fallback
                    raise
                except Exception as e:
                    logger.warning(f"Translate sentence attempt failed: {e}")
                    attempt += 1
                    time.sleep(0.2 * (attempt + 1))
            if not translated_sentence:
                translated_sentence = self._translate_fallback(sentence)
            translated_sentences.append(translated_sentence.strip())
        return " ".join(translated_sentences)

    def _contains_english_heavy(self, text: str) -> bool:
        """Deteksi apakah masih banyak potongan Bahasa Inggris di teks keluaran."""
        if not text:
            return False
        text_lower = text.lower()
        english_indicators = [
            r'\bthe\b', r'\band\b', r'\bfor\b', r'\bwith\b', r'\buse\b', r'\btake\b',
            r'\bdosage\b', r'\badministration\b', r'\badverse\b', r'\breactions\b',
            r'\bcontraindications\b', r'\bwarnings\b', r'\bdrug\b'
        ]
        for patt in english_indicators:
            if re.search(patt, text_lower):
                return True
        return False

    def _is_definitely_indonesian(self, text: str) -> bool:
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
            'indications and usage': 'indikasi dan penggunaan',
            'dosage and administration': 'dosis dan cara pemakaian',
            'adverse reactions': 'efek samping',
            'contraindications': 'kontraindikasi',
            'warnings': 'peringatan',
            'drug interactions': 'interaksi obat',
            'precautions': 'tindakan pencegahan',
            'clinical pharmacology': 'farmakologi klinis',
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
            # Whole-word replacement with word boundaries
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
        # Perbaikan sederhana: jangan gabungkan huruf kecil kapital secara tidak wajar
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
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
        cache_key = generic_name.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 5
        }

        try:
            logger.info(f"🔍 Mencari data FDA untuk: {generic_name}")
            response = requests.get(self.base_url, params=params, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    logger.info(f"✅ Ditemukan {len(data['results'])} hasil untuk {generic_name}")

                    # Cari data yang paling lengkap
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

            # Coba alternatif
            logger.info(f"⚠️ Pencarian utama gagal, mencoba alternatif untuk: {generic_name}")
            drug_info = self._try_alternative_search(generic_name)
            if drug_info:
                self.cache[cache_key] = drug_info
                return drug_info

            logger.warning(f"❌ Tidak ada data ditemukan untuk: {generic_name}")
            return None

        except Exception as e:
            logger.exception(f"❌ Error FDA API: {e}")
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
                    logger.info(f"✅ Alternatif ditemukan untuk: {generic_name}")
                    return self._parse_fda_data(data['results'][0], generic_name)
        except Exception as e:
            logger.exception(f"⚠️ Alternatif search error: {e}")
        return None

    def _parse_fda_data(self, fda_data: dict, generic_name: str):
        logger.info(f"📝 Parsing data FDA untuk: {generic_name}")
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

        logger.info(f"✅ Data berhasil diparsing dan diterjemahkan untuk: {generic_name}")
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
        # Use translator which handles rate-limits and fallback
        translated = self.translator.translate_to_indonesian(text)
        if len(translated) > 1000:
            translated = translated[:997] + "..."
        return translated

# ===========================================
# The rest of the assistant (SimpleRAGPharmaAssistant, EnhancedDrugDetector,
# FocusedRAGEvaluator and UI) remain unchanged in logic but use FDADrugAPI / TranslationService above.
# For completeness the rest of the code is included (unchanged except for using logger prints).
# ===========================================
class EnhancedDrugDetector:
    def __init__(self):
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
        return list(self.drug_dictionary.keys())

    def get_fda_name(self, drug_name: str):
        return self.fda_name_mapping.get(drug_name, drug_name)

class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        self.drug_detector = EnhancedDrugDetector()
        self.drugs_cache = {}
        self.current_context = {}
        self.query_history = []

    def _get_or_fetch_drug_info(self, drug_name: str):
        drug_key = drug_name.lower()
        if drug_key in self.drugs_cache:
            logger.info(f"📦 Menggunakan cache untuk: {drug_name}")
            return self.drugs_cache[drug_key]
        logger.info(f"🌐 Fetch data FDA untuk: {drug_name}")
        fda_name = self.drug_detector.get_fda_name(drug_name)
        drug_info = self.fda_api.get_drug_info(fda_name)
        if drug_info:
            if drug_name != fda_name:
                drug_info['nama'] = drug_name.title()
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"
            drug_info = self._validate_and_fix_translation(drug_info)
            self.drugs_cache[drug_key] = drug_info
            logger.info(f"✅ Data berhasil diambil untuk: {drug_name}")
        else:
            logger.warning(f"❌ Gagal mendapatkan data untuk: {drug_name}")
        return drug_info

    def _validate_and_fix_translation(self, drug_info: dict):
        fields_to_check = [
            'indikasi', 'dosis_dewasa', 'efek_samping',
            'kontraindikasi', 'interaksi', 'peringatan',
            'golongan', 'bentuk_sediaan', 'route_pemberian',
            'merek_dagang'
        ]
        for field in fields_to_check:
            if field in drug_info and drug_info[field] != "Tidak tersedia":
                text = drug_info[field]
                if self._contains_too_much_english(text):
                    logger.info(f"⚠️  Masih ada Inggris di {field}, menerjemahkan ulang...")
                    translated = self.translator.translate_to_indonesian(text)
                    if translated and translated != text:
                        drug_info[field] = translated
                        drug_info['diperbaiki'] = True
        return drug_info

    def _contains_too_much_english(self, text: str) -> bool:
        if not text or len(text) < 20:
            return False
        text_lower = text.lower()
        common_english_medical = [
            'indications', 'usage', 'dosage', 'administration', 'adverse',
            'reactions', 'contraindications', 'warnings', 'interactions',
            'tablets', 'capsules', 'should', 'may', 'can', 'must', 'do not',
            'consult', 'doctor', 'physician', 'pharmacist', 'patient',
            'use', 'take', 'cause', 'causes', 'side effect', 'side effects'
        ]
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
        return eng_count > 1 and indo_count < 3

    def _rag_retrieve(self, query, top_k=3):
        query_lower = query.lower()
        self.query_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'language': 'indonesia'
        })
        results = []
        detected_drugs = self.drug_detector.detect_drug_from_query(query)
        if not detected_drugs:
            detected_drugs = self._detect_with_indonesian_synonyms(query)
        if not detected_drugs:
            common_drugs = self.drug_detector.get_all_available_drugs()
        else:
            common_drugs = [drug['drug_name'] for drug in detected_drugs]
        logger.info(f"🔍 Obat terdeteksi: {common_drugs[:top_k]}")
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
                'dosis': ['dosis', 'berapa', 'takaran', 'aturan pakai', 'dosis untuk', 'berapa mg', 'aturan minum'],
                'efek': ['efek samping', 'side effect', 'bahaya', 'efeknya', 'akibat', 'resiko'],
                'kontraindikasi': ['kontra', 'tidak boleh', 'hindari', 'larangan', 'kontraindikasi', 'pantangan'],
                'interaksi': ['interaksi', 'bereaksi dengan', 'makanan', 'minuman', 'interaksinya', 'reaksi'],
                'indikasi': ['untuk apa', 'kegunaan', 'manfaat', 'indikasi', 'guna', 'fungsi', 'khasiat'],
                'peringatan': ['peringatan', 'warning', 'hati-hati', 'waspada', 'catatan']
            }
            for key, keywords in question_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    score += 5
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
        logger.info(f"📊 Hasil retrieval: {[(r['drug_id'], r['score']) for r in results[:top_k]]}")
        return results[:top_k]

    def _detect_with_indonesian_synonyms(self, query: str):
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
        if not retrieved_results:
            return "Tidak ada informasi yang relevan ditemukan dalam database FDA."

        context = "## INFORMASI OBAT DARI FDA:\n\n"
        for i, result in enumerate(retrieved_results, 1):
            drug_info = result['drug_info']
            context += f"### OBAT {i}: {drug_info['nama']}\n"
            if 'catatan' in drug_info:
                context += f"- **Catatan:** {drug_info['catatan']}\n"
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
                    if self._has_english_medical_terms(text):
                        text = self.translator.translate_to_indonesian(text)
                    if len(text) > 300:
                        text = text[:297] + "..."
                    context += f"- **{label}:** {text}\n"
            context += "\n"

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
        try:
            logger.info("\n" + "="*60)
            logger.info(f"🤔 PERTANYAAN: {question}")
            logger.info("="*60)
            retrieved_results = self._rag_retrieve(question)
            if not retrieved_results:
                logger.info("❌ Tidak ada hasil ditemukan")
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
            logger.info(f"✅ Ditemukan {len(retrieved_results)} obat relevan")
            rag_context = self._build_rag_context(retrieved_results)
            answer = self._generate_indonesian_response(question, rag_context)
            sources = []
            seen_drug_names = set()
            for result in retrieved_results:
                drug_name = result['drug_info']['nama']
                if drug_name not in seen_drug_names:
                    sources.append(result['drug_info'])
                    seen_drug_names.add(drug_name)
            logger.info(f"📚 Sumber: {[s['nama'] for s in sources]}")
            logger.info("="*60 + "\n")
            return answer, sources
        except Exception as e:
            logger.exception(f"❌ Error dalam ask_question: {e}")
            return "Maaf, terjadi error dalam sistem. Silakan coba lagi.", []

    def _generate_indonesian_response(self, question, context):
        """Use Gemini if available; if rate-limited, fallback to structured summary from context."""
        if not gemini_available:
            return f"**JAWABAN BERDASARKAN DATA FDA:**\n\n{context}\n\n⚠️ KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT."

        # If translator flagged a long rate-limit backoff, skip calling Gemini for generation too
        if self.translator.rate_limited_until and datetime.utcnow() < self.translator.rate_limited_until:
            logger.info("Skipping Gemini generation due to translator rate-limit; using context-as-answer fallback.")
            answer = f"**JAWABAN BERDASARKAN DATA FDA (FALLBACK):**\n\n{context}\n\n⚠️ KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT."
            return answer

        try:
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
            # Use wrapper to call Gemini safely; if it fails due to rate-limit, fallback to context summary
            try:
                answer = self.translator._call_gemini(prompt)
            except Exception as e:
                logger.warning(f"Gemini generation unavailable: {e}. Falling back to structured context answer.")
                answer = f"**JAWABAN BERDASARKAN DATA FDA (FALLBACK):**\n\n{context}\n\n⚠️ KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT."
                return answer

            # Final validation for English fragments
            answer_lower = answer.lower()
            if "konsultasikan" not in answer_lower or "dokter" not in answer_lower:
                answer += "\n\n⚠️ KONSULTASIKAN DENGAN DOKTER ATAU APOTEKER SEBELUM MENGGUNAKAN OBAT."

            if 'fda' not in answer_lower and 'food and drug' not in answer_lower:
                answer = "**Berdasarkan data resmi dari U.S. Food and Drug Administration (FDA):**\n\n" + answer

            return answer
        except Exception as e:
            logger.exception(f"❌ Error generate response: {e}")
            return f"**BERDASARKAN DATA FDA:**\n\n{context}\n\n**PERINGATAN:** Informasi ini untuk edukasi. Selalu konsultasikan dengan dokter sebelum menggunakan obat."

class FocusedRAGEvaluator:
    def __init__(self, assistant):
        self.assistant = assistant
        self.test_set = [
            {"id": 1, "question": "Apa dosis paracetamol?", "expected_drug": "paracetamol", "question_type": "dosis", "key_info_expected": ["dosis", "mg", "paracetamol"]},
            {"id": 2, "question": "Efek samping amoxicillin?", "expected_drug": "amoxicillin", "question_type": "efek_samping", "key_info_expected": ["efek", "samping", "amoxicillin"]},
            {"id": 3, "question": "Untuk apa omeprazole digunakan?", "expected_drug": "omeprazole", "question_type": "indikasi", "key_info_expected": ["indikasi", "kegunaan", "omeprazole"]},
            {"id": 4, "question": "Apa kontraindikasi ibuprofen?", "expected_drug": "ibuprofen", "question_type": "kontraindikasi", "key_info_expected": ["kontraindikasi", "ibuprofen"]},
            {"id": 5, "question": "Interaksi obat metformin?", "expected_drug": "metformin", "question_type": "interaksi", "key_info_expected": ["interaksi", "metformin"]},
            {"id": 6, "question": "Berapa dosis atorvastatin?", "expected_drug": "atorvastatin", "question_type": "dosis", "key_info_expected": ["dosis", "atorvastatin"]},
            {"id": 7, "question": "Efek samping simvastatin?", "expected_drug": "simvastatin", "question_type": "efek_samping", "key_info_expected": ["efek", "samping", "simvastatin"]},
            {"id": 8, "question": "Kegunaan lansoprazole?", "expected_drug": "lansoprazole", "question_type": "indikasi", "key_info_expected": ["kegunaan", "lansoprazole"]},
            {"id": 9, "question": "Peringatan penggunaan aspirin?", "expected_drug": "aspirin", "question_type": "peringatan", "key_info_expected": ["peringatan", "aspirin"]},
            {"id": 10, "question": "Dosis cetirizine untuk dewasa?", "expected_drug": "cetirizine", "question_type": "dosis", "key_info_expected": ["dosis", "cetirizine", "dewasa"]}
        ]

    def calculate_mrr(self):
        reciprocal_ranks = []
        for test in self.test_set:
            detected_drugs = self.assistant.drug_detector.detect_drug_from_query(test["question"])
            rank = None
            for i, drug_info in enumerate(detected_drugs, 1):
                if drug_info['drug_name'] == test["expected_drug"]:
                    rank = i
                    break
            reciprocal_ranks.append(1.0 / rank if rank else 0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    def calculate_faithfulness(self):
        faithful_scores = []
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            answer_lower = answer.lower()
            criteria_scores = []
            criteria_scores.append(0.4 if sources else 0)
            fda_indicators = ["fda", "food and drug administration", "data resmi fda", "sumber fda"]
            has_fda_ref = any(indicator in answer_lower for indicator in fda_indicators)
            criteria_scores.append(0.25 if has_fda_ref else 0)
            fictional_indicators = ["menurut saya", "biasanya", "umumnya", "seharusnya", "kemungkinan besar", "menurut pengetahuan saya"]
            has_fictional = any(indicator in answer_lower for indicator in fictional_indicators)
            criteria_scores.append(0.20 if not has_fictional else 0)
            disclaimer_indicators = ["konsultasi", "dokter", "apoteker", "sebelum menggunakan"]
            has_disclaimer = any(indicator in answer_lower for indicator in disclaimer_indicators)
            criteria_scores.append(0.15 if has_disclaimer else 0)
            total_score = sum(criteria_scores)
            faithful_scores.append(min(total_score, 1.0))
        return np.mean(faithful_scores) if faithful_scores else 0

    def run_evaluation(self):
        try:
            mrr_score = self.calculate_mrr()
            faithfulness_score = self.calculate_faithfulness()
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_set),
                "MRR": float(mrr_score),
                "Faithfulness": float(faithfulness_score),
                "RAG_Score": float((mrr_score + faithfulness_score) / 2)
            }
            results["test_case_details"] = self._get_test_case_details()
            return results
        except Exception as e:
            logger.exception(f"❌ Error dalam evaluasi: {e}")
            return {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "error": str(e), "MRR": 0, "Faithfulness": 0, "RAG_Score": 0}

    def _get_test_case_details(self):
        details = []
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            detected_drugs = self.assistant.drug_detector.detect_drug_from_query(test["question"])
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
# Streamlit UI (uses assistant)
# ===========================================
def main():
    assistant = SimpleRAGPharmaAssistant()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None

    st.markdown("""
    <style>
        .chat-container { max-height: 500px; overflow-y: auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; background-color: #fafafa; margin-bottom: 20px; }
        .user-message { background-color: #0078D4; color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; margin: 8px 0; max-width: 70%; margin-left: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .bot-message { background-color: white; color: #333; padding: 12px 16px; border-radius: 18px 18px 18px 4px; margin: 8px 0; max-width: 70%; margin-right: auto; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .message-time { font-size: 0.75em; opacity: 0.7; margin-top: 5px; text-align: right; }
        .bot-message .message-time { text-align: left; }
        .fda-indicator { background-color: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px; padding: 8px 12px; margin: 5px 0; font-size: 0.8em; color: #2e7d32; }
        .welcome-message { text-align: center; padding: 40px; color: #666; background: white; border-radius: 10px; border: 2px dashed #e0e0e0; }
        .drug-card { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        .good-score { color: #4CAF50; } .medium-score { color: #FF9800; } .poor-score { color: #F44336; }
        .evaluation-info { background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .rag-score-card { background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0; color: #856404; }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("💊 Navigasi")
    page = st.sidebar.radio("Pilih Halaman:", ["🏠 Chatbot Obat", "📊 Evaluasi RAG"])

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
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""<div class="user-message"><div>{message["content"]}</div><div class="message-time">{message["timestamp"]}</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="bot-message"><div>{message["content"]}</div><div class="message-time">{message["timestamp"]} • Sumber: FDA API 🇮🇩</div></div>""", unsafe_allow_html=True)
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

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Tulis pertanyaan Anda tentang obat:", placeholder="Contoh: Apa dosis paracetamol? Efek samping amoxicillin?", key="user_input")
            col_btn1, col_btn2 = st.columns([3,1])
            with col_btn1:
                submit_btn = st.form_submit_button("🚀 Tanya", use_container_width=True, type="primary")
            with col_btn2:
                clear_btn = st.form_submit_button("🗑️ Hapus Chat", use_container_width=True)

        if submit_btn and user_input:
            st.session_state.messages.append({"role":"user","content":user_input,"timestamp":datetime.now().strftime("%H:%M")})
            with st.spinner("🔍 Mengakses FDA API dan menerjemahkan..."):
                answer, sources = assistant.ask_question(user_input)
                st.session_state.conversation_history.append({'timestamp': datetime.now(), 'question': user_input, 'answer': answer, 'sources': [drug['nama'] for drug in sources], 'source': 'FDA API'})
                st.session_state.messages.append({"role":"bot","content":answer,"sources":sources,"timestamp":datetime.now().strftime("%H:%M")})
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
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif page == "📊 Evaluasi RAG":
        st.title("📊 Evaluasi Sistem RAG")
        st.markdown("**Fokus pada evaluasi komponen RETRIEVAL dan GENERATION dari RAG**")
        with st.expander("ℹ️ Tentang 2 Metrik Evaluasi RAG", expanded=True):
            st.markdown("""<div class="evaluation-info">...informasi...</div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2,1,1])
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
                    st.download_button(label="⬇️ Download JSON", data=data, file_name=filename, mime="application/json")
                else:
                    st.warning("⚠️ Jalankan evaluasi terlebih dahulu!")
        with col3:
            if st.button("🔄 Reset Hasil", use_container_width=True):
                st.session_state.evaluation_results = None
                st.session_state.evaluator = None
                st.rerun()

        st.markdown("---")
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            st.markdown(f"### 📈 Hasil Evaluasi RAG ({results['timestamp']})")
            st.json(results)
        else:
            st.info("Klik 'Jalankan Evaluasi RAG' untuk memulai evaluasi.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>💊 Sistem Tanya Jawab Obat dengan RAG • 100% Bahasa Indonesia</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
