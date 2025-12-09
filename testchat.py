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
# KELAS EVALUASI RAG YANG DIPERBAIKI
# ===========================================
class RAGEvaluator:
    def __init__(self, assistant):
        self.assistant = assistant
        
        # PERBAIKAN 1: Test set yang lebih realistis berdasarkan data FDA
        # Hanya berisi pertanyaan yang kemungkinan besar ada di FDA
        self.test_set = [
            {
                "id": 1,
                "question": "Apa dosis paracetamol?",
                "expected_drug": "paracetamol",
                "question_type": "dosis",
                "expected_info_type": "dosage_and_administration"
            },
            {
                "id": 2,
                "question": "Efek samping amoxicillin?",
                "expected_drug": "amoxicillin",
                "question_type": "efek_samping",
                "expected_info_type": "adverse_reactions"
            },
            {
                "id": 3,
                "question": "Untuk apa omeprazole digunakan?",
                "expected_drug": "omeprazole",
                "question_type": "indikasi",
                "expected_info_type": "indications_and_usage"
            },
            {
                "id": 4,
                "question": "Apa kontraindikasi ibuprofen?",
                "expected_drug": "ibuprofen",
                "question_type": "kontraindikasi",
                "expected_info_type": "contraindications"
            },
            {
                "id": 5,
                "question": "Interaksi obat metformin?",
                "expected_drug": "metformin",
                "question_type": "interaksi",
                "expected_info_type": "drug_interactions"
            },
            {
                "id": 6,
                "question": "Berapa dosis atorvastatin?",
                "expected_drug": "atorvastatin",
                "question_type": "dosis",
                "expected_info_type": "dosage_and_administration"
            },
            {
                "id": 7,
                "question": "Efek samping simvastatin?",
                "expected_drug": "simvastatin",
                "question_type": "efek_samping",
                "expected_info_type": "adverse_reactions"
            },
            {
                "id": 8,
                "question": "Kegunaan lansoprazole?",
                "expected_drug": "lansoprazole",
                "question_type": "indikasi",
                "expected_info_type": "indications_and_usage"
            },
            {
                "id": 9,
                "question": "Peringatan penggunaan aspirin?",
                "expected_drug": "aspirin",
                "question_type": "peringatan",
                "expected_info_type": "warnings"
            },
            {
                "id": 10,
                "question": "Dosis cetirizine?",
                "expected_drug": "cetirizine",
                "question_type": "dosis",
                "expected_info_type": "dosage_and_administration"
            }
        ]
        
        # Mapping antara question type dan keywords yang diharapkan
        self.question_type_keywords = {
            "dosis": ["mg", "dosis", "tablet", "kapsul", "sekali", "hari", "penggunaan", "diberikan"],
            "efek_samping": ["efek", "samping", "reaksi", "adverse", "gejala", "mual", "pusing", "diare"],
            "indikasi": ["untuk", "mengobati", "indikasi", "kegunaan", "penggunaan", "terapi", "penyakit"],
            "kontraindikasi": ["tidak", "boleh", "kontra", "hindari", "larangan", "dilarang", "jangan"],
            "interaksi": ["interaksi", "bereaksi", "bersamaan", "kombinasi", "makanan", "minuman", "alkohol"],
            "peringatan": ["peringatan", "warning", "hati-hati", "waspada", "perhatian", "risiko"]
        }
        
        # Kata-kata penting untuk semantic similarity
        self.important_words = ["mg", "dosis", "efek", "samping", "indikasi", "kontraindikasi", 
                              "interaksi", "peringatan", "FDA", "obat", "penggunaan", "tablet",
                              "reaksi", "gejala", "mengobati", "hindari", "kombinasi", "risiko"]
    
    # ===========================================
    # METRIK 1: MEAN RECIPROCAL RANK (MRR)
    # ===========================================
    def calculate_mrr(self):
        """Hitung MRR untuk drug detection accuracy"""
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
    
    # ===========================================
    # METRIK 2: FAITHFULNESS
    # ===========================================
    def calculate_faithfulness(self):
        """Hitung kesetiaan jawaban terhadap sumber FDA"""
        faithful_scores = []
        
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            answer_lower = answer.lower()
            
            # Check 1: Apakah ada sumber FDA?
            has_source = sources and len(sources) > 0
            
            # Check 2: Apakah jawaban mengandung referensi FDA?
            fda_indicators = ["fda", "food and drug administration", "data resmi", "sumber resmi"]
            has_fda_ref = any(indicator in answer_lower for indicator in fda_indicators)
            
            # Check 3: Apakah jawaban mengandung disclaimer?
            disclaimer_indicators = ["konsultasi", "dokter", "apoteker", "sebelum menggunakan", "peringatan medis"]
            has_disclaimer = any(indicator in answer_lower for indicator in disclaimer_indicators)
            
            # Check 4: Apakah jawaban mengandung klaim berlebihan?
            # Kata-kata yang menunjukkan informasi fiktif atau terlalu umum
            fictional_indicators = [
                "menurut penelitian saya", "biasanya", "umumnya", "seharusnya", 
                "kemungkinan besar", "rata-rata", "pada umumnya", "kebanyakan"
            ]
            has_fictional = any(indicator in answer_lower for indicator in fictional_indicators)
            
            # Check 5: Apakah jawaban mengakui keterbatasan data?
            limitation_indicators = ["tidak tersedia", "tidak ditemukan", "data terbatas", "informasi tidak lengkap"]
            acknowledges_limitation = any(indicator in answer_lower for indicator in limitation_indicators)
            
            # Skoring
            score = 0
            
            if has_source:
                score += 0.4  # 40% untuk memiliki sumber
            
            if has_fda_ref:
                score += 0.2  # 20% untuk menyebut FDA
            
            if has_disclaimer:
                score += 0.1  # 10% untuk disclaimer medis
            
            if not has_fictional:
                score += 0.2  # 20% untuk tidak membuat klaim fiktif
            
            if acknowledges_limitation:
                score += 0.1  # 10% untuk mengakui keterbatasan data
            
            faithful_scores.append(min(score, 1.0))  # Maksimal 1.0
        
        return np.mean(faithful_scores) if faithful_scores else 0
    
    # ===========================================
    # METRIK 3: ANSWER RELEVANCY
    # ===========================================
    def calculate_answer_relevancy(self):
        """Hitung relevansi jawaban terhadap pertanyaan"""
        relevancy_scores = []
        
        for test in self.test_set:
            answer, _ = self.assistant.ask_question(test["question"])
            answer_lower = answer.lower()
            
            # Dapatkan keywords berdasarkan question type
            q_type = test["question_type"]
            keywords = self.question_type_keywords.get(q_type, [])
            
            # Tambahkan drug name sebagai keyword
            drug_name = test["expected_drug"]
            keywords.append(drug_name)
            
            # Hitung keyword matches
            matches = 0
            for keyword in keywords:
                if keyword in answer_lower:
                    matches += 1
            
            # Normalize score
            total_keywords = len(keywords)
            score = matches / total_keywords if total_keywords > 0 else 0
            
            # Bonus jika jawaban langsung menjawab pertanyaan
            question_words = test["question"].lower().split()
            question_keywords = [word for word in question_words if len(word) > 3]
            question_matches = sum(1 for word in question_keywords if word in answer_lower)
            
            if len(question_keywords) > 0:
                question_score = question_matches / len(question_keywords)
                score = (score * 0.7) + (question_score * 0.3)  # Weighted average
            
            # Penalty jika jawaban terlalu pendek atau generic
            if len(answer) < 50 and "tidak tersedia" not in answer_lower:
                score *= 0.8  # Penalty 20%
            
            relevancy_scores.append(min(score, 1.0))
        
        return np.mean(relevancy_scores) if relevancy_scores else 0
    
    # ===========================================
    # METRIK 4: SEMANTIC SIMILARITY
    # ===========================================
    def calculate_semantic_similarity(self):
        """Hitung kesamaan semantik dengan jawaban ideal"""
        similarity_scores = []
        
        for test in self.test_set:
            answer, _ = self.assistant.ask_question(test["question"])
            
            # Generate expected answer template berdasarkan question type
            expected_answer = self._generate_expected_answer(test)
            
            if expected_answer:
                # Hitung similarity
                score = self._calculate_text_similarity(answer, expected_answer)
                similarity_scores.append(score)
        
        return np.mean(similarity_scores) if similarity_scores else 0
    
    def _generate_expected_answer(self, test):
        """Generate template jawaban ideal berdasarkan tipe pertanyaan"""
        drug = test["expected_drug"].title()
        q_type = test["question_type"]
        
        templates = {
            "dosis": f"Informasi dosis {drug} tersedia dalam data FDA. Dosis yang tepat tergantung kondisi pasien dan harus ditentukan oleh tenaga medis.",
            "efek_samping": f"Efek samping {drug} tercatat dalam database FDA. Pasien disarankan melaporkan efek samping yang tidak diinginkan.",
            "indikasi": f"{drug} diindikasikan untuk kondisi tertentu sesuai data FDA. Penggunaan harus sesuai resep dokter.",
            "kontraindikasi": f"Kontraindikasi {drug} tercantum dalam informasi FDA. Pasien dengan kondisi tertentu tidak boleh menggunakan obat ini.",
            "interaksi": f"Interaksi {drug} dengan obat lain terdapat dalam data FDA. Konsultasi dengan apoteker diperlukan sebelum penggunaan bersamaan.",
            "peringatan": f"Peringatan penggunaan {drug} tersedia dalam informasi FDA. Bacalah seluruh informasi sebelum penggunaan."
        }
        
        return templates.get(q_type, f"Informasi tentang {drug} tersedia dalam database FDA.")
    
    def _calculate_text_similarity(self, text1, text2):
        """Hitung similarity antara dua teks dengan metode hybrid"""
        # Preprocess text
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # 1. Jaccard Similarity pada kata-kata
        words1 = set(re.findall(r'\w+', text1_lower))
        words2 = set(re.findall(r'\w+', text2_lower))
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_score = len(intersection) / len(union) if union else 0
        
        # 2. Bonus untuk kata-kata penting yang match
        important_matches = sum(1 for word in self.important_words 
                              if word in words1 and word in words2)
        
        important_bonus = min(important_matches * 0.05, 0.3)  # Maksimal bonus 30%
        
        # 3. Penalty jika banyak kata yang tidak relevan
        irrelevant_words = ["hello", "hi", "terima kasih", "silahkan", "bisa", "membantu"]
        irrelevant_count = sum(1 for word in irrelevant_words if word in text1_lower)
        irrelevant_penalty = min(irrelevant_count * 0.02, 0.1)  # Maksimal penalty 10%
        
        # Final score
        final_score = jaccard_score + important_bonus - irrelevant_penalty
        
        return max(0, min(final_score, 1.0))
    
    # ===========================================
    # METODE UTAMA EVALUASI
    # ===========================================
    def run_complete_evaluation(self):
        """Jalankan semua evaluasi dan return hasil"""
        try:
            # Hitung semua metrik
            mrr_score = self.calculate_mrr()
            faithfulness_score = self.calculate_faithfulness()
            relevancy_score = self.calculate_answer_relevancy()
            similarity_score = self.calculate_semantic_similarity()
            
            # Compile results
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_set),
                "MRR": float(mrr_score),
                "Faithfulness": float(faithfulness_score),
                "Answer_Relevancy": float(relevancy_score),
                "Semantic_Similarity": float(similarity_score)
            }
            
            # Hitung overall score (weighted average)
            weights = {
                "MRR": 0.3,           # 30% - Paling penting untuk drug detection
                "Faithfulness": 0.4,  # 40% - Sangat penting untuk aplikasi medis
                "Answer_Relevancy": 0.2,  # 20% - Penting untuk user experience
                "Semantic_Similarity": 0.1   # 10% - Tambahan untuk kualitas jawaban
            }
            
            weighted_sum = (
                results["MRR"] * weights["MRR"] +
                results["Faithfulness"] * weights["Faithfulness"] +
                results["Answer_Relevancy"] * weights["Answer_Relevancy"] +
                results["Semantic_Similarity"] * weights["Semantic_Similarity"]
            )
            
            results["Overall_Score"] = float(weighted_sum)
            
            # Simpan detail test case results
            results["test_case_details"] = self._get_test_case_details()
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error dalam evaluasi: {str(e)}")
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "MRR": 0,
                "Faithfulness": 0,
                "Answer_Relevancy": 0,
                "Semantic_Similarity": 0,
                "Overall_Score": 0
            }
    
    def _get_test_case_details(self):
        """Ambil detail hasil untuk setiap test case"""
        details = []
        
        for test in self.test_set:
            answer, sources = self.assistant.ask_question(test["question"])
            
            # Deteksi drug
            detected_drugs = self.assistant.drug_detector.detect_drug_from_query(test["question"])
            detected_drug_names = [drug['drug_name'] for drug in detected_drugs]
            
            detail = {
                "test_id": test["id"],
                "question": test["question"],
                "expected_drug": test["expected_drug"],
                "detected_drugs": detected_drug_names,
                "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
                "has_sources": bool(sources),
                "source_count": len(sources) if sources else 0
            }
            
            details.append(detail)
        
        return details

# ===========================================
# KELAS-KELAS EXISTING (TIDAK BERUBAH)
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
            'salbutamol': ['albuterol', 'salbutamol', 'ventolin', 'salbu', 'asmasolon']
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
            8. SELALU sebutkan bahwa informasi berasal dari FDA

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
# FUNGSI UTAMA DENGAN PERBAIKAN
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
    # HALAMAN EVALUASI RAG (DIPERBAIKI)
    # ===========================================
    elif page == "📊 Evaluasi RAG System":
        st.title("📊 Evaluasi RAG System - 4 Metrik")
        st.markdown("Evaluasi performa sistem menggunakan 4 metrik standar RAG")
        
        # Informasi evaluasi
        with st.expander("ℹ️ Tentang 4 Metrik Evaluasi", expanded=True):
            st.markdown("""
            <div class="evaluation-info">
            ### **📊 Metrik Evaluasi RAG**
            
            **1. Mean Reciprocal Rank (MRR)**
            - **Apa**: Mengukur akurasi sistem dalam menemukan obat yang benar dari query
            - **Target**: > 0.8 (80%)
            - **Bobot**: 30% dalam overall score
            
            **2. Faithfulness**
            - **Apa**: Mengukur kesetiaan jawaban terhadap data sumber (FDA)
            - **Target**: > 0.85 (85%)
            - **Bobot**: 40% dalam overall score (paling penting untuk aplikasi medis)
            
            **3. Answer Relevancy**
            - **Apa**: Mengukur relevansi jawaban terhadap pertanyaan spesifik
            - **Target**: > 0.7 (70%)
            - **Bobot**: 20% dalam overall score
            
            **4. Semantic Similarity**
            - **Apa**: Mengukur kesamaan makna dengan jawaban ideal
            - **Target**: > 0.75 (75%)
            - **Bobot**: 10% dalam overall score
            </div>
            """, unsafe_allow_html=True)
        
        # Tombol untuk menjalankan evaluasi
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Jalankan Evaluasi Komprehensif", use_container_width=True, type="primary"):
                with st.spinner("Menjalankan evaluasi pada 10 test cases..."):
                    # Initialize evaluator
                    st.session_state.evaluator = RAGEvaluator(assistant)
                    
                    # Jalankan evaluasi
                    results = st.session_state.evaluator.run_complete_evaluation()
                    st.session_state.evaluation_results = results
                    
                    # Tampilkan notifikasi sukses
                    st.success("✅ Evaluasi selesai! Scroll ke bawah untuk melihat hasil.")
                    st.rerun()
        
        with col2:
            if st.button("📥 Simpan Hasil ke JSON", use_container_width=True):
                if st.session_state.evaluation_results:
                    # Generate filename dengan timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rag_evaluation_{timestamp}.json"
                    
                    # Simpan ke file
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.evaluation_results, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"✅ Hasil disimpan ke `{filename}`")
                    
                    # Tawarkan download
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = f.read()
                    
                    st.download_button(
                        label="⬇️ Download File JSON",
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
        
        # Garis pemisah
        st.markdown("---")
        
        # Tampilkan hasil evaluasi jika ada
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            st.markdown(f"### 📈 Hasil Evaluasi ({results['timestamp']})")
            st.markdown(f"**Test Cases:** {results['total_test_cases']} pertanyaan • **Status:** ✅ Selesai")
            
            # Tampilkan metrik dalam 4 kolom
            col1, col2, col3, col4 = st.columns(4)
            
            # Helper function untuk menentukan warna score
            def get_score_color(score, target):
                if score >= target:
                    return "good-score"
                elif score >= target * 0.8:  # 80% dari target
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
                    <div>Mean Reciprocal Rank</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"**Target:** >0.800 | **Baseline:** 0.930 (Samudra dkk.)")
            
            with col2:
                faithfulness = results["Faithfulness"]
                color_class = get_score_color(faithfulness, 0.85)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Faithfulness</div>
                    <div class="metric-value {color_class}">{faithfulness:.3f}</div>
                    <div>Kesetiaan ke Sumber</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"**Target:** >0.850 | **Baseline:** 0.620 (Samudra dkk.)")
            
            with col3:
                relevancy = results["Answer_Relevancy"]
                color_class = get_score_color(relevancy, 0.7)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Answer Relevancy</div>
                    <div class="metric-value {color_class}">{relevancy:.3f}</div>
                    <div>Relevansi Jawaban</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"**Target:** >0.700 | **Baseline:** 0.570 (Samudra dkk.)")
            
            with col4:
                similarity = results["Semantic_Similarity"]
                color_class = get_score_color(similarity, 0.75)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Semantic Similarity</div>
                    <div class="metric-value {color_class}">{similarity:.3f}</div>
                    <div>Kesamaan Semantik</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"**Target:** >0.750 | **Baseline:** 0.810 (Samudra dkk.)")
            
            # Overall Score
            st.markdown("---")
            overall = results["Overall_Score"]
            
            col_overall1, col_overall2 = st.columns([1, 3])
            
            with col_overall1:
                # Determine overall color
                if overall >= 0.8:
                    overall_color = "#4CAF50"  # Green
                    overall_status = "Baik"
                elif overall >= 0.6:
                    overall_color = "#FF9800"  # Orange
                    overall_status = "Cukup"
                else:
                    overall_color = "#F44336"  # Red
                    overall_status = "Perlu Perbaikan"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background: #f5f5f5; border: 2px solid {overall_color};">
                    <div style="font-size: 0.9em; color: #666;">Overall Score</div>
                    <div style="font-size: 2.5em; font-weight: bold; color: {overall_color};">
                        {overall:.3f}
                    </div>
                    <div style="font-size: 0.9em; color: {overall_color}; font-weight: bold;">{overall_status}</div>
                    <div style="font-size: 0.8em; color: #666;">Weighted Average</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_overall2:
                # Progress bars untuk setiap metrik
                st.markdown("**Detail Skor:**")
                
                metrics_data = [
                    ("MRR", results["MRR"], 0.8),
                    ("Faithfulness", results["Faithfulness"], 0.85),
                    ("Answer Relevancy", results["Answer_Relevancy"], 0.7),
                    ("Semantic Similarity", results["Semantic_Similarity"], 0.75)
                ]
                
                for metric_name, score, target in metrics_data:
                    # Tampilkan progress bar dengan target line
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        # Progress bar dengan warna berdasarkan performa
                        if score >= target:
                            bar_color = "green"
                        elif score >= target * 0.8:
                            bar_color = "orange"
                        else:
                            bar_color = "red"
                        
                        st.progress(
                            float(score), 
                            text=f"{metric_name}: {score:.3f} / {target}"
                        )
                    
                    with col_b:
                        # Persentase dari target
                        percentage = (score / target * 100) if target > 0 else 0
                        st.metric(label="% Target", value=f"{percentage:.1f}%")
            
            # Detail hasil dalam expander
            with st.expander("📋 Detail Hasil Evaluasi Lengkap"):
                # Tampilkan JSON results
                st.markdown("**Data Hasil (JSON):**")
                st.json(results)
                
                # Tampilkan test cases
                st.markdown("### 🧪 Test Cases yang Digunakan")
                
                # PERBAIKAN: Gunakan evaluator dari session state
                if st.session_state.evaluator:
                    evaluator = st.session_state.evaluator
                    
                    # Buat dataframe test cases
                    test_df = pd.DataFrame([
                        {
                            "No": test["id"],
                            "Pertanyaan": test["question"],
                            "Obat yang Diharapkan": test["expected_drug"],
                            "Tipe Pertanyaan": test["question_type"]
                        }
                        for test in evaluator.test_set
                    ])
                    
                    st.dataframe(test_df, use_container_width=True, hide_index=True)
                    
                    # Tampilkan contoh jawaban
                    st.markdown("### 🤖 Contoh Jawaban Sistem")
                    
                    # Pilih 3 test cases secara acak
                    import random
                    sample_tests = random.sample(evaluator.test_set, min(3, len(evaluator.test_set)))
                    
                    for i, test in enumerate(sample_tests):
                        with st.spinner(f"Mengambil jawaban untuk: '{test['question']}'..."):
                            answer, sources = assistant.ask_question(test["question"])
                            
                            with st.container():
                                st.markdown(f"**Test Case {test['id']}:** `{test['question']}`")
                                
                                # Tampilkan jawaban dengan formatting
                                st.markdown("**Jawaban Sistem:**")
                                st.info(answer)
                                
                                # Tampilkan informasi sumber
                                if sources:
                                    source_names = [s['nama'] for s in sources]
                                    st.markdown(f"**Sumber FDA:** {', '.join(source_names)}")
                                else:
                                    st.warning("Tidak ada sumber FDA ditemukan")
                                
                                st.markdown("---")
                else:
                    st.warning("Evaluator tidak tersedia. Jalankan evaluasi terlebih dahulu.")
        
        else:
            # Tampilkan informasi sebelum evaluasi
            st.info("""
            **📝 Informasi Evaluasi:**
            
            Sistem akan dievaluasi menggunakan **10 test cases** yang meliputi:
            - Pertanyaan tentang dosis obat
            - Pertanyaan tentang efek samping
            - Pertanyaan tentang indikasi penggunaan
            - Pertanyaan tentang kontraindikasi
            - Pertanyaan tentang interaksi obat
            
            **Klik tombol 'Jalankan Evaluasi Komprehensif' untuk memulai.**
            """)
            
            # Preview test cases
            st.markdown("### 🧪 Preview Test Cases")
            
            # Buat evaluator sementara untuk preview
            temp_evaluator = RAGEvaluator(assistant)
            
            preview_df = pd.DataFrame([
                {
                    "No": test["id"],
                    "Pertanyaan": test["question"],
                    "Obat": test["expected_drug"],
                    "Tipe": test["question_type"]
                }
                for test in temp_evaluator.test_set[:5]  # Hanya preview 5 pertama
            ])
            
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            st.caption(f"Total: {len(temp_evaluator.test_set)} test cases")

    # Footer (tampil di semua halaman)
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "💊 **Sistem Tanya Jawab Obat** • Data dari FDA API • Terjemahan Gemini AI • Evaluasi RAG 4 Metrik"
        "</div>", 
        unsafe_allow_html=True
    )

# Panggil main function
if __name__ == "__main__":
    main()
