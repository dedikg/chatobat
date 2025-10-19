import google.generativeai as genai
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

# Setup Gemini API
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_available = True

class EnhancedPharmaAssistant:
    def __init__(self):
        self.drugs_db = self._initialize_drug_database()
        self.conversation_history = []
        
    def _initialize_drug_database(self):
        """Initialize expanded drug database dengan gejala yang benar"""
        return {
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
    
    def _calculate_similarity_score(self, query, drug_info):
        """Enhanced semantic similarity scoring dengan symptom mapping"""
        query = query.lower()
        score = 0
        
        # 1. Direct symptom matching (HIGHEST PRIORITY)
        if 'gejala' in drug_info:
            symptoms = drug_info['gejala'].lower().split(',')
            for symptom in symptoms:
                symptom_clean = symptom.strip()
                if symptom_clean and symptom_clean in query:
                    score += 5
        
        # 2. Direct drug name match
        if drug_info['nama'].lower() in query:
            score += 5
            
        # 3. Indication keyword matching
        indication_lower = drug_info['indikasi'].lower()
        indication_keywords = [kw.strip() for kw in indication_lower.split(',')]
        for keyword in indication_keywords:
            if keyword and keyword in query:
                score += 3
        
        # 4. Brand name match
        for merek in drug_info['merek_dagang'].lower().split(','):
            merek_clean = merek.strip()
            if merek_clean and merek_clean in query:
                score += 3
        
        return score
    
    def _fallback_symptom_search(self, query, top_k=2):
        """Fallback search untuk query gejala"""
        symptom_drug_mapping = {
            'sakit kepala': ['paracetamol', 'ibuprofen'],
            'pusing': ['paracetamol'],
            'demam': ['paracetamol', 'ibuprofen'],
            'panas': ['paracetamol'],
            'pilek': ['loratadine'],
            'alergi': ['loratadine'],
            'maag': ['omeprazole'],
            'kolesterol': ['simvastatin'],
            'nyeri': ['paracetamol', 'ibuprofen']
        }
        
        results = []
        query_lower = query.lower()
        
        for symptom, drug_ids in symptom_drug_mapping.items():
            if symptom in query_lower:
                for drug_id in drug_ids:
                    if drug_id in self.drugs_db:
                        results.append({
                            'score': 4,
                            'drug_info': self.drugs_db[drug_id],
                            'drug_id': drug_id
                        })
        
        return results[:top_k]
    
    def semantic_search(self, query, top_k=3):
        """Enhanced semantic search dengan symptom understanding"""
        results = []
        
        for drug_id, drug_info in self.drugs_db.items():
            score = self._calculate_similarity_score(query, drug_info)
            
            if score > 0:
                results.append({
                    'score': score,
                    'drug_info': drug_info,
                    'drug_id': drug_id
                })
        
        # FALLBACK: Jika tidak ada hasil, cari berdasarkan gejala
        if not results:
            results = self._fallback_symptom_search(query, top_k)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return [result['drug_info'] for result in results[:top_k]]
    
    def ask_question(self, question):
        """Enhanced RAG dengan Gemini"""
        # Semantic search for relevant drugs
        relevant_drugs = self.semantic_search(question)
        
        if not relevant_drugs:
            available_drugs = ", ".join([drug['nama'] for drug in self.drugs_db.values()])
            return f"Maaf, tidak ditemukan informasi tentang obat tersebut dalam database kami. Coba tanyakan tentang: {available_drugs}.", []
        
        # Prepare enhanced context for Gemini
        context = "INFORMASI OBAT YANG RELEVAN:\n"
        for i, drug in enumerate(relevant_drugs, 1):
            context += f"""
            OBAT {i}:
            - NAMA: {drug['nama']}
            - MEREK: {drug['merek_dagang']}
            - GOLONGAN: {drug['golongan']}
            - INDIKASI: {drug['indikasi']}
            - DOSIS DEWASA: {drug['dosis_dewasa']}
            - DOSIS ANAK: {drug['dosis_anak']}
            - EFEK SAMPING: {drug['efek_samping']}
            - KONTRAINDIKASI: {drug['kontraindikasi']}
            - INTERAKSI: {drug['interaksi']}
            - GEJALA: {drug.get('gejala', 'Tidak tersedia')}
            """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Anda adalah asisten farmasi BPJS Kesehatan yang profesional.
            
            INFORMASI OBAT YANG TERSEDIA:
            {context}
            
            PERTANYAAN PASIEN: {question}
            
            INSTRUKSI:
            1. Jawab pertanyaan dengan AKURAT berdasarkan informasi obat di atas
            2. Gunakan bahasa Indonesia yang JELAS dan mudah dipahami
            3. Jika informasi tidak tersedia, jangan membuat-buat jawaban
            4. Sertakan nama obat yang relevan dalam jawaban
            5. Tetap singkat namun informatif
            
            JAWABAN:
            """
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=800,
                    top_p=0.8
                )
            )
            return response.text, relevant_drugs
            
        except Exception as e:
            print(f"⚠️ Gemini API Error: {e}")
            # Fallback to manual answer
            return self._generate_manual_answer(question, relevant_drugs), relevant_drugs
    
    def _generate_manual_answer(self, question, drugs):
        """Manual answer fallback"""
        answer_parts = [f"**Untuk: '{question}'**"]
        
        for drug in drugs:
            answer_parts.append(f"💊 **{drug['nama']}**")
            answer_parts.append(f"• Indikasi: {drug['indikasi']}")
            answer_parts.append(f"• Dosis: {drug['dosis_dewasa']}")
        
        return "\n".join(answer_parts)
