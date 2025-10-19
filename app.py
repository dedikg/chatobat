import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from datetime import datetime
import hashlib
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from typing import List, Dict, Tuple
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem RAG Tanya Jawab Informasi Obat",
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

class TrueRAGPharmaAssistant:
    def __init__(self):
        self.drugs_db = self._initialize_drug_database()
        self.current_context = {}
        self.embedding_model = None
        self.vector_database = self._build_vector_database()
        
    def _initialize_drug_database(self):
        """Initialize comprehensive drug database"""
        # Expanded database dengan informasi lebih detail
        drugs_db = {
            "paracetamol": {
                "nama": "Paracetamol",
                "golongan": "Analgesik dan Antipiretik",
                "indikasi": "Demam, nyeri ringan hingga sedang, sakit kepala, sakit gigi, nyeri otot, nyeri haid, migrain",
                "dosis_dewasa": "500-1000 mg setiap 4-6 jam, maksimal 4000 mg/hari",
                "dosis_anak": "10-15 mg/kgBB setiap 4-6 jam",
                "efek_samping": "Gangguan pencernaan, ruam kulit (jarang), hepatotoksik pada overdosis",
                "kontraindikasi": "Gangguan hati berat, hipersensitif, penyakit ginjal berat",
                "interaksi": "Alcohol meningkatkan risiko kerusakan hati, warfarin meningkatkan risiko perdarahan",
                "merek_dagang": "Panadol, Sanmol, Tempra, Biogesic",
                "kategori": "analgesik, antipiretik, nyeri, demam",
                "gejala": "sakit kepala, demam, nyeri, sakit gigi, nyeri haid, pusing, panas, migrain",
                "peringatan": "Jangan melebihi dosis maksimal, hindari alkohol, hati-hati pada pasien gangguan hati"
            },
            "amoxicillin": {
                "nama": "Amoxicillin",
                "golongan": "Antibiotik Beta-Laktam", 
                "indikasi": "Infeksi bakteri saluran napas, telinga, kulit, saluran kemih, radang tenggorokan, sinusitis, pneumonia",
                "dosis_dewasa": "250-500 mg setiap 8 jam atau 875 mg setiap 12 jam",
                "dosis_anak": "20-50 mg/kgBB/hari dibagi 3 dosis",
                "efek_samping": "Diare, mual, ruam kulit, reaksi alergi, kandidiasis",
                "kontraindikasi": "Alergi penisilin, mononukleosis infeksiosa, riwayat penyakit hati",
                "interaksi": "Mengurangi efektivitas kontrasepsi oral, probenesid meningkatkan kadar amoxicillin",
                "merek_dagang": "Amoxan, Kalmoxillin, Moxigra, Amoxiclav",
                "kategori": "antibiotik, infeksi, bakteri",
                "gejala": "infeksi, radang, demam karena infeksi, batuk berdahak, radang tenggorokan, sinusitis",
                "peringatan": "Lengkapi seluruh regimen antibiotik, perhatikan reaksi alergi"
            },
            # ... (tambahkan lebih banyak obat)
        }
        return drugs_db
    
    def _load_embedding_model(self):
        """Load sentence transformer model untuk embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            # Fallback ke model sederhana
            self.embedding_model = None
    
    def _build_vector_database(self):
        """Build vector database dari drug information"""
        vector_db = []
        
        for drug_id, drug_info in self.drugs_db.items():
            # Create multiple chunks dari setiap obat untuk retrieval yang lebih baik
            chunks = self._chunk_drug_information(drug_info)
            
            for chunk in chunks:
                vector_entry = {
                    'drug_id': drug_id,
                    'drug_name': drug_info['nama'],
                    'content': chunk['content'],
                    'content_type': chunk['type'],
                    'embedding': None,  # Will be computed on demand
                    'metadata': {
                        'category': drug_info['kategori'],
                        'symptoms': drug_info['gejala']
                    }
                }
                vector_db.append(vector_entry)
        
        return vector_db
    
    def _chunk_drug_information(self, drug_info):
        """Split drug information into retrievable chunks"""
        chunks = []
        
        # Chunk untuk indikasi
        chunks.append({
            'type': 'indikasi',
            'content': f"{drug_info['nama']} digunakan untuk: {drug_info['indikasi']}"
        })
        
        # Chunk untuk dosis
        chunks.append({
            'type': 'dosis',
            'content': f"Dosis {drug_info['nama']}: Dewasa - {drug_info['dosis_dewasa']}, Anak - {drug_info['dosis_anak']}"
        })
        
        # Chunk untuk efek samping
        chunks.append({
            'type': 'efek_samping',
            'content': f"Efek samping {drug_info['nama']}: {drug_info['efek_samping']}"
        })
        
        # Chunk untuk kontraindikasi
        chunks.append({
            'type': 'kontraindikasi',
            'content': f"Kontraindikasi {drug_info['nama']}: {drug_info['kontraindikasi']}"
        })
        
        # Chunk untuk interaksi
        chunks.append({
            'type': 'interaksi',
            'content': f"Interaksi {drug_info['nama']}: {drug_info['interaksi']}"
        })
        
        # Chunk untuk peringatan
        if 'peringatan' in drug_info:
            chunks.append({
                'type': 'peringatan',
                'content': f"Peringatan {drug_info['nama']}: {drug_info['peringatan']}"
            })
        
        return chunks
    
    def _compute_embeddings(self, text):
        """Compute embeddings untuk text"""
        if self.embedding_model is None:
            self._load_embedding_model()
        
        if self.embedding_model:
            return self.embedding_model.encode([text])[0]
        else:
            # Fallback ke TF-IDF like scoring
            return self._simple_text_vector(text)
    
    def _simple_text_vector(self, text):
        """Simple text vectorization fallback"""
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """True semantic search dengan vector similarity"""
        query_embedding = self._compute_embeddings(query)
        
        results = []
        
        for doc in self.vector_database:
            if isinstance(query_embedding, dict):
                # Fallback similarity calculation
                score = self._calculate_simple_similarity(query, doc['content'])
            else:
                doc_embedding = self._compute_embeddings(doc['content'])
                score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            
            if score > 0.1:  # Threshold similarity
                results.append({
                    'score': score,
                    'drug_id': doc['drug_id'],
                    'drug_name': doc['drug_name'],
                    'content': doc['content'],
                    'content_type': doc['content_type'],
                    'metadata': doc['metadata']
                })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _calculate_simple_similarity(self, query: str, document: str) -> float:
        """Simple similarity calculation fallback"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words or not doc_words:
            return 0.0
        
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def retrieve_information(self, query: str) -> Tuple[str, List[Dict]]:
        """Retrieve relevant information menggunakan RAG"""
        # Step 1: Semantic search untuk menemukan informasi relevan
        retrieved_docs = self._semantic_search(query, top_k=3)
        
        if not retrieved_docs:
            return "Maaf, tidak menemukan informasi yang relevan dalam database.", []
        
        # Step 2: Group by drug dan pilih yang paling relevan
        drug_scores = {}
        for doc in retrieved_docs:
            drug_id = doc['drug_id']
            if drug_id not in drug_scores:
                drug_scores[drug_id] = 0
            drug_scores[drug_id] += doc['score']
        
        # Pilih drug dengan score tertinggi
        best_drug_id = max(drug_scores, key=drug_scores.get)
        best_drug_info = self.drugs_db[best_drug_id]
        
        # Step 3: Prepare context untuk generator
        context = self._prepare_rag_context(retrieved_docs, best_drug_info)
        
        # Step 4: Generate response menggunakan RAG
        response = self._generate_rag_response(query, context, best_drug_info)
        
        return response, [best_drug_info]
    
    def _prepare_rag_context(self, retrieved_docs: List[Dict], best_drug_info: Dict) -> str:
        """Prepare context dari retrieved documents"""
        context = "INFORMASI OBAT YANG RELEVAN:\n\n"
        
        # Group documents by type untuk organisasi yang lebih baik
        doc_types = {}
        for doc in retrieved_docs:
            doc_type = doc['content_type']
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        # Organize context by document type
        for doc_type in ['indikasi', 'dosis', 'efek_samping', 'kontraindikasi', 'interaksi', 'peringatan']:
            if doc_type in doc_types:
                context += f"{doc_type.upper()}:\n"
                for doc in doc_types[doc_type]:
                    context += f"- {doc['content']}\n"
                context += "\n"
        
        # Tambahkan informasi umum obat
        context += f"INFORMASI UMUM:\n"
        context += f"- Nama: {best_drug_info['nama']}\n"
        context += f"- Golongan: {best_drug_info['golongan']}\n"
        context += f"- Merek Dagang: {best_drug_info['merek_dagang']}\n"
        
        return context
    
    def _generate_rag_response(self, query: str, context: str, drug_info: Dict) -> str:
        """Generate response menggunakan RAG pattern"""
        if not gemini_available:
            return self._generate_fallback_response(query, drug_info)
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Anda adalah asisten farmasi yang profesional. Gunakan informasi berikut untuk menjawab pertanyaan.

            {context}

            PERTANYAAN: {query}

            INSTRUKSI RAG:
            1. JAWAB BERDASARKAN INFORMASI YANG DIBERIKAN di atas
            2. Jangan membuat informasi yang tidak ada dalam konteks
            3. Jika informasi tidak cukup, katakan bahwa informasi terbatas
            4. Fokus pada obat yang paling relevan
            5. Gunakan bahasa Indonesia yang jelas dan profesional
            6. Sertakan peringatan penting jika ada

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
            return response.text
            
        except Exception as e:
            st.error(f"⚠️ Gemini API Error: {e}")
            return self._generate_fallback_response(query, drug_info)
    
    def _generate_fallback_response(self, query: str, drug_info: Dict) -> str:
        """Fallback response tanpa AI"""
        query_lower = query.lower()
        response_parts = [f"💊 **{drug_info['nama']}**"]
        
        if any(keyword in query_lower for keyword in ['dosis', 'berapa']):
            response_parts.append(f"**Dosis Dewasa:** {drug_info['dosis_dewasa']}")
            response_parts.append(f"**Dosis Anak:** {drug_info['dosis_anak']}")
        
        elif any(keyword in query_lower for keyword in ['efek', 'samping']):
            response_parts.append(f"**Efek Samping:** {drug_info['efek_samping']}")
        
        elif any(keyword in query_lower for keyword in ['kontra']):
            response_parts.append(f"**Kontraindikasi:** {drug_info['kontraindikasi']}")
        
        elif any(keyword in query_lower for keyword in ['interaksi']):
            response_parts.append(f"**Interaksi:** {drug_info['interaksi']}")
        
        elif any(keyword in query_lower for keyword in ['untuk apa', 'kegunaan', 'indikasi']):
            response_parts.append(f"**Indikasi:** {drug_info['indikasi']}")
        
        else:
            # Default comprehensive response
            response_parts.extend([
                f"**Indikasi:** {drug_info['indikasi']}",
                f"**Dosis Dewasa:** {drug_info['dosis_dewasa']}",
                f"**Efek Samping:** {drug_info['efek_samping']}"
            ])
        
        return "\n\n".join(response_parts)
    
    def ask_question(self, question: str) -> Tuple[str, List[Dict]]:
        """Main interface untuk bertanya dengan RAG"""
        # Update context management
        answer, sources = self.retrieve_information(question)
        self._update_conversation_context(question, answer, sources)
        
        return answer, sources
    
    def _update_conversation_context(self, question, answer, sources):
        """Simple context management"""
        if sources:
            self.current_context = {
                'current_drug': sources[0]['nama'],
                'timestamp': datetime.now()
            }

# Initialize true RAG assistant
@st.cache_resource
def load_rag_assistant():
    return TrueRAGPharmaAssistant()

assistant = load_rag_assistant()
