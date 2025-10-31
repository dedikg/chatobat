import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime
import time
import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any

# [KONFIGURASI DAN CLASS YANG SAMA SEBELUMNYA...]
# ... (FDADrugAPI, TranslationService, EnhancedDrugDetector, SimpleRAGPharmaAssistant)

class FDADrugAPI:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
    
    def get_drug_info(self, generic_name: str):
        """Ambil data obat langsung dari FDA API dan return raw data juga"""
        params = {
            'search': f'openfda.generic_name:"{generic_name}"',
            'limit': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    parsed_data = self._parse_fda_data(data['results'][0], generic_name)
                    return {
                        'parsed': parsed_data,
                        'raw': data['results'][0],  # Raw FDA data
                        'raw_full': data,           # Full API response
                        'api_url': response.url     # API endpoint yang dipanggil
                    }
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

class SimpleRAGPharmaAssistant:
    def __init__(self):
        self.fda_api = FDADrugAPI()
        self.translator = TranslationService()
        self.drug_detector = EnhancedDrugDetector()
        self.drugs_cache = {}
        self.current_context = {}
        self.last_raw_data = {}  # Store raw data untuk ditampilkan
        
    def _get_or_fetch_drug_info(self, drug_name: str):
        """Dapatkan data dari cache atau fetch dari FDA API dengan nama FDA yang benar"""
        drug_key = drug_name.lower()
        
        if drug_key in self.drugs_cache:
            return self.drugs_cache[drug_key]
        
        # Dapatkan nama FDA yang sebenarnya
        fda_name = self.drug_detector.get_fda_name(drug_name)
        
        # Fetch dari FDA API dengan nama FDA
        api_result = self.fda_api.get_drug_info(fda_name)
        
        if api_result:
            drug_info = api_result['parsed']
            
            # Update nama ke nama yang familiar untuk user
            if drug_name != fda_name:
                drug_info['nama'] = drug_name.title()
                drug_info['catatan'] = f"Di FDA dikenal sebagai {fda_name}"
            
            # Translate fields yang penting
            drug_info = self._translate_drug_info(drug_info)
            
            # Simpan raw data untuk ditampilkan
            self.drugs_cache[drug_key] = {
                'parsed': drug_info,
                'raw': api_result['raw'],
                'raw_full': api_result['raw_full'],
                'api_url': api_result['api_url'],
                'fda_name_used': fda_name
            }
        
        return self.drugs_cache.get(drug_key)
    
    def get_last_raw_data(self):
        """Get raw data dari pencarian terakhir"""
        return self.last_raw_data
    
    def ask_question(self, question):
        """Main RAG interface dengan FDA API"""
        try:
            # Reset last raw data
            self.last_raw_data = {}
            
            # Step 1: Retrieve relevant information dari FDA API
            retrieved_results = self._rag_retrieve(question)
            
            if not retrieved_results:
                available_drugs = ", ".join(self.drug_detector.get_all_available_drugs()[:10])
                return f"❌ Tidak ditemukan informasi yang relevan dalam database FDA untuk pertanyaan Anda.\n\n💡 **Coba tanyakan tentang:** {available_drugs}", []
            
            # Step 2: Build context dari data FDA
            rag_context = self._build_rag_context(retrieved_results)
            
            # Step 3: Generate response dengan RAG
            answer = self._generate_rag_response(question, rag_context)
            
            # Step 4: Get sources dan raw data
            sources = []
            raw_data_collection = []
            seen_drug_names = set()
            
            for result in retrieved_results:
                drug_name = result['drug_info']['parsed']['nama']
                if drug_name not in seen_drug_names:
                    sources.append(result['drug_info']['parsed'])
                    # Collect raw data untuk ditampilkan
                    raw_data_collection.append({
                        'drug_name': drug_name,
                        'raw_data': result['drug_info']['raw'],
                        'api_url': result['drug_info']['api_url'],
                        'fda_name_used': result['drug_info']['fda_name_used']
                    })
                    seen_drug_names.add(drug_name)
            
            # Simpan raw data terakhir
            self.last_raw_data = raw_data_collection
            
            # Update context
            self._update_conversation_context(question, answer, sources)
            
            return answer, sources
            
        except Exception as e:
            return "Maaf, terjadi error dalam sistem. Silakan coba lagi.", []
    
    # [METHODS LAINNYA YANG SAMA...]

def display_raw_fda_data(raw_data_collection):
    """Tampilkan raw FDA data dalam format JSON/XML"""
    if not raw_data_collection:
        return
    
    st.markdown("---")
    st.subheader("🔧 **Raw FDA API Data**")
    
    for i, raw_data in enumerate(raw_data_collection):
        with st.expander(f"📄 Raw Data untuk {raw_data['drug_name']} (FDA: {raw_data['fda_name_used']})"):
            
            # Tampilkan API URL yang dipanggil
            st.markdown(f"**API Endpoint:** `{raw_data['api_url']}`")
            
            # Tab untuk JSON dan XML
            tab1, tab2 = st.tabs(["📋 JSON Response", "📊 XML View"])
            
            with tab1:
                st.markdown("**JSON Response dari FDA API:**")
                st.json(raw_data['raw_data'])
            
            with tab2:
                st.markdown("**Structured XML View:**")
                try:
                    # Convert JSON ke XML-like structure
                    xml_output = json_to_xml_display(raw_data['raw_data'])
                    st.code(xml_output, language='xml')
                except Exception as e:
                    st.error(f"Error converting to XML: {e}")
                    st.json(raw_data['raw_data'])

def json_to_xml_display(data, indent=0):
    """Convert JSON structure to XML-like format untuk display"""
    if isinstance(data, dict):
        xml_str = ""
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                xml_str += "  " * indent + f"<{key}>\n"
                xml_str += json_to_xml_display(value, indent + 1)
                xml_str += "  " * indent + f"</{key}>\n"
            else:
                xml_str += "  " * indent + f"<{key}>{escape_xml(str(value))}</{key}>\n"
        return xml_str
    elif isinstance(data, list):
        xml_str = ""
        for i, item in enumerate(data):
            xml_str += "  " * indent + f"<item_{i}>\n"
            xml_str += json_to_xml_display(item, indent + 1)
            xml_str += "  " * indent + f"</item_{i}>\n"
        return xml_str
    else:
        return "  " * indent + f"{escape_xml(str(data))}\n"

def escape_xml(text):
    """Escape special XML characters"""
    escapes = {
        '<': '&lt;',
        '>': '&gt;',
        '&': '&amp;',
        '"': '&quot;',
        "'": '&apos;'
    }
    for char, escape in escapes.items():
        text = text.replace(char, escape)
    return text

def main():
    # Initialize assistant
    assistant = SimpleRAGPharmaAssistant()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'show_raw_data' not in st.session_state:
        st.session_state.show_raw_data = False

    # [CUSTOM CSS DAN HEADER YANG SAMA...]

    # Header
    st.title("💊 Sistem Tanya Jawab Obat - FDA API")
    st.markdown("Sistem informasi obat dengan data langsung dari **FDA API** dan terjemahan menggunakan **Gemini AI**")

    # Toggle untuk raw data
    st.sidebar.markdown("### ⚙️ Developer Options")
    st.session_state.show_raw_data = st.sidebar.checkbox(
        "🔧 Tampilkan Raw FDA Data", 
        value=st.session_state.show_raw_data,
        help="Tampilkan data mentah dari FDA API selama proses tanya jawab"
    )

    # [CHAT INTERFACE YANG SAMA...]

    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Tulis pertanyaan Anda tentang obat:",
            placeholder="Contoh: Apa dosis paracetamol? Efek samping amoxicillin? Interaksi obat?",
            key="user_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
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
        
        with col_btn3:
            debug_btn = st.form_submit_button(
                "🐛 Debug Mode", 
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
            
            # Tampilkan raw data jika toggle aktif
            if st.session_state.show_raw_data:
                raw_data_collection = assistant.get_last_raw_data()
                if raw_data_collection:
                    display_raw_fda_data(raw_data_collection)
        
        st.rerun()

    if debug_btn and user_input:
        # Debug mode - tampilkan proses detail
        st.session_state.show_raw_data = True
        st.info("🐛 **Debug Mode Aktif** - Menampilkan semua detail proses...")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        with st.spinner("🔍 Debug Mode - Melacak semua proses..."):
            # Step by step debugging
            st.markdown("### 🔍 **Debug Process**")
            
            # 1. Drug Detection
            st.markdown("#### 1. Drug Detection")
            detected_drugs = assistant.drug_detector.detect_drug_from_query(user_input)
            if detected_drugs:
                for drug in detected_drugs:
                    st.write(f"- **Detected:** {drug['drug_name']} → **FDA Name:** {drug['fda_name']}")
            else:
                st.write("- ❌ No drugs detected")
            
            # 2. FDA API Calls
            st.markdown("#### 2. FDA API Calls")
            if detected_drugs:
                for drug in detected_drugs:
                    with st.spinner(f"Calling FDA API for {drug['fda_name']}..."):
                        api_result = assistant.fda_api.get_drug_info(drug['fda_name'])
                        if api_result:
                            st.success(f"✅ FDA data found for {drug['fda_name']}")
                            st.json(api_result['raw'])  # Tampilkan raw response
                        else:
                            st.error(f"❌ No FDA data for {drug['fda_name']}")
            
            # 3. Final Answer
            st.markdown("#### 3. Final Answer Generation")
            answer, sources = assistant.ask_question(user_input)
            
            # Add bot message
            st.session_state.messages.append({
                "role": "bot", 
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Tampilkan raw data
            raw_data_collection = assistant.get_last_raw_data()
            if raw_data_collection:
                display_raw_fda_data(raw_data_collection)
        
        st.rerun()

    if clear_btn:
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

    # [SIDEBAR DAN FOOTER YANG SAMA...]

if __name__ == "__main__":
    main()
