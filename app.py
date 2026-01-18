import streamlit as st
from utils import predict_entities

st.set_page_config(
    page_title="NER Bahasa Indonesia",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .big-font {
        font-size: 48px !important;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #666;
        margin-bottom: 40px;
    }
    /* Highlight entity - warna soft tapi jelas di dark & light mode */
    .entity-per {
        background-color: #FFCDD2;
        color: #C62828;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
        margin: 1px;
    }
    .entity-org {
        background-color: #C8E6C9;
        color: #2E7D32;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
        margin: 1px;
    }
    .entity-loc {
        background-color: #FFF9C4;
        color: #F9A825;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
        margin: 1px;
    }
    .highlighted-text {
        font-size: 18px;
        line-height: 2.2;
        padding: 20px;
        background-color: var(--default-background-color);
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 20px 0;
    }
    .legend-box {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 30px 0;
        flex-wrap: wrap;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deteksi Otomatis Nama Orang, Organisasi, dan Lokasi dalam Teks Bahasa Indonesia</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "Masukkan teks berbahasa Indonesia di sini:",
    value="",
    height=200,
    placeholder="Contoh: Presiden Joko Widodo bertemu dengan Gubernur Bank Indonesia di Jakarta..."
)

if st.button("🔍 Analisis Entity", type="primary", use_container_width=True):
    if text_input.strip():
        with st.spinner("Sedang menganalisis teks..."):
            results = predict_entities(text_input)
        
        st.markdown("### 📄 Hasil Prediksi Entity")
        
        highlighted_text = ""
        current_entity = None
        for r in results:
            token = r['token']
            entity = r['entity']
            
            if entity in ["B-PER", "I-PER"]:
                cls = "entity-per"
                label = "PER"
            elif entity in ["B-ORG", "I-ORG"]:
                cls = "entity-org"
                label = "ORG"
            elif entity in ["B-LOC", "I-LOC"]:
                cls = "entity-loc"
                label = "LOC"
            else:
                cls = ""
                label = None
            
            if cls:
                highlighted_text += f"<span class='{cls}'>{token}</span> "
            else:
                highlighted_text += f"{token} "
        
        st.markdown(f'<div class="highlighted-text">{highlighted_text}</div>', unsafe_allow_html=True)
        
        st.markdown("### 🏷️ Keterangan Warna")
        st.markdown("""
        <div class="legend-box">
            <div class="legend-item">
                <span class="entity-per">Nama Orang</span>
                <span>(PER)</span>
            </div>
            <div class="legend-item">
                <span class="entity-org">Organisasi</span>
                <span>(ORG)</span>
            </div>
            <div class="legend-item">
                <span class="entity-loc">Lokasi</span>
                <span>(LOC)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Detail Token")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "token": "Token",
                "pos": "POS Tag",
                "entity": "Entity"
            }
        )
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
