# Entity Extraction pada Teks Berita Menggunakan FastText Embedding dan CRF

Projek ini merupakan implementasi **Named Entity Recognition (NER)** atau **Entity Extraction** pada teks berita berbahasa Indonesia. Pada GitHub ini juga sudah terdapat Python Notebook yang berisikan EDA dan Model secara lengkap. Tetapi pada GitHub ini terdapat satu file yang belum diupload untuk streamlit dikarenakan ukurannya yang besar. Pada bagian bawah terdapat link gdrive yang berisikan zip folder streamlit secara keseluruhan. 

**Metode yang digunakan:**
- **FastText Embedding** → representasi kata dengan subword information 
- **CRF (Conditional Random Fields)** → model sequence labeling untuk memprediksi tag entitas secara kontekstual

## Anggota Kelompok 3

| No | NIM        | Nama                                   |
|----|------------|----------------------------------------|
| 1  | 2208561053 | Ni Made Gita Satviki Nirmala           |
| 2  | 2208561065 | Maedelien Tiffany Kariesta Simatupang  |
| 3  | 2208561070 | Febrian Valentino Agape                |

**Dosen Pengampu**  
Dr. Anak Agung Istri Ngurah Eka Karyawati, S.Si., M.Eng.

## Dataset

**idner-news-2k** — Dataset berita Indonesia untuk Named Entity Recognition (NER)  
Link: [https://github.com/khairunnisaor/idner-news-2k](https://github.com/khairunnisaor/idner-news-2k)  

Dataset ini berformat CoNLL (train.txt, dev.txt, test.txt) dengan tag NER IOB:  
- PER (Person)  
- LOC (Location)  
- ORG (Organization)  
- O  

## Pre-trained FastText Model (Ukuran Besar ~700MB)

File `fasttext_compatible.bin.wv.vectors_ngrams.npy` (dan file terkait lainnya) terlalu besar untuk diupload ke GitHub.  

**Download dari Google Drive:**  
[Link Google Drive Lengkap](https://drive.google.com/file/d/13uZvdpFCEvYn4JJQvQXreKpUeDQ-Vts1/view?usp=drive_link)
