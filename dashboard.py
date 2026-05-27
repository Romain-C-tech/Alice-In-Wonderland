import nltk

def download_nltk_resource(resource, path):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

download_nltk_resource('wordnet', 'corpora/wordnet')
download_nltk_resource('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng')
download_nltk_resource('stopwords', 'corpora/stopwords')
download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('omw-1.4', 'corpora/omw-1.4')

import streamlit as st
import pandas as pd
from bookworm import card
from bookworm import get_data
from bookworm import download_book

# ----------------------------------------------------------------------------------------------------


search_author = st.sidebar.text_input("Search an author :")
button_author = st.sidebar.button("Download all his books")

if button_author:
    with st.sidebar.spinner("Loading..."):
            download_book(search_author)

# ----------------------------------------------------------------------------------------------------

def show_card(id):

    dico = card(id)

    info = dico["info"]
    lexdiv = dico["lexdiv"]
    topics = dico["topics"]
    entities = dico["entities"][1]
    summary = dico["summary"]
    similar = dico["similar"]
    cover = dico["cover"]
    cloud = dico["cloud"]

# ----------------------------------------------------------------------------------------------------

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(f"./Cover/{cover}")

    st.divider()

# ----------------------------------------------------------------------------------------------------

    with st.expander("ⓘ - Info"):
        st.subheader(f"Categorized in :")
        st.text(info["bookshelves"])
        st.subheader(f"ID :")
        st.text(id)

# ----------------------------------------------------------------------------------------------------

    with st.expander("𖠿 - Entities"):
        st.subheader(f"Most present character :")
        for i in range(len(entities[0])):
            st.text(f"- {i + 1} - {entities[0][i]}")

        st.subheader(f"Most present place :")
        for i in range(len(entities[1])):
            st.text(f"- {i + 1} - {entities[1][i]}")

# ----------------------------------------------------------------------------------------------------

    with st.expander("🕮 - Summary"):
        st.subheader(f"Book summary :")
        st.text(summary)

# ----------------------------------------------------------------------------------------------------

    with st.expander("𓍝 - Similar"):
        st.subheader("Similar book :")
        for i in range(len(similar)):
            st.text(f"- {i + 1} - {similar[i]}")
        
# ----------------------------------------------------------------------------------------------------

    with st.expander("🌣 - Lexdiv"):
        for key, value in lexdiv.items():
            st.text(f"{key.upper()} : {value}")

# ----------------------------------------------------------------------------------------------------

    with st.expander("🗒 - Topics"):
        st.image(f"./Clouds/{cloud}")
        i = 1
        list_word = ""
        for values in topics.values():
            st.subheader(f"Chapter {i}")
            for element in values:
                list_word += element + " - "
            st.text(list_word)
            list_word = ""
            st.divider()
            i += 1

# ----------------------------------------------------------------------------------------------------

@st.cache_data
def load_books():
    return get_data()

book = load_books()

book_selected_idx = None

with st.expander("Select a book to generate its card"):
    search = st.text_input("Search a book :")
    filtered = book[book["Title"].str.contains(search, case=False, na=False, regex=False)][["Text#", "Title", "Authors"]].drop_duplicates(subset="Text#").head(10)
    filtered_indexed = filtered.set_index("Text#")

    if not search:
        st.text("Type a book name to search")
    elif len(filtered) == 0:
        st.text("No book found")
    else:
        button =  st.button("Create a card")
        book_selected_idx = st.radio(
            "Select a book :",
            filtered_indexed.index,
            format_func=lambda i: f"{filtered_indexed.loc[i, 'Title']} (ID: {i})"
        )

# ----------------------------------------------------------------------------------------------------

st.divider()
if not search or filtered.empty:
    st.title("No book selected")
    st.text("Please select a valid name in order to create a card")
else:
    book_row = filtered_indexed.loc[book_selected_idx]
    st.title(book_row["Title"])
    st.subheader(f"By {book_row["Authors"]}")

# ----------------------------------------------------------------------------------------------------

if book_selected_idx:
    if button:
        with st.spinner("Loading..."):
            show_card(book_selected_idx)

# ----------------------------------------------------------------------------------------------------

st.markdown("""
    <style>
    div[role=radiogroup] label {
        border-bottom: 1px solid #ccc;
        padding-bottom: 8px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)