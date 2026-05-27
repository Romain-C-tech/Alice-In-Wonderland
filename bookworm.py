# ----------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import hstack
from wordcloud import WordCloud
from nltk import pos_tag
import pandas as pd
import numpy as np
import requests
import argparse
import spacy
import nltk
import sys
import re

# ----------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()
language_list = {

    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "sv": "swedish",
    "no": "norwegian",
    "da": "danish",
    "fi": "finnish",
    "pl": "polish",
    "cs": "czech",
    "sk": "slovak",
    "hu": "hungarian",
    "ro": "romanian",
    "el": "greek",
    "tr": "turkish",

    "ru": "russian",
    "uk": "ukrainian",
    "bg": "bulgarian",
    "sr": "serbian",
    "hr": "croatian",
    "bs": "bosnian",
    "sl": "slovenian",
    "lt": "lithuanian",
    "lv": "latvian",
    "et": "estonian",
    
    "ar": "arabic",
    "he": "hebrew",
    "fa": "persian",
    "ur": "urdu",

    "zh": "chinese",
    "zh-tw": "chinese (traditional)",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "bn": "bengali",
    "vi": "vietnamese",
    "th": "thai",
    "id": "indonesian",
    "ms": "malay",
    "tl": "tagalog",

    "sw": "swahili",
    "am": "amharic",
    "yo": "yoruba",
    "ha": "hausa",
    "zu": "zulu",

    "ca": "catalan",
    "eu": "basque",
    "gl": "galician",
    "cy": "welsh",
    "is": "icelandic",
    "mt": "maltese",
    "sq": "albanian",
    "mk": "macedonian",
    "hy": "armenian",
    "ka": "georgian",
    "az": "azerbaijani",
    "kk": "kazakh",
    "uz": "uzbek",
}

punctuations = "!\"#$%&'()*+,./_“"
data = None
regex = re.compile(r'[\n\r\t]+')
PUNCT = re.compile(f"[{re.escape(punctuations)}]")

# ----------------------------------------------------------------------------------------------------
# Error Handling
# ----------------------------------------------------------------------------------------------------

def FindLastId():
   """
    Find the last book ID in the Gutenberg catalog.

    Returns:
        int: The last book ID in the catalog.
    """
   global data
   if data is None:
        data = get_data()
   return data["Text#"].max()

def get_data():
    """
    Get the data from the Gutenberg catalog and preprocess it.

    Returns:
        pd.DataFrame: The preprocessed data from the Gutenberg catalog.
    """
    try :
        URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv?"
        books = pd.read_csv(URL)
        books["Authors"] = books["Authors"].fillna("Anonymous")
        books["Bookshelves"] = books["Bookshelves"].fillna("Uncategorized")
        books["Subjects"] = books["Subjects"].fillna("Unspecified")
        books["Language"] = books["Language"].str[:2]
        books = books[books["Type"] == "Text"]

        return books

    except requests.exceptions.ConnectionError:
        print("Error : Connection Error.")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("Error : Timeout.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP : {e.response.status_code} - {e.response.reason}")
        sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"Error exception : {e}")
        sys.exit(1)

def id_error(id):
    """
    Validate the book ID provided as a command-line argument.

    Returns:
        int: The validated book ID.
    """
    id = int(id)
    if id < 1:
        raise argparse.ArgumentTypeError("ID must be an integer")
    if id > FindLastId() :
        raise argparse.ArgumentTypeError("ID is out of range")
    return id

# ----------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--lexdiv",type = id_error ,help = "Return various book metrics into a dictionary.")
parser.add_argument("--topics", type = id_error, help = "Return the main topic top 10 words from each section.")
parser.add_argument("--entities", type = id_error, help = "Return the locations and characters of a book in a dictionary.")
parser.add_argument("--summarize", type = id_error, help = "Return a string summarizing a book in a string of few sentences.")
parser.add_argument("--similar", type = id_error, help = "Return a list of 5 similar books.")
parser.add_argument("--card", type = id_error, help = "Return gathers information about a book")
parser.add_argument("--authors", type = str, help = "Return the list of all book written or co-written by an author.")
parser.add_argument("--category", type = str, help = "Return the list of all book of a category.")

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------

def get_info(id):
    """
    Get the information of a book from the Gutenberg catalog.

    Returns:
        tuple: A tuple containing the book ID, authors, and bookshelves.
    """
    global data
    if data is None:
        data = get_data()

    book = data[data["Text#"] == id]

    book_id = book["Text#"].values[0]
    authors = book["Authors"].values[0]
    bookshelves = book["Bookshelves"].values[0]

    return book_id,authors,bookshelves

# Extract only text without header and footer
def read(id):
    """
    Read the text of a book from the Gutenberg catalog.

    Returns:
        str: The text of the book without the header and footer.
    """

    try :
        URL = f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        book = response.text

    except requests.exceptions.ConnectionError:
        print("Error : Connection Error.")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("Error : Timeout.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP : {e.response.status_code} - {e.response.reason}")
        sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"Error exception : {e}")
        sys.exit(1)
    
    start = book.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = book.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    
    if start == -1 or end == -1:
        print("Marqueurs introuvables")
    else:
        start = book.find("\n", start) + 1
        book = book[start:end].strip()
    return book

def get_stop_words(language : str):
    """
    Get the stop words for a given language.

    Returns:
        set: A set of stop words for the given language.
    """
    nltk.download('stopwords',quiet=True)
    full_language = language_list[language]
    return set(stopwords.words(full_language))

def get_pos_tags(tag):
    """
    Get the part of speech tag for a given tag.

    Returns:
        str: The part of speech tag for the given tag.
    """
    if tag.startswith("N"):
        return "n"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"   

def download_book(author : str):
    """
    The function saves the book as a text file in the "Books" directory.
    Args:        author (str): The name of the author whose books you want to download.
    """
    global data
    if data is None:
        data = get_data()
    books = data[data["Authors"].str.contains(author, case=False)]
    ids = books["Text#"].tolist()
    titles = books["Title"].tolist()
    
    for id, title in zip(ids, titles):
        title = re.sub(r'[\\/:*?"<>|()\[\],]', "", title.strip())
        open(f"./Books/{title}.txt", "w", encoding="utf-8").write(read(id))
# ----------------------------------------------------------------------------------------------------
# lexdiv
# ----------------------------------------------------------------------------------------------------

def lexdiv(id : int):
    """
    Calculate various lexical diversity metrics for a given book ID.

    Returns:
        dict: A dictionary containing the following metrics:
    """
    texte = read(id)
    
    raw_word_list = []
    word = ""
    
    for char in texte:
        if char in punctuations:
            raw_word_list.append(word)
            word = ""
        else:
            word += char
    raw_word_list.append(word)

    clean_word_list = []
    for element in raw_word_list:
        if element != "" and element != " ":
            clean_word_list.append(element)

# ----------------------------------------------------------------------------------------------------

    unique_word_list = list(set(clean_word_list))

# ----------------------------------------------------------------------------------------------------

    counter = Counter(clean_word_list)
    occurring_once_list = [word for word, counts in counter.items() if counts == 1]

# ----------------------------------------------------------------------------------------------------

    average = 0
    for element in clean_word_list:
        average += len(element)
    average /= len(clean_word_list)

# ----------------------------------------------------------------------------------------------------

    return {
        "tok": len(clean_word_list), # total number of word tokens
        "typ": len(unique_word_list), # number of unique word tokens
        "hap": len(occurring_once_list), # number of word tokens occurring only once
        "ttr": len(unique_word_list) / len(clean_word_list), # number of unique words tokens divided by number of word tokens
        "mwl": average, # mean number of characters per word token
        "mwf": len(clean_word_list) / len(unique_word_list) # number of word token divided by number of unique word tokens
    }

if args.lexdiv:
        result = lexdiv(args.lexdiv)
        print(result)

# ----------------------------------------------------------------------------------------------------
# Topics
# ----------------------------------------------------------------------------------------------------

def topics(id : int):
    """
        Find the words with the highest occurrence in each section of a book.

    Returns:
        dict: A dictionary where the keys are the section numbers and the values are lists of the 10 most common words in each section.
    """
    global data
    if data is None:
        data = get_data()

    book = data[data["Text#"] == id]
    language : str = book["Language"].values[0]
    stop_words = get_stop_words(language)

    texte = read(id)   
    topics = texte.split("\r\n\r\n\r\n\r\n\r\n")
    dico = {}
    
    for i, section in enumerate(topics):
        min_size = 100
        if len(section) > min_size :
            dico[i] = []
            clean_word_list = []
            
            section = regex.sub(" ", section)
            section = PUNCT.sub("", section)
            
            tokens = word_tokenize(section)
            tags = pos_tag(tokens)

            for word, tag in tags:
                if word.lower() == 'are' or word.lower() in ['is', 'am']:
                    clean_word_list.append(word)
                else :
                    clean_word_list.append(lemmatizer.lemmatize(word.lower(), get_pos_tags(tag)))
            
            section = clean_word_list
            section = [word for word in section if word not in stop_words and word != "" and word != " " and len(word) > 2]

            c = Counter(section)
            top_occurrence = c.most_common(10)
            for word, _ in top_occurrence :
                dico[i].append(word)
        
    return dico

if args.topics:
    result = topics(args.topics)
    print(result)

# ----------------------------------------------------------------------------------------------------
# Authors
# ----------------------------------------------------------------------------------------------------

def BooksTitleOfAnAuthor(name : str):
    """
    Find the books written by a given author.

        Returns:
            tuple: A tuple containing two lists: the first list contains the IDs of the books,
            and the second list contains the titles of the books.
    """
    global data
    if data is None:
        data = get_data()
    books = data[data["Authors"].str.contains(name, case=False)]
    ids = books["Text#"].tolist()
    titles = books["Title"].tolist()
    
    
    return ids, titles

if args.authors:
    ids, titles = BooksTitleOfAnAuthor(args.authors)
    if len(ids) == 0:
        print(f"No book found for the author {args.authors}")
    elif len(ids) > 20:
        print(f"Too many books found for the author {args.authors} : {len(ids)} books found. Only the first 20 books will be downloaded.")
        
        ids = ids[:20]
        titles = titles[:20]
        for id, title in zip(ids, titles):
            title = re.sub(r'[\\/:*?"<>|()\[\],]', "", title)
            open(f"./Books/{title}.txt", "w", encoding="utf-8").write(read(id))
    
    else :
        for id, title in zip(ids, titles):
            title = re.sub(r'[\\/:*?"<>|()\[\],]', "", title.strip())
            open(f"./Books/{title}.txt", "w", encoding="utf-8").write(read(id))

# ----------------------------------------------------------------------------------------------------
# Category
# ----------------------------------------------------------------------------------------------------

def BooksTitleOfACategory(category : str):
    """Find the books of a given category.

        Returns:
            tuple: A tuple containing two lists: the first list contains the IDs of the books,
            and the second list contains the titles of the books.
    """
    global data
    if data is None:
        data = get_data()
    books = data[data["Bookshelves"].str.contains(category, case=False)]
    ids = books["Text#"].tolist()
    titles = books["Title"].tolist()
    
    return ids, titles

if args.category:
    ids, titles = BooksTitleOfAnAuthor(args.category)
    if len(ids) == 0:
        print(f"No book found for the category {args.category}")
    elif len(ids) > 20:
        print(f"Too many books found for the category {args.category} : {len(ids)} books found. Only the first 20 books will be downloaded.")

        ids = ids[:20]
        titles = titles[:20]  
        for id, title in zip(ids, titles):
            title = re.sub(r'[\\/:*?"<>|()\[\],]', "", title)
            open(f"./Books/{title}.txt", "w", encoding="utf-8").write(read(id))
    
    else :
        for id, title in zip(ids, titles):
            title = re.sub(r'[\\/:*?"<>|()\[\],]', "", title.strip())
            open(f"./Books/{title}.txt", "w", encoding="utf-8").write(read(id))

# ----------------------------------------------------------------------------------------------------
# Summarize
# ----------------------------------------------------------------------------------------------------

# Method TF-IDF : Give a score for all sentences to take the 10 most important sentences.

def summarize(text_id : int,n_sentence : int):
    """
    Summarize a book by extracting the most important sentences based on their TF-IDF scores.

    Returns:
        str: A summary of the book consisting of the most important sentences.
    """
    global data
    if data is None:
        data = get_data()

    book = data[data["Text#"] == text_id]
    language : str = book["Language"].values[0]
    stop_words = get_stop_words(language)
    
    texte = read(text_id)

    sections = texte.split("\r\n\r\n\r\n\r\n\r\n")
    new_texte = [] # Future liste de tous les paragraphes
    for _,section in enumerate(sections):
        section = section.split("\r\n\r\n")
        for paragraphe in section :
            new_texte.append(paragraphe)

    vectorizer = TfidfVectorizer(stop_words = list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(new_texte)

    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    top_indices = np.argsort(scores)[-n_sentence:]
    top_indices_sorted = sorted(top_indices)

    resume = "\n\n".join([new_texte[i] for i in top_indices_sorted])

    return resume

if args.summarize:
    print(summarize(text_id = args.summarize, n_sentence = 2))

# ----------------------------------------------------------------------------------------------------
# Entities
# ----------------------------------------------------------------------------------------------------

def entities(id):
    """
    Extract the characters and locations from a book using named entity recognition.

    Returns:
        tuple: A tuple containing two elements:
            - A dictionary with two keys: "characters" and "locations". The value of "characters" is a list of unique character names, and the value of "locations" is a list of unique location names.
            - A list containing two elements: the first element is a list of the top 3 most common character names, and the second element is a list of the top 3 most common location names.
    """
    response = read(id)
    nlp = spacy.load("en_core_web_sm")

    regex = re.compile(r'[\n\r\t]+')
    text = regex.sub(" ", response)

    characters_set = set()
    locations_set = set()

    characters_counter = Counter()
    locations_counter = Counter()

    chunk_size = 500_000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        doc = nlp(chunk)
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                locations_set.add(ent.text)
                locations_counter[ent.text] += 1
            elif ent.label_ == "PERSON":
                characters_set.add(ent.text)
                characters_counter[ent.text] += 1

    top_characters = [name for name, _ in characters_counter.most_common(3)]
    top_locations = [name for name, _ in locations_counter.most_common(3)]

# ----------------------------------------------------------------------------------------------------

    return {"characters": list(characters_set), "locations": list(locations_set)}, [top_characters, top_locations]

# ----------------------------------------------------------------------------------------------------

if args.entities:
    print(entities(args.entities))

# ----------------------------------------------------------------------------------------------------
# Similar
# ----------------------------------------------------------------------------------------------------

def similar(book_id):
    """
    Find the 5 most similar books to a given book ID based on their titles, authors, subjects, bookshelves, and LoCC.

    Returns:
        list: A list of the titles of the 5 most similar books.
    """
    global data
    if data is None:
        data = get_data()

    slect_book = data.reset_index(drop=True)

# ----------------------------------------------------------------------------------------------------

    title_list = []
    authors_list = []
    subjects_list = []
    bookshelves_list = []
    loCC_list = []

    vectorize_list = [
        [title_list, "Title"],
        [authors_list, "Authors"],
        [subjects_list, "Subjects"],
        [bookshelves_list, "Bookshelves"],
        [loCC_list, "LoCC"]
    ]

# ----------------------------------------------------------------------------------------------------

    for elements in vectorize_list:
        for element in slect_book[elements[1]]:
            elements[0].append(str(element) if pd.notna(element) else "")

# ----------------------------------------------------------------------------------------------------

    X_list = []
    vectorizer = TfidfVectorizer()
    for elements in vectorize_list:
        X = vectorizer.fit_transform(elements[0])
        X_list.append(X)

    X_final = normalize(hstack(X_list), norm="l2")

# ----------------------------------------------------------------------------------------------------

    idx = slect_book.index[slect_book["Text#"] == book_id].tolist()
    if not idx:
        print(f"The book {book_id} is not in the list.")
        return
    idx = idx[0]

# ----------------------------------------------------------------------------------------------------

    scores = cosine_similarity(X_final[idx], X_final).flatten()
    top_indices = np.argsort(scores)[::-1][1:6]

    return_list = []
    for i in top_indices:
        return_list.append(slect_book.iloc[i]['Title'])

    return return_list

# ----------------------------------------------------------------------------------------------------

if args.similar:
    print(similar(args.similar))

# ----------------------------------------------------------------------------------------------------
# Book Cover
# ----------------------------------------------------------------------------------------------------

def book_cover(id):
    """
    Download the cover image of a book from the Gutenberg API.

    Returns:
        str: The filename of the downloaded cover image.
    """
    try :
        URL = f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.cover.medium.jpg"
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
        cover = response.content
        
        with open(f"./Cover/cover_{id}.jpg", "wb") as f:
            f.write(cover)
        return  f"cover_{id}.jpg"

    except requests.exceptions.ConnectionError:
        print("Error : Connection Error.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error : Timeout.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP : {e.response.status_code} - {e.response.reason}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error exception : {e}")
        sys.exit(1)

# ----------------------------------------------------------------------------------------------------
# Book Cloud
# ----------------------------------------------------------------------------------------------------

def book_cloud(id):
    """
    Generate a word cloud image for a given book ID.

    Returns:
        str: The filename of the generated word cloud image.
    """
    text = read(id)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(f"./Clouds/wordcloud_{id}.png")
    return f"wordcloud_{id}.png"

# ----------------------------------------------------------------------------------------------------
# Card
# ----------------------------------------------------------------------------------------------------

def card(id):
    """
    Gather information about a book, including its infos, lexical diversity metrics,topics, entities, summary, similar books, cover image, and word cloud.

    Returns:
        dict: A dictionary containing the following keys:
            - "info": A dictionary containing the book ID, authors, and bookshelves.
            - "lexdiv": A dictionary containing various lexical diversity metrics for the book.
            - "topics": A dictionary where the keys are the section numbers and the values are lists of the 10 most common words in each section.
            - "entities": A tuple containing two elements: a dictionary with two keys ("characters" and "locations") and a list of the top 3 most common character names and location names.
            - "summary": A string summarizing the book in a few sentences.
            - "similar": A list of the titles of 5 similar books.
            - "cover": The filename of the downloaded cover image for the book.
            - "cloud": The filename of the generated word cloud image for the book.
    """
    info = get_info(id = id)

    return {
        "info" : {
            "id": info[0].tolist(),
            "authors": info[1],
            "bookshelves": info[2]
        },
        "lexdiv" : lexdiv(id = id),
        "topics" : topics(id = id),
        "entities" : entities(id = id),
        "summary" : summarize(text_id= id , n_sentence = 2),
        "similar" : similar(book_id = id),
        "cover" : book_cover(id = id),
        "cloud" : book_cloud(id = id)
    }

if args.card:
    response = card(args.card)
    print(response)