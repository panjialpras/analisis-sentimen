import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

dataset = None

def load_dataset(file_path):
    dataset = pd.read_csv(file_path, sep=';', encoding='latin1')
    dataset.drop_duplicates(inplace=True)
    return dataset

# fungsi remove digunakan untuk menghapus karakter tertentu

def remove(text):
    text = re.sub('[0-9]+', '', str(text))
    text = re.sub(r'\$\w*', '', str(text))
    text = re.sub(r'rt :[\s]+', '', str(text))
    text = re.sub("b'|b\"", '', str(text))
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", str(text))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('<.*?>', '', str(text))
    text = re.sub("\n", " ", str(text))
    text = re.sub(r"\b[a-zA-Z]\b", "", str(text))
    text = re.sub(r',', '', str(text))
    text = re.sub(r'#', '', str(text))
    # text = ' '.join(text.split())
    text = text.encode('ascii', 'replace').decode('ascii')
    return text

# tokenizing memisah tiap kalimat menjadi beberapa pecahan kata

def tokenizing(text):
    x = text.split()
    return x

# Stopword menghapus kata yang tidak mengandung arti

def stopword(text, stop_remover):
    text = stop_remover.remove(text)
    return text


# Stemmer menghapus kata yang memiliki imbuhan
def stemming(text, stemmer):
    text = stemmer.stem(str(text))
    return text


# Normasisasi kata
norm = {'mahasi': 'mahasiswa', 'penga': 'Pengawasan ', 'kek': ' kayak ', 'dr.': 'dokter', ' udah ': ' sudah ', 'sdh ': 'sudah ', 'mw ': 'mau',
        ' ga ': 'tidak', 'yg ': 'yang ', 'tak ': 'tidak', 'hrs ': 'harus', 'bp ': 'bapak', 'krn': 'karena', 'dlm': 'dalam', ' dr ': 'dari',
        'kpd': 'kepada', 'klo': 'kalau', 'tdk': 'tidak', 'dgn': 'dengan', 'dg': 'dengan', 'trus': 'terus', 'bwh': 'bawah', 'tsb': 'tersebut',
        'tp': 'tapi', 'tpi': 'tapi', 'bkn': 'bukan', 'ttg ': 'tentang', 'gnt ': 'ganti', 'bhwa': 'bahwa', 'skrg': 'sekarang',
        ' ama ': ' sama', 'bnyk': 'banyak', 'gmn': 'bagaimana', 'dr. ': 'dokter', ' mo ': 'mau', 'atw ': 'atau', ' ri ': ' republik indonesia ', 'sbg ': 'sebagai', 'Yg': 'Yang'}


def normalize(str_text):
    for i in range(len(str_text)):
        for a in norm:
            str_text[i] = str_text[i].replace(a, norm[a])
    return str_text