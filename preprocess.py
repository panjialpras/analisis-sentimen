import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

dataset = None

def load_dataset(file_path):
    dataset = pd.read_csv(file_path, sep=';', encoding='latin1')
    dataset.drop_duplicates(inplace=True)
    return dataset

def text_clean(str_text):
    if not isinstance(str_text, str):
        str_text = str(str_text)
        str_text = str_text.lower()
        str_text = re.sub(r"b'", '', str_text)
        str_text = re.sub(r'RT', '', str_text)
        str_text = re.sub(r'#[\w]+',' ',str_text)
        str_text = re.sub(r'http\S+',' ',str_text)
        str_text = re.sub(r'[^a-zA-Z]+',' ',str_text)
    return str_text

norm = {'pgn':'ingin',' ntar ':'nanti ','kek ':' kayak ','dr.':'dokter',' udah ':' sudah ', 'sdh ' : 'sudah ','mw ':'mau',
        ' ga ':'tidak','yg ':'yang ','tak ':'tidak','hrs ':'harus','bp ':'bapak','krn':'karena','dlm':'dalam',' dr ':'dari',
        'kpd':'kepada','klo':'kalau','tdk':'tidak','dgn':'dengan','dg':'dengan','trus':'terus','bwh':'bawah','tsb':'tersebut',
        'tp':'tapi','tpi':'tapi','bkn':'bukan','ttg ':'tentang','gnt ':'ganti','bhwa':'bahwa','skrg':'sekarang',
        ' ama ':' sama','bnyk':'banyak','gmn':'bagaimana','dr. ':'dokter',' mo ':'mau','atw ':'atau',' ri ':' republik indonesia '
        ,'sbg ':'sebagai', 'Yg':'Yang'}

def normalize(str_text):
    for a in norm:
        str_text = str_text.replace(a, norm[a])
    return str_text

def case_folding(str_text):
    [str_text.lower() for text in str_text]
    return str_text

stemmer = StemmerFactory().create_stemmer()

def stemmer(str_text):
    stemmer = StemmerFactory().create_stemmer()
    str_text = stemmer.stem(str_text)
    return str_text

stop_remover = StopWordRemoverFactory().create_stop_word_remover()

def stopword(str_text):
    str_text = stop_remover.remove(str_text)
    return str_text

hapus = ["xe","x", "xa","xb","xc","xd","xe","xf","xg","xh","xi","xj","xk","xl","xm","xn", 
         "xo","xp","xq","xr","xs","xt","xu","xv","xz"]

def hapus_kata(str_text):
    for a in hapus:
        str_text = str_text.replace(a,"")
    return str_text

def tokenized(str_text):
    str_text = re.split('\W+', str_text)
    return str_text

dataset['tokenized'] = dataset['hapus_kata'].apply(lambda x: tokenized(x))

# def tokenized_text(str_text):
#     str_text['tokenized_text'] = str_text['tokenizing'].apply(lambda x: ' '.join(x))
#     return str_text