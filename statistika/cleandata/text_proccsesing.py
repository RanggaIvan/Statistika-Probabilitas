import pandas as pd
import re
import string

# ==========================================================
# 0. Baca data hasil balancing (undersampling)
# ==========================================================
df = pd.read_csv('tokopedia_reviews_olahraga_balanced.csv')

# Kolom teks ulasan
KOLOM_TEKS = 'review_text'

# ==========================================================
# 1. CLEANING & CASE FOLDING
#    - Hapus angka, tanda baca, emoji, karakter khusus
#    - Ubah ke huruf kecil
# ==========================================================
def cleaning_case_folding(teks):
    if pd.isna(teks):
        return ""

    teks = str(teks)

    # Hapus URL
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)

    # Hapus emoji & simbol unicode non-alfabet (emoji, simbol pictograph, dll)
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"  # simbol & pictograph, emoji tambahan
        "\U0001F600-\U0001F64F"  # emoticon
        "\U0001F680-\U0001F6FF"  # transport & maps
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # misc symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    teks = emoji_pattern.sub(' ', teks)

    # Case folding -> huruf kecil
    teks = teks.lower()

    # Hapus angka
    teks = re.sub(r'\d+', ' ', teks)

    # Hapus tanda baca & karakter khusus (sisakan huruf & spasi)
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = re.sub(r'[^a-z\s]', ' ', teks)

    # Rapikan spasi berlebih
    teks = re.sub(r'\s+', ' ', teks).strip()

    return teks


# ==========================================================
# 2. NORMALISASI SLANG / KATA GAUL
#    Kamus kata gaul umum e-commerce & bahasa sehari-hari -> kata baku
# ==========================================================
KAMUS_SLANG = {
    # Kata ganti & sapaan
    "gw": "saya", "gue": "saya", "gua": "saya", "ane": "saya", "aq": "saya",
    "sy": "saya", "aku": "saya", "w": "saya",
    "lu": "kamu", "loe": "kamu", "lo": "kamu", "elo": "kamu", "ente": "kamu",
    "km": "kamu", "kmu": "kamu", "u": "kamu",
    "mimin": "admin", "min": "admin",

    # Negasi
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "ngak": "tidak", "nggak": "tidak",
    "engga": "tidak", "enggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "kaga": "tidak", "kgk": "tidak", "g": "tidak", "gaa": "tidak",

    # Kata hubung / keterangan umum
    "tp": "tapi", "tpi": "tapi", "tapii": "tapi",
    "dgn": "dengan", "dg": "dengan", "sama": "dengan",
    "krn": "karena", "krna": "karena", "karna": "karena", "coz": "karena", "cz": "karena",
    "utk": "untuk", "buat": "untuk", "bwt": "untuk", "u/": "untuk",
    "yg": "yang", "yng": "yang",
    "dr": "dari", "drpd": "daripada", "drpada": "daripada",
    "jd": "jadi", "jdi": "jadi",
    "jg": "juga", "jga": "juga", "jugaa": "juga",
    "klo": "kalau", "kalo": "kalau", "klu": "kalau", "kl": "kalau",
    "krg": "kurang", "krng": "kurang",
    "spy": "supaya", "biar": "supaya",
    "stlh": "setelah", "stelah": "setelah",
    "sblm": "sebelum", "sblum": "sebelum",
    "skrg": "sekarang", "skarang": "sekarang", "skrng": "sekarang",
    "trs": "terus", "trus": "terus", "truz": "terus",
    "sm": "sama",

    # Kata waktu & lokasi
    "dtg": "datang", "dateng": "datang",
    "dmn": "dimana", "dimna": "dimana",
    "disni": "disini", "sni": "sini",
    "situ": "sana",
    "skrang": "sekarang",
    "bsk": "besok", "bsok": "besok",
    "kmrn": "kemarin", "kmarin": "kemarin", "kemaren": "kemarin",
    "td": "tadi", "tdi": "tadi",

    # Singkatan umum chat/e-commerce
    "byr": "bayar", "byar": "bayar",
    "brg": "barang", "brng": "barang", "brang": "barang",
    "pjl": "penjual", "seller": "penjual", "olshop": "toko online",
    "cod": "bayar di tempat",
    "kurir": "kurir", "exp": "ekspedisi", "ekspedisi": "ekspedisi",
    "resi": "resi",
    "orderan": "pesanan", "order": "pesan", "pesenan": "pesanan",
    "sesion": "sesuai",
    "respon": "tanggapan", "respn": "tanggapan",
    "recommended": "direkomendasikan", "rekomended": "direkomendasikan",
    "fast respon": "cepat tanggap", "fastrespon": "cepat tanggap",
    "cs": "layanan pelanggan", "wasap": "whatsapp", "wa": "whatsapp",
    "chat": "pesan", "pdhl": "padahal", "pdahal": "padahal",
    "bgt": "banget", "bngt": "banget", "bgttt": "banget", "bener": "benar",
    "bener2": "benar-benar", "beneran": "benar-benar",
    "emg": "memang", "emang": "memang",
    "gmn": "bagaimana", "gimana": "bagaimana", "gmna": "bagaimana",
    "knp": "kenapa", "knapa": "kenapa",
    "gpp": "tidak apa-apa", "gapapa": "tidak apa-apa", "gakpapa": "tidak apa-apa",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
    "blm": "belum", "blum": "belum", "belom": "belum",
    "sy": "saya",
    "gt": "begitu", "gitu": "begitu", "githu": "begitu",
    "gini": "begini", "gene": "begini",
    "bgtu": "begitu",
    "nih": "ini", "ni": "ini",
    "tuh": "itu", "tu": "itu",
    "aja": "saja", "ajah": "saja", "aj": "saja", "ajaa": "saja",
    "doang": "saja", "doank": "saja",
    "cm": "cuma", "cuman": "cuma", "cma": "cuma",
    "bgs": "bagus", "bgus": "bagus", "bagusss": "bagus",
    "jelek2": "jelek-jelek",
    "puas2": "puas-puas",
    "mantul": "mantap betul", "mantab": "mantap", "mantaap": "mantap",
    "top": "bagus", "topbgt": "bagus banget",
    "oke": "baik", "okee": "baik", "ok": "baik", "okey": "baik",
    "sip": "baik", "sippp": "baik",
    "recommend": "rekomendasi",
    "worth": "sepadan", "worthit": "sepadan", "worth it": "sepadan",
    "gacor": "bagus sekali",
    "parah": "sangat buruk",
    "ancur": "hancur", "hancur2an": "hancur",
    "jelek": "jelek", "jlk": "jelek",
    "kecewa": "kecewa", "kecewaa": "kecewa",
    "puas": "puas", "puasss": "puas",
    "rusak2": "rusak-rusak",
    "cacat": "cacat",
    "ori": "asli", "original": "asli",
    "kw": "tiruan", "palsu": "palsu",
    "cpt": "cepat", "cepet": "cepat", "cepetan": "cepat",
    "lambat": "lambat", "lmbt": "lambat", "lelet": "lambat", "lola": "lambat",
    "lama": "lama", "lamaa": "lama",
    "rapih": "rapi", "rapii": "rapi",
    "packing": "kemasan", "paking": "kemasan", "pcking": "kemasan",
    "packingnya": "kemasannya",
    "sesuai": "sesuai", "sesuaii": "sesuai",
    "ngga sesuai": "tidak sesuai",
    "gasesuai": "tidak sesuai",
    "sesion": "sesuai",
    "kirim": "kirim", "ngirim": "mengirim",
    "balesin": "membalas", "bales": "balas", "balas2": "balas",
    "tggu": "tunggu", "tunggu2": "tunggu-tunggu", "nunggu": "menunggu",
    "nyampe": "sampai", "sampe": "sampai", "nyampai": "sampai",
    "mo": "mau", "mw": "mau", "pengen": "ingin", "pengin": "ingin",
    "males": "malas", "mager": "malas",
    "bgtu": "begitu",
    "kzl": "kesal", "kesel": "kesal",
    "sebel": "sebal",
    "makasih": "terima kasih", "mksh": "terima kasih", "thx": "terima kasih",
    "thanks": "terima kasih", "trims": "terima kasih",
    "sy": "saya",
    "org": "orang", "orgnya": "orangnya",
    "pjng": "panjang",
    "pndek": "pendek",
    "bhn": "bahan", "bahannya": "bahannya",
    "wrn": "warna", "warnanya": "warnanya",
    "ukurannya": "ukurannya", "ukrn": "ukuran",
    "sesuaii": "sesuai",
    "pas": "pas",
    "kegedean": "terlalu besar", "gede": "besar", "gedean": "lebih besar",
    "kekecilan": "terlalu kecil", "kecil2": "kecil-kecil",
    "murah": "murah", "mahal": "mahal", "mahalan": "lebih mahal",
    "worth": "sepadan",
    "sllu": "selalu", "selalu": "selalu",
    "pertamax": "pertama",
    "sm2": "sama-sama",
    "sipp": "baik",
}

def normalisasi_slang(teks, kamus=KAMUS_SLANG):
    kata_kata = teks.split()
    hasil = [kamus.get(kata, kata) for kata in kata_kata]
    return " ".join(hasil)


# ==========================================================
# 3. STOPWORD REMOVAL
#    Daftar stopword bahasa Indonesia (kata hubung/kata umum
#    yang tidak bermakna sentimen)
# ==========================================================
STOPWORDS_ID = set("""
yang untuk pada ke para namun menurut antara dia dua ia seperti jika jika
sehingga kembali dan tidak ini karena kepada oleh saat harus sementara
setelah belum kami sekitar bagi serta di dari dalam akan atau juga hanya
sudah saya sekali kita adalah bisa dengan begitu mengapa kenapa yaitu
yakni daripada itu lain lalu lain kalau tapi tetapi bahwa apalagi
bagaimanapun bagaimana selain masih diperlukan sebab pun tak boleh
sepanjang tanpa tentu terhadap secara agak lagi lain oleh setiap
lakukan begitu banyak sebagai jadi pernah supaya sambil dapat kalaupun
gimana sesekali harusnya nya juga si tuh nih deh dong sih kok yah ya
loh lho kan pun oh eh wah kah ku mu nya ku kau engkau anda ente
kamu kau aku saya gue gua w lu loe elo mereka kalian
adalah ialah merupakan yaitu yakni bahwa untuk daripada demi guna
ada aja saja doang cuma hanya hingga sampai sejak selama semenjak
walaupun meskipun biarpun kendati sekalipun andaipun
para si sang tersebut ybs
pada di ke dari dalam luar atas bawah depan belakang antara
dan atau serta maupun baik entah
adapun apabila bila jika jikalau kalau manakala seandainya sekiranya
karena sebab oleh karena itu sehingga akibatnya maka
agar supaya biar
meski meskipun walau walaupun biarpun sekalipun kendati kendatipun
seperti bagai bagaikan laksana umpama ibarat sebagaimana
""".split())

# Tambahan stopword khas ulasan e-commerce yang netral (tidak bermuatan sentimen)
STOPWORDS_TAMBAHAN = {
    "nya", "yg", "utk", "dgn", "dr", "krn", "tsb", "dll", "dsb",
    "barang", "produk", "toko", "penjual", "pembeli", "tokopedia",
    "kurir", "paket", "order", "pesanan", "belanja"
}

STOPWORDS_ALL = STOPWORDS_ID | STOPWORDS_TAMBAHAN

# Kata negasi WAJIB dipertahankan -> jika ikut terhapus, makna sentimen bisa
# terbalik (mis. "tidak bagus" jadi "bagus" saja). Maka kata-kata ini
# dikeluarkan dari daftar stopword di atas.
KATA_NEGASI = {"tidak", "bukan", "jangan", "tanpa", "belum", "kurang"}
STOPWORDS_ALL = STOPWORDS_ALL - KATA_NEGASI

def stopword_removal(teks, stopwords=STOPWORDS_ALL):
    kata_kata = teks.split()
    hasil = [kata for kata in kata_kata if kata not in stopwords]
    return " ".join(hasil)


# ==========================================================
# 4. TERAPKAN SEMUA TAHAP PREPROCESSING
#    (tahap antara dihitung sementara, lalu dibuang -
#     hasil akhir disimpan di kolom baru, ulasan asli
#     di kolom review_text tetap dipertahankan)
# ==========================================================
teks_cleaned = df[KOLOM_TEKS].apply(cleaning_case_folding)
teks_normalized = teks_cleaned.apply(normalisasi_slang)
teks_final = teks_normalized.apply(stopword_removal)

# Simpan hasil preprocessing di kolom baru, ulasan asli tetap ada
df['review_text_preprocessed'] = teks_final

# ==========================================================
# 5. Tampilkan contoh hasil & simpan
# ==========================================================
print("=== Contoh Hasil (ulasan asli vs hasil preprocessing) ===")
print(df[[KOLOM_TEKS, 'review_text_preprocessed']].head(5).to_string())

df.to_csv('tokopedia_reviews_olahraga_preprocessed.csv', index=False)
print("\nData hasil preprocessing disimpan ke 'tokopedia_reviews_olahraga_preprocessed.csv'")