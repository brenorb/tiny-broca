"""Brazilian Portuguese front end: text -> letter tokens with a stress feature.

The token alphabet is the Portuguese written form itself: lowercase letters
(including the accented ones that are distinct graphemes), punctuation, and
the usual special tokens. Three decisions, documented against the English
original:

* **Characters, not phonemes.** The original front end mapped text to eSpeak
  phonemes through the ``dizione`` package (CMUdict + G2P + rules). No
  permissive Portuguese equivalent exists: eSpeak's Portuguese voice falls
  back to English phonemes for most words, and no copyleft pronouncing
  dictionary is available. Portuguese graphemes map to phones through a small,
  regular set of context rules, so a character-level front end whose encoder
  learns the context is the standard, fully permissive design.
* **The normaliser reads what a person reads aloud.** Digits, dates, times,
  money, symbols, units, ordinals, acronyms, abbreviations, roman numerals,
  URLs and common English loanwords become ordinary Portuguese words before
  tokenisation ("R$ 1.234,56", "14/03", "4 > 3", "2ª", "CPF", "online"), with
  Brazilian conventions (dot thousands separator, comma decimal, "duas horas",
  "primeiro de maio"). English loanwords are respelled the way Brazilians say
  them ("online" -> "onlaine", "shampoo" -> "xampu"), so the letter model never
  has to learn English spelling.
* **Stress from syllables.** The stress feature marks the stressed syllable
  found by a Portuguese syllabifier: an accent wins, else the oxytone endings
  (-r, -l, -z, -x, -i, -u, -im, -um, -is, -us, ...) stress the last syllable
  and everything else the penultimate; hiatus and diphthongs are split the
  Portuguese way (ra-iz, re-ais, sa-í-da, mui-to). Unstressed function words
  (de, que, se, o, a, ...) carry no stress.

Version 2 of the front end (tiny-broca v1) changed the normaliser and the
stress feature; the symbol inventory is unchanged, so version-1 checkpoints
load and fine-tune on version-2 tokens.

The forced aligner (``tiny_broca.align``) is a Portuguese character-level CTC
recogniser, so every written letter is a label it can emit; punctuation is
pause-capable and is never forced.
"""

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass

FRONTEND_VERSION = 2
NORMALIZER_VERSION = 2
SEPARATOR = "_"
FRONTENDS = ("pt",)
FRONTEND_VARIABLE = "TINY_BROCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "pt")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…", "-")

LETTERS = (
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "à", "á", "â", "ã", "ç", "é", "ê", "í", "ó", "ô", "õ", "ú", "ü", "ũ",
)
VOWELS = ("a", "e", "i", "o", "u", "à", "á", "â", "ã", "é", "ê", "í", "ó", "ô", "õ", "ú", "ü", "ũ")
# Kept names of the English front end: the set of tokens the aligner can emit.
PHONES = LETTERS
PHONE_SET = frozenset(PHONES)
VOWEL_SET = frozenset(VOWELS)
# Unstressed consonants whose frames should not be voiced (quality checks).
VOICELESS = frozenset(("p", "t", "k", "f", "s", "x"))
# Append-only: changing the order or removing a symbol invalidates checkpoints.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
LETTER_SET = frozenset(LETTERS)
STRESS_STRESSED = 1

_QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
    "–": "—", "―": "—", "‒": "—",
    "‐": "-", "‑": "-", "­": "",
    "−": "−",
})
_WORD_CHARS = LETTER_SET | {"'"}   # apostrophes stay inside words (d'água)


def frontend():
    """Name of the program that produces the tokens: ``"pt"``."""
    return _frontend


def set_frontend(name):
    """Select the front end for this process (only ``"pt"`` exists)."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def frontend_identity():
    return {"name": "pt", "version": FRONTEND_VERSION,
            "normalizer_version": NORMALIZER_VERSION, "symbols": list(SYMBOLS)}


def frontend_runtime():
    """What an annotation run records about the exact program behind its
    tokens. The normaliser is part of this package, so the identity covers it;
    the runtime records nothing extra."""
    return {"normalizer_version": NORMALIZER_VERSION}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Numbers in words (Brazilian conventions)
# ---------------------------------------------------------------------------

_UNITS = ("zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito",
          "nove", "dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis",
          "dezessete", "dezoito", "dezenove")
_TENS = ("", "dez", "vinte", "trinta", "quarenta", "cinquenta", "sessenta",
         "setenta", "oitenta", "noventa")
_HUNDREDS = ("", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
             "seiscentos", "setecentos", "oitocentos", "novecentos")
_MONTHS = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
           "agosto", "setembro", "outubro", "novembro", "dezembro")
_SCALE = ((10 ** 12, "trilhão", "trilhões"), (10 ** 9, "bilhão", "bilhões"),
          (10 ** 6, "milhão", "milhões"), (10 ** 3, "mil", "mil"))
_SCALE_WORDS = frozenset(("milhão", "milhões", "bilhão", "bilhões", "trilhão", "trilhões"))
_FEMININE_NUMBERS = {"um": "uma", "dois": "duas", "duzentos": "duzentas", "trezentos": "trezentas",
                     "quatrocentos": "quatrocentas", "quinhentos": "quinhentas", "seiscentos": "seiscentas",
                     "setecentos": "setecentas", "oitocentos": "oitocentas", "novecentos": "novecentas"}


def _below_thousand(n):
    parts = []
    if n >= 100:
        hundreds, rest = divmod(n, 100)
        parts.append("cem" if n == 100 else _HUNDREDS[hundreds])
        if rest:
            parts.append("e")
    if n % 100:
        tens, unit = divmod(n % 100, 10)
        if tens == 1:
            parts.append(_UNITS[10 + unit])     # dez, onze, ..., dezenove
        elif tens:
            parts.append(_TENS[tens])
            if unit:
                parts.append("e")
                parts.append(_UNITS[unit])
        elif unit:
            parts.append(_UNITS[unit])
    return " ".join(p for p in parts if p)


def _feminine(words):
    """Feminine agreement of a spelled number ("duzentas e uma"). The scale
    nouns milhão/bilhão are masculine, so only what follows the last of them
    agrees: "dois milhões e duzentas mil pessoas"."""
    parts = words.split(" ")
    last = max((i for i, w in enumerate(parts) if w in _SCALE_WORDS), default=-1)
    return " ".join(parts[:last + 1] + [_FEMININE_NUMBERS.get(w, w) for w in parts[last + 1:]])


def _single_round_group(n):
    while n and n % 1000 == 0:
        n //= 1000
    return n < 100 or (n < 1000 and n % 100 == 0)


def number_words(n, feminine=False):
    """Spell an integer 0..10**15 in Brazilian Portuguese."""
    n = int(abs(n))
    if n == 0:
        return "zero"
    if n < 1000:
        words = _below_thousand(n)
    elif n % 1000 == 0 and n < 10 ** 6:
        head = _below_thousand(n // 1000)
        words = "mil" if head == "um" else head + " mil"
    else:
        parts = []
        for scale, singular, plural in _SCALE:
            if n >= scale:
                count, n = divmod(n, scale)
                # Portuguese says "mil", never "um mil".
                head = "" if (count == 1 and scale == 10 ** 3) else (
                    _UNITS[count] if count == 1 else number_words(count))
                parts.append((head + " " if head else "") +
                             (singular if count == 1 else plural))
                # "e" joins a scale to what follows only when that is a single
                # group below a hundred or a round hundred: "quatro mil e noventa",
                # "mil e quinhentos", "um milhão e quinhentos mil", but "mil cento
                # e cinquenta", "um milhão duzentos e trinta e quatro mil".
                if n and _single_round_group(n):
                    parts.append("e")
        if n:
            parts.append(_below_thousand(n) if n < 1000 else number_words(n))
        words = " ".join(parts)
    return _feminine(words) if feminine else words


_ORD_UNITS = ("", "primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto",
              "sétimo", "oitavo", "nono")
_ORD_TENS = ("", "décimo", "vigésimo", "trigésimo", "quadragésimo", "quinquagésimo",
             "sexagésimo", "septuagésimo", "octogésimo", "nonagésimo")
_ORD_HUNDREDS = ("", "centésimo", "ducentésimo", "trecentésimo", "quadringentésimo",
                 "quingentésimo", "sexcentésimo", "septingentésimo", "octingentésimo",
                 "noningentésimo")
_ORDINAL_WORDS = frozenset(_ORD_UNITS + _ORD_TENS + _ORD_HUNDREDS + ("milésimo",)) - {""}


def ordinal_words(n, feminine=False):
    """Spell an ordinal: 1 -> "primeiro", 21 -> "vigésimo primeiro", 2 (f) -> "segunda"."""
    n = int(n)
    if n <= 0:
        return number_words(n)
    parts = []
    if n >= 1000:
        thousands, n = divmod(n, 1000)
        parts.append(("" if thousands == 1 else number_words(thousands) + " ") + "milésimo")
    hundreds, rest = divmod(n, 100)
    tens, unit = divmod(rest, 10)
    for table, index in ((_ORD_HUNDREDS, hundreds), (_ORD_TENS, tens), (_ORD_UNITS, unit)):
        if index:
            parts.append(table[index])
    words = " ".join(parts).split(" ")
    if feminine:
        words = [w[:-1] + "a" if w in _ORDINAL_WORDS else w for w in words]
    return " ".join(words)


def digits_words(text):
    return " ".join(_UNITS[int(c)] for c in text if c.isdigit())


def decimal_words(integer, fraction, feminine=False):
    """'3', '14' -> "três vírgula quatorze"; '3', '05' -> "três vírgula zero cinco"."""
    whole = number_words(int(integer.replace(".", "")), feminine)
    if not fraction or not fraction.strip("0"):
        return whole                                    # "150,00" -> "cento e cinquenta"
    if len(fraction) > 3 or (fraction[0] == "0" and len(fraction) > 1):
        frac = digits_words(fraction)                   # "3,05" -> "zero cinco"
    else:
        frac = number_words(int(fraction))              # "3,14" -> "quatorze"
    return f"{whole} vírgula {frac}"


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# English loanwords and brands as Brazilians say them, respelled in ordinary
# Portuguese spelling (accents mark the stress where the default rules would
# put it elsewhere). Keys are matched case-insensitively as whole words; a
# plural "-s" on the key carries over to the respelling ("sites" -> "saites").
_LOANWORDS_TSV = """
online	onlaine
tiny	táini
offline	oflaine
bitcoin	bitecóin
blockchain	bloquitchêin
shampoo	xampu
tradeoff	treidófe
trade-off	treidófe
show	xou
mouse	mause
delivery	delíveri
feedback	fidibéque
email	imêil
e-mail	imêil
site	saite
website	uebsaite
web	uébe
app	épe
wifi	uaifai
wi-fi	uaifai
iphone	aifone
ipad	aipéde
youtube	iutúbe
youtuber	iutúber
whatsapp	uotsápe
instagram	instagrã
facebook	feicebúque
twitter	tuíter
tiktok	tiquitóque
google	gúgou
gmail	gemêil
hotmail	rótmeil
outlook	autlúque
yahoo	iarrú
uber	úber
netflix	nétiflix
spotify	ispótifai
amazon	ámazon
microsoft	maicrossófte
apple	épou
samsung	sânsungue
software	sóftuer
hardware	rárduer
download	daunlôude
upload	aplôude
login	lóguin
logout	logáute
link	linque
streaming	istrímin
stream	istrime
streamer	istrímer
startup	startápe
marketing	márquetin
home office	rôme ófice
design	dizáine
designer	dizáiner
notebook	nôutibuque
laptop	léptope
smartphone	ismartifone
smartwatch	ismartuótche
tablet	táblete
podcast	pódcaste
playlist	pleilíste
deadline	dédilaine
freelancer	frilâncer
freela	frila
fast food	fést fúde
shopping	xópin
outdoor	autidór
layout	leiáute
hacker	réquer
crush	crãxe
spoiler	ispóiler
like	láique
story	istóri
stories	istóris
live	láive
game	gueime
gamer	guêimer
videogame	videogueime
nerd	nérde
geek	guíque
rock	róque
pop	pópe
jazz	djéz
blues	blus
funk	fanque
rap	répe
rapper	réper
hip hop	ripe rópe
hip-hop	ripe rópe
selfie	sélfie
hashtag	rexitégue
post	pôste
fake	feique
fake news	feique niús
news	niús
check-in	xequín
checkin	xequín
check-out	xecáute
checkout	xecáute
know-how	nourráu
happy hour	répi áuer
black friday	bléque fráidei
halloween	ralouín
bluetooth	blutúfe
pen drive	pen dráive
pendrive	pendráive
backup	becápe
bug	bágue
debug	dibágue
deploy	diplói
framework	freimuórque
frontend	frontiénde
front-end	frontiénde
backend	bequiénde
back-end	bequiénde
coworking	couórquin
coach	coutche
coaching	coutchin
mindset	maindséte
insight	insáite
networking	netuórquin
brainstorm	breinstórme
briefing	brífin
budget	bádjete
business	bíznes
branding	bréndin
e-commerce	icomérce
ecommerce	icomérce
outlet	áutlete
off	ófe
fitness	fítines
crossfit	crósfite
personal trainer	pérsonal trêiner
trainer	trêiner
skincare	isquinquér
look	lúque
outfit	áutfite
jeans	djins
short	xórte
shorts	xórtes
cropped	crópede
blazer	blêizer
hot dog	róte dógue
cheeseburger	xisbúrguer
milkshake	milquexeique
brownie	bráuni
cupcake	cápqueique
cookie	cúqui
bacon	bêicon
ketchup	quétchupe
drink	drinque
self-service	sélfe sérvice
pet shop	péte xópe
petshop	petexópe
pet	péte
playground	pleigráunde
camping	câmpin
trailer	trêiler
okay	oquêi
ok	oquêi
bye	bai
sorry	sóri
wow	uau
nice	náice
cool	cul
top	tópe
vibe	váibe
date	deite
match	métche
ranking	rânquin
level	lével
upgrade	apigreide
update	apdeite
reset	risséte
chat	xéte
chatbot	xetebóte
bot	bóte
spam	ispãme
influencer	influêncer
blogger	blóguer
blog	blógue
internet	internéte
touch	tátche
zoom	zum
meet	mite
call	cól
meeting	mítin
follow-up	folouápe
job	djóbe
trainee	treiní
fintech	fintéque
ethereum	etérium
crypto	crípto
token	tôquen
wallet	uólete
trader	trêider
trade	treide
trading	trêidin
big data	bigue dêita
machine learning	maxíne lérnin
cloud	cláude
gadget	gádjete
headset	redséte
headphone	redfone
player	plêier
playstation	pleistêixon
xbox	équisbóx
windows	uíndous
linux	línux
android	andróide
ios	ai ó ésse
macbook	mequibúque
chrome	crôume
firefox	fairefóx
chatgpt	xéte gê pê tê
openai	oupen ei ai
linkedin	línquedin
telegram	telegrã
pinterest	pínterest
reddit	rédite
twitch	tuítche
discord	discórde
kindle	quíndou
wikipedia	uiquipédia
waze	uêize
airbnb	érbienbi
ifood	aifúde
nubank	nubanque
picpay	piquepêi
shopee	xopí
shein	xiín
aliexpress	aliesprés
burger king	búrguer quim
mcdonald's	mequidônalds
mcdonalds	mequidônalds
starbucks	istarbáques
subway	sábuei
nike	náique
disney	dísnei
marvel	márvel
star wars	istar uórs
harry potter	rári póter
new york	niu iórque
los angeles	lós ândjeles
washington	uóxinton
hollywood	rólivude
wall street	uól istríte
big brother	bigue bróder
masterchef	máster xéfe
reality show	riáliti xou
stand-up	istendápe
talk show	tóque xou
fair play	fér plêi
stress	istrésse
lobby	lóbi
merchandising	merxandáizin
showroom	xourrum
spray	ispréi
jet ski	djéte isqui
skate	isqueite
playoff	pleiófe
round	ráunde
cover	câver
hit	rite
single	síngou
remix	rimíx
playback	pleibéque
cosplay	cosplêi
emoji	emôji
print	prínte
bullying	búlin
stalker	istólquer
hater	rêiter
k-pop	queipópe
kpop	queipópe
code	côude
feed	fíde
freezer	frízer
cashback	quéxibéque
call center	cól cênter
webcam	uebicã
drive	dráive
smart	ismarte
e-book	ibúque
ebook	ibúque
chip	xípe
nugget	nâguete
brunch	brãnche
muffin	mâfin
pancake	penqueique
waffle	uófou
donut	dônut
cheesecake	xisqueique
food truck	fúde trâque
drive-thru	dráive tru
petrobras	petrobrás
eletrobras	eletrobrás
detran	detrã
procon	procôm
enem	enêm
selic	selíque
samu	sâmu
unicamp	unicâmpi
unesp	unéspi
fuvest	fuvéste
sebrae	sebráe
senac	senáque
anatel	anatél
aneel	aneél
embratel	embratél
bovespa	bovéspa
infraero	infraéro
covid	côvide
on-line	onlaine
kubernetes	cubernétis
docker	dóquer
overfitting	overfítin
underfitting	anderfítin
database	dêitabeice
dataset	dêitasséte
data center	dêita cênter
data science	dêita sáience
data lake	dêita lêique
deep learning	dípe lérnin
fine-tuning	fáine túnin
fine tuning	fáine túnin
pipeline	paipláine
python	páiton
javascript	javascrípte
typescript	taipiscrípte
github	guíte râbe
gitlab	guíte lébe
git	guíte
commit	comíte
merge	mérgi
pull request	pul riquéste
branch	bréntche
dev	déve
devops	devópis
full stack	ful istéque
fullstack	ful istéque
full-stack	ful istéque
script	iscrípte
prompt	prômpite
plugin	plúguin
widget	uídjete
driver	dráiver
firmware	fírmuer
malware	máluer
spyware	ispáiuer
ransomware	rânsomuer
phishing	fíxin
firewall	fáieruól
proxy	próquici
server	sérver
host	rôste
hosting	rôstin
cache	quéxe
cluster	clâster
container	contêiner
endpoint	endipóinte
payload	peilôude
query	quéri
runtime	rantáime
stack	istéque
thread	tréde
array	arrêi
string	istríngue
loop	lúpe
bit	bíte
byte	báite
kilobyte	quilobáite
megabyte	megabáite
gigabyte	gigabáite
terabyte	terabáite
megabit	megabíte
gigabit	gigabíte
pixel	píquicel
transformer	transfórmer
embedding	embédin
benchmark	bêntchimarque
dashboard	déxibórde
scrum	iscrâm
sprint	isprínte
kanban	cânban
roadmap	rôudemépe
backlog	béquelógue
workflow	uórqueflou
onboarding	onbórdin
feature	fítcher
release	rilíce
rollback	rólbéque
patch	pétche
password	pássuorde
username	iúzerneime
browser	bráuzer
touchscreen	tátchiscrín
router	ráuter
vlog	vlógue
unboxing	anbóquicin
review	riviú
gameplay	gueimeplêi
lag	légue
ping	píngue
noob	núbe
skin	isquín
steam	istím
desktop	déskitópe
wireless	uáierles
power bank	páuer bénque
screenshot	iscrínxóte
scanner	iscâner
inbox	ínbox
downgrade	dáungreide
restart	ristárte
setup	setápe
crash	créxe
zap	zápe
joystick	djóistique
arcade	arquêide
mousepad	máusepéde
display	displêi
blu-ray	blu rêi
bluray	blu rêi
flash	fléxe
log	lógue
kernel	kérnel
unicode	iunicôude
assembly	assémbli
debian	débian
saas	sás
ram	râmi
mint	mínte
signal	síguinal
copilot	copáilote
gemini	djémini
claude	clôde
anthropic	antrópique
deepseek	dípe síque
midjourney	mídi djârni
nvidia	envídia
powerpoint	páuer póinte
office	ófice
word	uórde
photoshop	fotoxópe
teams	tíns
slack	isléque
notion	nôuxon
skype	iscáipe
tinder	tínder
snapchat	isnépixéte
spacex	ispêicex
oracle	óracou
adobe	adôbi
paypal	peipáu
stripe	istráipe
rappi	rápi
booking	búquin
tripadvisor	tripadváizor
prime	práime
prime video	práime vídeo
deezer	dízer
soundcloud	sáunde cláude
walmart	uólmarte
ikea	iquéia
reebok	ribóque
gucci	gútchi
louis vuitton	luí vitôn
rolex	rólex
porsche	pórche
audi	áudi
volkswagen	volquisváguen
ford	fórde
chevrolet	chevrolé
honda	rônda
toyota	toióta
hyundai	riundái
nissan	níssan
jeep	djípe
renault	renô
peugeot	pejô
citroën	citroén
citroen	citroén
mitsubishi	mitsubíxi
lamborghini	lamborguíni
fiat	fiáte
pepsi	pépsi
heineken	ráineken
budweiser	budváizer
nutella	nutéla
oreo	ório
pizza hut	pítsa râte
domino's	dóminos
outback	autibéque
bob's	bóbis
sony	sôni
philips	fílips
dell	déu
acer	êicer
xiaomi	xiaomí
huawei	ruauêi
nokia	nóquia
fortnite	fórtenaite
minecraft	máinecrafte
roblox	róblox
valorant	válorante
free fire	fri fáier
league of legends	líg ofe légendes
counter-strike	cáunter istráique
counter strike	cáunter istráique
mastercard	máster cárde
american express	américan esprés
vimeo	vímeo
bing	bíngue
yandex	iândex
trump	trâmpi
biden	báiden
elon	ílon
musk	mâsque
bill gates	bíu gueites
steve jobs	istívi djóbis
zuckerberg	zúquerbergue
shakespeare	xéquispir
beatles	bítous
michael	máicou
jackson	djéquisson
john	djón
james	djêimes
george	djórdji
justin	djâstin
bieber	bíber
kevin	quévin
kelly	quéli
william	uílian
willian	uílian
wesley	uésli
wilson	uílson
wagner	vágner
chris	crís
donald	dônald
tony	tôni
henry	rênri
peter	píter
robert	róberte
oliver	óliver
taylor swift	têilor suífte
beyoncé	bionsê
beyonce	bionsê
madonna	madôna
rihanna	riána
lady gaga	lêidi gága
drake	drêique
eminem	êminem
buffett	bâfete
parkinson	párquinson
eiffel	êifel
seattle	siátou
tennessee	tenessí
las vegas	lás végas
vegas	végas
miami	maiâmi
boston	bóston
texas	téquissas
manhattan	manréten
brooklyn	brúclin
silicon valley	sílicon váli
times square	táimis iscuér
taiwan	taiuã
york	iórque
stablecoin	istêibou cóin
altcoin	áute cóin
dogecoin	dógui cóin
staking	istêiquin
day trade	dêi treide
day trader	dêi trêider
swing trade	suíngue treide
hedge	rédji
hedge fund	rédji fânde
spread	ispréde
stop loss	istópe lós
private equity	práivete écuiti
venture capital	vêntcher cápital
valuation	valuêixon
equity	écuiti
cash	quéxe
cash flow	quéxe flôu
payback	peibéque
breakeven	breiquíven
funding	fândin
compliance	complâience
rating	rêitin
holding	rôldin
leasing	lízin
franchising	frenxáizin
factoring	fáctorin
dumping	dâmpin
lockdown	loquedáun
headhunter	réde rânter
board	bórde
stakeholder	isteiquirrôlder
performance	perfórmance
telemarketing	telemárquetin
marketplace	márquetepleice
dropshipping	dropxípin
lead	líde
target	tárguete
pitch	pítche
burnout	bârnáute
commodity	comóditi
commodities	comóditis
outsourcing	autsórcin
offshore	ofxór
overbooking	ôuverbúquin
cyber monday	sáiber mândei
voucher	váucher
manager	mânadjer
yuan	iuã
impeachment	impítchimente
bike	báique
e-bike	ibáique
surf	sârfe
windsurf	uínde sârfe
kitesurf	cáite sârfe
wakeboard	uêique bórde
snowboard	isnôu bórde
snowboarder	isnôu bórder
snowboarding	isnôu bórdin
trekking	trékin
rafting	ráftin
parkour	parcúr
jiu-jitsu	jiujítsu
jiu jitsu	jiujítsu
pit stop	píte istópe
pole position	pôu posíxon
rugby	rúgbi
golf	gólfe
frisbee	frísbi
paintball	pêintból
sport	ispórte
spinning	ispínin
personal	pérsonal
fit	fíte
whey	uêi
whey protein	uêi prôutin
low carb	lôu cárbe
vegan	végan
peeling	pílin
piercing	pírcin
tattoo	tatu
yoga	ióga
spa	ispá
legging	léguin
blush	blâxe
gloss	glós
primer	práimer
oversized	ôuversáized
fashion	féxon
plus size	plâs sáize
plus	plâs
make	mêique
air fryer	ér fráier
airfryer	ér fráier
closet	clósete
loft	lófte
flat	fléte
duplex	dúplex
triplex	tríplex
hall	ról
home	rôme
home theater	rôme tíater
home care	rôme quér
deck	déque
resort	rizórte
hostel	rôstel
tour	túr
city tour	síti túr
city	síti
jet lag	djéte légue
free shop	fri xópe
duty free	diúti fri
jet-ski	djéte isqui
cooler	cúler
blackout	blecáute
laser	lêizer
scooter	iscúter
pickup	picápe
pick-up	picápe
hatch	rétchi
diesel	dísel
airbag	érbégue
motorhome	motor rôme
motocross	motocrós
minivan	minivã
test drive	tést dráive
check-up	xecápe
checkup	xecápe
stand	istênde
stand-by	istendibái
standby	istendibái
shop	xópe
barber shop	bárber xópe
store	istór
play	plêi
stop	istópe
pop-up	popápe
popup	popápe
motoboy	motobói
office boy	ófice bói
cowboy	caubói
playboy	pleibói
boy	bói
baby	bêibi
baby shower	bêibi xáuer
teen	tín
kids	quídis
miss	mís
love	lâve
gay	guêi
sexy	séqui
drag	dréque
drag queen	dréque quín
nude	núde
free	fri
full	ful
full time	ful táime
part-time	párti táime
timing	táimin
boss	bós
light	láite
lite	láite
diet	dáiete
premium	prêmium
vintage	víntidji
dealer	díler
barman	bármen
bartender	bartênder
close	clôuse
kit	quíte
status	istátus
banner	bâner
boom	bum
tag	tégue
the	dâ
voice	vóice
music	miúzique
street	istríte
studio	istúdio
graffiti	grafíti
hipster	rípster
indie	índi
junior	júnior
matrix	mêitrix
maps	mépes
walkman	uóquimen
pub	pâbe
cheerleader	tchirlíder
black	bléque
power	páuer
green card	grín cárde
big tech	bigue téque
tech	téque
center	cênter
shopping center	xópin cênter
self service	sélfe sérvice
fast-food	fést fúde
milk-shake	milquexeique
hot-dog	róte dógue
set	séte
ferry	férri
remake	rimêique
reboot	ribúte
sitcom	sítcon
blockbuster	blóquebâster
best-seller	bést séler
best seller	bést séler
heavy metal	révi métau
punk	pânque
reggae	régue
soul	sôu
gospel	góspel
trap	trépe
folk	fóuque
country	cântri
backstage	béquistêidji
fandom	féndom
hype	ráipe
cringe	crínji
flop	flópe
shippar	xipár
stalkear	istolquiár
hackear	requiár
hackeou	requiôu
hackeado	requiádo
hackeada	requiáda
hackearam	requiáram
hackeando	requiândo
hacking	réquin
tweet	tuíte
retweet	rituíte
reels	ríus
trend	trénde
trending	tréndin
gif	guífe
mood	múde
random	rândom
bad	béde
chill	tchíl
feat	fíte
beat	bíte
groove	grúve
swing	suíngue
chef	xéfe
buffet	bufê
gourmet	gurmê
reveillon	reveiôn
réveillon	reveiôn
champagne	xampânhe
chantilly	xantilí
croissant	croassã
fondue	fondí
petit gateau	petí gatô
mousse	músse
bouquet	buquê
boutique	butíque
lingerie	lanjerrí
necessaire	necessér
nécessaire	necessér
déjà vu	dejavú
deja vu	dejavú
ballet	balé
cappuccino	capuchíno
espresso	esprésso
macchiato	maquiáto
latte	láte
smoothie	ismúdi
whisky	uísque
whiskey	uísque
gin	djím
chopp	xópe
cheddar	xédar
sushi	súxi
sashimi	saxími
temaki	temáqui
yakisoba	iaquissóba
ramen	râmen
kebab	quebábe
nachos	nátchos
pickles	píquous
snack	isnéque
shoyu	xôiu
wasabi	uassábi
gyoza	guiôza
poke	pôque
bowl	bôul
cream cheese	crím xís
catchup	quétchupe
barbecue	barbequiú
roast beef	rôste bífe
wrap	répe
bagel	bêiguel
panini	paníni
risotto	rizóto
spaghetti	espaguéte
gnocchi	nhóqui
bruschetta	brusquéta
tiramisu	tiramissú
prosecco	proséco
mozzarella	mussaréla
curry	cúrri
hamburguer	hambúrguer
funghi	fúngui
"""

# Acronyms, matched case-sensitively in upper case (a lower-case "s" plural is
# allowed: "ONGs", "CDs"). Unlisted acronyms are spelled letter by letter, or
# read as a word when Portuguese can pronounce them (NASA, FIFA).
_ACRONYMS_TSV = """
CPF	cê pê éfe
CEP	cépe
RG	érre gê
CNH	cê ene agá
CNPJ	cê ene pê jota
INSS	i ene ésse ésse
FGTS	éfe gê tê ésse
SUS	sus
PIX	pix
PIB	píbi
IBGE	i bê gê é
ONU	ônu
ONG	ôngue
ENEM	enêm
USP	úspi
UNICAMP	unicâmpi
PUC	púqui
MEC	méqui
UFRJ	u éfe érre jota
UFMG	u éfe eme gê
UNESP	unéspi
FGV	éfe gê vê
STF	ésse tê éfe
STJ	ésse tê jota
TSE	tê ésse é
TCU	tê cê u
PF	pê éfe
PM	pê eme
PRF	pê érre éfe
OAB	ó á bê
CBF	cê bê éfe
EUA	estados unidos
UE	união europeia
TV	tê vê
DVD	dê vê dê
CD	cê dê
PC	pê cê
HD	agá dê
SSD	ésse ésse dê
USB	u ésse bê
PDF	pê dê éfe
GPS	gê pê ésse
LED	léde
DNA	dê ene á
HIV	agá i vê
AIDS	áidis
COVID	côvide
UTI	u tê i
SAMU	sâmu
SESC	sésque
SENAI	senai
SENAC	senáque
SEBRAE	sebráe
IPVA	i pê vê á
IPTU	i pê tê u
IR	i érre
IOF	i ó éfe
ICMS	i cê eme ésse
MEI	mei
CLT	cê éle tê
PJ	pê jota
RH	érre agá
TI	tê i
IA	i á
CEO	ci ai ôu
API	á pê i
URL	u érre éle
HTML	agá tê eme éle
SQL	ésse quê éle
RAM	râmi
CPU	cê pê u
GPU	gê pê u
HDMI	agá dê eme i
QR	quê érre
SMS	ésse eme ésse
LGPD	éle gê pê dê
FAQ	éfe á quê
VIP	vípe
OK	oquêi
RS	érre ésse
SP	ésse pê
RJ	érre jota
MG	eme gê
DF	dê éfe
BR	bê érre
NBA	ene bê á
UFC	u éfe cê
MMA	eme eme á
BMW	bê eme dáblio
HP	agá pê
IBM	i bê eme
LG	éle gê
MBA	eme bê á
PHD	pê agá dê
TCC	tê cê cê
FUVEST	fuvéste
OMS	ó eme ésse
G20	gê vinte
MPB	eme pê bê
FM	éfe eme
ID	i dê
OTAN	otã
UNICEF	unicéfe
UNESCO	unésco
ANVISA	anvisa
ANATEL	anatél
DETRAN	detrã
PROCON	procôm
IBAMA	ibama
FUNAI	funai
EMBRAPA	embrapa
PETROBRAS	petrobrás
BNDES	bê ene dê é ésse
BC	bê cê
IPCA	i pê cê á
SELIC	selíque
CDI	cê dê i
CDB	cê dê bê
NFT	ene éfe tê
CPI	cê pê i
PEC	péque
DJ	didjêi
MP3	eme pê três
MP4	eme pê quatro
"""

_ABBREVIATIONS = (
    # (pattern, reading); patterns are case-insensitive and end at a word edge.
    (r"\bsras?\.", lambda m: "senhoras" if m.group(0).lower().startswith("sras") else "senhora"),
    (r"\bsrs?\.", lambda m: "senhores" if m.group(0).lower().startswith("srs") else "senhor"),
    (r"\bsrta\.", "senhorita"),
    (r"\bdras?\.", lambda m: "doutoras" if m.group(0).lower().startswith("dras") else "doutora"),
    (r"\bdrs?\.", lambda m: "doutores" if m.group(0).lower().startswith("drs") else "doutor"),
    (r"\bprofa\.", "professora"),
    (r"\bprof\.", "professor"),
    (r"\bav\.", "avenida"),
    (r"\bal\.(?=\s[A-ZÀ-Ú])", "alameda"),
    (r"\bpça\.", "praça"),
    (r"\bn[º°]\s?\.?|\bn\.\s?[º°]|\bnúm\.", "número "),
    (r"\btel\.", "telefone"),
    (r"\bcel\.", "celular"),
    (r"\betc\b\.(?=\s*$|\s+[A-ZÀ-Ú])", "etcétera."),
    (r"\betc\b\.?", "etcétera"),
    (r"\bséc\.", "século"),
    (r"(?-i:\bD)\.(?=\s(?-i:[A-ZÀ-Ú]))", "dom"),
    (r"\bp\.\s?ex\.", "por exemplo"),
    (r"\bvs\.", "versus"),
    (r"\baprox\.", "aproximadamente"),
    (r"\bobs\.", "observação"),
    (r"\bpágs?\.", lambda m: "páginas" if m.group(0).lower().startswith("págs") else "página"),
    (r"\bpg\.", "página"),
    (r"\bcap\.", "capítulo"),
    (r"\bart\.", "artigo"),
    (r"\bvol\.", "volume"),
    (r"\bltda\b\.?", "limitada"),
    (r"\bcia\.", "companhia"),
    (r"\bdepto\.", "departamento"),
    (r"\ba\.\s?c\.", "antes de cristo"),
    (r"\bd\.\s?c\.", "depois de cristo"),
    (r"\bsto\.", "santo"),
    (r"\bsta\.", "santa"),
    (r"\bexmo\.", "excelentíssimo"),
    (r"\bexma\.", "excelentíssima"),
    (r"\bfeat\.", "fíte"),
    (r"\bc/(?=\s)", "com"),
    (r"\bs/(?=\s)", "sem"),
    (r"\bp/(?=\s)", "para"),
)

# Chat abbreviations, matched as whole lower-case words.
_SLANG = {
    "vc": "você", "vcs": "vocês", "tb": "também", "tbm": "também", "pq": "porque",
    "q": "que", "blz": "beleza", "vlw": "valeu", "obg": "obrigado", "obgd": "obrigado",
    "msg": "mensagem", "hj": "hoje", "td": "tudo", "tds": "todos", "cmg": "comigo",
    "ctz": "certeza", "mt": "muito", "mto": "muito", "mts": "muitos", "qdo": "quando",
    "qnd": "quando", "dps": "depois", "agr": "agora", "fds": "fim de semana",
    "pfv": "por favor", "pfvr": "por favor", "sdd": "saudade", "sdds": "saudades",
    "bjs": "beijos", "bj": "beijo", "abs": "abraços", "kd": "cadê", "tmj": "tamo junto",
    "add": "adicionar", "aff": "afe", "rs": "risos", "rsrs": "risos", "rsrsrs": "risos",
}

_LETTER_NAMES = {
    "a": "á", "b": "bê", "c": "cê", "d": "dê", "e": "é", "f": "éfe", "g": "gê", "h": "agá",
    "i": "i", "j": "jota", "k": "cá", "l": "éle", "m": "eme", "n": "ene", "o": "ó", "p": "pê",
    "q": "quê", "r": "érre", "s": "ésse", "t": "tê", "u": "u", "v": "vê", "w": "dáblio",
    "x": "xis", "y": "ípsilon", "z": "zê",
}


def _read_tsv(source):
    table = {}
    for line in source.strip().splitlines():
        key, value = line.split("\t")
        table[key] = value
    return table


LOANWORDS = {k.lower(): v for k, v in _read_tsv(_LOANWORDS_TSV).items()}
ACRONYMS = _read_tsv(_ACRONYMS_TSV)

# Nouns that make a number feminine ("duas horas", "duzentas pessoas"); other
# nouns ending in -a are feminine too unless listed in _MASCULINE_A.
_FEMININE_NOUNS = frozenset("""
vez vezes mulher mulheres mãe mães colher colheres flor flores cor cores noite noites
tarde tardes parte partes classe classes fase fases frase frases base bases chave chaves
rede redes série séries espécie espécies fonte fontes lei leis mão mãos nuvem nuvens
ordem ordens imagem imagens viagem viagens mensagem mensagens garagem garagens
passagem passagens vantagem vantagens margem margens origem origens dor dores
gente pele peles neve febre sede sedes torre torres ponte pontes morte mortes
sorte arte artes face faces fé paz luz luzes voz vozes cruz cruzes vagem
libra libras gema gemas bala balas unidade unidades cidade cidades semana semanas
árvore árvores foto fotos moto motos questão questões região regiões razão razões semente sementes
raiz raízes grade grades tribo tribos parede paredes rede redes neve fome noção
""".split())
_MASCULINE_A = frozenset("""
dia dias problema problemas sistema sistemas tema temas programa programas mapa mapas
clima climas idioma idiomas planeta planetas cometa cometas poeta poetas telefonema
dilema dilemas esquema esquemas drama dramas diploma diplomas fantasma fantasmas
sofá sofás pijama pijamas cinema cinemas grama gramas quilograma quilogramas
miligrama miligramas aroma aromas delta alfa beta gama pirata piratas guarda guardas
atleta atletas motorista motoristas jornalista jornalistas artista artistas dentista
dentistas policial policiais papa papas profeta profetas monarca monarcas diadema
teorema teoremas trema emblema emblemas enigma enigmas paradigma paradigmas poema poemas
telefonemas coração corações
""".split())
_FEMININE_ENDINGS = ("ção", "ções", "são", "sões", "dade", "dades", "gem", "gens", "ice", "ices",
                     "eza", "ezas", "ura", "uras", "ância", "âncias", "ência", "ências", "ise", "ises",
                     "tude", "tudes", "triz", "trizes")


def _is_feminine(word):
    word = word.lower()
    if word in _FEMININE_NOUNS:
        return True
    if word in _MASCULINE_A:
        return False
    if word.endswith(_FEMININE_ENDINGS):
        return True
    return word.endswith(("a", "as", "ã", "ãs"))


# ---------------------------------------------------------------------------
# Text normalisation: what is written -> what is said
# ---------------------------------------------------------------------------

_NUMBER = r"\d{1,3}(?:\.\d{3})+|\d+"
_UPPER = "A-ZÀÁÂÃÇÉÊÍÓÔÕÚÜ"
_LOWER = "a-zàáâãçéêíóôõúü"

_ROMAN = re.compile(r"\b(?=[MDCLXVI])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_CARDINAL_CONTEXT = frozenset(("século", "séculos", "capítulo", "capítulos", "volume", "parte",
                                     "ato", "livro", "tomo", "fase", "título", "anexo", "nível",
                                     "classe", "seção", "bloco", "módulo", "temporada", "edição"))
_ROMAN_ARTICLE_CONTEXT = frozenset(("artigo", "inciso", "parágrafo"))
_QUEEN_NAMES = frozenset(("isabel", "elizabeth", "elisabete", "maria", "vitória", "catarina", "cristina", "leonor",
                          "beatriz", "joana", "teresa", "carlota", "ana", "margarida"))
_REGNAL_NAMES = _QUEEN_NAMES | frozenset("""
pedro joão paulo bento pio leão luís luiz henrique carlos elizabeth elisabete isabel fernando
felipe filipe afonso manuel manoel jorge eduardo guilherme napoleão ricardo gregório inocêncio
clemente urbano alexandre constantino frederico francisco josé maria sisto nicolau bonifácio
""".split())

_MONEY = re.compile(r"(?<![\w$])([-−])?\s?(R\$|US\$|U\$|€|£|\$)\s?([-−])?(" + _NUMBER + r")(?:,(\d{1,2}))?(?![\d.,]\d)"
                    r"(?:\s?(mil|milhão|milhao|milhões|milhoes|mi|bilhão|bilhao|bilhões|bilhoes|bi|tri|trilhão|trilhões)\b)?",
                    re.IGNORECASE)
_CURRENCIES = {"r$": ("real", "reais", False), "us$": ("dólar", "dólares", False),
               "u$": ("dólar", "dólares", False), "$": ("dólar", "dólares", False),
               "€": ("euro", "euros", False), "£": ("libra", "libras", True)}
_DATE_FULL = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4}|\d{2})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_MONTH_YEAR = re.compile(r"(?<![\d/])(\d{1,2})/(\d{4})\b")
_DATE_PARTIAL = re.compile(r"(?<![\d/.,])(\d{1,2})/(\d{1,2})(?![\d/]|,\d)")
_FIRST_OF = re.compile(r"\b(?:1|1º|1°)(?=\s+de\s+(?:" + "|".join(_MONTHS) + r")\b)", re.IGNORECASE)
_DATE_CONTEXT = frozenset(("dia", "dias", "em", "até", "desde", "no", "entre", "partir", "data", "datas", "vencimento", "prazo"))
_TIME_HMS = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
_TIME_HM = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s?(?:h|hs|hrs)\b)?")
_CLOCK = re.compile(r"\b(\d{1,2})\s?h\s?(\d{2})(?:\s?min)?\b", re.IGNORECASE)
_HOURS = re.compile(r"\b(\d{1,2})\s?(?:h|hs|hrs|hr)(?![\w²³])", re.IGNORECASE)
_ORDINAL_SIGN = re.compile(r"\b(\d{1,3}(?:\.\d{3})+|\d{1,4})\s?\.?\s?([ºª])(s?)")
_NUMBER_SIGN = re.compile(r"\b[nN]\s?\.?\s?[º°]\s?\.?\s?(?=\d)")
_ORDINAL_ASCII = re.compile(r"\b(\d{1,3})(o|a|os|as)\b")
_ORDINAL_DEGREE = re.compile(r"\b(\d{1,3})\s?°(?=\s+(?:lugar|andar|ano|anos|dia|turno|colocado|colocada|série|período|"
                             r"semestre|bimestre|trimestre|tempo|filho|filha|neto|aniversário|encontro|capítulo|"
                             r"volume|mandato|batalhão|distrito|grau|time|vez|edição|round|jogo|gol)\b)")
_PERCENT = re.compile(r"(?<![\w,.])([-−])?(" + _NUMBER + r")(?:,(\d+))?\s?%")
_DEGREES = re.compile(r"(?<![\w,.])([-−])?(" + _NUMBER + r")(?:,(\d+))?\s?°\s?([CF])?(?![\w])")
_PHONE = re.compile(r"(\+\s?55\s?)?\(\s?(\d{2})\s?\)\s?(9\s?\d{4}|\d{4})[-\s.]?(\d{4})\b|"
                    r"(\+\s?55\s?)?\b(\d{2})\s(9\d{4}|\d{4})-(\d{4})\b")
_PHONE_LOCAL = re.compile(r"(?<![\d-])(9?\d{4})-(\d{4})(?![\d-])")
_TOLL_FREE = re.compile(r"\b(0[38]00)[\s-]?(\d{3})[\s-]?(\d{4})\b")
_COUNTRY_CODE = re.compile(r"(?<![\w])\+(\d{1,3})(?=\s?\(?\d)")
_CEP = re.compile(r"\b(\d{5})-(\d{3})\b")
_CNPJ = re.compile(r"\b(\d{2})\.(\d{3})\.(\d{3})/(\d{4})-(\d{2})\b")
_GROUPS = re.compile(r"\b\d{1,3}(?:[.-]\d{1,3}){2,}\b")     # CPF, CNPJ-like, CEP
_PLATE = re.compile(r"\b([A-Z]{3})-?(\d)([A-Z0-9])(\d{2})\b")
_SCORE = re.compile(r"(\b[" + _UPPER + r"][" + _LOWER + r"]+\s+)(\d{1,2})\s?[xX]\s?(\d{1,2})(?=\s+[" + _UPPER + r"])")
_COEFFICIENT = re.compile(r"(?<![\w,.])(\d{1,3})([a-zA-Z])(?![\w])")
_SPACED_SCORE = re.compile(r"(?<![\w,.])(\d{1,2})\s+[xX]\s+(\d{1,2})(?!\d|,\d|\s*(?:cm|mm|m|metros|km|pol|polegadas)\b)")
_TIMES_X = re.compile(r"(?<![\w,.])(\d{1,3})\s?[xX](?![\w\d])(?!\s*\d)")
_BY_X = re.compile(r"(?<![\w,.])(\d+)\s?[xX]\s?(\d+)(?!\d|[,.]\d)")
_RANGE = re.compile(r"(?<![\d.,/:-])(\d{1,4})\s?[-—]\s?(\d{1,4})(?!\d|[.,/:-]\d)")
_NEGATIVE = re.compile(r"(?<![\w\d])[-−](?=\d)")
_DECIMAL = re.compile(r"(?<![\w,.])(" + _NUMBER + r"),(\d{1,6})(?!\d)")
_DOT_DECIMAL = re.compile(r"(?<![\w.])(\d+)\.(\d{1,2})(?!\d|\.\d)")
_INTEGER = re.compile(r"(?<![\w,])(" + _NUMBER + r")(?![\w]|[.,]\d)")
_FRACTIONS = {2: ("meio", "meios"), 3: ("terço", "terços"), 4: ("quarto", "quartos"), 5: ("quinto", "quintos"),
              6: ("sexto", "sextos"), 7: ("sétimo", "sétimos"), 8: ("oitavo", "oitavos"),
              9: ("nono", "nonos"), 10: ("décimo", "décimos"), 100: ("centésimo", "centésimos"),
              1000: ("milésimo", "milésimos")}

_UNITS_TABLE = (
    # symbol, singular, plural, feminine
    ("km/h", "quilômetro por hora", "quilômetros por hora", False),
    ("km/l", "quilômetro por litro", "quilômetros por litro", False),
    ("cm²", "centímetro quadrado", "centímetros quadrados", False),
    ("cm2", "centímetro quadrado", "centímetros quadrados", False),
    ("Mbps", "megabit por segundo", "megabits por segundo", False),
    ("Gbps", "gigabit por segundo", "gigabits por segundo", False),
    ("kbps", "quilobit por segundo", "quilobits por segundo", False),
    ("m/s", "metro por segundo", "metros por segundo", False),
    ("km²", "quilômetro quadrado", "quilômetros quadrados", False),
    ("km2", "quilômetro quadrado", "quilômetros quadrados", False),
    ("m²", "metro quadrado", "metros quadrados", False),
    ("m2", "metro quadrado", "metros quadrados", False),
    ("m³", "metro cúbico", "metros cúbicos", False),
    ("m3", "metro cúbico", "metros cúbicos", False),
    ("kWh", "quilouóte hora", "quilouótes hora", False),
    ("mAh", "miliampére hora", "miliampéres hora", False),
    ("kcal", "quilocaloria", "quilocalorias", True),
    ("cal", "caloria", "calorias", True),
    ("km", "quilômetro", "quilômetros", False),
    ("cm", "centímetro", "centímetros", False),
    ("mm", "milímetro", "milímetros", False),
    ("kg", "quilo", "quilos", False),
    ("mg", "miligrama", "miligramas", False),
    ("ml", "mililitro", "mililitros", False),
    ("mL", "mililitro", "mililitros", False),
    ("ha", "hectare", "hectares", False),
    ("min", "minuto", "minutos", False),
    ("seg", "segundo", "segundos", False),
    ("ms", "milissegundo", "milissegundos", False),
    ("GB", "gigabaite", "gigabaites", False),
    ("MB", "megabaite", "megabaites", False),
    ("KB", "quilobaite", "quilobaites", False),
    ("kB", "quilobaite", "quilobaites", False),
    ("TB", "terabaite", "terabaites", False),
    ("GHz", "gigarrértz", "gigarrértz", False),
    ("MHz", "megarrértz", "megarrértz", False),
    ("Hz", "rértz", "rértz", False),
    ("kW", "quilouóte", "quilouótes", False),
    ("W", "uóte", "uótes", False),
    ("V", "volt", "volts", False),
    ("pol", "polegada", "polegadas", True),
    ("g", "grama", "gramas", False),
    ("t", "tonelada", "toneladas", True),
    ("l", "litro", "litros", False),
    ("L", "litro", "litros", False),
    ("m", "metro", "metros", False),
    ("s", "segundo", "segundos", False),
)
_UNIT_NAMES = {symbol: (singular, plural, feminine) for symbol, singular, plural, feminine in _UNITS_TABLE}
_UNIT = re.compile(r"(?<![\w,.])([-−])?(" + _NUMBER + r")(?:,(\d+))?\s?(" +
                   "|".join(re.escape(u[0]) for u in _UNITS_TABLE) + r")(?![\w²³/])")
_TECH_NUMBER = re.compile(r"\b(\d{1,3})([A-KM-UX-Z])\b")   # 4G, 3D, 4K, seat 23C (not L, V, W: units)
# Single capital letters standing alone (plano B, vitamina C, Raio X); vowels only after a naming word.
_LONE_CONSONANT = re.compile(r"(?<![\w'.])([B-DF-HJ-NP-TV-Z])(?![\w'])")
_LONE_VOWEL = re.compile(r"\b(plano|vitamina|letra|classe|tipo|grupo|nota|série|opção|item|bloco|setor|ala|portão|"
                         r"fila|turma|seção|categoria|modelo|versão|lado|prédio|torre|sala)\s+([AEIOU])(?![\w'])",
                         re.IGNORECASE)
_LETTER_HYPHEN_DIGIT = re.compile(r"(?<=[A-Za-zÀ-ú])-(?=\d)")
_LONE_HASH = re.compile(r"(?<!\S)#(?!\S)")
_LONE_STAR = re.compile(r"(?<!\d )(?<!\S)\*(?!\S)(?! \d)")
_ABOUT = re.compile(r"(?<![\w~])~\s?(?=[\dR$])")
_STREET = re.compile(r"(\b(?:[Nn]a|[Aa]|[Dd]a|[Pp]ela|[Àà]|[Dd]e|[Ee]m)\s+|^)R\.(?=\s[A-ZÀ-Ú])")
_PER = re.compile(r"\s?/\s?(mês|dia|hora|semana|ano|noite|pessoa|unidade|litro|quilo|kg|aula|mensal)\b")
_KM_MARKER = re.compile(r"(?<![\w\d])km\.?\s+(?=\d)", re.IGNORECASE)
_CONTEXT_SCORE = re.compile(r"\b(venceu|perdeu|ganhou|empatou|terminou|ficou|placar|por|foi)\s+(\d{1,2})\s?[xX]\s?(\d{1,2})\b"
                            r"(?!\s*(?:cm|mm|m|km|pol|metros?|centímetros?)\b)", re.IGNORECASE)
_DIMENSION = re.compile(r"(?<![\w,.])(\d+(?:,\d+)?)\s?[xX×]\s?(\d+(?:,\d+)?)(?:\s?[xX×]\s?(\d+(?:,\d+)?))?"
                        r"(?:\s?(cm|mm|m|km|pol)(?![\w²³/])|(?=\s+(?:metros?|centímetros?|milímetros?|quilômetros?|polegadas?)\b))")

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"\b(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:com|br|org|net|gov|edu|io|app|dev|info|me|tv|co|ai)"
                  r"(?:\.[a-z]{2})?(?:/[\w./%?=&#-]*)?(?![\w@])", re.IGNORECASE)
_HANDLE = re.compile(r"(?<![\w@])@(\w+)")
_HASHTAG = re.compile(r"(?<![\w#])#(\w+)")
_DOMAIN_WORDS = {"com": "com", "br": "bê érre", "org": "órgue", "net": "néte", "gov": "góve",
                 "edu": "édu", "io": "i ó", "app": "épe", "dev": "déve", "info": "info", "me": "mi",
                 "tv": "tê vê", "co": "cô", "ai": "ei ai", "www": "dáblio dáblio dáblio",
                 "http": "agá tê tê pê", "https": "agá tê tê pê ésse"}

_LAUGH_K = re.compile(r"\b[kK]{3,}\b")
_LAUGH_HA = re.compile(r"\b(?:[hH][aAuUeE]){2,}[hH]?\b")

_MATH = (
    (">=", " maior ou igual a "), ("<=", " menor ou igual a "), ("≥", " maior ou igual a "),
    ("≤", " menor ou igual a "), ("!=", " diferente de "), ("≠", " diferente de "), ("==", " igual a "),
    ("=", " igual a "), (">", " maior que "), ("<", " menor que "), ("±", " mais ou menos "),
    ("+", " mais "), ("×", " \u2297 "), ("÷", " dividido por "), ("^", " elevado a "),
    ("√", " raiz quadrada de "), ("≈", " aproximadamente "), ("∞", " infinito "), ("π", " pi "),
    ("−", " menos "), ("&", " e "), ("§", " parágrafo "), ("@", " arroba "),
)
_OPERATOR_CHARS = "<>=+×÷^≥≤≠±−"
_OPERAND = r"(?:\d|[a-zA-Z](?![\w']))"
_VARIABLE = re.compile(r"(?<![\w'])([a-zA-Z])(?=\s*[!" + re.escape(_OPERATOR_CHARS) + r"]+\s*" + _OPERAND + r")|"
                       r"(?<=[" + re.escape(_OPERATOR_CHARS) + r"])(\s*)([a-zA-Z])(?![\w'])")
_SPACED_MINUS = re.compile(r"(?<=[\d\w])\s+-\s+(?=[\d\w])")
_SPACED_TIMES = re.compile(r"(?<=\d)\s+[xX]\s+(?=\d)")
_STAR = re.compile(r"(?<=\d)\s?\*\s?(?=\d)")
_SLASH_DIVIDE = re.compile(r"(?<=\d)\s?/\s?(?=\d)")

_CONSONANTS_ONLY = re.compile(r"(?<![\w'])[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{2,6}(?![\w'])")
_KEEP = frozenset(LETTERS) | frozenset(PUNCTUATION) | {" "}
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.?!;:…)])")
_REPEATED_PUNCT = re.compile(r"([?!])[?!]+|([,;:])[,;:]+")


# Words after which a number is a code read digit by digit.
_IDENTIFIER_CONTEXT = frozenset(("protocolo", "senha", "código", "pin", "ramal", "matrícula", "final", "agência",
                                 "conta", "token", "chassi", "renavam", "lote"))
_NOT_NOUNS = frozenset(("a", "o", "as", "os", "e", "ou", "de", "do", "da", "dos", "das", "em", "no", "na",
                        "por", "para", "pra", "com", "que", "se", "à", "às", "ao", "aos", "até", "mais", "menos",
                        "cada", "nada", "agora", "ainda", "fora", "embora", "quase", "apenas", "só", "já", "lá",
                        "cá", "ali", "aqui"))


_NEUTRAL_ADJECTIVES = frozenset(("grande", "grandes", "maior", "maiores", "melhor", "melhores", "pior", "piores",
                                 "enorme", "enormes", "simples", "jovem", "jovens", "principal", "principais",
                                 "importante", "importantes", "diferente", "diferentes", "excelente", "excelentes",
                                 "incrível", "incríveis", "breve", "breves", "leve", "leves", "forte", "fortes",
                                 "interessante", "interessantes", "feliz", "felizes", "verde", "verdes"))


def _next_word(text, end):
    """The noun a number counts: the next word, looking past "de", "mil",
    the second number of a span ("de 231 a 1037 cidades"), a partitive
    ("2 em cada 10 brasileiras") and a gender-neutral adjective ("22 grandes
    cidades")."""
    m = re.match(r"\s+(?:(?:a|e|ou|até)\s+[\d.,]+\s+)?(?:em\s+cada\s+[\d.,]+\s+)?(?:mil\s+)?(?:de\s+)?"
                 r"([^\W\d_]+)(?:\s+([^\W\d_]+))?", text[end:])
    if not m:
        return ""
    word = m.group(1)
    if word.lower() in _NEUTRAL_ADJECTIVES and m.group(2):
        word = m.group(2)
    if word.lower() in _NOT_NOUNS:
        return ""
    return word


def _previous_word(text, start):
    m = re.search(r"([^\W\d_]+)\W*$", text[:start])
    return m.group(1).lower() if m else ""


def _count_words(n, feminine, next_word, has_de=False):
    """A cardinal before a noun: gender agreement, and "de" after exact
    millions ("um milhão de pessoas") unless the text already has it."""
    words = number_words(n, feminine)
    if words.split(" ")[-1] in _SCALE_WORDS and next_word and next_word.lower() not in ("de", "e") and not has_de:
        words += " de"
    return words


def _money(m):
    minus = m.group(1) or m.group(3)
    symbol, integer, cents, scale = m.group(2).lower(), m.group(4), m.group(5), m.group(6)
    singular, plural, feminine = _CURRENCIES[symbol]
    value = int(integer.replace(".", ""))
    if scale:
        scale = scale.lower().replace("milhao", "milhão").replace("milhoes", "milhões").replace(
            "bilhao", "bilhão").replace("bilhoes", "bilhões")
        if scale == "mil":
            amount = value * 1000 + (int((cents + "00")[:3]) if cents else 0)
            words = number_words(amount, feminine) + " " + plural
        else:
            one = value == 1 and not cents
            scale = {"tri": "trilhão" if one else "trilhões", "bi": "bilhão" if one else "bilhões",
                     "mi": "milhão" if one else "milhões"}.get(scale, scale)
            head = decimal_words(integer, cents) if cents else number_words(value)
            words = f"{head} {scale} de {plural}"
    else:
        cents_value = int((cents + "0")[:2]) if cents else 0
        if value == 0 and cents_value:
            words = number_words(cents_value) + (" centavo" if cents_value == 1 else " centavos")
        else:
            words = number_words(value, feminine)
            if words.split(" ")[-1] in _SCALE_WORDS:
                words += " de"
            words += " " + (singular if value == 1 else plural)
            if cents_value:
                words += " e " + number_words(cents_value) + (" centavo" if cents_value == 1 else " centavos")
    return (" menos " if minus else " ") + words + " "


def _day_words(day):
    return "primeiro" if day == 1 else number_words(day)


def _date_words(day, month, year=None):
    words = f"{_day_words(day)} de {_MONTHS[month - 1]}"
    if year is not None:
        words += f" de {number_words(year)}"
    return words


def _date_full(m):
    day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return m.group(0)
    if len(year) == 2:
        year = ("20" if int(year) <= 40 else "19") + year
    return " " + _date_words(day, month, int(year)) + " "


def _date_iso(m):
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return m.group(0)
    return " " + _date_words(day, month, year) + " "


def _month_year(m):
    month, year = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return m.group(0)
    return f" {_MONTHS[month - 1]} de {number_words(year)} "


def _fraction_words(numerator, denominator, next_word=""):
    singular, plural = _FRACTIONS[denominator]
    if denominator == 2 and numerator == 1:
        return "meia" if next_word and _is_feminine(next_word) else "meio"
    return f"{number_words(numerator)} {singular if numerator == 1 else plural}"


_MEASURE_NOUNS = frozenset("""xícara xícaras copo copos colher colheres litro litros quilo quilos kg g grama gramas
hora horas dúzia dúzias pizza pizzas parte partes página páginas tablete tabletes lata latas pacote pacotes
""".split())


def _partial_date(text):
    def replace(m):
        a, b = m.group(1), m.group(2)
        day, month = int(a), int(b)
        before = _previous_word(text, m.start())
        after = text[m.end():]
        if (day, month) == (24, 7):
            return " vinte e quatro por sete "
        word = re.match(r"\s+([^\W\d_]+)", after)
        word = word.group(1).lower() if word else ""
        is_date = 1 <= month <= 12 and 1 <= day <= 31
        span = re.match(r"\s+(?:a|até|e|ao)\s+\d{1,2}/\d{1,2}", after)
        fraction_like = (month in (2, 3, 4, 8) and day < month and not a.startswith("0") and not b.startswith("0")
                         and before not in _DATE_CONTEXT and not span
                         and (word in ("de", "da", "do", "dos", "das") or word in _MEASURE_NOUNS or not is_date))
        if fraction_like or not is_date:
            if month in _FRACTIONS and day < month:
                return " " + _fraction_words(day, month, _next_word(text, m.end())) + " "
            return f" {number_words(day)} barra {number_words(month)} "
        return " " + _date_words(day, month) + " "
    return _DATE_PARTIAL.sub(replace, text)


def _clock_words(hour, minute, hours_word, duration=False):
    if not 0 <= hour <= 24 or not 0 <= minute <= 59:
        return None
    if duration:
        head = number_words(hour, feminine=True) + (" hora" if hour == 1 else " horas")
        if minute == 0:
            return head
        return head + (" e meia" if minute == 30 else f" e {number_words(minute)}")
    if hour in (0, 24):
        head = "meia noite"
    elif hour == 12:
        head = "meio dia"
    else:
        head = number_words(hour, feminine=True)
    named = hour in (0, 12, 24)
    if minute == 0:
        return head if named else head + (" hora" if hour == 1 else " horas")
    if minute == 30:
        return head + " e meia"
    if hours_word and not named:
        head += " hora" if hour == 1 else " horas"
    return f"{head} e {number_words(minute)}"


def _unit_count(n, singular, plural, feminine=False):
    return f"{number_words(n, feminine)} {singular if n == 1 else plural}"


def _time_hms(m):
    hours, minutes, seconds = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if hours > 24 or minutes > 59 or seconds > 59:
        return m.group(0)
    return (f" {_unit_count(hours, 'hora', 'horas', True)}, {_unit_count(minutes, 'minuto', 'minutos')} e "
            f"{_unit_count(seconds, 'segundo', 'segundos')} ")


def _time_hm(m):
    first, second = int(m.group(1)), int(m.group(2))
    if first > 24 and second <= 59:
        # A duration in minutes and seconds ("corri em 25:30").
        return f" {_unit_count(first, 'minuto', 'minutos')} e {_unit_count(second, 'segundo', 'segundos')} "
    words = _clock_words(first, second, True)
    return m.group(0) if words is None else " " + words + " "


_CLOCK_CONTEXT = frozenset(("às", "as", "das", "até", "desde", "pelas", "à", "a", "para", "entre"))
# Words before "1h30" that make it a duration ("tem 1h30", "em 2h15"); otherwise it is a time of day.
_DURATION_CONTEXT = frozenset(("em", "durante", "por", "dura", "durou", "duram", "leva", "levou", "levam", "tem",
                               "teve", "têm", "há", "faz", "fez", "após", "mais", "menos", "quase", "cerca", "só",
                               "apenas", "uns", "umas", "total"))


def _clocks(text):
    def replace(m):
        before = _previous_word(text, m.start())
        duration = before in _DURATION_CONTEXT or (
            before not in _CLOCK_CONTEXT
            and bool(re.match(r"\s+de\s+(?!manhã|tarde|noite|madrugada|\d)", text[m.end():])))
        words = _clock_words(int(m.group(1)), int(m.group(2)), False, duration=duration)
        return m.group(0) if words is None else " " + words + " "
    return _CLOCK.sub(replace, text)


_NAMED_HOURS = ((r"\b(?:(À)|à)s(\s+)(meio dia)", lambda m: ("Ao" if m.group(1) else "ao") + m.group(2) + m.group(3)),
                (r"\b([Àà])s(\s+)(meia noite)", r"\1\2\3"),
                (r"\b([Aa])s(\s+)(meio dia)", r"o\2\3"), (r"\b([Aa])s(\s+)(meia noite)", r"\1\2\3"),
                (r"\b([Dd])as(\s+)(meio dia)", r"\1o\2\3"), (r"\b([Dd])as(\s+)(meia noite)", r"\1a\2\3"),
                (r"\b([Pp]el)as(\s+)(meio dia)", r"\1o\2\3"), (r"\b([Pp]el)as(\s+)(meia noite)", r"\1a\2\3"))


def _hours(m):
    hour = int(m.group(1))
    if hour > 24:
        return m.group(0)
    return " " + number_words(hour, feminine=True) + (" hora " if hour == 1 else " horas ")


def _ordinal_sign(m):
    words = ordinal_words(int(m.group(1).replace(".", "")), feminine=m.group(2) == "ª")
    if m.group(3):
        words = " ".join(w + "s" if w[:-1] + "o" in _ORDINAL_WORDS or w in _ORDINAL_WORDS else w for w in words.split(" "))
    return " " + words + " "


def _ordinal_ascii(m):
    suffix = m.group(2)
    words = ordinal_words(int(m.group(1)), feminine=suffix.startswith("a"))
    if suffix.endswith("s"):
        words = " ".join(w + "s" if w in _ORDINAL_WORDS or w[:-1] + "o" in _ORDINAL_WORDS else w
                         for w in words.split(" "))
    return " " + words + " "


def _signed_number(minus, integer, fraction, feminine=False):
    words = decimal_words(integer, fraction, feminine) if fraction else number_words(int(integer.replace(".", "")), feminine)
    return ("menos " if minus else "") + words


def _percent(m):
    return " " + _signed_number(m.group(1), m.group(2), m.group(3)) + " por cento "


def _degrees(m):
    value = _signed_number(m.group(1), m.group(2), m.group(3))
    one = m.group(2) == "1" and not m.group(3)
    scale = " fahrenheit" if m.group(4) == "F" else ""
    return f" {value} {'grau' if one else 'graus'}{scale} "


def _unit(m):
    minus, integer, fraction, symbol = m.groups()
    singular, plural, feminine = _UNIT_NAMES[symbol]
    one = integer == "1" and not (fraction and fraction.strip("0"))
    return " " + _signed_number(minus, integer, fraction, feminine) + " " + (singular if one else plural) + " "


def _spell_digits_groups(*groups):
    return ", ".join(digits_words(g) for g in groups)


def _phone(m):
    country = m.group(1) or m.group(5)
    groups = [g for g in (m.groups()[1:4] if m.group(2) else m.groups()[5:8]) if g]
    area, rest = groups[0], [g.replace(" ", "") for g in groups[1:]]
    prefix = "mais cinquenta e cinco, " if country else ""
    return " " + prefix + number_words(int(area)) + ", " + _spell_digits_groups(*rest) + " "


def _phone_local(m):
    first, second = m.group(1), m.group(2)
    if len(first) == 4 and 1900 <= int(first) <= 2100 and 1900 <= int(second) <= 2100 and int(second) > int(first):
        return m.group(0)                               # a span of years, read by the range rule
    return " " + _spell_digits_groups(first, second) + " "


def _plate(m):
    letters = " ".join(_LETTER_NAMES[c.lower()] for c in m.group(1))
    middle = m.group(3)
    middle = _LETTER_NAMES[middle.lower()] if middle.isalpha() else _UNITS[int(middle)]
    return f" {letters}, {_UNITS[int(m.group(2))]} {middle} {digits_words(m.group(4))} "


def _range(m):
    first, second = int(m.group(1)), int(m.group(2))
    if second <= first:
        return m.group(0)
    return f" {number_words(first)} a {number_words(second)} "


def _roman_value(numeral):
    total = 0
    for i, c in enumerate(numeral):
        value = _ROMAN_VALUES[c]
        if i + 1 < len(numeral) and _ROMAN_VALUES[numeral[i + 1]] > value:
            total -= value
        else:
            total += value
    return total


def _roman(text):
    def replace(m):
        numeral = m.group(0)
        value = _roman_value(numeral)
        if not 0 < value < 4000:
            return numeral
        before_text = text[:m.start()]
        before = re.search(r"([^\W\d_]+)\.?\s+$", before_text)
        before = before.group(1) if before else ""
        after = re.match(r"\s+([" + _UPPER + r"][^\W\d_]*)", text[m.end():])
        if (before.lower() in _ROMAN_CARDINAL_CONTEXT or before.lower() in ("cap", "vol", "sec", "séc")
                or re.match(r"\s+de\s+(?:" + "|".join(_MONTHS) + r")\b", text[m.end():], re.IGNORECASE)):
            return number_words(value)
        if before.lower() in _ROMAN_ARTICLE_CONTEXT or before.lower() == "art":
            return ordinal_words(value) if value < 10 else number_words(value)
        if before.lower() in _REGNAL_NAMES and before[:1].isupper() and value <= 30:
            return ordinal_words(value, feminine=before.lower() in _QUEEN_NAMES) if value <= 10 else number_words(value)
        if after and (len(numeral) > 1 or before.lower() in ("", "a", "o", "na", "no", "da", "do", "pela", "pelo", "à", "ao")) \
                and value <= 100:
            return ordinal_words(value, feminine=_is_feminine(after.group(1)))
        return numeral
    return _ROMAN.sub(replace, text)


def _spell_letters(word):
    return " ".join(_LETTER_NAMES[c] for c in word.lower() if c in _LETTER_NAMES)


_CODAS = frozenset("srlnmxz")
_ONSETS = frozenset(("pr", "br", "tr", "dr", "cr", "gr", "fr", "vr", "pl", "bl", "cl", "gl", "fl", "ch", "lh", "nh"))


def _pronounceable(word):
    """Can an upper-case acronym be read as a Portuguese word (NASA, FIFA)?"""
    w = word.lower()
    if len(w) <= 2 or not any(c in "aeiou" for c in w):
        return False
    for m in re.finditer(r"[^aeiou]+", w):
        cluster = m.group(0)
        if m.start() == 0:
            ok = len(cluster) == 1 or cluster in _ONSETS
        elif m.end() == len(w):
            ok = cluster in _CODAS
        else:
            ok = (len(cluster) == 1 or cluster in _ONSETS or (len(cluster) == 2 and cluster[0] in _CODAS)
                  or (len(cluster) == 3 and cluster[0] in _CODAS and cluster[1:] in _ONSETS))
        if not ok:
            return False
    return True


def _acronym_reading(token):
    if token in ACRONYMS:
        return ACRONYMS[token]
    plural = token.endswith("s") and token[:-1] in ACRONYMS
    if plural:
        return ACRONYMS[token[:-1]] + "s"
    core = token[:-1] if token.endswith("s") and token[:-1].isupper() else token
    suffix = "s" if core != token else ""
    if any(c.isdigit() for c in core):
        parts = re.findall(r"[A-Z]+|\d+", core)
        return " ".join(_spell_letters(p) if p.isalpha() else number_words(int(p)) for p in parts) + suffix
    if _pronounceable(core):
        return core.lower() + suffix
    return _spell_letters(core) + suffix


_ACRONYM = re.compile(r"(?<![\w'])([A-Z][A-Z0-9]{1,9}s?)(?![\w'])")


def _acronyms(text):
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6 and len(text.split()) >= 3:
        # Shouted text: only listed acronyms are spelled; the rest is ordinary words.
        return _ACRONYM.sub(lambda m: (" " + ACRONYMS[m.group(1)] + " ") if m.group(1) in ACRONYMS else m.group(1), text)

    def replace(m):
        token = m.group(1)
        if token.isdigit() or (token[0].isupper() and token[1:].islower()):
            return token
        if token in _UNIT_NAMES and re.search(r"\d\s?$", text[:m.start()]):
            return token
        if not any(c.isupper() for c in token[1:]) and not token[1:].isdigit():
            return token
        return " " + _acronym_reading(token) + " "
    return _ACRONYM.sub(replace, text)


def _speak_domain_part(part):
    part_lower = part.lower()
    if part_lower in _DOMAIN_WORDS:
        return _DOMAIN_WORDS[part_lower]
    if part_lower in LOANWORDS:
        return LOANWORDS[part_lower]
    if part.isdigit():
        return number_words(int(part)) if len(part) <= 4 else digits_words(part)
    return part_lower


def _speak_address(address, separators):
    out = []
    for piece in re.split(r"([./_\-?=&#%])", address):
        if not piece:
            continue
        if piece in separators:
            out.append(separators[piece])
        else:
            out.extend(_speak_domain_part(p) for p in re.findall(r"[^\W\d_]+|\d+", piece))
    return " ".join(w for w in out if w)


_URL_SEPARATORS = {".": "ponto", "/": "barra", "_": "underline", "-": "hífen", "?": "", "=": "igual",
                   "&": "e", "#": "", "%": "por cento"}


def _email(m):
    local, domain = m.group(0).split("@", 1)
    return " " + _speak_address(local, _URL_SEPARATORS) + " arroba " + _speak_address(domain, _URL_SEPARATORS) + " "


def _url(m):
    address = re.sub(r"^https?://", "", m.group(0), flags=re.IGNORECASE)
    tail = ""
    while address and address[-1] in ".,;:!?":
        tail, address = address[-1] + tail, address[:-1]
    return " " + _speak_address(address.rstrip("/"), _URL_SEPARATORS) + " " + tail


_LOANWORD_KEYS = sorted(LOANWORDS, key=len, reverse=True)
# English adds -es only after s, x, z, ch and sh ("coaches"); anywhere else
# key + es is a Portuguese word: postes, botes, chefes, deves.
_LOANWORD = re.compile(r"(?<![\w'-])(" + "|".join(re.escape(k) for k in _LOANWORD_KEYS)
                       + r")(s|(?<=[sxz])es|(?<=[cs]h)es)?(?![\w'-])", re.IGNORECASE)


def _loanword(m):
    respelled = LOANWORDS[m.group(1).lower()]
    if m.group(2):
        respelled = respelled + ("s" if respelled[-1] in "aeiouáéíóúâêôãõ" or not respelled.endswith("s") else "")
    return respelled


def _dimension(m):
    sides = [decimal_words(*v.split(",")) if "," in v else number_words(int(v)) for v in m.groups()[:3] if v]
    unit = m.group(4)
    words = " por ".join(sides)
    return f" {words} {_UNIT_NAMES[unit][1]} " if unit else f" {words} "


_SLANG_RE = re.compile(r"(?<![\w'])(" + "|".join(map(re.escape, sorted(_SLANG, key=len, reverse=True))) + r")(?![\w'])",
                       re.IGNORECASE)


def _slang(m):
    token = m.group(1)
    if token in ACRONYMS or (len(token) == 1 and not token.islower()):
        return token
    if token.islower() or (token[0].isupper() and token[1:].islower()) or token.lower() in ("vc", "vcs"):
        reading = _SLANG[token.lower()]
        return reading[0].upper() + reading[1:] if token[0].isupper() else reading
    return token


def _integers(text):
    def replace(m):
        digits = m.group(1)
        plain = digits.replace(".", "")
        before = _previous_word(text, m.start())
        if (len(plain) > 15 or (len(plain) > 1 and plain.startswith("0")) or (len(plain) >= 10 and "." not in digits)
                or before in _IDENTIFIER_CONTEXT):
            return " " + digits_words(plain) + " "
        value = int(plain)
        next_word = _next_word(text, m.end())
        following = text[m.end():]
        time_of_day = bool(re.match(r"\s+(?:e\s+meia\s+)?da\s+(?:manhã|tarde|noite|madrugada)\b", following))
        feminine = (before in ("à", "às", "das", "pelas") or time_of_day
                    or (before in ("as", "são") and not next_word)
                    or (bool(next_word) and _is_feminine(next_word)))
        if before == "dia" and value == 1:
            return " primeiro "
        return " " + _count_words(value, feminine, next_word, bool(re.match(r"\s+de\b", following))) + " "
    return _INTEGER.sub(replace, text)


def _decimals(text):
    def replace(m):
        next_word = _next_word(text, m.end())
        return " " + decimal_words(m.group(1), m.group(2), bool(next_word) and _is_feminine(next_word)) + " "
    return _DECIMAL.sub(replace, text)


def _math(text):
    text = _STAR.sub(" \u2297 ", text)
    if "=" in text or any(op in text for op in ("<", ">", "≥", "≤", "≠")):
        text = _SPACED_TIMES.sub(" \u2297 ", text)
        text = _SPACED_MINUS.sub(" menos ", text)
        text = _SLASH_DIVIDE.sub(" dividido por ", text)
    def variable(m):
        letter = m.group(1) or m.group(3)
        # "a", "o" and "e" before a word are an article or a conjunction ("> o preço").
        if m.group(3) and letter in "aoeAOE" and re.match(r"\s+[^\W\d_]{2,}", text[m.end():]):
            return m.group(0)
        return " " + _LETTER_NAMES[letter.lower()] + " "
    text = _VARIABLE.sub(variable, text)
    for symbol, words in _MATH:
        text = text.replace(symbol, words)
    return text


def _cleanup(text):
    out = []
    for c in text:
        if c in _KEEP or c.lower() in LETTER_SET:
            out.append(c)
        elif c.isdigit():
            out.append(" " + _UNITS[int(c)] + " ")
        elif c.isalpha():
            base = unicodedata.normalize("NFKD", c)
            base = "".join(b for b in base if not unicodedata.combining(b))
            base = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ı": "i"}.get(base, base)
            out.append(base if all(b.lower() in LETTER_SET for b in base) else " ")
        elif c in "[{":
            out.append("(")
        elif c in "]}":
            out.append(")")
        elif c in "|•·":
            out.append(",")
        else:
            out.append(" ")
    return "".join(out)


def spoken_form(text):
    """The words a Brazilian reader would say for ``text``, case preserved.

    This is what the teacher TTS reads when the training audio is generated,
    so the audio and the student's tokens always agree. ``normalize_text``
    lower-cases it.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # The number sign and ordinal indicators before NFKC folds º and ª into o and a.
    text = _NUMBER_SIGN.sub(" número ", text)
    text = _ORDINAL_SIGN.sub(_ordinal_sign, text)
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_TRANSLATION)
    text = text.replace("⁄", "/").replace("​", "").replace("️", "").replace("‍", "")
    text = text.replace("ª", " a").replace("º", " o")
    # Two hyphens and three dots are the ASCII forms of the inventory symbols;
    # without this "..." would become three pause tokens.
    text = text.replace("--", "—").replace("...", "…")
    text = _PLATE.sub(_plate, text)
    text = _LETTER_HYPHEN_DIGIT.sub(" ", text)
    text = _LONE_HASH.sub(" cerquilha ", text)
    text = _LONE_STAR.sub(" asterisco ", text)
    text = _ABOUT.sub(" cerca de ", text)
    text = _STREET.sub(lambda m: m.group(1) + "Rua", text)
    text = _EMAIL.sub(_email, text)
    text = _URL.sub(_url, text)
    text = _HANDLE.sub(lambda m: " arroba " + m.group(1) + " ", text)
    text = _HASHTAG.sub(lambda m: (" número " + m.group(1) + " ") if m.group(1).isdigit()
                        else " rexitégue " + m.group(1) + " ", text)
    for pattern, reading in _ABBREVIATIONS:
        text = re.sub(pattern, (lambda m, r=reading: " " + r(m) + " ") if callable(reading) else " " + reading + " ",
                      text, flags=re.IGNORECASE)
    text = _LAUGH_K.sub(" cacacá ", text)
    text = _LAUGH_HA.sub(" rárárá ", text)
    text = _roman(text)
    text = _TECH_NUMBER.sub(lambda m: f" {number_words(int(m.group(1)))} {_LETTER_NAMES[m.group(2).lower()]} ", text)
    text = _MONEY.sub(_money, text)
    text = _acronyms(text)
    text = _CNPJ.sub(lambda m: " " + _spell_digits_groups(*m.groups()) + " ", text)
    text = _DATE_ISO.sub(_date_iso, text)
    text = _DATE_FULL.sub(_date_full, text)
    text = _FIRST_OF.sub(" primeiro", text)
    text = _DATE_MONTH_YEAR.sub(_month_year, text)
    text = _TIME_HMS.sub(_time_hms, text)
    text = _TIME_HM.sub(_time_hm, text)
    text = _clocks(text)
    text = _HOURS.sub(_hours, text)
    for pattern, replacement in _NAMED_HOURS:
        text = re.sub(pattern, replacement, text)
    text = _ORDINAL_DEGREE.sub(lambda m: " " + ordinal_words(int(m.group(1))) + " ", text)
    text = _ORDINAL_ASCII.sub(_ordinal_ascii, text)
    text = _PERCENT.sub(_percent, text)
    text = _DEGREES.sub(_degrees, text)
    text = _DIMENSION.sub(_dimension, text)
    text = _UNIT.sub(_unit, text)
    text = _PER.sub(lambda m: " por " + {"kg": "quilo", "mensal": "mês"}.get(m.group(1), m.group(1)), text)
    text = _KM_MARKER.sub(" quilômetro ", text)
    text = _TOLL_FREE.sub(lambda m: " " + _spell_digits_groups(*m.groups()) + " ", text)
    text = _PHONE.sub(_phone, text)
    text = _CEP.sub(lambda m: " " + _spell_digits_groups(m.group(1), m.group(2)) + " ", text)
    text = _GROUPS.sub(lambda m: m.group(0) if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", m.group(0))
                       else " " + _spell_digits_groups(*re.split(r"[.-]", m.group(0))) + " ", text)
    text = _PHONE_LOCAL.sub(_phone_local, text)
    text = _COUNTRY_CODE.sub(lambda m: " mais " + number_words(int(m.group(1))) + " ", text)
    text = _SCORE.sub(lambda m: f"{m.group(1)}{number_words(int(m.group(2)))} a {number_words(int(m.group(3)))}", text)
    text = _CONTEXT_SCORE.sub(lambda m: f"{m.group(1)} {number_words(int(m.group(2)))} a {number_words(int(m.group(3)))}", text)
    if "=" in text or any(op in text for op in ("<", ">", "≥", "≤", "≠")):
        text = _COEFFICIENT.sub(lambda m: f" {number_words(int(m.group(1)))} {_LETTER_NAMES[m.group(2).lower()]} ", text)
    else:
        text = _SPACED_SCORE.sub(lambda m: f" {number_words(int(m.group(1)))} a {number_words(int(m.group(2)))} ", text)
    if "=" not in text:
        text = _BY_X.sub(lambda m: f" {number_words(int(m.group(1)))} por {number_words(int(m.group(2)))} ", text)
    text = _TIMES_X.sub(lambda m: " " + number_words(int(m.group(1)), feminine=True) + " vezes ", text)
    text = _partial_date(text)
    if "=" not in text:
        text = _RANGE.sub(_range, text)
    text = _math(text)
    text = _NEGATIVE.sub(" menos ", text)
    text = _decimals(text)
    text = _DOT_DECIMAL.sub(lambda m: f" {number_words(int(m.group(1)))} ponto {digits_words(m.group(2)) if m.group(2).startswith('0') else number_words(int(m.group(2)))} ", text)
    text = _integers(text)
    text = text.replace("\u2297", " vezes ")
    text = _LOANWORD.sub(_loanword, text)
    text = _SLANG_RE.sub(_slang, text)
    text = _LONE_VOWEL.sub(lambda m: m.group(1) + " " + _LETTER_NAMES[m.group(2).lower()], text)
    text = _LONE_CONSONANT.sub(lambda m: " " + _LETTER_NAMES[m.group(1).lower()] + " ", text)
    text = _CONSONANTS_ONLY.sub(lambda m: " " + _spell_letters(m.group(0)) + " ", text)
    text = _cleanup(text)
    text = _REPEATED_PUNCT.sub(lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"\s+", " ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = re.sub(r"\(\s+", "(", text).strip()
    return text[:1].upper() + text[1:]


def normalize_text(text):
    """What the model reads: the spoken form, lower-cased.

    Digits follow Brazilian conventions: "1.234" is a thousand and "3,14" a
    decimal; money uses reais and centavos; dates read day-month-year; times
    read in hours and minutes; document identifiers read digit by digit;
    symbols, units, ordinals, acronyms and loanwords become words.
    """
    return re.sub(r"\s+", " ", spoken_form(text).lower()).strip()


# For comparing a transcription against its text: numbers are spoken by the
# teacher but written back as digits by Whisper ("R$ 12,50" is read as "doze
# vírgula cinquenta"), so a raw word WER would be meaningless.
_NUMBER_WORDS = frozenset(("zero", "um", "uma", "dois", "duas", "três", "tres", "quatro",
                           "cinco", "seis", "sete", "oito", "nove", "dez", "onze", "doze",
                           "treze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove",
                           "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta",
                           "oitenta", "noventa", "cem", "cento", "mil", "milhão", "milhao",
                           "bilhão", "bilhao"))
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def wer_normalize(text):
    """Casefold, strip accents and punctuation, mask numbers and number words."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"r\s?\$\s?", " <num> ", text)
    text = re.sub(r"%", " <num> ", text)
    text = _NUM.sub(" <num> ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if not w.startswith("<") and w not in _NUMBER_WORDS
             and w.rstrip("s") not in _NUMBER_WORDS]
    return " ".join(words).strip()


def spoken_normalize(text):
    """For scoring a transcription without masking anything: both sides go
    through the normaliser (so "R$ 3,00" and "três reais" agree), then accents,
    punctuation and the quatorze/catorze spelling variants are folded."""
    text = normalize_text(text).replace("catorze", "quatorze")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Syllables and stress
# ---------------------------------------------------------------------------

_VOWEL_LETTERS = frozenset("aeiouyáàâãéêíóôõúüũ")
_WEAK = frozenset("iuy")
_ACUTE_CIRCUMFLEX = frozenset("áéíóúâêô")
_TILDE = frozenset("ãõũ")
_ONSET_CLUSTERS = frozenset(("pr", "br", "tr", "dr", "cr", "gr", "fr", "vr", "pl", "bl", "cl", "gl",
                             "fl", "tl", "kr", "kl", "ch", "lh", "nh", "qu", "gu", "qü", "gü"))
_HIATUS_CODA = frozenset("lmnrz")
_OXYTONE_ENDINGS = ("r", "l", "z", "x", "i", "u", "y", "im", "um", "om", "is", "us", "ys", "ins",
                    "uns", "ons", "ims")
# Words the default rule gets wrong.
_OXYTONE_WORDS = frozenset(("porque", "porquê"))
# Unstressed function words: articles, contractions, clitic pronouns and
# monosyllabic conjunctions and prepositions.
CLITICS = frozenset(("o", "a", "os", "as", "e", "de", "do", "da", "dos", "das", "em", "no", "na", "nos",
                     "nas", "ao", "aos", "à", "às", "que", "se", "me", "te", "lhe", "lhes", "por", "pra",
                     "pro", "pras", "pros", "com", "sem", "mas", "ou", "nem", "num", "numa"))


def _letter_kinds(word):
    """'V' for a letter that can be a syllable nucleus, 'C' otherwise. The u of
    qu/gu before a vowel is part of the onset (aquilo, água), as is y before a
    vowel at the start of a word (Yasmin)."""
    kinds = []
    n = len(word)
    for i, c in enumerate(word):
        if c not in _VOWEL_LETTERS:
            kinds.append("C")
        elif c in "uü" and i > 0 and word[i - 1] in "qg" and i + 1 < n and word[i + 1] in _VOWEL_LETTERS:
            kinds.append("C")
        elif c == "y" and i == 0 and n > 1 and word[1] in _VOWEL_LETTERS:
            kinds.append("C")
        else:
            kinds.append("V")
    return kinds


def _falling_diphthong(word, kinds, j):
    """Does the vowel at ``j`` glide onto the vowel before it?"""
    first, second = word[j - 1], word[j]
    if first in _TILDE and second in "eo":
        return True                                     # mãe, pão, põe
    if second not in _WEAK or first == second:
        return False
    # An unaccented i/u before l, m, n, r, z closing its syllable, or before
    # nh, is a hiatus: ra-iz, ju-iz, ru-im, ca-ir, ra-i-nha, a-in-da.
    rest = word[j + 1:]
    if rest[:2] == "nh":
        return False
    if rest[:2] in ("rr", "ss"):
        return True
    if rest[:1] in _HIATUS_CODA and (len(rest) == 1 or kinds[j + 2] == "C"):
        return False
    return True


def syllables(word):
    """Letter spans (start, end) of the syllables of a lower-case word."""
    n = len(word)
    kinds = _letter_kinds(word)
    nuclei = []
    i = 0
    while i < n:
        if kinds[i] != "V":
            i += 1
            continue
        j = i + 1
        if j < n and kinds[j] == "V" and _falling_diphthong(word, kinds, j):
            j += 1
        nuclei.append((i, j))
        i = j
    if not nuclei:
        return [(0, n)] if n else []
    starts = [0]
    for (_, end), (start, _) in zip(nuclei, nuclei[1:]):
        run = start - end
        if run <= 1:
            starts.append(end if run == 0 else start - 1)
        elif word[start - 2:start] in _ONSET_CLUSTERS:
            starts.append(start - 2)
        else:
            starts.append(start - 1)
    ends = starts[1:] + [n]
    return list(zip(starts, ends))


def _syllable_starts(word):
    """Start indices of the syllables of a word (kept for older callers)."""
    return [s for s, _ in syllables(word)]


def _stressed_range(word):
    """(start, end) of the stressed syllable of a word.

    An acute or circumflex accent wins, then a tilde; otherwise the oxytone
    endings stress the last syllable and every other word the penultimate.
    """
    spans = syllables(word)
    if not spans:
        return 0, len(word)
    if word in _OXYTONE_WORDS and len(spans) > 1:
        return spans[-1]
    for marks in (_ACUTE_CIRCUMFLEX, _TILDE):
        for start, end in spans:
            if any(c in marks for c in word[start:end]):
                return start, end
    if len(spans) == 1:
        return spans[0]
    return spans[-1] if word.endswith(_OXYTONE_ENDINGS) else spans[-2]


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple

    def __len__(self):
        return len(self.ids)


def tokenize(text):
    """Build the token sequence from normalised text.

    Returns a :class:`Phonemized` whose symbols are letters, punctuation and
    word spaces, with a stress feature per token (1 on the letters of the
    stressed syllable of each word, 0 elsewhere and on unstressed function
    words) and the list of characters that fell outside the inventory.
    """
    symbols = [BOS]
    stress = [0]
    unknown = []
    word = []

    def flush_word():
        if not word:
            return
        text_ = "".join(word)
        letters = text_.replace("'", "")
        if text_ in CLITICS or not letters:
            start = end = -1
        else:
            start, end = _stressed_range(letters)
            # Map the span on the apostrophe-free word back onto the word.
            index = [i for i, c in enumerate(word) if c != "'"]
            start, end = index[start], index[end - 1] + 1
        for i, c in enumerate(word):
            symbols.append(c)
            stress.append(STRESS_STRESSED if start <= i < end and c != "'" else 0)
        word.clear()

    for c in text:
        if c == " ":
            flush_word()
            if symbols[-1] != SP:
                symbols.append(SP)
                stress.append(0)
        elif c in LETTER_SET or c in _WORD_CHARS:
            word.append(c)
        elif c in SYMBOL_TO_ID:
            flush_word()
            symbols.append(c)
            stress.append(0)
        else:
            flush_word()
            symbols.append(UNK)
            stress.append(0)
            unknown.append(c)
    flush_word()
    symbols.append(EOS)
    stress.append(0)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols),
                      tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty text")
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("nothing to read")
    return tokenize(normalized)


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
