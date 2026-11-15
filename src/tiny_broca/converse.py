"""Combinatorial Brazilian Portuguese conversational sentences.

A deterministic, seeded generator. The slot pools are written for this project;
the sentences it produces form the conversational part of the training corpus
(read by Qwen3-TTS, like the LLM-written sentences of the original dataset, but
generated from slots instead of a one-shot model call, which keeps the corpus
reproducible).

Slot keys: person, feeling, place, food, drink, object, relative, topic, venue,
meeting, cause, frequency, hour.
"""

import random
import re
from dataclasses import dataclass


@dataclass
class Family:
    template: str
    slots: tuple


PERSONS = (
    "Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe", "Gabriela", "Hugo",
    "Isabela", "João", "Karina", "Lucas", "Marina", "Nelson", "Otávio", "Paula",
    "Rafael", "Sabrina", "Thiago", "Vera", "Beatriz", "Caio", "Débora", "Eduardo",
    "Fernanda", "Gustavo", "Heitor", "Larissa", "Matheus", "Renata", "Sofia",
    "Duda", "Tia Cida", "Vó Zefa", "Padrinho", "Dona Maria", "Seu João",
)
FEELINGS = ("muito feliz", "um pouco nervoso", "bem cansado", "bem animado",
            "preocupado", "contente", "bem surpreso", "com saudade", "estressado",
            "empolgado", "bem tranquilo", "desanimado")
PLACES = ("a padaria", "o mercado", "a praça", "o banco", "a papelaria",
          "a farmácia", "o posto", "a academia", "o shopping", "a escola",
          "o escritório", "o bar da esquina", "a casa da minha mãe", "o trabalho",
          "a praia", "o metrô", "o aeroporto")
FOODS = ("um café com leite", "um pão de queijo", "um açaí", "um hambúrguer",
         "uma pizza", "um pastel", "um sanduíche de frango", "um bolo de cenoura",
         "uma coxinha", "um brigadeiro", "uma tapioca", "um feijão com arroz",
         "uma lasanha", "um churrasco", "uma moqueca")
DRINKS = ("água", "suco de laranja", "café", "cerveja gelada", "refrigerante",
          "chá gelado", "sumô de uva")
OBJECTS = ("o celular", "a chave", "a carteira", "o guarda-chuva", "o livro",
           "o controle", "a blusa", "os óculos", "a bolsa", "o recibo",
           "o embrulho", "o notebook", "o carregador", "a foto", "o bilhete")
RELATIVES = ("meu filho", "minha irmã", "meu chefe", "meu sogro", "minha cunhada",
             "meu vizinho", "a minha prima", "meu padrinho", "a secretária",
             "o meu médico", "meu cunhado")
TOPICS = ("a novela", "o jogo de ontem", "o filme novo", "a série", "a música",
          "a festa", "a viagem", "o show", "a prova", "o projeto", "o relatório",
          "a reunião", "o trânsito", "o tempo", "o preço da passagem")
VENUES = ("a padaria", "o banco", "a papelaria", "a farmácia", "a loja",
          "o mercado", "o cinema", "a agência", "a cafeteria", "a papelaria")
MEETINGS = ("a reunião", "o show", "o jogo", "a prova", "o voo", "o dentista",
            "o jantar", "a aula", "o treino", "o almoço")
CAUSES = ("porque choveu", "por causa do trânsito", "que o forno quebrou",
          "que a internet caiu", "que esqueci em casa", "que tava ocupado",
          "porque o carro quebrou", "que doeu a cabeça", "que foi cancelado")
FREQUENCIES = ("sempre", "às vezes", "na semana passada", "ontem à noite",
               "no fim de semana", "no mês que vem", "daqui a pouco", "toda manhã",
               "todo domingo", "há dois dias", "nesse verão")
PREFIXES = ("Olha, ", "Então, ", "É que ", "Sabe, ", "Pois é, ", "Olha só, ",
            "De qualquer forma, ", "Sabe de uma coisa? ", "Olha, ", "Então, ")
SUFFIXES = (" né?", " viu?", " tá bom?", " faz sentido", " não é?", " hein?",
            " de verdade", " com certeza", " hoje em dia", " da última vez")


def _hour(rng):
    h = rng.randint(6, 21)
    m = rng.choice([0, 15, 30, 45])
    if m == 0:
        return f"{h} horas" if h != 1 else "1 hora"
    if m == 30 and rng.random() < 0.6:
        return f"{h} e meia"
    return f"{h} e {['quinze', 'trinta', 'quarenta e cinco'][[15, 30, 45].index(m)]}"


_RAW = [
    # Greetings / small talk
    ("Oi, {person}, tudo bem? Vi que você veio, que bom.", ("person",)),
    ("Ei, {person}, faz tempo que a gente não se vê.", ("person",)),
    ("Bom dia, {person}. Dormiu bem? Eu acordei cedo hoje.", ("person",)),
    ("Oi, {person}, como você está? Eu estou {feeling}.", ("person", "feeling")),
    ("{person}, você comeu alguma coisa hoje?", ("person",)),
    ("Oi, {person}, queria um {drink}, por favor?", ("person", "drink")),

    # Plans
    ("A gente pode se encontrar na {place} depois?", ("place",)),
    ("Vamos passar na {place} no caminho?", ("place",)),
    ("Eu vou à {place} amanhã de manhã.", ("place",)),
    ("Posso te buscar às {hour}? Aí a gente vai junto.", ("hour",)),
    ("Te encontro na {place}, tudo bem?", ("place",)),

    # Food and drinks
    ("Vou pedir {food}, e você vai querer alguma coisa?", ("food",)),
    ("Eu quero um {drink} gelado e {food}.", ("drink", "food")),
    ("Aquele {food} daqui é muito bom.", ("food",)),
    ("Comi {food} no almoço e ainda tô com fome.", ("food",)),
    ("Passou {drink} pra mim? Tô com sede.", ("drink",)),

    # Work and errands
    ("A reunião foi cancelada, então eu saio mais cedo.", (),),
    ("Meu chefe mandou mais e-mail agora há pouco.", (),),
    ("Eu vou resolver isso ainda hoje, pode ficar tranquilo.", (),),
    ("Preciso entregar o relatório até sexta, então vou ficar aqui mais um pouquinho.", (),),
    ("Esqueci {object} em casa de novo.", ("object",)),
    ("Você viu {object}? Tô procurando desde cedo.", ("object",)),
    ("Deixa eu guardar {object} pra você.", ("object",)),
    ("O boleto vence semana que vem, não precisa se preocupar.", (),),
    ("Fui ao banco e a fila andou, saí rápido.", (),),

    # Feelings and reactions
    ("Nossa, {person}, eu tava {feeling} depois de saber disso.", ("person", "feeling")),
    ("Eu fiquei {feeling} com a notícia.", ("feeling",)),
    ("Tô {feeling} hoje, mas amanhã vai melhorar.", ("feeling",)),
    ("Meu, {person}, eu sei que dá pra resolver isso.", ("person",)),
    ("Poxa, que chato. Eu tava contando com isso.", (),),
    ("Sério? Eu nunca imaginei que fosse assim.", (),),
    ("Mentira? Faz assim de novo que eu não acreditei.", (),),

    # Excuses and delays
    ("Desculpa ter demorado, {cause}.", ("cause",)),
    ("Peço desculpa, {cause}. Não teve como chegar antes.", ("cause",)),
    ("Cheguei atrasado, {cause}. A partir de agora prometo melhorar.", ("cause",)),

    # Weather and city
    ("Olha esse tempo, parece que vai chover a tarde.", (),),
    ("Tá fazendo muito calor, né? Passa um gelo aí.", (),),
    ("O trânsito hoje estava um absurdo na {place}.", ("place",)),
    ("Foi um dia de sol, perfeito pra ir à praia.", (),),
    ("Amanhã promete chuva de novo, leva guarda-chuva.", (),),

    # Family and people
    ("{relative} passou aqui hoje de manhã.", ("relative",)),
    ("{relative} vai viajar no fim do mês.", ("relative",)),
    ("Meus pais moram em São Paulo, mas a gente fala toda semana.", (),),
    ("A festa de aniversário da minha prima é sábado.", (),),
    ("Chama {person} pra gente, o pessoal vai adorar.", ("person",)),

    # Media and leisure
    ("Você viu {topic}? Tá todo mundo falando.", ("topic",)),
    ("Assisti {topic} ontem à noite, não foi nada demais.", ("topic",)),
    ("Vi {topic} ontem e fiquei pensando nele o dia inteiro.", ("topic",)),
    ("Coloquei {topic} pra tocar, fica aí.", ("topic",)),
    ("No fim de semana a gente vai no cinema, quer ir junto?", (),),
    ("Fui à academia {frequency} e voltei destruído.", ("frequency",)),

    # Questions
    ("Você sabe se {venue} vai abrir hoje?", ("venue",)),
    ("A que horas é {meeting}?", ("meeting",)),
    ("Onde você guardou {object}?", ("object",)),
    ("Por que você não me avisou antes?", (),),
    ("Como a gente resolve isso?", (),),
    ("Quando {object} chega?", ("object",)),
    ("Você topa ir à {place} depois do trabalho?", ("place",)),
    ("Será que dá pra adiar pra {frequency}?", ("frequency",)),
    ("A gente combina de se encontrar {frequency}?", ("frequency",)),
    ("Você tem troco pra dez?", (),),
    ("Faz quanto tempo você não vai à {place}?", ("place",)),

    # Exclamations
    ("Que dia lindo hoje!", (),),
    ("Uau, que coisa linda!", (),),
    ("Nossa, olha que bagunça!", (),),
    ("Caramba, eu não acredito nisso!", (),),
    ("Que maravilha, que alívio!", (),),
    ("Que horror, de novo?", (),),

    # Shopping and money
    ("Passou {drink} pra mim? Eu te devolvo depois.", ("drink",)),
    ("O preço subiu de novo, tá tudo caro.", (),),
    ("Comprei {object} com desconto, fiquei feliz.", ("object",)),
    ("O cartão não passou, o caixa disse que é falha.", (),),
    ("Guarda o troco, por favor.", (),),
]
FAMILIES = tuple(Family(t, s) for t, s in _RAW)

_POOL = {
    "person": PERSONS,
    "feeling": FEELINGS,
    "place": PLACES,
    "food": FOODS,
    "drink": DRINKS,
    "object": OBJECTS,
    "relative": RELATIVES,
    "topic": TOPICS,
    "venue": VENUES,
    "meeting": MEETINGS,
    "cause": CAUSES,
    "frequency": FREQUENCIES,
}


def _fill(template, slots, rng):
    values = {k: rng.choice(v) for k, v in _POOL.items()}
    values["hour"] = _hour(rng)
    text = template
    for slot in slots:
        text = text.replace("{%s}" % slot, values[slot])
    if rng.random() < 0.4:
        text = rng.choice(PREFIXES) + text[0].lower() + text[1:]
    if not text.endswith("?") and rng.random() < 0.2:
        text = text.rstrip(".") + rng.choice(SUFFIXES)
    return text


def generate(count, seed=20261115):
    """Return ``count`` unique conversational pt-BR sentences."""
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < count:
        family = rng.choice(FAMILIES)
        text = _fill(family.template, family.slots, rng)
        text = re.sub(r"\s+", " ", text).strip()
        if 8 < len(text) < 160 and text not in seen:
            seen.add(text)
            out.append(text)
    return out


if __name__ == "__main__":
    for s in generate(12):
        print("-", s)
