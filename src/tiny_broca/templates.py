"""Number, date, money and identifier templates for Brazilian Portuguese.

The sentences in this module exercise the text-normalization rules of the
front end (numbers, dates, money in the Brazilian convention of dot as
thousands separator and comma as decimal, phone numbers, document
identifiers, plates, percentages, times). They form the template part of
batch 2 of the corpus, mirroring the original dataset.
"""

import random
import re

MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho",
         "julho", "agosto", "setembro", "outubro", "novembro", "dezembro")
MES_CURTO = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")
DIAS = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")
UNITS = ("reais", "centavos", "dólares", "pessoas", "quilômetros", "metros", "minutos",
         "horas", "dias", "anos", "vezes", "por cento", "graus", "pontos", "vezes",
         "contas", "itens", "unidades", "litros", "caixas", "páginas", "linhas",
         "alunos", "funcionários", "cidades", "estados", "times", "rodadas", "gols")
CITIES = ("São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre",
          "Salvador", "Recife", "Fortaleza", "Goiânia", "Brasília", "Campinas",
          "Florianópolis", "Manaus", "Belém", "Vitória", "Natal", "Cuiabá", "Joinville")
STREETS = ("Rua das Flores", "Avenida Brasil", "Rua XV de Novembro", "Travessa do Comércio",
           "Rua Augusta", "Avenida Paulista", "Rua do Carmo", "Rua da Praia")
EVENTS = ("a reunião", "o show", "o casamento", "a prova", "o voo", "o treino",
          "o aniversário", "o lançamento", "a palestra", "o funeral")
REASONS = ("por causa da chuva", "por causa do trânsito", "porque o avião atrasou",
           "porque adoei", "por causa da obra na rua", "porque o sistema caiu",
           "porque esqueci", "por causa da festa")


def money(rng, lo=1.0, hi=99999.0):
    v = round(rng.uniform(lo, hi), 2)
    s = ("%.2f" % v).replace(".", "§").replace(",", ".").replace("§", ",")
    int_part, frac = s.split(",")
    out = []
    while len(int_part) > 3:
        out.append(int_part[-3:])
        int_part = int_part[:-3]
    out.insert(0, int_part)
    return "R$ " + ".".join(out) + "," + frac


def phone(rng):
    ddd = rng.choice((11, 21, 31, 41, 47, 51, 61, 71, 81, 85))
    a, b = "".join(rng.choice("0123456789") for _ in range(5)), "".join(rng.choice("0123456789") for _ in range(4))
    return f"({ddd}) {a}-{b}"


def cpf(rng):
    a, b, c = "".join(rng.choice("0123456789") for _ in range(3))
    return f"{a}.{b}.{c}-00"


def cnpj(rng):
    a = "".join(rng.choice("0123456789") for _ in range(2))
    b = "".join(rng.choice("0123456789") for _ in range(3))
    c = "".join(rng.choice("0123456789") for _ in range(3))
    return f"{a}.{b}.{c}/0001-90"


def cep(rng):
    return "%05d-%03d" % (rng.randint(0, 99999), rng.randint(0, 999))


def plate(rng):
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3)) + \
        str(rng.randint(0, 9)) + "D" + str(rng.randint(10, 99))


def date_short(rng):
    return "%02d/%02d/%d" % (rng.randint(1, 28), rng.randint(1, 12), rng.randint(2018, 2026))


def date_long(rng):
    return "%d de %s de %d" % (rng.randint(1, 28), rng.choice(MESES), rng.randint(2018, 2026))


def time(rng):
    style = rng.random()
    h = rng.randint(5, 23)
    m = rng.choice((0, 5, 15, 30, 45))
    if style < 0.3:
        return f"{h:02d}:{m:02d}"
    if style < 0.6:
        return f"{h}h{m:02d}"
    if h == 1:
        return f"à 1 hora" if m == 0 else f"à 1 e {m if m != 30 else 'meia'}"
    return f"às {h} e {m or 'meia'}"


_T = (
    # Money
    ("A conta de luz veio em {mlow}, um valor alto.", ("mlow",)),
    ("Paguei {mmid} no mercado e já estou economizando.", ("mmid",)),
    ("O aluguel sobiu para {mhi} este mês.", ("mhi",)),
    ("Guardei {mmid} da festa, separei tudo por conta.", ("mmid",)),
    ("Quanto ficou {mmid}? Quero deixar o troco.", ("mmid",)),
    ("O boleto de {mhi} vence dia {d}.", ("mhi", "d")),
    ("O frete saiu por {mlow}, bem mais barato que o previsto.", ("mlow",)),
    ("O carro novo custa {mbig}, mas parcelou bem.", ("mbig",)),
    # Counts
    ("Chegaram {n} {u} antes do prazo, ótimo.", ("n", "u")),
    ("Faltam {n} {u} para terminar o serviço.", ("n", "u")),
    ("Subiu para {n} {u} desde janeiro, quase o dobro.", ("n", "u")),
    ("De {n1} a {n2} {u}, dependendo do dia.", ("n1", "n2", "u")),
    ("Foram {n} {u} no total, mais que o esperado.", ("n", "u")),
    ("Aumento de {p} por cento no preço, outra vez.", ("p",)),
    ("A previsão é de {temp} graus amanhã, use um casaco.", ("temp",)),
    ("O jogo terminou {n} a {n2}, foi apertado.", ("n", "n2")),
    # Dates and times
    ("A entrega está prevista para {dt}.", ("dt",)),
    ("Marquei {ev} para {dt}, não esquece.", ("ev", "dt")),
    ("Fui em {dt} e a cidade estava linda.", ("dt",)),
    ("O {ev} foi em {dt}, que dia bom.", ("ev", "dt")),
    ("Te ligo {t}, tudo bem?", ("t",)),
    ("A reunião começou {t}, fui direto.", ("t",)),
    ("O voo pousa {t}, te aviso quando sair no ar.", ("t",)),
    ("É {day} que eu trabalho, sempre o mesmo dia.", ("day",)),
    # Identifiers
    ("Meu telefone é {p}, me liga se precisar.", ("p",)),
    ("Anota meu número, {p}.", ("p",)),
    ("O CPF da conta é {c}, confere no contrato.", ("c",)),
    ("O CNPJ da empresa é {j}, emita a nota com ele.", ("j",)),
    ("O CEP aqui é {e}, usa esse no endereço.", ("e",)),
    ("A placa do carro é {pl}, um azul claro.", ("pl",)),
    # Places with addresses
    ("Encontro no {st}, número {n}, perto da esquina.", ("st", "n")),
    ("Moro no {st}, {n}, {city}.", ("st", "n", "city")),
    ("A loja é no {st}, {n}, entre o banco e a farmácia.", ("st", "n")),
    ("Vou até {city} no fim de semana, {n} {u} de estrada.", ("city", "n", "u")),
    # Percentages and scores
    ("Subiu {p} por cento no ano, o esperado era menos.", ("p",)),
    ("Desconto de {p} por cento pra quem pagar à vista.", ("p",)),
    ("O time marcou {n} pontos na rodada, ficou em {o}º lugar.", ("n", "o")),
    ("Acertou {n} de {n2} perguntas, bom resultado.", ("n", "n2")),
    # Exclamatory numbers
    ("Caramba, {n} {u} de fila, melhor desistir.", ("n", "u")),
    ("Mais {n} {u} e eu desisto, que dia.", ("n", "u")),
    ("Ainda bem que são só {n} {u} a caminho.", ("n", "u")),
)

def _fill(template, slots, rng):
    def value(name):
        if name == "dt":
            return rng.choice([date_short, date_long])(rng)
        if name == "day":
            return rng.choice(DIAS)
        if name == "t":
            return time(rng)
        if name == "u":
            return rng.choice(UNITS)
        if name == "ev":
            return rng.choice(EVENTS)
        if name == "st":
            return rng.choice(STREETS)
        if name == "city":
            return rng.choice(CITIES)
        if name == "c":
            return cpf(rng)
        if name == "j":
            return cnpj(rng)
        if name == "e":
            return cep(rng)
        if name == "pl":
            return plate(rng)
        if name == "o":
            return str(rng.randint(1, 20))
        if name == "mlow":
            return money(rng, 10, 900)
        if name == "mmid":
            return money(rng, 50, 2500)
        if name == "mhi":
            return money(rng, 800, 6000)
        if name == "mbig":
            return money(rng, 60000, 350000)
        if name == "m":
            return money(rng)
        if name == "temp":
            return str(rng.randint(-5, 45))
        if name == "n":
            if "n2" in slots:
                top = int(_values["n2"])
                return str(rng.randint(0, top))
            return str(rng.randint(2, 9999))
        if name == "n2":
            if "n1" in slots:
                return str(int(_values["n1"]) + rng.randint(1, 999))
            return str(_values["n2"])
        if name == "n1":
            return str(_values["n1"]) if "n1" in _values else str(rng.randint(2, 500))
        if name == "p":
            return str(rng.randint(3, 60))
        if name == "d":
            return str(rng.randint(1, 28))
        raise KeyError(name)

    # Resolve paired slots so comparisons stay sensible (n <= n2 when both).
    _values = {}
    if "n2" in slots:
        _values["n2"] = rng.randint(5, 50) if "n" in slots else rng.randint(10, 9999)
    if "n1" in slots and "n2" in slots:
        _values["n1"] = rng.randint(2, 500)

    text = template
    for slot in slots:
        text = text.replace("{%s}" % slot, value(slot))
    return text


def generate(count, seed=20261116):
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < count:
        template, slots = rng.choice(_T)
        text = _fill(template, slots, rng)
        text = re.sub(r"\s+", " ", text).strip()
        if text not in seen and len(text) > 8:
            seen.add(text)
            out.append(text)
    return out


if __name__ == "__main__":
    for s in generate(14):
        print("-", s)
