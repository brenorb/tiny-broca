import unittest
from unittest.mock import patch

from tiny_broca import text
from tiny_broca.config import Config
from tiny_broca.text import (BOS, EOS, LETTERS, PHONE_SET, PHONES, SP, SYMBOLS, SYMBOL_TO_ID, UNK, VOWELS,
                             frontend_fingerprint, normalize_text, phonemize, tokenize)


class FrontEndTests(unittest.TestCase):
    def test_inventory_is_consistent_with_the_model_vocabulary(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(Config().vocab_size, len(SYMBOLS))
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(len(PHONES), len(LETTERS))
        for c in "ãõçéê":     # accented letters are distinct graphemes
            self.assertIn(c, PHONE_SET)
        self.assertTrue(all(v in PHONE_SET for v in VOWELS))
        with self.assertRaises(ValueError):
            Config(vocab_size=259)     # the byte vocabulary is gone

    def test_numbers_follow_brazilian_conventions(self):
        self.assertEqual(text.number_words(1000), "mil")          # never "um mil"
        self.assertEqual(text.number_words(1005), "mil e cinco")
        self.assertEqual(text.number_words(1050), "mil e cinquenta")
        self.assertEqual(text.number_words(1150), "mil cento e cinquenta")
        self.assertEqual(text.number_words(1234), "mil duzentos e trinta e quatro")
        self.assertEqual(text.number_words(4090), "quatro mil e noventa")
        self.assertEqual(text.number_words(4150), "quatro mil cento e cinquenta")
        self.assertEqual(text.number_words(2026), "dois mil e vinte e seis")
        self.assertEqual(text.number_words(115), "cento e quinze")
        self.assertEqual(text.number_words(15), "quinze")
        self.assertEqual(text.number_words(1000090), "um milhão e noventa")
        self.assertEqual(text.number_words(1050000), "um milhão e cinquenta mil")

    def test_normalisation_spells_out_symbols(self):
        self.assertEqual(normalize_text("R$ 4.090,50"), "quatro mil e noventa reais e cinquenta centavos")
        self.assertEqual(normalize_text("R$ 1500"), "mil e quinhentos reais")
        self.assertEqual(normalize_text("15/03/2026"), "quinze de março de dois mil e vinte e seis")
        self.assertEqual(normalize_text("14:30"), "quatorze e meia")
        self.assertEqual(normalize_text("100%"), "cem por cento")
        self.assertEqual(normalize_text("2,5"), "dois vírgula cinco")
        self.assertEqual(normalize_text("123.456.789-09"), "um dois três, quatro cinco seis, sete oito nove, zero nove")

    def test_decimals_read_brazilian_style(self):
        # A dot-grouped integer with a comma decimal: the group is part of the
        # number, so the whole "1.234,56" is one reading.
        self.assertEqual(normalize_text("1.234,56"), "mil duzentos e trinta e quatro vírgula cinquenta e seis")
        self.assertEqual(normalize_text("3,14"), "três vírgula quatorze")
        self.assertEqual(normalize_text("3,05"), "três vírgula zero cinco")
        self.assertEqual(normalize_text("150,00"), "cento e cinquenta")
        self.assertEqual(normalize_text("R$ 150,00"), "cento e cinquenta reais")
        self.assertEqual(normalize_text("R$ 1.234,56"),
                         "mil duzentos e trinta e quatro reais e cinquenta e seis centavos")

    def test_normalisation_unifies_quotes_and_case(self):
        self.assertEqual(normalize_text("“Olá”  ela disse -- ‘não’ (à parte)"),
                         '"olá" ela disse — \'não\' (à parte)')
        self.assertEqual(normalize_text("TÁ BOM"), "tá bom")

    def test_letters_carry_stress_from_the_syllable(self):
        result = phonemize("Olá, meu café.")
        self.assertEqual(result.symbols, (BOS, "o", "l", "á", ",", SP, "m", "e", "u", SP, "c", "a", "f", "é", ".", EOS))
        self.assertEqual(result.stress, (0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(result.unknown, ())
        # An explicit accent beats the position rules (lâ.mpa.da: the accented
        # syllable is the first; the consonant cluster approximation covers l-â-m).
        self.assertEqual(phonemize("lâmpada").symbols, (BOS, "l", "â", "m", "p", "a", "d", "a", EOS))
        self.assertEqual(phonemize("lâmpada").stress, (0, 1, 1, 1, 0, 0, 0, 0, 0))
        # Default: the penultimate syllable (casa = CA.sa); "casa" has two, so the first.
        self.assertEqual(phonemize("casa").stress, (0, 1, 1, 0, 0, 0))
        self.assertEqual(phonemize("telefone").stress, (0, 0, 0, 0, 0, 1, 1, 0, 0, 0))

    def test_out_of_inventory_characters_become_unknown(self):
        self.assertIn("w", PHONE_SET)
        result = phonemize("ok")     # every letter is inventory, so nothing is unknown
        self.assertEqual(result.unknown, ())
        self.assertEqual(tokenize("§").symbols, (BOS, UNK, EOS))     # section sign is not a letter
        self.assertEqual(tokenize("§").unknown, ("§",))
        with self.assertRaises(ValueError):
            phonemize("   ")

    def test_fingerprint_tracks_inventory_and_version(self):
        baseline = frontend_fingerprint()
        with patch.object(text, "FRONTEND_VERSION", 999):
            self.assertNotEqual(frontend_fingerprint(), baseline)
        with patch.object(text, "SYMBOLS", SYMBOLS + ("new",)):
            self.assertNotEqual(frontend_fingerprint(), baseline)


class NormaliserV2Tests(unittest.TestCase):
    """Front end version 2: everything a reader says aloud becomes words."""

    def check(self, cases):
        for written, spoken in cases:
            with self.subTest(written=written):
                self.assertEqual(normalize_text(written), spoken)
                self.assertEqual(phonemize(written).unknown, ())

    def test_reported_v0_failures(self):
        self.check([
            ("Vou comprar online.", "vou comprar onlaine."),
            ("4 > 3", "quatro maior que três"),
            ("R$3,00", "três reais"),
            ("14/03", "quatorze de março"),
        ])

    def test_symbols_and_math(self):
        self.check([
            ("2 + 2 = 4", "dois mais dois igual a quatro"),
            ("x + 2 = 5", "xis mais dois igual a cinco"),
            ("2 x 3 = 6", "dois vezes três igual a seis"),
            ("Se a > b então b < a.", "se á maior que bê então bê menor que á."),
            ("5 >= 3 e 2 != 3", "cinco maior ou igual a três e dois diferente de três"),
            ("Tom & Jerry", "tom e jerry"),
        ])

    def test_dates(self):
        self.check([
            ("Dia 1º de maio.", "dia primeiro de maio."),
            ("01/01/2026", "primeiro de janeiro de dois mil e vinte e seis"),
            ("Vence em 05/12.", "vence em cinco de dezembro."),
            ("2025-03-14", "quatorze de março de dois mil e vinte e cinco"),
            ("Em 03/2025.", "em março de dois mil e vinte e cinco."),
            ("1/2 xícara e 3/4 de litro", "meia xícara e três quartos de litro"),
        ])

    def test_times_agree_in_gender(self):
        self.check([
            ("Às 14h.", "às quatorze horas."),
            ("Às 2 da tarde.", "às duas da tarde."),
            ("À 1 hora.", "à uma hora."),
            ("21:00", "vinte e uma horas"),
            ("Das 9h às 18h30.", "das nove horas às dezoito e meia."),
            ("12:00", "meio dia"),
        ])

    def test_ordinals(self):
        self.check([
            ("O 1º lugar e a 2ª colocada.", "o primeiro lugar e a segunda colocada."),
            ("21º andar", "vigésimo primeiro andar"),
            ("a 3a série", "a terceira série"),
        ])
        self.assertEqual(text.ordinal_words(11), "décimo primeiro")
        self.assertEqual(text.ordinal_words(100), "centésimo")
        self.assertEqual(text.ordinal_words(32, feminine=True), "trigésima segunda")

    def test_money(self):
        self.check([
            ("R$ 0,50", "cinquenta centavos"),
            ("R$ 1,99", "um real e noventa e nove centavos"),
            ("R$ 10,5", "dez reais e cinquenta centavos"),
            ("R$ 1 milhão", "um milhão de reais"),
            ("R$ 1.000.000,00", "um milhão de reais"),
            ("R$ 2,5 milhões", "dois vírgula cinco milhões de reais"),
            ("R$ 5 mil", "cinco mil reais"),
            ("US$ 10", "dez dólares"),
            ("€ 20", "vinte euros"),
            ("£ 2", "duas libras"),
        ])

    def test_numbers_units_and_agreement(self):
        self.check([
            ("12,5%", "doze vírgula cinco por cento"),
            ("36,6°C", "trinta e seis vírgula seis graus"),
            ("-5 °C", "menos cinco graus"),
            ("5 km e 1,5 kg", "cinco quilômetros e um vírgula cinco quilos"),
            ("1 km", "um quilômetro"),
            ("2 pessoas e 200 casas", "duas pessoas e duzentas casas"),
            ("2 dias e 1 semana", "dois dias e uma semana"),
            ("1.000.000 pessoas", "um milhão de pessoas"),
            ("123456", "cento e vinte e três mil quatrocentos e cinquenta e seis"),
            ("007", "zero zero sete"),
            ("10-15 anos", "dez a quinze anos"),
            ("Parcelei em 10x.", "parcelei em dez vezes."),
            ("Brasil 2x1 Argentina", "brasil dois a um argentina"),
            ("4G e 5G", "quatro gê e cinco gê"),
        ])

    def test_identifiers(self):
        self.check([
            ("(11) 98765-4321", "onze, nove oito sete seis cinco, quatro três dois um"),
            ("70040-010", "sete zero zero quatro zero, zero um zero"),
            ("12.345.678/0001-90", "um dois, três quatro cinco, seis sete oito, zero zero zero um, nove zero"),
        ])

    def test_acronyms_abbreviations_and_roman_numerals(self):
        self.check([
            ("O CPF e o RG.", "o cê pê éfe e o érre gê."),
            ("A NASA e a FIFA.", "a nasa e a fifa."),
            ("Dois PCs e três CDs.", "dois pê cês e três cê dês."),
            ("O IBGE.", "o i bê gê é."),
            ("O Sr. Silva e a Dra. Ana, etc.", "o senhor silva e a doutora ana, etcétera"),
            ("Século XXI e Dom Pedro II", "século vinte e um e dom pedro segundo"),
            ("A II Guerra Mundial", "a segunda guerra mundial"),
            ("vc vai hj? blz", "você vai hoje? beleza"),
        ])
        self.assertEqual(normalize_text("TÁ BOM"), "tá bom")      # shouted words are not acronyms

    def test_loanwords_are_respelled(self):
        self.check([
            ("Bitcoin, shampoo, blockchain e tradeoff.", "bitecóin, xampu, bloquitchêin e treidófe."),
            ("O Wi-Fi do iPhone.", "o uaifai do aifone."),
            ("Os sites e os links.", "os saites e os linques."),
            ("Fiz o check-in.", "fiz o xequín."),
            ("Trabalho em home office.", "trabalho em rôme ófice."),
        ])

    def test_addresses(self):
        self.check([
            ("fulano.silva@gmail.com", "fulano ponto silva arroba gemêil ponto com"),
            ("www.google.com.br", "dáblio dáblio dáblio ponto gúgou ponto com ponto bê érre"),
        ])

    def test_nothing_outside_the_inventory_survives(self):
        for written in ("Olá 👋🏽 mundo!", "Preço: ~R$ 10 ± 2", "Müller e Gisèle", "#tbt @fulano", "≈ 3 ∞"):
            with self.subTest(written=written):
                self.assertEqual(phonemize(written).unknown, ())


class StressV2Tests(unittest.TestCase):
    def stressed(self, word):
        start, end = text._stressed_range(word)
        return word[start:end]

    def test_oxytone_endings(self):
        for word, syllable in (("aqui", "qui"), ("caju", "ju"), ("rapaz", "paz"), ("comum", "mum"),
                               ("jardim", "dim"), ("abacaxi", "xi"), ("feliz", "liz"), ("brasil", "sil"),
                               ("falou", "lou"), ("museu", "seu"), ("porque", "que")):
            with self.subTest(word=word):
                self.assertEqual(self.stressed(word), syllable)

    def test_paroxytones_and_accents(self):
        for word, syllable in (("casa", "ca"), ("homem", "ho"), ("jovens", "jo"), ("falam", "fa"),
                               ("lápis", "lá"), ("táxi", "tá"), ("também", "bém"), ("órfão", "ór"),
                               ("coração", "ção"), ("dia", "di"), ("história", "tó")):
            with self.subTest(word=word):
                self.assertEqual(self.stressed(word), syllable)

    def test_hiatus_and_diphthongs(self):
        cases = {"raiz": "ra-iz", "juiz": "ju-iz", "ruim": "ru-im", "rainha": "ra-i-nha", "ainda": "a-in-da",
                 "reais": "re-ais", "saída": "sa-í-da", "muito": "mui-to", "praia": "prai-a", "água": "á-gua",
                 "aquilo": "a-qui-lo", "guerra": "guer-ra", "quatro": "qua-tro", "pão": "pão"}
        for word, split in cases.items():
            with self.subTest(word=word):
                self.assertEqual("-".join(word[a:b] for a, b in text.syllables(word)), split)
        self.assertEqual(self.stressed("reais"), "ais")
        self.assertEqual(self.stressed("raiz"), "iz")

    def test_function_words_are_unstressed(self):
        result = phonemize("a casa de que")
        stress = dict(zip(range(len(result.symbols)), result.stress))
        self.assertEqual([s for sym, s in zip(result.symbols, result.stress) if sym == "a"][0], 0)
        self.assertEqual(result.stress[result.symbols.index("d")], 0)
        self.assertEqual(result.stress[result.symbols.index("c")], 1)


if __name__ == "__main__":
    unittest.main()
