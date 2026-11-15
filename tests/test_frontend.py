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
        self.assertEqual(normalize_text("14:30"), "catorze e meia")
        self.assertEqual(normalize_text("100%"), "cem por cento")
        self.assertEqual(normalize_text("2,5"), "dois vírgula cinco")
        self.assertEqual(normalize_text("123.456.789-09"), "um dois três, quatro cinco seis, sete oito nove, zero nove")

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


if __name__ == "__main__":
    unittest.main()
