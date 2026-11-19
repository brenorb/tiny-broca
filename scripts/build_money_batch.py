#!/usr/bin/env python3
"""Build batch3: money amounts in the range the base corpus under-represented.

The 104k-step model rushed and garbled long spelled-out values ("mil duzentos
e quarenta e cinco reais e dez centavos"): after normalisation, only 0.4% of
training utterances carry a 6+ word number span, and none is as dense as that
one. This batch concentrates on 1,000-99,999.99 values (5-13 word spans after
the normalizer), in varied conversational frames, so the fine-tune teaches
pacing where it was extrapolating.

    python scripts/build_money_batch.py [--count 400] [--seed 20261119]

Writes data/corpus/batch3/train.parquet with the standard corpus schema
(id, batch, split, text, source).
"""
import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Conversational frames. {v} is the pt-BR formatted value ("1.234,56").
FRAMES = (
    "A conta do aluguel fechou em {v} este mês.",
    "Meu aluguel este mês foi de {v}.",
    "Paguei {v} de aluguel e restou pouca coisa.",
    "A conta do cartão veio de {v}, mais alta que o esperado.",
    "O carro que eu quero está por {v}.",
    "A viagem custou {v} no total, com passagens e hotel.",
    "Ainda devo {v} para o banco.",
    "O empréstimo ficou em {v} ao todo.",
    "Gastei {v} com o presente de aniversário.",
    "A passagem aérea saiu por {v}.",
    "A obra do banheiro custou {v}.",
    "Pedi {v} emprestado para cobrir o aluguel.",
    "A taxa do financiamento somou {v} no ano.",
    "Vendi o notebook por {v}, bem acima do que paguei.",
    "O salário deste mês rendeu {v} no fim.",
    "A mensalidade da escola subiu para {v}.",
    "O seguro do carro custa {v} por ano.",
    "A reforma ficou em {v}, dentro do orçamento.",
    "Ganhei {v} de bônus no trimestre.",
    "A fatura do cartão fechou em {v}.",
)


def money(rng, lo, hi):
    v = round(rng.uniform(lo, hi), 2)
    s = ("%.2f" % v).replace(".", "§").replace(",", ".").replace("§", ",")
    int_part, frac = s.split(",")
    out = []
    while len(int_part) > 3:
        out.append(int_part[-3:])
        int_part = int_part[:-3]
    out.insert(0, int_part)
    return (".".join(out) + "," + frac) if rng.random() < 0.75 else (".".join(out))


def build(count, seed):
    rng = random.Random(seed)
    seen = set()
    rows = []
    while len(rows) < count:
        frame = rng.choice(FRAMES)
        # 900-99,999.99: 5-13 word spans; keep some sub-1000 for the boundary.
        if rng.random() < 0.1:
            v = money(rng, 120, 899)
        elif rng.random() < 0.55:
            v = money(rng, 900, 9999)
        else:
            v = money(rng, 10000, 99999)
        text = frame.replace("{v}", "R$ " + v)
        if text in seen:
            continue
        seen.add(text)
        rows.append((f"broca03_{len(rows):05d}", 3, "train", text, "money"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = ap.parse_args()

    rows = build(args.count, args.seed)
    table = pa.table({
        "id": [r[0] for r in rows],
        "batch": [r[1] for r in rows],
        "split": [r[2] for r in rows],
        "text": [r[3] for r in rows],
        "source": [r[4] for r in rows],
    })
    out = args.corpus / "batch3"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "train.parquet")
    print(f"wrote {len(rows)} money sentences to {out / 'train.parquet'}")
    for r in rows[:3]:
        print(" ", r[3])


if __name__ == "__main__":
    main()
