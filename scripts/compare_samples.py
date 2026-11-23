#!/usr/bin/env python
"""Side-by-side listening page: v0 (front end v1) against v1 (front end v2).

    python scripts/compare_samples.py --v1 runs/tiny-broca-v1/inference.pt --out runs/compare-v0-v1

Writes <out>/audio/*.wav, <out>/index.html (both voices, both models, and
what parakeet hears in each clip) and <out>/compare.json. v0 reads with the
v0 front end from git tag v0 (runs/v0-snapshot), exactly as released.
"""
import argparse
import html
import importlib.util
import json
import subprocess
from pathlib import Path

import soundfile as sf
import torch

from tiny_broca.infer import load_model
from tiny_broca.text import normalize_text, phonemize

SENTENCES = (
    ("Os quatro problemas relatados", "Vou comprar o ingresso online hoje."),
    ("Os quatro problemas relatados", "Se 4 > 3, então 3 < 4."),
    ("Os quatro problemas relatados", "O café custou R$3,00."),
    ("Os quatro problemas relatados", "A prova é dia 14/03."),
    ("Palavras em inglês", "Ele investiu em Bitcoin e estuda blockchain."),
    ("Palavras em inglês", "Acabou o shampoo, e o tradeoff foi comprar online."),
    ("Palavras em inglês", "Mandei o link do podcast pelo WhatsApp e pelo e-mail."),
    ("Palavras em inglês", "O delivery trouxe o mouse e o notebook."),
    ("Dinheiro", "Paguei R$ 0,50 no café e R$ 47,35 no almoço."),
    ("Dinheiro", "A conta do aluguel fechou em R$ 1.245,10."),
    ("Dinheiro", "A empresa faturou R$ 2,5 milhões e lucrou R$ 1 milhão."),
    ("Números longos", "No relatório de 2023, a receita foi de R$ 1.234.567,89, com crescimento de 12,5%."),
    ("Datas, horas e ordinais", "A reunião é às 14h30 do dia 1º de maio, no 21º andar."),
    ("Datas, horas e ordinais", "Das 2 às 4 da tarde, a 2ª turma faz a prova."),
    ("Siglas e abreviações", "O Sr. Silva levou o CPF, o RG e o comprovante ao INSS."),
    ("Siglas e abreviações", "Dom Pedro II governou o Brasil no século XIX."),
    ("Unidades e porcentagens", "Corri 5 km em 25:30, perdi 1,5 kg e a febre baixou para 36,6°C."),
    ("Tônica", "O urubu de Itu comeu caju e abacaxi aqui no jardim."),
    ("Tônica", "O juiz tirou a raiz do problema: os reais sumiram."),
    ("Perguntas", "Você vai à festa amanhã?"),
    ("Frase comum", "Depois da tempestade, o ar da montanha parecia fresco e limpo."),
)


def v0_front_end():
    spec = importlib.util.spec_from_file_location("text_v0", "runs/v0-snapshot/src/tiny_broca/text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthesize(model, phones, voice):
    args = (torch.tensor([phones.ids]), torch.tensor([phones.stress]), torch.tensor([len(phones)]),
            torch.tensor([model.config.speakers.index(voice)]))
    audio, lengths, _ = model.synthesize(*args, seed=0, unvoiced_noise=getattr(model, "unvoiced_noise", False))
    return audio[0, :int(lengths[0])].numpy()


def heard(path):
    result = subprocess.run(["fscript", str(path), "--model", "parakeet-v3-int8", "--text=plain", "-D", "--raw", "--stdout"],
                            capture_output=True, text=True)
    if result.returncode:  # an empty transcript would reject the clip for good
        raise RuntimeError(f"fscript failed on {path}: {(result.stderr or result.stdout).strip()[:300]}")
    return " ".join(result.stdout.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v0", type=Path, default=Path("runs/tiny-broca-ft2/inference.pt"))
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/compare-v0-v1"))
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    old = v0_front_end()
    models = {"v0": load_model(args.v0, check_frontend=False), "v1": load_model(args.v1, check_frontend=False)}
    rows = []
    for number, (group, text) in enumerate(SENTENCES):
        row = {"group": group, "text": text, "v0_reads": old.normalize_text(text), "v1_reads": normalize_text(text), "clips": {}}
        for voice in ("F1", "M1"):
            for name, model in models.items():
                phones = old.phonemize(text) if name == "v0" else phonemize(text)
                path = args.out / "audio" / f"{number:02d}_{voice}_{name}.wav"
                sf.write(path, synthesize(model, phones, voice), model.config.sample_rate, subtype="PCM_16")
                row["clips"][f"{voice}_{name}"] = {"file": f"audio/{path.name}",
                                                  "heard": "" if args.no_asr else heard(path)}
        rows.append(row)
        print(number, text, flush=True)
    (args.out / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    write_page(rows, args.out / "index.html")


def write_page(rows, path):
    cells = []
    group = None
    for row in rows:
        if row["group"] != group:
            group = row["group"]
            cells.append(f'<tr class="group"><th colspan="3">{html.escape(group)}</th></tr>')
        audio = []
        for voice in ("F1", "M1"):
            for name in ("v0", "v1"):
                clip = row["clips"][f"{voice}_{name}"]
                heard_text = f'<div class="heard">ouvido: {html.escape(clip["heard"])}</div>' if clip["heard"] else ""
                audio.append(f'<div class="clip"><span class="tag {name}">{name} {voice}</span>'
                             f'<audio controls preload="none" src="{clip["file"]}"></audio>{heard_text}</div>')
        reads = (f'<div class="reads"><b>v0 lê:</b> {html.escape(row["v0_reads"])}</div>'
                 f'<div class="reads"><b>v1 lê:</b> {html.escape(row["v1_reads"])}</div>')
        cells.append(f'<tr><td class="text">{html.escape(row["text"])}{reads}</td>'
                     f'<td>{"".join(audio[:2])}</td><td>{"".join(audio[2:])}</td></tr>')
    path.write_text(f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>tiny-broca: v0 × v1</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; margin: 24px; color: #1d1d1f; background: #fafafa; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} p.lead {{ color: #555; margin: 0 0 18px; max-width: 70ch; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border-bottom: 1px solid #e5e5e5; padding: 10px; vertical-align: top; text-align: left; }}
tr.group th {{ background: #f0f0f3; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #444; }}
td.text {{ width: 34%; font-weight: 600; }} .reads {{ font-weight: 400; color: #555; font-size: 13px; margin-top: 6px; }}
.clip {{ margin-bottom: 10px; }} audio {{ width: 100%; height: 32px; }}
.tag {{ font-size: 12px; font-weight: 700; padding: 1px 6px; border-radius: 4px; }}
.tag.v0 {{ background: #eee; color: #555; }} .tag.v1 {{ background: #e3f1e5; color: #1f6b2c; }}
.heard {{ font-size: 12px; color: #777; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #161618; color: #eee; }} table {{ background: #1e1e21; }}
  th, td {{ border-color: #333; }} tr.group th {{ background: #26262a; color: #bbb; }} .reads, .heard {{ color: #aaa; }} }}
</style></head><body>
<h1>tiny-broca: v0 × v1</h1>
<p class="lead">Mesmas frases nas duas vozes. "v0 lê" e "v1 lê" mostram o texto que cada front end entrega ao modelo;
"ouvido" é a transcrição automática (parakeet) de cada clipe.</p>
<table><tr><th>Frase</th><th>F1</th><th>M1</th></tr>
{''.join(cells)}
</table></body></html>
""", encoding="utf-8")


if __name__ == "__main__":
    main()
