# -*- coding: utf-8 -*-
"""Record the top-2 routing decisions of a trained 16-expert MoE.

Model : microsoft/Phi-tiny-MoE-instruct (SlimMoE; 32 MoE layers, 16 experts,
        top-2 routing).  Router logits are captured with forward hooks on each
        layer's gate; in eval mode the sparse-mixer selects the two highest
        logits, so top-2 of the captured logits is the executed routing.
Data  : three domains -- WikiText-2 (test), GSM8K (test), and Python source
        code from this repository's src/ tree.  Each domain is cut into
        512-token windows (= 16 sources x 32 tokens, the replay batch).
Output: routing_counts.json  {domain: {layer: [[count_e for e in 0..15] per window]}}
"""
from __future__ import print_function
import glob, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = "microsoft/Phi-tiny-MoE-instruct"
WIN = 512
NWIN = int(os.environ.get("NWIN", "24"))
torch.set_num_threads(16)


def corpus_texts():
    import pandas as pd
    cdir = os.path.join(HERE, "corpora")
    wiki = pd.read_parquet(glob.glob(os.path.join(cdir, "Salesforce__wikitext", "**", "*.parquet"), recursive=True)[0])
    wiki_text = "".join(t for t in wiki["text"].tolist())
    gsm = pd.read_parquet(glob.glob(os.path.join(cdir, "openai__gsm8k", "**", "*.parquet"), recursive=True)[0])
    gsm_text = "\n\n".join("Question: %s\nAnswer: %s" % (q, a) for q, a in zip(gsm["question"], gsm["answer"]))
    code_files = sorted(glob.glob(os.path.join(ROOT, "src", "*.py")))
    code_text = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in code_files)
    return {"wiki": wiki_text, "math": gsm_text, "code": code_text}


def load_model():
    # Only the library's own PhiMoE implementation and safetensors weights are
    # used; the repository's remote code is never executed.
    from transformers import AutoTokenizer, PhimoeForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    model = PhimoeForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, use_safetensors=True,
        trust_remote_code=False)
    model.eval()
    return tok, model, "native-phimoe"


def gates(model):
    # transformers 5.x: each MoE block owns a PhimoeTopKRouter (an nn.Linear
    # subclass) that returns (logits, routing weights, selected experts).
    found = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "PhimoeTopKRouter":
            layer = int(name.split(".layers.")[1].split(".")[0])
            found.append((layer, mod))
    found.sort(key=lambda item: item[0])
    return found


def main():
    t0 = time.time()
    tok, model, how = load_model()
    gl = gates(model)
    print("loaded via", how, "gates:", len(gl), "%.0fs" % (time.time() - t0), flush=True)
    assert len(gl) == 32, len(gl)
    captured = {}

    def hook_for(layer):
        def hook(mod, inp, out):
            # out[2]: the experts the sparse mixer actually selected (top-2)
            captured[layer] = out[2].detach().reshape(-1, out[2].shape[-1])
        return hook
    for layer, mod in gl:
        mod.register_forward_hook(hook_for(layer))

    texts = corpus_texts()
    result = {"model": MODEL, "load": how, "window_tokens": WIN, "domains": {}}
    for dom, text in texts.items():
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        nwin = min(NWIN, ids.numel() // WIN)
        per_layer = {str(l): [] for l, _ in gl}
        for w in range(nwin):
            chunk = ids[w * WIN:(w + 1) * WIN].unsqueeze(0)
            captured.clear()
            with torch.no_grad():
                model(input_ids=chunk, use_cache=False)
            for l, _ in gl:
                top2 = captured[l]
                assert top2.shape == (WIN, 2), top2.shape
                cnt = torch.bincount(top2.reshape(-1), minlength=16).tolist()
                per_layer[str(l)].append(cnt)
            print(dom, "window", w + 1, "/", nwin, "%.0fs" % (time.time() - t0), flush=True)
        result["domains"][dom] = {"tokens": int(ids.numel()), "windows": nwin,
                                  "counts": per_layer}
        with open(os.path.join(ROOT, "data", "realtrace", "routing_counts.json"), "w") as fh:
            json.dump(result, fh)
    print("DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
