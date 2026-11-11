# -*- coding: utf-8 -*-
"""Download the real-MoE assets used by the ISPA revision (model + corpora).

Model weights go to the default HF cache on C: (D: is nearly full).
"""
import os, sys, time
from huggingface_hub import snapshot_download, hf_hub_download

t0 = time.time()
here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(here, "corpora")
os.makedirs(data_dir, exist_ok=True)
for repo, fn in [("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
                 ("openai/gsm8k", "main/test-00000-of-00001.parquet")]:
    p = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset",
                        local_dir=os.path.join(data_dir, repo.replace("/", "__")))
    print("corpus", repo, p, flush=True)
p = snapshot_download("microsoft/Phi-tiny-MoE-instruct",
                      allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model"])
print("model", p, flush=True)
print("DONE %.0fs" % (time.time() - t0), flush=True)
