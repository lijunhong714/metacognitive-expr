"""Question-level sequences used only by AKT_LPKT."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AKTLPKTQueDataset(Dataset):
    def __init__(self, file_path, folds, max_concepts, at2idx, it2idx):
        self.max_concepts = max_concepts
        folds = sorted(folds)
        # Independent cache version: no native AKT/LPKT/KTQueDataset cache is read or replaced.
        cache_path = file_path + "_" + "_".join(map(str, folds)) + "_akt_lpkt_qv1.pkl"
        if os.path.exists(cache_path):
            self.dori = pd.read_pickle(cache_path)
        else:
            self.dori = self._load(file_path, folds, at2idx, it2idx)
            pd.to_pickle(self.dori, cache_path)

    def _load(self, file_path, folds, at2idx, it2idx):
        df = pd.read_csv(file_path)
        df = df[df["fold"].isin(folds)]
        rows = {key: [] for key in ("qseqs", "cseqs", "rseqs", "tseqs", "utseqs", "itseqs", "smasks")}
        has_duration = "usetimes" in df.columns
        for _, row in df.iterrows():
            q = [int(x) for x in row["questions"].split(",")]
            r = [int(x) for x in row["responses"].split(",")]
            concepts = []
            for item in row["concepts"].split(","):
                kc = [] if item == "-1" else [int(x) for x in item.split("_")]
                if len(kc) > self.max_concepts:
                    raise ValueError(f"KC count exceeds max_concepts={self.max_concepts}")
                concepts.append(kc + [-1] * (self.max_concepts - len(kc)))
            if "timestamps" in df.columns:
                times = [int(float(x)) for x in row["timestamps"].split(",")]
                previous = times[:1] + times[:-1]
                gaps = np.clip((np.array(times) - np.array(previous)) // 60000, -1, 43200)
            else:
                times = [0] * len(q)
                gaps = [1] * len(q)
            intervals = [it2idx.get(str(x), it2idx["-1"]) for x in gaps]
            if has_duration:
                durations = [int(float(x)) // 1000 for x in row["usetimes"].split(",")]
                durations = [at2idx.get(str(x), at2idx["-1"]) for x in durations]
                rows["utseqs"].append(durations)
            for key, value in (("qseqs", q), ("cseqs", concepts), ("rseqs", r),
                               ("tseqs", times), ("itseqs", intervals),
                               ("smasks", [int(x) for x in row["selectmasks"].split(",")])):
                rows[key].append(value)
        if not rows["qseqs"]:
            raise ValueError(f"No AKT_LPKT rows for folds {folds}: {file_path}")
        for key in rows:
            if key != "utseqs" or has_duration:
                rows[key] = torch.tensor(rows[key], dtype=torch.float if key == "rseqs" else torch.long)
        # One valid question produces one target; KC padding never defines the loss mask.
        rows["masks"] = (rows["rseqs"][:, :-1] != -1) & (rows["rseqs"][:, 1:] != -1)
        rows["smasks"] = rows["smasks"][:, 1:] != -1
        return rows

    def __len__(self):
        return self.dori["rseqs"].size(0)

    def __getitem__(self, index):
        mask = self.dori["masks"][index]
        result = {"masks": mask, "smasks": self.dori["smasks"][index]}
        for key in ("qseqs", "cseqs", "rseqs", "tseqs", "utseqs", "itseqs"):
            values = self.dori[key]
            if key == "utseqs" and isinstance(values, list):
                result[key] = []
                result["shft_" + key] = []
                continue
            values = values[index]
            # cseqs is [T,K]; retain -1 KC padding. Other series are [T].
            if key == "cseqs":
                result[key], result["shft_" + key] = values[:-1], values[1:]
            else:
                result[key] = values[:-1] * mask
                result["shft_" + key] = values[1:] * mask
        return result
