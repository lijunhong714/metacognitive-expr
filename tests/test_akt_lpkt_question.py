"""Small AKT_LPKT question-level integration checks, including causality."""

import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pykt.datasets.akt_lpkt_que_dataloader import AKTLPKTQueDataset
from pykt.models.akt_lpkt import AKTLPKT
from pykt.models.train_model import model_forward
from pykt.models.evaluate_model import evaluate


def run_case(k, has_duration):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train_valid_sequences_quelevel.csv"
        concepts = "_".join(map(str, range(k)))
        rows = []
        for responses in ("1,0,1,-1", "0,1,0,-1"):
            row = {
                "fold": 0,
                "questions": "0,1,2,-1",
                "concepts": f"{concepts},1,2,-1",
                "responses": responses,
                "selectmasks": "1,1,1,-1",
                "timestamps": "0,60000,120000,-1",
            }
            if has_duration:
                row["usetimes"] = "1000,2000,3000,-1"
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        mapping = {"0": 0, "1": 1, "2": 2, "-1": 3}
        dataset = AKTLPKTQueDataset(str(path), {0}, k, mapping, mapping)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        assert batch["cseqs"].shape == (2, 3, k)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c = torch.cat((batch["cseqs"][:, :1], batch["shft_cseqs"]), 1).to(device)
        q = torch.cat((batch["qseqs"][:, :1], batch["shft_qseqs"]), 1).to(device)
        r = torch.cat((batch["rseqs"][:, :1], batch["shft_rseqs"]), 1).long().to(device)
        it = torch.cat((batch["itseqs"][:, :1], batch["shft_itseqs"]), 1).to(device)
        at = None
        if has_duration:
            at = torch.cat((batch["utseqs"][:, :1], batch["shft_utseqs"]), 1).to(device)
        model = AKTLPKT(
            n_question=max(k, 3), n_pid=4, n_at=3, n_it=3, n_phi=0,
            d_model=16, n_blocks=1, dropout=0, d_ff=32,
            final_fc_dim=32, num_attn_heads=4, d_k=8,
            input_level="question",
        ).to(device)
        assert model.regulator.q_matrix is None
        assert model.qa_embed_diff.num_embeddings == 2
        mask = r.ne(-1)
        model.eval()
        with torch.no_grad():
            pred, _, aux = model(c, r, q, it, at, mask=mask, return_aux=True)
            flipped = r.clone()
            flipped[:, 1] = 1 - flipped[:, 1]
            pred_flipped, _ = model(c, flipped, q, it, at, mask=mask)
        assert pred.shape == q.shape  # [B,T]: exactly one prediction per question
        assert aux["omega"].shape == q.shape
        assert aux["pi"].shape[:2] == q.shape
        assert aux["delta_abs"].shape == q.shape
        assert aux["gamma_l"].shape[:2] == q.shape
        assert aux["gamma_f"].shape == (2, 4, max(k, 3) + 1)
        assert aux["h_concept"].shape == aux["gamma_f"].shape
        torch.testing.assert_close(pred[:, 1], pred_flipped[:, 1], rtol=0, atol=1e-7)
        assert batch["smasks"].sum().item() == 4  # two valid next questions per row
        model.train()
        result, reg = model(c, r, q, it, at, mask=mask)
        selection = batch["smasks"].to(device)
        loss = torch.nn.functional.binary_cross_entropy(
            result[:, 1:][selection],
            r[:, 1:][selection].float(),
        ) + reg
        loss.backward()
        assert torch.isfinite(loss)
        assert model.q_embed.weight.grad is not None
        # Check the actual shared dispatch points for this model name.
        dispatch_loss = model_forward(model, batch)
        assert torch.isfinite(dispatch_loss)
        auc, acc = evaluate(model, DataLoader(dataset, batch_size=2), "akt_lpkt")
        assert 0 <= auc <= 1 and 0 <= acc <= 1


def test_question_level_shapes_causality_and_gradients():
    for k, has_duration in ((1, True), (1, True), (7, False), (5, False)):
        run_case(k, has_duration)
