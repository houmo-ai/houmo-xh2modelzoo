from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel
from xhmodel_merak.xh_llm.models.emotion2vec.iemocap_protocol import (
    IEMOCAP_LABELS,
    IEMOCAP_SESSION_COUNTS,
    LABEL_TO_INDEX,
    compute_metrics,
)


class IEMOCAPFeatureDataset(Dataset):
    def __init__(self, feature_dir: str | Path, manifest: list[dict]):
        self.feature_dir = Path(feature_dir)
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        item = self.manifest[index]
        features = np.load(self.feature_dir / f"{item['utterance_id']}.npy")
        return torch.from_numpy(features).float(), LABEL_TO_INDEX[item["label"]]


def collate_features(samples):
    features, labels = zip(*samples, strict=True)
    lengths = torch.tensor([feature.shape[0] for feature in features])
    padded = pad_sequence(features, batch_first=True)
    padding_mask = torch.arange(padded.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    return padded, padding_mask, torch.tensor(labels, dtype=torch.long)


class OfficialIEMOCAPClassifier(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 4):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, 256)
        self.post_net = nn.Linear(256, output_dim)
        self.activate = nn.ReLU()

    def forward(self, features, padding_mask):
        features = self.activate(self.pre_net(features))
        valid = (~padding_mask).unsqueeze(-1).to(features.dtype)
        features = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.post_net(features)


def build_manifest(iemocap_root: str | Path) -> list[dict]:
    root = Path(iemocap_root)
    if not root.exists():
        raise FileNotFoundError(root)
    manifest: list[dict] = []
    valid_labels = set(IEMOCAP_LABELS)
    for session_index in range(1, 6):
        session_dir = root / f"Session{session_index}"
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        wav_root = session_dir / "sentences" / "wav"
        if not eval_dir.is_dir() or not wav_root.is_dir():
            raise ValueError(f"incomplete IEMOCAP session directory: {session_dir}")
        session_items: list[dict] = []
        for eval_file in sorted(eval_dir.glob("*.txt")):
            for line in eval_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.startswith("[") or "\t" not in line:
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    continue
                utterance_id, raw_label = fields[1], fields[2]
                label = "hap" if raw_label == "exc" else raw_label
                if label not in valid_labels:
                    continue
                dialog_id = utterance_id.rsplit("_", 1)[0]
                wav_path = wav_root / dialog_id / f"{utterance_id}.wav"
                if not wav_path.is_file():
                    raise FileNotFoundError(wav_path)
                session_items.append(
                    {
                        "utterance_id": utterance_id,
                        "label": label,
                        "session": session_index,
                        "audio": str(wav_path),
                    }
                )
        session_items.sort(key=lambda item: item["utterance_id"])
        expected = IEMOCAP_SESSION_COUNTS[session_index - 1]
        if len(session_items) != expected:
            raise ValueError(
                f"Session{session_index} contains {len(session_items)} target utterances, expected {expected}"
            )
        manifest.extend(session_items)
    if len(manifest) != 5531:
        raise ValueError(f"IEMOCAP manifest contains {len(manifest)} utterances, expected 5531")
    return manifest


def extract_features(manifest: list[dict], model: Emotion2vecHMONNXModel, feature_dir: str | Path) -> None:
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest, start=1):
        output = feature_dir / f"{item['utterance_id']}.npy"
        if output.exists():
            continue
        result = model.extract_file(item["audio"])
        np.save(output, result["frame_features"].cpu().numpy())
        if index % 100 == 0:
            print(f"extracted {index}/{len(manifest)}")


def make_loaders(dataset, fold: int, batch_size: int, generator: torch.Generator):
    test_start = sum(IEMOCAP_SESSION_COUNTS[:fold])
    test_end = test_start + IEMOCAP_SESSION_COUNTS[fold]
    test_indices = list(range(test_start, test_end))
    train_val_indices = list(range(0, test_start)) + list(range(test_end, len(dataset)))
    train_count = int(0.8 * len(train_val_indices))
    val_count = len(train_val_indices) - train_count
    train_subset, val_subset = random_split(Subset(dataset, train_val_indices), [train_count, val_count], generator)
    test_subset = Subset(dataset, test_indices)
    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate_features, num_workers=0)
    return (
        DataLoader(train_subset, shuffle=True, **loader_kwargs),
        DataLoader(val_subset, shuffle=False, **loader_kwargs),
        DataLoader(test_subset, shuffle=False, **loader_kwargs),
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for features, padding_mask, labels in loader:
        features, padding_mask, labels = features.to(device), padding_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features, padding_mask), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for features, padding_mask, labels in loader:
        logits = model(features.to(device), padding_mask.to(device))
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return compute_metrics(targets, predictions, num_classes=len(IEMOCAP_LABELS))


def run_five_fold(manifest, feature_dir, batch_size, epochs, learning_rate, seed, output_dir, feature_dim=1024):
    torch.manual_seed(seed)
    split_generator = torch.Generator().manual_seed(seed)
    dataset = IEMOCAPFeatureDataset(feature_dir, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in range(5):
        train_loader, val_loader, test_loader = make_loaders(dataset, fold, batch_size, split_generator)
        model = OfficialIEMOCAPClassifier(input_dim=feature_dim).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=1e-3, step_size_up=10)
        criterion = nn.CrossEntropyLoss()
        best_wa = -1.0
        checkpoint = output_dir / f"model_{fold + 1}.pth"
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)
            if val_metrics["wa"] > best_wa:
                best_wa = val_metrics["wa"]
                torch.save(model.state_dict(), checkpoint)
            print(f"fold={fold + 1} epoch={epoch + 1} loss={loss:.6f} val={val_metrics}")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test_metrics = evaluate(model, test_loader, device)
        results.append(test_metrics)
        print(f"fold={fold + 1} test={test_metrics}")
    averages = {key: float(np.mean([result[key] for result in results])) for key in results[0]}
    report = {"folds": results, "average": averages}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap-root", required=True)
    parser.add_argument("--meta")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", default="work_dirs/emotion2vec_iemocap_eval")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.extract_only and args.evaluate_only:
        raise ValueError("--extract-only and --evaluate-only are mutually exclusive")
    manifest = build_manifest(args.iemocap_root)
    manifest_path = Path(args.feature_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.evaluate_only:
        if not args.meta:
            raise ValueError("--meta is required for feature extraction")
        meta = Emotion2vecModelMeta.from_json_file(args.meta)
        extract_features(manifest, Emotion2vecHMONNXModel(meta), args.feature_dir)
    if not args.extract_only:
        print(
            run_five_fold(
                manifest,
                args.feature_dir,
                args.batch_size,
                args.epochs,
                args.learning_rate,
                args.seed,
                args.output_dir,
                args.feature_dim,
            )
        )


if __name__ == "__main__":
    main()
