"""Real-data regressions for classifier splitting and checkpoint continuation."""

from __future__ import annotations

import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from scenicdriver.classifier import model as classifier_model
from scenicdriver.classifier import train as training
from scenicdriver.classifier.inference import RESISC45_MEAN, RESISC45_STD, PreprocessingContract
from scenicdriver.scenic_scorer import regression


@pytest.fixture
def image_dataset(tmp_path):
    def create(name="images", counts=3):
        root = tmp_path / name
        for label, class_name in enumerate(classifier_model.TERRAIN_CLASSES):
            class_dir = root / class_name
            class_dir.mkdir(parents=True)
            count = counts if isinstance(counts, int) else counts[label]
            for sample in range(count):
                color = ((label * 5 + sample * 17) % 256, label * 3, sample * 70)
                Image.new("RGB", (8, 8), color).save(class_dir / f"sample-{sample}.png")
        return root

    return create


@pytest.fixture(autouse=True)
def small_transforms(monkeypatch):
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    monkeypatch.setattr(training, "get_training_transform", lambda _preprocessing: transform)
    monkeypatch.setattr(training, "get_inference_transform", lambda _preprocessing: transform)


@pytest.fixture
def tiny_classifier(monkeypatch):
    instances = []

    class TinyClassifier(nn.Module):
        def __init__(self, num_classes, pretrained, device):
            super().__init__()
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6), nn.Tanh())
            self.classifier = nn.Linear(6, num_classes)
            self.initial_parameters = {
                name: parameter.detach().clone() for name, parameter in self.named_parameters()
            }
            instances.append(self)

        def forward(self, images):
            return self.classifier(self.backbone(images))

        def get_model_info(self):
            count = sum(parameter.numel() for parameter in self.parameters())
            return {"architecture": "tiny", "total_params": count, "trainable_params": count}

    monkeypatch.setattr(classifier_model, "LandscapeClassifier", TinyClassifier)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield instances
    torch.set_num_threads(previous_threads)


def _members(loader, root):
    subset = loader.dataset
    return {
        (Path(subset.dataset.samples[index][0]).relative_to(root).as_posix(),
         subset.dataset.samples[index][1])
        for index in subset.indices
    }


def _run_training(root, output, **kwargs):
    training.train(
        root,
        output,
        batch_size=256,
        num_workers=0,
        device="cpu",
        learning_rate=0.02,
        label_smoothing=0.0,
        **kwargs,
    )


def test_split_seed_changes_membership_reproducibly(image_dataset):
    root = image_dataset()
    first_train, first_val = training.create_dataloaders(root, num_workers=0, seed=73)
    repeat_train, repeat_val = training.create_dataloaders(root, num_workers=0, seed=73)
    other_train, other_val = training.create_dataloaders(root, num_workers=0, seed=42)

    assert _members(first_train, root) == _members(repeat_train, root)
    assert _members(first_val, root) == _members(repeat_val, root)
    assert _members(first_train, root) != _members(other_train, root)
    assert _members(first_val, root) != _members(other_val, root)
    assert _members(first_train, root).isdisjoint(_members(first_val, root))
    assert len(_members(first_train, root) | _members(first_val, root)) == 135


@pytest.mark.parametrize("folder_change", ["rename", "missing", "extra"])
def test_rejects_noncanonical_class_folders(image_dataset, folder_change):
    root = image_dataset(counts=2)
    first_class = root / classifier_model.TERRAIN_CLASSES[0]
    if folder_change == "rename":
        first_class.rename(root / "not_a_terrain_class")
    elif folder_change == "missing":
        for image in first_class.iterdir():
            image.unlink()
        first_class.rmdir()
    else:
        extra = root / "not_a_terrain_class"
        extra.mkdir()
        Image.new("RGB", (8, 8)).save(extra / "sample.png")

    with pytest.raises(ValueError):
        training.create_dataloaders(root, num_workers=0)


@pytest.mark.parametrize("val_split", [-0.1, 0.0, 1.0, 1.1])
def test_rejects_invalid_validation_fraction(image_dataset, val_split):
    root = image_dataset(counts=2)
    with pytest.raises(ValueError):
        training.create_dataloaders(root, num_workers=0, val_split=val_split)


@pytest.mark.parametrize("val_split", [0.01, 0.99])
def test_every_class_retains_training_and_validation_samples(image_dataset, val_split):
    root = image_dataset()
    train_loader, val_loader = training.create_dataloaders(
        root, num_workers=0, val_split=val_split
    )
    expected_labels = set(range(len(classifier_model.TERRAIN_CLASSES)))
    assert {label for _, label in _members(train_loader, root)} == expected_labels
    assert {label for _, label in _members(val_loader, root)} == expected_labels


def test_training_dataset_smaller_than_batch_yields_real_batch(image_dataset):
    root = image_dataset(counts=2)
    train_loader, _ = training.create_dataloaders(root, batch_size=256, num_workers=0)
    batches = list(train_loader)
    assert len(batches) == 1
    images, labels = batches[0]
    assert images.shape == (45, 3, 8, 8)
    assert set(labels.tolist()) == set(range(45))


def test_singleton_class_stays_in_training(image_dataset):
    root = image_dataset(counts=[1] + [2] * 44)
    train_loader, val_loader = training.create_dataloaders(root, num_workers=0)
    assert {label for _, label in _members(train_loader, root)} == set(range(45))
    assert {label for _, label in _members(val_loader, root)} == set(range(1, 45))
    assert ("airplane/sample-0.png", 0) in _members(train_loader, root)


def test_all_singletons_reject_empty_validation_set(image_dataset):
    root = image_dataset(counts=1)
    with pytest.raises(ValueError):
        training.create_dataloaders(root, num_workers=0)


def test_missing_resume_fails_before_model_construction(image_dataset, tmp_path, tiny_classifier):
    root = image_dataset(counts=2)
    with pytest.raises(FileNotFoundError):
        _run_training(root, tmp_path / "output", epochs=1, resume=tmp_path / "missing.pt")
    assert tiny_classifier == []


def test_training_seed_matches_actual_split_and_updates_parameters(
    image_dataset, tmp_path, tiny_classifier, monkeypatch
):
    root = image_dataset()
    expected_train, expected_val = training.create_dataloaders(root, num_workers=0, seed=73)
    expected = (_members(expected_train, root), _members(expected_val, root))
    original_create = training.create_dataloaders
    observed = []

    def capture_actual_split(*args, **kwargs):
        loaders = original_create(*args, **kwargs)
        observed.append(tuple(_members(loader, Path(args[0])) for loader in loaders))
        return loaders

    monkeypatch.setattr(training, "create_dataloaders", capture_actual_split)
    output = tmp_path / "seed-73"
    _run_training(root, output, epochs=1, warmup_epochs=2, seed=73)
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)

    assert observed == [expected]
    assert checkpoint["split_identity"]["seed"] == 73
    identity = checkpoint["split_identity"]["sha256"]
    assert len(identity) == 64
    int(identity, 16)
    instance = tiny_classifier[0]
    for group in ("backbone", "classifier"):
        assert any(
            not torch.equal(parameter.detach(), instance.initial_parameters[name])
            for name, parameter in instance.named_parameters()
            if name.startswith(group + ".")
        )
    assert math.isfinite(checkpoint["history"]["train_loss"][0])
    assert math.isfinite(checkpoint["history"]["val_loss"][0])

    relocated = image_dataset(name="relocated")
    _run_training(relocated, tmp_path / "repeat", epochs=1, warmup_epochs=2, seed=73)
    _run_training(root, tmp_path / "other", epochs=1, warmup_epochs=2, seed=42)
    repeated = torch.load(tmp_path / "repeat/checkpoint.pt", weights_only=True)
    other = torch.load(tmp_path / "other/checkpoint.pt", weights_only=True)
    assert repeated["split_identity"] == checkpoint["split_identity"]
    assert other["split_identity"]["sha256"] != identity

    sample = relocated / "airplane/sample-0.png"
    sample.rename(sample.with_name("renamed.png"))
    _run_training(relocated, tmp_path / "renamed", epochs=1, warmup_epochs=2, seed=73)
    renamed = torch.load(tmp_path / "renamed/checkpoint.pt", weights_only=True)
    assert renamed["split_identity"]["seed"] == 73
    assert renamed["split_identity"]["sha256"] != identity


def test_cosine_checkpoint_supports_weights_only_loading(image_dataset, tmp_path, tiny_classifier):
    root = image_dataset(counts=2)
    output = tmp_path / "cosine"
    _run_training(root, output, epochs=3, warmup_epochs=1, seed=73)
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["history"]["lr"] == pytest.approx([0.002, 0.002, 0.001])
    assert all(type(rate) is float for rate in checkpoint["history"]["lr"])


def test_resume_preserves_history_and_full_warmup_cosine_schedule(
    image_dataset, tmp_path, tiny_classifier, monkeypatch
):
    root = image_dataset(counts=2)
    full_output = tmp_path / "full"
    resumed_output = tmp_path / "resumed"
    _run_training(root, full_output, epochs=5, warmup_epochs=2, seed=73)
    full = torch.load(full_output / "checkpoint.pt", weights_only=True)
    original_train_epoch = training.train_epoch
    completed = 0

    class ControlledInterruption(Exception):
        pass

    def interrupt_before_third_epoch(*args, **kwargs):
        nonlocal completed
        if completed == 2:
            raise ControlledInterruption
        loss = original_train_epoch(*args, **kwargs)
        completed += 1
        return loss

    monkeypatch.setattr(training, "train_epoch", interrupt_before_third_epoch)
    with pytest.raises(ControlledInterruption):
        _run_training(root, resumed_output, epochs=5, warmup_epochs=2, seed=73)
    prefix = torch.load(resumed_output / "checkpoint.pt", weights_only=True)
    assert prefix["epoch"] == 1
    for key in ("train_loss", "val_loss", "val_acc", "lr"):
        assert prefix["history"][key] == full["history"][key][:2]

    monkeypatch.setattr(training, "train_epoch", original_train_epoch)
    _run_training(
        root, resumed_output, epochs=5, warmup_epochs=2, seed=73,
        resume=resumed_output / "checkpoint.pt",
    )
    resumed = torch.load(resumed_output / "checkpoint.pt", weights_only=True)
    expected_rates = [0.001, 0.002, 0.002, 0.0015, 0.0005]
    assert full["history"]["lr"] == pytest.approx(expected_rates)
    assert resumed["history"]["lr"] == pytest.approx(expected_rates)
    for key in ("train_loss", "val_loss", "val_acc", "lr"):
        assert resumed["history"][key][:2] == prefix["history"][key]
        assert len(resumed["history"][key]) == 5
    assert resumed["scheduler_state_dict"] == full["scheduler_state_dict"]
    assert resumed["split_identity"] == prefix["split_identity"] == full["split_identity"]
    persisted_history = json.loads((resumed_output / "training_history.json").read_text())
    assert persisted_history == resumed["history"]


def test_checkpoint_history_guard_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="history"):
        training._history_from_checkpoint({})
    with pytest.raises(ValueError, match="history"):
        training._history_from_checkpoint({"history": [0.1, 0.2]})
    for missing in ("train_loss", "val_loss", "val_acc", "lr"):
        incomplete = {
            "history": {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        }
        del incomplete["history"][missing]
        with pytest.raises(ValueError, match="history"):
            training._history_from_checkpoint(incomplete)

    complete = {"train_loss": [1.0], "val_loss": [0.5], "val_acc": [0.25], "lr": [0.001]}
    assert training._history_from_checkpoint({"history": complete}) is complete


def test_split_identity_guard_requires_recorded_and_matching_identity():
    identity = {"seed": 73, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="split_identity"):
        training._verify_split_identity({}, identity)
    with pytest.raises(ValueError, match="split_identity"):
        training._verify_split_identity(
            {"split_identity": {"seed": 73, "sha256": "1" * 64}}, identity
        )
    training._verify_split_identity({"split_identity": dict(identity)}, identity)


@pytest.mark.parametrize(
    "tamper", ["missing_history", "not_dict_history", "missing_split_identity"]
)
def test_resume_rejects_malformed_checkpoint(image_dataset, tmp_path, tiny_classifier, tamper):
    root = image_dataset(counts=2)
    output = tmp_path / tamper
    _run_training(root, output, epochs=1, warmup_epochs=2, seed=73)
    path = output / "checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    if tamper == "missing_history":
        checkpoint.pop("history")
        expected = "history"
    elif tamper == "not_dict_history":
        checkpoint["history"] = [0.1]
        expected = "history"
    else:
        checkpoint.pop("split_identity")
        expected = "split_identity"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=expected):
        _run_training(root, output, epochs=2, warmup_epochs=2, seed=73, resume=path)


@pytest.mark.parametrize("change", ["renamed_sample", "different_seed"])
def test_resume_rejects_dataset_split_change(image_dataset, tmp_path, tiny_classifier, change):
    root = image_dataset(counts=2)
    output = tmp_path / change
    _run_training(root, output, epochs=1, warmup_epochs=2, seed=73)
    path = output / "checkpoint.pt"
    if change == "renamed_sample":
        sample = root / "airplane/sample-0.png"
        sample.rename(sample.with_name("renamed.png"))
        resume_seed = 73
    else:
        resume_seed = 42

    with pytest.raises(ValueError, match="split_identity"):
        _run_training(root, output, epochs=2, warmup_epochs=2, seed=resume_seed, resume=path)


@pytest.fixture
def tiny_vit(monkeypatch):
    """Real LandscapeClassifier with a size-agnostic tiny backbone in place of ViT-B/16."""

    def create_model(*_args, **_kwargs):
        backbone = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(12, 6))
        backbone.num_features = 6
        return backbone

    monkeypatch.setattr(classifier_model.timm, "create_model", create_model)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def _export_argv(labels, raw_dir, checkpoint, output, *extra):
    return [
        "export", "--labels-csv", str(labels), "--raw-dir", str(raw_dir),
        "--classifier-ckpt", str(checkpoint), "--output", str(output), "--device", "cpu",
        *extra,
    ]


def test_train_save_load_export_and_score_derive_one_normalization(
    image_dataset, tmp_path, tiny_vit, monkeypatch
):
    from scenicdriver.active_learning import scoring
    from scenicdriver.heuristics.labeler import _load_classifier
    from scripts.modeling import export_regression_dataset as exporter

    root = image_dataset(counts=2)
    output = tmp_path / "trained"
    _run_training(root, output, epochs=1, warmup_epochs=2, seed=73)
    checkpoint = output / "checkpoint.pt"
    expected = PreprocessingContract.for_normalization(
        "resisc45", class_names=classifier_model.TERRAIN_CLASSES
    )

    # Training defaults to RESISC45 and records the full contract as primitives.
    raw = torch.load(checkpoint, weights_only=True)
    assert raw["checkpoint_format"] == classifier_model.CHECKPOINT_FORMAT
    assert raw["preprocessing"] == {
        "normalization": "resisc45",
        "mean": list(RESISC45_MEAN),
        "std": list(RESISC45_STD),
        "image_size": 224,
        "class_names": classifier_model.TERRAIN_CLASSES,
    }
    assert classifier_model.load_classifier_checkpoint(checkpoint)[1] == expected

    # Exporter derives its transform from the checkpoint.
    raw_dir = tmp_path / "raw"
    for kind in ("satellite", "terrain"):
        (raw_dir / "images" / kind).mkdir(parents=True)
        Image.new("RGB", (8, 8), (40, 90, 20)).save(raw_dir / "images" / kind / "a.png")
    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,scenic_score\nimages/satellite/a.png,5\n", encoding="utf-8")
    seen = []
    real_transform = exporter.get_inference_transform

    def spy_transform(preprocessing):
        seen.append(preprocessing)
        return real_transform(preprocessing)

    monkeypatch.setattr(exporter, "get_inference_transform", spy_transform)
    monkeypatch.setattr(
        exporter, "_terrain_features_from_images", lambda *_args: np.zeros(6, dtype=np.float32)
    )
    exported = tmp_path / "dataset.npz"
    monkeypatch.setattr(sys, "argv", _export_argv(labels, raw_dir, checkpoint, exported))
    exporter.main()
    assert seen == [expected]
    with np.load(exported) as dataset:
        assert dataset["vit_embeddings"].shape == (1, 6)
        assert dataset["class_logits"].shape == (1, 45)

    # A conflicting CLI override fails and publishes nothing.
    conflicting = tmp_path / "conflicting.npz"
    monkeypatch.setattr(
        sys, "argv",
        _export_argv(labels, raw_dir, checkpoint, conflicting, "--normalization", "imagenet"),
    )
    with pytest.raises(ValueError, match="conflicts"):
        exporter.main()
    assert not conflicting.exists()

    # A failed export (zero samples) never leaves an output at the canonical path.
    missing_labels = tmp_path / "missing.csv"
    missing_labels.write_text(
        "image_path,scenic_score\nimages/satellite/absent.png,5\n", encoding="utf-8"
    )
    failed = tmp_path / "failed.npz"
    monkeypatch.setattr(
        sys, "argv", _export_argv(missing_labels, raw_dir, checkpoint, failed, "--skip-missing")
    )
    with pytest.raises(ValueError, match="No samples exported"):
        exporter.main()
    assert not failed.exists()
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith(".")) == []

    # Active scorer derives transform, class order, and cache identity from it.
    regression_path = tmp_path / "regression.pt"
    regression_model = regression.ScenicRegressionModel(vit_dim=6, terrain_dim=6, num_classes=45)
    torch.save(
        {
            "model_state_dict": regression_model.state_dict(),
            "vit_dim": 6, "terrain_dim": 6, "num_classes": 45,
        },
        regression_path,
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": str(regression_path)}}))
    loaded = scoring.load_scoring_models(
        registry_path=registry, classifier_checkpoint=checkpoint, device="cpu"
    )
    assert loaded.class_names == tuple(classifier_model.TERRAIN_CLASSES)
    assert loaded.classifier_preprocess_identity == scoring.classifier_preprocess_identity(expected)
    normalize = loaded.classifier_transform.transforms[-1]
    assert tuple(normalize.mean) == RESISC45_MEAN and tuple(normalize.std) == RESISC45_STD
    with pytest.raises(ValueError, match="conflicts"):
        scoring.load_scoring_models(
            registry_path=registry, classifier_checkpoint=checkpoint, device="cpu",
            classifier_normalization="imagenet",
        )

    # Heuristic labeler inference derives the same contract.
    _, transform, _, class_names = _load_classifier(
        use_classifier=True, classifier_best_ckpt=checkpoint, device="cpu", warnings=[]
    )
    assert class_names == classifier_model.TERRAIN_CLASSES
    assert tuple(transform.transforms[-1].mean) == RESISC45_MEAN


def test_legacy_bare_checkpoint_rejected_with_guidance_and_explicit_migration(
    tmp_path, tiny_vit
):
    legacy = tmp_path / "legacy.pt"
    torch.save(
        classifier_model.LandscapeClassifier(pretrained=False, device="cpu").state_dict(), legacy
    )
    with pytest.raises(ValueError, match="legacy bare state dicts.*Retrain.*migrate"):
        classifier_model.LandscapeClassifier(pretrained=False, pretrained_path=legacy, device="cpu")

    migrated = tmp_path / "migrated.pt"
    classifier_model.migrate_legacy_checkpoint(legacy, migrated, normalization="imagenet")
    loaded = classifier_model.LandscapeClassifier(
        pretrained=False, pretrained_path=migrated, device="cpu"
    )
    assert loaded.preprocessing.normalization == "imagenet"


def test_classifier_checkpoint_load_refuses_pickle_code_execution(tmp_path, tiny_vit):
    marker = tmp_path / "executed"
    checkpoint = tmp_path / "malicious.pt"
    torch.save(
        {
            "checkpoint_format": classifier_model.CHECKPOINT_FORMAT,
            "checkpoint_schema_version": classifier_model.CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": {},
            "payload": _MaliciousPayload(marker),
        },
        checkpoint,
    )
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        classifier_model.load_classifier_checkpoint(checkpoint)
    assert not marker.exists()


def test_interrupted_checkpoint_write_keeps_previous_checkpoint(
    image_dataset, tmp_path, tiny_classifier, monkeypatch
):
    root = image_dataset(counts=2)
    output = tmp_path / "atomic"
    _run_training(root, output, epochs=1, warmup_epochs=2, seed=73)
    previous = (output / "checkpoint.pt").read_bytes()

    def interrupted_save(value, handle):
        handle.write(b"partial checkpoint")
        raise OSError("killed mid-write")

    monkeypatch.setattr(regression.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="killed mid-write"):
        _run_training(
            root, output, epochs=3, warmup_epochs=2, seed=73, resume=output / "checkpoint.pt"
        )
    monkeypatch.undo()
    assert (output / "checkpoint.pt").read_bytes() == previous
    assert not [path for path in output.iterdir() if path.name.endswith(".tmp")]


def test_resume_restores_rng_streams_like_continuous_run(
    image_dataset, tmp_path, tiny_classifier, monkeypatch
):
    base = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])

    def stochastic(image):
        tensor = base(image)
        return (
            tensor
            + 0.1 * torch.rand_like(tensor)
            + float(np.random.rand()) * 0.01
            + random.random() * 0.01
        )

    monkeypatch.setattr(training, "get_training_transform", lambda _preprocessing: stochastic)
    root = image_dataset(counts=2)
    _run_training(root, tmp_path / "full", epochs=4, warmup_epochs=1, seed=73)
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)

    original_train_epoch = training.train_epoch
    completed = 0

    class Interrupted(Exception):
        pass

    def interrupt_before_third_epoch(*args, **kwargs):
        nonlocal completed
        if completed == 2:
            raise Interrupted
        completed += 1
        return original_train_epoch(*args, **kwargs)

    monkeypatch.setattr(training, "train_epoch", interrupt_before_third_epoch)
    resumed_output = tmp_path / "resumed"
    with pytest.raises(Interrupted):
        _run_training(root, resumed_output, epochs=4, warmup_epochs=1, seed=73)
    prefix = torch.load(resumed_output / "checkpoint.pt", weights_only=True)
    assert prefix["epoch"] == 1
    for key in ("python", "numpy", "torch"):
        assert key in prefix["rng_state"]

    monkeypatch.setattr(training, "train_epoch", original_train_epoch)
    with pytest.raises(ValueError, match="conflicts"):
        _run_training(
            root, resumed_output, epochs=4, warmup_epochs=1, seed=73,
            normalization="imagenet", resume=resumed_output / "checkpoint.pt",
        )
    _run_training(
        root, resumed_output, epochs=4, warmup_epochs=1, seed=73,
        resume=resumed_output / "checkpoint.pt",
    )
    resumed = torch.load(resumed_output / "checkpoint.pt", weights_only=True)

    assert torch.equal(resumed["rng_state"]["torch"], full["rng_state"]["torch"])
    assert torch.equal(resumed["rng_state"]["numpy"]["keys"], full["rng_state"]["numpy"]["keys"])
    assert resumed["rng_state"]["numpy"]["pos"] == full["rng_state"]["numpy"]["pos"]
    assert resumed["rng_state"]["python"] == full["rng_state"]["python"]
    assert torch.equal(resumed["loader_generator_state"], full["loader_generator_state"])
    for name, tensor in full["model_state_dict"].items():
        assert torch.equal(resumed["model_state_dict"][name], tensor), name
    assert resumed["history"] == full["history"]
