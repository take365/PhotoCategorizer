"""Zero-shot classification utilities for PhotoCategorizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel

import open_clip

from .utils import Config, ZeroShotConfig, iter_batches, resolve_device


@dataclass(slots=True)
class ZeroShotResult:
    """Model prediction bundle for a single image."""

    path: Path
    label: str
    score: float
    runner_up_label: str | None
    runner_up_score: float
    margin: float
    needs_review: bool
    scores: Dict[str, float]


class _SiglipBackend:
    """Wrapper around HuggingFace SigLIP for zero-shot scoring."""

    DEFAULT_CHECKPOINT = "google/siglip-base-patch16-224"

    def __init__(self, checkpoint: str | None, device: torch.device) -> None:
        self.device = device
        ckpt = checkpoint or self.DEFAULT_CHECKPOINT
        self.model = SiglipModel.from_pretrained(ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(prompts), padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs)
        return _normalize(text_features)

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        return _normalize(image_features)


class _OpenClipBackend:
    """Wrapper around OpenCLIP."""

    DEFAULT_MODEL = "ViT-B-32"
    DEFAULT_CHECKPOINT = "laion2b_s34b_b79k"

    def __init__(self, checkpoint: str | None, device: torch.device) -> None:
        model_name, pretrained = self._parse_model_spec(checkpoint)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device)
        self.model.eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _parse_model_spec(checkpoint: str | None) -> tuple[str, str]:
        if checkpoint and "::" in checkpoint:
            model_name, pretrained = checkpoint.split("::", maxsplit=1)
            return model_name, pretrained
        if checkpoint:
            # If only one part provided assume it is the pretrained tag.
            return _OpenClipBackend.DEFAULT_MODEL, checkpoint
        return _OpenClipBackend.DEFAULT_MODEL, _OpenClipBackend.DEFAULT_CHECKPOINT

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self._tokenizer(list(prompts))
        tokens = tokens.to(self.device)
        text_features = self.model.encode_text(tokens)
        return _normalize(text_features)

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        processed = torch.stack([
            self._preprocess(image).to(self.device) for image in images
        ])
        image_features = self.model.encode_image(processed)
        return _normalize(image_features)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


class ZeroShotClassifier:
    """Zero-shot classifier supporting SigLIP and OpenCLIP backends."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.labels = list(config.classes.keys())
        self.device = resolve_device(config.zero_shot.device)
        self.primary = self._create_backend(config.zero_shot)
        self.alternate = None
        if config.zero_shot.alternate_model:
            alt_cfg = ZeroShotConfig(
                model=config.zero_shot.alternate_model,
                checkpoint=config.zero_shot.alternate_checkpoint,
                device=config.zero_shot.device,
                batch_size=config.zero_shot.batch_size,
                score_threshold=config.zero_shot.score_threshold,
                margin_threshold=config.zero_shot.margin_threshold,
                multilingual=config.zero_shot.multilingual,
                alternate_model=None,
                alternate_checkpoint=None,
                alternate_weight=0.0,
            )
            self.alternate = self._create_backend(alt_cfg)
        self.prompt_texts, self.prompt_class_indices = self._flatten_prompts(
            config.classes
        )
        self.primary_text_features = self.primary.encode_text(self.prompt_texts)
        self.alternate_text_features = (
            self.alternate.encode_text(self.prompt_texts) if self.alternate else None
        )

    def _create_backend(self, zero_shot_cfg: ZeroShotConfig):  # type: ignore[no-untyped-def]
        model_key = zero_shot_cfg.model.lower()
        if model_key == "siglip":
            return _SiglipBackend(zero_shot_cfg.checkpoint, self.device)
        if model_key == "openclip":
            return _OpenClipBackend(zero_shot_cfg.checkpoint, self.device)
        raise ValueError(f"Unsupported zero-shot model: {zero_shot_cfg.model}")

    def classify(self, paths: Sequence[Path]) -> List[ZeroShotResult]:
        """Run zero-shot classification for *paths*."""

        predictions: List[ZeroShotResult] = []
        batch_size = self.config.zero_shot.batch_size
        prompt_counts = self._prompt_counts_tensor()
        for batch in tqdm(
            iter_batches(list(paths), batch_size),
            total=(len(paths) + batch_size - 1) // batch_size,
            desc="Zero-shot",
        ):
            images = [self._load_image(path) for path in batch]
            primary_scores = self._forward(self.primary, self.primary_text_features, images, prompt_counts)
            if self.alternate and self.alternate_text_features is not None:
                alt_scores = self._forward(
                    self.alternate, self.alternate_text_features, images, prompt_counts
                )
                weight = float(self.config.zero_shot.alternate_weight)
                primary_scores = (1 - weight) * primary_scores + weight * alt_scores
            predictions.extend(self._scores_to_results(batch, primary_scores))
        return predictions

    def _forward(
        self,
        backend,  # type: ignore[no-untyped-def]
        text_features: torch.Tensor,
        images: Sequence[Image.Image],
        prompt_counts: torch.Tensor,
    ) -> np.ndarray:
        image_features = backend.encode_images(images)
        logits = image_features @ text_features.T
        prompt_probs = torch.softmax(logits, dim=1)
        class_probs = torch.zeros(
            (prompt_probs.size(0), len(self.labels)), device=prompt_probs.device
        )
        for prompt_idx, class_idx in enumerate(self.prompt_class_indices):
            class_probs[:, class_idx] += prompt_probs[:, prompt_idx]
        class_probs /= prompt_counts
        return class_probs.detach().cpu().numpy()

    def _scores_to_results(
        self, paths: Sequence[Path], scores: np.ndarray
    ) -> List[ZeroShotResult]:
        results: List[ZeroShotResult] = []
        threshold = self.config.zero_shot.score_threshold
        margin_threshold = self.config.zero_shot.margin_threshold
        for index, path in enumerate(paths):
            row = scores[index]
            order = np.argsort(-row)
            top_idx = int(order[0])
            runner_up_idx = int(order[1]) if len(order) > 1 else top_idx
            top_label = self.labels[top_idx]
            runner_up_label = self.labels[runner_up_idx] if runner_up_idx != top_idx else None
            top_score = float(row[top_idx])
            runner_up_score = float(row[runner_up_idx]) if runner_up_label else 0.0
            margin = top_score - runner_up_score
            needs_review = not (top_score >= threshold and margin >= margin_threshold)
            score_dict = {label: float(row[i]) for i, label in enumerate(self.labels)}
            results.append(
                ZeroShotResult(
                    path=path,
                    label=top_label,
                    score=top_score,
                    runner_up_label=runner_up_label,
                    runner_up_score=runner_up_score,
                    margin=float(margin),
                    needs_review=needs_review,
                    scores=score_dict,
                )
            )
        return results

    def _flatten_prompts(self, classes: Dict[str, List[str]]):
        prompts: List[str] = []
        prompt_class_indices: List[int] = []
        for class_index, label in enumerate(self.labels):
            class_prompts = classes[label]
            if not class_prompts:
                raise ValueError(f"Class '{label}' must include at least one prompt")
            for prompt in class_prompts:
                prompts.append(prompt)
                prompt_class_indices.append(class_index)
        return prompts, prompt_class_indices

    def _prompt_counts_tensor(self) -> torch.Tensor:
        counts = torch.zeros(len(self.labels), dtype=torch.float32, device=self.device)
        for _, class_idx in enumerate(self.prompt_class_indices):
            counts[class_idx] += 1.0
        return counts.view(1, -1)

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")


def classify_zero_shot(config: Config, image_paths: Sequence[Path]) -> List[ZeroShotResult]:
    """Convenience function wrapping ZeroShotClassifier."""

    classifier = ZeroShotClassifier(config)
    return classifier.classify(image_paths)
