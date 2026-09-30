"""Matched group-aware support selection and reproducible specialist training."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .baselines import schema_key, state_text, tfidf_matrix, tokens
from .datasets import audit_splits, fingerprint, load_cases, request_fingerprint
from .experiments import write_cases


def prepare_support(train_path, heldout_paths, output, *, shots=(8, 32, 128), seeds=(0, 1, 2)):
    if len(set(seeds)) < 3 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("at least three distinct nonnegative seeds are required")
    if not shots or any(type(k) is not int or k < 1 for k in shots):
        raise ValueError("shots must be positive integers")
    cases = load_cases(train_path)
    if any(c.split != "train" or len(c.request.questions) != 1 or c.request.media for c in cases):
        raise ValueError("specialist support requires text-only single-question training cases")
    if not heldout_paths or not audit_splits([train_path, *heldout_paths])["valid"]:
        raise ValueError("disjoint held-out data is required")
    schemas = {schema_key(c.request.questions[0]) for c in cases}
    if len(schemas) != 1:
        raise ValueError("one fixed schema per support curve")
    by_label = defaultdict(list)
    for case in cases:
        q = case.request.questions[0]
        by_label[case.gold[q.id]].append(case)
    labels = cases[0].request.questions[0].labels()
    if set(by_label) != set(labels):
        raise ValueError("training data must cover every class")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    entries, selections = [], []
    for seed in seeds:
        rng = random.Random(seed)
        # Select one representative per source group globally before stratifying.
        groups = defaultdict(list)
        for case in cases:
            groups[case.group_id or case.id].append(case)
        pools = defaultdict(list)
        for key in sorted(groups):
            case = rng.choice(groups[key])
            pools[case.gold[case.request.questions[0].id]].append(case)
        for label in labels:
            rng.shuffle(pools[label])
        for k in shots:
            if any(len(pools[label]) < k for label in labels):
                raise ValueError(f"insufficient distinct source groups for {k} shots/class")
            selected = [case for label in labels for case in pools[label][:k]]
            name = f"shots-{k}-seed-{seed}"
            selections.append((name, selected))
            entries.append(
                {
                    "id": name,
                    "seed": seed,
                    "shots_per_class": k,
                    "support_ids": [c.id for c in selected],
                    "dataset": f"{name}.jsonl",
                    "dataset_sha256": fingerprint(selected),
                }
            )
    output.mkdir(parents=True)
    for name, selected in selections:
        write_cases(output / f"{name}.jsonl", selected)
    manifest = {
        "schema_version": 2,
        "schema": next(iter(schemas)),
        "train_sha256": fingerprint(cases),
        "heldout_sha256": [fingerprint(load_cases(p)) for p in heldout_paths],
        "cells": entries,
        "selection": "one representative per source group, nested supports within seed",
    }
    write_json(output / "support.json", manifest)
    return manifest


def fit_baseline(
    dataset, output, *, method="tfidf", seed=0, model_id=None, revision=None, steps=20
):
    cases = load_cases(dataset)
    if any(c.split != "train" or c.request.media or len(c.request.questions) != 1 for c in cases):
        raise ValueError("training requires single-question text-only train split")
    if method not in ("tfidf", "prior", "setfit"):
        raise ValueError("unknown training method")
    if steps < 1 or seed < 0:
        raise ValueError("positive steps and nonnegative seed required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    groups = defaultdict(list)
    for case in cases:
        groups[schema_key(case.request.questions[0])].append(case)
    if method == "setfit" and (len(groups) != 1 or not model_id or not revision):
        raise ValueError("SetFit requires one schema and a pinned embedding checkpoint")
    if method == "setfit" and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("SetFit requires an immutable embedding revision")
    schemas = {}
    for key, selected in groups.items():
        labels = sorted(selected[0].request.questions[0].labels())
        y = [labels.index(c.gold[c.request.questions[0].id]) for c in selected]
        if set(y) != set(range(len(labels))):
            raise ValueError("every class needs training examples")
        text = [state_text(c.request.state) for c in selected]
        schema = {
            "labels": labels,
            "prior": (np.bincount(y, minlength=len(labels)) / len(y)).tolist(),
        }
        if method == "tfidf":
            from sklearn.linear_model import LogisticRegression

            vocabulary = {
                term: i for i, term in enumerate(sorted({t for s in text for t in tokens(s)}))
            }
            if not vocabulary:
                raise ValueError("empty training vocabulary")
            df = np.zeros(len(vocabulary))
            for s in text:
                for term in set(tokens(s)):
                    df[vocabulary[term]] += 1
            idf = np.log((1 + len(text)) / (1 + df)) + 1
            x = tfidf_matrix(text, vocabulary, idf)
            model = LogisticRegression(random_state=seed, max_iter=1000).fit(x, y)
            schema.update(
                vocabulary=vocabulary,
                idf=idf.tolist(),
                coefficients=model.coef_.tolist(),
                intercept=model.intercept_.tolist(),
            )
        elif method == "setfit":
            from datasets import Dataset
            from setfit import SetFitModel, Trainer, TrainingArguments
            from transformers import set_seed

            set_seed(seed)
            model = SetFitModel.from_pretrained(model_id, revision=revision)
            args = TrainingArguments(
                output_dir=str(output / "training"),
                seed=seed,
                max_steps=steps,
                num_iterations=20,
                num_epochs=1,
                report_to="none",
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=Dataset.from_dict({"text": text, "label": y})
            )
            trainer.train()
            model.save_pretrained(str(output / "model"))
        schemas[key] = schema
    artifact = {
        "schema_version": 2,
        "method": method,
        "seed": seed,
        "schemas": schemas,
        "train_sha256": fingerprint(cases),
        "train_ids": [c.id for c in cases],
        "train_groups": sorted({c.group_id or c.id for c in cases}),
        "train_requests": [request_fingerprint(c.request) for c in cases],
        "tokenization": "ascii_words_and_non_ascii_characters",
        "model_id": model_id,
        "revision": revision,
        "steps": steps if method == "setfit" else None,
    }
    output.mkdir(parents=True, exist_ok=method == "setfit")
    if method == "setfit":
        from .checkpoints import directory_digest

        artifact["checkpoint_sha256"] = directory_digest(output / "model")
    write_json(output / "classifier.json", artifact)
    return {
        "artifact_file": str(output / "classifier.json"),
        "artifact_sha256": hashlib.sha256((output / "classifier.json").read_bytes()).hexdigest(),
        "support_ids": artifact["train_ids"],
        "method": method,
    }
