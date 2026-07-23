#!/usr/bin/env python3
"""Controlled PNG representation benchmark.

Research question: can changing only the PNG representation improve the
reliability of an unchanged visual provider against an unchanged compiled
ZeroModel policy? See
`docs/research/controlled-png-representation-benchmark.md`.

This is an experiment harness, not a new production package: it reuses the
Stage 2D provider-evaluation aggregate
(`zeromodel.video.domains.video_action_set.provider_evaluation_dto`) as the
system of record, and reuses `examples/local_model_zero_arcade_test.py`
verbatim for the fixture state universe, renderer, response parser,
`ScriptedProvider`/`OllamaProvider`, and policy compiler. The only new
production-adjacent surface is `examples/arcade_png_interventions.py`
(deterministic PNG intervention recipes) and
`examples/arcade_png_representation_runner.py` (execution engine); this
module is CLI, orchestration, and evidence-package output only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    MaterializedProviderEvaluationRunDTO,
)

import examples.local_model_zero_arcade_test as arcade
from examples.arcade_png_interventions import (
    ALL_VARIANTS,
    COMBINED_VARIANT,
    COOLDOWN_VARIANTS,
    LANE_VARIANTS,
    REFERENCE_VARIANTS,
    UNLABELLED_VARIANT,
    ArcadePngInterventionRecipe,
    apply_recipe,
    build_recipe,
)
from examples.arcade_png_representation_comparison import (
    COOLDOWN_TARGET_METRICS,
    GENERIC_TARGET_METRICS,
    LANE_TARGET_METRICS,
    build_comparison_rows,
    build_compatibility_statement,
    classify_variant,
    validate_comparable_runs,
    write_comparison_csv,
    write_comparison_json,
    write_comparison_md,
)
from examples.arcade_png_representation_runner import (
    build_provider_configuration,
    build_scripted_replies_for_variant,
    find_resumable_run,
    run_variant,
)

DEFAULT_VARIANTS = REFERENCE_VARIANTS + COOLDOWN_VARIANTS + LANE_VARIANTS


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "ollama"), default="fake")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--fixture", choices=("smoke", "canonical"), default="smoke")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--store", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compile-reports", action="store_true")
    parser.add_argument("--combined-cooldown", choices=COOLDOWN_VARIANTS)
    parser.add_argument("--combined-lane", choices=LANE_VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--write-pngs", dest="write_pngs", action="store_true", default=True
    )
    parser.add_argument("--no-write-pngs", dest="write_pngs", action="store_false")
    args = parser.parse_args(argv)
    args.variants = [
        entry.strip() for entry in args.variants.split(",") if entry.strip()
    ]
    _validate_arguments(args)
    return args


def _validate_arguments(args: argparse.Namespace) -> None:
    for variant_id in args.variants:
        if variant_id not in ALL_VARIANTS:
            raise SystemExit(f"unknown --variants entry: {variant_id!r}")
    if COMBINED_VARIANT in args.variants:
        if args.combined_cooldown is None or args.combined_lane is None:
            raise SystemExit(
                "--combined-cooldown and --combined-lane are required when "
                "combined-v1 is requested"
            )
        if not (args.resume or any(v in COOLDOWN_VARIANTS for v in args.variants)):
            raise SystemExit(
                "combined-v1 must not be selected before its isolated cooldown/lane "
                "family variants have been evaluated - include them in --variants "
                "(or use --resume against an experiment that already ran them)"
            )
    if args.store == "sqlite" and args.sqlite_path is None:
        raise SystemExit("--sqlite-path is required when --store sqlite")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be in [0,1]")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    if args.resume and args.output_dir is None:
        raise SystemExit(
            "--resume requires an explicit --output-dir naming the prior experiment"
        )
    if (
        args.output_dir is not None
        and args.output_dir.exists()
        and not args.overwrite_output
        and not args.resume
    ):
        raise SystemExit(
            "--output-dir already exists; pass --overwrite-output or --resume"
        )


def _build_shared_provider(
    args: argparse.Namespace,
    *,
    states: Sequence[arcade.ArcadeState],
    recipes: Mapping[str, ArcadePngInterventionRecipe],
) -> arcade.Provider:
    if args.backend == "ollama":
        return arcade.OllamaProvider(args.base_url, args.model, args.timeout, args.seed)
    combined_replies: dict[str, arcade.ProviderReply] = {}
    for recipe in recipes.values():
        steps_by_state = [
            apply_recipe(recipe, arcade.render(state, recipe.base_render_mode))
            for state in states
        ]
        combined_replies.update(
            build_scripted_replies_for_variant(states, steps_by_state)
        )
    return arcade.ScriptedProvider(combined_replies)


def _write_variant_images(
    *,
    variant_dir: Path,
    recipe: ArcadePngInterventionRecipe,
    states: Sequence[arcade.ArcadeState],
) -> None:
    images_dir = variant_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(states):
        image = arcade.render(state, recipe.base_render_mode)
        final_image = apply_recipe(recipe, image)[-1][0]
        path = images_dir / f"{index:03d}-{arcade.safe_name(state.row_id)}.png"
        final_image.save(path, format="PNG", optimize=False)


def _summary_presentation(
    saved_run: MaterializedProviderEvaluationRunDTO,
    *,
    recipe: ArcadePngInterventionRecipe,
) -> dict[str, object]:
    summary = saved_run.summary
    factor_correct = summary.factor_correct_counts.to_value()
    factor_denominators = summary.factor_denominators.to_value()
    factor_accuracy = {
        key: (factor_correct[key] / factor_denominators[key])
        for key in factor_denominators
        if factor_denominators[key]
    }
    return {
        "variant_id": recipe.variant_id,
        "recipe_id": recipe.recipe_id,
        "base_render_mode": recipe.base_render_mode,
        "run_id": saved_run.run.run_id,
        "policy_artifact_id": saved_run.run.policy_artifact_id,
        "attempted": summary.attempted_count,
        "accepted": summary.accepted_count,
        "rejected": summary.rejected_count,
        "exact_state_correct": summary.exact_count,
        "exact_state_accuracy_over_attempted": (
            summary.exact_count / summary.attempted_count
            if summary.attempted_count
            else None
        ),
        "action_correct": summary.action_correct_count,
        "action_accuracy_over_attempted": (
            summary.action_correct_count / summary.attempted_count
            if summary.attempted_count
            else None
        ),
        "action_equivalent_count": summary.action_equivalent_count,
        "action_changing_count": summary.action_changing_count,
        "factor_accuracy_over_accepted": factor_accuracy,
        "latency_us": {
            "median": summary.latency_median_us,
            "p95": summary.latency_p95_us,
            "min": summary.latency_min_us,
            "max": summary.latency_max_us,
        },
        "rejection_reasons": dict(summary.rejection_reason_counts.to_value()),
    }


def _write_variant_output(
    *,
    variant_dir: Path,
    recipe: ArcadePngInterventionRecipe,
    saved_run: MaterializedProviderEvaluationRunDTO,
    records: list[dict[str, object]],
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "run.json").write_text(
        json.dumps(saved_run.run.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "recipe.json").write_text(
        json.dumps(recipe.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (variant_dir / "summary.json").write_text(
        json.dumps(
            _summary_presentation(saved_run, recipe=recipe), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    with (variant_dir / "cases.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _compile_variant_report(
    saved_run: MaterializedProviderEvaluationRunDTO, *, output: Path
) -> None:
    from zeromodel.artifacts import (
        InMemoryArtifactStore,
        load_compiled_report_aggregate,
    )
    from zeromodel.core import LayoutRecipe

    from examples.provider_evaluation_report_adapter import (
        compile_provider_evaluation_report,
    )

    layout_recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "arcade-png-representation-benchmark-priority-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    artifact_store = InMemoryArtifactStore()
    compiled = compile_provider_evaluation_report(
        saved_run, layout_recipe=layout_recipe, store=artifact_store
    )
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=artifact_store
    )
    (output / "report.json").write_text(
        json.dumps(
            {
                "compiled_report_artifact_id": compiled.artifact_ref.artifact_id,
                "adapted_report_id": aggregate.adapted_report.adapted_report_id,
                "vpm_artifact_id": aggregate.vpm_artifact.artifact_id,
                "subject_count": len(aggregate.adapted_report.subjects),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _target_metrics_for_variant(variant_id: str) -> Sequence[str]:
    if variant_id in COOLDOWN_VARIANTS:
        return COOLDOWN_TARGET_METRICS
    if variant_id in LANE_VARIANTS:
        return LANE_TARGET_METRICS
    return GENERIC_TARGET_METRICS


def _write_comparison(
    output_dir: Path,
    materialized_runs: Mapping[str, MaterializedProviderEvaluationRunDTO],
    recipe_ids: Mapping[str, str],
) -> None:
    validate_comparable_runs(tuple(materialized_runs.values()))
    rows = build_comparison_rows(materialized_runs, recipe_ids)
    baseline_id = (
        UNLABELLED_VARIANT
        if UNLABELLED_VARIANT in materialized_runs
        else next(iter(sorted(materialized_runs)))
    )
    baseline_run = materialized_runs[baseline_id]
    classifications = {}
    for variant_id, run in materialized_runs.items():
        if variant_id == baseline_id:
            continue
        classifications[variant_id] = classify_variant(
            baseline=baseline_run,
            candidate=run,
            target_metrics=_target_metrics_for_variant(variant_id),
        )
    statement = (
        build_compatibility_statement(baseline_run.run)
        + f" Baseline variant: {baseline_id}."
    )
    write_comparison_json(
        output_dir / "comparison.json",
        rows=rows,
        classifications=classifications,
        compatibility_statement=statement,
    )
    write_comparison_csv(output_dir / "comparison.csv", rows)
    write_comparison_md(
        output_dir / "comparison.md",
        rows=rows,
        classifications=classifications,
        compatibility_statement=statement,
    )


def _write_experiment_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    stamp: str,
    policy_artifact_id: str,
    provider_configuration_id: str,
    fixture_identity: str,
    case_mode: str,
    resumed_variants: set[str],
) -> None:
    payload = {
        "schema_version": arcade.SCHEMA_VERSION,
        "python": sys.version,
        "created": stamp,
        "fixture_identity": fixture_identity,
        "case_mode": case_mode,
        "policy_artifact_id": policy_artifact_id,
        "provider_configuration_id": provider_configuration_id,
        "variants": args.variants,
        "resumed_variants": sorted(resumed_variants),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve_output_dir(args: argparse.Namespace, *, stamp: str) -> Path:
    if args.output_dir is not None:
        if args.output_dir.exists() and args.overwrite_output and not args.resume:
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return args.output_dir
    output_dir = (
        Path("./local-results") / f"arcade-png-benchmark-{args.fixture}-{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run(args: argparse.Namespace) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _resolve_output_dir(args, stamp=stamp)
    print(f"[INFO] Results will be written to {output_dir}")

    reader, artifact_hex = arcade.policy_reader()
    policy_artifact_id = f"sha256:{artifact_hex}"
    states = arcade.smoke_states() if args.fixture == "smoke" else arcade.all_states()
    if args.max_cases:
        states = states[: args.max_cases]

    case_mode = "arcade-smoke-v1" if args.fixture == "smoke" else "arcade-canonical-v1"
    fixture_identity = f"arcade-png-representation-benchmark:{args.fixture}"

    provider_configuration = build_provider_configuration(
        backend=args.backend, model=args.model, seed=args.seed
    )

    recipes = {
        variant_id: build_recipe(
            variant_id,
            combined_cooldown=args.combined_cooldown,
            combined_lane=args.combined_lane,
        )
        for variant_id in args.variants
    }

    runtime = arcade._build_runtime(args)
    facade = runtime.video_action_set
    # Each variant gets its own benchmark identity and episode plan (and
    # therefore its own frame_id namespace): the identity/episode-plan
    # aggregate requires a unique (seed_digest, split, ordinal) triple per
    # plan, and `ObservationDTO.frame_id` requires a purely decimal sequence
    # suffix, so distinct representation variants of the same fixture state
    # cannot share one episode_id. This identity/plan diversity is bookkeeping
    # only - it is not part of the provider-evaluation comparability
    # fingerprint (`FIXED_IDENTITY_FIELDS` in
    # `arcade_png_representation_comparison.py`), which is what actually
    # governs whether two runs may be compared.
    identities_by_variant = {
        variant_id: arcade._build_benchmark_identity(
            model=args.model,
            artifact_id=policy_artifact_id,
            stamp=f"{stamp}-{variant_id}",
        )
        for variant_id in args.variants
    }
    plans_by_variant = {
        variant_id: arcade._build_episode_plan(
            identity=identities_by_variant[variant_id],
            episode_id=f"development:arcade-png-representation-benchmark-{stamp}-{variant_id}",
            frame_count=len(states),
        )
        for variant_id in args.variants
    }
    for variant_id in args.variants:
        facade.save_identity(identities_by_variant[variant_id])
        facade.save_episode_plan(plans_by_variant[variant_id])

    provider = _build_shared_provider(args, states=states, recipes=recipes)

    materialized_runs: dict[str, MaterializedProviderEvaluationRunDTO] = {}
    recipe_ids: dict[str, str] = {}
    resumed_variants: set[str] = set()

    for variant_id in args.variants:
        recipe = recipes[variant_id]
        plan = plans_by_variant[variant_id]
        identity = identities_by_variant[variant_id]
        recipe_ids[variant_id] = recipe.recipe_id
        variant_dir = output_dir / variant_id

        existing = None
        if args.resume:
            existing = find_resumable_run(
                facade=facade,
                fixture_identity=fixture_identity,
                provider_configuration_id=provider_configuration.provider_configuration_id,
                policy_artifact_id=policy_artifact_id,
                case_mode=case_mode,
                variant_id=variant_id,
            )
        if existing is not None:
            print(
                f"[INFO] --resume reusing existing run for {variant_id}: {existing.run.run_id}"
            )
            materialized_runs[variant_id] = existing
            resumed_variants.add(variant_id)
            continue

        print(f"[INFO] Running variant {variant_id} ({len(states)} case(s))")
        saved_run, records = run_variant(
            variant_id=variant_id,
            recipe=recipe,
            states=states,
            provider=provider,
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
            facade=facade,
            identity=identity,
            plan=plan,
            fixture_identity=fixture_identity,
            case_mode=case_mode,
            confidence_threshold=args.confidence_threshold,
            metadata={
                "model": args.model,
                "backend": args.backend,
                "recipe_id": recipe.recipe_id,
            },
        )
        materialized_runs[variant_id] = saved_run
        _write_variant_output(
            variant_dir=variant_dir, recipe=recipe, saved_run=saved_run, records=records
        )
        if args.write_pngs:
            _write_variant_images(variant_dir=variant_dir, recipe=recipe, states=states)
        if args.compile_reports:
            _compile_variant_report(saved_run, output=variant_dir)

    _write_comparison(output_dir, materialized_runs, recipe_ids)
    _write_experiment_manifest(
        output_dir,
        args=args,
        stamp=stamp,
        policy_artifact_id=policy_artifact_id,
        provider_configuration_id=provider_configuration.provider_configuration_id,
        fixture_identity=fixture_identity,
        case_mode=case_mode,
        resumed_variants=resumed_variants,
    )
    print(f"[INFO] Experiment complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
