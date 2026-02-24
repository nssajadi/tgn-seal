import os


def build_base_name(model, dataset, variant=None, extra_suffix=None):
    """Create base filename without run index."""
    parts = [model]

    if variant:
        parts.append(variant)

    parts.append(dataset)

    if extra_suffix:
        parts.append(extra_suffix)

    return "-".join(parts)


def build_model_result_paths(
        model,
        dataset,
        num_runs,
        *,
        prefix="results",
        variant=None,
        extra_suffix=None,
):
    """
    Generate file paths for multiple runs of one model.

    Run 0 → no suffix
    Run i>0 → _i
    """
    base = build_base_name(model, dataset, variant, extra_suffix)

    paths = [
        os.path.join(prefix, f"{base}.pkl" if i == 0 else f"{base}_{i}.pkl")
        for i in range(num_runs)
    ]

    return paths


def build_models_result_paths(dataset, num_runs, config, prefix="results"):
    """
    Build dict: model label -> list of result file paths
    """
    return {
        label: build_model_result_paths(
            model,
            dataset,
            num_runs,
            prefix=prefix,
            variant=variant,
            extra_suffix=extra,
        )
        for label, (model, variant, extra) in config.items()
    }
