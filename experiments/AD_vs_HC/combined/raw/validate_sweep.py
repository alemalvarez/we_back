from typing import List, Tuple, Any
from itertools import product
import yaml  # type: ignore

from core.validate_kernel import validate_kernel


def _extract_values(param: Any) -> List[str]:
    """Return list of string-encoded options from a sweep parameter node.

    Supports either a single 'value' or a list under 'values'.
    """
    if isinstance(param, dict):
        if "values" in param and isinstance(param["values"], list):
            return [str(v) for v in param["values"]]
        if "value" in param:
            return [str(param["value"])]
    # Fallback: treat as single string
    return [str(param)]


def _parse_tuple_list(encoded: str) -> List[Tuple[int, int]]:
    """Parse underscore-encoded 2D tuple list like '50_2__20_8__20_2'."""
    groups = [g for g in encoded.strip().split("__") if g]
    tuples: List[Tuple[int, int]] = []
    for g in groups:
        parts = [p for p in g.split("_") if p]
        if len(parts) != 2:
            raise ValueError(f"Expected 2 parts per group, got {len(parts)} in '{g}'")
        tuples.append((int(parts[0]), int(parts[1])))
    return tuples


def _encode_repeated(value: Tuple[int, int], count: int) -> str:
    return "__".join(f"{value[0]}_{value[1]}" for _ in range(count))


def validate_sweep(yaml_path: str, input_shape: Tuple[int, int] = (1000, 68)) -> None:
    """
    Read a sweep yaml, parse underscore-encoded conv params, validate all
    kernel/stride/padding combinations using validate_kernel, and print a summary.
    """
    with open(yaml_path, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    params = sweep_cfg.get("parameters", {})

    ks_options = _extract_values(params.get("kernel_sizes"))
    if not ks_options:
        print("No kernel_sizes found in sweep yaml.")
        return

    stride_node = params.get("strides")
    pad_node = params.get("paddings")

    results: List[tuple[str, str, str, bool]] = []

    combo_index = 1
    for ks_str in ks_options:
        kernels = _parse_tuple_list(ks_str)
        num_layers = len(kernels)

        # Prepare stride/padding option strings; default if missing
        stride_options = _extract_values(stride_node) if stride_node is not None else [
            _encode_repeated((1, 1), num_layers)
        ]
        padding_options = _extract_values(pad_node) if pad_node is not None else [
            _encode_repeated((0, 0), num_layers)
        ]

        for st_str, pd_str in product(stride_options, padding_options):
            try:
                strides = _parse_tuple_list(st_str)
                paddings = _parse_tuple_list(pd_str)
            except Exception:
                is_valid = False
            else:
                # Length mismatch or invalid shapes handled in validate_kernel
                is_valid = validate_kernel(kernels, strides, paddings, input_shape=input_shape)

            print(
                f"Combo {combo_index}: kernels='{ks_str}', strides='{st_str}', paddings='{pd_str}' -> valid={is_valid}"
            )
            results.append((ks_str, st_str, pd_str, is_valid))
            combo_index += 1

    # Summary
    total = len(results)
    valid_count = sum(1 for _, _, _, ok in results if ok)
    print(f"Validated {total} combinations. Valid: {valid_count}, Invalid: {total - valid_count}.")


if __name__ == "__main__":
    validate_sweep("experiments/AD_vs_HC/combined/raw/sweep_improved2d.yaml")