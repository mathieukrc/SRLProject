
import sys
import types

import numpy as np
import torch


# Temporary stub for the DSRL dependency.
# Only used to import the dataset utilities for isolated tests.
dsrl_stub = types.ModuleType("dsrl")
sys.modules["dsrl"] = dsrl_stub

from data.synthetic import (
    random_chain_start_picker,
    density_matched_chain_start_picker,
    noise_augmentation_chain_start_picker,
    uncertainty_guided_chain_start_picker,
    violation_guided_chain_start_picker,
    combined_chain_start_picker,
)


class DummyDataset:
    def __init__(self, states):
        self.data = {
            "observations": torch.as_tensor(
                states, dtype=torch.float32
            )
        }


def dummy_uncertainty(states):
    return 1.0 - torch.abs(states[:, 0])


def dummy_violation(states):
    return states[:, 1] > 0.0


def main():
    rng = np.random.default_rng(42)

    states = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(1000, 4)
    )

    dataset = DummyDataset(states)
    budget = 500

    methods = {
        "random": random_chain_start_picker(
            dataset, budget
        ),
        "density_matched": density_matched_chain_start_picker(
            dataset, budget
        ),
        "noise_augmentation": noise_augmentation_chain_start_picker(
            dataset, budget
        ),
        "uncertainty_guided": uncertainty_guided_chain_start_picker(
            dataset, budget, dummy_uncertainty
        ),
        "violation_guided": violation_guided_chain_start_picker(
            dataset, budget, dummy_violation
        ),
        "combined": combined_chain_start_picker(
            dataset,
            budget,
            dummy_uncertainty,
            dummy_violation
        ),
    }

    for name, generated in methods.items():
        assert generated.shape == (budget, 4)
        assert torch.isfinite(generated).all()

        print(
            f"{name}: shape={tuple(generated.shape)} - PASSED"
        )

    print("\nAll isolated tests passed!")

    print("\n--- Targeting behavior tests ---")

    random_states = methods["random"]
    uncertainty_states = methods["uncertainty_guided"]
    violation_states = methods["violation_guided"]
    combined_states = methods["combined"]

    # 1. Uncertainty-guided sampling
    random_uncertainty = dummy_uncertainty(random_states).mean()
    guided_uncertainty = dummy_uncertainty(uncertainty_states).mean()

    print("\nUncertainty-guided:")
    print("Random mean uncertainty:", random_uncertainty.item())
    print("Guided mean uncertainty:", guided_uncertainty.item())

    assert guided_uncertainty > random_uncertainty
    print("PASSED")

    # 2. Violation-guided sampling
    random_violation_rate = dummy_violation(random_states).float().mean()
    guided_violation_rate = dummy_violation(violation_states).float().mean()

    print("\nViolation-guided:")
    print("Random violation rate:", random_violation_rate.item())
    print("Guided violation rate:", guided_violation_rate.item())

    assert guided_violation_rate >= random_violation_rate
    print("PASSED")

    # 3. Combined sampling
    combined_violation_rate = dummy_violation(combined_states).float().mean()

    print("\nCombined:")
    print("Combined violation rate:", combined_violation_rate.item())

    assert combined_violation_rate >= random_violation_rate

    # Compare uncertainty only among violating states
    combined_mask = dummy_violation(combined_states)
    violation_mask = dummy_violation(violation_states)

    if combined_mask.any() and violation_mask.any():
        combined_uncertainty = dummy_uncertainty(
            combined_states[combined_mask]
        ).mean()

        violation_uncertainty = dummy_uncertainty(
            violation_states[violation_mask]
        ).mean()

        print("Combined uncertainty:", combined_uncertainty.item())
        print("Violation-only uncertainty:", violation_uncertainty.item())

        assert combined_uncertainty >= violation_uncertainty

    print("PASSED")

    print("\nAll targeting behavior tests passed!")

    print("\n--- Reproducibility tests ---")

    functions = {
        "random": lambda: random_chain_start_picker(
            dataset, budget, seed=42
        ),
        "noise": lambda: noise_augmentation_chain_start_picker(
            dataset, budget, seed=42
        ),
        "uncertainty": lambda: uncertainty_guided_chain_start_picker(
            dataset, budget, dummy_uncertainty, seed=42
        ),
        "violation": lambda: violation_guided_chain_start_picker(
            dataset, budget, dummy_violation, seed=42
        ),
        "combined": lambda: combined_chain_start_picker(
            dataset,
            budget,
            dummy_uncertainty,
            dummy_violation,
            seed=42,
        ),
    }

    for name, function in functions.items():
        first = function()
        second = function()

        assert torch.equal(first, second), (
            f"{name} is not reproducible"
        )

        print(f"{name}: reproducible - PASSED")

if __name__ == "__main__":
    main()
