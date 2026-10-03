from __future__ import annotations

import numpy as np

from src.drift.control_charts import (
    DriftMonitor,
    DriftMonitorConfig,
)


def print_observations(
    title: str,
    observations,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'idx':>4} | "
        f"{'signal':>10} | "
        f"{'z':>8} | "
        f"{'EWMA':>10} | "
        f"{'CUSUM+':>10} | "
        f"{'CUSUM-':>10} | "
        f"{'status'}"
    )

    print("-" * 100)

    for obs in observations:

        print(
            f"{obs.index:4d} | "
            f"{obs.signal:10.4f} | "
            f"{obs.z_score:8.4f} | "
            f"{obs.ewma:10.4f} | "
            f"{obs.cusum_positive:10.4f} | "
            f"{obs.cusum_negative:10.4f} | "
            f"{obs.status}"
        )


def main():

    config = DriftMonitorConfig(
        ewma_lambda=0.20,
        ewma_limit=3.0,
        cusum_k=0.5,
        cusum_h=5.0,
        persistence=3,
    )

    # ================================================================
    # 1. Stable sequence
    # ================================================================

    stable_monitor = DriftMonitor(
        reference_mean=10.0,
        reference_std=2.0,
        config=config,
    )

    stable_signals = [
        9.8,
        10.2,
        10.1,
        9.9,
        10.0,
        10.3,
        9.7,
        10.1,
    ]

    stable_observations = stable_monitor.update_many(
        stable_signals
    )

    print_observations(
        "STABLE TEMPORAL SEQUENCE",
        stable_observations,
    )

    if any(
        obs.status != "STABLE"
        for obs in stable_observations
    ):
        raise AssertionError(
            "La séquence stable a généré une alarme de drift."
        )

    # ================================================================
    # 2. Persistent positive drift
    # ================================================================

    drift_monitor = DriftMonitor(
        reference_mean=10.0,
        reference_std=2.0,
        config=config,
    )

    drift_signals = [
        10.0,
        10.2,
        10.1,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
    ]

    drift_observations = drift_monitor.update_many(
        drift_signals
    )

    print_observations(
        "PERSISTENT POSITIVE DRIFT",
        drift_observations,
    )

    persistent = [
        obs
        for obs in drift_observations
        if obs.status == "PERSISTENT_DRIFT"
    ]

    if not persistent:
        raise AssertionError(
            "La dérive persistante n'a pas été détectée."
        )

    if persistent[0].consecutive_alarms < config.persistence:
        raise AssertionError(
            "PERSISTENT_DRIFT détecté avant le nombre "
            "d'alarmes requis."
        )

    # ================================================================
    # 3. State evolution checks
    # ================================================================

    final_observation = drift_observations[-1]

    if final_observation.status != "PERSISTENT_DRIFT":
        raise AssertionError(
            "La dernière observation devrait être "
            "PERSISTENT_DRIFT."
        )

    if final_observation.ewma <= 0.0:
        raise AssertionError(
            "L'EWMA devrait indiquer une dérive positive."
        )

    if final_observation.cusum_positive <= 0.0:
        raise AssertionError(
            "Le CUSUM positif devrait être activé."
        )

    # ================================================================
    # 4. Reproducibility
    # ================================================================

    monitor_a = DriftMonitor(
        reference_mean=10.0,
        reference_std=2.0,
        config=config,
    )

    monitor_b = DriftMonitor(
        reference_mean=10.0,
        reference_std=2.0,
        config=config,
    )

    result_a = monitor_a.update_many(
        drift_signals
    )

    result_b = monitor_b.update_many(
        drift_signals
    )

    values_a = [
        (
            obs.z_score,
            obs.ewma,
            obs.cusum_positive,
            obs.cusum_negative,
            obs.status,
        )
        for obs in result_a
    ]

    values_b = [
        (
            obs.z_score,
            obs.ewma,
            obs.cusum_positive,
            obs.cusum_negative,
            obs.status,
        )
        for obs in result_b
    ]

    if values_a != values_b:
        raise AssertionError(
            "Le DriftMonitor n'est pas reproductible."
        )

    print()
    print("=" * 100)
    print("TEMPORAL DRIFT MONITOR V2 OK")
    print("=" * 100)


if __name__ == "__main__":
    main()