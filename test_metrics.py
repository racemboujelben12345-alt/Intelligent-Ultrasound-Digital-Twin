from src.image_analysis.basic_metrics import analyze_image


IMAGE_PATH = (
    "data/raw/public_ultrasound/"
    "us_simulation/abdominal_US/AUS/"
    "images/train/ct14-1.png"
)


def main():
    print("=" * 70)
    print("SCAN A DIGITAL TWIN V2")
    print("ADVANCED IMAGE CHARACTERIZATION TEST")
    print("=" * 70)

    metrics = analyze_image(IMAGE_PATH)

    # ==============================================================
    # IMAGE INFORMATION
    # ==============================================================

    print("\n--- IMAGE INFORMATION ---")

    print(f"Path              : {metrics['image_path']}")
    print(f"Width             : {metrics['width']}")
    print(f"Height            : {metrics['height']}")

    # ==============================================================
    # INTENSITY
    # ==============================================================

    print("\n--- INTENSITY ---")

    print(f"Minimum           : {metrics['min_intensity']:.3f}")
    print(f"Maximum           : {metrics['max_intensity']:.3f}")
    print(f"Mean              : {metrics['mean_intensity']:.3f}")
    print(f"Median            : {metrics['median_intensity']:.3f}")
    print(f"Standard deviation: {metrics['std_intensity']:.3f}")
    print(f"Dynamic range     : {metrics['dynamic_range']:.3f}")

    # ==============================================================
    # DISTRIBUTION
    # ==============================================================

    print("\n--- INTENSITY DISTRIBUTION ---")

    print(f"P01               : {metrics['percentile_1']:.3f}")
    print(f"P05               : {metrics['percentile_5']:.3f}")
    print(f"P25               : {metrics['percentile_25']:.3f}")
    print(f"P50               : {metrics['percentile_50']:.3f}")
    print(f"P75               : {metrics['percentile_75']:.3f}")
    print(f"P95               : {metrics['percentile_95']:.3f}")
    print(f"P99               : {metrics['percentile_99']:.3f}")

    print(
        f"Coefficient variation: "
        f"{metrics['coefficient_variation']:.6f}"
    )

    # ==============================================================
    # IMAGE QUALITY
    # ==============================================================

    print("\n--- IMAGE QUALITY ---")

    print(
        f"Contrast           : "
        f"{metrics['contrast_std']:.3f}"
    )

    print(
        f"Laplacian sharpness: "
        f"{metrics['sharpness_laplacian']:.3f}"
    )

    # ==============================================================
    # TEXTURE
    # ==============================================================

    print("\n--- TEXTURE ---")

    print(
        f"Entropy            : "
        f"{metrics['entropy']:.6f}"
    )

    print(
        f"Speckle proxy      : "
        f"{metrics['speckle_proxy']:.6f}"
    )

    # ==============================================================
    # STRUCTURE
    # ==============================================================

    print("\n--- STRUCTURE ---")

    print(
        f"Edge density       : "
        f"{metrics['edge_density']:.6f}"
    )

    print(
        f"Gradient mean      : "
        f"{metrics['gradient_mean']:.6f}"
    )

    print(
        f"Gradient std       : "
        f"{metrics['gradient_std']:.6f}"
    )

    print(
        f"Gradient maximum   : "
        f"{metrics['gradient_max']:.6f}"
    )

    # ==============================================================
    # UNIFORMITY
    # ==============================================================

    print("\n--- UNIFORMITY ---")

    print(
        f"Uniformity         : "
        f"{metrics['uniformity']:.6f}"
    )

    # ==============================================================
    # END
    # ==============================================================

    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS TERMINATED")
    print("=" * 70)


if __name__ == "__main__":
    main()