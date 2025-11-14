"""Quick test script for saliency extraction functionality."""

import numpy as np

from chess_sandbox.concept_extraction.model.features import compute_square_saliency


def test_compute_square_saliency_3d():
    """Test saliency computation with 3D activations (single position)."""
    print("Testing compute_square_saliency with 3D input...")

    # Simulate activations from a conv layer: (channels, height, width)
    activations = np.random.rand(64, 8, 8)

    # Test mean aggregation
    saliency_mean = compute_square_saliency(activations, aggregation="mean")
    assert saliency_mean.shape == (8, 8), f"Expected (8, 8), got {saliency_mean.shape}"
    assert saliency_mean.min() >= 0, "Saliency should be non-negative"
    print(
        f"  ✓ Mean aggregation: shape={saliency_mean.shape}, "
        f"range=[{saliency_mean.min():.4f}, {saliency_mean.max():.4f}]"
    )

    # Test max aggregation
    saliency_max = compute_square_saliency(activations, aggregation="max")
    assert saliency_max.shape == (8, 8)
    assert saliency_max.min() >= 0
    print(
        f"  ✓ Max aggregation: shape={saliency_max.shape}, "
        f"range=[{saliency_max.min():.4f}, {saliency_max.max():.4f}]"
    )

    # Test L2 aggregation
    saliency_l2 = compute_square_saliency(activations, aggregation="l2")
    assert saliency_l2.shape == (8, 8)
    assert saliency_l2.min() >= 0
    print(f"  ✓ L2 aggregation: shape={saliency_l2.shape}, range=[{saliency_l2.min():.4f}, {saliency_l2.max():.4f}]")

    # Verify that max >= mean (for positive values)
    assert np.all(saliency_max >= saliency_mean - 1e-6), "Max should be >= mean for abs values"
    print("  ✓ Max >= Mean verified")


def test_compute_square_saliency_4d():
    """Test saliency computation with 4D activations (batch)."""
    print("\nTesting compute_square_saliency with 4D input (batch)...")

    # Simulate batch of activations: (batch, channels, height, width)
    batch_size = 5
    activations = np.random.rand(batch_size, 64, 8, 8)

    saliency = compute_square_saliency(activations, aggregation="mean")
    assert saliency.shape == (batch_size, 8, 8), f"Expected ({batch_size}, 8, 8), got {saliency.shape}"
    assert saliency.min() >= 0
    print(f"  ✓ Batch processing: shape={saliency.shape}, range=[{saliency.min():.4f}, {saliency.max():.4f}]")


def test_square_saliency_response():
    """Test the ConceptSaliencyResponse model."""
    print("\nTesting ConceptSaliencyResponse...")

    from chess_sandbox.concept_extraction.labelling.labeller import Concept
    from chess_sandbox.concept_extraction.model.inference import ConceptSaliencyResponse

    # Create mock data
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    concepts = [
        Concept(name="opening", validated_by="probe", temporal="actual"),
        Concept(name="material_equality", validated_by="probe", temporal="actual"),
    ]

    # Create a saliency map with higher values in the center
    saliency_map = np.random.rand(8, 8) * 0.5
    saliency_map[3:5, 3:5] += 0.5  # Make center squares more salient

    # Create response
    response = ConceptSaliencyResponse.from_concepts_and_saliency(
        fen=fen,
        concepts=concepts,
        threshold=0.5,
        saliency_map=saliency_map,
        top_k=5,
        include_full_map=False,
    )

    assert response.fen == fen
    assert len(response.concepts) == 2
    assert len(response.top_squares) == 5
    assert response.saliency_map is None  # Not included
    print(f"  ✓ Response created with {len(response.concepts)} concepts")
    print("  ✓ Top 5 salient squares:")
    for i, sq in enumerate(response.top_squares, 1):
        print(f"     {i}. {sq.square}: {sq.saliency:.4f}")

    # Test with full map
    response_full = ConceptSaliencyResponse.from_concepts_and_saliency(
        fen=fen,
        concepts=concepts,
        threshold=0.5,
        saliency_map=saliency_map,
        top_k=10,
        include_full_map=True,
    )
    assert response_full.saliency_map is not None
    assert len(response_full.saliency_map) == 8
    assert len(response_full.saliency_map[0]) == 8
    print(f"  ✓ Full saliency map included: {len(response_full.saliency_map)}x{len(response_full.saliency_map[0])}")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting error handling...")

    # Test wrong spatial dimensions
    try:
        activations = np.random.rand(64, 7, 7)  # Wrong dimensions
        compute_square_saliency(activations)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught expected error for 7x7: {e}")

    # Test wrong number of dimensions
    try:
        activations = np.random.rand(64, 8)  # 2D instead of 3D/4D
        compute_square_saliency(activations)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught expected error for 2D: {e}")

    # Test invalid aggregation method
    try:
        activations = np.random.rand(64, 8, 8)
        compute_square_saliency(activations, aggregation="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught expected error for invalid aggregation: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("SALIENCY EXTRACTION TESTS")
    print("=" * 70)

    test_compute_square_saliency_3d()
    test_compute_square_saliency_4d()
    test_square_saliency_response()
    test_error_handling()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
