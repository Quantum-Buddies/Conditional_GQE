"""Unit tests for QD-GRPO correctness (cache isolation, descriptor, reward signs).

Tests the specific bugs identified in the audit:
1. DedupCache must include molecule context in key (no cross-molecule leakage)
2. Entanglement descriptor must not count single-qubit X as entangling
3. MMD diversity reward sign must follow Chemeleon2 (negative MMD, LOO = mmd_loo - mmd_full)
4. Creativity reward must be min_dist (not 1 - min_dist)
5. PerMoleculeArchives must isolate archives per molecule
6. FMO2 select_elite_for_fragment must return lowest energy circuit
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gqe.rl.map_elites import (
    DedupCache, MAPElitesArchive, PerMoleculeArchives,
    compute_circuit_features, _circuit_cache_key,
)


class TestDedupCacheIsolation:
    """Bug 1: Cache key must include molecule context."""

    def test_same_operators_different_molecules_get_different_cache(self):
        cache_h2 = DedupCache(molecule_id="h2", n_qubits=4, n_electrons=2)
        cache_lih = DedupCache(molecule_id="lih", n_qubits=12, n_electrons=4)

        ops = ["XYYX", "YXXY"]
        cache_h2.put(ops, -1.0)
        cache_lih.put(ops, -2.5)

        assert cache_h2.get(ops) == -1.0
        assert cache_lih.get(ops) == -2.5
        assert cache_h2.get(ops) != cache_lih.get(ops)

    def test_cache_key_changes_with_molecule(self):
        ops = ["XYYX", "YXXY"]
        key_h2 = _circuit_cache_key(ops, "h2", 4, 2, 5, 0.01)
        key_lih = _circuit_cache_key(ops, "lih", 12, 4, 5, 0.01)
        assert key_h2 != key_lih

    def test_cache_key_changes_with_qubit_count(self):
        ops = ["XYYX"]
        key_4q = _circuit_cache_key(ops, "h2", 4, 2, 5, 0.01)
        key_6q = _circuit_cache_key(ops, "h2", 6, 3, 5, 0.01)
        assert key_4q != key_6q

    def test_cache_key_changes_with_optimizer_iters(self):
        ops = ["XYYX"]
        key_5iter = _circuit_cache_key(ops, "h2", 4, 2, 5, 0.01)
        key_50iter = _circuit_cache_key(ops, "h2", 4, 2, 50, 0.01)
        assert key_5iter != key_50iter

    def test_get_or_compute_does_not_leak_across_molecules(self):
        """The exact smoke test from the audit: cache ["X"] at -1, then
        evaluate with a function that returns -2. Must NOT return cached -1."""
        cache = DedupCache(molecule_id="h2", n_qubits=4, n_electrons=2)
        cache.put(["X"], -1.0)

        cache_other = DedupCache(molecule_id="lih", n_qubits=12, n_electrons=4)
        result, was_cached = cache_other.get_or_compute(["X"], lambda ops: -2.0)
        assert not was_cached, "Cache leaked across molecules!"
        assert result == -2.0, f"Expected -2.0, got {result} (cached value leaked)"


class TestEntanglementDescriptor:
    """Bug 5: Single-qubit X is NOT entangling."""

    def test_single_qubit_x_not_entangling(self):
        features = compute_circuit_features(["XIII"], n_qubits=4, max_seq_len=64)
        assert features["entanglement_density"] == 0.0, \
            "Single-qubit X should not count as entangling"

    def test_two_qubit_xy_is_entangling(self):
        features = compute_circuit_features(["XXII"], n_qubits=4, max_seq_len=64)
        assert features["entanglement_density"] == 1.0, \
            "Two-qubit XX should count as entangling"

    def test_mixed_batch(self):
        features = compute_circuit_features(
            ["XIII", "XXII", "YIII", "YYII"], n_qubits=4, max_seq_len=64
        )
        assert features["entanglement_density"] == 0.5, \
            "2 of 4 operators are multi-qubit X/Y → density should be 0.5"

    def test_z_only_not_entangling(self):
        features = compute_circuit_features(["ZZII", "IZZI"], n_qubits=4, max_seq_len=64)
        assert features["entanglement_density"] == 0.0

    def test_xiii_and_xxii_get_different_features(self):
        """Audit found XIII and XXII got identical features. They must differ now."""
        f1 = compute_circuit_features(["XIII"], n_qubits=4, max_seq_len=64)
        f2 = compute_circuit_features(["XXII"], n_qubits=4, max_seq_len=64)
        assert f1["entanglement_density"] != f2["entanglement_density"], \
            "XIII and XXII must have different entanglement densities"


class TestPerMoleculeArchives:
    """Bug 2: Archives must be per-molecule, not global."""

    def test_separate_archives_per_molecule(self):
        archives = PerMoleculeArchives(n_bins_entanglement=5, n_bins_depth=5)
        archives.insert("h2", ["XXII"], -1.1, 4)
        archives.insert("lih", ["XXIIIIIIIIII"], -7.5, 12)

        h2_archive = archives.get("h2")
        lih_archive = archives.get("lih")

        assert len(h2_archive) == 1
        assert len(lih_archive) == 1
        assert h2_archive.best_energy() == -1.1
        assert lih_archive.best_energy() == -7.5

    def test_different_molecules_do_not_share_cells(self):
        archives = PerMoleculeArchives(n_bins_entanglement=5, n_bins_depth=5)
        # Insert same operators for two molecules — they should go to separate archives
        archives.insert("h2", ["XXII"], -1.1, 4)
        archives.insert("lih", ["XXII"], -7.5, 12)

        # Insert a worse circuit for h2 — should NOT affect lih
        archives.insert("h2", ["YYII"], -0.9, 4)

        h2_best = archives.get("h2").best_energy()
        lih_best = archives.get("lih").best_energy()
        assert h2_best == -1.1  # original elite retained
        assert lih_best == -7.5  # unaffected by h2 insertion

    def test_total_coverage_and_elites(self):
        archives = PerMoleculeArchives(n_bins_entanglement=5, n_bins_depth=5)
        # Use operators with different entanglement densities and depths to fill different cells
        archives.insert("h2", ["XXII"], -1.1, 4)           # high entanglement, short
        archives.insert("h2", ["ZZII", "ZIZI", "ZIIZ"], -0.8, 4)  # no entanglement, longer
        archives.insert("lih", ["XXIIIIIIIIII"], -7.5, 12)  # different molecule

        assert archives.total_elites() == 3
        assert archives.total_coverage() > 0.0


class TestMMDDiversitySign:
    """Bug 3: MMD diversity reward sign was reversed from Chemeleon2."""

    def test_diverse_batch_gets_higher_reward_than_redundant(self):
        from src.gqe.models.train_rl_dapo import compute_batch_diversity_mmd
        vocab = {"XXII": 0, "YYII": 1, "ZZII": 2, "XIXI": 3}

        # Diverse batch: all different operators
        diverse = [["XXII"], ["YYII"], ["ZZII"], ["XIXI"]]
        # Redundant batch: all same
        redundant = [["XXII"], ["XXII"], ["XXII"], ["XXII"]]

        ref = [["ZZII"], ["XIXI"]]

        diverse_rewards = compute_batch_diversity_mmd(diverse, vocab, ref)
        redundant_rewards = compute_batch_diversity_mmd(redundant, vocab, ref)

        # After min-max normalization, both have mean 0.5 by construction.
        # But the diverse batch should have higher max reward (some circuits
        # are clearly more diverse than others) while the redundant batch
        # has zero variance (all identical → all same reward).
        assert diverse_rewards.max() > redundant_rewards.max(), \
            "Diverse batch should have higher peak MMD diversity reward than redundant one"
        assert diverse_rewards.std() > redundant_rewards.std(), \
            "Diverse batch should have higher reward variance than redundant one"


class TestCreativityRewardSign:
    """Bug 4: Creativity reward was inverted (1 - min_dist instead of min_dist)."""

    def test_novel_circuit_gets_higher_reward_than_duplicate(self):
        from src.gqe.models.train_rl_dapo import compute_creativity_batch

        # Batch with a novel circuit and a duplicate
        novel = [["XXII", "YYII"], ["ZZII", "XIXI"]]
        duplicate = [["XXII", "YYII"], ["XXII", "YYII"]]

        seen = {("XXII", "YYII")}  # first circuit is "seen"

        novel_rewards = compute_creativity_batch(novel, seen)
        dup_rewards = compute_creativity_batch(duplicate, seen)

        # The novel circuit (index 1) should get higher reward than the duplicate (index 1)
        assert novel_rewards[1] > dup_rewards[1], \
            "Novel circuit should get higher creativity reward than duplicate"

    def test_unique_and_novel_gets_max_reward(self):
        from src.gqe.models.train_rl_dapo import compute_creativity_batch

        ops = [["XXII"], ["YYII"]]
        rewards = compute_creativity_batch(ops, seen_operators=None)
        # Both are unique in batch and novel (no seen set)
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0

    def test_duplicate_and_seen_gets_zero(self):
        from src.gqe.models.train_rl_dapo import compute_creativity_batch

        ops = [["XXII"], ["XXII"]]
        seen = {("XXII",)}
        rewards = compute_creativity_batch(ops, seen)
        # Both are duplicates in batch AND seen before
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0


class TestFMO2Selection:
    """Bug 8: FMO2 circuit library selection was not implemented."""

    def test_select_elite_returns_lowest_energy(self):
        archive = MAPElitesArchive(n_bins_entanglement=5, n_bins_depth=5, max_seq_len=64)
        archive.insert(["XXII"], -1.1, 4)
        archive.insert(["YYII"], -0.9, 4)
        archive.insert(["XXYY"], -1.3, 4)

        elite = archive.select_elite_for_fragment(target_n_qubits=4)
        assert elite is not None
        assert elite["energy"] == -1.3  # lowest energy

    def test_select_with_max_operators_filter(self):
        archive = MAPElitesArchive(n_bins_entanglement=5, n_bins_depth=5, max_seq_len=64)
        archive.insert(["XXII", "YYII", "ZZII"], -1.5, 4)
        archive.insert(["XXII"], -1.0, 4)

        elite = archive.select_elite_for_fragment(target_n_qubits=4, max_operators=2)
        assert elite is not None
        assert len(elite["operators"]) <= 2
        assert elite["energy"] == -1.0  # only the 1-operator circuit qualifies

    def test_select_from_empty_archive(self):
        archive = MAPElitesArchive(n_bins_entanglement=5, n_bins_depth=5, max_seq_len=64)
        elite = archive.select_elite_for_fragment(target_n_qubits=4)
        assert elite is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
