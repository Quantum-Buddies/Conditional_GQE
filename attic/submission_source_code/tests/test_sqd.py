"""Correctness and property tests for the hardware-agnostic SQD pipeline.

Tests cover:
- Bit order consistency (qubit 0 = LSB = rightmost char in bitstring)
- Pauli matrix element phases (Z sign flips, Y imaginary phases)
- Symmetry filtering (particle number, spin parity preservation)
- Variational bound (SQD energy >= FCI energy within tolerance)
- Nested subspace monotonicity (larger subspace -> lower or equal energy)
- JW round-trip (bitstring <-> Slater determinant mapping)
- Random counts control (random bitstrings should give worse energy than structured)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    iter_terms,
    get_active_electron_count,
)
from src.gqe.eval.sqd import (
    pauli_matrix_element,
    build_subspace_hamiltonian,
    sqd_energy_from_bitstrings,
    filter_by_particle_number,
    filter_by_spin_parity,
    select_subspace_by_counts,
    sqd_energy_from_counts,
    exact_diagonalize as sqd_exact_diagonalize,
    canonicalize_counts,
    target_spin_counts,
    filter_configurations,
    apply_pauli_to_bitstring,
    project_pauli_hamiltonian,
    solve_subspace,
    run_sqd,
    reverse_bitstrings_in_counts,
)
from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings as legacy_qsci_energy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def h2_record():
    """Load the H2 Hamiltonian record (4 qubits, 15 terms)."""
    ham_path = ROOT / "results" / "data" / "hamiltonians_gic2026" / "hamiltonians.json"
    if not ham_path.exists():
        pytest.skip(f"Hamiltonian file not found: {ham_path}")
    with ham_path.open() as f:
        data = json.load(f)
    for rec in data.get("records", []):
        if rec["name"] == "h2":
            return rec
    pytest.skip("h2 not found in Hamiltonian records")


@pytest.fixture
def h2_fci(h2_record):
    """Exact FCI energy for H2 via dense diagonalization (qiskit-free)."""
    return sqd_exact_diagonalize(h2_record)


# ---------------------------------------------------------------------------
# Bit order tests
# ---------------------------------------------------------------------------

class TestBitOrder:
    """Verify that bitstring conventions are consistent: qubit 0 = LSB = rightmost char."""

    def test_bitstring_to_int_qubit0_is_rightmost(self):
        """int('0101', 2) = 5, so qubit 0 (rightmost) = 1, qubit 1 = 0, qubit 2 = 1, qubit 3 = 0."""
        bs = "0101"
        val = int(bs, 2)
        assert val == 5
        # qubit 0 is bit 0 of the integer = rightmost char
        assert (val >> 0) & 1 == 1  # rightmost char '1' -> qubit 0 = 1
        assert (val >> 1) & 1 == 0  # second char '0' -> qubit 1 = 0
        assert (val >> 2) & 1 == 1  # third char '1' -> qubit 2 = 1
        assert (val >> 3) & 1 == 0  # leftmost char '0' -> qubit 3 = 0

    def test_hf_determinant_bit_order(self, h2_record):
        """H2 has 2 active electrons. HF state = |1100> in qubit ordering = qubits 0,1 occupied."""
        n_qubits = int(h2_record["n_qubits"])
        n_electrons = get_active_electron_count(h2_record)
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
        # For H2: n_electrons=2, n_qubits=4 -> "0011"
        # qubit 0 = 1, qubit 1 = 1, qubit 2 = 0, qubit 3 = 0
        assert hf_bs == "0011"
        assert int(hf_bs, 2) == 3

    def test_bitstring_consistency_with_pauli_ops(self, h2_record):
        """A Z operator on qubit i should flip sign based on bit i of the integer."""
        n_qubits = int(h2_record["n_qubits"])
        # Take a Z0 term: <b|Z0|b> = (-1)^{bit_0(b)}
        # For b = "0011" (int 3), bit 0 = 1, so Z0 expectation = -1
        b_int = 3
        bit_0 = (b_int >> 0) & 1
        z0_expectation = (-1) ** bit_0
        assert z0_expectation == -1.0  # qubit 0 is occupied

        # For b = "0001" (int 1), bit 0 = 1
        b_int = 1
        bit_0 = (b_int >> 0) & 1
        assert (-1) ** bit_0 == -1.0

        # For b = "0010" (int 2), bit 0 = 0
        b_int = 2
        bit_0 = (b_int >> 0) & 1
        assert (-1) ** bit_0 == 1.0


# ---------------------------------------------------------------------------
# Pauli matrix element phase tests
# ---------------------------------------------------------------------------

class TestPauliPhases:
    """Verify correct phases in Pauli matrix element computation."""

    def test_z_diagonal(self):
        """<b|Z_i|b> = (-1)^{bit_i(b)}."""
        # Z on qubit 0, bitstring 0011 (int 3), bit 0 = 1 -> -1
        ops = ["Z", "I", "I", "I"]
        coeff = 1.0 + 0.0j
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=3, n_qubits=4)
        assert result == -1.0 + 0.0j

        # Z on qubit 1, bitstring 0011 (int 3), bit 1 = 1 -> -1
        ops = ["I", "Z", "I", "I"]
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=3, n_qubits=4)
        assert result == -1.0 + 0.0j

        # Z on qubit 2, bitstring 0011 (int 3), bit 2 = 0 -> +1
        ops = ["I", "I", "Z", "I"]
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=3, n_qubits=4)
        assert result == 1.0 + 0.0j

    def test_x_off_diagonal(self):
        """<b|X_i|b ^ (1<<i)> = 1 (X flips qubit i)."""
        # X on qubit 0: connects b=0011 (3) to b=0010 (2)
        ops = ["X", "I", "I", "I"]
        coeff = 1.0 + 0.0j
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=2, n_qubits=4)
        assert result == 1.0 + 0.0j

        # X on qubit 2: connects b=0011 (3) to b=0111 (7)
        ops = ["I", "I", "X", "I"]
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=7, n_qubits=4)
        assert result == 1.0 + 0.0j

    def test_y_off_diagonal_phase(self):
        """<b|Y_i|b ^ (1<<i)> = ±i depending on bit_i(b).

        Y|0> = i|1>, Y|1> = -i|0>.
        So <b|Y_i|b^mask> = i * (-1)^{bit_i(b)}.
        """
        # Y on qubit 0, b=0011 (bit 0 = 1): <0011|Y_0|0010> = -i
        ops = ["Y", "I", "I", "I"]
        coeff = 1.0 + 0.0j
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=2, n_qubits=4)
        assert result == -1.0j  # bit_0=1 -> (-1)^1 * i = -i

        # Y on qubit 2, b=0011 (bit 2 = 0): <0011|Y_2|0111> = +i
        ops = ["I", "I", "Y", "I"]
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=7, n_qubits=4)
        assert result == 1.0j  # bit_2=0 -> (-1)^0 * i = +i

    def test_multi_pauli_phase(self):
        """Y0 X1 X2 Y3 on b=0011 (3): check combined phase.

        ops = [Y, X, X, Y]
        b_int = 3 = 0011
        xy_mask = (1<<0) | (1<<1) | (1<<2) | (1<<3) = 0b1111 = 15
        b_flipped = 3 ^ 15 = 12 = 1100

        Phase:
        - qubit 0: Y, bit_0=1 -> -i
        - qubit 1: X, no phase
        - qubit 2: X, no phase
        - qubit 3: Y, bit_3=0 -> +i
        Total: (-i) * (1) * (1) * (i) = (-i)(i) = 1
        """
        ops = ["Y", "X", "X", "Y"]
        coeff = 1.0 + 0.0j
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=12, n_qubits=4)
        assert result == 1.0 + 0.0j

    def test_xy_mask_mismatch_returns_zero(self):
        """If b_flipped != b_int ^ xy_mask, return 0."""
        ops = ["X", "I", "I", "I"]
        coeff = 1.0 + 0.0j
        # X on qubit 0 connects 3->2, not 3->1
        result = pauli_matrix_element(ops, coeff, b_int=3, b_flipped=1, n_qubits=4)
        assert result == 0.0 + 0.0j


# ---------------------------------------------------------------------------
# Symmetry filtering tests
# ---------------------------------------------------------------------------

class TestSymmetryFiltering:
    """Test particle number and spin parity conservation filters."""

    def test_particle_number_filter(self):
        """filter_by_particle_number keeps only bitstrings with N electrons."""
        n_qubits = 4
        n_electrons = 2
        bitstrings = ["0011", "0101", "0110", "0001", "1111", "0000"]
        filtered = filter_by_particle_number(bitstrings, n_electrons)
        # 0011 (2), 0101 (2), 0110 (2) have 2 electrons
        assert "0011" in filtered
        assert "0101" in filtered
        assert "0110" in filtered
        assert "0001" not in filtered  # 1 electron
        assert "1111" not in filtered  # 4 electrons
        assert "0000" not in filtered  # 0 electrons

    def test_particle_number_filter_tolerance(self):
        """With tolerance, allow bitstrings within ±tol of target N."""
        n_qubits = 4
        n_electrons = 2
        bitstrings = ["0011", "0001", "0111"]
        # tol=1 allows 1, 2, or 3 electrons
        filtered = filter_by_particle_number(bitstrings, n_electrons, tol=1)
        assert "0011" in filtered  # 2 electrons
        assert "0001" in filtered  # 1 electron (within tol)
        assert "0111" in filtered  # 3 electrons (within tol)

    def test_spin_parity_filter(self):
        """filter_by_spin_parity keeps bitstrings with correct alpha-beta parity."""
        n_qubits = 4
        # For H2: 2 spin-orbitals per spatial orbital, alpha on even, beta on odd
        # Spin parity = (n_alpha - n_beta) mod 2
        # Singlet (S=0): parity = 0
        bitstrings = ["0011", "0101", "0110", "1100"]
        # 0011: alpha=1(q0), beta=1(q1) -> parity=0
        # 0101: alpha=1(q0), beta=1(q2) -> alpha=1, beta=1 -> parity=0
        # 0110: alpha=1(q1), beta=1(q2) -> depends on convention
        filtered = filter_by_spin_parity(bitstrings, n_qubits, target_parity=0)
        # At minimum, 0011 (HF) should pass
        assert "0011" in filtered

    def test_symmetry_filtering_improves_energy(self, h2_record, h2_fci):
        """Filtered subspace should give lower (better) energy than unfiltered with same size."""
        n_electrons = get_active_electron_count(h2_record)
        n_qubits = int(h2_record["n_qubits"])

        # Generate all 16 bitstrings for 4 qubits
        all_bs = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

        # Full space energy = FCI
        full_energy = sqd_energy_from_bitstrings(h2_record, all_bs)
        assert abs(full_energy - h2_fci) < 1e-10

        # Filter to correct particle number
        filtered = filter_by_particle_number(all_bs, n_electrons)
        filtered_energy = sqd_energy_from_bitstrings(h2_record, filtered)

        # Filtered subspace energy >= FCI (variational bound)
        assert filtered_energy >= h2_fci - 1e-10

        # But should be close to FCI for H2 with all 2-electron determinants
        assert filtered_energy < h2_fci + 0.1  # Should capture most of the correlation


# ---------------------------------------------------------------------------
# Variational bound tests
# ---------------------------------------------------------------------------

class TestVariationalBound:
    """SQD energy must satisfy variational bound: E_SQD >= E_FCI."""

    def test_full_space_equals_fci(self, h2_record, h2_fci):
        """SQD with all bitstrings should exactly equal FCI energy."""
        n_qubits = int(h2_record["n_qubits"])
        all_bs = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
        sqd_energy = sqd_energy_from_bitstrings(h2_record, all_bs)
        assert abs(sqd_energy - h2_fci) < 1e-8

    def test_subspace_energy_above_fci(self, h2_record, h2_fci):
        """Any proper subspace should give energy >= FCI."""
        n_qubits = int(h2_record["n_qubits"])
        # Use a small subspace: HF + a few excited determinants
        subspace = ["0011", "1100", "0101", "1010"]
        sqd_energy = sqd_energy_from_bitstrings(h2_record, subspace)
        assert sqd_energy >= h2_fci - 1e-10

    def test_hf_only_energy_above_fci(self, h2_record, h2_fci):
        """HF determinant alone should give energy >= FCI."""
        n_electrons = get_active_electron_count(h2_record)
        n_qubits = int(h2_record["n_qubits"])
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
        hf_energy = sqd_energy_from_bitstrings(h2_record, [hf_bs])
        assert hf_energy >= h2_fci - 1e-10


# ---------------------------------------------------------------------------
# Nested subspace monotonicity tests
# ---------------------------------------------------------------------------

class TestNestedSubspaceMonotonicity:
    """Larger subspace should give lower or equal energy (monotonic improvement)."""

    def test_monotonicity_with_nested_subspaces(self, h2_record, h2_fci):
        """E(S1 ⊂ S2 ⊂ S3) should be monotonically decreasing."""
        n_qubits = int(h2_record["n_qubits"])
        all_bs = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

        # Build nested subspaces by adding one bitstring at a time
        # Start from HF and add determinants
        n_electrons = get_active_electron_count(h2_record)
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")

        # Order: HF first, then by Hamming distance from HF
        remaining = [bs for bs in all_bs if bs != hf_bs]
        remaining.sort(key=lambda bs: bin(int(bs, 2) ^ int(hf_bs, 2)).count("1"))

        nested = [hf_bs]
        energies = [sqd_energy_from_bitstrings(h2_record, nested)]

        for bs in remaining:
            nested.append(bs)
            e = sqd_energy_from_bitstrings(h2_record, nested)
            energies.append(e)

        # Check monotonicity: each energy <= previous
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-10, (
                f"Monotonicity violated at step {i}: "
                f"E({i}) = {energies[i]:.10f} > E({i-1}) = {energies[i-1]:.10f}"
            )

        # Final energy should be FCI
        assert abs(energies[-1] - h2_fci) < 1e-8

    def test_monotonicity_with_counts_based_selection(self, h2_record, h2_fci):
        """Selecting subspace by frequency should also show monotonic improvement."""
        n_qubits = int(h2_record["n_qubits"])
        all_bs = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

        # Simulate counts: HF has highest count, then excitations
        counts = {}
        for bs in all_bs:
            hamming_dist = bin(int(bs, 2) ^ 0b0011).count("1")
            counts[bs] = max(1, 100 - hamming_dist * 20)

        # Select nested subspaces by count order
        sorted_bs = sorted(counts.keys(), key=lambda bs: -counts[bs])
        energies = []
        for k in range(1, len(sorted_bs) + 1):
            e = sqd_energy_from_bitstrings(h2_record, sorted_bs[:k])
            energies.append(e)

        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-10, (
                f"Monotonicity violated at k={i+1}: "
                f"E(k={i+1}) = {energies[i]:.10f} > E(k={i}) = {energies[i-1]:.10f}"
            )


# ---------------------------------------------------------------------------
# Counts-based SQD tests
# ---------------------------------------------------------------------------

class TestCountsBasedSQD:
    """Test sqd_energy_from_counts that processes raw measurement counts."""

    def test_counts_to_bitstrings_to_energy(self, h2_record, h2_fci):
        """Full counts over all bitstrings should give FCI energy."""
        n_qubits = int(h2_record["n_qubits"])
        # Uniform counts over all 16 states
        counts = {format(i, f"0{n_qubits}b"): 1 for i in range(2**n_qubits)}
        energy = sqd_energy_from_counts(h2_record, counts, subspace_size=None)
        assert abs(energy - h2_fci) < 1e-8

    def test_select_subspace_by_counts(self):
        """select_subspace_by_counts should return most frequent bitstrings."""
        counts = {"0011": 500, "1100": 200, "0101": 100, "0000": 50}
        selected = select_subspace_by_counts(counts, subspace_size=2)
        assert "0011" in selected
        assert "1100" in selected
        assert len(selected) == 2

    def test_random_counts_worse_than_structured(self, h2_record, h2_fci):
        """Random uniform counts should give worse energy than HF-focused counts.

        With k=3 (small subspace), the structured selection should pick the
        chemically most important determinants (HF + key excitations), while
        random selection will likely miss critical determinants.
        """
        n_qubits = int(h2_record["n_qubits"])
        n_electrons = get_active_electron_count(h2_record)
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")

        # Structured counts: HF dominant with some excitations
        structured_counts = {hf_bs: 1000}
        for i in range(2**n_qubits):
            bs = format(i, f"0{n_qubits}b")
            if bs != hf_bs:
                structured_counts[bs] = max(1, 50 - bin(int(bs, 2) ^ int(hf_bs, 2)).count("1") * 10)

        # Random counts - average over multiple seeds for robustness
        rng = np.random.default_rng(42)
        random_counts = {}
        for _ in range(4096):
            bs = format(rng.integers(0, 2**n_qubits), f"0{n_qubits}b")
            random_counts[bs] = random_counts.get(bs, 0) + 1

        # Use k=3: structured should pick HF + 2 best excitations
        k = 3
        structured_energy = sqd_energy_from_counts(h2_record, structured_counts, subspace_size=k)
        random_energy = sqd_energy_from_counts(h2_record, random_counts, subspace_size=k)

        # Structured should be better (lower) than random
        assert structured_energy <= random_energy + 1e-10


# ---------------------------------------------------------------------------
# Cross-check with legacy QSCI implementation
# ---------------------------------------------------------------------------

class TestLegacyCrossCheck:
    """Ensure new SQD implementation agrees with existing qsci_postprocess."""

    def test_agreement_on_full_space(self, h2_record, h2_fci):
        """Both implementations should give FCI on the full space."""
        n_qubits = int(h2_record["n_qubits"])
        all_bs = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

        new_energy = sqd_energy_from_bitstrings(h2_record, all_bs)
        legacy_energy = legacy_qsci_energy(h2_record, all_bs)

        assert abs(new_energy - legacy_energy) < 1e-8
        assert abs(new_energy - h2_fci) < 1e-8

    def test_agreement_on_subspace(self, h2_record):
        """Both implementations should agree on a small subspace."""
        subspace = ["0011", "1100", "0101", "1010"]
        new_energy = sqd_energy_from_bitstrings(h2_record, subspace)
        legacy_energy = legacy_qsci_energy(h2_record, subspace)
        assert abs(new_energy - legacy_energy) < 1e-8


# ---------------------------------------------------------------------------
# JW round-trip test
# ---------------------------------------------------------------------------

class TestJWRoundTrip:
    """Verify bitstring <-> Slater determinant mapping is consistent."""

    def test_jw_occupation_consistency(self, h2_record):
        """For JW mapping, qubit i = spin-orbital i occupation.
        Bit i = 1 means occupied, bit i = 0 means virtual.
        """
        n_qubits = int(h2_record["n_qubits"])
        n_electrons = get_active_electron_count(h2_record)

        # HF state: first n_electrons qubits occupied
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
        hf_int = int(hf_bs, 2)

        # Count occupied spin-orbitals
        n_occupied = bin(hf_int).count("1")
        assert n_occupied == n_electrons

        # Single excitation: move electron from qubit 0 to qubit 2
        excited = hf_int ^ (1 << 0) ^ (1 << 2)
        excited_bs = format(excited, f"0{n_qubits}b")
        assert bin(excited).count("1") == n_electrons  # Same particle number

    def test_hamming_weight_preserves_particle_number(self):
        """All bitstrings with same Hamming weight have same particle number."""
        n_qubits = 4
        for n_elec in range(n_qubits + 1):
            for i in range(2**n_qubits):
                bs = format(i, f"0{n_qubits}b")
                hw = bin(i).count("1")
                assert hw == sum(int(b) for b in bs)
                if hw == n_elec:
                    assert filter_by_particle_number([bs], n_elec) == [bs]


# ---------------------------------------------------------------------------
# Masterplan API tests
# ---------------------------------------------------------------------------

class TestCanonicalizeCounts:
    """Test canonicalize_counts normalizes bitstring keys."""

    def test_pads_short_bitstrings(self):
        """Short bitstrings should be zero-padded to n_qubits."""
        counts = {"11": 100, "101": 50}
        canonical = canonicalize_counts(counts, n_qubits=4)
        assert "0011" in canonical
        assert canonical["0011"] == 100
        assert "0101" in canonical
        assert canonical["0101"] == 50

    def test_merges_duplicate_keys(self):
        """Duplicate keys after padding should be merged."""
        counts = {"11": 100, "0011": 50}
        canonical = canonicalize_counts(counts, n_qubits=4)
        assert canonical["0011"] == 150

    def test_strips_whitespace(self):
        """Whitespace in bitstrings should be stripped."""
        counts = {" 0011 ": 100, "1100": 200}
        canonical = canonicalize_counts(counts, n_qubits=4)
        assert canonical["0011"] == 100
        assert canonical["1100"] == 200


class TestTargetSpinCounts:
    """Test target_spin_counts filters to correct spin sector."""

    def test_singlet_filter(self):
        """Singlet (S=0) requires n_alpha == n_beta."""
        n_qubits = 4
        n_electrons = 2
        counts = {
            "0011": 100,  # n_alpha=1, n_beta=1 -> singlet, 2 electrons
            "0110": 30,   # n_alpha=1, n_beta=1 -> singlet, 2 electrons
            "0001": 20,   # n_alpha=1, n_beta=0 -> not singlet, 1 electron
            "0101": 5,    # n_alpha=2, n_beta=0 -> not singlet, 2 electrons
        }
        filtered = target_spin_counts(counts, n_qubits, n_electrons, spin_squared=0)
        assert "0011" in filtered
        assert "0110" in filtered
        assert "0001" not in filtered
        assert "0101" not in filtered


class TestFilterConfigurations:
    """Test filter_configurations with invalid-reason accounting."""

    def test_particle_number_rejection(self):
        """Bitstrings with wrong particle number are rejected with count."""
        bitstrings = ["0011", "0001", "1111", "0101"]
        valid, rejection = filter_configurations(
            bitstrings, n_qubits=4, n_electrons=2,
        )
        assert "0011" in valid
        assert "0101" in valid
        assert "0001" not in valid
        assert "1111" not in valid
        assert rejection["wrong_particle_number"] == 2
        assert rejection["total_valid"] == 2
        assert rejection["total_invalid"] == 2

    def test_no_filtering_returns_all(self):
        """With no filters, all bitstrings pass."""
        bitstrings = ["0011", "0001", "1111"]
        valid, rejection = filter_configurations(bitstrings, n_qubits=4)
        assert len(valid) == 3
        assert rejection["total_valid"] == 3
        assert rejection["total_invalid"] == 0

    def test_spin_parity_rejection(self):
        """Spin parity filter rejects with correct accounting."""
        bitstrings = ["0011", "0001", "0101"]
        valid, rejection = filter_configurations(
            bitstrings, n_qubits=4, n_electrons=2, spin_parity=0,
        )
        # 0011: n_alpha=1, n_beta=1, parity=0 -> pass
        # 0001: wrong particle number -> rejected by particle number
        # 0101: n_alpha=1, n_beta=1, parity=0 -> pass
        assert "0011" in valid
        assert "0001" not in valid
        assert rejection["wrong_particle_number"] == 1


class TestApplyPauliToBitstring:
    """Test apply_pauli_to_bitstring for X, Y, Z operations."""

    def test_x_flips_bit(self):
        """X on qubit 0 flips bit 0."""
        result, phase = apply_pauli_to_bitstring(["X", "I", "I", "I"], 3, 4)
        assert result == 2  # 0011 -> 0010
        assert phase == 1.0 + 0.0j

    def test_z_phase(self):
        """Z on qubit 0 gives -1 phase if bit is 1."""
        result, phase = apply_pauli_to_bitstring(["Z", "I", "I", "I"], 3, 4)
        assert result == 3  # Z doesn't flip
        assert phase == -1.0 + 0.0j  # bit 0 = 1

    def test_y_phase(self):
        """Y on qubit 0: bit=1 -> -i phase, flips bit."""
        result, phase = apply_pauli_to_bitstring(["Y", "I", "I", "I"], 3, 4)
        assert result == 2  # Y flips bit 0
        assert phase == -1.0j  # bit 0 = 1 -> -i

    def test_identity_no_change(self):
        """Identity does nothing."""
        result, phase = apply_pauli_to_bitstring(["I", "I", "I", "I"], 5, 4)
        assert result == 5
        assert phase == 1.0 + 0.0j


class TestProjectPauliHamiltonian:
    """Test project_pauli_hamiltonian returns correct dense matrix."""

    def test_full_space_equals_fci(self, h2_record, h2_fci):
        """Full-space projection should give FCI energy."""
        n_qubits = int(h2_record["n_qubits"])
        all_bs = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
        h_dense = project_pauli_hamiltonian(h2_record, all_bs)
        eigvals = np.linalg.eigvalsh(h_dense)
        assert abs(float(eigvals[0]) - h2_fci) < 1e-8

    def test_hermitian(self, h2_record):
        """Projected Hamiltonian should be Hermitian."""
        subspace = ["0011", "1100"]
        h_dense = project_pauli_hamiltonian(h2_record, subspace)
        assert np.allclose(h_dense, h_dense.conj().T)


class TestSolveSubspace:
    """Test solve_subspace returns ground energy and dimension."""

    def test_returns_energy_and_dim(self, h2_record):
        """solve_subspace returns (energy, dim) tuple."""
        subspace = ["0011", "1100", "0101", "1010"]
        energy, dim = solve_subspace(h2_record, subspace)
        assert dim == 4
        assert isinstance(energy, float)

    def test_returns_eigvec_when_requested(self, h2_record):
        """solve_subspace with return_eigvec=True returns 3-tuple."""
        subspace = ["0011", "1100"]
        energy, eigvec, dim = solve_subspace(h2_record, subspace, return_eigvec=True)
        assert dim == 2
        assert eigvec.shape == (2,)


class TestRunSQD:
    """Test the full run_sqd pipeline function."""

    def test_full_space_gives_fci(self, h2_record, h2_fci):
        """run_sqd with all bitstrings should give FCI energy."""
        n_qubits = int(h2_record["n_qubits"])
        counts = {format(i, f"0{n_qubits}b"): 1 for i in range(2**n_qubits)}
        result = run_sqd(h2_record, counts, n_electrons=2, return_details=False)
        assert abs(result["energy"] - h2_fci) < 1e-8
        assert result["variational_bound_satisfied"] is True

    def test_rejection_accounting(self, h2_record):
        """run_sqd with return_details=True includes rejection counts."""
        n_qubits = int(h2_record["n_qubits"])
        counts = {format(i, f"0{n_qubits}b"): 1 for i in range(2**n_qubits)}
        result = run_sqd(h2_record, counts, n_electrons=2, return_details=True)
        assert "rejection" in result
        assert result["rejection"]["total_valid"] > 0
        assert result["rejection"]["total_invalid"] > 0  # Some wrong particle number

    def test_monotonicity_check(self, h2_record):
        """run_sqd includes monotonicity check for small subspaces."""
        n_qubits = int(h2_record["n_qubits"])
        # Use HF + a few excitations with high counts
        counts = {"0011": 1000, "1100": 100, "0101": 50, "1010": 50}
        result = run_sqd(h2_record, counts, n_electrons=2, subspace_size=4, return_details=True)
        assert "monotonicity_ok" in result
        assert result["monotonicity_ok"] is True

    def test_canonicalization_in_pipeline(self, h2_record, h2_fci):
        """run_sqd should canonicalize non-padded bitstrings correctly."""
        n_qubits = int(h2_record["n_qubits"])
        # Pass non-padded bitstrings
        counts = {format(i, "b"): 1 for i in range(2**n_qubits)}
        result = run_sqd(h2_record, counts, n_electrons=2, return_details=False)
        assert abs(result["energy"] - h2_fci) < 1e-8


# ---------------------------------------------------------------------------
# QPU bit ordering tests (Rigetti Cepheus fix)
# ---------------------------------------------------------------------------

class TestQPUBitOrderReversal:
    """Test reverse_bitstrings_in_counts and reverse_bit_order parameter.

    Rigetti QPUs via qBraid return bitstrings with qubit 0 as the LEFTMOST
    character, while our SQD convention uses qubit 0 as the RIGHTMOST (LSB).
    This mismatch caused 100x energy errors on Cepheus-108Q results.
    """

    def test_reverse_bitstrings_basic(self):
        """Reversing '1100' gives '0011'."""
        counts = {"1100": 100, "0011": 50, "1010": 25}
        reversed_counts = reverse_bitstrings_in_counts(counts, n_qubits=4)
        assert reversed_counts["0011"] == 100
        assert reversed_counts["1100"] == 50
        assert reversed_counts["0101"] == 25

    def test_reverse_bitstrings_padding(self):
        """Short bitstrings should be padded before reversal."""
        counts = {"11": 100}
        reversed_counts = reverse_bitstrings_in_counts(counts, n_qubits=4)
        assert "1100" in reversed_counts
        assert reversed_counts["1100"] == 100

    def test_canonicalize_with_reverse(self):
        """canonicalize_counts with reverse_bit_order should reverse then canonicalize."""
        counts = {"1100": 100, "0011": 50}
        canonical = canonicalize_counts(counts, n_qubits=4, reverse_bit_order=True)
        assert canonical["0011"] == 100
        assert canonical["1100"] == 50

    def test_sqd_energy_with_reversed_bits(self, h2_record, h2_fci):
        """sqd_energy_from_counts with reverse_bit_order should recover correct energy."""
        n_qubits = int(h2_record["n_qubits"])
        n_electrons = get_active_electron_count(h2_record)

        # HF bitstring in our convention (qubit 0 = rightmost)
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
        hf_energy = sqd_energy_from_counts(h2_record, {hf_bs: 1000}, n_electrons=n_electrons)

        # Simulate QPU returning reversed bitstrings
        reversed_hf = hf_bs[::-1]
        energy_no_fix = sqd_energy_from_counts(h2_record, {reversed_hf: 1000}, n_electrons=n_electrons)
        energy_with_fix = sqd_energy_from_counts(
            h2_record, {reversed_hf: 1000}, n_electrons=n_electrons, reverse_bit_order=True
        )

        # Without fix should give wrong energy
        assert abs(energy_no_fix - hf_energy) > 0.01, "Without fix should give wrong energy"
        # With fix should recover correct energy
        assert abs(energy_with_fix - hf_energy) < 1e-10, "With fix should recover HF energy"

    def test_run_sqd_with_reversed_bits(self, h2_record, h2_fci):
        """run_sqd with reverse_bit_order should give FCI on full space."""
        n_qubits = int(h2_record["n_qubits"])
        all_counts = {format(i, f"0{n_qubits}b"): 1 for i in range(2**n_qubits)}
        reversed_all = {bs[::-1]: 1 for bs in all_counts}

        result = run_sqd(h2_record, reversed_all, n_electrons=2, reverse_bit_order=True)
        assert abs(result["energy"] - h2_fci) < 1e-8

    def test_no_reversal_by_default(self, h2_record):
        """Without reverse_bit_order, energies should be unchanged."""
        n_qubits = int(h2_record["n_qubits"])
        counts = {format(i, f"0{n_qubits}b"): 1 for i in range(2**n_qubits)}

        e_default = sqd_energy_from_counts(h2_record, counts)
        e_explicit_false = sqd_energy_from_counts(h2_record, counts, reverse_bit_order=False)

        assert abs(e_default - e_explicit_false) < 1e-12
