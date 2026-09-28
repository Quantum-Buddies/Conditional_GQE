// Fast parity computation C++ extension for QWC result parsing.
//
// Compile: pip install . (or python setup.py build_ext --inplace)
// Usage:
//   from src.gqe.accel._fast_parity import fast_expectations
//   energy, exps = fast_expectations(counts, masks, coeffs, n_qubits, n_shots)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>

namespace py = pybind11;

// Compute parity of a bitstring under a mask: popcount(bs & mask) % 2
static inline int parity(const std::string& bs, uint64_t mask, int n_qubits) {
    uint64_t bs_val = 0;
    for (int q = 0; q < n_qubits && q < 64; q++) {
        if (bs[q] == '1') {
            bs_val |= (1ULL << q);
        }
    }
    return __builtin_popcountll(bs_val & mask) % 2;
}

// Compute expectations for all terms in a QWC group from shared counts
py::tuple fast_expectations(
    const py::dict& counts,
    const py::array_t<int64_t>& masks_arr,
    const py::array_t<double>& coeffs_arr,
    int n_qubits,
    int n_shots
) {
    auto masks = masks_arr.unchecked<1>();
    auto coeffs = coeffs_arr.unchecked<1>();
    int n_terms = masks.shape(0);

    // Extract counts
    std::vector<std::string> bitstrings;
    std::vector<double> count_vals;
    for (auto item : counts) {
        bitstrings.push_back(item.first.cast<std::string>());
        count_vals.push_back(item.second.cast<double>());
    }
    int n_bs = bitstrings.size();

    // Compute expectations
    std::vector<double> expectations(n_terms, 0.0);
    double energy = 0.0;

    for (int ti = 0; ti < n_terms; ti++) {
        uint64_t mask = masks(ti);
        double exp = 0.0;
        for (int bi = 0; bi < n_bs; bi++) {
            int p = parity(bitstrings[bi], mask, n_qubits);
            double sign = (p == 1) ? -1.0 : 1.0;
            exp += sign * count_vals[bi] / n_shots;
        }
        expectations[ti] = exp;
        energy += coeffs(ti) * exp;
    }

    // Build result tuple
    py::list exp_list;
    for (int i = 0; i < n_terms; i++) {
        exp_list.append(expectations[i]);
    }

    return py::make_tuple(energy, exp_list);
}

// Batch parity for multiple bitstrings and masks
py::array_t<int8_t> batch_parity(
    const py::array_t<uint8_t>& bitstrings_arr,
    const py::array_t<int64_t>& masks_arr,
    int n_qubits
) {
    auto bs = bitstrings_arr.unchecked<2>();
    auto masks = masks_arr.unchecked<1>();
    int n_bs = bs.shape(0);
    int n_terms = masks.shape(0);

    auto result = py::array_t<int8_t>({n_terms, n_bs});
    auto r = result.mutable_unchecked<2>();

    for (int ti = 0; ti < n_terms; ti++) {
        uint64_t mask = masks(ti);
        for (int bi = 0; bi < n_bs; bi++) {
            uint64_t bs_val = 0;
            for (int q = 0; q < n_qubits && q < 64; q++) {
                if (bs(bi, q)) {
                    bs_val |= (1ULL << q);
                }
            }
            r(ti, bi) = static_cast<int8_t>(__builtin_popcountll(bs_val & mask) % 2);
        }
    }

    return result;
}

PYBIND11_MODULE(_fast_parity, m) {
    m.doc() = "Fast parity computation for QWC result parsing";
    m.def("fast_expectations", &fast_expectations,
          "Compute expectations for all terms in a QWC group from shared counts",
          py::arg("counts"), py::arg("masks"), py::arg("coeffs"),
          py::arg("n_qubits"), py::arg("n_shots"));
    m.def("batch_parity", &batch_parity,
          "Batch parity computation for multiple bitstrings and masks",
          py::arg("bitstrings"), py::arg("masks"), py::arg("n_qubits"));
}
