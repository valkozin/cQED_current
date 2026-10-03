# Cavity-induced persistent current in a π loop

A superconducting ring closed by a π junction and threaded by the flux of a quantum LC mode.
This folder holds a check of the original mean-field solution, an exact DMRG solution, the interaction-driven (U) π junction,
and the body of a PRL draft.

| Path | Contents |
|---|---|
| `notebooks/` | Original Mathematica mean-field notebook |
| `docs/01_review_ru.md` | Review of the approach, the physics and the mean-field solution (in Russian) |
| `docs/02_dmrg_results_ru.md` | DMRG results and comparison with mean field (in Russian) |
| `paper/` | PRL body (`main.tex` → `body.tex`, `results.tex`, `feasibility.tex`, `refs.bib`); `main.pdf` |
| `figs/` | Figures (pdf/png), produced by `code/make_figures.py` |
| `data/` | All numerical results (json/npy) and run logs |

## Code (`code/`)
* `model.py`, `mean_field.py`: line-by-line Python port of the notebook. It reproduces every curve of the notebook.
* `bo.py`: adiabatic (Born–Oppenheimer) solution. The photon moves in the exact electronic E(φ_cl + gX).
* `bdg_general.py`, `mf_general.py`: BdG and mean field for general rings (weak link / dot).
* `dmrg_ring.py`: TeNPy model of the ring plus photon, i.e. the full electron–photon problem with the Peierls vertex e^{ig(a+a†)}.
* `run_dmrg_notebook.py`, `run_dmrg_scan.py`, `run_udot_phase.py`: production DMRG runs.
* `bo_scan.py`, `bo_udot.py`, `bo_phase_diagram.py`: adiabatic scans.
* `tests/`: DMRG against exact BdG (g=0) and against exact diagonalization (small ring + photon).

Requirements: `numpy scipy matplotlib physics-tenpy==1.1.1`. The runs took about 3 h on 4 cores.
Example: `cd code && python run_dmrg_scan.py udot:2 10 0.5,1,1.5,2,3 256 50 ../data/out.json 0.00761`.
