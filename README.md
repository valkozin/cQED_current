# Zero-point flux fluctuations switch a quantum-dot Josephson loop from 0 to π

A superconducting ring closed by a π junction and threaded by the flux of a quantum LC mode.
This folder holds a check of the original mean-field solution, an exact DMRG solution, the interaction-driven (U) π junction,
and the body of a PRL draft.

| Path | Contents |
|---|---|
| `notebooks/` | Original Mathematica mean-field notebook |
| `docs/01_review_ru.md` | Review of the approach, the physics and the mean-field solution (in Russian) |
| `docs/02_dmrg_results_ru.md` | DMRG results and comparison with mean field (in Russian) |
| `docs/03_phase_diagram_U_g_ru.md` | First (adiabatic) (U,g) diagram, L=14 (in Russian) |
| `docs/04_phase_diagram_full_dmrg_ru.md` | Full-DMRG (U,g) phase diagram, protocol, convergence, comparison (in Russian) |
| `docs/05_revision_ru.md` | October 2026 revision: what was recomputed with the two-start protocol, unified parameters (L=10), text merge (in Russian) |
| `docs/06`, `docs/07`, `docs/09` | Frequency regime, strong π junction (t_J=t, U=6Δ) and physical units (imported from the side branch; in Russian) |
| `docs/10_prl_revision_ru.md` | PRL revision: novelty audit with prior art, the new flux-driven 0–π switch, list of fixes (in Russian) |
| `paper/` | PRL manuscript (`main.tex` → `body.tex`, `cavity0pi.tex`, `phase_diagram.tex`, `results.tex`, `feasibility.tex`, `supplement.tex`; `refs.bib`); `main.pdf` (main text + Supplemental Material) |
| `figs/` | Figures (pdf/png), produced by `code/make_figures.py` and `code/make_phase_figs.py` (`fig_main_phase_diagram` is the superseded L=14 adiabatic diagram, kept for reference) |
| `talk/` | Beamer seminar talks: full version `talk.tex` → `talk.pdf`, short version `talk_short.tex` → `talk_short.pdf`; teaching figures from `make_talk_figs.py`, `make_short_figs.py` |
| `data/` | All numerical results (json/npy) and run logs; `data/full/` the full-DMRG (U,g) grid, `data/rerun/` the beta scans with the two-start protocol, `data/nonadiabatic/` the flux-driven 0–π switch (full DMRG at ħω = 0.01–1 Δ, both parity sectors, plus the Lang–Firsov reference), `data/om0.003_tJ1.0/`, `data/tJ1.0/` the strong junction |

## Code (`code/`)
* `model.py`, `mean_field.py`: line-by-line Python port of the notebook. It reproduces every curve of the notebook.
* `bo.py`: adiabatic (Born–Oppenheimer) solution. The photon moves in the exact electronic E(φ_cl + gX).
* `bdg_general.py`, `mf_general.py`: BdG and mean field for general rings (weak link / dot).
* `dmrg_ring.py`: TeNPy model of the ring plus photon, i.e. the full electron–photon problem with the Peierls vertex e^{ig(a+a†)}.
* `run_dmrg_notebook.py`, `run_udot_phase.py`: production DMRG runs (notebook model, E(phi) at L=14); `run_dmrg_scan.py`: the original beta scans (superseded).
* `run_beta_scan.py`: beta scans (half-flux weak link at L=14, seed-free cat state of the U=2 pi loop at L=10) with the two-start protocol, sum rule and excited state.
* `bo_scan.py`, `bo_udot.py`, `bo_phase_diagram.py`: adiabatic scans (the last two default to the L=10 harmonics `data/harmonics_udot_U2_Nr10.npy`).
* `run_full_grid.py`: production full-DMRG (U,g) grid with two initial states and seed annealing; `repair_points.py`, `validate_grid.py`.
* `ring_ephi.py`, `ring_ephi_even.py`: E(phi;U) of the ring at g=0 in both parity sectors (input of the effective model).
* `effective_grid.py`: effective circuit model (fermions integrated out) on the same grid; `mf_grid.py`: mean field on the same grid.
* `make_phase_figs.py`: figures of the phase-diagram section (`figs/fig_phase_*.pdf`); `make_fig_main.py`: main phase-diagram figure with the seed-free order parameter δ(U,g).
* `run_nonadiabatic.py`: full DMRG in both parity sectors at finite ħω (flux-driven 0–π switch); `analyze_nonadiabatic.py`: U_c(g,ω) against the circuit model and the Lang–Firsov limit, `figs/fig_cavity_0pi.pdf`.
* `bo_parity.py`: circuit model diagonalized in the two flux-parity blocks (tunnel splitting below 1e-8 ħω).
* `tests/`: DMRG against exact BdG (g=0) and against exact diagonalization (small ring + photon).

Requirements: `numpy scipy matplotlib physics-tenpy==1.1.1`. The first runs took about 3 h on 4 cores; the full-DMRG (U,g) grid about 20 CPU hours (`data/full/worker.sh`).
Example: `cd code && python run_dmrg_scan.py udot:2 10 0.5,1,1.5,2,3 256 50 ../data/out.json 0.00761`.
