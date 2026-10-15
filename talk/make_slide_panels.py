"""Slide-sized single panels cut from the paper figures (code/make_figures.py), without overlapping titles.
Run from talk/:  python make_slide_panels.py   -> figs/main_a.pdf, figs/main_bc.pdf, figs/overview_b.pdf"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_agg import FigureCanvasAgg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
sys.path.insert(0, '.')
import make_figures as mf

captured = []
_save = matplotlib.figure.Figure.savefig
matplotlib.figure.Figure.savefig = lambda self, *a, **k: captured.append(self) if self not in captured else None


def crop(fig, axes, name, pad=0.04):
    FigureCanvasAgg(fig); fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = Bbox.union([ax.get_tightbbox(r) for ax in axes]).transformed(fig.dpi_scale_trans.inverted())
    _save(fig, os.path.join(OUT, name), bbox_inches=bb.padded(pad))


# ---- main phase diagram
mf.fig_main(); fig = captured[-1]
a, b, axb, axc, cb = fig.axes
for ax in fig.axes:
    for loc in ('left', 'center', 'right'):
        ax.set_title('', loc=loc)
    for t in list(ax.texts):
        if t.get_text().startswith('('):
            t.remove()
a.set_title(r'$U=2t$', fontsize=8)
axb.set_title('exact (DMRG $E(\\varphi;U)$ + quantum photon)', fontsize=7)
axc.set_title('mean field (HF + product state)', fontsize=7)
crop(fig, [a, b], 'main_a.pdf')
crop(fig, [axb, axc, cb], 'main_bc.pdf')

# ---- overview: phase diagram of delta only
mf.fig_overview(); fig = captured[-1]
axs = fig.axes
for ax in axs:
    for t in list(ax.texts):
        if t.get_text().startswith('('):
            t.remove()
crop(fig, axs[1:], 'overview_b.pdf')
print('done')
