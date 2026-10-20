
from multiprocessing import Pool, cpu_count
from time import perf_counter

import matplotlib.pyplot as plt
from mpmath import mp, mpmathify, polylog
from numpy import abs as np_abs, array_split, concat, empty_like, exp, log, log10, maximum, pi, sqrt, where, isnan, isinf
from numpy.random import seed, uniform, choice

from scipy.special import spence as spence_std
from scipy_mod.special import spence as spence_mod


MACHINE_EPSILON = 2.22044604925031308e-16

mp.dps = 25

seed(1)
ntest = 1_000_000


def mp_spence_chunk(chunk):
    res = empty_like(chunk)
    for k, zk in enumerate(chunk):
        res[k] = complex(polylog(2, 1 - mpmathify(zk)))
    return res


def spence_mp_parallel(z, dps=25, chunks_per_proc: int = 4):

    mp.dps = dps

    proc = cpu_count()
    n_chunks = proc * chunks_per_proc
    chunks = array_split(z, n_chunks)

    with Pool(proc) as pool:
        results = pool.map(mp_spence_chunk, chunks)

    return concat(results)


def loguniform(a, b, size):
    return exp(uniform(log(a), log(b), size=size))


def uniform_box(zmin, zmax):
    return uniform(zmin.real, zmax.real, size=(ntest,)) + 1.0j * uniform(zmin.imag, zmax.imag, size=(ntest,))


def uniform_annulus(center, radius_range):
    radius_min, radius_max = radius_range
    theta = uniform(0.0, pi + pi, size=(ntest,))
    r = sqrt(uniform(radius_min * radius_min, radius_max * radius_max, size=(ntest,)))
    return center + r * exp(1.0j * theta)


def loguniform_annulus(center, radius_range):
    radius_min, radius_max = radius_range
    theta = uniform(0.0, pi + pi, size=(ntest,))
    r = loguniform(radius_min, radius_max, size=(ntest,))
    return center + r * exp(1.0j * theta)


class Test:

    def __init__(self, z, name: str = "", verbose: bool = True):

        self.z = z
        self.name = name
        self.verbose = verbose

        self.std_time = self.mod_time = None
        self.std_val = self.mod_val = self.actual_val = None

    def say(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def test(self):

        self.say("\n——————————————————————————————————————————————————")
        self.say(f"Testing: {self.name}")

        tastd = perf_counter()
        self.std_val = spence_std(self.z)
        tbstd = perf_counter()

        nan_std = isnan(self.std_val) | isinf(self.std_val)

        if nan_std.any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("CRITICAL: StdNaN", self.z[nan_std])
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.std_time = tbstd - tastd
        self.say(f"std time : {self.std_time}")

        tamod = perf_counter()
        self.mod_val = spence_mod(self.z)
        tbmod = perf_counter()

        nan_mod = isnan(self.mod_val) | isinf(self.mod_val)

        if nan_mod.any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("CRITICAL: ModNaN", self.z[nan_mod])
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.mod_time = tbmod - tamod
        self.say(f"mod time : {self.mod_time}")

        self.actual_val = spence_mp_parallel(self.z, chunks_per_proc=4)
        denom = np_abs(self.actual_val)
        denom = 1.0 / where(denom == 0.0, 1.0, denom)

        self.std_relerr = np_abs(self.actual_val - self.std_val) * denom
        self.mod_relerr = np_abs(self.actual_val - self.mod_val) * denom


standard = [
    Test(uniform_annulus(1.0, (0.0, 0.25)), "|z - 1| < 1/4"),
    Test(uniform_annulus(1.0, (0.25, 1.0)), "1/4 < |z - 1| < 1"),
    Test(uniform_annulus(0.0, (0.0, 0.25)), "|z| < 1/4"),
    Test(uniform_annulus(0.0, (0.25, 1.0)), "1/4 < |z| < 1"),
    Test(loguniform_annulus(0.0, (1.0, 100.0)), "1 < |z| < 100"),
    Test(uniform_box(-10.0 - 10.0j, 10.0 + 10.0j), "{Re(z), Im(z)} ∈ (-10, 10)")
]


extremes = [
    Test(loguniform_annulus(1.0, (1.0e-16, 1.0e-2)), "10^-16 < |z - 1| < 10^-2"),
    Test(loguniform_annulus(0.0, (1.0e-16, 1.0e-2)), "10^-16 < |z| < 10^-2"),
    Test(loguniform_annulus(0.0, (1.0e-300, 1.0e-16)), "10^-300 < |z| < 10^-16"),
    Test(loguniform_annulus(0.0, (1.0e+2, 1.0e+8)), "10^2 < |z| < 10^8"),
    Test(loguniform_annulus(0.0, (1.0e+8, 1.0e+50)), "10^8 < |z| < 10^50"),
    Test(loguniform_annulus(0.0, (1.0e+50, 1.0e+300)), "10^50 < |z| < 10^300")
]


danger = [
    Test(uniform(-5.0, 5.0, size=(ntest,)) + 
         1.0j * choice([-1.0, 1.0], size=(ntest,)) * loguniform(1.0e-300, 1.0e-2, size=(ntest,)), 
         "Near Im(z) = 0, Re(z) ∈ (-5, 5)"),
    Test(uniform(-5.0, 5.0, size=(ntest,)) + 0.0j, "Im(z) = 0, Re(z) ∈ (-5, 5)"),
    Test(choice([-1.0, 1.0], size=(ntest,)) * loguniform(1.0e-300, 1.0e-1, size=(ntest,)) + 
         1.0j * uniform(-5.0, 5.0, size=(ntest,)), 
         "Near Re(z) = 0, Im(z) ∈ (-5, 5)"),
    Test(1.0 + choice([-1.0, 1.0], size=(ntest,)) * loguniform(1.0e-16, 1.0e-1, size=(ntest,)) + 
         1.0j * uniform(-5.0, 5.0, size=(ntest,)), 
         "Near Re(z) = 1, Im(z) ∈ (-5, 5)"),
    Test(0.5 + choice([-1.0, 1.0], size=(ntest,)) * loguniform(1.0e-16, 1.0e-1, size=(ntest,)) + 
         1.0j * uniform(-5.0, 5.0, size=(ntest,)), 
         "Near Re(z) = 1/2, Im(z) ∈ (-5, 5)"),
    Test(uniform_annulus(0.0, (0.99, 1.01)), "Near |z| = 1"),
    Test(uniform_annulus(1.0, (0.99, 1.01)), "Near |z - 1| = 1")
]


special = [
    Test(uniform_box(-50.0 + 0.0j, 50.0 + 0.0j), "Im(z) = 0")
]


def visualize_test_results(tests: list[Test], description: str = "", error_filename: str = "plot1", 
                           speed_filename: str = "plot2", image_dpi: int = 200):

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(tests)), 6))

    group_spacing = 3.0   # distance between group centers
    offset = 0.5          # half the distance between the two violins in a group
    width = 0.8

    std_color, mod_color = "tab:blue", "tab:green"
    centers = []

    for k, t in enumerate(tests):

        t.test()

        center = k * group_spacing
        centers.append(center)

        std_err = log10(maximum(t.std_relerr, 1.0e-18))
        mod_err = log10(maximum(t.mod_relerr, 1.0e-18))

        for data, pos, color in (
            (std_err, center - offset, std_color),
            (mod_err, center + offset, mod_color),
        ):
            parts = ax.violinplot(data, positions=[pos], widths=width, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.6)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                parts[key].set_edgecolor(color)

        # Peak error annotation above each violin
        for err, pos, color in (
            (t.std_relerr, center - offset, std_color),
            (t.mod_relerr, center + offset, mod_color),
        ):
            peak = err.max()
            ax.annotate(f"{peak:.3g}", xy=(pos, log10(max(peak, 1.0e-18))),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(centers)
    ax.set_xticklabels([t.name for t in tests])
    ax.set_ylabel(r"$\log_{10}$(relative error)")
    ax.set_title(f"Relative errors of the Spence function implementations ({description}). n = {f"{ntest:_}".replace("_", " ")}.")
    ax.grid(axis="y", alpha=0.3)

    ax.axhline(log10(MACHINE_EPSILON), color="gray", linestyle="--", linewidth=1.2, zorder=0)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.legend(handles=[
        Patch(facecolor=std_color, alpha=0.6, label="scipy.special.spence"),
        Patch(facecolor=mod_color, alpha=0.6, label="New spence implementation"),
        Line2D([0], [0], color="gray", linestyle="--", label="Machine Epsilon"),
    ])

    fig.tight_layout()
    fig.savefig(error_filename, dpi=image_dpi)
    plt.close(fig)

    # ---------------------------------------------------------------- speed plot
    std_times = [t.std_time for t in tests]
    mod_times = [t.mod_time for t in tests]

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(tests)), 6))

    ax.bar([c - offset for c in centers], std_times, width=width * 0.9,
           color=std_color, alpha=0.6, label="scipy.special.spence")
    ax.bar([c + offset for c in centers], mod_times, width=width * 0.9,
           color=mod_color, alpha=0.6, label="New spence implementation")

    # Annotate each bar with its total time
    for c, s, m in zip(centers, std_times, mod_times):
        ax.text(c - offset, s, f"{s:.3g}", ha="center", va="bottom", fontsize=8, color=std_color)
        ax.text(c + offset, m, f"{m:.3g}", ha="center", va="bottom", fontsize=8, color=mod_color)

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{t.name}\n{(t.std_time / t.mod_time):.3g}x Speedup" for t in tests])
    ax.set_ylabel("Total time (s)")
    ax.set_title(f"Speeds of the Spence function implementations ({description}). n = {f"{ntest:_}".replace("_", " ")}.")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(speed_filename, dpi=image_dpi)
    plt.close(fig)

    print("\n\nFinished!\n")


visualize_test_results(standard, description="Normal values",
                       error_filename="normal_error", speed_filename="normal_speed")

visualize_test_results(extremes, description="Extreme values",
                       error_filename="extreme_error", speed_filename="extreme_speed")

visualize_test_results(danger, description="Dangerous regions",
                       error_filename="danger_error", speed_filename="danger_speed")
