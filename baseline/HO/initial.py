import numpy as np

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import MetropolisSampler
from qflow.training import EnergyCallback, ParameterCallback, train

from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.optimizers import AdamOptimizer

from qflow.mpi import mpiprint, master_rank


def plot_training(energies, parameters):
    _, (eax, pax) = plt.subplots(ncols=2)
    eax.plot(energies, label=r"$\langle E_L\rangle$ [a.u]")
    eax.set_xlabel(r"% of training")
    eax.axhline(y=3, label="Exact", linestyle="--", color="k", alpha=0.5)
    eax.legend()

    pax.plot(np.asarray(parameters)[:, [0, 3]])
    pax.set_xlabel(r"% of training")
    pax.legend([r"$\alpha_G$", r"$\beta_{PJ}$"])

    matplotlib2tikz.save(__file__ + ".tex")


P, D = 1, 1  # Particles, dimensions
system = np.empty((P, D))

H = CoulombHarmonicOscillator()

psi = SimpleGaussian(alpha=0.5)
psi_sampler = MetropolisSampler(system, psi, step_size=1)

psi_energies = EnergyCallback(samples=100000)
psi_parameters = ParameterCallback()


train(
    psi,
    H,
    psi_sampler,
    iters=2000,
    samples=1000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_energies, psi_parameters),
)

stats = [
    compute_statistics_for_series(
        H.local_energy_array(psi_sampler, psi, 2 ** 22), method="blocking"
    ),
]

print(psi_parameters)
print(psi_energies)
print(stats)
labels = [r"$\Phi$", r"$\psi_{PJ}$"]

mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))
mpiprint(psi.parameters)
