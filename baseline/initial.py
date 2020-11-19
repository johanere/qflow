import numpy as np

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import MetropolisSampler
from qflow.training import EnergyCallback, ParameterCallback, train

from qflow.statistics import compute_statistics_for_series
from qflow.optimizers import AdamOptimizer

P, D = 1, 1  # Particles, dimensions
system = np.empty((P, D))
H = CoulombHarmonicOscillator()

psi = SimpleGaussian(alpha=0.5)

psi_sampler = MetropolisSampler(system, psi, step_size=1)
psi_energies = EnergyCallback(samples=100000)
psi_parameters = ParameterCallback()

print(psi_parameters)

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
