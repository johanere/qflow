#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "metropolissampler.hpp"

MetropolisSampler::MetropolisSampler(const System &system,
                                     const Wavefunction &wavefunction,
                                     Real step)
    : Sampler(system, wavefunction, step)
{
    initialize_system();
}

void MetropolisSampler::initialize_system() {
    for (int i = 0; i < _system_old.cols(); ++i) {
        for (int d = 0; d < _system_old.rows(); ++d) {
            _system_old(d, i) = _step * centered(rand_gen);
        }
    }
    _system_new = _system_old;
    _psi_old = (*_wavefunction)(_system_old);
}

void MetropolisSampler::perturb_system() {
    for (int d = 0; d < _system_new.rows(); ++d) {
        _system_new(d, _particle_to_move) += _step * centered(rand_gen);
    }
    _psi_new = (*_wavefunction)(_system_new);
}

Real MetropolisSampler::acceptance_probability() const {
    return square(_psi_new) / square(_psi_old);
}
