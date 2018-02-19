#ifndef VMCSOLVER_HPP
#define VMCSOLVER_HPP

#include <ostream>
#include <armadillo>

namespace VMC {

namespace Constants {
} // namespace Constants

enum Dimensions {
    DIM_1 = 1, DIM_2, DIM_3
};
enum class HOType {
    SYMMETRIC, ELLIPTICAL
};
enum class InteractionType {
    OFF = 0, ON
};

enum class AnalyticAcceleration {
    OFF = 0, ON
};

struct Results {
    double E, E2, variance, alpha, beta;
};

struct VMCConfiguration {
    int n_particles;
    Dimensions dims;
    HOType ho_type;
    InteractionType interaction;
    AnalyticAcceleration acceleration;
    double omega_ho;
    double omega_z;
    double a;
    double h;
    double h2;
    double step_length;
};

class VMCSolver {
private:
    const VMCConfiguration _config;
    double _alpha = 0.5, _beta = 1;

public:
    VMCSolver(const VMCConfiguration &config);

    double V_ext(const arma::mat &R) const;

    double V_int(const arma::mat &R) const;

    double Psi_f(const arma::mat &R) const;

    double Psi_g(const arma::mat &R) const;

    double Psi(const arma::mat &R) const;

    double E_kinetic(arma::mat &R) const;

    double E_local(arma::mat &R) const;

    Results run_MC(const int n_cycles) const;

    Results vmc(const int n_cycles,
                std::ostream &out,
                const double alpha_min,
                const double alpha_max,
                const double alpha_n,
                const double beta_min = 1,
                const double beta_max = 1,
                const double beta_n = 1);
};

}  // namespace VMC

#endif // VMCSOLVER_HPP
