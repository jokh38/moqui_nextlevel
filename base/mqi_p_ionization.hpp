#ifndef MQI_P_IONIZATION_HPP
#define MQI_P_IONIZATION_HPP

#include <moqui/base/mqi_interaction.hpp>

namespace mqi
{



///< delta_ionization
///< analytical model
template<typename R>
class p_ionization_tabulated : public interaction<R, mqi::PROTON>
{
    ///< constant value to calculate scattering angle
    const R Es = 13.9;   // I 75 eV
    ///< Cross-section & restricted stopping power
    ///< Table's energy step: Ei, Ef, dE
    const R  Ei;
    const R  Ef;
    const R  E_step;
    cudaTextureObject_t tex_;
public:
    CUDA_HOST_DEVICE
    p_ionization_tabulated(R m, R M, R s, cudaTextureObject_t tex) :
        Ei(m), Ef(M), E_step(s), tex_(tex) {}

    CUDA_HOST_DEVICE
    ~p_ionization_tabulated() {
    }

    ///< Cross-section
    CUDA_HOST_DEVICE
    virtual R
    cross_section(const relativistic_quantities<R>& rel, const material_t<R>& mat) {
        R cs = 0;
        if (rel.Ek >= Ei && rel.Ek <= Ef) {
            // The u-coordinate is the energy index + 0.5 for center sampling
            float u = (rel.Ek - Ei) / this->E_step + 0.5f;
            // The v-coordinate corresponds to the table index (0 for cs_p_ion_table)
            cs = tex2D<float>(this->tex_, u, 0.5f);
        }
        cs *= mat.rho_mass;
        return cs;
    }

    ///< dEdx
    CUDA_DEVICE
    virtual inline R
    dEdx(const relativistic_quantities<R>& rel, const material_t<R>& mat) {
        R pw = 0;
        if (rel.Ek >= Ei && rel.Ek <= Ef) {
            float u = (rel.Ek - Ei) / this->E_step + 0.5f;
            // v-coordinate is 1.5f for restricted_stopping_power_table
            pw = tex2D<float>(this->tex_, u, 1.5f);
        } else if (rel.Ek < Ei && rel.Ek > 0) {
            pw = tex2D<float>(this->tex_, 0.5f, 1.5f); // Read first element
            assert(pw >= 0);
        }
        return -1.0 * pw;
    }

    ///< sample delta-energy
    CUDA_DEVICE
    inline R
    sample_delta_energy(const R Te_max, const mqi_rng* rng) {
        R eta = mqi_uniform<R>(rng);
        return Te_max * this->T_cut / ((1.0 - eta) * Te_max + eta * this->T_cut);
    }

    ///< Energy loss (positive)
    CUDA_DEVICE
    virtual inline R
    energy_loss(const relativistic_quantities<R>& rel,
                material_t<R>&                    mat,
                const R                           step_length,
                mqi_rng*                          rng) {
        ///< n is left index of energy & range steps table
        R length_in_water = step_length * mat.stopping_power_ratio(rel.Ek) * mat.rho_mass / this->units.water_density;
        //R length_in_water = step_length * 1 * mat.rho_mass / this->units.water_density;
        uint16_t n  = uint16_t((rel.Ek - this->Ei) / this->E_step);
        R        x0 = this->Ei + n * this->E_step;
        R        x1 = x0 + this->E_step;
        if (x0 > rel.Ek) n -= 1;
        if (x1 < rel.Ek) n += 1;

        float u = (rel.Ek - Ei) / this->E_step + 0.5f;
        R r = tex2D<float>(this->tex_, u, 2.5f);
        if (r < length_in_water) return rel.Ek;   //< maximum energy loss
        r -= length_in_water;                     //< update residual range

        // This part is tricky. The original code iterates to find the new energy.
        // A direct texture-based approach for the inverse lookup is not straightforward.
        // For now, I will keep the loop, but this could be a performance bottleneck.
        // A better approach might be to pre-calculate an inverse table or use a more
        // sophisticated search on the texture, but that's a larger change.
        do {
            if (r >= tex2D<float>(this->tex_, n + 0.5f, 2.5f)) break;
        } while (--n > 0);

        R r0 = tex2D<float>(this->tex_, n + 0.5f, 2.5f);
        R r1 = tex2D<float>(this->tex_, n + 1.5f, 2.5f);
        R e0 = this->Ei + n * this->E_step;
        R e1 = e0 + this->E_step;

        R dE_mean = rel.Ek - mqi::intpl1d(r, r0, r1, e0, e1);
        R dE_var  = this->energy_straggling(rel, mat, length_in_water);
        R ret     = mqi::mqi_normal(rng, dE_mean, mqi::mqi_sqrt(dE_var));
        if (ret < 0) ret *= -1.0;
        return ret;
    }

    ///< energy_straggling variance
    CUDA_DEVICE
    inline R
    energy_straggling(const relativistic_quantities<R>& rel,
                      const material_t<R>&              mat,
                      const R                           step_length) {
        R Te   = (rel.Te_max >= 0.08511) ? 0.08511 : rel.Te_max;
        R O_sq = mat.dedx_term0() * mat.rho_mass / this->units.water_density * step_length;
        O_sq *= Te / rel.beta_sq * (1.0 - 0.5 * rel.beta_sq);
        return O_sq;
    }

    ///< calculate radiation length based on density (mm)
    ///< TODO: this better to be in material
    /// rho_mass (g/mm^3)
    /// From M. Fippel and M. Soukup, Med. Phys. Vol. 31, No. 8, 2004
    CUDA_DEVICE
    virtual R
    radiation_length(R density) {
        R radiation_length_mat = 0.0;
        R f                    = 0.0;
        density *= 1000.0;

        //// Fippel
        if (density <= 0.26) {
            f = 0.9857 + 0.0085 * density;
        } else if (density > 0.26 && density <= 0.9) {
            f = 1.0446 - 0.2180 * density;
        } else if (density > 0.9) {
            f = 1.19 + 0.44 * mqi::mqi_ln(density - 0.44);
        }

        radiation_length_mat =
          (this->units.water_density * this->units.radiation_length_water) / (density * 0.001 * f);
        return radiation_length_mat;
    }

    ///< CSDA method is special to p_ionization
    CUDA_DEVICE
    virtual void
    along_step(track_t<R>&       trk,
               track_stack_t<R>& stk,
               mqi_rng*          rng,
               const R           len,
               material_t<R>&    mat) {
        mqi::relativistic_quantities<R> rel(trk.vtx0.ke, this->units.Mp);
        ///< CSDA energy loss
#ifdef DEBUG
        printf("len %f rsp %f density %f water density %f length in water %f mm\n",
               len,
               mat.stopping_power_ratio(rel.Ek),
               mat.rho_mass,
               this->units.water_density,
               len * mat.stopping_power_ratio(rel.Ek) * mat.rho_mass / this->units.water_density);
#endif
        R dE = this->energy_loss(rel, mat, len, rng);
        ///< Update track (KE & POS & DIR)
        R r = 1.0;
        if (dE >= trk.vtx0.ke) {
            r = trk.vtx0.ke / dE;
            trk.stop();
        }
        assert(dE * r >= 0);
        ///< Multiple Coulomb SCattering (MSC)
        R P                    = rel.momentum();
        R radiation_length_mat = this->radiation_length(mat.rho_mass);

        R th_sq = ((this->Es / P) * (this->Es / P) / rel.beta_sq) * len / radiation_length_mat;
        R th    = mqi::mqi_sqrt(th_sq);
        th      = mqi::mqi_normal<R>(rng, 0, mqi::mqi_sqrt(2.0f) * th);
        if (th < 0) th *= -1.0;
        R phi = 2.0 * M_PI * mqi::mqi_uniform<R>(rng);
#if !defined(__CUDACC__)
        if (std::isnan(th) || std::isnan(phi) || std::isinf(th) || std::isinf(phi))
            printf("p ion1 dE %f ke %f Es %f P %f len %f th_sq %f th %f phi %f\n",
                   dE,
                   trk.vtx0.ke,
                   this->Es,
                   P,
                   len,
                   th_sq,
                   th,
                   phi);
        if (std::isnan(r)) printf("p ion1 r %f\n", r);
#endif
        trk.update_post_vertex_direction(th, phi);
        if (mqi::mqi_abs(trk.vtx0.dir.dot(trk.vtx1.dir) /
                           (trk.vtx0.dir.norm() * trk.vtx1.dir.norm()) -
                         mqi::mqi_cos(th)) < 1e-3) {
        } else {
#ifdef DEBUG
            printf("cos(th) %f dot %f\n",
                   mqi::mqi_cos(th),
                   trk.vtx0.dir.dot(trk.vtx1.dir) / (trk.vtx0.dir.norm() * trk.vtx1.dir.norm()));
            printf("vtx0 ");
            trk.vtx0.dir.dump();
            printf("vtx1 ");
            trk.vtx1.dir.dump();
#endif
        }
        assert(mqi::mqi_abs(trk.vtx0.dir.dot(trk.vtx1.dir) /
                              (trk.vtx0.dir.norm() * trk.vtx1.dir.norm()) -
                            mqi::mqi_cos(th)) < 1e-3);
        trk.deposit(dE * r);
        trk.update_post_vertex_position(r * len);
        trk.update_post_vertex_energy(dE * r);
    }

    ///< DoIt method to update track's KE, pos, dir, dE, status
    ///< compute energy loss, vertex, secondaries
    CUDA_DEVICE
    virtual void
    post_step(track_t<R>&       trk,
              track_stack_t<R>& stk,
              mqi_rng*          rng,
              const R           len,
              material_t<R>&    mat,
              bool              score_local_deposit) {
        //This method in p_ion should get called after CSDA
        mqi::relativistic_quantities<R> rel(trk.vtx1.ke, this->units.Mp);

        ///< Delta generation (local absorb)
        R Te, n;

        /// Sampling and Rejection from Geant4
        while (true) {
            n  = mqi::mqi_uniform<R>(rng);
            Te = this->T_cut * rel.Te_max;
            Te /= ((1.0 - n) * rel.Te_max + n * this->T_cut);
            if (mqi::mqi_uniform<R>(rng) <
                1.0 - rel.beta_sq * Te / rel.Te_max + Te * Te / (2.0 * rel.Et_sq)) {
                break;
            }
        }
        assert(Te >= 0);

        ///< Te is assumed to be absorbed locally
        /// Remove in release
#ifdef __PHYSICS_DEBUG__
        track_t<R> daughter(trk);
        daughter.dE       = Te;
        daughter.primary  = false;
        daughter.process  = mqi::D_ION;
        daughter.vtx0.ke  = 0;
        daughter.vtx1.ke  = 0;
        daughter.status   = CREATED;
        daughter.vtx0.pos = trk.c_node->geo->rotation_matrix_fwd *
                              (daughter.vtx0.pos - trk.c_node->geo->translation_vector) +
                            trk.c_node->geo->translation_vector;
        daughter.vtx0.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx0.dir;
        daughter.vtx1.pos = trk.c_node->geo->rotation_matrix_fwd *
                              (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                            trk.c_node->geo->translation_vector;
        daughter.vtx1.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
        stk.push_secondary(daughter);
#else
        trk.deposit(Te);
#endif
        trk.update_post_vertex_energy(Te);
    }

    ///< DoIt method to update track's KE, pos, dir, dE, status
    ///< compute energy loss, vertex, secondaries
    CUDA_DEVICE
    virtual void
    last_step(track_t<R>& trk, material_t<R>& mat) {
        mqi::relativistic_quantities<R> rel(trk.vtx0.ke, this->units.Mp);
        R                               length_in_water = 0;
        if (trk.dE > 0) length_in_water = -trk.dE / this->dEdx(rel, mat);
        R step_length = length_in_water * this->units.water_density / (mat.stopping_power_ratio(trk.vtx0.ke) * mat.rho_mass);
        //R step_length = length_in_water * this->units.water_density / (1 * mat.rho_mass);
        trk.update_post_vertex_position(step_length);
    }
};

}   // namespace mqi

#endif