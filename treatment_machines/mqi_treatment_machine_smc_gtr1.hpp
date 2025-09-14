#ifndef MQI_TREATMENT_MACHINE_SMC_GTR1_H
#define MQI_TREATMENT_MACHINE_SMC_GTR1_H

// Samsung Medical Center Log file based QA treatment machine
// Gantry 1
// Modified by Chanil Jeon (2024-06-04 ver)
// Sungkyunkwan University, Samsung Medical Center

#include <moqui/base/materials/mqi_patient_materials.hpp>
#include <moqui/base/mqi_treatment_machine_ion.hpp>
#include <moqui/base/distributions/mqi_phsp6d_ray.hpp>
#include <moqui/treatment_machines/spline_interp.hpp>

namespace mqi{

//namespace smc{
template<typename R>
class gtr1_material_t: public patient_material_t<R> {
public:
    // Customize material not implemented
    CUDA_HOST_DEVICE
    gtr1_material_t(): patient_material_t<R>(){;}

    CUDA_HOST_DEVICE
    gtr1_material_t(int16_t hu): patient_material_t<R>(hu){;}

    CUDA_HOST_DEVICE
    ~gtr1_material_t(){;}
};
/**
 * The <code>gtr1</code> class represents beam model for Sumitomo IMPT machine in SMC
 */
template <typename T>
class gtr1 : public treatment_machine_ion<T> {
protected:

public:

    // MU count to paticles in MC
    tk::spline particleCountCalibInterp;

    // Basic beam property
    tk::spline beamEnergySpreadInterp;
    tk::spline beamSpotSizeInterp;
    tk::spline beamAngularSpreadInterp;
    tk::spline beamDivergenceInterp;

    // Samsung Medical Center focal length value in Raystation
    // Added in 2024-06 by Chanil Jeon
    gtr1()
    {
        treatment_machine_ion<T>::SAD_[0] = 2696.0 ;
        treatment_machine_ion<T>::SAD_[1] = 2180.0 ;

        // Particle count calibration
        std::vector<double> beamEnergyForParticleCountCalib = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        //std::vector<double> beamEnergyForParticleCountCalib = { 70,76,84,92,100,106,112,118,126,134,142,146,150,154,158,162,170,174,178,182,186,190,198,202,206,214,218,222,226,230 };
        //std::vector<double> correctionParticleCountCalib = { 1, 1.065789513, 1.157943145, 1.2411556, 1.322953524, 1.395086046, 1.458739464, 1.513003859, 1.588552029, 1.66206693, 1.740579354, 1.780360004, 1.81434049, 1.848968773, 1.878347883, 1.91131038, 1.971904065, 2.003357087, 2.02810976, 2.058108192, 2.082444044, 2.108758102, 2.168636068, 2.20480403, 2.229879571, 2.308058176, 2.342322973, 2.381755055, 2.447585594, 2.511318715 };
        //std::vector<double> correctionParticleCountCalib = { 1, 1, 1 ,1 ,1 ,1 ,1 , 1 ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
        // this coefficient from 1million pristine
        //std::vector<double> correctionParticleCountCalib = { 1, 1.061202271, 1.152820922, 1.236176934, 1.315117669, 1.382373511, 1.444784618, 1.498103851, 1.584308142, 1.674583334, 1.740840309, 1.78211077, 1.815233009, 1.841546506, 1.873692717, 1.905568701, 1.963021494, 1.988838732, 2.014072993, 2.039611548, 2.064422174, 2.087853402, 2.151044281, 2.181950614, 2.208757017, 2.294883724, 2.33007653, 2.383619505, 2.475416227, 2.557063682};
        // this coefficeint from MC 
        //std::vector<double> correctionParticleCountCalib = { 1,1.060192371,1.146351916,1.227613124,1.306995149,1.371408536,1.430267126,1.48170962,1.562567026,1.635595481,1.705426149,1.744083164,1.775614932,1.80393292,1.83498652,1.86517345,1.920894724,1.946018216,1.971215037,1.996262731,2.021044522,2.045252565,2.103326605,2.132643026,2.157061866,2.234066845,2.267527471,2.306289072,2.372575026,2.43335286};
        // this coeiff new IDD 1mil and mcc
        //std::vector<double> correctionParticleCountCalib = { 1,1.042782203,1.125767439,1.2066805,1.283998322,1.350656474,1.413222686,1.464800686,1.551708925,1.638878772,1.705614436,1.746032617,1.776739459,1.805903666,1.836278699,1.866029589,1.921338606,1.950454191,1.974127237,1.999175855,2.021234107,2.047093542,2.107498016,2.140352464,2.166919681,2.247673987,2.285710347,2.338027899,2.425465293,2.507162943};
        // this coeiff new IDD and G1 data
        //std::vector<double> correctionParticleCountCalib = { 1,1.038971606,1.113049394,1.196195214,1.268389399,1.340031118,1.366475507,1.419510274,1.517871938,1.610132839,1.623037313,1.695937882,1.729128829,1.764211305,1.793367157,1.826367283,1.88783435,1.921629516,1.949175672,1.97770874,2.00190331,2.028659394,2.092364648,2.120932612,2.149684843,2.223622782,2.251734179,2.274334166,2.308709723,2.347431196};
        // IDD protons_per_Dose
        std::vector<double> correctionParticleCountCalib = { 1,1.043006406,1.129069661,1.188123361,1.272423541,1.327760158,1.402250896,1.464817523,1.54437654,1.633037205,1.707137572,1.747818,1.794019454,1.839009303,1.871200912,1.917344007,1.997447604,2.037391134,2.082718296,2.123947322,2.159990357,2.211800433,2.289580024,2.336590651,2.381026637,2.469431923,2.515466677,2.569724012,2.610777699,2.664859917};
        particleCountCalibInterp.set_points(beamEnergyForParticleCountCalib, correctionParticleCountCalib, tk::spline::cspline);
        particleCountCalibInterp.make_monotonic();

        // Beam Energy spread interpolation
        std::vector<double> beamEnergyForBeamEnergySpread = { 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230 };
        std::vector<double> correctionBeamEnergySpread = { 0.502793, 0.577766, 0.648957, 0.714868, 0.774127, 0.825533, 0.868103, 0.901097, 0.92404, 0.936734, 0.939254, 0.931934, 0.915346, 0.89027, 0.857653, 0.818572, 0.774184 };
        beamEnergySpreadInterp.set_points(beamEnergyForBeamEnergySpread, correctionBeamEnergySpread, tk::spline::cspline);
        beamEnergySpreadInterp.make_monotonic();

        // Beam Spot size interpolation
        std::vector<double> beamEnergyForBeamSpotSize = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        std::vector<double> correctionBeamSpotSize = { 10.70, 9.94, 9.10, 8.45, 7.88, 7.52, 7.18, 6.88, 6.52, 6.22, 5.94, 5.82, 5.70, 5.57, 5.47, 5.37, 5.16, 5.05, 4.94, 4.86, 4.76, 4.66, 4.50, 4.42, 4.34, 4.22, 4.15, 4.03, 3.85, 3.58 };
        beamSpotSizeInterp.set_points(beamEnergyForBeamSpotSize, correctionBeamSpotSize, tk::spline::cspline);
        beamSpotSizeInterp.make_monotonic();

        // Beam angular spread interpolation
        std::vector<double> beamEnergyForBeamAngularSpread = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        std::vector<double> correctionBeamAngularSpread = { 0.004675, 0.00424, 0.00379, 0.003605, 0.003315, 0.003045, 0.00294, 0.0028, 0.00253, 0.002435, 0.002285, 0.002205, 0.002295, 0.00224, 0.002075, 0.002155, 0.002075, 0.001865, 0.00196, 0.001855, 0.001885, 0.00195, 0.00205, 0.001945, 0.002, 0.00184, 0.0018, 0.001795, 0.00171, 0.001695 };
        beamAngularSpreadInterp.set_points(beamEnergyForBeamAngularSpread, correctionBeamAngularSpread, tk::spline::cspline);
        beamAngularSpreadInterp.make_monotonic();

        // Beam divergence interpolation
        std::vector<double> beamEnergyForBeamDivergence = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        std::vector<double> correctionBeamDivergence = { 0.0871, 0.0736, 0.0606, 0.0515, 0.0438, 0.0389, 0.0347, 0.0311, 0.027, 0.0236, 0.0206, 0.0193, 0.018, 0.0164, 0.0154, 0.0144, 0.0124, 0.0116, 0.0109, 0.0104, 0.00949, 0.00892, 0.00799, 0.00766, 0.00739, 0.00671, 0.0065, 0.00638, 0.0066, 0.00697 }; 
        beamDivergenceInterp.set_points(beamEnergyForBeamDivergence, correctionBeamDivergence, tk::spline::cspline);
        beamDivergenceInterp.make_monotonic();
    }

    ~gtr1(){;}

    /// User method to characterize MODULATED beamlet based on spot information from DICOM.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) 
    {
        return mqi::beamlet<T>();
    }

    /// User method to characterize MODULATED beamlet based on spot information ans source to isocenter distnace from DICOM.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm)
    {
        return mqi::beamlet<T>();
    }

    /// User method to characterize UNIFORM/MODULATED_SPEC beamlet based on spot information from DICOM.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1)
    {
        return mqi::beamlet<T>();
    }

    // Log file MU count to particle count conversion formula used in TOPAS MC
    // Particle count in MC = MU Count * (Dose / MU Count) * (Particle / dose) * (Dose monitor range--> implemented in pre-program) 
    size_t
    characterize_history(
        const mqi::beam_module_ion::logspot& s)
    {
        /* SMC Log file version */
        //std::cout << "Beam energy : " << s.e << std::endl;
        //std::cout << "Proton / dose interp : " << protonPerDoseInterp(s.e) << std::endl;
        //std::cout << "Dose Per MU count interp : " << dosePerMUCountInterp(s.e) << std::endl;
        int particleFromMUCount = s.muCount * particleCountCalibInterp(s.e); // * protonPerDoseInterp(s.e) * dosePerMUCountInterp(s.e);

        return particleFromMUCount;
    }

    // Characterize beamlet information using log file information
    // Using mqi::beam_module_ion::logspot, with an added feature in mqi_beam_module_ion
    // Added by Chanil Jeon
    mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::logspot& s,
        const float                       source_to_isocenter_mm,
        const bool rsuse)
    {
        // Range shifter correction
        float newBeamStartingPos{ -source_to_isocenter_mm };
        if (rsuse) newBeamStartingPos = newBeamStartingPos - 50;

        // Spot beam's energy
        // Constant energy 
        double energySpread = this->beamEnergySpreadInterp(s.e);

        // Gaussian energy spread distribution
        auto energy = new mqi::norm_1d<T>({ s.e }, { energySpread });

        // Caculate direction based on SAD and spot's position
        mqi::vec3<T> dir(std::atan(s.x/treatment_machine_ion<T>::SAD_[0]),
                         std::atan(s.y/treatment_machine_ion<T>::SAD_[1]),
                         -1.0);

        // Determine X,Y position of isocenter
        mqi::vec3<T> pos(0, 0, -newBeamStartingPos);
        pos.x = (treatment_machine_ion<T>::SAD_[0] - pos.z) * dir.x ;
        pos.y = (treatment_machine_ion<T>::SAD_[1] - pos.z) * dir.y ;

        // Spot size interpolation equation (70-230 MeV)
        double spotSize = this->beamSpotSizeInterp(s.e);

        // Angular spread interpolation equation (70-230 MeV)
        double angularSpread = this->beamAngularSpreadInterp(s.e);

        // Divergence interpolation equation (70-230 MeV)
        double divergence = this->beamDivergenceInterp(s.e);

        //Define phsp distribution
        std::array<T,6> beamlet_mean = { pos.x, pos.y, pos.z, dir.x, dir.y, dir.z };
        std::array<T,6> beamlet_sigm = { spotSize , spotSize, 0, angularSpread, angularSpread, 0};
        std::array<T,2> beamlet_divergence = { divergence, divergence };
        auto beamlet = new mqi::phsp_6d_ray<T>(beamlet_mean, beamlet_sigm, beamlet_divergence, newBeamStartingPos);

        return mqi::beamlet<T>(energy, beamlet);
    }

    mqi::rangeshifter*
    characterize_rangeshifter(
        const mqi::dataset* ds,
        mqi::modality_type m)
    {
        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        //1. rangeshifter sequence
        //For MDACC, this sequence doesn't have much information.
        auto  rs_ds = (*ds)( seq_tags->at("rs")) ;
        assert(rs_ds.size() >=1);

       //2. Snout position from control point 0
        std::vector<float> ftmp;
        auto layer0    = (*ds)(seq_tags->at("ctrl"))[0]; //layer0 for snout position
        layer0->get_values( "SnoutPosition", ftmp);

        mqi::vec3<float>    lxyz(400.0, 400.0, 0.0);
        mqi::vec3<float>    pxyz(0.0, 0.0, ftmp[0]);
        mqi::mat3x3<float>  rxyz(0.0, 0.0, 0.0);

        //3. There must be at least one range shifter sequence
        for(auto r : rs_ds)
        {
            std::vector<std::string> rs_id(0);
            r->get_values("RangeShifterID", rs_id);

            std::cout<< "RangeShifterID detected.. : " << rs_id[0] << std::endl;
            rs_id[0].erase(std::remove(rs_id[0].begin(), rs_id[0].end(), ' '), rs_id[0].end()); // Erase blank

            /* Changed, because G2 rangeshifter is fixed to name of SNOUT_DEG_B  */
            if (rs_id[0] == "SNOUT_DEG_B" || rs_id[0] == "SNOUT_DEG_S") lxyz.z = 40;
            else lxyz.z = 0.0;
            assert(lxyz.z > 0);
        }

        pxyz.z += lxyz.z / 2;
        std::cout << "Range shifter thickness determined.. : " << lxyz.z <<" (mm) and position: " << pxyz.z <<" (mm)" << std::endl;
        pxyz.dump();

        return new mqi::rangeshifter(lxyz, pxyz, rxyz);
    }

    mqi::aperture*
    characterize_aperture(
        const mqi::dataset* ds,
        mqi::modality_type m
    ){
        auto xypts = this->characterize_aperture_opening(ds,m);

        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        //1. block sequence
        auto  blk_ds = (*ds)( seq_tags->at("blk")) ;
        assert(blk_ds.size() >=1);

        std::vector<float> ftmp;

        //note:
        // 1. Assumed opening points are physical points of geometry.
        //    (there must be tag for this) => this should be taken cared by users
        //apt_lunder -> set_dimension(150.0, 67.5);
        //400 is temporal
        blk_ds[0]->get_values( "BlockThickness", ftmp);
        mqi::vec3<float> lxyz(400.0, 400.0, ftmp[0]);

        //I omitted reading patient_side or snout_side
        blk_ds[0]->get_values("IsocenterToBlockTrayDistance", ftmp);
        mqi::vec3<float> pxyz(0.0, 0.0, ftmp[0]);

        mqi::mat3x3<float> rxyz(0.0, 0.0, 0.0);

        return new mqi::aperture(xypts, lxyz, pxyz, rxyz);
    }

    // Returns beamsource model from csv file having log file information
    // Log file based implementation for Samsung Medical Center by Chanil Jeon
    mqi::beamsource<T>
    create_beamsource(const mqi::logfiles_t& logfileData,
                      const mqi::coordinate_transform<T> pcoord,
                      const float source_to_isocenter_mm = 465.0,
                      const bool rsuse = false) 
    {
        // Definition of source to isocenter distance
        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;

        // Creating beam source with log file information
        mqi::beamsource<T> beamsource;

        for (int i = 0; i < logfileData.beamInfo.size(); i++)
        {
            for (int j = 0; j < logfileData.beamInfo[i].size(); j++)
            {
                mqi::beam_module_ion::logspot logSpotInfo;
                logSpotInfo.e = logfileData.beamEnergyInfo[i][j]; // Log file spot energy (single energy layer)

                // In single energy layer information
                for (int k = 0; k < logfileData.beamInfo[i][j].muCount.size(); k++)
                {
                    logSpotInfo.muCount = logfileData.beamInfo[i][j].muCount[k];
                    logSpotInfo.x = logfileData.beamInfo[i][j].posX[k];
                    logSpotInfo.y = logfileData.beamInfo[i][j].posY[k];
                    beamsource.append_beamlet_log(this->characterize_beamlet(logSpotInfo, treatment_machine<T>::source_to_isocenter_mm_, rsuse), this->characterize_history(logSpotInfo), pcoord);
                }
            }
        }
        return beamsource;
    }
};

}
//}
#endif
