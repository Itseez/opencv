// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include <atomic>

namespace cv {
UsacParams::UsacParams() {
    confidence=0.99;
    isParallel=false;
    loIterations=5;
    loMethod=LOCAL_OPTIM_INNER_LO;
    loSampleSize=14;
    maxIterations=5000;
    neighborsSearch=NEIGH_GRID;
    randomGeneratorState=0;
    sampler=SAMPLING_UNIFORM;
    score=SCORE_METHOD_MSAC;
    threshold=1.5;
    final_polisher=COV_POLISHER;
    final_polisher_iterations=3;
}

namespace usac {
int mergePoints (InputArray pts1_, InputArray pts2_, Mat &pts, EstimationMethod estimator);
void setParameters (int flag, Ptr<Model> &params, EstimationMethod estimator, double thr,
                    int max_iters, double conf, bool mask_needed);

class RansacOutputImpl : public RansacOutput {
private:
    std::vector<int> inliers;
    cv::Mat model, K1, K2;
    // vector of number_inliers size
    // vector of points size, true if inlier, false - outlier
    std::vector<bool> inliers_mask;
    // vector of points size, value of i-th index corresponds to error of i-th point if i is inlier.
    std::vector<float> residuals;
    int number_inliers, number_iterations;
    ModelConfidence conf;
public:
    RansacOutputImpl (const cv::Mat &model_, const std::vector<bool> &inliers_mask_, int number_inliers_,
            int number_iterations_, ModelConfidence conf_, const std::vector<float> &errors_) {
        model_.copyTo(model);
        inliers_mask = inliers_mask_;
        number_inliers = number_inliers_;
        number_iterations = number_iterations_;
        residuals = errors_;
        conf = conf_;
    }

    // Return inliers' indices of size  = number of inliers
    const std::vector<int> &getInliers() override {
        if (inliers.empty()) {
            inliers.reserve(number_inliers);
            int pt_cnt = 0;
            for (bool is_inlier : inliers_mask) {
                if (is_inlier)
                    inliers.emplace_back(pt_cnt);
                pt_cnt++;
            }
        }
        return inliers;
    }
    const std::vector<bool> &getInliersMask() const override {
        return inliers_mask;
    }
    int getNumberOfInliers() const override {
        return number_inliers;
    }
    const Mat &getModel() const override {
        return model;
    }
    int getNumberOfIters() const override {
        return number_iterations;
    }
    ModelConfidence getConfidence() const override {
        return conf;
    }
    const std::vector<float> &getResiduals() const override {
        return residuals;
    }
};

Ptr<RansacOutput> RansacOutput::create(const cv::Mat &model_, const std::vector<bool> &inliers_mask_, int number_inliers_,
            int number_iterations_, ModelConfidence conf, const std::vector<float> &errors_) {
    return makePtr<RansacOutputImpl>(model_, inliers_mask_, number_inliers_,
            number_iterations_, conf, errors_);
}

double getLambda (std::vector<int> &supports, double cdf_thr, int points_size,
        int sample_size, bool is_independent, int &min_non_random_inliers) {
    std::sort(supports.begin(), supports.end());
    double lambda = supports.size() % 2 ? (supports[supports.size()/2] + supports[supports.size()/2+1])*0.5 : supports[supports.size()/2];
    const double cdf = lambda + cdf_thr*sqrt(lambda * (1 - lambda / (is_independent ? points_size - sample_size : points_size)));
    int lower_than_cdf = 0; lambda = 0;
    for (const auto &inl : supports)
        if (inl < cdf) {
            lambda += inl; lower_than_cdf++;
        } else break; // list is sorted
    lambda /= lower_than_cdf;
    if (lambda < 1 || lower_than_cdf == 0) lambda = 1;
    // use 0.9999 quantile https://keisan.casio.com/exec/system/14060745333941
    if (! is_independent) // do not calculate it for all inliers
        min_non_random_inliers = (int)(lambda + 2.32*sqrt(lambda * (1 - lambda / points_size))) + 1;
    return lambda;
}

class Ransac {
public:
    const Ptr<const Model> params;
    Ptr<Estimator> _estimator;
    Ptr<Error> _error;
    Ptr<Quality> _quality;
    Ptr<Sampler> _sampler;
    Ptr<Termination> _termination;
    Ptr<ModelVerifier> _model_verifier;
    Ptr<Degeneracy> _degeneracy;
    Ptr<LocalOptimization> _local_optimization;
    Ptr<FinalModelPolisher> polisher;
    Ptr<GammaValues> _gamma_generator;
    Ptr<MinimalSolver> _min_solver;
    Ptr<NonMinimalSolver> _lo_solver, _fo_solver;
    Ptr<RandomGenerator> _lo_sampler;
    Ptr<WeightFunction> _weight_fnc;

    int points_size, _state, filtered_points_size;
    double threshold, max_thr;
    bool parallel;

    Matx33d T1, T2;
    Mat points, K1, K2, calib_points, image_points, norm_points, filtered_points;
    Ptr<NeighborhoodGraph> graph;
    std::vector<Ptr<NeighborhoodGraph>> layers;

    Ransac (const Ptr<const Model> &params_, cv::InputArray points1, cv::InputArray points2,
            cv::InputArray K1_, cv::InputArray K2_, cv::InputArray dist_coeff1, cv::InputArray dist_coeff2) : params(params_) {
        _state = params->getRandomGeneratorState();
        threshold = params->getThreshold();
        max_thr = std::max(threshold, params->getMaximumThreshold());
        parallel = params->isParallel();
        Mat undist_points1, undist_points2;
        if (params->isPnP()) {
            if (! K1_.empty()) {
                K1 = K1_.getMat().clone(); K1.convertTo(K1, CV_64F);
                if (! dist_coeff1.empty()) {
                    // undistortPoints also calibrate points using K
                    undistortPoints(points1.isContinuous() ? points1 : points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
                    points_size = mergePoints(undist_points1, points2, points, params->getEstimator());
                    Utils::normalizeAndDecalibPointsPnP (K1, points, calib_points);
                } else {
                    points_size = mergePoints(points1, points2, points, params->getEstimator());
                    Utils::calibrateAndNormalizePointsPnP(K1, points, calib_points);
                }
            } else points_size = mergePoints(points1, points2, points, params->getEstimator());
        } else {
            if (params->isEssential()) {
                CV_CheckEQ((int)(!K1_.empty() && !K2_.empty()), 1, "Intrinsic matrix must not be empty!");
                K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
                K2 = K2_.getMat(); K2.convertTo(K2, CV_64F);
                if (! dist_coeff1.empty() || ! dist_coeff2.empty()) {
                    // undistortPoints also calibrate points using K
                    if (! dist_coeff1.empty()) undistortPoints(points1.isContinuous() ? points1 : points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
                    else undist_points1 = points1.getMat();
                    if (! dist_coeff2.empty()) undistortPoints(points2.isContinuous() ? points2 : points2.getMat().clone(), undist_points2, K2_, dist_coeff2);
                    else undist_points2 = points2.getMat();
                    points_size = mergePoints(undist_points1, undist_points2, calib_points, params->getEstimator());
                } else {
                    points_size = mergePoints(points1, points2, points, params->getEstimator());
                    Utils::calibratePoints(K1, K2, points, calib_points);
                }
                threshold = Utils::getCalibratedThreshold(threshold, K1, K2);
                max_thr = Utils::getCalibratedThreshold(max_thr, K1, K2);
            } else {
                points_size = mergePoints(points1, points2, points, params->getEstimator());
                if (params->isFundamental() && ! K1_.empty() && ! K2_.empty()) {
                    K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
                    K2 = K2_.getMat(); K2.convertTo(K2, CV_64F);
                    Utils::calibratePoints(K1, K2, points, calib_points);
                }
            }
        }

        if (params->getSampler() == SamplingMethod::SAMPLING_NAPSAC || params->getLO() == LocalOptimMethod::LOCAL_OPTIM_GC) {
            if (params->getNeighborsSearch() == NeighborSearchMethod::NEIGH_GRID) {
                graph = GridNeighborhoodGraph::create(points, points_size,
                        params->getCellSize(), params->getCellSize(), params->getCellSize(), params->getCellSize(), 10);
            } else if (params->getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_KNN) {
                graph = FlannNeighborhoodGraph::create(points, points_size,params->getKNN(), false, 5, 1);
            } else if (params->getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_RADIUS) {
                graph = RadiusSearchNeighborhoodGraph::create(points, points_size, params->getGraphRadius(), 5, 1);
            } else CV_Error(cv::Error::StsNotImplemented, "Graph type is not implemented!");
        }

        if (params->getSampler() == SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
            CV_CheckEQ((int)params->isPnP(), 0, "ProgressiveNAPSAC for PnP is not implemented!");
            const auto &cell_number_per_layer = params->getGridCellNumber();
            layers.reserve(cell_number_per_layer.size());
            const auto * const pts = (float *) points.data;
            float img1_width = 0, img1_height = 0, img2_width = 0, img2_height = 0;
            for (int i = 0; i < 4 * points_size; i += 4) {
                if (pts[i    ] > img1_width ) img1_width  = pts[i    ];
                if (pts[i + 1] > img1_height) img1_height = pts[i + 1];
                if (pts[i + 2] > img2_width ) img2_width  = pts[i + 2];
                if (pts[i + 3] > img2_height) img2_height = pts[i + 3];
            }
            // Create grid graphs (overlapping layes of given cell numbers)
            for (int layer_idx = 0; layer_idx < (int)cell_number_per_layer.size(); layer_idx++) {
                const int cell_number = cell_number_per_layer[layer_idx];
                if (layer_idx > 0)
                    if (cell_number_per_layer[layer_idx-1] <= cell_number)
                        CV_Error(cv::Error::StsError, "Progressive NAPSAC sampler: "
                                                      "Cell number in layers must be in decreasing order!");
                layers.emplace_back(GridNeighborhoodGraph::create(points, points_size,
            (int)(img1_width / (float)cell_number), (int)(img1_height / (float)cell_number),
            (int)(img2_width / (float)cell_number), (int)(img2_height / (float)cell_number), 10));
            }
        }

        // update points by calibrated for Essential matrix after graph is calculated
        if (params->isEssential()) {
            image_points = points;
            points = calib_points;
            // if maximum calibrated threshold significanlty differs threshold then set upper bound
            if (max_thr > 10*threshold)
                max_thr = 10*threshold;
        }

        // Since error function output squared error distance, so make
        // threshold squared as well
        threshold *= threshold;

        if ((params->isHomography() || (params->isFundamental() && (K1.empty() || K2.empty() || !params->isLarssonOptimization())) || params->isPtsetReg()) && (params->getLO() != LOCAL_OPTIM_NULL || params->getFinalPolisher() == COV_POLISHER)) {
            const auto normTr = NormTransform::create(points);
            std::vector<int> sample (points_size);
            for (int i = 0; i < points_size; i++) sample[i] = i;
                normTr->getNormTransformation(norm_points, sample, points_size, T1, T2);
        }

        if (params->getScore() == SCORE_METHOD_MAGSAC || params->getLO() == LOCAL_OPTIM_SIGMA || params->getFinalPolisher() == MAGSAC)
            _gamma_generator = GammaValues::create(params->getDegreesOfFreedom()); // is thread safe
        initialize (_state, _min_solver, _lo_solver, _error, _estimator, _degeneracy, _quality,
                _model_verifier, _local_optimization, _termination, _sampler, _lo_sampler, _weight_fnc, false/*parallel*/);
        if (params->getFinalPolisher() != NONE_POLISHER) {
            polisher = NonMinimalPolisher::create(_quality, _fo_solver,
                params->getFinalPolisher() == MAGSAC ? _weight_fnc : nullptr, params->getFinalLSQIterations(), 0.99);
        }
    }

    void initialize (int state, Ptr<MinimalSolver> &min_solver, Ptr<NonMinimalSolver> &non_min_solver,
            Ptr<Error> &error, Ptr<Estimator> &estimator, Ptr<Degeneracy> &degeneracy, Ptr<Quality> &quality,
            Ptr<ModelVerifier> &verifier, Ptr<LocalOptimization> &lo, Ptr<Termination> &termination,
            Ptr<Sampler> &sampler, Ptr<RandomGenerator> &lo_sampler, Ptr<WeightFunction> &weight_fnc, bool parallel_call) {

        const int min_sample_size = params->getSampleSize(), prosac_termination_length = std::min((int)(.5*points_size), 100);
        // inner inlier threshold will be used in LO to obtain inliers
        // additionally in DEGENSAC for F
        double inner_inlier_thr_sqr = threshold;
        if (params->isHomography() && inner_inlier_thr_sqr < 5.25) inner_inlier_thr_sqr = 5.25; // at least 2.5 px
        else if (params->isFundamental() && inner_inlier_thr_sqr < 4) inner_inlier_thr_sqr = 4; // at least 2 px

        if (params->getFinalPolisher() == MAGSAC || params->getLO() == LOCAL_OPTIM_SIGMA)
            weight_fnc = MagsacWeightFunction::create(_gamma_generator, params->getDegreesOfFreedom(), params->getUpperIncompleteOfSigmaQuantile(), params->getC(), params->getMaximumThreshold());
        else weight_fnc = nullptr;

        switch (params->getError()) {
            case ErrorMetric::SYMM_REPR_ERR:
                error = ReprojectionErrorSymmetric::create(points); break;
            case ErrorMetric::FORW_REPR_ERR:
                switch(params->getEstimator()){
                    case EstimationMethod::AFFINE:
                    case EstimationMethod::SE2:
                    case EstimationMethod::SIM2:
                        error = ReprojectionErrorAffine::create(points);
                        break;
                    case EstimationMethod::SO3:
                    case EstimationMethod::SE3:
                    case EstimationMethod::SIM3:
                        error = ReprojectionErrorAffine3D::create(points);
                        break;
                    default:
                        error = ReprojectionErrorForward::create(points);
                }
                break;
            case ErrorMetric::SAMPSON_ERR:
                error = SampsonError::create(points); break;
            case ErrorMetric::SGD_ERR:
                error = SymmetricGeometricDistance::create(points); break;
            case ErrorMetric::RERPOJ:
                error = ReprojectionErrorPmatrix::create(points); break;
            default: CV_Error(cv::Error::StsNotImplemented , "Error metric is not implemented!");
        }

        const double k_mlesac = params->getKmlesac ();
        switch (params->getScore()) {
            case ScoreMethod::SCORE_METHOD_RANSAC :
                quality = RansacQuality::create(points_size, threshold, error); break;
            case ScoreMethod::SCORE_METHOD_MSAC :
                quality = MsacQuality::create(points_size, threshold, error, k_mlesac); break;
            case ScoreMethod::SCORE_METHOD_MAGSAC :
                quality = MagsacQuality::create(max_thr, points_size, error, _gamma_generator,
                    threshold, params->getDegreesOfFreedom(),  params->getSigmaQuantile(),
                    params->getUpperIncompleteOfSigmaQuantile()); break;
            case ScoreMethod::SCORE_METHOD_LMEDS :
                quality = LMedsQuality::create(points_size, threshold, error); break;
            default: CV_Error(cv::Error::StsNotImplemented, "Score is not implemented!");
        }

        const auto is_ge_solver = params->getRansacSolver() == GEM_SOLVER;
        if (params->isHomography()) {
            degeneracy = HomographyDegeneracy::create(points);
            min_solver = HomographyMinimalSolver4pts::create(points, is_ge_solver);
            non_min_solver = HomographyNonMinimalSolver::create(norm_points, T1, T2, true);
            estimator = HomographyEstimator::create(min_solver, non_min_solver, degeneracy);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) {
                if (params->getFinalPolisher() == COV_POLISHER)
                     _fo_solver = CovarianceHomographySolver::create(norm_points, T1, T2);
                else _fo_solver = HomographyNonMinimalSolver::create(points);
            }
        } else if (params->isFundamental()) {
            if (K1.empty() || K2.empty()) {
                degeneracy = FundamentalDegeneracy::create(state++, quality, points, min_sample_size,
                   params->getPlaneAndParallaxIters(), std::max(threshold, 8.) /*sqr homogr thr*/, inner_inlier_thr_sqr, K1, K2);
            } else degeneracy = FundamentalDegeneracyViaE::create(quality, points, calib_points, K1, K2, true/*is F*/);
            if (min_sample_size == 7) {
                min_solver = FundamentalMinimalSolver7pts::create(points, is_ge_solver);
            } else min_solver = FundamentalMinimalSolver8pts::create(points);
            if (params->isLarssonOptimization() && !K1.empty() && !K2.empty()) {
                non_min_solver = LarssonOptimizer::create(calib_points, K1, K2, params->getLevMarqItersLO(), true/*F*/);
            } else {
                if (weight_fnc)
                    non_min_solver = EpipolarNonMinimalSolver::create(points, true);
                else
                    non_min_solver = EpipolarNonMinimalSolver::create(norm_points, T1, T2, true);
            }
            estimator = FundamentalEstimator::create(min_solver, non_min_solver, degeneracy);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) {
                if (params->isLarssonOptimization() && !K1.empty() && !K2.empty())
                     _fo_solver = LarssonOptimizer::create(calib_points, K1, K2, params->getLevMarqIters(), true/*F*/);
                else if (params->getFinalPolisher() == COV_POLISHER)
                     _fo_solver = CovarianceEpipolarSolver::create(norm_points, T1, T2);
                else _fo_solver = EpipolarNonMinimalSolver::create(points, true);
            }
        } else if (params->isEssential()) {
            if (params->getEstimator() == EstimationMethod::ESSENTIAL) {
                min_solver = EssentialMinimalSolver5pts::create(points, !is_ge_solver, true/*Nister*/);
                degeneracy = EssentialDegeneracy::create(points, min_sample_size);
            }
            non_min_solver = LarssonOptimizer::create(calib_points, K1, K2, params->getLevMarqItersLO(), false/*E*/);
            estimator = EssentialEstimator::create(min_solver, non_min_solver, degeneracy);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER)
                _fo_solver = LarssonOptimizer::create(calib_points, K1, K2, params->getLevMarqIters(), false/*E*/);
        } else if (params->isPnP()) {
            degeneracy = makePtr<Degeneracy>();
            if (min_sample_size == 3) {
                min_solver = P3PSolver::create(points, calib_points, K1);
                non_min_solver = DLSPnP::create(points, calib_points, K1);
            } else {
                if (is_ge_solver)
                    min_solver = PnPMinimalSolver6Pts::create(points);
                else min_solver = PnPSVDSolver::create(points);
                non_min_solver = PnPNonMinimalSolver::create(points);
            }
            estimator = PnPEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) _fo_solver = non_min_solver;
        } else if (params->getEstimator() == EstimationMethod::AFFINE) {
            degeneracy = makePtr<Degeneracy>();
            min_solver = AffineMinimalSolver::create(points);
            non_min_solver = AffineNonMinimalSolver::create(points, cv::noArray(), cv::noArray());
            estimator = AffineEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) {
                if (params->getFinalPolisher() == COV_POLISHER)
                    _fo_solver = CovarianceAffineSolver::create(points);
                else _fo_solver = non_min_solver;
            }
        } else if (params->getEstimator() == EstimationMethod::SE2) {
            degeneracy = makePtr<Degeneracy>();
            min_solver = SE2MinimalSolver::create(points);
            non_min_solver = SE2NonMinimalSolver::create(points);
            estimator = AffineEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) _fo_solver = non_min_solver;
        } else if (params->getEstimator() == EstimationMethod::SIM2) {
            degeneracy = makePtr<Degeneracy>();
            min_solver = SIM2MinimalSolver::create(points);
            non_min_solver = SIM2NonMinimalSolver::create(points);
            estimator = AffineEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) _fo_solver = non_min_solver;
        } else if (params->getEstimator() == EstimationMethod::SO3) {
            degeneracy = makePtr<Degeneracy>();
            min_solver = SO3MinimalSolver::create(points);
            non_min_solver = SO3NonMinimalSolver::create(points);
            estimator = AffineEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) _fo_solver = non_min_solver;
        } else if (params->getEstimator() == EstimationMethod::SE3) {
            degeneracy = makePtr<Degeneracy>();
            min_solver = SE3MinimalSolver::create(points);
            non_min_solver = SE3NonMinimalSolver::create(points);
            estimator = AffineEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) _fo_solver = non_min_solver;
        } else if (params->getEstimator() == EstimationMethod::SIM3) {
            degeneracy = makePtr<Degeneracy>();
            min_solver = SIM3MinimalSolver::create(points);
            non_min_solver = SIM3NonMinimalSolver::create(points);
            estimator = AffineEstimator::create(min_solver, non_min_solver);
            if (!parallel_call && params->getFinalPolisher() != NONE_POLISHER) _fo_solver = non_min_solver;
        } else CV_Error(cv::Error::StsNotImplemented, "Estimator not implemented!");

        switch (params->getSampler()) {
            case SamplingMethod::SAMPLING_UNIFORM:
                sampler = UniformSampler::create(state++, min_sample_size, points_size);
                break;
            case SamplingMethod::SAMPLING_PROSAC:
                if (!parallel_call) // for parallel only one PROSAC sampler
                    sampler = ProsacSampler::create(state++, points_size, min_sample_size, params->getProsacMaxSamples());
                break;
            case SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC:
                sampler = ProgressiveNapsac::create(state++, points_size, min_sample_size, layers, 20); break;
            case SamplingMethod::SAMPLING_NAPSAC:
                sampler = NapsacSampler::create(state++, points_size, min_sample_size, graph); break;
            default: CV_Error(cv::Error::StsNotImplemented, "Sampler is not implemented!");
        }

        const bool is_sprt = params->getVerifier() == VerificationMethod::SPRT_VERIFIER || params->getVerifier() == VerificationMethod::ASPRT;
        if (is_sprt)
            verifier = AdaptiveSPRT::create(state++, quality, points_size, params->getScore() == ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
                params->getSPRTepsilon(), params->getSPRTdelta(), params->getTimeForModelEstimation(),
                params->getSPRTavgNumModels(), params->getScore(), k_mlesac, params->getVerifier() == VerificationMethod::ASPRT);
        else if (params->getVerifier() == VerificationMethod::NULL_VERIFIER)
            verifier = ModelVerifier::create(quality);
        else CV_Error(cv::Error::StsNotImplemented, "Verifier is not imeplemented!");

        if (params->getSampler() == SamplingMethod::SAMPLING_PROSAC) {
            if (parallel_call) {
                termination = ProsacTerminationCriteria::create(nullptr, error,
                    points_size, min_sample_size, params->getConfidence(), params->getMaxIters(), prosac_termination_length, 0.05, 0.05, threshold,
                    _termination.dynamicCast<ProsacTerminationCriteria>()->getNonRandomInliers());
            } else {
                termination = ProsacTerminationCriteria::create(sampler.dynamicCast<ProsacSampler>(), error,
                    points_size, min_sample_size, params->getConfidence(), params->getMaxIters(), prosac_termination_length, 0.05, 0.05, threshold,
                    std::vector<int>());
            }
        } else if (params->getSampler() == SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
            if (is_sprt)
                 termination = SPRTPNapsacTermination::create(verifier.dynamicCast<AdaptiveSPRT>(),
                        params->getConfidence(), points_size, min_sample_size,
                        params->getMaxIters(), params->getRelaxCoef());
            else termination = StandardTerminationCriteria::create (params->getConfidence(),
                    points_size, min_sample_size, params->getMaxIters());
        } else if (is_sprt && params->getLO() == LocalOptimMethod::LOCAL_OPTIM_NULL) {
            termination = SPRTTermination::create(verifier.dynamicCast<AdaptiveSPRT>(),
                 params->getConfidence(), points_size, min_sample_size, params->getMaxIters());
        } else {
            termination = StandardTerminationCriteria::create
              (params->getConfidence(), points_size, min_sample_size, params->getMaxIters());
        }

        // if normal ransac or parallel call, avoid redundant init
        if ((! params->isParallel() || parallel_call) && params->getLO() != LocalOptimMethod::LOCAL_OPTIM_NULL) {
            lo_sampler = UniformRandomGenerator::create(state, points_size, params->getLOSampleSize());
            const auto lo_termination = StandardTerminationCriteria::create(params->getConfidence(), points_size, min_sample_size, params->getMaxIters());
            switch (params->getLO()) {
                case LocalOptimMethod::LOCAL_OPTIM_INNER_LO: case LocalOptimMethod::LOCAL_OPTIM_SIGMA:
                    lo = SimpleLocalOptimization::create(quality, non_min_solver, lo_termination, lo_sampler,
                         weight_fnc, params->getLOInnerMaxIters(), inner_inlier_thr_sqr, true); break;
                case LocalOptimMethod::LOCAL_OPTIM_INNER_AND_ITER_LO:
                    lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler,
                         points_size, threshold, true, params->getLOIterativeSampleSize(),
                         params->getLOInnerMaxIters(), params->getLOIterativeMaxIters(),
                         params->getLOThresholdMultiplier()); break;
                case LocalOptimMethod::LOCAL_OPTIM_GC:
                    lo = GraphCut::create(estimator, quality, graph, lo_sampler, threshold,
                       params->getGraphCutSpatialCoherenceTerm(), params->getLOInnerMaxIters(), lo_termination); break;
                default: CV_Error(cv::Error::StsNotImplemented , "Local Optimization is not implemented!");
            }
        }
    }

    int getIndependentInliers (const Mat &model_, const std::vector<int> &sample,
                                     std::vector<int> &inliers, const int num_inliers_) {
        bool is_F = params->isFundamental();
        Mat model = model_;
        int sample_size = 0;
        if (is_F) sample_size = 7;
        else if (params->isHomography()) sample_size = 4;
        else if (params->isEssential()) {
            is_F = true;
            // convert E to F
            model = Mat(Matx33d(K2).inv().t() * Matx33d(model) * Matx33d(K1).inv());
            sample_size = 5;
        } else if (params->isPnP() || params->isPtsetReg()) {
            if (params->getEstimator() == EstimationMethod::SE2 || params->getEstimator() == EstimationMethod::SIM2)
                sample_size = 2;
            else sample_size = 3;
        } else
            CV_Error(cv::Error::StsNotImplemented, "Method for independent inliers is not implemented for this problem");
        if (num_inliers_ <= sample_size) return 0; // minimal sample size generates model
        model.convertTo(model, CV_32F);
        int num_inliers = num_inliers_, num_pts_near_ep = 0,
                num_pts_validatin_or_constr = 0, pt1 = 0;
        const auto * const pts = params->isEssential() ? (float *) image_points.data : (float *) points.data;
        // scale for thresholds should be used
        const float ep_thr_sqr = 0.000001f, line_thr = 0.01f, neigh_thr = 4.0f;
        float sign1=0,a1=0, b1=0, c1=0, a2=0, b2=0, c2=0, ep1_x, ep1_y, ep2_x, ep2_y;
        const auto * const m = (float *) model.data;
        Vec3f ep1;
        bool do_or_test = false, ep1_inf = false, ep2_inf = false;
        if (is_F) { // compute epipole and sign of the first point for orientation test
            model *= (1/norm(model));
            ep1 = Utils::getRightEpipole(model);
            const Vec3f ep2 = Utils::getLeftEpipole(model);
            if (fabsf(ep1[2]) < DBL_EPSILON) {
                ep1_inf = true;
            } else {
                ep1_x = ep1[0] / ep1[2];
                ep1_y = ep1[1] / ep1[2];
            }
            if (fabsf(ep2[2]) < DBL_EPSILON) {
                ep2_inf = true;
            } else {
                ep2_x = ep2[0] / ep2[2];
                ep2_y = ep2[1] / ep2[2];
            }
        }
        const auto * const e1 = ep1.val; // of size 3x1

        // we move sample points to the end, so every inlier will be checked by sample point
        int num_sample_in_inliers = 0;
        if (!sample.empty()) {
            num_sample_in_inliers = 0;
            int temp_idx = num_inliers;
            for (int i = 0; i < temp_idx; i++) {
                const int inl = inliers[i];
                for (int s : sample) {
                    if (inl == s) {
                        std::swap(inliers[i], inliers[--temp_idx]);
                        i--; // we need to check inlier that we just swapped
                        num_sample_in_inliers++;
                        break;
                    }
                }
            }
        }

        if (is_F) {
            int MIN_TEST = std::min(15, num_inliers);
            for (int i = 0; i < MIN_TEST; i++) {
                pt1 = 4*inliers[i];
                sign1 = (m[0]*pts[pt1+2]+m[3]*pts[pt1+3]+m[6])*(e1[1]-e1[2]*pts[pt1+1]);
                int validate = 0;
                for (int j = 0; j < MIN_TEST; j++) {
                    if (i == j) continue;
                    const int inl_idx = 4*inliers[j];
                    if (sign1*(m[0]*pts[inl_idx+2]+m[3]*pts[inl_idx+3]+m[6])*(e1[1]-e1[2]*pts[inl_idx+1])<0)
                        validate++;
                }
                if (validate < MIN_TEST/2) {
                    do_or_test = true; break;
                }
            }
        }

        // verification does not include sample points as they are surely random
        const int max_verify = num_inliers - num_sample_in_inliers;
        if (max_verify <= 0)
            return 0;
        int num_non_random_inliers = num_inliers - sample_size;
        auto removeDependentPoints = [&] (bool do_orient_test, bool check_epipoles) {
            for (int i = 0; i < max_verify; i++) {
                // checks over inliers if they are dependent to other inliers
                const int inl_idx = 4*inliers[i];
                const auto x1 = pts[inl_idx], y1 = pts[inl_idx+1], x2 = pts[inl_idx+2], y2 = pts[inl_idx+3];
                if (is_F) {
                    // epipolar line on image 2 = l2
                    a2 = m[0] * x1 + m[1] * y1 + m[2];
                    b2 = m[3] * x1 + m[4] * y1 + m[5];
                    c2 = m[6] * x1 + m[7] * y1 + m[8];
                    // epipolar line on image 1 = l1
                    a1 = m[0] * x2 + m[3] * y2 + m[6];
                    b1 = m[1] * x2 + m[4] * y2 + m[7];
                    c1 = m[2] * x2 + m[5] * y2 + m[8];
                    if ((!ep1_inf && fabsf(x1-ep1_x)+fabsf(y1-ep1_y) < neigh_thr) ||
                        (!ep2_inf && fabsf(x2-ep2_x)+fabsf(y2-ep2_y) < neigh_thr)) {
                        num_non_random_inliers--;
                        num_pts_near_ep++;
                        continue; // is dependent, continue to the next point
                    } else if (check_epipoles) {
                        if (a2 * a2 + b2 * b2 + c2 * c2 < ep_thr_sqr ||
                            a1 * a1 + b1 * b1 + c1 * c1 < ep_thr_sqr) {
                            num_non_random_inliers--;
                            num_pts_near_ep++;
                            continue; // is dependent, continue to the next point
                        }
                    }
                    else if (do_orient_test && pt1 != inl_idx && sign1*(m[0]*x2+m[3]*y2+m[6])*(e1[1]-e1[2]*y1)<0) {
                        num_non_random_inliers--;
                        num_pts_validatin_or_constr++;
                        continue;
                    }
                    const auto mag2 = 1 / sqrt(a2 * a2 + b2 * b2), mag1 = 1/sqrt(a1 * a1 + b1 * b1);
                    a2 *= mag2; b2 *= mag2; c2 *= mag2;
                    a1 *= mag1; b1 *= mag1; c1 *= mag1;
                }

                for (int j = i+1; j < num_inliers; j++) {// verify through all including sample points
                    const int inl_idx_j = 4*inliers[j];
                    const auto X1 = pts[inl_idx_j], Y1 = pts[inl_idx_j+1], X2 = pts[inl_idx_j+2], Y2 = pts[inl_idx_j+3];
                    // use L1 norm instead of L2 for faster evaluation
                    if (fabsf(X1-x1) + fabsf(Y1 - y1) < neigh_thr || fabsf(X2-x2) + fabsf(Y2 - y2) < neigh_thr) {
                        num_non_random_inliers--;
                        // num_pts_bad_conditioning++;
                        break; // is dependent stop verification
                    } else if (is_F) {
                        if (fabsf(a2 * X2 + b2 * Y2 + c2) < line_thr && //|| // xj'^T F   xi
                            fabsf(a1 * X1 + b1 * Y1 + c1) < line_thr) { // xj^T  F^T xi'
                            num_non_random_inliers--;
                            break; // is dependent stop verification
                        }
                    }
                }
            }
        };
        if (params->isPnP()) {
            for (int i = 0; i < max_verify; i++) {
                const int inl_idx = 5*inliers[i];
                const auto x = pts[inl_idx], y = pts[inl_idx+1], X = pts[inl_idx+2], Y = pts[inl_idx+3], Z = pts[inl_idx+4];
                for (int j = i+1; j < num_inliers; j++) {
                    const int inl_idx_j = 5*inliers[j];
                    if (fabsf(x-pts[inl_idx_j  ]) + fabsf(y-pts[inl_idx_j+1]) < neigh_thr ||
                        fabsf(X-pts[inl_idx_j+2]) + fabsf(Y-pts[inl_idx_j+3]) + fabsf(Z-pts[inl_idx_j+4]) < neigh_thr) {
                        num_non_random_inliers--;
                        break;
                    }
                }
            }
        } else {
            removeDependentPoints(do_or_test, !ep1_inf && !ep2_inf);
            if (is_F) {
                const bool is_pts_vald_constr_normal = (double)num_pts_validatin_or_constr / num_inliers < 0.6;
                const bool is_pts_near_ep_normal = (double)num_pts_near_ep / num_inliers < 0.6;
                if (!is_pts_near_ep_normal || !is_pts_vald_constr_normal) {
                    num_non_random_inliers = num_inliers-sample_size;
                    num_pts_near_ep = 0; num_pts_validatin_or_constr = 0;
                    removeDependentPoints(is_pts_vald_constr_normal, is_pts_near_ep_normal);
                }
            }
        }
        return num_non_random_inliers;
    }

    bool run(Ptr<RansacOutput> &ransac_output) {
        if (points_size < params->getSampleSize())
            return false;
        const bool LO = params->getLO() != LocalOptimMethod::LOCAL_OPTIM_NULL,
            IS_FUNDAMENTAL = params->isFundamental(), IS_NON_RAND_TEST = params->isNonRandomnessTest();
        const int MAX_MODELS_ADAPT = 21, MAX_ITERS_ADAPT = MAX_MODELS_ADAPT/*assume at least 1 model from 1 sample*/,
            sample_size = params->getSampleSize();
        const double IOU_SIMILARITY_THR = 0.80;
        std::vector<int> non_degen_sample, best_sample;

        double lambda_non_random_all_inliers = -1;
        int final_iters, num_total_tested_models = 0;

        // non-random
        const int MAX_TEST_MODELS_NONRAND = IS_NON_RAND_TEST ? MAX_MODELS_ADAPT : 0;
        std::vector<Mat> models_for_random_test; models_for_random_test.reserve(MAX_TEST_MODELS_NONRAND);
        std::vector<std::vector<int>> samples_for_random_test; samples_for_random_test.reserve(MAX_TEST_MODELS_NONRAND);

        bool last_model_from_LO = false;
        Mat best_model, best_model_not_from_LO, K1_approx, K2_approx;
        Score best_score, best_score_model_not_from_LO;
        std::vector<bool> best_inliers_mask(points_size);
        if (! parallel) {
            // adaptive sprt test
            double IoU = 0, mean_num_est_models = 0;
            bool adapt = IS_NON_RAND_TEST || params->getVerifier() == VerificationMethod ::ASPRT, was_LO_run = false;
            int min_non_random_inliers = 30, iters = 0, num_estimations = 0, max_iters = params->getMaxIters();
            Mat non_degenerate_model, lo_model;
            Score current_score, non_degenerate_model_score, best_score_sample;
            std::vector<bool> model_inliers_mask (points_size);
            std::vector<Mat> models(_estimator->getMaxNumSolutions());
            std::vector<int> sample(_estimator->getMinimalSampleSize()), supports;
            supports.reserve(3*MAX_MODELS_ADAPT); // store model supports during adaption
            auto update_best = [&] (const Mat &new_model, const Score &new_score, bool from_lo=false) {
                _quality->getInliers(new_model, model_inliers_mask);
                IoU = Utils::intersectionOverUnion(best_inliers_mask, model_inliers_mask);
                best_inliers_mask = model_inliers_mask;

                if (!best_model.empty() && (int)models_for_random_test.size() < MAX_TEST_MODELS_NONRAND && IoU < IOU_SIMILARITY_THR && !from_lo) { // use IoU to not save similar models
                    // save old best model for non-randomness test if necessary
                    models_for_random_test.emplace_back(best_model.clone());
                    samples_for_random_test.emplace_back(best_sample);
                }

                // update score, model, inliers and max iterations
                best_score = new_score;
                new_model.copyTo(best_model);

                if (!from_lo) {
                    best_sample = sample;
                    if (IS_FUNDAMENTAL) { // avoid degeneracy after LO run
                        // save last model not from LO
                        best_model.copyTo(best_model_not_from_LO);
                        best_score_model_not_from_LO = best_score;
                    }
                }

                _model_verifier->update(best_score, iters);
                max_iters = _termination->update(best_model, best_score.inlier_number);
                // max_iters = std::max(max_iters, std::min(10, params->getMaxIters()));
                if (!adapt) // update quality and verifier to save evaluation time of a model
                    _quality->setBestScore(best_score.score);
                last_model_from_LO = from_lo;
            };

            auto run_lo = [&] (const Mat &_model, const Score &_score, bool force_lo) {
                was_LO_run = true;
                _local_optimization->setCurrentRANSACiter(force_lo ? iters : -1);
                Score lo_score;
                if (_local_optimization->refineModel(_model, _score, lo_model, lo_score) && lo_score.isBetter(best_score))
                    update_best(lo_model, lo_score, true);
            };

            for (; iters < max_iters; iters++) {
                _sampler->generateSample(sample);
                int number_of_models;
                if (adapt) {
                    number_of_models = _estimator->estimateModels(sample, models);
                    mean_num_est_models += number_of_models;
                    num_estimations++;
                } else {
                    number_of_models = _estimator->estimateModels(sample, models);
                }
                for (int i = 0; i < number_of_models; i++) {
                    num_total_tested_models++;
                    if (adapt) {
                        current_score = _quality->getScore(models[i]);
                        supports.emplace_back(current_score.inlier_number);
                        if (IS_NON_RAND_TEST && best_score_sample.isBetter(current_score)) {
                            models_for_random_test.emplace_back(models[i].clone());
                            samples_for_random_test.emplace_back(sample);
                        }
                    } else {
                        if (! _model_verifier->isModelGood(models[i], current_score))
                            continue;
                    }
                    if (current_score.isBetter(best_score_sample)) {
                        if (_degeneracy->recoverIfDegenerate(sample, models[i], current_score,
                                   non_degenerate_model, non_degenerate_model_score)) {
                            // check if best non degenerate model is better than so far the best model
                            if (non_degenerate_model_score.isBetter(best_score)) {
                                update_best(non_degenerate_model, non_degenerate_model_score);
                                best_score_sample = current_score.isBetter(best_score) ? best_score : current_score;
                            } else continue;
                        } else {
                            best_score_sample = current_score;
                            update_best(models[i], current_score);
                        }

                        if (LO && ((iters < max_iters && best_score.inlier_number > min_non_random_inliers && IoU < IOU_SIMILARITY_THR)))
                            run_lo(best_model, best_score, false);

                    } // end of if so far the best score
                } // end loop of number of models
                if (adapt && iters >= MAX_ITERS_ADAPT && num_total_tested_models >= MAX_MODELS_ADAPT) {
                    adapt = false;
                    lambda_non_random_all_inliers = getLambda(supports, 2.32, points_size, sample_size, false, min_non_random_inliers);
                    _model_verifier->updateSPRT(params->getTimeForModelEstimation(), 1.0, mean_num_est_models/num_estimations, lambda_non_random_all_inliers/points_size,(double)std::max(min_non_random_inliers, best_score.inlier_number)/points_size, best_score);
                }
            } // end main while loop
            final_iters = iters;
            if (! was_LO_run && !best_model.empty() && LO)
                run_lo(best_model, best_score, true);
        } else { // parallel VSAC
            const int MAX_THREADS = getNumThreads(), growth_max_samples = params->getProsacMaxSamples();
            const bool is_prosac = params->getSampler() == SamplingMethod::SAMPLING_PROSAC;
            std::atomic_bool success(false);
            std::atomic_int num_hypothesis_tested(0), thread_cnt(0), max_number_inliers(0), subset_size, termination_length;
            std::atomic<float> best_score_all(std::numeric_limits<float>::max());
            std::vector<Score> best_scores(MAX_THREADS), best_scores_not_LO;
            std::vector<Mat> best_models(MAX_THREADS), best_models_not_LO, K1_apx, K2_apx;
            std::vector<int> num_tested_models_threads(MAX_THREADS), growth_function, non_random_inliers;
            std::vector<std::vector<Mat>> tested_models_threads(MAX_THREADS);
            std::vector<std::vector<std::vector<int>>> tested_samples_threads(MAX_THREADS);
            std::vector<std::vector<int>> best_samples_threads(MAX_THREADS);
            std::vector<bool> last_model_from_LO_vec;
            std::vector<double> lambda_non_random_all_inliers_vec(MAX_THREADS);
            if (IS_FUNDAMENTAL) {
                last_model_from_LO_vec = std::vector<bool>(MAX_THREADS);
                best_models_not_LO = std::vector<Mat>(MAX_THREADS);
                best_scores_not_LO = std::vector<Score>(MAX_THREADS);
                K1_apx = std::vector<Mat>(MAX_THREADS);
                K2_apx = std::vector<Mat>(MAX_THREADS);
            }
            if (is_prosac) {
                growth_function = _sampler.dynamicCast<ProsacSampler>()->getGrowthFunction();
                subset_size = 2*sample_size; // n,  size of the current sampling pool
                termination_length = points_size;
            }
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            parallel_for_(Range(0, MAX_THREADS), [&](const Range & /*range*/) {
            if (!success) { // cover all if not success to avoid thread creating new variables
                const int thread_rng_id = thread_cnt++;
                bool adapt = params->getVerifier() == VerificationMethod ::ASPRT || IS_NON_RAND_TEST;
                int thread_state = _state + thread_rng_id, min_non_random_inliers = 0, num_tested_models = 0,
                    num_estimations = 0, mean_num_est_models = 0, iters, max_iters = params->getMaxIters();
                double IoU = 0, lambda_non_random_all_inliers_thread = -1;
                std::vector<Mat> tested_models_thread; tested_models_thread.reserve(MAX_TEST_MODELS_NONRAND);
                std::vector<std::vector<int>> tested_samples_thread; tested_samples_thread.reserve(MAX_TEST_MODELS_NONRAND);
                Ptr<UniformRandomGenerator> random_gen;
                if (is_prosac) random_gen = UniformRandomGenerator::create(thread_state);
                Ptr<Error> error;
                Ptr<Estimator> estimator;
                Ptr<Degeneracy> degeneracy;
                Ptr<Quality> quality;
                Ptr<ModelVerifier> model_verifier;
                Ptr<Sampler> sampler;
                Ptr<RandomGenerator> lo_sampler;
                Ptr<Termination> termination;
                Ptr<LocalOptimization> local_optimization;
                Ptr<MinimalSolver> min_solver;
                Ptr<NonMinimalSolver> non_min_solver;
                Ptr<WeightFunction> weight_fnc;
                initialize (thread_state, min_solver, non_min_solver, error, estimator, degeneracy, quality,
                        model_verifier, local_optimization, termination, sampler, lo_sampler, weight_fnc, true);
                bool is_last_from_LO_thread = false;
                Mat best_model_thread, non_degenerate_model, lo_model, best_not_LO_thread;
                Score best_score_thread, current_score, non_denegenerate_model_score, lo_score, best_score_all_threads, best_not_LO_score_thread;
                std::vector<int> sample(estimator->getMinimalSampleSize()), best_sample_thread, supports;
                supports.reserve(3*MAX_MODELS_ADAPT); // store model supports
                std::vector<bool> best_inliers_mask_local(points_size, false), model_inliers_mask(points_size, false);
                std::vector<Mat> models(estimator->getMaxNumSolutions());
                auto update_best = [&] (const Score &new_score, const Mat &new_model, bool from_LO=false) {
                    // update best score of all threads
                    if (max_number_inliers < new_score.inlier_number) max_number_inliers = new_score.inlier_number;
                    if (best_score_all > new_score.score)
                        best_score_all = new_score.score;
                    best_score_all_threads = Score(max_number_inliers, best_score_all);
                    //
                    quality->getInliers(new_model, model_inliers_mask);
                    IoU = Utils::intersectionOverUnion(best_inliers_mask_local, model_inliers_mask);
                    if (!best_model_thread.empty() && (int)tested_models_thread.size() < MAX_TEST_MODELS_NONRAND && IoU < IOU_SIMILARITY_THR) {
                        tested_models_thread.emplace_back(best_model_thread.clone());
                        tested_samples_thread.emplace_back(best_sample_thread);
                    }
                    if (!adapt) { // update quality and verifier
                        quality->setBestScore(best_score_all);
                        model_verifier->update(best_score_all_threads, iters);
                    }
                    // copy new score to best score
                    best_score_thread = new_score;
                    best_sample_thread = sample;
                    best_inliers_mask_local = model_inliers_mask;
                    // remember best model
                    new_model.copyTo(best_model_thread);

                    // update upper bound of iterations
                    if (is_prosac) {
                        int new_termination_length;
                        max_iters = termination.dynamicCast<ProsacTerminationCriteria>()->
                                updateTerminationLength(best_model_thread, best_score_thread.inlier_number, new_termination_length);
                        // update termination length
                        if (new_termination_length < termination_length)
                            termination_length = new_termination_length;
                    } else max_iters = termination->update(best_model_thread, max_number_inliers);
                    if (IS_FUNDAMENTAL) {
                        is_last_from_LO_thread = from_LO;
                        if (!from_LO) {
                            best_model_thread.copyTo(best_not_LO_thread);
                            best_not_LO_score_thread = best_score_thread;
                        }
                    }
                };
                bool was_LO_run = false;
                auto runLO = [&] (int current_ransac_iters) {
                    was_LO_run = true;
                    local_optimization->setCurrentRANSACiter(current_ransac_iters);
                    if (local_optimization->refineModel(best_model_thread, best_score_thread, lo_model,
                            lo_score) && lo_score.isBetter(best_score_thread))
                        update_best(lo_score, lo_model, true);
                };
                for (iters = 0; iters < max_iters && !success; iters++) {
                    success = num_hypothesis_tested++ > max_iters;
                    if (iters % 10 && !adapt) {
                        // Synchronize threads. just to speed verification of model.
                        quality->setBestScore(std::min(best_score_thread.score, (float)best_score_all));
                        model_verifier->update(best_score_thread.inlier_number > max_number_inliers ? best_score_thread : best_score_all_threads, iters);
                    }

                    if (is_prosac) {
                        if (num_hypothesis_tested > growth_max_samples) {
                            // if PROSAC has not converged to solution then do uniform sampling.
                            random_gen->generateUniqueRandomSet(sample, sample_size, points_size);
                        } else {
                            if (num_hypothesis_tested >= growth_function[subset_size-1] && subset_size < termination_length-MAX_THREADS) {
                                subset_size++;
                                if (subset_size >= points_size) subset_size = points_size-1;
                            }
                            if (growth_function[subset_size-1] < num_hypothesis_tested) {
                                // The sample contains m-1 points selected from U_(n-1) at random and u_n
                                random_gen->generateUniqueRandomSet(sample, sample_size-1, subset_size-1);
                                sample[sample_size-1] = subset_size-1;
                            } else
                                // Select m points from U_n at random.
                                random_gen->generateUniqueRandomSet(sample, sample_size, subset_size);
                        }
                    } else sampler->generateSample(sample); // use local sampler

                    const int number_of_models = estimator->estimateModels(sample, models);
                    if (adapt) {
                        num_estimations++; mean_num_est_models += number_of_models;
                    }
                    for (int i = 0; i < number_of_models; i++) {
                        num_tested_models++;
                        if (adapt) {
                            current_score = quality->getScore(models[i]);
                            supports.emplace_back(current_score.inlier_number);
                        } else if (! model_verifier->isModelGood(models[i], current_score))
                            continue;

                        if (current_score.isBetter(best_score_all_threads)) {
                            if (degeneracy->recoverIfDegenerate(sample, models[i], current_score,
                                    non_degenerate_model, non_denegenerate_model_score)) {
                                // check if best non degenerate model is better than so far the best model
                                if (non_denegenerate_model_score.isBetter(best_score_thread))
                                    update_best(non_denegenerate_model_score, non_degenerate_model);
                                else continue;
                            } else update_best(current_score, models[i]);
                            if (LO && num_hypothesis_tested < max_iters && IoU < IOU_SIMILARITY_THR &&
                                    best_score_thread.inlier_number > min_non_random_inliers)
                                runLO(iters);
                        } // end of if so far the best score
                        else if ((int)tested_models_thread.size() < MAX_TEST_MODELS_NONRAND) {
                            tested_models_thread.emplace_back(models[i].clone());
                            tested_samples_thread.emplace_back(sample);
                        }
                        if (num_hypothesis_tested > max_iters) {
                            success = true; break;
                        }
                    } // end loop of number of models
                    if (adapt && iters >= MAX_ITERS_ADAPT && num_tested_models >= MAX_MODELS_ADAPT) {
                        adapt = false;
                        lambda_non_random_all_inliers_thread = getLambda(supports, 2.32, points_size, sample_size, false, min_non_random_inliers);
                        model_verifier->updateSPRT(params->getTimeForModelEstimation(), 1, (double)mean_num_est_models/num_estimations, lambda_non_random_all_inliers_thread/points_size,
                             (double)std::max(min_non_random_inliers, best_score.inlier_number)/points_size, best_score_all_threads);
                    }
                    if (!adapt && LO && num_hypothesis_tested < max_iters && !was_LO_run && !best_model_thread.empty() &&
                            best_score_thread.inlier_number > min_non_random_inliers)
                        runLO(iters);
                } // end of loop over iters
                if (! was_LO_run && !best_model_thread.empty() && LO)
                    runLO(-1 /*use full iterations of LO*/);
                best_model_thread.copyTo(best_models[thread_rng_id]);
                best_scores[thread_rng_id] = best_score_thread;
                num_tested_models_threads[thread_rng_id] = num_tested_models;
                tested_models_threads[thread_rng_id] = tested_models_thread;
                tested_samples_threads[thread_rng_id] = tested_samples_thread;
                best_samples_threads[thread_rng_id] = best_sample_thread;
                if (IS_FUNDAMENTAL) {
                    best_scores_not_LO[thread_rng_id] = best_not_LO_score_thread;
                    best_not_LO_thread.copyTo(best_models_not_LO[thread_rng_id]);
                    last_model_from_LO_vec[thread_rng_id] = is_last_from_LO_thread;
                }
                lambda_non_random_all_inliers_vec[thread_rng_id] = lambda_non_random_all_inliers_thread;
            }}); // end parallel
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            // find best model from all threads' models
            best_score = best_scores[0];
            int best_thread_idx = 0;
            for (int i = 1; i < MAX_THREADS; i++)
                if (best_scores[i].isBetter(best_score)) {
                    best_score = best_scores[i];
                    best_thread_idx = i;
                }
            best_model = best_models[best_thread_idx];
            if (IS_FUNDAMENTAL) {
                last_model_from_LO = last_model_from_LO_vec[best_thread_idx];
                K1_approx = K1_apx[best_thread_idx];
                K2_approx = K2_apx[best_thread_idx];
            }
            final_iters = num_hypothesis_tested;
            best_sample = best_samples_threads[best_thread_idx];
            int num_lambdas = 0;
            double avg_lambda = 0;
            for (int i = 0; i < MAX_THREADS; i++) {
                if (IS_FUNDAMENTAL && best_scores_not_LO[i].isBetter(best_score_model_not_from_LO)) {
                    best_score_model_not_from_LO = best_scores_not_LO[i];
                    best_models_not_LO[i].copyTo(best_model_not_from_LO);
                }
                if (IS_NON_RAND_TEST && lambda_non_random_all_inliers_vec[i] > 0) {
                    num_lambdas ++;
                    avg_lambda += lambda_non_random_all_inliers_vec[i];
                }
                num_total_tested_models += num_tested_models_threads[i];
                if ((int)models_for_random_test.size() < MAX_TEST_MODELS_NONRAND) {
                    for (int m = 0; m < (int)tested_models_threads[i].size(); m++) {
                        models_for_random_test.emplace_back(tested_models_threads[i][m].clone());
                        samples_for_random_test.emplace_back(tested_samples_threads[i][m]);
                        if ((int)models_for_random_test.size() == MAX_TEST_MODELS_NONRAND)
                            break;
                    }
                }
            }
            if (IS_NON_RAND_TEST && num_lambdas > 0 && avg_lambda > 0)
                lambda_non_random_all_inliers = avg_lambda / num_lambdas;
        }
        if (best_model.empty()) {
            ransac_output = RansacOutput::create(best_model, std::vector<bool>(), best_score.inlier_number, final_iters, ModelConfidence::RANDOM, std::vector<float>());
            return false;
        }
        if (last_model_from_LO && IS_FUNDAMENTAL && K1.empty() && K2.empty()) {
            Score new_score; Mat new_model;
            const double INL_THR = 0.80;
            if (parallel)
                _quality->getInliers(best_model, best_inliers_mask);
            // run additional degeneracy check for F:
            if (_degeneracy.dynamicCast<FundamentalDegeneracy>()->verifyFundamental(best_model, best_score, best_inliers_mask, new_model, new_score)) {
                // so-far-the-best F is degenerate
                // Update best F using non-degenerate F or the one which is not from LO
                if (new_score.isBetter(best_score_model_not_from_LO) && new_score.inlier_number > INL_THR*best_score.inlier_number) {
                    best_score = new_score;
                    new_model.copyTo(best_model);
                } else if (best_score_model_not_from_LO.inlier_number > INL_THR*best_score.inlier_number) {
                    best_score = best_score_model_not_from_LO;
                    best_model_not_from_LO.copyTo(best_model);
                }
            } else { // so-far-the-best F is not degenerate
                if (new_score.isBetter(best_score)) {
                     // if new model is better then update
                    best_score = new_score;
                    new_model.copyTo(best_model);
                }
            }
        }
        if (params->getFinalPolisher() != PolishingMethod::NONE_POLISHER) {
            Mat polished_model;
            Score polisher_score;
            if (polisher->polishSoFarTheBestModel(best_model, best_score, // polish final model
                  polished_model, polisher_score) && polisher_score.isBetter(best_score)) {
                best_score = polisher_score;
                polished_model.copyTo(best_model);
            }
        }

        ///////////////// get inliers of the best model and points' residuals ///////////////
        std::vector<bool> inliers_mask; std::vector<float> residuals;
        if (params->isMaskRequired()) {
            inliers_mask = std::vector<bool>(points_size);
            residuals = _error->getErrors(best_model);
            _quality->getInliers(residuals, inliers_mask, threshold);
        }

        ModelConfidence model_conf = ModelConfidence::UNKNOWN;
        if (IS_NON_RAND_TEST) {
            std::vector<int> temp_inliers(points_size);
            const int non_random_inls_best_model = getIndependentInliers(best_model, best_sample, temp_inliers,
                         _quality->getInliers(best_model, temp_inliers));
            // quick test on lambda from all inliers (= upper bound of independent inliers)
            // if model with independent inliers is not random for Poisson with all inliers then it is not random using independent inliers too
            if (pow(Utils::getPoissonCDF(lambda_non_random_all_inliers, non_random_inls_best_model), num_total_tested_models) < 0.9999) {
                std::vector<int> inliers_list(models_for_random_test.size());
                for (int m = 0; m < (int)models_for_random_test.size(); m++)
                    inliers_list[m] = getIndependentInliers(models_for_random_test[m], samples_for_random_test[m],
                        temp_inliers, _quality->getInliers(models_for_random_test[m], temp_inliers));
                int min_non_rand_inliers;
                const double lambda = getLambda(inliers_list, 1.644, points_size, sample_size, true, min_non_rand_inliers);
                const double cdf_lambda = Utils::getPoissonCDF(lambda, non_random_inls_best_model), cdf_N = pow(cdf_lambda, num_total_tested_models);
                model_conf = cdf_N < 0.9999 ? ModelConfidence ::RANDOM : ModelConfidence ::NON_RANDOM;
            } else model_conf = ModelConfidence ::NON_RANDOM;
        }
        ransac_output = RansacOutput::create(best_model, inliers_mask, best_score.inlier_number, final_iters, model_conf, residuals);
        return true;
    }
};

/*
 * pts1, pts2 are matrices either N x a, N x b or a x N or b x N, where N > a and N > b
 * pts1 are image points, if pnp pts2 are object points otherwise - image points as well.
 * output is matrix of size N x (a + b)
 * return points_size = N
 */
int mergePoints (InputArray pts1_, InputArray pts2_, Mat &pts, EstimationMethod estimator) {

    Mat pts1 = pts1_.getMat(), pts2 = pts2_.getMat();
    auto convertPoints = [] (Mat &points, int pt_dim) {
        points.convertTo(points, CV_32F); // convert points to have float precision
        if (points.channels() > 1)
            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
        if (points.rows < points.cols)
            transpose(points, points); // transpose so points will be in rows
        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
            points = points.colRange(0, pt_dim);
    };

    switch (estimator) {
        case EstimationMethod::HOMOGRAPHY:
        case EstimationMethod::FUNDAMENTAL:
        case EstimationMethod::FUNDAMENTAL8:
        case EstimationMethod::ESSENTIAL:
        case EstimationMethod::AFFINE:
        case EstimationMethod::SE2:
        case EstimationMethod::SIM2:
            convertPoints(pts1, 2); // pts1 are always image points
            convertPoints(pts2, 2);
            break;
        case EstimationMethod::SO3:
        case EstimationMethod::SE3:
        case EstimationMethod::SIM3:
            convertPoints(pts1, 3);
            convertPoints(pts2, 3);
            break;
        case EstimationMethod::P3P:
        case EstimationMethod::P6P:
            convertPoints(pts1, 2); // pts1 are always image points
            convertPoints(pts2, 3); // for PnP points are 3D
            break;
        default: CV_Error(cv::Error::StsNotImplemented, "Estimator has not implemented yet!");
    }

    // points are of size [Nx2 Nx2] = Nx4 for H, F, E
    // points are of size [Nx3 Nx3] = Nx6 for SO3, SE3, SIM3
    // points are of size [Nx2 Nx3] = Nx5 for PnP
    hconcat(pts1, pts2, pts);
    return pts.rows;
}

void saveMask (OutputArray mask, const std::vector<bool> &inliers_mask) {
    if (mask.needed()) {
        const int points_size = (int) inliers_mask.size();
        Mat tmp_mask(points_size, 1, CV_8U);
        auto * maskptr = tmp_mask.ptr<uchar>();
        for (int i = 0; i < points_size; i++)
            maskptr[i] = (uchar) inliers_mask[i];
        tmp_mask.copyTo(mask);
    }
}
void setParameters (Ptr<Model> &params, EstimationMethod estimator, const UsacParams &usac_params,
        bool mask_needed) {
    params = Model::create(usac_params.threshold, estimator, usac_params.sampler,
            usac_params.confidence, usac_params.maxIterations, usac_params.score);
    params->setLocalOptimization(usac_params.loMethod);
    params->setLOSampleSize(usac_params.loSampleSize);
    params->setLOIterations(usac_params.loIterations);
    params->setParallel(usac_params.isParallel);
    params->setNeighborsType(usac_params.neighborsSearch);
    params->setRandomGeneratorState(usac_params.randomGeneratorState);
    params->maskRequired(mask_needed);
}

void setParameters (int flag, Ptr<Model> &params, EstimationMethod estimator, double thr,
        int max_iters, double conf, bool mask_needed) {
    switch (flag) {
        case USAC_DEFAULT:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_AND_ITER_LO);
            break;
        case USAC_MAGSAC:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MAGSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_SIGMA);
            params->setLOSampleSize(params->isHomography() ? 75 : 50);
            params->setLOIterations(params->isHomography() ? 15 : 10);
            break;
        case USAC_PARALLEL:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setParallel(true);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_LO);
            break;
        case USAC_ACCURATE:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_GC);
            params->setLOSampleSize(20);
            params->setLOIterations(25);
            break;
        case USAC_FAST:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_AND_ITER_LO);
            params->setLOIterations(5);
            params->setLOIterativeIters(3);
            break;
        case USAC_PROSAC:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_PROSAC, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_LO);
            break;
        case USAC_FM_8PTS:
            params = Model::create(thr, EstimationMethod::FUNDAMENTAL8,SamplingMethod::SAMPLING_UNIFORM,
                    conf, max_iters,ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_LO);
            break;
        default: CV_Error(cv::Error::StsBadFlag, "Incorrect flag for USAC!");
    }
    // do not do too many iterations for PnP
    if (estimator == EstimationMethod::P3P) {
        if (params->getLOInnerMaxIters() > 10)
            params->setLOIterations(10);
        params->setLOIterativeIters(0);
        params->setFinalLSQ(3);
    }

    params->maskRequired(mask_needed);
}

Mat findHomography (InputArray srcPoints, InputArray dstPoints, int method, double thr,
        OutputArray mask, const int max_iters, const double confidence) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::HOMOGRAPHY, thr, max_iters, confidence, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, srcPoints, dstPoints,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel() / ransac_output->getModel().at<double>(2,2);
    }
    if (mask.needed()){
        mask.create(std::max(srcPoints.getMat().rows, srcPoints.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat findFundamentalMat( InputArray points1, InputArray points2, int method, double thr,
        double confidence, int max_iters, OutputArray mask ) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::FUNDAMENTAL, thr, max_iters, confidence, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, points1, points2,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel();
    }
    if (mask.needed()){
        mask.create(std::max(points1.getMat().rows, points1.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat findEssentialMat (InputArray points1, InputArray points2, InputArray cameraMatrix1,
        int method, double prob, double thr, OutputArray mask, int maxIters) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::ESSENTIAL, thr, maxIters, prob, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, points1, points2,
            ransac_output, cameraMatrix1, cameraMatrix1, noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel();
    }
    if (mask.needed()){
        mask.create(std::max(points1.getMat().rows, points1.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
       InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec,
       bool /*useExtrinsicGuess*/, int max_iters, float thr, double conf,
       OutputArray inliers, int method) {
    Ptr<Model> params;
    setParameters(method, params, cameraMatrix.empty() ? EstimationMethod ::P6P : EstimationMethod ::P3P,
            thr, max_iters, conf, inliers.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, imagePoints, objectPoints,
            ransac_output, cameraMatrix, noArray(), distCoeffs, noArray())) {
        if (inliers.needed()) {
            const auto &inliers_mask = ransac_output->getInliersMask();
            Mat inliers_;
            for (int i = 0; i < (int)inliers_mask.size(); i++)
                if (inliers_mask[i])
                    inliers_.push_back(i);
            inliers_.copyTo(inliers);
        }
        const Mat &model = ransac_output->getModel();
        model.col(0).copyTo(rvec);
        model.col(1).copyTo(tvec);
        return true;
    }
    return false;
}

Mat estimateAffine2D(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::AFFINE, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,2);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat estimateSE2(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod ::SE2, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,2);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat estimateSIM2(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod ::SIM2, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,2);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat estimateSO3(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod ::SO3, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,3);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat estimateSE3(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod ::SE3, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,3);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat estimateSIM3(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod ::SIM3, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to,
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,3);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

class ModelImpl : public Model {
private:
    // main parameters:
    double threshold;
    EstimationMethod estimator;
    SamplingMethod sampler;
    double confidence;
    int max_iterations;
    ScoreMethod score;
    int sample_size;

    // Larsson parameters
    bool is_larsson_optimization = true;
    int larsson_leven_marq_iters_lo = 10, larsson_leven_marq_iters_fo = 15;

    // solver for a null-space extraction
    MethodSolver null_solver = GEM_SOLVER;

    // prosac
    int prosac_max_samples = 200000;

    // for neighborhood graph
    int k_nearest_neighbors = 8;//, flann_search_params = 5, num_kd_trees = 1; // for FLANN
    int cell_size = 50; // pixels, for grid neighbors searching
    int radius = 30; // pixels, for radius-search neighborhood graph
    NeighborSearchMethod neighborsType = NeighborSearchMethod::NEIGH_GRID;

    // Local Optimization parameters
    LocalOptimMethod lo = LocalOptimMethod ::LOCAL_OPTIM_INNER_LO;
    int lo_sample_size=12, lo_inner_iterations=20, lo_iterative_iterations=8,
            lo_thr_multiplier=10, lo_iter_sample_size = 30;

    // Graph cut parameters
    const double spatial_coherence_term = 0.975;

    // apply polisher for final RANSAC model
    PolishingMethod polisher = PolishingMethod ::COV_POLISHER;

    // preemptive verification test
    VerificationMethod verifier = VerificationMethod ::ASPRT;

    // sprt parameters
    // lower bound estimate is 2% of inliers
    // model estimation to verification time = ratio of time needed to estimate model
    // to verification of one point wrt the model
    double sprt_eps = 0.02, sprt_delta = 0.008, avg_num_models, model_est_to_ver_time;

    // estimator error
    ErrorMetric est_error;

    // progressive napsac
    double relax_coef = 0.1;
    // for building neighborhood graphs
    const std::vector<int> grid_cell_number = {10, 5, 2};

    //for final least squares polisher
    int final_lsq_iters = 7;

    bool need_mask = true, // do we need inlier mask in the end
        is_parallel = false, // use parallel RANSAC
        is_nonrand_test = false; // is test for the final model non-randomness

    // state of pseudo-random number generator
    int random_generator_state = 0;

    // number of iterations of plane-and-parallax in DEGENSAC^+
    int plane_and_parallax_max_iters = 300;

    // magsac parameters:
    int DoF = 2;
    double sigma_quantile = 3.04, upper_incomplete_of_sigma_quantile = 0.00419,
            lower_incomplete_of_sigma_quantile = 0.8629, C = 0.5, maximum_thr = 7.5;
    double k_mlesac = 2.25; // parameter for MLESAC model evaluation
public:
    ModelImpl (double threshold_, EstimationMethod estimator_, SamplingMethod sampler_, double confidence_,
            int max_iterations_, ScoreMethod score_) :
           threshold(threshold_), estimator(estimator_), sampler(sampler_), confidence(confidence_), max_iterations(max_iterations_), score(score_) {
        switch (estimator_) {
            case (EstimationMethod::AFFINE):
                avg_num_models = 1; model_est_to_ver_time = 50;
                sample_size = 3; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::SE2):
            case (EstimationMethod::SIM2):
                avg_num_models = 1; model_est_to_ver_time = 100;
                sample_size = 2; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::SO3):
            case (EstimationMethod::SE3):
            case (EstimationMethod::SIM3):
                avg_num_models = 1; model_est_to_ver_time = 100;
                sample_size = 3; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::HOMOGRAPHY):
                avg_num_models = 0.8; model_est_to_ver_time = 200;
                sample_size = 4; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::FUNDAMENTAL):
                DoF = 4; C = 0.25; sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.003657; lower_incomplete_of_sigma_quantile = 1.3012;
                maximum_thr = 2.5;
                avg_num_models = 1.5; model_est_to_ver_time = 200;
                sample_size = 7; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::FUNDAMENTAL8):
                avg_num_models = 1; model_est_to_ver_time = 100; maximum_thr = 2.5;
                sample_size = 8; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::ESSENTIAL):
                DoF = 4; C = 0.25; sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.003657; lower_incomplete_of_sigma_quantile = 1.3012;
                avg_num_models = 3.93; model_est_to_ver_time = 1000; maximum_thr = 2;
                sample_size = 5; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::P3P):
                avg_num_models = 1.38; model_est_to_ver_time = 800;
                sample_size = 3; est_error = ErrorMetric ::RERPOJ; break;
            case (EstimationMethod::P6P):
                avg_num_models = 1; model_est_to_ver_time = 300;
                sample_size = 6; est_error = ErrorMetric ::RERPOJ; break;
            default: CV_Error(cv::Error::StsNotImplemented, "Estimator has not implemented yet!");
        }

        if (score_ == ScoreMethod::SCORE_METHOD_MAGSAC)
            polisher = PolishingMethod::MAGSAC;

        // for PnP problem we can use only KNN graph
        if (estimator_ == EstimationMethod::P3P || estimator_ == EstimationMethod::P6P) {
            polisher = LSQ_POLISHER;
            neighborsType = NeighborSearchMethod::NEIGH_FLANN_KNN;
            k_nearest_neighbors = 2;
        }
    }

    // setters
    void setNonRandomnessTest (bool set) override { is_nonrand_test = set; }
    void setVerifier (VerificationMethod verifier_) override { verifier = verifier_; }
    void setPolisher (PolishingMethod polisher_) override { polisher = polisher_; }
    void setParallel (bool is_parallel_) override { is_parallel = is_parallel_; }
    void setError (ErrorMetric error_) override { est_error = error_; }
    void setLocalOptimization (LocalOptimMethod lo_) override { lo = lo_; }
    void setKNearestNeighhbors (int knn_) override { k_nearest_neighbors = knn_; }
    void setNeighborsType (NeighborSearchMethod neighbors) override { neighborsType = neighbors; }
    void setCellSize (int cell_size_) override { cell_size = cell_size_; }
    void setLOIterations (int iters) override { lo_inner_iterations = iters; }
    void setLOSampleSize (int lo_sample_size_) override { lo_sample_size = lo_sample_size_; }
    void maskRequired (bool need_mask_) override { need_mask = need_mask_; }
    void setRandomGeneratorState (int state) override { random_generator_state = state; }
    void setLOIterativeIters (int iters) override { lo_iterative_iterations = iters; }
    void setFinalLSQ (int iters) override { final_lsq_iters = iters; }

    // getters
    int getProsacMaxSamples() const override { return prosac_max_samples; }
    int getLevMarqIters () const override { return larsson_leven_marq_iters_fo; }
    int getLevMarqItersLO () const override { return larsson_leven_marq_iters_lo; }
    bool isNonRandomnessTest () const override { return is_nonrand_test; }
    bool isMaskRequired () const override { return need_mask; }
    NeighborSearchMethod getNeighborsSearch () const override { return neighborsType; }
    int getKNN () const override { return k_nearest_neighbors; }
    ErrorMetric getError () const override { return est_error; }
    EstimationMethod getEstimator () const override { return estimator; }
    int getSampleSize () const override { return sample_size; }
    int getFinalLSQIterations () const override { return final_lsq_iters; }
    int getDegreesOfFreedom () const override { return DoF; }
    double getSigmaQuantile () const override { return sigma_quantile; }
    double getUpperIncompleteOfSigmaQuantile () const override {
        return upper_incomplete_of_sigma_quantile;
    }
    double getLowerIncompleteOfSigmaQuantile () const override {
        return lower_incomplete_of_sigma_quantile;
    }
    double getC () const override { return C; }
    double getKmlesac () const override { return k_mlesac; }
    double getMaximumThreshold () const override { return maximum_thr; }
    double getGraphCutSpatialCoherenceTerm () const override { return spatial_coherence_term; }
    int getLOSampleSize () const override { return lo_sample_size; }
    MethodSolver getRansacSolver () const override { return null_solver; }
    PolishingMethod getFinalPolisher () const override { return polisher; }
    int getLOThresholdMultiplier() const override { return lo_thr_multiplier; }
    int getLOIterativeSampleSize() const override { return lo_iter_sample_size; }
    int getLOIterativeMaxIters() const override { return lo_iterative_iterations; }
    int getLOInnerMaxIters() const override { return lo_inner_iterations; }
    int getPlaneAndParallaxIters () const override { return plane_and_parallax_max_iters; }
    LocalOptimMethod getLO () const override { return lo; }
    ScoreMethod getScore () const override { return score; }
    int getMaxIters () const override { return max_iterations; }
    double getConfidence () const override { return confidence; }
    double getThreshold () const override { return threshold; }
    VerificationMethod getVerifier () const override { return verifier; }
    SamplingMethod getSampler () const override { return sampler; }
    int getRandomGeneratorState () const override { return random_generator_state; }
    double getSPRTdelta () const override { return sprt_delta; }
    double getSPRTepsilon () const override { return sprt_eps; }
    double getSPRTavgNumModels () const override { return avg_num_models; }
    int getCellSize () const override { return cell_size; }
    int getGraphRadius() const override { return radius; }
    double getTimeForModelEstimation () const override { return model_est_to_ver_time; }
    double getRelaxCoef () const override { return relax_coef; }
    const std::vector<int> &getGridCellNumber () const override { return grid_cell_number; }
    bool isLarssonOptimization () const override { return is_larsson_optimization; }
    bool isParallel () const override { return is_parallel; }
    bool isFundamental () const override {
        return estimator == EstimationMethod::FUNDAMENTAL ||
               estimator == EstimationMethod::FUNDAMENTAL8;
    }
    bool isHomography () const override { return estimator == EstimationMethod::HOMOGRAPHY; }
    bool isEssential () const override { return estimator == EstimationMethod::ESSENTIAL; }
    bool isPnP() const override {
        return estimator == EstimationMethod ::P3P || estimator == EstimationMethod ::P6P;
    }
    bool isPtsetReg() const override {
        return estimator == EstimationMethod ::AFFINE ||
               estimator == EstimationMethod ::SE2 || estimator == EstimationMethod ::SIM2 ||
               estimator == EstimationMethod ::SO3 ||
               estimator == EstimationMethod ::SE3 || estimator == EstimationMethod ::SIM3;
    }
};

Ptr<Model> Model::create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
                         double confidence_, int max_iterations_, ScoreMethod score_) {
    return makePtr<ModelImpl>(threshold_, estimator_, sampler_, confidence_,
                              max_iterations_, score_);
}

bool run (const Ptr<const Model> &params, InputArray points1, InputArray points2,
       Ptr<RansacOutput> &ransac_output, InputArray K1_, InputArray K2_,
       InputArray dist_coeff1, InputArray dist_coeff2) {
    Ransac ransac (params, points1, points2, K1_, K2_, dist_coeff1, dist_coeff2);
    if (ransac.run(ransac_output)) {
        if (params->isPnP()) {
            // convert R to rodrigues and back and recalculate inliers which due to numerical
            // issues can differ
            Mat out, newP;
            Matx33d R, newR, K1;
            Vec3d t, rvec;
            if (K1_.empty()) {
                usac::Utils::decomposeProjection (ransac_output->getModel(), K1, R, t);
                Rodrigues(R, rvec);
                hconcat(rvec, t, out);
                hconcat(out, K1, out);
            } else {
                K1 = ransac.K1;
                const Mat Rt = Mat(Matx33d(K1).inv() * Matx34d(ransac_output->getModel()));
                t = Rt.col(3);
                Rodrigues(Rt.colRange(0,3), rvec);
                hconcat(rvec, t, out);
            }
            // Matx33d _K1(K1);
            Rodrigues(rvec, newR);
            hconcat(K1 * Matx33d(newR), K1 * Vec3d(t), newP);
            std::vector<bool> inliers_mask(ransac.points_size);
            ransac._quality->getInliers(newP, inliers_mask);
            ransac_output = RansacOutput::create(out, inliers_mask, ransac_output->getNumberOfInliers(),
                ransac_output->getNumberOfIters(), ransac_output->getConfidence(), ransac_output->getResiduals());
        }
        return true;
    }
    return false;
}
}}
