// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_USAC_USAC_HPP
#define OPENCV_USAC_USAC_HPP

namespace cv { namespace usac {
enum EstimationMethod { HOMOGRAPHY=0, FUNDAMENTAL=1, FUNDAMENTAL8=2, ESSENTIAL=3, AFFINE=4, P3P=5, P6P=6, SE2=7, SIM2=8, SO3=9, SE3=10, SIM3=11};
enum VerificationMethod { NULL_VERIFIER=0, SPRT_VERIFIER=1, ASPRT=2 };
enum ErrorMetric {DIST_TO_LINE=0, SAMPSON_ERR=1, SGD_ERR=2, SYMM_REPR_ERR=3, FORW_REPR_ERR=4, RERPOJ=5};
enum MethodSolver { GEM_SOLVER=0, SVD_SOLVER=1 };
enum ModelConfidence {RANDOM=0, NON_RANDOM=1, UNKNOWN=2};

// Abstract Error class
class Error : public Algorithm {
public:
    // set model to use getError() function
    virtual void setModelParameters (const Mat &model) = 0;
    // returns error of point wih @point_idx w.r.t. model
    virtual float getError (int point_idx) const = 0;
    virtual const std::vector<float> &getErrors (const Mat &model) = 0;
};

// Symmetric Reprojection Error for Homography
class ReprojectionErrorSymmetric : public Error {
public:
    static Ptr<ReprojectionErrorSymmetric> create(const Mat &points);
};

// Forward Reprojection Error for Homography
class ReprojectionErrorForward : public Error {
public:
    static Ptr<ReprojectionErrorForward> create(const Mat &points);
};

// Sampson Error for Fundamental matrix
class SampsonError : public Error {
public:
    static Ptr<SampsonError> create(const Mat &points);
};

// Symmetric Geometric Distance (to epipolar lines) for Fundamental and Essential matrix
class SymmetricGeometricDistance : public Error {
public:
    static Ptr<SymmetricGeometricDistance> create(const Mat &points);
};

// Reprojection Error for Projection matrix
class ReprojectionErrorPmatrix : public Error {
public:
    static Ptr<ReprojectionErrorPmatrix> create(const Mat &points);
};

// Reprojection Error for Affine matrix
class ReprojectionErrorAffine : public Error {
public:
    static Ptr<ReprojectionErrorAffine> create(const Mat &points);
};

// Reprojection Error for Affine 3D matrix
class ReprojectionErrorAffine3D : public Error {
public:
    static Ptr<ReprojectionErrorAffine3D> create(const Mat &points);
};

class TrifocalTensorReprError : public Error {
public:
    static Ptr<TrifocalTensorReprError> create(const Mat &points);
};

// Normalizing transformation of data points
class NormTransform : public Algorithm {
public:
    /*
     * @norm_points is output matrix of size pts_size x 4
     * @sample constains indices of points
     * @sample_number is number of used points in sample <0; sample_number)
     * @T1, T2 are output transformation matrices
     */
    virtual void getNormTransformation (Mat &norm_points, const std::vector<int> &sample,
                                        int sample_number, Matx33d &T1, Matx33d &T2) const = 0;
    static Ptr<NormTransform> create (const Mat &points);
};

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SOLVER ///////////////////////////////////////////
class MinimalSolver : public Algorithm {
public:
    // Estimate models from minimal sample. models.size() == number of found solutions
    virtual int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;
    // return minimal sample size required for estimation.
    virtual int getSampleSize() const = 0;
    // return maximum number of possible solutions.
    virtual int getMaxNumberOfSolutions () const = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyMinimalSolver4pts : public MinimalSolver {
public:
    static Ptr<HomographyMinimalSolver4pts> create(const Mat &points, bool use_ge);
};
class PnPSVDSolver : public MinimalSolver {
public:
    static Ptr<PnPSVDSolver> create (const Mat &points);
};

//-------------------------- FUNDAMENTAL MATRIX -----------------------
class FundamentalMinimalSolver7pts : public MinimalSolver {
public:
    static Ptr<FundamentalMinimalSolver7pts> create(const Mat &points, bool use_ge);
};

class FundamentalMinimalSolver8pts : public MinimalSolver {
public:
    static Ptr<FundamentalMinimalSolver8pts> create(const Mat &points_);
};

//-------------------------- ESSENTIAL MATRIX -----------------------
class EssentialMinimalSolver5pts : public MinimalSolver {
public:
    static Ptr<EssentialMinimalSolver5pts> create(const Mat &points, bool use_svd, bool is_nister);
};

//-------------------------- PNP -----------------------
class PnPMinimalSolver6Pts : public MinimalSolver {
public:
    static Ptr<PnPMinimalSolver6Pts> create(const Mat &points_);
};

class P3PSolver : public MinimalSolver {
public:
    static Ptr<P3PSolver> create(const Mat &points_, const Mat &calib_norm_pts, const Mat &K);
};

//-------------------------- AFFINE -----------------------
class AffineMinimalSolver : public MinimalSolver {
public:
    static Ptr<AffineMinimalSolver> create(const Mat &points_);
};

//-------------------------- SE/SIM -----------------------
class SE2MinimalSolver : public MinimalSolver {
public:
    static Ptr<SE2MinimalSolver> create(const Mat &points_);
};
// class PartialNdMinimalSolver : public MinimalSolver {
// public:
//     static Ptr<PartialNdMinimalSolver> create(const Mat &points_, const int Nd=2, const bool is_similarity=false);
// };
class SIM2MinimalSolver : public MinimalSolver {
public:
    static Ptr<SIM2MinimalSolver> create(const Mat &points_);
};
class SO3MinimalSolver : public MinimalSolver {
public:
    static Ptr<SO3MinimalSolver> create(const Mat &points_);
};
class SE3MinimalSolver : public MinimalSolver {
public:
    static Ptr<SE3MinimalSolver> create(const Mat &points_);
};
class SIM3MinimalSolver : public MinimalSolver {
public:
    static Ptr<SIM3MinimalSolver> create(const Mat &points_);
};

class TrifocalTensorMinimalSolver : public MinimalSolver {
public:
    static Ptr<TrifocalTensorMinimalSolver> create(const Mat &points_);
    virtual void getFundamentalMatricesFromTensor (const cv::Mat &tensor, cv::Mat &F21, cv::Mat &F31) = 0;
};

//////////////////////////////////////// NON MINIMAL SOLVER ///////////////////////////////////////
class NonMinimalSolver : public Algorithm {
public:
    virtual int estimate (const Mat &model, const std::vector<int> &sample, int sample_size, std::vector<Mat>
            &models, const std::vector<double> &weights) const {
        CV_UNUSED(model);
        return estimate(sample, sample_size, models, weights);
    }
    // Estimate models from non minimal sample. models.size() == number of found solutions
    virtual int estimate (const std::vector<int> &sample, int sample_size,
          std::vector<Mat> &models, const std::vector<double> &weights) const = 0;
    // return minimal sample size required for non-minimal estimation.
    virtual int getMinimumRequiredSampleSize() const = 0;
    // return maximum number of possible solutions.
    virtual int getMaxNumberOfSolutions () const = 0;
    virtual int estimate (const std::vector<bool>& mask, std::vector<Mat>& models,
            const std::vector<double>& weights) = 0;
    virtual void enforceRankConstraint (bool enforce) = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<HomographyNonMinimalSolver> create(const Mat &points_, bool use_ge_=false);
    static Ptr<HomographyNonMinimalSolver> create(const Mat &points_, const Matx33d &T1, const Matx33d &T2, bool use_ge);
};

//-------------------------- FUNDAMENTAL MATRIX -----------------------
class EpipolarNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<EpipolarNonMinimalSolver> create(const Mat &points_, bool is_fundamental);
    static Ptr<EpipolarNonMinimalSolver> create(const Mat &points_, const Matx33d &T1, const Matx33d &T2, bool use_ge);
};

//-------------------------- ESSENTIAL MATRIX -----------------------
class EssentialNonMinimalSolverViaF : public NonMinimalSolver {
public:
    static Ptr<EssentialNonMinimalSolverViaF> create(const Mat &points_, const cv::Mat &K1, const Mat &K2);
};

class EssentialNonMinimalSolverViaT : public NonMinimalSolver {
public:
    static Ptr<EssentialNonMinimalSolverViaT> create(const Mat &points_);
};

//-------------------------- PNP -----------------------
class PnPNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<PnPNonMinimalSolver> create(const Mat &points);
};

class DLSPnP : public NonMinimalSolver {
public:
    static Ptr<DLSPnP> create(const Mat &points_, const Mat &calib_norm_pts, const Mat &K);
};

//-------------------------- AFFINE -----------------------
class AffineNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<AffineNonMinimalSolver> create(const Mat &points, InputArray T1, InputArray T2);
};

class LarssonOptimizer : public NonMinimalSolver {
public:
    static Ptr<LarssonOptimizer> create(const Mat &calib_points_, const Matx33d &K1_, const Matx33d &K2_, int max_iters_, bool is_fundamental_);
};

//-------------------------- SE/SIM -----------------------
class SE2NonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<SE2NonMinimalSolver> create(const Mat &points_);
};
class SIM2NonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<SIM2NonMinimalSolver> create(const Mat &points_);
};
class SO3NonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<SO3NonMinimalSolver> create(const Mat &points_);
};
class SE3NonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<SE3NonMinimalSolver> create(const Mat &points_);
};
class SIM3NonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<SIM3NonMinimalSolver> create(const Mat &points_);
};

////////////////////////////////////////// SCORE ///////////////////////////////////////////
class Score {
public:
    int inlier_number;
    float score;
    Score () { // set worst case
        inlier_number = 0;
        score = std::numeric_limits<float>::max();
    }
    Score (int inlier_number_, float score_) { // copy constructor
        inlier_number = inlier_number_;
        score = score_;
    }
    // Compare two scores. Objective is minimization of score. Lower score is better.
    inline bool isBetter (const Score &score2) const {
        return score < score2.score;
    }
};

class GammaValues : public Algorithm {
public:
    virtual ~GammaValues() override = default;
    static Ptr<GammaValues> create(int DoF, int max_size_table=500);
    virtual const std::vector<double> &getCompleteGammaValues() const = 0;
    virtual const std::vector<double> &getIncompleteGammaValues() const = 0;
    virtual const std::vector<double> &getGammaValues() const = 0;
    virtual double getScaleOfGammaCompleteValues () const = 0;
    virtual double getScaleOfGammaValues () const = 0;
    virtual int getTableSize () const = 0;
};

////////////////////////////////////////// QUALITY ///////////////////////////////////////////
class Quality : public Algorithm {
public:
    virtual ~Quality() override = default;
    /*
     * Calculates number of inliers and score of the @model.
     * return Score with calculated inlier_number and score.
     * @model: Mat current model, e.g., H matrix.
     */
    virtual Score getScore (const Mat &model) const = 0;
    virtual Score getScore (const std::vector<float>& errors) const = 0;
    // get @inliers of the @model. Assume threshold is given
    // @inliers must be preallocated to maximum points size.
    virtual int getInliers (const Mat &model, std::vector<int> &inliers) const = 0;
    // get @inliers of the @model for given threshold
    virtual int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const = 0;
    // Set the best score, so evaluation of the model can terminate earlier
    virtual void setBestScore (float best_score_) = 0;
    // set @inliers_mask: true if point i is inlier, false - otherwise.
    virtual int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const = 0;
    virtual int getPointsSize() const = 0;
    virtual double getThreshold () const = 0;
    virtual Ptr<Error> getErrorFnc () const = 0;
    static int getInliers (const Ptr<Error> &error, const Mat &model,
            std::vector<bool> &inliers_mask, double threshold);
    static int getInliers (const Ptr<Error> &error, const Mat &model,
            std::vector<int>  &inliers,      double threshold);
    static int getInliers (const std::vector<float> &errors, std::vector<bool> &inliers,
            double threshold);
    static int getInliers (const std::vector<float> &errors, std::vector<int> &inliers,
            double threshold);
    Score selectBest (const std::vector<Mat> &models, int num_models, Mat &best) {
        if (num_models == 0) return {};
        int best_idx = 0;
        Score best_score = getScore(models[0]);
        for (int i = 1; i < num_models; i++) {
            const auto sc = getScore(models[i]);
            if (sc.isBetter(best_score)) {
                best_score = sc;
                best_idx = i;
            }
        }
        models[best_idx].copyTo(best);
        return best_score;
    }
};

// RANSAC (binary) quality
class RansacQuality : public Quality {
public:
    static Ptr<RansacQuality> create(int points_size_, double threshold_,const Ptr<Error> &error_);
};

// M-estimator quality - truncated Squared error
class MsacQuality : public Quality {
public:
    static Ptr<MsacQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_, double k_msac=2.25);
};

// Marginlizing Sample Consensus quality, D. Barath et al.
class MagsacQuality : public Quality {
public:
    static Ptr<MagsacQuality> create(double maximum_thr, int points_size_,const Ptr<Error> &error_,
                             const Ptr<GammaValues> &gamma_generator,
                             double tentative_inlier_threshold_, int DoF, double sigma_quantile,
                             double upper_incomplete_of_sigma_quantile);
};

// Least Median of Squares Quality
class LMedsQuality : public Quality {
public:
    static Ptr<LMedsQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_);
};

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// DEGENERACY //////////////////////////////////
class Degeneracy : public Algorithm {
private:
    Mat homogr;
public:
    virtual ~Degeneracy() override = default;
    /*
     * Check if sample causes degenerate configurations.
     * For example, test if points are collinear.
     */
    virtual bool isSampleGood (const std::vector<int>& sample) const {
        CV_UNUSED(sample);
        return true;
    }
    /*
     * Check if model satisfies constraints.
     * For example, test if epipolar geometry satisfies oriented constraint.
     */
    virtual bool isModelValid (const Mat& model, const std::vector<int>& sample) const {
        CV_UNUSED(model);
        CV_UNUSED(sample);
        return true;
    }
    /*
     * Fix degenerate model.
     * Return true if model is degenerate, false - otherwise
     */
    virtual bool recoverIfDegenerate (const std::vector<int> &sample,const Mat &best_model, const Score &score,
                  Mat &non_degenerate_model, Score &non_degenerate_model_score) {
        CV_UNUSED(sample);
        CV_UNUSED(best_model);
        CV_UNUSED(score);
        CV_UNUSED(non_degenerate_model);
        CV_UNUSED(non_degenerate_model_score);
        return false;
    }
};

class EpipolarGeometryDegeneracy : public Degeneracy {
public:
    static void recoverRank (Mat &model, bool is_fundamental_mat);
    static Ptr<EpipolarGeometryDegeneracy> create (const Mat &points_, int sample_size_);
};

class EssentialDegeneracy : public EpipolarGeometryDegeneracy {
public:
    static Ptr<EssentialDegeneracy>create (const Mat &points, int sample_size);
};

class HomographyDegeneracy : public Degeneracy {
public:
    static Ptr<HomographyDegeneracy> create(const Mat &points_);
};

class FundamentalDegeneracyViaE : public EpipolarGeometryDegeneracy {
public:
    static Ptr<FundamentalDegeneracyViaE> create (const Ptr<Quality> &quality, const Mat &pts,
            const Mat &calib_pts, const Matx33d &K1, const Matx33d &K2, bool is_f_objective);
};

class FundamentalDegeneracy : public EpipolarGeometryDegeneracy {
public:
    virtual void setPrincipalPoint (double px_, double py_) = 0;
    virtual void setPrincipalPoint (double px_, double py_, double px2_, double py2_) = 0;
    virtual bool verifyFundamental (const Mat &F_best, const Score &F_score, const std::vector<bool> &inliers_mask, cv::Mat &F_new, Score &new_score) = 0;
    static Ptr<FundamentalDegeneracy> create (int state, const Ptr<Quality> &quality_,
        const Mat &points_, int sample_size_, int max_iters_plane_and_parallax,
        double homography_threshold, double f_inlier_thr_sqr, const Mat true_K1=Mat(), const Mat true_K2=Mat());
};

/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// ESTIMATOR //////////////////////////////////
class Estimator : public Algorithm{
public:
    virtual int estimateModelNonMinimalSample (const Mat& model, const std::vector<int> &sample, int sample_size, std::vector<Mat>
        &models, const std::vector<double> &weights) const {
        CV_UNUSED(model);
        return estimateModelNonMinimalSample(sample, sample_size, models, weights);
    }
    /*
     * Estimate models with minimal solver.
     * Return number of valid solutions after estimation.
     * Return models accordingly to number of solutions.
     * Note, vector of models must allocated before.
     * Note, not all degenerate tests are included in estimation.
     */
    virtual int
    estimateModels (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;
    /*
     * Estimate model with non-minimal solver.
     * Return number of valid solutions after estimation.
     * Note, not all degenerate tests are included in estimation.
     */
    virtual int
    estimateModelNonMinimalSample (const std::vector<int> &sample, int sample_size,
                       std::vector<Mat> &models, const std::vector<double> &weights) const = 0;
    // return minimal sample size required for minimal estimation.
    virtual int getMinimalSampleSize () const = 0;
    // return minimal sample size required for non-minimal estimation.
    virtual int getNonMinimalSampleSize () const = 0;
    // return maximum number of possible solutions of minimal estimation.
    virtual int getMaxNumSolutions () const = 0;
    // return maximum number of possible solutions of non-minimal estimation.
    virtual int getMaxNumSolutionsNonMinimal () const = 0;
    virtual void enforceRankConstraint (bool enforce) = 0;
};

class HomographyEstimator : public Estimator {
public:
    static Ptr<HomographyEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

class FundamentalEstimator : public Estimator {
public:
    static Ptr<FundamentalEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

class EssentialEstimator : public Estimator {
public:
    static Ptr<EssentialEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

class AffineEstimator : public Estimator {
public:
    static Ptr<AffineEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_);
};

class PnPEstimator : public Estimator {
public:
    static Ptr<PnPEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// MODEL VERIFIER ////////////////////////////////////
class ModelVerifier : public Algorithm {
public:
    virtual ~ModelVerifier() override = default;
    // Return true if model is good, false - otherwise.
    virtual bool isModelGood(const Mat &model, Score &score) = 0;
    // update verifier by given inlier number
    virtual void update (const Score &score, int iteration) = 0;
    virtual void reset() = 0;
    virtual void updateSPRT (double time_model_est, double time_corr_ver, double new_avg_models, double new_delta, double new_epsilon, const Score &best_score) = 0;
    static Ptr<ModelVerifier> create(const Ptr<Quality> &qualtiy);
};

struct SPRT_history {
    /*
     * delta:
     * The probability of a data point being consistent
     * with a 'bad' model is modeled as a probability of
     * a random event with Bernoulli distribution with parameter
     * delta : p(1|Hb) = delta.

     * epsilon:
     * The probability p(1|Hg) = epsilon
     * that any randomly chosen data point is consistent with a 'good' model
     * is approximated by the fraction of inliers epsilon among the data
     * points

     * A is the decision threshold, the only parameter of the Adapted SPRT
     */
    double epsilon, delta, A;
    // number of samples processed by test
    int tested_samples; // k
    SPRT_history () {
        tested_samples = 0;
    }
};

///////////////////////////////// SPRT VERIFIER /////////////////////////////////////////
/*
* Matas, Jiri, and Ondrej Chum. "Randomized RANSAC with sequential probability ratio test."
* Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1. Vol. 2. IEEE, 2005.
*/
class AdaptiveSPRT : public ModelVerifier {
public:
    virtual const std::vector<SPRT_history> &getSPRTvector () const = 0;
    virtual int avgNumCheckedPts () const = 0;
    static Ptr<AdaptiveSPRT> create (int state, const Ptr<Quality> &quality, int points_size_,
         double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
         double time_sample, double avg_num_models, ScoreMethod score_type_,
         double k_mlesac, bool is_adaptive = true);
};

//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SAMPLER ///////////////////////////////////////
class Sampler : public Algorithm {
public:
    virtual ~Sampler() override = default;
    // set new points size
    virtual void setNewPointsSize (int points_size) = 0;
    // generate sample. Fill @sample with indices of points.
    virtual void generateSample (std::vector<int> &sample) = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// NEIGHBORHOOD GRAPH /////////////////////////////////////////
class NeighborhoodGraph : public Algorithm {
public:
    virtual ~NeighborhoodGraph() override = default;
    // Return neighbors of the point with index @point_idx_ in the graph.
    virtual const std::vector<int> &getNeighbors(int point_idx_) const = 0;
    virtual const std::vector<std::vector<int>> &getGraph () const = 0;
};

class RadiusSearchNeighborhoodGraph : public NeighborhoodGraph {
public:
    static Ptr<RadiusSearchNeighborhoodGraph> create (const Mat &points, int points_size,
            double radius_, int flann_search_params, int num_kd_trees);
};

class FlannNeighborhoodGraph : public NeighborhoodGraph {
public:
    static Ptr<FlannNeighborhoodGraph> create(const Mat &points, int points_size,
      int k_nearest_neighbors_, bool get_distances, int flann_search_params, int num_kd_trees);
    virtual const std::vector<double> &getNeighborsDistances (int idx) const = 0;
};

class GridNeighborhoodGraph : public NeighborhoodGraph {
public:
    static Ptr<GridNeighborhoodGraph> create(const Mat &points, int points_size,
            int cell_size_x_img1_, int cell_size_y_img1_,
            int cell_size_x_img2_, int cell_size_y_img2_, int max_neighbors);
};

////////////////////////////////////// UNIFORM SAMPLER ////////////////////////////////////////////
class UniformSampler : public Sampler {
public:
    static Ptr<UniformSampler> create(int state, int sample_size_, int points_size_);
};

/////////////////////////////////// PROSAC (SIMPLE) SAMPLER ///////////////////////////////////////
class ProsacSimpleSampler : public Sampler {
public:
    static Ptr<ProsacSimpleSampler> create(int state, int points_size_, int sample_size_,
           int max_prosac_samples_count);
};

////////////////////////////////////// PROSAC SAMPLER ////////////////////////////////////////////
class ProsacSampler : public Sampler {
public:
    static Ptr<ProsacSampler> create(int state, int points_size_, int sample_size_,
                                     int growth_max_samples);
    // return number of samples generated (for prosac termination).
    virtual int getKthSample () const = 0;
    // return constant reference of growth function of prosac sampler (for prosac termination)
    virtual const std::vector<int> &getGrowthFunction () const = 0;
    virtual void setTerminationLength (int termination_length) = 0;
};

////////////////////////// NAPSAC (N adjacent points sample consensus) SAMPLER ////////////////////
class NapsacSampler : public Sampler {
public:
    static Ptr<NapsacSampler> create(int state, int points_size_, int sample_size_,
          const Ptr<NeighborhoodGraph> &neighborhood_graph_);
};

////////////////////////////////////// P-NAPSAC SAMPLER /////////////////////////////////////////
class ProgressiveNapsac : public Sampler {
public:
    static Ptr<ProgressiveNapsac> create(int state, int points_size_, int sample_size_,
            const std::vector<Ptr<NeighborhoodGraph>> &layers, int sampler_length);
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// TERMINATION ///////////////////////////////////////////
class Termination : public Algorithm {
public:
    // update termination object by given @model and @inlier number.
    // and return maximum number of predicted iteration
    virtual int update(const Mat &model, int inlier_number) const = 0;
};

//////////////////////////////// STANDARD TERMINATION ///////////////////////////////////////////
class StandardTerminationCriteria : public Termination {
public:
    static Ptr<StandardTerminationCriteria> create(double confidence, int points_size_,
               int sample_size_, int max_iterations_);
};

///////////////////////////////////// SPRT TERMINATION //////////////////////////////////////////
class SPRTTermination : public Termination {
public:
    static Ptr<SPRTTermination> create(const Ptr<AdaptiveSPRT> &sprt,
               double confidence, int points_size_, int sample_size_, int max_iterations_);
};

///////////////////////////// PROGRESSIVE-NAPSAC-SPRT TERMINATION /////////////////////////////////
class SPRTPNapsacTermination : public Termination {
public:
    static Ptr<SPRTPNapsacTermination> create(const Ptr<AdaptiveSPRT> &
        sprt, double confidence, int points_size_, int sample_size_,
        int max_iterations_, double relax_coef_);
};

////////////////////////////////////// PROSAC TERMINATION /////////////////////////////////////////
class ProsacTerminationCriteria : public Termination {
public:
    virtual const std::vector<int> &getNonRandomInliers () const = 0;
    virtual int updateTerminationLength (const Mat &model, int inliers_size, int &found_termination_length) const = 0;
    static Ptr<ProsacTerminationCriteria> create(const Ptr<ProsacSampler> &sampler_,
         const Ptr<Error> &error_, int points_size_, int sample_size, double confidence,
         int max_iters, int min_termination_length, double beta, double non_randomness_phi,
         double inlier_thresh, const std::vector<int> &non_rand_inliers);
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// UTILS ////////////////////////////////////////////////
namespace Utils {
    void densitySort (const Mat &points, int knn, Mat &sorted_points, std::vector<int> &sorted_mask);
    /*
     * calibrate points: [x'; 1] = K^-1 [x; 1]
     * @points is matrix N x 4.
     * @norm_points is output matrix N x 4 with calibrated points.
     */
    void calibratePoints (const Mat &K1, const Mat &K2, const Mat &points, Mat &norm_points);
    void calibrateAndNormalizePointsPnP (const Mat &K, const Mat &pts, Mat &calib_norm_pts);
    void normalizeAndDecalibPointsPnP (const Mat &K, Mat &pts, Mat &calib_norm_pts);
    void decomposeProjection (const Mat &P, Matx33d &K_, Matx33d &R, Vec3d &t, bool same_focal=false);
    double getCalibratedThreshold (double threshold, const Mat &K1, const Mat &K2);
    float findMedian (std::vector<float> &array);
    double intersectionOverUnion (const std::vector<bool> &a, const std::vector<bool> &b);
    void triangulatePoints (const Mat &points, const Mat &E_, const Mat &K1, const Mat &K2, Mat &points3D, Mat &R, Mat &t,
        std::vector<bool> &good_mask, std::vector<double> &depths1, std::vector<double> &depths2);
    void triangulatePoints (const Mat &E, const Mat &points1, const Mat &points2,  Mat &corr_points1, Mat &corr_points2,
               const Mat &K1, const Mat &K2, Mat &points3D, Mat &R, Mat &t, const std::vector<bool> &good_point_mask);
    int triangulatePointsRt (const Mat &points, Mat &points3D, const Mat &K1_, const Mat &K2_,
        const cv::Mat &R, const cv::Mat &t_vec, std::vector<bool> &good_mask, std::vector<double> &depths1, std::vector<double> &depths2);
    int decomposeHomography (const Matx33d &Hnorm, std::vector<Matx33d> &R, std::vector<Vec3d> &t);
    double getPoissonCDF (double lambda, int tentative_inliers);
    void getClosePoints (const cv::Mat &points, std::vector<std::vector<int>> &close_points, float close_thr_sqr);
    Vec3d getLeftEpipole (const Mat &F);
    Vec3d getRightEpipole (const Mat &F);
    int removeClosePoints (const Mat &points, Mat &new_points, float thr);
}
namespace Math {
    // return skew symmetric matrix
    Matx33d getSkewSymmetric(const Vec3d &v_);
    // eliminate matrix with m rows and n columns to be upper triangular.
    bool eliminateUpperTriangular (std::vector<double> &a, int m, int n);
    Matx33d rotVec2RotMat (const Vec3d &v);
    Vec3d rotMat2RotVec (const Matx33d &R);
}

class SolverPoly: public Algorithm {
public:
    virtual int getRealRoots (const std::vector<double> &coeffs, std::vector<double> &real_roots) = 0;
    static Ptr<SolverPoly> create();
};

///////////////////////////////////////// RANDOM GENERATOR /////////////////////////////////////
class RandomGenerator : public Algorithm {
public:
    virtual ~RandomGenerator() override = default;
    // interval is <0, max_range);
    virtual void resetGenerator (int max_range) = 0;
    // return sample filled with random numbers
    virtual void generateUniqueRandomSet (std::vector<int> &sample) = 0;
    // fill @sample of size @subset_size with random numbers in range <0, @max_range)
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int subset_size,
                                                                    int max_range) = 0;
    // fill @sample of size @sample.size() with random numbers in range <0, @max_range)
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int max_range) = 0;
    // return subset=sample size
    virtual void setSubsetSize (int subset_sz) = 0;
    virtual int getSubsetSize () const = 0;
    // return random number from <0, max_range), where max_range is from constructor
    virtual int getRandomNumber () = 0;
    // return random number from <0, max_rng)
    virtual int getRandomNumber (int max_rng) = 0;
    virtual const std::vector<int> &generateUniqueRandomSubset (std::vector<int> &array1,
            int size1) = 0;
};

class UniformRandomGenerator : public RandomGenerator {
public:
    static Ptr<UniformRandomGenerator> create (int state);
    static Ptr<UniformRandomGenerator> create (int state, int max_range, int subset_size_);
};

class WeightFunction : public Algorithm {
public:
    virtual int getInliersWeights (const std::vector<float> &errors, std::vector<int> &inliers, std::vector<double> &weights) const = 0;
    virtual int getInliersWeights (const std::vector<float> &errors, std::vector<int> &inliers, std::vector<double> &weights, double thr_sqr) const = 0;
    virtual double getThreshold() const = 0;
};

class GaussWeightFunction : public WeightFunction {
public:
     static Ptr<GaussWeightFunction> create (double thr, double sigma, double outlier_prob);
};

class MagsacWeightFunction : public WeightFunction {
public:
    static Ptr<MagsacWeightFunction> create (const Ptr<GammaValues> &gamma_generator_,
            int DoF_, double upper_incomplete_of_sigma_quantile, double C_, double max_sigma_);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// LOCAL OPTIMIZATION /////////////////////////////////////////
class LocalOptimization : public Algorithm {
public:
    virtual ~LocalOptimization() override = default;
    /*
     * Refine so-far-the-best RANSAC model in local optimization step.
     * @best_model: so-far-the-best model
     * @new_model: output refined new model.
     * @new_model_score: score of @new_model.
     * Returns bool if model was refined successfully, false - otherwise
     */
    virtual bool refineModel (const Mat &best_model, const Score &best_model_score,
                              Mat &new_model, Score &new_model_score) = 0;
    virtual void setCurrentRANSACiter (int /*ransac_iter*/) {}
    virtual int getNumLOoptimizations () const { return 0; }
};

//////////////////////////////////// GRAPH CUT LO ////////////////////////////////////////
class GraphCut : public LocalOptimization {
public:
    static Ptr<GraphCut>
    create(const Ptr<Estimator> &estimator_,
           const Ptr<Quality> &quality_, const Ptr<NeighborhoodGraph> &neighborhood_graph_,
           const Ptr<RandomGenerator> &lo_sampler_, double threshold_,
           double spatial_coherence_term, int gc_iters, Ptr<Termination> termination_= nullptr);
};

//////////////////////////////////// INNER + ITERATIVE LO ///////////////////////////////////////
class InnerIterativeLocalOptimization : public LocalOptimization {
public:
    static Ptr<InnerIterativeLocalOptimization>
    create(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
           const Ptr<RandomGenerator> &lo_sampler_, int pts_size, double threshold_,
           bool is_iterative_, int lo_iter_sample_size_, int lo_inner_iterations,
           int lo_iter_max_iterations, double threshold_multiplier);
};

class SimpleLocalOptimization : public LocalOptimization {
public:
    static Ptr<SimpleLocalOptimization> create
        (const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &estimator_,
         const Ptr<Termination> termination_, const Ptr<RandomGenerator> &random_gen,
         const Ptr<WeightFunction> weight_fnc_,
         int max_lo_iters_, double inlier_thr_sqr, bool updated_lo=false);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// FINAL MODEL POLISHER //////////////////////////////////////
class FinalModelPolisher : public Algorithm {
public:
    virtual ~FinalModelPolisher() override = default;
    /*
     * Polish so-far-the-best RANSAC model in the end of RANSAC.
     * @model: input final RANSAC model.
     * @new_model: output polished model.
     * @new_score: score of output model.
     * Return true if polishing was successful, false - otherwise.
     */
    virtual bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
            Mat &new_model, Score &new_model_score) = 0;
};

class NonMinimalPolisher : public FinalModelPolisher {
public:
    static Ptr<NonMinimalPolisher> create(const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &solver_,
            Ptr<WeightFunction> weight_fnc_, int max_iters_, double iou_thr_);
};
class CovarianceSolver : public NonMinimalSolver {
public:
    ~CovarianceSolver() override = default;
    int estimate (const std::vector<int> &, int , std::vector<Mat> &,
          const std::vector<double> &) const override {
        assert(false && "not implemeted!");
        return 0;
    }
    virtual int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &weights) override = 0;
    virtual void reset() = 0;
};
class CovarianceEpipolarSolver : public CovarianceSolver {
public:
    static Ptr<CovarianceEpipolarSolver> create (const Mat &points, bool is_fundamental);
    static Ptr<CovarianceEpipolarSolver> create (const Mat &points, const Matx33d &T1, const Matx33d &T2);
};
class CovarianceHomographySolver : public CovarianceSolver {
public:
    static Ptr<CovarianceHomographySolver> create (const Mat &points);
    static Ptr<CovarianceHomographySolver> create (const Mat &points, const Matx33d &T1, const Matx33d &T2);
};
class CovarianceAffineSolver : public CovarianceSolver {
public:
    static Ptr<CovarianceAffineSolver> create (const Mat &points, const Matx33d &T1, const Matx33d &T2);
    static Ptr<CovarianceAffineSolver> create (const Mat &points);
};

/////////////////////////////////// POSE LIB ////////////////////////////////////////
struct CameraPose {
    cv::Matx33d R;
    cv::Vec3d t;
    double alpha = 1.0; // either focal length or scale
};
typedef std::vector<CameraPose> CameraPoseVector;

struct BundleOptions {
    int max_iterations = 100;
    enum LossType {
        MLESAC,
    } loss_type = LossType::MLESAC; // CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-8;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
};

bool satisfyCheirality (const cv::Matx33d& R, const cv::Vec3d &t, const cv::Vec3d &x1, const cv::Vec3d &x2);

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_relpose(const cv::Mat &correspondences_,
                   const std::vector<int> &sample_,
                   const int sample_size_,
                   CameraPose *pose,
                   const BundleOptions &opt = BundleOptions(),
                   const double *weights = nullptr);

/////////////////////////////////// RANSAC OUTPUT ///////////////////////////////////
class RansacOutput : public Algorithm {
public:
    virtual ~RansacOutput() override = default;
    static Ptr<RansacOutput> create(const Mat &model_,
        const std::vector<bool> &inliers_mask_, int number_inliers_,
        int number_iterations_, ModelConfidence conf, const std::vector<float> &errors_);

    // Return inliers' indices. size of vector = number of inliers
    virtual const std::vector<int > &getInliers() = 0;
    // Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
    virtual const std::vector<bool> &getInliersMask() const = 0;
    virtual int getNumberOfInliers() const = 0;
    virtual const Mat &getModel() const = 0;
    virtual int getNumberOfIters() const = 0;
    virtual ModelConfidence getConfidence() const = 0;
    virtual const std::vector<float> &getResiduals() const = 0;
};

////////////////////////////////////////////// MODEL /////////////////////////////////////////////

class Model : public Algorithm {
public:
    virtual bool isFundamental () const = 0;
    virtual bool isHomography () const = 0;
    virtual bool isEssential () const = 0;
    virtual bool isPnP () const = 0;
    virtual bool isPtsetReg () const = 0;

    // getters
    virtual int getSampleSize () const = 0;
    virtual bool isParallel() const = 0;
    virtual PolishingMethod getFinalPolisher () const = 0;
    virtual LocalOptimMethod getLO () const = 0;
    virtual ErrorMetric getError () const = 0;
    virtual EstimationMethod getEstimator () const = 0;
    virtual ScoreMethod getScore () const = 0;
    virtual int getMaxIters () const = 0;
    virtual double getConfidence () const = 0;
    virtual double getThreshold () const = 0;
    virtual VerificationMethod getVerifier () const = 0;
    virtual SamplingMethod getSampler () const = 0;
    virtual double getTimeForModelEstimation () const = 0;
    virtual double getSPRTdelta () const = 0;
    virtual double getSPRTepsilon () const = 0;
    virtual double getSPRTavgNumModels () const = 0;
    virtual NeighborSearchMethod getNeighborsSearch () const = 0;
    virtual int getKNN () const = 0;
    virtual int getCellSize () const = 0;
    virtual int getGraphRadius() const = 0;
    virtual double getRelaxCoef () const = 0;
    virtual bool isNonRandomnessTest () const = 0;

    virtual int getFinalLSQIterations () const = 0;
    virtual int getDegreesOfFreedom () const = 0;
    virtual double getSigmaQuantile () const = 0;
    virtual double getUpperIncompleteOfSigmaQuantile () const = 0;
    virtual double getLowerIncompleteOfSigmaQuantile () const = 0;
    virtual double getC () const = 0;
    virtual double getMaximumThreshold () const = 0;
    virtual double getGraphCutSpatialCoherenceTerm () const = 0;
    virtual double getKmlesac () const = 0;
    virtual int getLOSampleSize () const = 0;
    virtual int getLOThresholdMultiplier() const = 0;
    virtual int getLOIterativeSampleSize() const = 0;
    virtual int getLOIterativeMaxIters() const = 0;
    virtual int getLOInnerMaxIters() const = 0;
    virtual const std::vector<int> &getGridCellNumber () const = 0;
    virtual int getRandomGeneratorState () const = 0;
    virtual MethodSolver getRansacSolver () const = 0;
    virtual int getPlaneAndParallaxIters () const = 0;
    virtual int getLevMarqIters () const = 0;
    virtual int getLevMarqItersLO () const = 0;
    virtual bool isLarssonOptimization () const = 0;
    virtual int getProsacMaxSamples() const = 0;

    // setters
    virtual void setNonRandomnessTest (bool set) = 0;
    virtual void setLocalOptimization (LocalOptimMethod lo_) = 0;
    virtual void setKNearestNeighhbors (int knn_) = 0;
    virtual void setNeighborsType (NeighborSearchMethod neighbors) = 0;
    virtual void setCellSize (int cell_size_) = 0;
    virtual void setParallel (bool is_parallel) = 0;
    virtual void setVerifier (VerificationMethod verifier_) = 0;
    virtual void setPolisher (PolishingMethod polisher_) = 0;
    virtual void setError (ErrorMetric error_) = 0;
    virtual void setLOIterations (int iters) = 0;
    virtual void setLOIterativeIters (int iters) = 0;
    virtual void setLOSampleSize (int lo_sample_size) = 0;
    virtual void setRandomGeneratorState (int state) = 0;
    virtual void setFinalLSQ (int iters) = 0;

    virtual void maskRequired (bool required) = 0;
    virtual bool isMaskRequired () const = 0;
    static Ptr<Model> create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
         double confidence_=0.95, int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::SCORE_METHOD_MSAC);
};

double getLambda (std::vector<int> &supports, double cdf_thr, int points_size,
      int sample_size, bool is_indendent_inliers, int &min_non_random_inliers);

Mat findHomography(InputArray srcPoints, InputArray dstPoints, int method,
                   double ransacReprojThreshold, OutputArray mask,
                   const int maxIters, const double confidence);

Mat findFundamentalMat( InputArray points1, InputArray points2,
    int method, double ransacReprojThreshold, double confidence,
    int maxIters, OutputArray mask=noArray());

bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
         InputArray cameraMatrix, InputArray distCoeffs,
         OutputArray rvec, OutputArray tvec,
         bool useExtrinsicGuess, int iterationsCount,
         float reprojectionError, double confidence,
         OutputArray inliers, int flags);

Mat findEssentialMat( InputArray points1, InputArray points2,
                      InputArray cameraMatrix1,
                      int method, double prob,
                      double threshold, OutputArray mask,
                      int maxIters);

Mat estimateAffine2D(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

Mat estimateSE2(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

Mat estimateSIM2(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

Mat estimateSO3(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

Mat estimateSE3(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

Mat estimateSIM3(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

void saveMask (OutputArray mask, const std::vector<bool> &inliers_mask);
void setParameters (Ptr<Model> &params, EstimationMethod estimator, const UsacParams &usac_params,
        bool mask_need);
bool run (const Ptr<const Model> &params, InputArray points1, InputArray points2,
      Ptr<RansacOutput> &ransac_output, InputArray K1_, InputArray K2_,
      InputArray dist_coeff1, InputArray dist_coeff2);
}}

#endif //OPENCV_USAC_USAC_HPP
