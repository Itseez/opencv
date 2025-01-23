// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#endif

namespace cv { namespace usac {
class HomographyMinimalSolver4ptsImpl : public HomographyMinimalSolver4pts {
private:
    Mat points_mat;
    const bool use_ge;
public:
    explicit HomographyMinimalSolver4ptsImpl (const Mat &points_, bool use_ge_) :
        points_mat(points_), use_ge(use_ge_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }

    int estimate (const std::vector<int>& sample, std::vector<Mat> &models) const override {
        const float * points = points_mat.ptr<float>();
        int m = 8, n = 9;
        std::vector<double> A(72, 0);
        int cnt = 0;
        for (int i = 0; i < 4; i++) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl], y1 = points[smpl+1], x2 = points[smpl+2], y2 = points[smpl+3];

            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            cnt += 3; // skip zeros
            A[cnt++] = x2*x1;
            A[cnt++] = x2*y1;
            A[cnt++] = x2;

            cnt += 3;
            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            A[cnt++] = y2*x1;
            A[cnt++] = y2*y1;
            A[cnt++] = y2;
        }

        if (use_ge) {
            if (!Math::eliminateUpperTriangular(A, m, n))
                return 0;

            models = std::vector<Mat>{ Mat_<double>(3,3) };
            auto * h = (double *) models[0].data;
            h[8] = 1.;

            // start from the last row
            for (int i = m-1; i >= 0; i--) {
                double acc = 0;
                for (int j = i+1; j < n; j++)
                    acc -= A[i*n+j]*h[j];

                h[i] = acc / A[i*n+i];
                // due to numerical errors return 0 solutions
                if (std::isnan(h[i]))
                    return 0;
            }
        } else {
            Mat U, Vt, D;
            cv::Matx<double, 8, 9> A_svd(&A[0]);
            SVD::compute(A_svd, D, U, Vt, SVD::FULL_UV+SVD::MODIFY_A);
            models = std::vector<Mat> { Vt.row(Vt.rows-1).reshape(0, 3) };
        }
        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 4; }
};
Ptr<HomographyMinimalSolver4pts> HomographyMinimalSolver4pts::create(const Mat &points, bool use_ge) {
    return makePtr<HomographyMinimalSolver4ptsImpl>(points, use_ge);
}

class HomographyNonMinimalSolverImpl : public HomographyNonMinimalSolver {
private:
    Mat points_mat;
    const bool do_norm, use_ge;
    Ptr<NormTransform> normTr;
    Matx33d _T1, _T2;
public:
    explicit HomographyNonMinimalSolverImpl (const Mat &norm_points_, const Matx33d &T1, const Matx33d &T2, bool use_ge_) :
            points_mat(norm_points_), do_norm(false), use_ge(use_ge_), _T1(T1), _T2(T2)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }
    explicit HomographyNonMinimalSolverImpl (const Mat &points_, bool use_ge_) :
        points_mat(points_), do_norm(true), use_ge(use_ge_), normTr (NormTransform::create(points_))
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Matx33d T1, T2;
        Mat norm_points_;
        if (do_norm)
            normTr->getNormTransformation(norm_points_, sample, sample_size, T1, T2);
        const float * const npts = do_norm ? norm_points_.ptr<float>() : points_mat.ptr<float>();

        Mat H;
        if (use_ge) {
            double a1[8] = {0, 0, -1, 0, 0, 0, 0, 0},
                   a2[8] = {0, 0, 0, 0, 0, -1, 0, 0};
            std::vector<double> AtAb(72, 0); // 8x9
            if (weights.empty()) {
                for (int i = 0; i < sample_size; i++) {
                    const int idx = do_norm ? 4*i : 4*sample[i];
                    const double x1 = npts[idx], y1 = npts[idx+1], x2 = npts[idx+2], y2 = npts[idx+3];
                    a1[0] = -x1;
                    a1[1] = -y1;
                    a1[6] = x2*x1;
                    a1[7] = x2*y1;

                    a2[3] = -x1;
                    a2[4] = -y1;
                    a2[6] = y2*x1;
                    a2[7] = y2*y1;

                    // calculate covariance for eigen
                    for (int j = 0; j < 8; j++) {
                        for (int z = j; z < 8; z++)
                            AtAb[j * 9 + z] +=  a1[j]*a1[z] + a2[j]*a2[z];
                        AtAb[j * 9 + 8] += a1[j]*x2 + a2[j]*y2;
                    }
                }
            } else { // use weights
                for (int i = 0; i < sample_size; i++) {
                    const double weight = weights[i];
                    if (weight < FLT_EPSILON) continue;
                    const int idx = do_norm ? 4*i : 4*sample[i];
                    const double x1 = npts[idx], y1 = npts[idx+1], x2 = npts[idx+2], y2 = npts[idx+3];
                    const double minus_weight_times_x1 = -weight * x1,
                                 minus_weight_times_y1 = -weight * y1,
                                 weight_times_x2 =  weight * x2,
                                 weight_times_y2 =  weight * y2;

                    a1[0] = minus_weight_times_x1;
                    a1[1] = minus_weight_times_y1;
                    a1[2] = -weight;
                    a1[6] = weight_times_x2 * x1;
                    a1[7] = weight_times_x2 * y1;

                    a2[3] = minus_weight_times_x1;
                    a2[4] = minus_weight_times_y1;
                    a2[5] = -weight;
                    a2[6] = weight_times_y2 * x1;
                    a2[7] = weight_times_y2 * y1;

                    for (int j = 0; j < 8; j++) {
                        for (int z = j; z < 8; z++)
                            AtAb[j * 9 + z] +=  a1[j]*a1[z] + a2[j]*a2[z];
                        AtAb[j * 9 + 8] += a1[j]*weight_times_x2 + a2[j]*weight_times_y2;
                    }
                }
            }
            for (int j = 1; j < 8; j++)
                for (int z = 0; z < j; z++)
                    AtAb[j*9+z] = AtAb[z*9+j];
            if (!Math::eliminateUpperTriangular(AtAb, 8, 9))
                return 0;
            H = Mat_<double>(3,3);
            auto * h = (double *) H.data;
            h[8] = 1.;
            const int m = 8, n = 9;
            // start from the last row
            for (int i = m-1; i >= 0; i--) {
                double acc = 0;
                for (int j = i+1; j < n; j++)
                    acc -= AtAb[i*n+j]*h[j];
                h[i] = acc / AtAb[i*n+i];
                if (std::isnan(h[i]))
                    return 0; // numerical imprecision
            }
        } else {
            double a1[9] = {0, 0, -1, 0, 0, 0, 0, 0, 0},
                   a2[9] = {0, 0, 0, 0, 0, -1, 0, 0, 0}, AtA[81] = {0};
            if (weights.empty()) {
                for (int i = 0; i < sample_size; i++) {
                    const int smpl = do_norm ? 4*i : 4*sample[i];
                    const auto x1 = npts[smpl], y1 = npts[smpl+1], x2 = npts[smpl+2], y2 = npts[smpl+3];

                    a1[0] = -x1;
                    a1[1] = -y1;
                    a1[6] = x2*x1;
                    a1[7] = x2*y1;
                    a1[8] = x2;

                    a2[3] = -x1;
                    a2[4] = -y1;
                    a2[6] = y2*x1;
                    a2[7] = y2*y1;
                    a2[8] = y2;

                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
                }
            } else { // use weights
                for (int i = 0; i < sample_size; i++) {
                    const double weight = weights[i];
                    if (weight < FLT_EPSILON) continue;
                    const int smpl = do_norm ? 4*i : 4*sample[i];
                    const auto x1 = npts[smpl], y1 = npts[smpl+1], x2 = npts[smpl+2], y2 = npts[smpl+3];
                    const double minus_weight_times_x1 = -weight * x1,
                            minus_weight_times_y1 = -weight * y1,
                            weight_times_x2 =  weight * x2,
                            weight_times_y2 =  weight * y2;

                    a1[0] = minus_weight_times_x1;
                    a1[1] = minus_weight_times_y1;
                    a1[2] = -weight;
                    a1[6] = weight_times_x2 * x1;
                    a1[7] = weight_times_x2 * y1;
                    a1[8] = weight_times_x2;

                    a2[3] = minus_weight_times_x1;
                    a2[4] = minus_weight_times_y1;
                    a2[5] = -weight;
                    a2[6] = weight_times_y2 * x1;
                    a2[7] = weight_times_y2 * y1;
                    a2[8] = weight_times_y2;

                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
                }
            }
            // copy symmetric part of covariance matrix
            for (int j = 1; j < 9; j++)
                for (int z = 0; z < j; z++)
                    AtA[j*9+z] = AtA[z*9+j];

#ifdef HAVE_EIGEN
            H = Mat_<double>(3,3);
            // extract the last null-vector
            Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Eigen::Matrix<double, 9, 9>
                    (Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> (
                            (Eigen::Matrix<double, 9, 9> (AtA))).householderQ()).col(8);
#else
            Matx<double, 9, 9> Vt;
            Vec<double, 9> D;
            if (! eigen(Matx<double, 9, 9>(AtA), D, Vt)) return 0;
            H = Mat_<double>(3, 3, Vt.val + 72/*=8*9*/);
#endif
        }
        const auto * const h = (double *) H.data;
        const auto * const t1 = do_norm ? T1.val : _T1.val, * const t2 = do_norm ? T2.val : _T2.val;
        // H = T2^-1 H T1
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(h[0]/t2[0] - (h[6]*t2[2])/t2[0]),
                t1[0]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]), h[2]/t2[0] + t1[2]*(h[0]/t2[0] -
                (h[6]*t2[2])/t2[0]) + t1[5]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]) - (h[8]*t2[2])/t2[0],
                t1[0]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]), t1[0]*(h[4]/t2[0] - (h[7]*t2[5])/t2[0]),
                h[5]/t2[0] + t1[2]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]) + t1[5]*(h[4]/t2[0] -
                (h[7]*t2[5])/t2[0]) - (h[8]*t2[5])/t2[0], t1[0]*h[6], t1[0]*h[7],
                h[8] + h[6]*t1[2] + h[7]*t1[5])) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    int getMinimumRequiredSampleSize() const override { return 4; }
    int getMaxNumberOfSolutions () const override { return 1; }
    void enforceRankConstraint (bool /*enforce*/) override {}
};
Ptr<HomographyNonMinimalSolver> HomographyNonMinimalSolver::create(const Mat &points_, bool use_ge_) {
    return makePtr<HomographyNonMinimalSolverImpl>(points_, use_ge_);
}
Ptr<HomographyNonMinimalSolver> HomographyNonMinimalSolver::create(const Mat &points_, const Matx33d &T1, const Matx33d &T2, bool use_ge) {
    return makePtr<HomographyNonMinimalSolverImpl>(points_, T1, T2, use_ge);
}

class CovarianceHomographySolverImpl : public CovarianceHomographySolver {
private:
    Mat norm_pts;
    Matx33d T1, T2;
    float * norm_points;
    std::vector<bool> mask;
    int points_size;
    double covariance[81] = {0}, * t1, * t2;
public:
    explicit CovarianceHomographySolverImpl (const Mat &norm_points_, const Matx33d &T1_, const Matx33d &T2_)
            : norm_pts(norm_points_), T1(T1_), T2(T2_) {
        points_size = norm_points_.rows;
        norm_points = (float *) norm_pts.data;
        t1 = T1.val; t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }
    explicit CovarianceHomographySolverImpl (const Mat &points_) {
        points_size = points_.rows;
        // normalize points
        std::vector<int> sample(points_size);
        for (int i = 0; i < points_size; i++) sample[i] = i;
        const Ptr<NormTransform> normTr = NormTransform::create(points_);
        normTr->getNormTransformation(norm_pts, sample, points_size, T1, T2);
        norm_points = (float *) norm_pts.data;
        t1 = T1.val; t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }
    void reset () override {
        // reset covariance matrix to zero and mask to false
        std::fill(covariance, covariance+81, 0);
        std::fill(mask.begin(), mask.end(), false);
    }

    /*
     * Find homography using 4-point algorithm with covariance matrix and PCA
     */
    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &/*weights*/) override {
        double a1[9] = {0, 0, -1, 0, 0, 0, 0, 0, 0},
               a2[9] = {0, 0, 0, 0, 0, -1, 0, 0, 0};

        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                a1[0] = -x1;
                a1[1] = -y1;
                a1[6] = x2*x1;
                a1[7] = x2*y1;
                a1[8] = x2;

                a2[3] = -x1;
                a2[4] = -y1;
                a2[6] = y2*x1;
                a2[7] = y2*y1;
                a2[8] = y2;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] +=-a1[j]*a1[z] - a2[j]*a2[z];
                else
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                covariance[j*9+z] = covariance[z*9+j];

#ifdef HAVE_EIGEN
        Mat H = Mat_<double>(3,3);
        // extract the last null-vector
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Eigen::Matrix<double, 9, 9>
                (Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> (
                        (Eigen::Matrix<double, 9, 9> (covariance))).householderQ()).col(8);
#else
       Matx<double, 9, 9> Vt;
       Vec<double, 9> D;
       if (! eigen(Matx<double, 9, 9>(covariance), D, Vt)) return 0;
       Mat H = Mat_<double>(3, 3, Vt.val + 72/*=8*9*/);
#endif

        const auto * const h = (double *) H.data;
        // H = T2^-1 H T1
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(h[0]/t2[0] - (h[6]*t2[2])/t2[0]),
           t1[0]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]), h[2]/t2[0] + t1[2]*(h[0]/t2[0] -
           (h[6]*t2[2])/t2[0]) + t1[5]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]) - (h[8]*t2[2])/t2[0],
           t1[0]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]), t1[0]*(h[4]/t2[0] - (h[7]*t2[5])/t2[0]),
           h[5]/t2[0] + t1[2]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]) + t1[5]*(h[4]/t2[0] -
           (h[7]*t2[5])/t2[0]) - (h[8]*t2[5])/t2[0], t1[0]*h[6], t1[0]*h[7],
           h[8] + h[6]*t1[2] + h[7]*t1[5])) };

        return 1;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 4; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<CovarianceHomographySolver> CovarianceHomographySolver::create (const Mat &points) {
    return makePtr<CovarianceHomographySolverImpl>(points);
}
Ptr<CovarianceHomographySolver> CovarianceHomographySolver::create (const Mat &points, const Matx33d &T1, const Matx33d &T2) {
    return makePtr<CovarianceHomographySolverImpl>(points, T1, T2);
}

class AffineMinimalSolverImpl : public AffineMinimalSolver {
private:
    Mat points_mat;
public:
    explicit AffineMinimalSolverImpl (const Mat &points_) :
            points_mat(points_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }
    /*
        Affine transformation
        x1 y1 1 0  0  0   a   u1
        0  0  0 x1 y1 1   b   v1
        x2 y2 1 0  0  0   c   u2
        0  0  0 x2 y2 1 * d = v2
        x3 y3 1 0  0  0   e   u3
        0  0  0 x3 y3 1   f   v3
    */
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2];
        const float * points = points_mat.ptr<float>();
        const auto
                x1 = points[smpl1], y1 = points[smpl1+1], u1 = points[smpl1+2], v1 = points[smpl1+3],
                x2 = points[smpl2], y2 = points[smpl2+1], u2 = points[smpl2+2], v2 = points[smpl2+3],
                x3 = points[smpl3], y3 = points[smpl3+1], u3 = points[smpl3+2], v3 = points[smpl3+3];

        // covers degeneracy test when all 3 points are collinear.
        // In this case denominator will be 0
        double denominator = x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2;
        if (fabs(denominator) < FLT_EPSILON) // check if denominator is zero
            return 0;
        denominator = 1. / denominator;

        double a =  (u1*y2 - u2*y1 - u1*y3 + u3*y1 + u2*y3 - u3*y2) * denominator;
        double b = -(u1*x2 - u2*x1 - u1*x3 + u3*x1 + u2*x3 - u3*x2) * denominator;
        double c = u1 - a * x1 - b * y1; // ax1 + by1 + c = u1
        double d =  (v1*y2 - v2*y1 - v1*y3 + v3*y1 + v2*y3 - v3*y2) * denominator;
        double e = -(v1*x2 - v2*x1 - v1*x3 + v3*x1 + v2*x3 - v3*x2) * denominator;
        double f = v1 - d * x1 - e * y1; // dx1 + ey1 + f = v1

        models = std::vector<Mat>{ Mat(Matx33d(a, b, c, d, e, f, 0, 0, 1)) };
        return 1;
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<AffineMinimalSolver> AffineMinimalSolver::create(const Mat &points_) {
    return makePtr<AffineMinimalSolverImpl>(points_);
}

class AffineNonMinimalSolverImpl : public AffineNonMinimalSolver {
private:
    Mat points_mat;
    Ptr<NormTransform> normTr;
    Matx33d _T1, _T2;
    bool do_norm;
public:
    explicit AffineNonMinimalSolverImpl (const Mat &points_, InputArray T1, InputArray T2) :
            points_mat(points_) {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
        if (!T1.empty() && !T2.empty()) {
            do_norm = false;
            _T1 = T1.getMat();
            _T2 = T2.getMat();
        } else {
            do_norm = true;
            normTr = NormTransform::create(points_);
        }
    }

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;
        Matx33d T1, T2;
        Mat norm_points_;
        if (do_norm)
            normTr->getNormTransformation(norm_points_, sample, sample_size, T1, T2);
        const float * const pts = normTr ? norm_points_.ptr<float>() : points_mat.ptr<float>();

        // do Least Squares
        // Ax = b   ->  A^T Ax = A^T b
        // x = (A^T A)^-1 A^T b
        double AtA[36] = {0}, Ab[6] = {0};
        double r1[6] = {0, 0, 1, 0, 0, 0}; // row 1 of A
        double r2[6] = {0, 0, 0, 0, 0, 1}; // row 2 of A

        if (weights.empty())
            for (int p = 0; p < sample_size; p++) {
                const int idx = do_norm ? 4*p : 4*sample[p];
                const auto x1=pts[idx], y1=pts[idx+1], x2=pts[idx+2], y2=pts[idx+3];
                r1[0] = x1;
                r1[1] = y1;

                r2[3] = x1;
                r2[4] = y1;

                for (int j = 0; j < 6; j++) {
                    for (int z = j; z < 6; z++)
                        AtA[j * 6 + z] += r1[j] * r1[z] + r2[j] * r2[z];
                    Ab[j] += r1[j]*x2 + r2[j]*y2;
                }
            }
        else
            for (int p = 0; p < sample_size; p++) {
                const auto weight = weights[p];
                if (weight < FLT_EPSILON) continue;
                const int idx = do_norm ? 4*p : 4*sample[p];
                const double weight_times_x1 = weight * pts[idx  ],
                             weight_times_y1 = weight * pts[idx+1],
                             weight_times_x2 = weight * pts[idx+2],
                             weight_times_y2 = weight * pts[idx+3];

                r1[0] = weight_times_x1;
                r1[1] = weight_times_y1;
                r1[2] = weight;

                r2[3] = weight_times_x1;
                r2[4] = weight_times_y1;
                r2[5] = weight;

                for (int j = 0; j < 6; j++) {
                    for (int z = j; z < 6; z++)
                        AtA[j * 6 + z] += r1[j] * r1[z] + r2[j] * r2[z];
                    Ab[j] += r1[j]*weight_times_x2 + r2[j]*weight_times_y2;
                }
            }

        // copy symmetric part
        for (int j = 1; j < 6; j++)
            for (int z = 0; z < j; z++)
                AtA[j*6+z] = AtA[z*6+j];

        Vec6d aff;
        if (!solve(Matx66d(AtA), Vec6d(Ab), aff))
            return 0;
        const double h[9] = {aff(0), aff(1), aff(2),
                             aff(3), aff(4), aff(5),
                             0, 0, 1};
        const auto * const t1 = normTr ? T1.val : _T1.val, * const t2 = normTr ? T2.val : _T2.val;
        // A = T2^-1 A T1
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(h[0]/t2[0] - (h[6]*t2[2])/t2[0]),
                t1[0]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]), h[2]/t2[0] + t1[2]*(h[0]/t2[0] -
                (h[6]*t2[2])/t2[0]) + t1[5]*(h[1]/t2[0] - (h[7]*t2[2])/t2[0]) - (h[8]*t2[2])/t2[0],
                t1[0]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]), t1[0]*(h[4]/t2[0] - (h[7]*t2[5])/t2[0]),
                h[5]/t2[0] + t1[2]*(h[3]/t2[0] - (h[6]*t2[5])/t2[0]) + t1[5]*(h[4]/t2[0] -
                (h[7]*t2[5])/t2[0]) - (h[8]*t2[5])/t2[0], t1[0]*h[6], t1[0]*h[7],
                h[8] + h[6]*t1[2] + h[7]*t1[5])) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}

    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<AffineNonMinimalSolver> AffineNonMinimalSolver::create(const Mat &points_, InputArray T1, InputArray T2) {
    return makePtr<AffineNonMinimalSolverImpl>(points_, T1, T2);
}

class CovarianceAffineSolverImpl : public CovarianceAffineSolver {
private:
    Mat norm_pts;
    Matx33d T1, T2;
    float * norm_points;
    std::vector<bool> mask;
    int points_size;
    double covariance[36] = {0}, Ab[6] = {0}, * t1, * t2;
public:
    explicit CovarianceAffineSolverImpl (const Mat &norm_points_, const Matx33d &T1_, const Matx33d &T2_)
            : norm_pts(norm_points_), T1(T1_), T2(T2_) {
        points_size = norm_points_.rows;
        norm_points = (float *) norm_pts.data;
        t1 = T1.val; t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }
    explicit CovarianceAffineSolverImpl (const Mat &points_) {
        points_size = points_.rows;
        // normalize points
        std::vector<int> sample(points_size);
        for (int i = 0; i < points_size; i++) sample[i] = i;
        const Ptr<NormTransform> normTr = NormTransform::create(points_);
        normTr->getNormTransformation(norm_pts, sample, points_size, T1, T2);
        norm_points = (float *) norm_pts.data;
        t1 = T1.val; t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
    }
    void reset () override {
        std::fill(covariance, covariance+36, 0);
        std::fill(Ab, Ab+6, 0);
        std::fill(mask.begin(), mask.end(), false);
    }
    /*
     * Find affine transformation using linear method with covariance matrix and PCA
     */
    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &) override {
        double r1[6] = {0, 0, 1, 0, 0, 0}; // row 1 of A
        double r2[6] = {0, 0, 0, 0, 0, 1}; // row 2 of A
        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                r1[0] = x1;
                r1[1] = y1;

                r2[3] = x1;
                r2[4] = y1;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 6; j++) {
                        for (int z = j; z < 6; z++)
                            covariance[j*6+z] +=-r1[j]*r1[z] - r2[j]*r2[z];
                        Ab[j] +=-r1[j]*x2 - r2[j]*y2;
                    }
                else
                    for (int j = 0; j < 6; j++) {
                        for (int z = j; z < 6; z++)
                            covariance[j*6+z] += r1[j]*r1[z] + r2[j]*r2[z];
                        Ab[j] += r1[j]*x2 + r2[j]*y2;
                    }
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 6; j++)
            for (int z = 0; z < j; z++)
                covariance[j*6+z] = covariance[z*6+j];

        Vec6d aff;
        if (!solve(Matx66d(covariance), Vec6d(Ab), aff))
            return 0;
        double a[9] = { aff(0), aff(1), aff(2), aff(3), aff(4), aff(5), 0, 0, 1 };
        models = std::vector<Mat>{ Mat(Matx33d(t1[0]*(a[0]/t2[0] - (a[6]*t2[2])/t2[0]),
           t1[0]*(a[1]/t2[0] - (a[7]*t2[2])/t2[0]), a[2]/t2[0] + t1[2]*(a[0]/t2[0] -
           (a[6]*t2[2])/t2[0]) + t1[5]*(a[1]/t2[0] - (a[7]*t2[2])/t2[0]) - (a[8]*t2[2])/t2[0],
           t1[0]*(a[3]/t2[0] - (a[6]*t2[5])/t2[0]), t1[0]*(a[4]/t2[0] - (a[7]*t2[5])/t2[0]),
           a[5]/t2[0] + t1[2]*(a[3]/t2[0] - (a[6]*t2[5])/t2[0]) + t1[5]*(a[4]/t2[0] -
           (a[7]*t2[5])/t2[0]) - (a[8]*t2[5])/t2[0], t1[0]*a[6], t1[0]*a[7],
           a[8] + a[6]*t1[2] + a[7]*t1[5])) };
        return 1;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<CovarianceAffineSolver> CovarianceAffineSolver::create (const Mat &points, const Matx33d &T1, const Matx33d &T2) {
    return makePtr<CovarianceAffineSolverImpl>(points, T1, T2);
}
Ptr<CovarianceAffineSolver> CovarianceAffineSolver::create (const Mat &points) {
    return makePtr<CovarianceAffineSolverImpl>(points);
}

template<int dim>
Mat solve_weighted_umeyama(const Mat &points_, const bool is_scale, const bool is_centered, const std::vector<double> &weights_=std::vector<double>());

class SE2MinimalSolverImpl : public SE2MinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
public:
    explicit SE2MinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
            // , points((float *) points_.data)
            {}
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        Mat p_mat;
        std::vector<cv::Mat> arr{points_mat->row(sample[0]), points_mat->row(sample[1]),};
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<2>(p_mat, false, false) };
        return 1;
    }
    int getSampleSize() const override { return 2; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SE2MinimalSolver> SE2MinimalSolver::create(const Mat &points_) {
    return makePtr<SE2MinimalSolverImpl>(points_);
}

class SIM2MinimalSolverImpl : public SIM2MinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
public:
    explicit SIM2MinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
            // , points((float *) points_.data)
            {}
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        Mat p_mat;
        std::vector<cv::Mat> arr{points_mat->row(sample[0]), points_mat->row(sample[1]),};
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<2>(p_mat, true, false) };
        return 1;
    }
    int getSampleSize() const override { return 2; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SIM2MinimalSolver> SIM2MinimalSolver::create(const Mat &points_) {
    return makePtr<SIM2MinimalSolverImpl>(points_);
}

class SO3MinimalSolverImpl : public SO3MinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
public:
    explicit SO3MinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
            // , points((float *) points_.data)
            {}
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        Mat p_mat;
        std::vector<cv::Mat> arr{points_mat->row(sample[0]), points_mat->row(sample[1]), points_mat->row(sample[2]),};
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<3>(p_mat, false, true) };
        return 1;
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SO3MinimalSolver> SO3MinimalSolver::create(const Mat &points_) {
    return makePtr<SO3MinimalSolverImpl>(points_);
}

class SE3MinimalSolverImpl : public SE3MinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
public:
    explicit SE3MinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
            // , points((float *) points_.data)
            {}
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        Mat p_mat;
        std::vector<cv::Mat> arr{points_mat->row(sample[0]), points_mat->row(sample[1]), points_mat->row(sample[2]),};
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<3>(p_mat, false, false) };
        return 1;
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SE3MinimalSolver> SE3MinimalSolver::create(const Mat &points_) {
    return makePtr<SE3MinimalSolverImpl>(points_);
}

class SIM3MinimalSolverImpl : public SIM3MinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
public:
    explicit SIM3MinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
            // , points((float *) points_.data)
            {}
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        Mat p_mat;
        std::vector<cv::Mat> arr{points_mat->row(sample[0]), points_mat->row(sample[1]), points_mat->row(sample[2]),};
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<3>(p_mat, true, false) };
        return 1;
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SIM3MinimalSolver> SIM3MinimalSolver::create(const Mat &points_) {
    return makePtr<SIM3MinimalSolverImpl>(points_);
}

class SE2NonMinimalSolverImpl : public SE2NonMinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
    // const NormTransform<double> norm_transform;
public:
    explicit SE2NonMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
    /*, points((float*) points_.data), norm_transform(points_)*/ {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Mat p_mat;
        std::vector<cv::Mat> arr(sample_size);
        for (int p = 0; p < sample_size; p++) {
            arr[p] = points_mat->row(sample[p]);
        }
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<2>(p_mat, false, false, weights) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 2; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SE2NonMinimalSolver> SE2NonMinimalSolver::create(const Mat &points_) {
    return makePtr<SE2NonMinimalSolverImpl>(points_);
}

class SIM2NonMinimalSolverImpl : public SIM2NonMinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
    // const NormTransform<double> norm_transform;
public:
    explicit SIM2NonMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
    /*, points((float*) points_.data), norm_transform(points_)*/ {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Mat p_mat;
        std::vector<cv::Mat> arr(sample_size);
        for (int p = 0; p < sample_size; p++) {
            arr[p] = points_mat->row(sample[p]);
        }
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<2>(p_mat, true, false, weights) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 2; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SIM2NonMinimalSolver> SIM2NonMinimalSolver::create(const Mat &points_) {
    return makePtr<SIM2NonMinimalSolverImpl>(points_);
}

class SO3NonMinimalSolverImpl : public SO3NonMinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
    // const NormTransform<double> norm_transform;
public:
    explicit SO3NonMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
    /*, points((float*) points_.data), norm_transform(points_)*/ {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Mat p_mat;
        std::vector<cv::Mat> arr(sample_size);
        for (int p = 0; p < sample_size; p++) {
            arr[p] = points_mat->row(sample[p]);
        }
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<3>(p_mat, false, true, weights) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SO3NonMinimalSolver> SO3NonMinimalSolver::create(const Mat &points_) {
    return makePtr<SO3NonMinimalSolverImpl>(points_);
}

class SE3NonMinimalSolverImpl : public SE3NonMinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
    // const NormTransform<double> norm_transform;
public:
    explicit SE3NonMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
    /*, points((float*) points_.data), norm_transform(points_)*/ {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Mat p_mat;
        std::vector<cv::Mat> arr(sample_size);
        for (int p = 0; p < sample_size; p++) {
            arr[p] = points_mat->row(sample[p]);
        }
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<3>(p_mat, false, false, weights) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SE3NonMinimalSolver> SE3NonMinimalSolver::create(const Mat &points_) {
    return makePtr<SE3NonMinimalSolverImpl>(points_);
}

class SIM3NonMinimalSolverImpl : public SIM3NonMinimalSolver {
private:
    const Mat * points_mat;
    // const float * const points;
    // const NormTransform<double> norm_transform;
public:
    explicit SIM3NonMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_)
    /*, points((float*) points_.data), norm_transform(points_)*/ {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Mat p_mat;
        std::vector<cv::Mat> arr(sample_size);
        for (int p = 0; p < sample_size; p++) {
            arr[p] = points_mat->row(sample[p]);
        }
        cv::vconcat(arr, p_mat);
        models = std::vector<Mat>{ solve_weighted_umeyama<3>(p_mat, true, false, weights) };
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<SIM3NonMinimalSolver> SIM3NonMinimalSolver::create(const Mat &points_) {
    return makePtr<SIM3NonMinimalSolverImpl>(points_);
}


template<int dim>
Matx<double, dim+1, dim+1> buildRot(Matx<double, dim, dim> rot)
{
    auto res = Matx<double, dim+1, dim+1>::eye();
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            res(i, j) = rot(i, j);
        }
    }
    return res;
}

template<int dim>
Matx<double, dim+1, dim+1> buildTrans(Vec<double, dim> trans)
{
    auto res = Matx<double, dim+1, dim+1>::eye();
    for (int i = 0; i < dim; i++)
    {
        res(i, dim) = trans(i);
    }
    return res;
}

template<int dim>
Matx<double, dim+1, dim+1> buildScale(double scale)
{
    auto res = Matx<double, dim+1, dim+1>::eye() * scale;
    res (dim, dim) = 1;
    return res;
}

template<int dim>
Mat solve_weighted_umeyama(const Mat &points_, const bool is_scale, const bool is_centered, const std::vector<double> &weights)
{
    // DOI: 10.1109/34.88573
    // https://web.stanford.edu/class/cs273/refs/umeyama.pdf

    typedef Vec<double, dim> Vecdd;
    typedef Matx<double, dim, dim> Matxdd;
    const int N = points_.rows;

    bool noWeights = weights.empty();

    Mat convertedPoints;
    points_.convertTo(convertedPoints, CV_64F);
    // accumulating squared sums & dot products for further (co)variance calculation
    // wmean(X) = sum(w_i * x_i)/sum(w_i)
    // cov(X, Y) = sum(w_i*(x_i - wmean(X))^T*(y_i - wmean(Y)))/sum(w_i) =
    // sum(w_i*x_i^T*y_i)/sum(w_i) - wmean(X)^T*wmean(Y)
    // var(X) = tr(cov(X, X))
    Matxdd covSum;
    Vecdd wsum1, wsum2;
    double wsqsum1 = 0;
    double totalWeight = 0;
    for (int n = 0; n < N; n++)
    {
        double weight = noWeights ? 1. : weights[n];

        Vecdd sample1 = convertedPoints.at<Vecdd>(n, 0);
        Vecdd sample2 = convertedPoints.at<Vecdd>(n, 1);

        Matxdd covsq = sample2 * sample1.t();

        covSum += weight * covsq;
        wsum1  += weight * sample1;
        wsum2  += weight * sample2;
        wsqsum1 += weight * sample1.ddot(sample1);

        totalWeight += weight;
    }

    double invWeight = 1. / totalWeight;
    Vecdd center1 = wsum1 * invWeight * (is_centered?0:1);
    Vecdd center2 = wsum2 * invWeight * (is_centered?0:1);
    double sigma2 = wsqsum1 * invWeight - center1.ddot(center1);
    Matxdd ABt = covSum * invWeight - center2 * center1.t();

    Matxdd u, vt;
    Vecdd w;
    SVDecomp(ABt, w, u, vt, SVD::FULL_UV);

    Matxdd s = Matxdd::eye();
    double scaleDet = determinant(u) * determinant(vt);
    s(dim-1,dim-1) = scaleDet;

    // Scale
    w(dim-1) *= scaleDet;

    Matxdd rot = u * s * vt;

    double traceds = sum(w)[0];

    double scale = traceds / sigma2;

    auto hInv = buildTrans<dim>(-center1);
    auto hTsl = buildTrans<dim>( center2);
    auto hRot = buildRot<dim>(rot);
    auto hScl = buildScale<dim>(is_scale ? scale : 1.0);
    return Mat(hTsl * hScl * hRot * hInv)(Rect(0,0,dim+1,dim));
}

}}
