// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Zhangjie Chen <zhangjiec01@gmail.com>

#include "opencv2/core.hpp"
#include "opencv2/3d.hpp"
#include "queue"
#include "unordered_map"

namespace cv{
SpectralCluster::SpectralCluster() {
    pImpl = makePtr<Impl>(0.1f, 0.05f);
}

SpectralCluster::SpectralCluster(float delta_val, float eta_val) {
    CV_Assert(delta_val >= 0.f && delta_val <= 1.f);
    CV_Assert(eta_val >= 0.f && eta_val <= 1.f);
    pImpl = makePtr<Impl>(delta_val, eta_val);
}

class SpectralCluster::Impl {
public:
    float delta;
    float eta;

    Impl(float delta_val, float eta_val) : delta(delta_val), eta(eta_val) {}
    ~Impl() = default;

    static void getLaplacianMat(cv::Mat& in, cv::Mat& out);

    void getAdjacentDistanceMat(cv::Mat& out, const std::vector<cv::Point3f>& vertices,
                                const std::vector<std::vector<int32_t>>& indices) const;

    static void getAffinityMat(cv::Mat& in, cv::Mat& out, const std::vector<std::vector<int32_t>>& indices);

    static float getGeodesicDistance(const std::vector<cv::Point3f>& face1, const std::vector<cv::Point3f>& face2,
                                     const std::pair<cv::Point3f, cv::Point3f>& edge);

    float getAngleDistance(const std::vector<cv::Point3f>& face1, const std::vector<cv::Point3f>& face2) const;

    static cv::Point3f calculateFaceCentroid(const std::vector<cv::Point3f>& face);

    static cv::Point3f calculateFaceNormal(const std::vector<cv::Point3f>& face);
};

void SpectralCluster::Impl::getLaplacianMat(Mat &in, Mat &out) {
    Mat degree_mat;
    reduce(in, degree_mat, 1, REDUCE_SUM);
    degree_mat = 1.0f / degree_mat;
    sqrt(degree_mat, degree_mat);
    Mat degree_mat_sqrt;
    repeat(degree_mat, 1, in.rows, degree_mat_sqrt);
    out = ((in.mul(degree_mat_sqrt).t()).mul(degree_mat_sqrt)).t();
}

void SpectralCluster::Impl::getAffinityMat(Mat &in, Mat &out, const std::vector<std::vector<int32_t>> &indices) {
    int num_faces = int(indices.size());

    // using dijkstra algorithm to get the affinity matrix
    out = Mat::ones(num_faces, num_faces, CV_32F) * (double)std::numeric_limits<float>::infinity();
    struct dist_node {
        int index;
        float distance;
        dist_node(int i, float d) : index(i), distance(d) {}
        bool operator<(const dist_node& other) const { return distance > other.distance; }
    };

    cv::parallel_for_(cv::Range(0, num_faces), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            std::vector<bool> visited(num_faces, false);
            std::priority_queue<dist_node> pq;
            pq.emplace(i, 0.f);

            while (!pq.empty()) {
                dist_node current = pq.top();
                pq.pop();

                if (visited[current.index])
                    continue;

                visited[current.index] = true;

                for (int neighbor = 0; neighbor < num_faces; ++neighbor) {
                    float weight = in.at<float>(current.index, neighbor);
                    if (weight > 0 && !visited[neighbor]) {
                        float distance = current.distance + weight;
                        if (distance < out.at<float>(i, neighbor)) {
                            out.at<float>(i, neighbor) = distance;
                            pq.emplace(neighbor, distance);
                        }
                    }
                }
            }
        }
    });

    out.setTo(0, out == std::numeric_limits<float>::infinity());

    Scalar sum = cv::sum(out);
    auto sum_val = static_cast<float>(sum[0]);
    float sigma = sum_val / static_cast<float>(num_faces * num_faces);
    sigma = 2.0f * sigma * sigma;

    // Gaussian Kernel
    out.forEach<float>([sigma](float& elem, const int* position) -> void {
        CV_UNUSED(position);
        elem = (elem == 0.f) ? 0.f : exp(-elem / sigma);
    });

    // set diagonal elements to 1
    for (int i = 0; i < num_faces; ++i)
        out.at<float>(i, i) = 1.f;
}

void SpectralCluster::Impl::getAdjacentDistanceMat(Mat &out, const std::vector<Point3f> &vertices,
                                             const std::vector<std::vector<int32_t>> &indices) const {
    int num_faces = int(indices.size());
    int num_adjacency = 0;
    struct PairHash {
        std::size_t operator() (const std::pair<int, int>& pair) const {
            return std::hash<int>{}(pair.first) ^ std::hash<int>{}(pair.second);
        }
    };
    std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash> map;
    map.reserve(num_faces * 3);
    for (int i = 0; i < num_faces; ++i) {
        const std::vector<int32_t> &face = indices[i];
        for (int j = 0; j < 3; ++j) {
            int v1 = face[j], v2 = face[(j + 1) % 3];
            if (v1 > v2) std::swap(v1, v2);
            auto it = map.find({v1, v2});
            if (it != map.end())
                it->second.second = i;
            else
                map[{v1, v2}] = std::make_pair(i, -1);
        }
    }
    // init
    Mat g_dist_mat = Mat::zeros(num_faces, num_faces, CV_32F);
    Mat a_dist_mat = Mat::zeros(num_faces, num_faces, CV_32F);
    std::vector<Point3f> f1, f2;
    std::pair<Point3f, Point3f> edge;
    // compute distances
    for (const auto& entry : map) {
        const std::pair<int, int>& adjacent_faces = entry.second;
        if (adjacent_faces.second == -1)
            continue;
        ++num_adjacency;
        const std::vector<int32_t>& face1 = indices[adjacent_faces.first];
        const std::vector<int32_t>& face2 = indices[adjacent_faces.second];

        f1 = {vertices[face1[0]], vertices[face1[1]], vertices[face1[2]]};
        f2 = {vertices[face2[0]], vertices[face2[1]], vertices[face2[2]]};
        edge = {vertices[entry.first.first], vertices[entry.first.second]};

        g_dist_mat.at<float>(adjacent_faces.first, adjacent_faces.second) = getGeodesicDistance(f1, f2, edge);
        a_dist_mat.at<float>(adjacent_faces.first, adjacent_faces.second) = getAngleDistance(f1, f2);
    }

    // normalize
    g_dist_mat /= (cv::sum(g_dist_mat)[0] / num_adjacency);
    a_dist_mat /= (cv::sum(a_dist_mat)[0] / num_adjacency);
    // make symmetric
    g_dist_mat += g_dist_mat.t();
    a_dist_mat += a_dist_mat.t();
    // compute output distance matrix
    out = this->delta * g_dist_mat + (1.f - this->delta) * a_dist_mat;
}

float SpectralCluster::Impl::getGeodesicDistance(const std::vector<Point3f>& face1, const std::vector<Point3f> &face2,
                                           const std::pair<Point3f, Point3f> &edge) {
    cv::Point3f centroid1 = calculateFaceCentroid(face1);
    cv::Point3f centroid2 = calculateFaceCentroid(face2);
    cv::Point3f edge_center = (edge.first + edge.second) / 2.f;
    return float(norm(centroid1 - edge_center)) + float(norm(centroid2 - edge_center));
}

Point3f SpectralCluster::Impl::calculateFaceCentroid(const std::vector<cv::Point3f>& face) {
    cv::Point3f centroid(0.0f, 0.0f, 0.0f);
    for(const cv::Point3f& p : face)
        centroid += p;
    return centroid / 3.f;
}

float SpectralCluster::Impl::getAngleDistance(const std::vector<Point3f>& face1, const std::vector<Point3f> &face2) const {
    cv::Vec3f normal1 = calculateFaceNormal(face1);
    cv::Vec3f normal2 = calculateFaceNormal(face2);
    float cos_angle = normal1.dot(normal2);
    float angular_distance = 1.0f - cos_angle;

    if (cos_angle < 0)
        angular_distance *= this->eta;
    return angular_distance;
}

Point3f SpectralCluster::Impl::calculateFaceNormal(const std::vector<Point3f>& face) {
    Vec3f normal = Vec3f(face[1] - face[0]).cross(Vec3f(face[2] - face[0]));
    normal /= cv::norm(normal);
    return normal;
}

void SpectralCluster::cluster(const std::vector<cv::Point3f> &vertices, const std::vector<std::vector<int32_t>> &indices,
                              int k, OutputArray result) {
    CV_Assert(k > 1);
    for (const auto & index : indices)
        CV_Assert(index.size() == 3);

    Mat distance_mat;

    pImpl->getAdjacentDistanceMat(distance_mat, vertices, indices);

    Mat affinity_mat;
    pImpl->getAffinityMat(distance_mat, affinity_mat, indices);

    Mat laplacian_mat;
    pImpl->getLaplacianMat(affinity_mat, laplacian_mat);

    // eigen decomposition
    Mat eigen_values, eigen_vectors;
    eigen(laplacian_mat, eigen_values, eigen_vectors);
    eigen_vectors = eigen_vectors.rowRange(0, k);
    Mat eigen_vectors_t;
    eigen_vectors_t = eigen_vectors.t();

    // normalize each row
    float norm_val;
    for (int i = 0; i < eigen_vectors_t.rows; ++i) {
        norm_val = static_cast<float>(norm(eigen_vectors_t.row(i)));
        norm_val = norm_val > std::numeric_limits<float>::epsilon() ? norm_val : std::numeric_limits<float>::epsilon();
        eigen_vectors_t.row(i) /= norm_val;
    }

    Mat label_result;
    cv::kmeans(eigen_vectors_t, k, label_result,
               TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
               5, KMEANS_PP_CENTERS);

    if (result.needed())
        label_result.copyTo(result);
}

}
