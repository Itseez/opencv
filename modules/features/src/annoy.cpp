// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#define ANNOYLIB_MULTITHREADED_BUILD

#include "precomp.hpp"
#include "../3rdparty/annoy/annoylib.h"
#include <opencv2/core/utils/logger.hpp>


namespace cv
{

struct Random
{
    static const uint64 default_seed = 0xffffffff;
    #if __cplusplus < 201103L
    typedef uint64 seed_type;
    #endif

    RNG rng;

    Random(uint64 seed = default_seed)
    {
        rng.state = seed;
    }

    inline int flip()
    {
        // Draw random 0 or 1
        return rng.next() & 1;
    }

    inline size_t index(size_t n)
    {
        // Draw random integer between 0 and n-1 where n is at most the number of data points you have
        return rng(unsigned(n));
    }

    inline void set_seed(uint64 seed)
    {
        rng.state = seed;
    }
};

template <typename DataType, typename DistanceType>
class ANNIndexImpl : public ANNIndex
{
public:
    ANNIndexImpl(int dimension) : dim(dimension)
    {
        index = makePtr<::cvannoy::AnnoyIndex<int, DataType, DistanceType, Random, ::cvannoy::AnnoyIndexMultiThreadedBuildPolicy>>(dimension);
    }

    void addItems(InputArray _dataset) CV_OVERRIDE
    {
        CV_Assert(!_dataset.empty());

        Mat features = _dataset.getMat();
        CV_Assert(features.cols == dim);
        CV_Assert(features.type() == cv::DataType<DataType>::type);

        int num = features.rows;
        char* msg = nullptr;
        if (!index->add_item(0, features.ptr<DataType>(0), &msg))
        {
            if (msg)
            {
                String errorMsg = msg;
                free(msg);
                CV_Error(Error::StsError, errorMsg);
            }
            else
            {
                CV_Error(Error::StsError, "Fail to add an item.");
            }
        }
        for (int i = 1; i < num; ++i)
            index->add_item(i, features.ptr<DataType>(i));
    }

    void build(int trees) CV_OVERRIDE
    {
        if (index->get_n_items() <= 0)
            CV_Error(Error::StsError, "No items added. Please add items before building the index.");

        if (trees <= 0)
            trees = -1;

        char* msg = nullptr;
        if (!index->build(trees, -1, &msg))
        {
            if (msg)
            {
                String errorMsg = msg;
                free(msg);
                CV_Error(Error::StsError, errorMsg);
            }
            else
            {
                CV_Error(Error::StsError, "Fail to build the index.");
            }
        }
    }

    void knnSearch(InputArray _query, OutputArray _indices, OutputArray _dists, int knn, int search_k) CV_OVERRIDE
    {
        CV_Assert(!_query.empty() && _query.isContinuous());
        Mat query = _query.getMat(), indices, dists;
        CV_Assert(query.type() == cv::DataType<DataType>::type);
        CV_Assert(knn > 0 && knn <= index->get_n_items());

        int numQuery = query.rows;
        if (_indices.needed())
        {
            indices = _indices.getMat();
            if (!indices.isContinuous() || indices.type() != CV_32S ||
                indices.rows != numQuery || indices.cols != knn)
            {
                if (!indices.isContinuous())
                    _indices.release();
                _indices.create(numQuery, knn, CV_32S);
                indices = _indices.getMat();
            }
        }
        else
            indices.create(numQuery, knn, CV_32S);

        if (_dists.needed())
        {
            dists = _dists.getMat();
            if (!dists.isContinuous() || dists.type() != cv::DataType<DataType>::type ||
                dists.rows != numQuery || dists.cols != knn)
            {
                if (!_dists.isContinuous())
                    _dists.release();
                _dists.create(numQuery, knn, cv::DataType<DataType>::type);
                dists = _dists.getMat();
            }
        }
        else
            dists.create(numQuery, knn, cv::DataType<DataType>::type);

        auto processBatch = [&](const Range& range)
        {
            std::vector<int> nns;
            std::vector<DataType> distances;

            for (int i = range.start; i < range.end; ++i)
            {
                index->get_nns_by_vector(query.ptr<DataType>(i), knn, search_k, &nns, &distances);

                std::copy(nns.begin(), nns.end(), indices.ptr<int>(i));
                std::copy(distances.begin(), distances.end(), dists.ptr<DataType>(i));

                nns.clear();
                distances.clear();
            }
        };

        parallel_for_(Range(0, numQuery), processBatch);
    }

    void save(const String &filename, bool prefault) CV_OVERRIDE
    {
        char* msg = nullptr;
        if (!index->save(filename.c_str(), prefault, &msg))
        {
            if (msg)
            {
                String errorMsg = msg;
                free(msg);
                CV_Error(Error::StsError, errorMsg);
            }
            else
            {
                CV_Error(Error::StsError, "Fail to save the index.");
            }
        }
    }

    void load(const String &filename, bool prefault) CV_OVERRIDE
    {
        char* msg = nullptr;
        if (!index->load(filename.c_str(), prefault, &msg))
        {
            if (msg)
            {
                String errorMsg = msg;
                free(msg);
                CV_Error(Error::StsError, errorMsg);
            }
            else
            {
                CV_Error(Error::StsError, "Fail to load the index.");
            }
        }
    }

    int getTreeNumber() CV_OVERRIDE
    {
        return index->get_n_trees();
    }

    int getItemNumber() CV_OVERRIDE
    {
        return index->get_n_items();
    }

    bool setOnDiskBuild(const String &filename) CV_OVERRIDE
    {
        char* msg = nullptr;
        if (index->on_disk_build(filename.c_str(), &msg))
            return true;
        else
        {
            if (msg)
            {
                String errorMsg = msg;
                CV_LOG_ERROR(NULL, errorMsg);
                free(msg);
            }
            else
            {
                CV_LOG_ERROR(NULL, "Cannot set build on disk.");
            }
            return false;
        }
    }

    void setSeed(int seed) CV_OVERRIDE
    {
        index->set_seed(static_cast<uint32_t>(seed));
    }

private:
    int dim;
    Ptr<::cvannoy::AnnoyIndex<int, DataType, DistanceType, Random, ::cvannoy::AnnoyIndexMultiThreadedBuildPolicy>> index;
};

Ptr<ANNIndex> ANNIndex::create(int dim, ANNIndex::Distance distType)
{
    switch (distType)
    {
        case ANNIndex::DIST_EUCLIDEAN:
            return makePtr<ANNIndexImpl<float, ::cvannoy::Euclidean>>(dim);
            break;
        case ANNIndex::DIST_MANHATTAN:
            return makePtr<ANNIndexImpl<float, ::cvannoy::Manhattan>>(dim);
            break;
        case ANNIndex::DIST_ANGULAR:
            return makePtr<ANNIndexImpl<float, ::cvannoy::Angular>>(dim);
            break;
        case ANNIndex::DIST_HAMMING:
            return makePtr<ANNIndexImpl<uchar, ::cvannoy::Hamming>>(dim);
            break;
        case ANNIndex::DIST_DOTPRODUCT:
            return makePtr<ANNIndexImpl<float, ::cvannoy::DotProduct>>(dim);
            break;
        default:
            CV_Error(Error::StsBadArg, "Unknown/unsupported distance type");
    }
};

}