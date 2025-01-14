/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/****************************************************************************************\
*                       Algorithmic tests for descriptor matchers                        *
\****************************************************************************************/
class CV_DescriptorMatcherTest : public cvtest::BaseTest
{
public:
    CV_DescriptorMatcherTest( const string& _name, const Ptr<DescriptorMatcher>& _dmatcher, float _badPart ) :
        badPart(_badPart), name(_name), dmatcher(_dmatcher)
        {}
protected:
    static const int dim = 500;
    static const int queryDescCount = 300; // must be even number because we split train data in some cases in two
    static const int countFactor = 4; // do not change it
    const float badPart;

    virtual void run( int );
    void generateData( Mat& query, Mat& train );

#if 0
    void emptyDataTest(); // FIXIT not used
#endif
    void matchTest( const Mat& query, const Mat& train );
    void knnMatchTest( const Mat& query, const Mat& train );
    void radiusMatchTest( const Mat& query, const Mat& train );

    string name;
    Ptr<DescriptorMatcher> dmatcher;

private:
    CV_DescriptorMatcherTest& operator=(const CV_DescriptorMatcherTest&) { return *this; }
};

#if 0
void CV_DescriptorMatcherTest::emptyDataTest()
{
    assert( !dmatcher.empty() );
    Mat queryDescriptors, trainDescriptors, mask;
    vector<Mat> trainDescriptorCollection, masks;
    vector<DMatch> matches;
    vector<vector<DMatch> > vmatches;

    try
    {
        dmatcher->match( queryDescriptors, trainDescriptors, matches, mask );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "match() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->knnMatch( queryDescriptors, trainDescriptors, vmatches, 2, mask );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "knnMatch() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->radiusMatch( queryDescriptors, trainDescriptors, vmatches, 10.f, mask );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "radiusMatch() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->add( trainDescriptorCollection );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "add() on empty descriptors must not generate exception.\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->match( queryDescriptors, matches, masks );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "match() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->knnMatch( queryDescriptors, vmatches, 2, masks );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "knnMatch() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->radiusMatch( queryDescriptors, vmatches, 10.f, masks );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "radiusMatch() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

}
#endif

void CV_DescriptorMatcherTest::generateData( Mat& query, Mat& train )
{
    RNG& rng = theRNG();

    // Generate query descriptors randomly.
    // Descriptor vector elements are integer values.
    Mat buf( queryDescCount, dim, CV_32SC1 );
    rng.fill( buf, RNG::UNIFORM, Scalar::all(0), Scalar(3) );
    buf.convertTo( query, CV_32FC1 );

    // Generate train descriptors as follows:
    // copy each query descriptor to train set countFactor times
    // and perturb some one element of the copied descriptors in
    // in ascending order. General boundaries of the perturbation
    // are (0.f, 1.f).
    train.create( query.rows*countFactor, query.cols, CV_32FC1 );
    float step = 1.f / countFactor;
    for( int qIdx = 0; qIdx < query.rows; qIdx++ )
    {
        Mat queryDescriptor = query.row(qIdx);
        for( int c = 0; c < countFactor; c++ )
        {
            int tIdx = qIdx * countFactor + c;
            Mat trainDescriptor = train.row(tIdx);
            queryDescriptor.copyTo( trainDescriptor );
            int elem = rng(dim);
            float diff = rng.uniform( step*c, step*(c+1) );
            trainDescriptor.at<float>(0, elem) += diff;
        }
    }
}

void CV_DescriptorMatcherTest::matchTest( const Mat& query, const Mat& train )
{
    dmatcher->clear();

    // test const version of match()
    {
        vector<DMatch> matches;
        dmatcher->match( query, train, matches );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test match() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch& match = matches[i];
                if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0) )
                    badCount++;
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test match() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test const version of match() for the same query and test descriptors
    {
        vector<DMatch> matches;
        dmatcher->match( query, query, matches );

        if( (int)matches.size() != query.rows )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test match() function for the same query and test descriptors (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch& match = matches[i];
                //std::cout << match.distance << std::endl;

                if( match.queryIdx != (int)i || match.trainIdx != (int)i || std::abs(match.distance) > FLT_EPSILON )
                {
                    ts->printf( cvtest::TS::LOG, "Bad match (i=%d, queryIdx=%d, trainIdx=%d, distance=%f) while test match() function for the same query and test descriptors (1).\n",
                                i, match.queryIdx, match.trainIdx, match.distance );
                    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                }
            }
        }
    }

    // test version of match() with add()
    {
        vector<DMatch> matches;
        // make add() twice to test such case
        dmatcher->add( vector<Mat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher->add( vector<Mat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<Mat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = Mat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher->match( query, matches, masks );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test match() function (2).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch& match = matches[i];
                int shift = dmatcher->isMaskSupported() ? 1 : 0;
                {
                    if( i < queryDescCount/2 )
                    {
                        if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + shift) || (match.imgIdx != 0) )
                            badCount++;
                    }
                    else
                    {
                        if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + shift) || (match.imgIdx != 1) )
                            badCount++;
                    }
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test match() function (2).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            }
        }
    }
}

void CV_DescriptorMatcherTest::knnMatchTest( const Mat& query, const Mat& train )
{
    dmatcher->clear();

    // test const version of knnMatch()
    {
        const int knn = 3;

        vector<vector<DMatch> > matches;
        dmatcher->knnMatch( query, train, matches, knn );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test knnMatch() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != knn )
                    badCount++;
                else
                {
                    int localBadCount = 0;
                    for( int k = 0; k < knn; k++ )
                    {
                        DMatch& match = matches[i][k];
                        if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor+k) || (match.imgIdx != 0) )
                            localBadCount++;
                    }
                    badCount += localBadCount > 0 ? 1 : 0;
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test knnMatch() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of knnMatch() with add()
    {
        const int knn = 2;
        vector<vector<DMatch> > matches;
        // make add() twice to test such case
        dmatcher->add( vector<Mat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher->add( vector<Mat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<Mat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = Mat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher->knnMatch( query, matches, knn, masks );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test knnMatch() function (2).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            int shift = dmatcher->isMaskSupported() ? 1 : 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != knn )
                    badCount++;
                else
                {
                    int localBadCount = 0;
                    for( int k = 0; k < knn; k++ )
                    {
                        DMatch& match = matches[i][k];
                        {
                            if( i < queryDescCount/2 )
                            {
                                if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + k + shift) ||
                                    (match.imgIdx != 0) )
                                    localBadCount++;
                            }
                            else
                            {
                                if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + k + shift) ||
                                    (match.imgIdx != 1) )
                                    localBadCount++;
                            }
                        }
                    }
                    badCount += localBadCount > 0 ? 1 : 0;
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test knnMatch() function (2).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            }
        }
    }
}

void CV_DescriptorMatcherTest::radiusMatchTest( const Mat& query, const Mat& train )
{
    dmatcher->clear();
    // test const version of match()
    {
        const float radius = 1.f/countFactor;
        vector<vector<DMatch> > matches;
        dmatcher->radiusMatch( query, train, matches, radius );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != 1 )
                    badCount++;
                else
                {
                    DMatch& match = matches[i][0];
                    if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0) )
                        badCount++;
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test radiusMatch() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of match() with add()
    {
        int n = 3;
        const float radius = 1.f/countFactor * n;
        vector<vector<DMatch> > matches;
        // make add() twice to test such case
        dmatcher->add( vector<Mat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher->add( vector<Mat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<Mat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = Mat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher->radiusMatch( query, matches, radius, masks );

        //int curRes = cvtest::TS::OK;
        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }

        int badCount = 0;
        int shift = dmatcher->isMaskSupported() ? 1 : 0;
        int needMatchCount = dmatcher->isMaskSupported() ? n-1 : n;
        for( size_t i = 0; i < matches.size(); i++ )
        {
            if( (int)matches[i].size() != needMatchCount )
                badCount++;
            else
            {
                int localBadCount = 0;
                for( int k = 0; k < needMatchCount; k++ )
                {
                    DMatch& match = matches[i][k];
                    {
                        if( i < queryDescCount/2 )
                        {
                            if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + k + shift) ||
                                (match.imgIdx != 0) )
                                localBadCount++;
                        }
                        else
                        {
                            if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + k + shift) ||
                                (match.imgIdx != 1) )
                                localBadCount++;
                        }
                    }
                }
                badCount += localBadCount > 0 ? 1 : 0;
            }
        }
        if( (float)badCount > (float)queryDescCount*badPart )
        {
            //curRes = cvtest::TS::FAIL_INVALID_OUTPUT;
            ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test radiusMatch() function (2).\n",
                        (float)badCount/(float)queryDescCount );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        }
    }
}

void CV_DescriptorMatcherTest::run( int )
{
    Mat query, train;
    generateData( query, train );

    matchTest( query, train );

    knnMatchTest( query, train );

    radiusMatchTest( query, train );
}

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST( Features2d_DescriptorMatcher_BruteForce, regression )
{
    CV_DescriptorMatcherTest test( "descriptor-matcher-brute-force",
                                  DescriptorMatcher::create("BruteForce"), 0.01f );
    test.safe_run();
}

#ifdef HAVE_OPENCV_FLANN
TEST( Features2d_DescriptorMatcher_FlannBased, regression )
{
    CV_DescriptorMatcherTest test( "descriptor-matcher-flann-based",
                                  DescriptorMatcher::create("FlannBased"), 0.04f );
    test.safe_run();
}
#endif

TEST( Features2d_DMatch, read_write )
{
    FileStorage fs(".xml", FileStorage::WRITE + FileStorage::MEMORY);
    vector<DMatch> matches;
    matches.push_back(DMatch(1,2,3,4.5f));
    fs << "Match" << matches;
    String str = fs.releaseAndGetString();
    ASSERT_NE( strstr(str.c_str(), "4.5"), (char*)0 );
}

#ifdef HAVE_OPENCV_FLANN
TEST( Features2d_FlannBasedMatcher, read_write )
{
    static const char* ymlfile = "%YAML:1.0\n---\n"
    "format: 3\n"
    "indexParams:\n"
    "   -\n"
    "      name: algorithm\n"
    "      type: 9\n"  // FLANN_INDEX_TYPE_ALGORITHM
    "      value: 6\n"// this line is changed!
    "   -\n"
    "      name: trees\n"
    "      type: 4\n"
    "      value: 4\n"
    "searchParams:\n"
    "   -\n"
    "      name: checks\n"
    "      type: 4\n"
    "      value: 32\n"
    "   -\n"
    "      name: eps\n"
    "      type: 5\n"
    "      value: 4.\n"// this line is changed!
    "   -\n"
    "      name: explore_all_trees\n"
    "      type: 8\n"
    "      value: 0\n"
    "   -\n"
    "      name: sorted\n"
    "      type: 8\n"    // FLANN_INDEX_TYPE_BOOL
    "      value: 1\n";

    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
    FileStorage fs_in(ymlfile, FileStorage::READ + FileStorage::MEMORY);
    matcher->read(fs_in.root());
    FileStorage fs_out(".yml", FileStorage::WRITE + FileStorage::MEMORY);
    matcher->write(fs_out);
    std::string out = fs_out.releaseAndGetString();

    EXPECT_EQ(ymlfile, out);
}
#endif

TEST(Features2d_DMatch, issue_11855)
{
    Mat sources = (Mat_<uchar>(2, 3) << 1, 1, 0,
                                        1, 1, 1);
    Mat targets = (Mat_<uchar>(2, 3) << 1, 1, 1,
                                        0, 0, 0);
    Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING, true);
    vector<vector<DMatch> > match;
    bf->knnMatch(sources, targets, match, 1, noArray(), true);

    ASSERT_EQ((size_t)1, match.size());
    ASSERT_EQ((size_t)1, match[0].size());
    EXPECT_EQ(1, match[0][0].queryIdx);
    EXPECT_EQ(0, match[0][0].trainIdx);
    EXPECT_EQ(0.0f, match[0][0].distance);
}

TEST(Features2d_DMatch, issue_17771)
{
    Mat sources = (Mat_<uchar>(2, 3) << 1, 1, 0,
                                        1, 1, 1);
    Mat targets = (Mat_<uchar>(2, 3) << 1, 1, 1,
                                        0, 0, 0);
    UMat usources = sources.getUMat(ACCESS_READ);
    UMat utargets = targets.getUMat(ACCESS_READ);
    vector<vector<DMatch> > match;
    Ptr<BFMatcher> ubf = BFMatcher::create(NORM_HAMMING);
    Mat mask = (Mat_<uchar>(2, 2) << 1, 0, 0, 1);
    EXPECT_NO_THROW(ubf->knnMatch(usources, utargets, match, 1, mask, true));
}

}} // namespace
