// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CONTOURS_COMMON_HPP
#define OPENCV_CONTOURS_COMMON_HPP

#include "precomp.hpp"

#include <array>

namespace cv {

static const schar MAX_SIZE = 16;

static const cv::Point chainCodeDeltas[8] =
    {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};

static inline int getDelta(schar s, size_t step)
{
    CV_DbgAssert(s >= 0 && s < 16);
    const cv::Point res = chainCodeDeltas[s % 8];
    return res.x + res.y * (int)step;
}

inline schar clamp_direction(schar dir)
{
    return std::min(dir, (schar)15);
}

// BLOCK_SIZE_ELEM - number of elements in a block
// STATIC_CAPACITY_BYTES - static memory in bytes for preallocated blocks
template <typename T, size_t BLOCK_SIZE_ELEM = 1024, size_t STATIC_CAPACITY_BYTES = 4096>
class BlockStorage {
    public:
        using value_type = T;
        typedef struct {value_type data[BLOCK_SIZE_ELEM];} block_type;

        BlockStorage()
        {
            const size_t minDynamicBlocks = !staticBlocksCount ? 1 : 0;
            for(size_t i = 0 ; i<minDynamicBlocks ; ++i)
                dynamicBlocks.push_back(new block_type);
        }
        BlockStorage(const BlockStorage&) = delete;
        BlockStorage(BlockStorage&&) noexcept = default;
        ~BlockStorage() {
            for(const auto & block : dynamicBlocks) {
                delete block;
            }
        }
        BlockStorage& operator=(const BlockStorage&) = delete;
        BlockStorage& operator=(BlockStorage&&) noexcept = default;

        void clear(void) {
            const size_t minDynamicBlocks = !staticBlocksCount ? 1 : 0;
            for(size_t i = minDynamicBlocks, count = dynamicBlocks.size() ; i<count ; ++i ) {
                delete dynamicBlocks[i];
            }
            dynamicBlocks.resize(minDynamicBlocks);
            sz = 0;
        }

        void push_back(const value_type& value) {
            const size_t blockIndex = sz / BLOCK_SIZE_ELEM;
            const size_t currentBlocksCount = staticBlocksCount+dynamicBlocks.size();
            if (blockIndex == currentBlocksCount)
                dynamicBlocks.push_back(new block_type);
            block_type& cur_block =
                (blockIndex < staticBlocksCount) ? staticBlocks[blockIndex] :
                *dynamicBlocks[blockIndex-staticBlocksCount];
            cur_block.data[sz % BLOCK_SIZE_ELEM] = value;
            ++sz;
        }

        size_t size() const { return sz; }

        const value_type& at(size_t index) const {
            const size_t blockIndex = index / BLOCK_SIZE_ELEM;
            const block_type& cur_block =
                (blockIndex < staticBlocksCount) ? staticBlocks[blockIndex] :
                *dynamicBlocks[blockIndex-staticBlocksCount];
            return cur_block.data[index % BLOCK_SIZE_ELEM];
        }
        value_type& at(size_t index) {
            const size_t blockIndex = index / BLOCK_SIZE_ELEM;
            block_type& cur_block =
                (blockIndex < staticBlocksCount) ? staticBlocks[blockIndex] :
                *dynamicBlocks[blockIndex-staticBlocksCount];
            return cur_block.data[index % BLOCK_SIZE_ELEM];
        }
        const value_type& operator[](size_t index) const {return at(index);}
        value_type& operator[](size_t index) {return at(index);}
    public:
        friend class RangeIterator;
        class RangeIterator
        {
            public:
                RangeIterator(const BlockStorage* _owner, size_t _first, size_t _last)
                             :owner(_owner),remaining(_last-_first),
                              blockIndex(_first/BLOCK_SIZE_ELEM),offset(_first%BLOCK_SIZE_ELEM) {
                }
            private:
                const BlockStorage* owner = nullptr;
                size_t remaining = 0;
                size_t blockIndex = 0;
                size_t offset = 0;
            public:
                bool done(void) const {return !remaining;}
                std::pair<const value_type*, size_t> operator*(void) const {return get();}
                std::pair<const value_type*, size_t> get(void) const {
                    const block_type& cur_block =
                        (blockIndex < owner->staticBlocksCount) ? owner->staticBlocks[blockIndex] :
                        *owner->dynamicBlocks[blockIndex-owner->staticBlocksCount];
                    const value_type* rangeStart = cur_block.data+offset;
                    const size_t rangeLength = std::min(remaining, BLOCK_SIZE_ELEM-offset);
                    return std::make_pair(rangeStart, rangeLength);
                }
                RangeIterator& operator++() {
                    std::pair<const value_type*, size_t> range = get();
                    remaining -= range.second;
                    offset = 0;
                    ++blockIndex;
                    return *this;
                }
        };
        RangeIterator getRangeIterator(size_t first, size_t last) const {
          return RangeIterator(this, first, last);
        }
    private:
        std::array<block_type, STATIC_CAPACITY_BYTES/(BLOCK_SIZE_ELEM*sizeof(value_type))> staticBlocks;
        const size_t staticBlocksCount = STATIC_CAPACITY_BYTES/(BLOCK_SIZE_ELEM*sizeof(value_type));
        std::vector<block_type*> dynamicBlocks;
        size_t sz = 0;
};

template<typename T>
class vectorOfRanges
{
    public:
        vectorOfRanges(void) = default;
        vectorOfRanges(const vectorOfRanges&) = default;
        vectorOfRanges(vectorOfRanges&& other) noexcept = default;
        ~vectorOfRanges() = default;
    public:
        vectorOfRanges& operator=(const vectorOfRanges&) = default;
        vectorOfRanges& operator=(vectorOfRanges&& other) noexcept = default;
    public:
        bool empty(void) const {return !_size;}
        size_t size(void) const {return _size;}
        T at(size_t index) const {
            for(const auto& range : _ranges) {
                if (index < range.second)
                    return static_cast<T>(range.first+index);
                else
                    index -= range.second;
          }
          return _ranges[0].first;//should not occur
        }
        T back(void) const {return at(_size-1);}
    public:
        void push_back(const T& value) {
            if (_ranges.empty() || (value != back()+1))
                _ranges.push_back(std::make_pair(value, 1));
            else
                ++_ranges.back().second;
            ++_size;
        }
        void pop_back(void) {
            if (_ranges.back().second == 1)
                _ranges.pop_back();
            else
                --_ranges.back().second;
            --_size;
        }
    private:
        std::vector<std::pair<T, size_t> > _ranges;
        size_t _size = 0;
};

template <typename T>
class TreeNode
{
private:
    int self_;

public:
    // tree hierarchy (parent - children)
    int parent;
    int first_child;
    // 1st linked list - bidirectional - sibling children
    int prev;
    int next;
    // 2nd linked list - unidirectional - not related to 1st list
    int ctable_next;
    T body;

public:
    TreeNode(int self, T&& body_) :
        self_(self), parent(-1), first_child(-1), prev(-1), next(-1), ctable_next(-1), body(std::move(body_))
    {
        CV_Assert(self >= 0);
    }
    TreeNode(const TreeNode&) = delete;
    TreeNode(TreeNode&&) noexcept = default;
    TreeNode& operator=(const TreeNode&) = delete;
    TreeNode& operator=(TreeNode&&) noexcept = default;
    int self() const
    {
        return self_;
    }
};

template <typename T>
class Tree
{
public:
    Tree() {}
    Tree(const Tree&) = delete;
    Tree(Tree&&) = delete;
    Tree& operator=(const Tree&) = delete;
    Tree& operator=(Tree&&) = delete;
    ~Tree() = default;
private:
    std::vector<TreeNode<T>> nodes;

public:
    TreeNode<T>& newElem(T && body_)
    {
        const size_t idx = nodes.size();
        CV_DbgAssert(idx < (size_t)std::numeric_limits<int>::max());
        nodes.emplace_back(std::move(TreeNode<T>((int)idx, std::move(body_))));
        return nodes[idx];
    }
    TreeNode<T>& elem(int idx)
    {
        CV_DbgAssert(idx >= 0 && (size_t)idx < nodes.size());
        return nodes[(size_t)idx];
    }
    const TreeNode<T>& elem(int idx) const
    {
        CV_DbgAssert(idx >= 0 && (size_t)idx < nodes.size());
        return nodes[(size_t)idx];
    }
    int lastSibling(int e) const
    {
        if (e != -1)
        {
            while (true)
            {
                const TreeNode<T>& cur_elem = elem(e);
                if (cur_elem.next == -1)
                    break;
                e = cur_elem.next;
            }
        }
        return e;
    }
    void addSiblingAfter(int prev, int idx)
    {
        TreeNode<T>& prev_item = nodes[prev];
        TreeNode<T>& child = nodes[idx];
        child.parent = prev_item.parent;
        if (prev_item.next != -1)
        {
            ((TreeNode<T>&)nodes[prev_item.next]).prev = idx;
            child.next = prev_item.next;
        }
        child.prev = prev;
        prev_item.next = idx;
    }
    void addChild(int parent_idx, int child_idx)
    {
        TreeNode<T>& parent = nodes[parent_idx];
        TreeNode<T>& child = nodes[child_idx];
        if (parent.first_child != -1)
        {
            TreeNode<T>& fchild_ = nodes[parent.first_child];
            fchild_.prev = child_idx;
            child.next = parent.first_child;
        }
        parent.first_child = child_idx;
        child.parent = parent_idx;
        child.prev = -1;
    }
    bool isEmpty() const
    {
        return nodes.size() == 0;
    }
    size_t size() const
    {
        return nodes.size();
    }
};

template <typename T>
class TreeIterator
{
public:
    TreeIterator(Tree<T>& tree_) : tree(tree_)//,levels(&tree._treeIteratorArena)
    {
        CV_Assert(!tree.isEmpty());
        levels.push_back(0);
    }
    bool isDone() const
    {
        return levels.empty();
    }
    const TreeNode<T>& getNext_s()
    {
        int idx = levels.back();
        levels.pop_back();
        const TreeNode<T>& res = tree.elem(idx);
        int cur = tree.lastSibling(res.first_child);
        while (cur != -1)
        {
            levels.push_back(cur);
            cur = tree.elem(cur).prev;
        }
        return res;
    }

private:
    Tree<T>& tree;
    vectorOfRanges<int> levels;
};

//==============================================================================

class ContourPointsStorage
{
    public:
        typedef cv::Point point_storage_t;
        typedef BlockStorage<point_storage_t> storage_t;
    public:
        ContourPointsStorage(void) = delete;
        ContourPointsStorage(storage_t* _storage):storage(_storage) {}
        ContourPointsStorage(const ContourPointsStorage&) = delete;
        ContourPointsStorage(ContourPointsStorage&&) noexcept = default;
        ~ContourPointsStorage() = default;
        ContourPointsStorage& operator=(const ContourPointsStorage&) = delete;
        ContourPointsStorage& operator=(ContourPointsStorage&&) noexcept = default;
    public:
        storage_t::RangeIterator getRangeIterator(void) const {return storage->getRangeIterator(first, last);}
    public:
        bool empty(void) const {return first == last;}
        size_t size(void) const {return last - first;}
    public:
        void clear(void) {first = last;}
        bool resize(size_t newSize) {
            bool ok = (newSize <= size());
            if (ok)
                last = first+newSize;
            return ok;
        }
        void push_back(const point_storage_t& value) {
            if (empty()) {
                first = storage->size();
            }
            storage->push_back(value);
            last = storage->size();
        }
        const cv::Point& at(size_t index) const {return storage->at(first+index);}
        cv::Point& at(size_t index) {return storage->at(first+index);}
        const cv::Point& operator[](size_t index) const {return at(index);}
        cv::Point& operator[](size_t index) {return at(index);}
    private:
        storage_t* storage = nullptr;
        size_t first = 0;
        size_t last = 0;
};

class ContourCodesStorage
{
    public:
        typedef schar code_storage_t;
        typedef BlockStorage<code_storage_t, 1024, 0> storage_t;
    public:
        ContourCodesStorage(void) = delete;
        ContourCodesStorage(storage_t* _storage):storage(_storage) {}
        ContourCodesStorage(const ContourCodesStorage&) = delete;
        ContourCodesStorage(ContourCodesStorage&&) noexcept = default;
        ~ContourCodesStorage() = default;
        ContourCodesStorage& operator=(const ContourCodesStorage&) = delete;
        ContourCodesStorage& operator=(ContourCodesStorage&&) noexcept = default;
    public:
        storage_t::RangeIterator getRangeIterator(void) const {return storage->getRangeIterator(first, last);}
    public:
        bool empty(void) const {return first == last;}
        size_t size(void) const {return last - first;}
    public:
        void clear(void) {first = last;}
        bool resize(size_t newSize) {
            bool ok = (newSize <= size());
            if (ok)
                last = first+newSize;
            return ok;
        }
        void push_back(const code_storage_t& value) {
            if (empty()) {
                first = storage->size();
            }
            storage->push_back(value);
            last = storage->size();
        }
        const schar& at(size_t index) const {return storage->at(first+index);}
        schar& at(size_t index) {return storage->at(first+index);}
        const schar& operator[](size_t index) const {return at(index);}
        schar& operator[](size_t index) {return at(index);}
    private:
        storage_t* storage = nullptr;
        size_t first = 0;
        size_t last = 0;
};
class Contour
{
public:
    ContourPointsStorage pts;
    cv::Rect brect;
    cv::Point origin;
    ContourCodesStorage codes;
    bool isHole = false;
    bool isChain = false;

    explicit Contour(ContourPointsStorage::storage_t* pointStorage_,
                     ContourCodesStorage::storage_t* codesStorage_)
                    :pts(pointStorage_),codes(codesStorage_) {
    }
    Contour(const Contour&) = delete;
    Contour(Contour&& other) noexcept = default;
    Contour& operator=(const Contour&) = delete;
    Contour& operator=(Contour&& other) noexcept = default;
    ~Contour() = default;
    void updateBoundingRect() {}
    bool isEmpty() const
    {
        return pts.size() == 0 && codes.size() == 0;
    }
    size_t size() const
    {
        return isChain ? codes.size() : pts.size();
    }
    void addPoint(const Point& pt)
    {
        pts.push_back(pt);
    }
    void copyTo(void* data) const
    {
        // NOTE: Mat::copyTo doesn't work because it creates new Mat object
        //       instead of reusing existing vector data
        if (isChain)
        {
            /*memcpy(data, codes.data(), codes.size() * sizeof(typename decltype(codes)::value_type));*/
            schar* dst = reinterpret_cast<schar*>(data);
            for(auto rangeIterator = codes.getRangeIterator() ; !rangeIterator.done() ; ++rangeIterator)
            {
                const auto range = *rangeIterator;
                memcpy(dst, range.first, range.second*sizeof(schar));
                dst += range.second;
            }
        }
        else
        {
            /*for (size_t i = 0, count = pts.size() ; i < count ; ++i)
                ((Point*)data)[i] = pts.at(i);
                */
            cv::Point* dst = reinterpret_cast<cv::Point*>(data);
            for(auto rangeIterator = pts.getRangeIterator() ; !rangeIterator.done() ; ++rangeIterator)
            {
                const auto range = *rangeIterator;
                memcpy(dst, range.first, range.second*sizeof(cv::Point));
                dst += range.second;
            }
        }
    }
};

typedef TreeNode<Contour> CNode;
typedef Tree<Contour> CTree;
typedef TreeIterator<Contour> CIterator;


void contourTreeToResults(CTree& tree,
                          int res_type,
                          cv::OutputArrayOfArrays& _contours,
                          cv::OutputArray& _hierarchy);


void approximateChainTC89(const ContourCodesStorage& chain, const Point& origin, const int method,
                          ContourPointsStorage& output);

}  // namespace cv

#endif  // OPENCV_CONTOURS_COMMON_HPP
