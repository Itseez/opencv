// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "persistence.hpp"
#include "persistence_impl.hpp"
#include "persistence_base64_encoding.hpp"
#include <unordered_map>
#include <iterator>

#include <opencv2/core/utils/logger.hpp>

#define CV_SEQ_ELTYPE_PTR            CV_MAKE_TYPE(CV_8U, 8 /*sizeof(void*)*/)

namespace cv
{

namespace fs
{

int strcasecmp(const char* s1, const char* s2)
{
    const char* dummy="";
    if(!s1) s1=dummy;
    if(!s2) s2=dummy;

    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);
    size_t i, len = std::min(len1, len2);
    for( i = 0; i < len; i++ )
    {
        int d = tolower((int)s1[i]) - tolower((int)s2[i]);
        if( d != 0 )
            return d;
    }
    return len1 < len2 ? -1 : len1 > len2 ? 1 : 0;
}

char* itoa( int _val, char* buffer, int /*radix*/ )
{
    const int radix = 10;
    char* ptr=buffer + 23 /* enough even for 64-bit integers */;
    unsigned val = abs(_val);

    *ptr = '\0';
    do
    {
        unsigned r = val / radix;
        *--ptr = (char)(val - (r*radix) + '0');
        val = r;
    }
    while( val != 0 );

    if( _val < 0 )
        *--ptr = '-';

    return ptr;
}

char* itoa( int64_t _val, char* buffer, int /*radix*/, bool _signed)
{
    const int radix = 10;
    char* ptr=buffer + 23 /* enough even for 64-bit integers */;
    int sign = _signed && _val < 0 ? -1 : 1;
    uint64_t val = !_signed ? (uint64_t)_val : abs(_val);

    *ptr = '\0';
    do
    {
        uint64_t r = val / radix;
        *--ptr = (char)(val - (r*radix) + '0');
        val = r;
    }
    while( val != 0 );

    if( sign < 0 )
        *--ptr = '-';

    return ptr;
}

char* doubleToString( char* buf, size_t bufSize, double value, bool explicitZero )
{
    Cv64suf val;
    unsigned ieee754_hi;

    val.f = value;
    ieee754_hi = (unsigned)(val.u >> 32);

    if( (ieee754_hi & 0x7ff00000) != 0x7ff00000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
        {
            if( explicitZero )
                snprintf( buf, bufSize, "%d.0", ivalue );
            else
                snprintf( buf, bufSize, "%d.", ivalue );
        }
        else
        {
            // binary64 has 52 bit fraction with hidden bit.
            // 53 * log_10(2) is 15.955. So "%.16f" should be fine, but its test fails.
            snprintf( buf, bufSize, "%.17g", value );

            char* ptr = buf;
            if( *ptr == '+' || *ptr == '-' )
                ptr++;
            for( ; cv_isdigit(*ptr); ptr++ )
                ;
            if( *ptr == ',' )
                *ptr = '.';
        }
    }
    else
    {
        unsigned ieee754_lo = (unsigned)val.u;
        if( (ieee754_hi & 0x7fffffff) + (ieee754_lo != 0) > 0x7ff00000 )
            strcpy( buf, ".Nan" );
        else
            strcpy( buf, (int)ieee754_hi < 0 ? "-.Inf" : ".Inf" );
    }

    return buf;
}

char* floatToString( char* buf, size_t bufSize, float value, bool halfprecision, bool explicitZero )
{
    Cv32suf val;
    unsigned ieee754;
    val.f = value;
    ieee754 = val.u;

    if( (ieee754 & 0x7f800000) != 0x7f800000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
        {
            if( explicitZero )
                snprintf( buf, bufSize, "%d.0", ivalue );
            else
                snprintf( buf, bufSize, "%d.", ivalue );
        }
        else
        {
            if (halfprecision)
            {
                // bfloat16 has 7 bit fraction with hidden bit.
                // binary16 has 10 bit fraction with hidden bit.
                // 11 * log_10(2) is 3.311. So "%.4f" should be fine, but its test fails.
                snprintf(buf, bufSize, "%.5g", value);
            }
            else
            {
                // binray32 has 23 bit fraction with hidden bit.
                // 24 * log_10(2) is 7.225. So "%.8f" should be fine, but its test fails.
                snprintf(buf, bufSize, "%.9g", value);
            }

            char* ptr = buf;
            if( *ptr == '+' || *ptr == '-' )
                ptr++;
            for( ; cv_isdigit(*ptr); ptr++ )
                ;
            if( *ptr == ',' )
                *ptr = '.';
        }
    }
    else
    {
        if( (ieee754 & 0x7fffffff) != 0x7f800000 )
            strcpy( buf, ".Nan" );
        else
            strcpy( buf, (int)ieee754 < 0 ? "-.Inf" : ".Inf" );
    }

    return buf;
}

static const char symbols[] = "ucwsifdhHbUIn";

static char typeSymbol(int depth)
{
    CV_StaticAssert(CV_64F == 6, "");
    CV_CheckDepth(depth, depth >= 0 && depth <= CV_32U, "");
    return symbols[depth];
}

static int symbolToType(char c)
{
    if (c == 'r')
        return CV_SEQ_ELTYPE_PTR;
    const char* pos = strchr( symbols, c );
    if( !pos )
        CV_Error( cv::Error::StsBadArg, "Invalid data type specification" );
    return static_cast<int>(pos - symbols);
}

char* encodeFormat(int elem_type, char* dt, size_t dt_len)
{
    int cn = (elem_type == CV_SEQ_ELTYPE_PTR/*CV_USRTYPE1*/) ? 1 : CV_MAT_CN(elem_type);
    char symbol = (elem_type == CV_SEQ_ELTYPE_PTR/*CV_USRTYPE1*/) ? 'r' : typeSymbol(CV_MAT_DEPTH(elem_type));
    snprintf(dt, dt_len, "%d%c", cn, symbol);
    return dt + (cn == 1 ? 1 : 0);
}

// Deprecated due to size of dt buffer being unknowable.
char* encodeFormat(int elem_type, char* dt)
{
    constexpr size_t max = 20+1+1; // UINT64_MAX + one char + nul termination.
    return encodeFormat(elem_type, dt, max);
}

int decodeFormat( const char* dt, int* fmt_pairs, int max_len )
{
    int fmt_pair_count = 0;
    int i = 0, k = 0, len = dt ? (int)strlen(dt) : 0;

    if( !dt || !len )
        return 0;

    CV_Assert( fmt_pairs != 0 && max_len > 0 );
    fmt_pairs[0] = 0;
    max_len *= 2;

    for( ; k < len; k++ )
    {
        char c = dt[k];

        if( cv_isdigit(c) )
        {
            int count = c - '0';
            if( cv_isdigit(dt[k+1]) )
            {
                char* endptr = 0;
                count = (int)strtol( dt+k, &endptr, 10 );
                k = (int)(endptr - dt) - 1;
            }

            if( count <= 0 )
                CV_Error( cv::Error::StsBadArg, "Invalid data type specification" );

            fmt_pairs[i] = count;
        }
        else
        {
            int depth = symbolToType(c);
            if( fmt_pairs[i] == 0 )
                fmt_pairs[i] = 1;
            fmt_pairs[i+1] = depth;
            if( i > 0 && fmt_pairs[i+1] == fmt_pairs[i-1] )
                fmt_pairs[i-2] += fmt_pairs[i];
            else
            {
                i += 2;
                if( i >= max_len )
                    CV_Error( cv::Error::StsBadArg, "Too long data type specification" );
            }
            fmt_pairs[i] = 0;
        }
    }

    fmt_pair_count = i/2;
    return fmt_pair_count;
}

int calcElemSize( const char* dt, int initial_size )
{
    int size = 0;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], i, fmt_pair_count;
    int comp_size;

    fmt_pair_count = decodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    fmt_pair_count *= 2;
    for( i = 0, size = initial_size; i < fmt_pair_count; i += 2 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[i+1]);
        size = cvAlign( size, comp_size );
        size += comp_size * fmt_pairs[i];
    }
    if( initial_size == 0 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[1]);
        size = cvAlign( size, comp_size );
    }
    return size;
}


int calcStructSize( const char* dt, int initial_size )
{
    int size = calcElemSize( dt, initial_size );
    size_t elem_max_size = 0;
    for ( const char * type = dt; *type != '\0'; type++ )
    {
        char v = *type;
        if (v >= '0' && v <= '9')
            continue;  // skip vector size
        switch (v)
        {
        case 'u': { elem_max_size = std::max( elem_max_size, sizeof(uchar ) ); break; }
        case 'b': { elem_max_size = std::max( elem_max_size, sizeof(bool  ) ); break; }
        case 'c': { elem_max_size = std::max( elem_max_size, sizeof(schar ) ); break; }
        case 'w': { elem_max_size = std::max( elem_max_size, sizeof(ushort) ); break; }
        case 's': { elem_max_size = std::max( elem_max_size, sizeof(short ) ); break; }
        case 'i': { elem_max_size = std::max( elem_max_size, sizeof(int   ) ); break; }
        case 'n': { elem_max_size = std::max( elem_max_size, sizeof(unsigned) ); break; }
        case 'f': { elem_max_size = std::max( elem_max_size, sizeof(float ) ); break; }
        case 'd': { elem_max_size = std::max( elem_max_size, sizeof(double) ); break; }
        case 'h': { elem_max_size = std::max( elem_max_size, sizeof(hfloat)); break; }
        case 'H': { elem_max_size = std::max( elem_max_size, sizeof(bfloat)); break; }
        case 'I': { elem_max_size = std::max( elem_max_size, sizeof(int64_t)); break; }
        case 'U': { elem_max_size = std::max( elem_max_size, sizeof(uint64_t)); break; }
        default:
            CV_Error_(Error::StsNotImplemented, ("Unknown type identifier: '%c' in '%s'", (char)(*type), dt));
        }
    }
    size = cvAlign( size, static_cast<int>(elem_max_size) );
    return size;
}

int decodeSimpleFormat( const char* dt )
{
    int elem_type = -1;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;

    fmt_pair_count = decodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    if( fmt_pair_count != 1 || fmt_pairs[0] >= CV_CN_MAX)
        CV_Error( cv::Error::StsError, "Too complex format for the matrix" );

    elem_type = CV_MAKETYPE( fmt_pairs[1], fmt_pairs[0] );

    return elem_type;
}

}

#if defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64) || \
    (defined (__LITTLE_ENDIAN__) && __LITTLE_ENDIAN__)
#define CV_LITTLE_ENDIAN_MEM_ACCESS 1
#else
#define CV_LITTLE_ENDIAN_MEM_ACCESS 0
#endif

static inline int readInt(const uchar* p)
{
    // On little endian CPUs, both branches produce the same result. On big endian, only the else branch does.
#if CV_LITTLE_ENDIAN_MEM_ACCESS
    int val;
    memcpy(&val, p, sizeof(val));
    return val;
#else
    int val = (int)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
    return val;
#endif
}

static inline int64_t readLong(const uchar* p)
{
    // On little endian CPUs, both branches produce the same result. On big endian, only the else branch does.
#if CV_LITTLE_ENDIAN_MEM_ACCESS
    int64_t val;
    memcpy(&val, p, sizeof(val));
    return val;
#else
    unsigned val0 = (unsigned)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
    unsigned val1 = (unsigned)(p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24));
    return val0 | ((int64_t)val1 << 32);
#endif
}

static inline double readReal(const uchar* p)
{
    // On little endian CPUs, both branches produce the same result. On big endian, only the else branch does.
#if CV_LITTLE_ENDIAN_MEM_ACCESS
    double val;
    memcpy(&val, p, sizeof(val));
    return val;
#else
    unsigned val0 = (unsigned)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
    unsigned val1 = (unsigned)(p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24));
    Cv64suf val;
    val.u = val0 | ((uint64)val1 << 32);
    return val.f;
#endif
}

template <typename T>
static inline void writeInt(uchar* p, T ival)
{
    // On little endian CPUs, both branches produce the same result. On big endian, only the else branch does.
#if CV_LITTLE_ENDIAN_MEM_ACCESS
    memcpy(p, &ival, sizeof(ival));
#else
    for (size_t i = 0, j = 0; i < sizeof(ival); ++i, j += 8)
        p[i] = (uchar)(ival >> j);
#endif
}

static inline void writeReal(uchar* p, double fval)
{
    // On little endian CPUs, both branches produce the same result. On big endian, only the else branch does.
#if CV_LITTLE_ENDIAN_MEM_ACCESS
    memcpy(p, &fval, sizeof(fval));
#else
    Cv64suf v;
    v.f = fval;
    p[0] = (uchar)v.u;
    p[1] = (uchar)(v.u >> 8);
    p[2] = (uchar)(v.u >> 16);
    p[3] = (uchar)(v.u >> 24);
    p[4] = (uchar)(v.u >> 32);
    p[5] = (uchar)(v.u >> 40);
    p[6] = (uchar)(v.u >> 48);
    p[7] = (uchar)(v.u >> 56);
#endif
}



void FileStorage::Impl::init() {
    flags = 0;
    buffer.clear();
    bufofs = 0;
    state = UNDEFINED;
    is_using_base64 = false;
    state_of_writing_base64 = FileStorage_API::Base64State::Uncertain;
    is_write_struct_delayed = false;
    delayed_struct_key = nullptr;
    delayed_struct_flags = 0;
    delayed_type_name = nullptr;
    base64_writer = nullptr;
    is_opened = false;
    dummy_eof = false;
    write_mode = false;
    mem_mode = false;
    space = 0;
    wrap_margin = 71;
    fmt = 0;
    file = 0;
    gzfile = 0;
    empty_stream = true;

    strbufv.clear();
    strbuf = 0;
    strbufsize = strbufpos = 0;
    roots.clear();

    fs_data.clear();
    fs_data_ptrs.clear();
    fs_data_blksz.clear();
    freeSpaceOfs = 0;

    str_hash.clear();
    str_hash_data.clear();
    str_hash_data.resize(1);
    str_hash_data[0] = '\0';

    filename.clear();
    lineno = 0;
}

FileStorage::Impl::Impl(FileStorage *_fs) {
    fs_ext = _fs;
    init();
}

FileStorage::Impl::~Impl() {
    release();
}

void FileStorage::Impl::release(String *out) {
    if (is_opened) {
        if (out)
            out->clear();
        if (write_mode) {
            while (write_stack.size() > 1) {
                endWriteStruct();
            }
            flush();
            if (fmt == FileStorage::FORMAT_XML)
                puts("</opencv_storage>\n");
            else if (fmt == FileStorage::FORMAT_JSON)
                puts("}\n");
        }
        if (mem_mode && out) {
            *out = cv::String(outbuf.begin(), outbuf.end());
        }
    }
    closeFile();
    init();
}

void FileStorage::Impl::analyze_file_name(const std::string &file_name, std::vector<std::string> &params) {
    params.clear();
    static const char not_file_name = '\n';
    static const char parameter_begin = '?';
    static const char parameter_separator = '&';

    if (file_name.find(not_file_name, (size_t) 0) != std::string::npos)
        return;

    size_t beg = file_name.find_last_of(parameter_begin);
    params.push_back(file_name.substr((size_t) 0, beg));

    if (beg != std::string::npos) {
        size_t end = file_name.size();
        beg++;
        for (size_t param_beg = beg, param_end = beg;
             param_end < end;
             param_beg = param_end + 1) {
            param_end = file_name.find_first_of(parameter_separator, param_beg);
            if ((param_end == std::string::npos || param_end != param_beg) && param_beg + 1 < end) {
                params.push_back(file_name.substr(param_beg, param_end - param_beg));
            }
        }
    }
}

bool FileStorage::Impl::open(const char *filename_or_buf, int _flags, const char *encoding) {
    bool ok = true;
    release();

    bool append = (_flags & 3) == FileStorage::APPEND;
    mem_mode = (_flags & FileStorage::MEMORY) != 0;

    write_mode = (_flags & 3) != 0;
    bool write_base64 = (write_mode || append) && (_flags & FileStorage::BASE64) != 0;

    bool isGZ = false;
    size_t fnamelen = 0;

    std::vector<std::string> params;
    //if ( !mem_mode )
    {
        analyze_file_name(filename_or_buf, params);
        if (!params.empty())
            filename = params[0];

        if (!write_base64 && params.size() >= 2 &&
            std::find(params.begin() + 1, params.end(), std::string("base64")) != params.end())
            write_base64 = (write_mode || append);
    }

    if (filename.size() == 0 && !mem_mode && !write_mode)
        CV_Error(cv::Error::StsNullPtr, "NULL or empty filename");

    if (mem_mode && append)
        CV_Error(cv::Error::StsBadFlag, "FileStorage::APPEND and FileStorage::MEMORY are not currently compatible");

    flags = _flags;

    if (!mem_mode) {
        char *dot_pos = strrchr((char *) filename.c_str(), '.');
        char compression = '\0';

        if (dot_pos && dot_pos[1] == 'g' && dot_pos[2] == 'z' &&
            (dot_pos[3] == '\0' || (cv_isdigit(dot_pos[3]) && dot_pos[4] == '\0'))) {
            if (append) {
                CV_Error(cv::Error::StsNotImplemented, "Appending data to compressed file is not implemented");
            }
            isGZ = true;
            compression = dot_pos[3];
            if (compression)
                dot_pos[3] = '\0', fnamelen--;
        }

        if (!isGZ) {
            file = fopen(filename.c_str(), !write_mode ? "rt" : !append ? "wt" : "a+t");
            if (!file)
            {
                CV_LOG_ERROR(NULL, "Can't open file: '" << filename << "' in " << (!write_mode ? "read" : !append ? "write" : "append") << " mode");
                return false;
            }
        } else {
#if USE_ZLIB
            char mode[] = {write_mode ? 'w' : 'r', 'b', compression ? compression : '3', '\0'};
            gzfile = gzopen(filename.c_str(), mode);
            if (!gzfile)
            {
                CV_LOG_ERROR(NULL, "Can't open archive: '" << filename << "' mode=" << mode);
                return false;
            }
#else
            CV_Error(cv::Error::StsNotImplemented, "There is no compressed file storage support in this configuration");
#endif
        }
    }

    // FIXIT release() must do that, use CV_Assert() here instead
    roots.clear();
    fs_data.clear();

    wrap_margin = 71;
    fmt = FileStorage::FORMAT_AUTO;

    if (write_mode) {
        fmt = flags & FileStorage::FORMAT_MASK;

        if (mem_mode)
            outbuf.clear();

        if (fmt == FileStorage::FORMAT_AUTO && !filename.empty()) {
            const char *dot_pos = NULL;
            const char *dot_pos2 = NULL;
            // like strrchr() implementation, but save two last positions simultaneously
            for (const char *pos = &filename[0]; pos[0] != 0; pos++) {
                if (pos[0] == '.') {
                    dot_pos2 = dot_pos;
                    dot_pos = pos;
                }
            }
            if (fs::strcasecmp(dot_pos, ".gz") == 0 && dot_pos2 != NULL) {
                dot_pos = dot_pos2;
            }
            fmt = (fs::strcasecmp(dot_pos, ".xml") == 0 || fs::strcasecmp(dot_pos, ".xml.gz") == 0)
                  ? FileStorage::FORMAT_XML
                  : (fs::strcasecmp(dot_pos, ".json") == 0 || fs::strcasecmp(dot_pos, ".json.gz") == 0)
                    ? FileStorage::FORMAT_JSON
                    : FileStorage::FORMAT_YAML;
        } else if (fmt == FileStorage::FORMAT_AUTO) {
            fmt = FileStorage::FORMAT_XML;
        }

        // we use factor=6 for XML (the longest characters (' and ") are encoded with 6 bytes (&apos; and &quot;)
        // and factor=4 for YAML ( as we use 4 bytes for non ASCII characters (e.g. \xAB))
        int buf_size = CV_FS_MAX_LEN * (fmt == FileStorage::FORMAT_XML ? 6 : 4) + 1024;

        if (append) {
            fseek(file, 0, SEEK_END);
            if (ftell(file) == 0)
                append = false;
        }

        write_stack.clear();
        empty_stream = true;
        write_stack.push_back(FStructData("", FileNode::MAP | FileNode::EMPTY, 0));
        buffer.reserve(buf_size + 1024);
        buffer.resize(buf_size);
        bufofs = 0;
        is_using_base64 = write_base64;
        state_of_writing_base64 = FileStorage_API::Base64State::Uncertain;

        if (fmt == FileStorage::FORMAT_XML) {
            size_t file_size = file ? (size_t) ftell(file) : (size_t) 0;
            if (!append || file_size == 0) {
                if (encoding && *encoding != '\0') {
                    if (fs::strcasecmp(encoding, "UTF-16") == 0) {
                        release();
                        CV_Error(cv::Error::StsBadArg, "UTF-16 XML encoding is not supported! Use 8-bit encoding\n");
                    }

                    CV_Assert(strlen(encoding) < 1000);
                    char buf[1100];
                    snprintf(buf, sizeof(buf), "<?xml version=\"1.0\" encoding=\"%s\"?>\n", encoding);
                    puts(buf);
                } else
                    puts("<?xml version=\"1.0\"?>\n");
                puts("<opencv_storage>\n");
            } else {
                int xml_buf_size = 1 << 10;
                char substr[] = "</opencv_storage>";
                int last_occurrence = -1;
                xml_buf_size = MIN(xml_buf_size, int(file_size));
                fseek(file, -xml_buf_size, SEEK_END);
                // find the last occurrence of </opencv_storage>
                for (;;) {
                    int line_offset = (int) ftell(file);
                    const char *ptr0 = this->gets(xml_buf_size);
                    const char *ptr = NULL;
                    if (!ptr0)
                        break;
                    ptr = ptr0;
                    for (;;) {
                        ptr = strstr(ptr, substr);
                        if (!ptr)
                            break;
                        last_occurrence = line_offset + (int) (ptr - ptr0);
                        ptr += strlen(substr);
                    }
                }
                if (last_occurrence < 0) {
                    release();
                    CV_Error(cv::Error::StsError, "Could not find </opencv_storage> in the end of file.\n");
                }
                closeFile();
                file = fopen(filename.c_str(), "r+t");
                CV_Assert(file != 0);
                fseek(file, last_occurrence, SEEK_SET);
                // replace the last "</opencv_storage>" with " <!-- resumed -->", which has the same length
                puts(" <!-- resumed -->");
                fseek(file, 0, SEEK_END);
                puts("\n");
            }

            emitter_do_not_use_direct_dereference = createXMLEmitter(this);
        } else if (fmt == FileStorage::FORMAT_YAML) {
            if (!append)
                puts("%YAML:1.0\n---\n");
            else
                puts("...\n---\n");

            emitter_do_not_use_direct_dereference = createYAMLEmitter(this);
        } else {
            CV_Assert(fmt == FileStorage::FORMAT_JSON);
            if (!append)
                puts("{\n");
            else {
                bool valid = false;
                long roffset = 0;
                for (;
                        fseek(file, roffset, SEEK_END) == 0;
                        roffset -= 1) {
                    const char end_mark = '}';
                    if (fgetc(file) == end_mark) {
                        fseek(file, roffset, SEEK_END);
                        valid = true;
                        break;
                    }
                }

                if (valid) {
                    closeFile();
                    file = fopen(filename.c_str(), "r+t");
                    CV_Assert(file != 0);
                    fseek(file, roffset, SEEK_END);
                    fputs(",", file);
                } else {
                    CV_Error(cv::Error::StsError, "Could not find '}' in the end of file.\n");
                }
            }
            write_stack.back().indent = 4;
            emitter_do_not_use_direct_dereference = createJSONEmitter(this);
        }
        is_opened = true;
    } else {
        const size_t buf_size0 = 40;
        buffer.resize(buf_size0);
        if (mem_mode) {
            strbuf = (char *) filename_or_buf;
            strbufsize = strlen(strbuf);
        }

        const char *yaml_signature = "%YAML";
        const char *json_signature = "{";
        const char *xml_signature = "<?xml";
        char *buf = this->gets(16);
        CV_Assert(buf);
        char *bufPtr = cv_skip_BOM(buf);
        size_t bufOffset = bufPtr - buf;

        if (strncmp(bufPtr, yaml_signature, strlen(yaml_signature)) == 0)
            fmt = FileStorage::FORMAT_YAML;
        else if (strncmp(bufPtr, json_signature, strlen(json_signature)) == 0)
            fmt = FileStorage::FORMAT_JSON;
        else if (strncmp(bufPtr, xml_signature, strlen(xml_signature)) == 0)
            fmt = FileStorage::FORMAT_XML;
        else if (strbufsize == bufOffset)
            CV_Error(cv::Error::StsBadArg, "Input file is invalid");
        else
            CV_Error(cv::Error::StsBadArg, "Unsupported file storage format");

        rewind();
        strbufpos = bufOffset;
        bufofs = 0;

        try {
            char *ptr = bufferStart();
            ptr[0] = ptr[1] = ptr[2] = '\0';
            FileNode root_nodes(fs_ext, 0, 0);

            uchar *rptr = reserveNodeSpace(root_nodes, 9);
            *rptr = FileNode::SEQ;
            writeInt(rptr + 1, 4);
            writeInt(rptr + 5, 0);

            roots.clear();

            switch (fmt) {
                case FileStorage::FORMAT_XML:
                    parser_do_not_use_direct_dereference = createXMLParser(this);
                    break;
                case FileStorage::FORMAT_YAML:
                    parser_do_not_use_direct_dereference = createYAMLParser(this);
                    break;
                case FileStorage::FORMAT_JSON:
                    parser_do_not_use_direct_dereference = createJSONParser(this);
                    break;
                default:
                    parser_do_not_use_direct_dereference = Ptr<FileStorageParser>();
            }

            if (!parser_do_not_use_direct_dereference.empty()) {
                ok = getParser().parse(ptr);
                if (ok) {
                    finalizeCollection(root_nodes);

                    CV_Assert(!fs_data_ptrs.empty());
                    FileNode roots_node(fs_ext, 0, 0);
                    size_t i, nroots = roots_node.size();
                    FileNodeIterator it = roots_node.begin();

                    for (i = 0; i < nroots; i++, ++it)
                        roots.push_back(*it);
                }
            }
        }
        catch (...)
        {
            // FIXIT log error message
            is_opened = true;
            release();
            throw;
        }

        // release resources that we do not need anymore
        closeFile();
        is_opened = true;
        std::vector<char> tmpbuf;
        std::swap(buffer, tmpbuf);
        bufofs = 0;
    }
    return ok;
}

void FileStorage::Impl::puts(const char *str) {
    CV_Assert(write_mode);
    if (mem_mode)
        std::copy(str, str + strlen(str), std::back_inserter(outbuf));
    else if (file)
        fputs(str, file);
#if USE_ZLIB
    else if (gzfile)
        gzputs(gzfile, str);
#endif
    else
        CV_Error(cv::Error::StsError, "The storage is not opened");
}

char *FileStorage::Impl::getsFromFile(char *buf, int count) {
    if (file)
        return fgets(buf, count, file);
#if USE_ZLIB
    if (gzfile)
        return gzgets(gzfile, buf, count);
#endif
    CV_Error(cv::Error::StsError, "The storage is not opened");
}

char *FileStorage::Impl::gets(size_t maxCount) {
    if (strbuf) {
        size_t i = strbufpos, len = strbufsize;
        const char *instr = strbuf;
        for (; i < len; i++) {
            char c = instr[i];
            if (c == '\0' || c == '\n') {
                if (c == '\n')
                    i++;
                break;
            }
        }
        size_t count = i - strbufpos;
        if (maxCount == 0 || maxCount > count)
            maxCount = count;
        buffer.resize(std::max(buffer.size(), maxCount + 8));
        memcpy(&buffer[0], instr + strbufpos, maxCount);
        buffer[maxCount] = '\0';
        strbufpos = i;
        return maxCount > 0 ? &buffer[0] : 0;
    }

    const size_t MAX_BLOCK_SIZE = INT_MAX / 2; // hopefully, that will be enough
    if (maxCount == 0)
        maxCount = MAX_BLOCK_SIZE;
    else
        CV_Assert(maxCount < MAX_BLOCK_SIZE);
    size_t ofs = 0;

    for (;;) {
        int count = (int) std::min(buffer.size() - ofs - 16, maxCount);
        char *ptr = getsFromFile(&buffer[ofs], count + 1);
        if (!ptr)
            break;
        int delta = (int) strlen(ptr);
        ofs += delta;
        maxCount -= delta;
        if (delta == 0 || ptr[delta - 1] == '\n' || maxCount == 0)
            break;
        if (delta == count)
            buffer.resize((size_t) (buffer.size() * 1.5));
    }
    return ofs > 0 ? &buffer[0] : 0;
}

char *FileStorage::Impl::gets() {
    char *ptr = this->gets(0);
    if (!ptr) {
        ptr = bufferStart();  // FIXIT Why do we need this hack? What is about other parsers JSON/YAML?
        *ptr = '\0';
        setEof();
        return 0;
    } else {
        size_t l = strlen(ptr);
        if (l > 0 && ptr[l - 1] != '\n' && ptr[l - 1] != '\r' && !eof()) {
            ptr[l] = '\n';
            ptr[l + 1] = '\0';
        }
    }
    lineno++;
    return ptr;
}

bool FileStorage::Impl::eof() {
    if (dummy_eof)
        return true;
    if (strbuf)
        return strbufpos >= strbufsize;
    if (file)
        return feof(file) != 0;
#if USE_ZLIB
    if (gzfile)
        return gzeof(gzfile) != 0;
#endif
    return false;
}

void FileStorage::Impl::setEof() {
    dummy_eof = true;
}

void FileStorage::Impl::closeFile() {
    if (file)
        fclose(file);
#if USE_ZLIB
    else if (gzfile)
        gzclose(gzfile);
#endif
    file = 0;
    gzfile = 0;
    strbuf = 0;
    strbufpos = 0;
    is_opened = false;
}

void FileStorage::Impl::rewind() {
    if (file)
        ::rewind(file);
#if USE_ZLIB
    else if (gzfile)
        gzrewind(gzfile);
#endif
    strbufpos = 0;
}

char *FileStorage::Impl::resizeWriteBuffer(char *ptr, int len) {
    const char *buffer_end = &buffer[0] + buffer.size();
    if (ptr + len < buffer_end)
        return ptr;

    const char *buffer_start = &buffer[0];
    int written_len = (int) (ptr - buffer_start);

    CV_Assert(written_len <= (int) buffer.size());
    int new_size = (int) ((buffer_end - buffer_start) * 3 / 2);
    new_size = MAX(written_len + len, new_size);
    buffer.reserve(new_size + 256);
    buffer.resize(new_size);
    bufofs = written_len;
    return &buffer[0] + bufofs;
}

char *FileStorage::Impl::flush() {
    char *buffer_start = &buffer[0];
    char *ptr = buffer_start + bufofs;

    if (ptr > buffer_start + space) {
        ptr[0] = '\n';
        ptr[1] = '\0';
        puts(buffer_start);
        bufofs = 0;
    }

    int indent = write_stack.back().indent;

    if (space != indent) {
        memset(buffer_start, ' ', indent);
        space = indent;
    }
    bufofs = space;
    ptr = buffer_start + bufofs;

    return ptr;
}

void FileStorage::Impl::endWriteStruct() {
    CV_Assert(write_mode);

    check_if_write_struct_is_delayed(false);
    if (state_of_writing_base64 != FileStorage_API::Uncertain)
        switch_to_Base64_state(FileStorage_API::Uncertain);

    CV_Assert(!write_stack.empty());

    FStructData &current_struct = write_stack.back();
    if (fmt == FileStorage::FORMAT_JSON && !FileNode::isFlow(current_struct.flags) && write_stack.size() > 1)
        current_struct.indent = write_stack[write_stack.size() - 2].indent;

    getEmitter().endWriteStruct(current_struct);

    write_stack.pop_back();
    if (!write_stack.empty())
        write_stack.back().flags &= ~FileNode::EMPTY;
}

void FileStorage::Impl::startWriteStruct_helper(const char *key, int struct_flags,
                                                const char *type_name) {
    CV_Assert(write_mode);

    struct_flags = (struct_flags & (FileNode::TYPE_MASK | FileNode::FLOW)) | FileNode::EMPTY;
    if (!FileNode::isCollection(struct_flags))
        CV_Error(cv::Error::StsBadArg,
                 "Some collection type: FileNode::SEQ or FileNode::MAP must be specified");

    if (type_name && type_name[0] == '\0')
        type_name = 0;

    FStructData s = getEmitter().startWriteStruct(write_stack.back(), key, struct_flags, type_name);

    write_stack.push_back(s);
    size_t write_stack_size = write_stack.size();
    if (write_stack_size > 1)
        write_stack[write_stack_size - 2].flags &= ~FileNode::EMPTY;

    if (fmt != FileStorage::FORMAT_JSON && !FileNode::isFlow(s.flags))
        flush();

    if (fmt == FileStorage::FORMAT_JSON && type_name && type_name[0] && FileNode::isMap(struct_flags)) {
        getEmitter().write("type_id", type_name, false);
    }
}

void FileStorage::Impl::startWriteStruct(const char *key, int struct_flags,
                                         const char *type_name) {
    check_if_write_struct_is_delayed(false);
    if (state_of_writing_base64 == FileStorage_API::NotUse)
        switch_to_Base64_state(FileStorage_API::Uncertain);

    if (state_of_writing_base64 == FileStorage_API::Uncertain && FileNode::isSeq(struct_flags)
        && is_using_base64 && type_name == 0) {
        /* Uncertain whether output Base64 data */
        make_write_struct_delayed(key, struct_flags, type_name);
    } else if (type_name && memcmp(type_name, "binary", 6) == 0) {
        /* Must output Base64 data */
        if ((FileNode::TYPE_MASK & struct_flags) != FileNode::SEQ)
            CV_Error(cv::Error::StsBadArg, "must set 'struct_flags |= CV_NODE_SEQ' if using Base64.");
        else if (state_of_writing_base64 != FileStorage_API::Uncertain)
            CV_Error(cv::Error::StsError, "function \'cvStartWriteStruct\' calls cannot be nested if using Base64.");

        startWriteStruct_helper(key, struct_flags, "binary");

        if (state_of_writing_base64 != FileStorage_API::Uncertain)
            switch_to_Base64_state(FileStorage_API::Uncertain);
        switch_to_Base64_state(FileStorage_API::InUse);
    } else {
        /* Won't output Base64 data */
        if (state_of_writing_base64 == FileStorage_API::InUse)
            CV_Error(cv::Error::StsError, "At the end of the output Base64, `cvEndWriteStruct` is needed.");

        startWriteStruct_helper(key, struct_flags, type_name);

        if (state_of_writing_base64 != FileStorage_API::Uncertain)
            switch_to_Base64_state(FileStorage_API::Uncertain);
        switch_to_Base64_state(FileStorage_API::NotUse);
    }
}

void FileStorage::Impl::writeComment(const char *comment, bool eol_comment) {
    CV_Assert(write_mode);
    getEmitter().writeComment(comment, eol_comment);
}

void FileStorage::Impl::startNextStream() {
    CV_Assert(write_mode);
    if (!empty_stream) {
        while (!write_stack.empty())
            endWriteStruct();
        flush();
        getEmitter().startNextStream();
        empty_stream = true;
        write_stack.push_back(FStructData("", FileNode::EMPTY, 0));
        bufofs = 0;
    }
}

void FileStorage::Impl::write(const String &key, int value) {
    CV_Assert(write_mode);
    getEmitter().write(key.c_str(), value);
}

void FileStorage::Impl::write(const String &key, int64_t value) {
    CV_Assert(write_mode);
    getEmitter().write(key.c_str(), value);
}

void FileStorage::Impl::write(const String &key, double value) {
    CV_Assert(write_mode);
    getEmitter().write(key.c_str(), value);
}

void FileStorage::Impl::write(const String &key, const String &value) {
    CV_Assert(write_mode);
    getEmitter().write(key.c_str(), value.c_str(), false);
}

void FileStorage::Impl::writeRawData(const std::string &dt, const void *_data, size_t len) {
    CV_Assert(write_mode);

    if (is_using_base64 || state_of_writing_base64 == FileStorage_API::Base64State::InUse) {
        writeRawDataBase64(_data, len, dt.c_str());
        return;
    } else if (state_of_writing_base64 == FileStorage_API::Base64State::Uncertain) {
        switch_to_Base64_state(FileStorage_API::Base64State::NotUse);
    }

    size_t elemSize = fs::calcStructSize(dt.c_str(), 0);
    CV_Assert(elemSize);
    CV_Assert(len % elemSize == 0);
    len /= elemSize;

    bool explicitZero = fmt == FileStorage::FORMAT_JSON;
    const uchar *data0 = (const uchar *) _data;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS * 2], k, fmt_pair_count;
    char buf[256] = "";

    fmt_pair_count = fs::decodeFormat(dt.c_str(), fmt_pairs, CV_FS_MAX_FMT_PAIRS);

    if (!len)
        return;

    if (!data0)
        CV_Error(cv::Error::StsNullPtr, "Null data pointer");

    if (fmt_pair_count == 1) {
        fmt_pairs[0] *= (int) len;
        len = 1;
    }

    for (; len--; data0 += elemSize) {
        int offset = 0;
        for (k = 0; k < fmt_pair_count; k++) {
            int i, count = fmt_pairs[k * 2];
            int elem_type = fmt_pairs[k * 2 + 1];
            int elem_size = CV_ELEM_SIZE(elem_type);
            const char *ptr;

            offset = cvAlign(offset, elem_size);
            const uchar *data = data0 + offset;

            for (i = 0; i < count; i++) {
                switch (elem_type) {
                    case CV_8U:
                        ptr = fs::itoa(*(uchar *) data, buf, 10);
                        data++;
                        break;
                    case CV_Bool:
                        ptr = fs::itoa(*(uchar *) data != 0, buf, 10);
                        data++;
                        break;
                    case CV_8S:
                        ptr = fs::itoa(*(char *) data, buf, 10);
                        data++;
                        break;
                    case CV_16U:
                        ptr = fs::itoa(*(ushort *) data, buf, 10);
                        data += sizeof(ushort);
                        break;
                    case CV_16S:
                        ptr = fs::itoa(*(short *) data, buf, 10);
                        data += sizeof(short);
                        break;
                    case CV_32U:
                        ptr = fs::itoa((int64_t)*(unsigned*) data, buf, 10, false);
                        data += sizeof(unsigned);
                        break;
                    case CV_32S:
                        ptr = fs::itoa(*(int *) data, buf, 10);
                        data += sizeof(int);
                        break;
                    case CV_64U:
                        ptr = fs::itoa(*(uint64_t*) data, buf, 10, false);
                        data += sizeof(uint64_t);
                        break;
                    case CV_64S:
                        ptr = fs::itoa(*(int64_t*) data, buf, 10, true);
                        data += sizeof(int64_t);
                        break;
                    case CV_32F:
                        ptr = fs::floatToString(buf, sizeof(buf), *(float *) data, false, explicitZero);
                        data += sizeof(float);
                        break;
                    case CV_64F:
                        ptr = fs::doubleToString(buf, sizeof(buf), *(double *) data, explicitZero);
                        data += sizeof(double);
                        break;
                    case CV_16F:
                        ptr = fs::floatToString(buf, sizeof(buf), (float) *(hfloat *) data, true, explicitZero);
                        data += sizeof(hfloat);
                        break;
                    case CV_16BF:
                        ptr = fs::floatToString(buf, sizeof(buf), (float) *(bfloat *) data, true, explicitZero);
                        data += sizeof(bfloat);
                        break;
                    default:
                        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported type");
                        return;
                }

                getEmitter().writeScalar(0, ptr);
            }

            offset = (int) (data - data0);
        }
    }
}

void FileStorage::Impl::workaround() {
    check_if_write_struct_is_delayed(false);

    if (state_of_writing_base64 != FileStorage_API::Base64State::Uncertain)
        switch_to_Base64_state(FileStorage_API::Base64State::Uncertain);
}

void FileStorage::Impl::switch_to_Base64_state(FileStorage_API::Base64State new_state) {
    const char *err_unkonwn_state = "Unexpected error, unable to determine the Base64 state.";
    const char *err_unable_to_switch = "Unexpected error, unable to switch to this state.";

    /* like a finite state machine */
    switch (state_of_writing_base64) {
        case FileStorage_API::Base64State::Uncertain:
            switch (new_state) {
                case FileStorage_API::Base64State::InUse:
                {
                    CV_DbgAssert(base64_writer == 0);
                    bool can_indent = (fmt != cv::FileStorage::Mode::FORMAT_JSON);
                    base64_writer = new base64::Base64Writer(*this, can_indent);
                    if (!can_indent) {
                        char *ptr = bufferPtr();
                        *ptr++ = '\0';
                        puts(bufferStart());
                        setBufferPtr(bufferStart());
                        memset(bufferStart(), 0, static_cast<int>(space));
                        puts("\"$base64$");
                    }
                    break;
                }
                case FileStorage_API::Base64State::Uncertain:
                    break;
                case FileStorage_API::Base64State::NotUse:
                    break;
                default:
                    CV_Error(cv::Error::StsError, err_unkonwn_state);
                    break;
            }
            break;
        case FileStorage_API::Base64State::InUse:
            switch (new_state) {
                case FileStorage_API::Base64State::InUse:
                case FileStorage_API::Base64State::NotUse:
                    CV_Error(cv::Error::StsError, err_unable_to_switch);
                    break;
                case FileStorage_API::Base64State::Uncertain:
                    delete base64_writer;
                    base64_writer = 0;
                    if ( fmt == cv::FileStorage::FORMAT_JSON )
                    {
                        puts("\"");
                        setBufferPtr(bufferStart());
                        flush();
                        memset(bufferStart(), 0, static_cast<int>(space) );
                        setBufferPtr(bufferStart());
                    }
                    break;
                default:
                    CV_Error(cv::Error::StsError, err_unkonwn_state);
                    break;
            }
            break;
        case FileStorage_API::Base64State::NotUse:
            switch (new_state) {
                case FileStorage_API::Base64State::InUse:
                case FileStorage_API::Base64State::NotUse:
                    CV_Error(cv::Error::StsError, err_unable_to_switch);
                    break;
                case FileStorage_API::Base64State::Uncertain:
                    break;
                default:
                    CV_Error(cv::Error::StsError, err_unkonwn_state);
                    break;
            }
            break;
        default:
            CV_Error(cv::Error::StsError, err_unkonwn_state);
            break;
    }

    state_of_writing_base64 = new_state;
}

void FileStorage::Impl::make_write_struct_delayed(const char *key, int struct_flags, const char *type_name) {
    CV_Assert(is_write_struct_delayed == false);
    CV_DbgAssert(delayed_struct_key == nullptr);
    CV_DbgAssert(delayed_struct_flags == 0);
    CV_DbgAssert(delayed_type_name == nullptr);

    delayed_struct_flags = struct_flags;

    if (key != nullptr) {
        delayed_struct_key = new char[strlen(key) + 1U];
        strcpy(delayed_struct_key, key);
    }

    if (type_name != nullptr) {
        delayed_type_name = new char[strlen(type_name) + 1U];
        strcpy(delayed_type_name, type_name);
    }

    is_write_struct_delayed = true;
}

void FileStorage::Impl::check_if_write_struct_is_delayed(bool change_type_to_base64) {
    if (is_write_struct_delayed) {
        /* save data to prevent recursive call errors */
        std::string struct_key;
        std::string type_name;
        int struct_flags = delayed_struct_flags;

        if (delayed_struct_key != nullptr && *delayed_struct_key != '\0') {
            struct_key.assign(delayed_struct_key);
        }
        if (delayed_type_name != nullptr && *delayed_type_name != '\0') {
            type_name.assign(delayed_type_name);
        }

        /* reset */
        delete[] delayed_struct_key;
        delete[] delayed_type_name;
        delayed_struct_key = nullptr;
        delayed_struct_flags = 0;
        delayed_type_name = nullptr;

        is_write_struct_delayed = false;

        /* call */
        if (change_type_to_base64) {
            startWriteStruct_helper(struct_key.c_str(), struct_flags, "binary");
            if (state_of_writing_base64 != FileStorage_API::Uncertain)
                switch_to_Base64_state(FileStorage_API::Uncertain);
            switch_to_Base64_state(FileStorage_API::InUse);
        } else {
            startWriteStruct_helper(struct_key.c_str(), struct_flags, type_name.c_str());
            if (state_of_writing_base64 != FileStorage_API::Uncertain)
                switch_to_Base64_state(FileStorage_API::Uncertain);
            switch_to_Base64_state(FileStorage_API::NotUse);
        }
    }
}

void FileStorage::Impl::writeRawDataBase64(const void *_data, size_t len, const char *dt) {
    CV_Assert(write_mode);

    check_if_write_struct_is_delayed(true);

    if (state_of_writing_base64 == FileStorage_API::Base64State::Uncertain) {
        switch_to_Base64_state(FileStorage_API::Base64State::InUse);
    } else if (state_of_writing_base64 != FileStorage_API::Base64State::InUse) {
        CV_Error(cv::Error::StsError, "Base64 should not be used at present.");
    }

    base64_writer->write(_data, len, dt);
}

FileNode FileStorage::Impl::getFirstTopLevelNode() const {
    return roots.empty() ? FileNode() : roots[0];
}

FileNode FileStorage::Impl::root(int streamIdx) const {
    return streamIdx >= 0 && streamIdx < (int) roots.size() ? roots[streamIdx] : FileNode();
}

FileNode FileStorage::Impl::operator[](const String &nodename) const {
    return this->operator[](nodename.c_str());
}

FileNode FileStorage::Impl::operator[](const char * /*nodename*/) const {
    return FileNode();
}

int FileStorage::Impl::getFormat() const { return fmt; }

char *FileStorage::Impl::bufferPtr() const { return (char *) (&buffer[0] + bufofs); }

char *FileStorage::Impl::bufferStart() const { return (char *) &buffer[0]; }

char *FileStorage::Impl::bufferEnd() const { return (char *) (&buffer[0] + buffer.size()); }

void FileStorage::Impl::setBufferPtr(char *ptr) {
    char *bufferstart = bufferStart();
    CV_Assert(ptr >= bufferstart && ptr <= bufferEnd());
    bufofs = ptr - bufferstart;
}

int FileStorage::Impl::wrapMargin() const { return wrap_margin; }

FStructData &FileStorage::Impl::getCurrentStruct() {
    CV_Assert(!write_stack.empty());
    return write_stack.back();
}

void FileStorage::Impl::setNonEmpty() {
    empty_stream = false;
}

void FileStorage::Impl::processSpecialDouble(char *buf, double *value, char **endptr) {
    FileStorage_API *fs = this;
    char c = buf[0];
    int inf_hi = 0x7ff00000;

    if (c == '-' || c == '+') {
        inf_hi = c == '-' ? 0xfff00000 : 0x7ff00000;
        c = *++buf;
    }

    if (c != '.')
        CV_PARSE_ERROR_CPP("Bad format of floating-point constant");

    Cv64suf v;
    v.f = 0.;
    if (toupper(buf[1]) == 'I' && toupper(buf[2]) == 'N' && toupper(buf[3]) == 'F')
        v.u = (uint64) inf_hi << 32;
    else if (toupper(buf[1]) == 'N' && toupper(buf[2]) == 'A' && toupper(buf[3]) == 'N')
        v.u = (uint64) -1;
    else
        CV_PARSE_ERROR_CPP("Bad format of floating-point constant");
    *value = v.f;
    *endptr = buf + 4;
}

double FileStorage::Impl::strtod(char *ptr, char **endptr) {
    double fval = ::strtod(ptr, endptr);
    if (**endptr == '.') {
        char *dot_pos = *endptr;
        *dot_pos = ',';
        double fval2 = ::strtod(ptr, endptr);
        *dot_pos = '.';
        if (*endptr > dot_pos)
            fval = fval2;
        else
            *endptr = dot_pos;
    }

    if (*endptr == ptr || cv_isalpha(**endptr))
        processSpecialDouble(ptr, &fval, endptr);

    return fval;
}

void FileStorage::Impl::convertToCollection(int type, FileNode &node) {
    CV_Assert(type == FileNode::SEQ || type == FileNode::MAP);

    int node_type = node.type();
    if (node_type == type)
        return;

    bool named = node.isNamed();
    uchar *ptr = node.ptr() + 1 + (named ? 4 : 0);

    int64_t ival = 0;
    double fval = 0;
    std::string sval;
    bool add_first_scalar = false;

    if (node_type != FileNode::NONE) {
        // scalar nodes can only be converted to sequences, e.g. in XML:
        // <a>5[parser_position]... => create 5 with name "a"
        // <a>5 6[parser_position]... => 5 is converted to [5] and then 6 is added to it
        //
        // otherwise we don't know where to get the element names from
        CV_Assert(type == FileNode::SEQ);
        if (node_type == FileNode::INT) {
            ival = readLong(ptr);
            add_first_scalar = true;
        } else if (node_type == FileNode::REAL) {
            fval = readReal(ptr);
            add_first_scalar = true;
        } else if (node_type == FileNode::STRING) {
            sval = std::string(node);
            add_first_scalar = true;
        } else
            CV_Error_(Error::StsError, ("The node of type %d cannot be converted to collection", node_type));
    }

    ptr = reserveNodeSpace(node, 1 + (named ? 4 : 0) + 4 + 4);
    *ptr++ = (uchar) (type | (named ? FileNode::NAMED : 0));
    // name has been copied automatically
    if (named)
        ptr += 4;
    // set raw_size(collection)==4, nelems(collection)==1
    writeInt(ptr, 4);
    writeInt(ptr + 4, 0);

    if (add_first_scalar)
        addNode(node, std::string(), node_type,
                node_type == FileNode::INT ? (const void *) &ival :
                node_type == FileNode::REAL ? (const void *) &fval :
                node_type == FileNode::STRING ? (const void *) sval.c_str() : 0,
                -1);
}

// a) allocates new FileNode (for that just set blockIdx to the last block and ofs to freeSpaceOfs) or
// b) reallocates just created new node (blockIdx and ofs must be taken from FileNode).
//    If there is no enough space in the current block (it should be the last block added so far),
//    the last block is shrunk so that it ends immediately before the reallocated node. Then,
//    a new block of sufficient size is allocated and the FileNode is placed in the beginning of it.
// The case (a) can be used to allocate the very first node by setting blockIdx == ofs == 0.
// In the case (b) the existing tag and the name are copied automatically.
uchar *FileStorage::Impl::reserveNodeSpace(FileNode &node, size_t sz) {
    bool shrinkBlock = false;
    size_t shrinkBlockIdx = 0, shrinkSize = 0;

    uchar *ptr = 0, *blockEnd = 0;

    if (!fs_data_ptrs.empty()) {
        size_t blockIdx = node.blockIdx;
        size_t ofs = node.ofs;
        CV_Assert(blockIdx == fs_data_ptrs.size() - 1);
        CV_Assert(ofs <= fs_data_blksz[blockIdx]);
        CV_Assert(freeSpaceOfs <= fs_data_blksz[blockIdx]);
        //CV_Assert( freeSpaceOfs <= ofs + sz );

        ptr = fs_data_ptrs[blockIdx] + ofs;
        blockEnd = fs_data_ptrs[blockIdx] + fs_data_blksz[blockIdx];

        CV_Assert(ptr >= fs_data_ptrs[blockIdx] && ptr <= blockEnd);
        if (ptr + sz <= blockEnd) {
            freeSpaceOfs = ofs + sz;
            return ptr;
        }

        if (ofs ==
            0)  // FileNode is a first component of this block. Resize current block instead of allocation of new one.
        {
            fs_data[blockIdx]->resize(sz);
            ptr = &fs_data[blockIdx]->at(0);
            fs_data_ptrs[blockIdx] = ptr;
            fs_data_blksz[blockIdx] = sz;
            freeSpaceOfs = sz;
            return ptr;
        }

        shrinkBlock = true;
        shrinkBlockIdx = blockIdx;
        shrinkSize = ofs;
    }

    size_t blockSize = std::max((size_t) CV_FS_MAX_LEN * 4 - 256, sz) + 256;
    Ptr<std::vector<uchar> > pv = makePtr<std::vector<uchar> >(blockSize);
    fs_data.push_back(pv);
    uchar *new_ptr = &pv->at(0);
    fs_data_ptrs.push_back(new_ptr);
    fs_data_blksz.push_back(blockSize);
    node.blockIdx = fs_data_ptrs.size() - 1;
    node.ofs = 0;
    freeSpaceOfs = sz;

    if (ptr && ptr + 5 <= blockEnd) {
        new_ptr[0] = ptr[0];
        if (ptr[0] & FileNode::NAMED) {
            new_ptr[1] = ptr[1];
            new_ptr[2] = ptr[2];
            new_ptr[3] = ptr[3];
            new_ptr[4] = ptr[4];
        }
    }

    if (shrinkBlock) {
        fs_data[shrinkBlockIdx]->resize(shrinkSize);
        fs_data_blksz[shrinkBlockIdx] = shrinkSize;
    }

    return new_ptr;
}

unsigned FileStorage::Impl::getStringOfs(const std::string &key) const {
    str_hash_t::const_iterator it = str_hash.find(key);
    return it != str_hash.end() ? it->second : 0;
}

FileNode FileStorage::Impl::addNode(FileNode &collection, const std::string &key,
                                    int elem_type, const void *value, int len) {
    FileStorage_API *fs = this;
    bool noname = key.empty() || (fmt == FileStorage::FORMAT_XML && strcmp(key.c_str(), "_") == 0);
    convertToCollection(noname ? FileNode::SEQ : FileNode::MAP, collection);

    bool isseq = collection.empty() ? false : collection.isSeq();
    if (noname != isseq)
        CV_PARSE_ERROR_CPP(noname ? "Map element should have a name" :
                           "Sequence element should not have name (use <_></_>)");
    unsigned strofs = 0;
    if (!noname) {
        strofs = getStringOfs(key);
        if (!strofs) {
            strofs = (unsigned) str_hash_data.size();
            size_t keysize = key.size() + 1;
            str_hash_data.resize(strofs + keysize);
            memcpy(&str_hash_data[0] + strofs, &key[0], keysize);
            str_hash.insert(std::make_pair(key, strofs));
        }
    }

    uchar *cp = collection.ptr();

    size_t blockIdx = fs_data_ptrs.size() - 1;
    size_t ofs = freeSpaceOfs;
    FileNode node(fs_ext, blockIdx, ofs);

    size_t sz0 = 1 + (noname ? 0 : 4) + 8;
    uchar *ptr = reserveNodeSpace(node, sz0);

    *ptr++ = (uchar) (elem_type | (noname ? 0 : FileNode::NAMED));
    if (elem_type == FileNode::NONE)
        freeSpaceOfs -= 8;

    if (!noname) {
        writeInt(ptr, (int) strofs);
        ptr += 4;
    }

    if (elem_type == FileNode::SEQ || elem_type == FileNode::MAP) {
        writeInt(ptr, 4);
        writeInt(ptr, 0);
    }

    if (value)
        node.setValue(elem_type, value, len);

    if (collection.isNamed())
        cp += 4;
    int nelems = readInt(cp + 5);
    writeInt(cp + 5, nelems + 1);

    return node;
}

void FileStorage::Impl::finalizeCollection(FileNode &collection) {
    if (!collection.isSeq() && !collection.isMap())
        return;
    uchar *ptr0 = collection.ptr(), *ptr = ptr0 + 1;
    if (*ptr0 & FileNode::NAMED)
        ptr += 4;
    size_t blockIdx = collection.blockIdx;
    size_t ofs = collection.ofs + (size_t) (ptr + 8 - ptr0);
    size_t rawSize = 4;
    unsigned sz = (unsigned) readInt(ptr + 4);
    if (sz > 0) {
        size_t lastBlockIdx = fs_data_ptrs.size() - 1;

        for (; blockIdx < lastBlockIdx; blockIdx++) {
            rawSize += fs_data_blksz[blockIdx] - ofs;
            ofs = 0;
        }
    }
    rawSize += freeSpaceOfs - ofs;
    writeInt(ptr, (int) rawSize);
}

void FileStorage::Impl::normalizeNodeOfs(size_t &blockIdx, size_t &ofs) const {
    while (ofs >= fs_data_blksz[blockIdx]) {
        if (blockIdx == fs_data_blksz.size() - 1) {
            CV_Assert(ofs == fs_data_blksz[blockIdx]);
            break;
        }
        ofs -= fs_data_blksz[blockIdx];
        blockIdx++;
    }
}

FileStorage::Impl::Base64State FileStorage::Impl::get_state_of_writing_base64() {
    return state_of_writing_base64;
}

int FileStorage::Impl::get_space() {
    return space;
}


FileStorage::Impl::Base64Decoder::Base64Decoder() {
    ofs = 0;
    ptr = 0;
    indent = 0;
    totalchars = 0;
    eos = true;
}

void FileStorage::Impl::Base64Decoder::init(const Ptr<FileStorageParser> &_parser, char *_ptr, int _indent) {
    parser_do_not_use_direct_dereference = _parser;
    ptr = _ptr;
    indent = _indent;
    encoded.clear();
    decoded.clear();
    ofs = 0;
    totalchars = 0;
    eos = false;
}

bool FileStorage::Impl::Base64Decoder::readMore(int needed) {
    static const uchar base64tab[] =
            {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0, 63,
                    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0, 0, 0, 0, 0,
                    0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };

    if (eos)
        return false;

    size_t sz = decoded.size();
    CV_Assert(ofs <= sz);
    sz -= ofs;
    for (size_t i = 0; i < sz; i++)
        decoded[i] = decoded[ofs + i];

    decoded.resize(sz);
    ofs = 0;

    CV_Assert(ptr);
    char *beg = 0, *end = 0;
    bool ok = getParser().getBase64Row(ptr, indent, beg, end);
    ptr = end;
    std::copy(beg, end, std::back_inserter(encoded));
    totalchars += end - beg;

    if (!ok || beg == end) {
        // in the end of base64 sequence pad it with '=' characters so that
        // its total length is multiple of
        eos = true;
        size_t tc = totalchars;
        for (; tc % 4 != 0; tc++)
            encoded.push_back('=');
    }

    int i = 0, j, n = (int) encoded.size();
    if (n > 0) {
        const uchar *tab = base64tab;
        char *src = &encoded[0];

        for (; i <= n - 4; i += 4) {
            // dddddd cccccc bbbbbb aaaaaa => ddddddcc ccccbbbb bbaaaaaa
            uchar d = tab[(int) (uchar) src[i]], c = tab[(int) (uchar) src[i + 1]];
            uchar b = tab[(int) (uchar) src[i + 2]], a = tab[(int) (uchar) src[i + 3]];

            decoded.push_back((uchar) ((d << 2) | (c >> 4)));
            decoded.push_back((uchar) ((c << 4) | (b >> 2)));
            decoded.push_back((uchar) ((b << 6) | a));
        }
    }

    if (i > 0 && encoded[i - 1] == '=') {
        if (i > 1 && encoded[i - 2] == '=' && !decoded.empty())
            decoded.pop_back();
        if (!decoded.empty())
            decoded.pop_back();
    }

    n -= i;
    for (j = 0; j < n; j++)
        encoded[j] = encoded[i + j];
    encoded.resize(n);

    return (int) decoded.size() >= needed;
}

uchar FileStorage::Impl::Base64Decoder::getUInt8() {
    size_t sz = decoded.size();
    if (ofs >= sz && !readMore(1))
        return (uchar) 0;
    return decoded[ofs++];
}

ushort FileStorage::Impl::Base64Decoder::getUInt16() {
    size_t sz = decoded.size();
    if (ofs + 2 > sz && !readMore(2))
        return (ushort) 0;
    ushort val = (decoded[ofs] + (decoded[ofs + 1] << 8));
    ofs += 2;
    return val;
}

int FileStorage::Impl::Base64Decoder::getInt32() {
    size_t sz = decoded.size();
    if (ofs + 4 > sz && !readMore(4))
        return 0;
    int ival = readInt(&decoded[ofs]);
    ofs += 4;
    return ival;
}

double FileStorage::Impl::Base64Decoder::getFloat64() {
    size_t sz = decoded.size();
    if (ofs + 8 > sz && !readMore(8))
        return 0;
    double fval = readReal(&decoded[ofs]);
    ofs += 8;
    return fval;
}

bool FileStorage::Impl::Base64Decoder::endOfStream() const { return eos; }

char *FileStorage::Impl::Base64Decoder::getPtr() const { return ptr; }


char *FileStorage::Impl::parseBase64(char *ptr, int indent, FileNode &collection) {
    const int BASE64_HDR_SIZE = 24;
    char dt[BASE64_HDR_SIZE + 1] = {0};
    base64decoder.init(parser_do_not_use_direct_dereference, ptr, indent);

    int i, k;

    for (i = 0; i < BASE64_HDR_SIZE; i++)
        dt[i] = (char) base64decoder.getUInt8();
    for (i = 0; i < BASE64_HDR_SIZE; i++)
        if (isspace(dt[i]))
            break;
    dt[i] = '\0';

    CV_Assert(!base64decoder.endOfStream());

    int fmt_pairs[CV_FS_MAX_FMT_PAIRS * 2];
    int fmt_pair_count = fs::decodeFormat(dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS);
    int64_t ival = 0;
    double fval = 0;

    for (;;) {
        for (k = 0; k < fmt_pair_count; k++) {
            int elem_type = fmt_pairs[k * 2 + 1];
            int count = fmt_pairs[k * 2];

            for (i = 0; i < count; i++) {
                int node_type = FileNode::INT;
                switch (elem_type) {
                    case CV_8U:
                        ival = base64decoder.getUInt8();
                        break;
                    case CV_8S:
                        ival = (char) base64decoder.getUInt8();
                        break;
                    case CV_16U:
                        ival = base64decoder.getUInt16();
                        break;
                    case CV_16S:
                        ival = (short) base64decoder.getUInt16();
                        break;
                    case CV_32S:
                        ival = base64decoder.getInt32();
                        break;
                    case CV_32F: {
                        Cv32suf v;
                        v.i = base64decoder.getInt32();
                        fval = v.f;
                        node_type = FileNode::REAL;
                    }
                        break;
                    case CV_64F:
                        fval = base64decoder.getFloat64();
                        node_type = FileNode::REAL;
                        break;
                    case CV_16F:
                        fval = float(hfloatFromBits(base64decoder.getUInt16()));
                        node_type = FileNode::REAL;
                        break;
                    default:
                        CV_Error(Error::StsUnsupportedFormat, "Unsupported type");
                }

                if (base64decoder.endOfStream())
                    break;
                addNode(collection, std::string(), node_type,
                        node_type == FileNode::INT ? (void *) &ival : (void *) &fval, -1);
            }
        }
        if (base64decoder.endOfStream())
            break;
    }

    finalizeCollection(collection);
    return base64decoder.getPtr();
}

void FileStorage::Impl::parseError(const char *func_name, const std::string &err_msg, const char *source_file,
                                   int source_line) {
    std::string msg = format("%s(%d): %s", filename.c_str(), lineno, err_msg.c_str());
    error(Error::StsParseError, func_name, msg.c_str(), source_file, source_line);
}

const uchar *FileStorage::Impl::getNodePtr(size_t blockIdx, size_t ofs) const {
    CV_Assert(blockIdx < fs_data_ptrs.size());
    CV_Assert(ofs < fs_data_blksz[blockIdx]);

    return fs_data_ptrs[blockIdx] + ofs;
}

std::string FileStorage::Impl::getName(size_t nameofs) const {
    CV_Assert(nameofs < str_hash_data.size());
    return std::string(&str_hash_data[nameofs]);
}

FileStorage *FileStorage::Impl::getFS() { return fs_ext; }


FileStorage::FileStorage()
    : state(0)
{
    p = makePtr<FileStorage::Impl>(this);
}

FileStorage::FileStorage(const String& filename, int flags, const String& encoding)
    : state(0)
{
    p = makePtr<FileStorage::Impl>(this);
    bool ok = p->open(filename.c_str(), flags, encoding.c_str());
    if(ok)
        state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
}

void FileStorage::startWriteStruct(const String& name, int struct_flags, const String& typeName)
{
    p->startWriteStruct(name.size() ? name.c_str() : 0, struct_flags, typeName.size() ? typeName.c_str() : 0);
    elname = String();
    if ((struct_flags & FileNode::TYPE_MASK) == FileNode::SEQ)
        state = FileStorage::VALUE_EXPECTED;
    else
        state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
}

void FileStorage::endWriteStruct()
{
    p->endWriteStruct();
    state = p->write_stack.empty() || FileNode::isMap(p->write_stack.back().flags) ?
        FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP :
        FileStorage::VALUE_EXPECTED;
    elname = String();
}

FileStorage::~FileStorage()
{
}

bool FileStorage::open(const String& filename, int flags, const String& encoding)
{
    try
    {
        bool ok = p->open(filename.c_str(), flags, encoding.c_str());
        if(ok)
            state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
        return ok;
    }
    catch (...)
    {
        release();
        throw;  // re-throw
    }
}

bool FileStorage::isOpened() const { return p->is_opened; }

void FileStorage::release()
{
    p->release();
}

FileNode FileStorage::root(int i) const
{
    if( p.empty() || p->roots.empty() || i < 0 || i >= (int)p->roots.size() )
        return FileNode();

    return p->roots[i];
}

FileNode FileStorage::getFirstTopLevelNode() const
{
    FileNode r = root();
    FileNodeIterator it = r.begin();
    return it != r.end() ? *it : FileNode();
}

std::string FileStorage::getDefaultObjectName(const std::string& _filename)
{
    static const char* stubname = "unnamed";
    const char* filename = _filename.c_str();
    const char* ptr2 = filename + _filename.size();
    const char* ptr = ptr2 - 1;
    cv::AutoBuffer<char> name_buf(_filename.size()+1);

    while( ptr >= filename && *ptr != '\\' && *ptr != '/' && *ptr != ':' )
    {
        if( *ptr == '.' && (!*ptr2 || strncmp(ptr2, ".gz", 3) == 0) )
            ptr2 = ptr;
        ptr--;
    }
    ptr++;
    if( ptr == ptr2 )
        CV_Error( cv::Error::StsBadArg, "Invalid filename" );

    char* name = name_buf.data();

    // name must start with letter or '_'
    if( !cv_isalpha(*ptr) && *ptr!= '_' ){
        *name++ = '_';
    }

    while( ptr < ptr2 )
    {
        char c = *ptr++;
        if( !cv_isalnum(c) && c != '-' && c != '_' )
            c = '_';
        *name++ = c;
    }
    *name = '\0';
    name = name_buf.data();
    if( strcmp( name, "_" ) == 0 )
        strcpy( name, stubname );
    return name;
}


int FileStorage::getFormat() const
{
    return p->fmt;
}

FileNode FileStorage::operator [](const char* key) const
{
    return this->operator[](std::string(key));
}

FileNode FileStorage::operator [](const std::string& key) const
{
    FileNode res;
    for (size_t i = 0; i < p->roots.size(); i++)
    {
        res = p->roots[i][key];
        if (!res.empty())
            break;
    }
    return res;
}

String FileStorage::releaseAndGetString()
{
    String buf;
    p->release(&buf);
    return buf;
}

void FileStorage::writeRaw( const String& fmt, const void* vec, size_t len )
{
    p->writeRawData(fmt, (const uchar*)vec, len);
}

void FileStorage::writeComment( const String& comment, bool eol_comment )
{
    p->writeComment(comment.c_str(), eol_comment);
}

void writeScalar( FileStorage& fs, int value )
{
    fs.p->write(String(), value);
}

void writeScalar( FileStorage& fs, int64_t value )
{
    fs.p->write(String(), value);
}

void writeScalar( FileStorage& fs, float value )
{
    fs.p->write(String(), (double)value);
}

void writeScalar( FileStorage& fs, double value )
{
    fs.p->write(String(), value);
}

void writeScalar( FileStorage& fs, const String& value )
{
    fs.p->write(String(), value);
}

void write( FileStorage& fs, const String& name, int value )
{
    fs.p->write(name, value);
}

void write( FileStorage& fs, const String& name, int64_t value )
{
    fs.p->write(name, value);
}

void write( FileStorage& fs, const String& name, float value )
{
    fs.p->write(name, (double)value);
}

void write( FileStorage& fs, const String& name, double value )
{
    fs.p->write(name, value);
}

void write( FileStorage& fs, const String& name, const String& value )
{
    fs.p->write(name, value);
}

void FileStorage::write(const String& name, int val) { p->write(name, val); }
void FileStorage::write(const String& name, int64_t val) { p->write(name, val); }
void FileStorage::write(const String& name, double val) { p->write(name, val); }
void FileStorage::write(const String& name, const String& val) { p->write(name, val); }
void FileStorage::write(const String& name, const Mat& val) { cv::write(*this, name, val); }
void FileStorage::write(const String& name, const std::vector<String>& val) { cv::write(*this, name, val); }

FileStorage& operator << (FileStorage& fs, const String& str)
{
    enum { NAME_EXPECTED = FileStorage::NAME_EXPECTED,
        VALUE_EXPECTED = FileStorage::VALUE_EXPECTED,
        INSIDE_MAP = FileStorage::INSIDE_MAP };
    const char* _str = str.c_str();
    if( !fs.isOpened() || !_str )
        return fs;
    Ptr<FileStorage::Impl>& fs_impl = fs.p;
    char c = *_str;

    if( c == '}' || c == ']' )
    {
        if( fs_impl->write_stack.empty() )
            CV_Error_( cv::Error::StsError, ("Extra closing '%c'", *_str) );

        fs_impl->workaround();

        int struct_flags = fs_impl->write_stack.back().flags;
        char expected_bracket = FileNode::isMap(struct_flags) ? '}' : ']';
        if( c != expected_bracket )
            CV_Error_( cv::Error::StsError, ("The closing '%c' does not match the opening '%c'", c, expected_bracket));
        fs_impl->endWriteStruct();
        CV_Assert(!fs_impl->write_stack.empty());
        struct_flags = fs_impl->write_stack.back().flags;
        fs.state = FileNode::isMap(struct_flags) ? INSIDE_MAP + NAME_EXPECTED : VALUE_EXPECTED;
        fs.elname = String();
    }
    else if( fs.state == NAME_EXPECTED + INSIDE_MAP )
    {
        if (!cv_isalpha(c) && c != '_')
            CV_Error_( cv::Error::StsError, ("Incorrect element name %s; should start with a letter or '_'", _str) );
        fs.elname = str;
        fs.state = VALUE_EXPECTED + INSIDE_MAP;
    }
    else if( (fs.state & 3) == VALUE_EXPECTED )
    {
        if( c == '{' || c == '[' )
        {
            int struct_flags = c == '{' ? FileNode::MAP : FileNode::SEQ;
            fs.state = struct_flags == FileNode::MAP ? INSIDE_MAP + NAME_EXPECTED : VALUE_EXPECTED;
            _str++;
            if( *_str == ':' )
            {
                _str++;
                if( !*_str )
                    struct_flags |= FileNode::FLOW;
            }
            fs_impl->startWriteStruct(!fs.elname.empty() ? fs.elname.c_str() : 0, struct_flags, *_str ? _str : 0 );
            fs.elname = String();
        }
        else
        {
            write( fs, fs.elname, (c == '\\' && (_str[1] == '{' || _str[1] == '}' ||
                                _str[1] == '[' || _str[1] == ']')) ? String(_str+1) : str );
            if( fs.state == INSIDE_MAP + VALUE_EXPECTED )
                fs.state = INSIDE_MAP + NAME_EXPECTED;
        }
    }
    else
        CV_Error( cv::Error::StsError, "Invalid fs.state" );
    return fs;
}


FileNode::FileNode()
    : fs(NULL)
{
    blockIdx = ofs = 0;
}

FileNode::FileNode(FileStorage::Impl* _fs, size_t _blockIdx, size_t _ofs)
    : fs(_fs)
{
    blockIdx = _blockIdx;
    ofs = _ofs;
}

FileNode::FileNode(const FileStorage* _fs, size_t _blockIdx, size_t _ofs)
    : FileNode(_fs->p.get(), _blockIdx, _ofs)
{
    // nothing
}

FileNode::FileNode(const FileNode& node)
{
    fs = node.fs;
    blockIdx = node.blockIdx;
    ofs = node.ofs;
}

FileNode& FileNode::operator=(const FileNode& node)
{
    fs = node.fs;
    blockIdx = node.blockIdx;
    ofs = node.ofs;
    return *this;
}

FileNode FileNode::operator[](const std::string& nodename) const
{
    if(!fs)
        return FileNode();

    CV_Assert( isMap() );

    unsigned key = fs->getStringOfs(nodename);
    size_t i, sz = size();
    FileNodeIterator it = begin();

    for( i = 0; i < sz; i++, ++it )
    {
        FileNode n = *it;
        const uchar* p = n.ptr();
        unsigned key2 = (unsigned)readInt(p + 1);
        CV_Assert( key2 < fs->str_hash_data.size() );
        if( key == key2 )
            return n;
    }
    return FileNode();
}

FileNode FileNode::operator[](const char* nodename) const
{
    return this->operator[](std::string(nodename));
}

FileNode FileNode::operator[](int i) const
{
    if(!fs)
        return FileNode();

    CV_Assert( isSeq() );

    int sz = (int)size();
    CV_Assert( 0 <= i && i < sz );

    FileNodeIterator it = begin();
    it += i;

    return *it;
}

std::vector<String> FileNode::keys() const
{
    CV_Assert(isMap());

    std::vector<String> res;
    res.reserve(size());
    for (FileNodeIterator it = begin(); it != end(); ++it)
    {
        res.push_back((*it).name());
    }
    return res;
}

int FileNode::type() const
{
    const uchar* p = ptr();
    if(!p)
        return NONE;
    return (*p & TYPE_MASK);
}

bool FileNode::isMap(int flags) { return (flags & TYPE_MASK) == MAP; }
bool FileNode::isSeq(int flags) { return (flags & TYPE_MASK) == SEQ; }
bool FileNode::isCollection(int flags) { return isMap(flags) || isSeq(flags); }
bool FileNode::isFlow(int flags) { return (flags & FLOW) != 0; }
bool FileNode::isEmptyCollection(int flags) { return (flags & EMPTY) != 0; }

bool FileNode::empty() const   { return fs == 0; }
bool FileNode::isNone() const  { return type() == NONE; }
bool FileNode::isSeq() const   { return type() == SEQ; }
bool FileNode::isMap() const   { return type() == MAP; }
bool FileNode::isInt() const   { return type() == INT;  }
bool FileNode::isReal() const  { return type() == REAL; }
bool FileNode::isString() const { return type() == STRING;  }
bool FileNode::isNamed() const
{
    const uchar* p = ptr();
    if(!p)
        return false;
    return (*p & NAMED) != 0;
}

std::string FileNode::name() const
{
    const uchar* p = ptr();
    if(!p)
        return std::string();
    size_t nameofs = p[1] | (p[2]<<8) | (p[3]<<16) | (p[4]<<24);
    return fs->getName(nameofs);
}

FileNode::operator int() const
{
    const uchar* p = ptr();
    if(!p)
        return 0;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return readInt(p);
    }
    else if( type == REAL )
    {
        return cvRound(readReal(p));
    }
    else
        return 0x7fffffff;
}

FileNode::operator int64_t() const
{
    const uchar* p = ptr();
    if(!p)
        return 0;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return readLong(p);
    }
    else if( type == REAL )
    {
        return cvRound(readReal(p));
    }
    else
        return 0x7fffffff;
}

FileNode::operator float() const
{
    const uchar* p = ptr();
    if(!p)
        return 0.f;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return (float)readInt(p);
    }
    else if( type == REAL )
    {
        return (float)readReal(p);
    }
    else
        return FLT_MAX;
}

FileNode::operator double() const
{
    const uchar* p = ptr();
    if(!p)
        return 0.f;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return (double)readInt(p);
    }
    else if( type == REAL )
    {
        return readReal(p);
    }
    else
        return DBL_MAX;
}

double FileNode::real() const  { return double(*this); }
std::string FileNode::string() const
{
    const uchar* p = ptr();
    if( !p || (*p & TYPE_MASK) != STRING )
        return std::string();
    p += (*p & NAMED) ? 5 : 1;
    size_t sz = (size_t)(unsigned)readInt(p);
    return std::string((const char*)(p + 4), sz - 1);
}
Mat FileNode::mat() const { Mat value; read(*this, value, Mat()); return value; }

FileNodeIterator FileNode::begin() const { return FileNodeIterator(*this, false); }
FileNodeIterator FileNode::end() const   { return FileNodeIterator(*this, true); }

void FileNode::readRaw( const std::string& fmt, void* vec, size_t len ) const
{
    FileNodeIterator it = begin();
    it.readRaw( fmt, vec, len );
}

size_t FileNode::size() const
{
    const uchar* p = ptr();
    if( !p )
        return 0;
    int tag = *p;
    int tp = tag & TYPE_MASK;
    if( tp == MAP || tp == SEQ )
    {
        if( tag & NAMED )
            p += 4;
        return (size_t)(unsigned)readInt(p + 5);
    }
    return tp != NONE;
}

size_t FileNode::rawSize() const
{
    const uchar* p0 = ptr(), *p = p0;
    if( !p )
        return 0;
    int tag = *p++;
    int tp = tag & TYPE_MASK;
    if( tag & NAMED )
        p += 4;
    size_t sz0 = (size_t)(p - p0);
    if( tp == INT )
        return sz0 + 8;
    if( tp == REAL )
        return sz0 + 8;
    if( tp == NONE )
        return sz0;
    CV_Assert( tp == STRING || tp == SEQ || tp == MAP );
    return sz0 + 4 + readInt(p);
}

uchar* FileNode::ptr()
{
    return !fs ? 0 : (uchar*)fs->getNodePtr(blockIdx, ofs);
}

const uchar* FileNode::ptr() const
{
    return !fs ? 0 : fs->getNodePtr(blockIdx, ofs);
}

void FileNode::setValue( int type, const void* value, int len )
{
    uchar *p = ptr();
    CV_Assert(p != 0);

    int tag = *p;
    int current_type = tag & TYPE_MASK;
    CV_Assert( current_type == NONE || current_type == type );

    int sz = 1;

    if( tag & NAMED )
        sz += 4;

    if( type == INT )
        sz += 8;
    else if( type == REAL )
        sz += 8;
    else if( type == STRING )
    {
        if( len < 0 )
            len = (int)strlen((const char*)value);
        sz += 4 + len + 1; // besides the string content,
                           // take the size (4 bytes) and the final '\0' into account
    }
    else
        CV_Error(Error::StsNotImplemented, "Only scalar types can be dynamically assigned to a file node");

    p = fs->reserveNodeSpace(*this, sz);
    *p++ = (uchar)(type | (tag & NAMED));
    if( tag & NAMED )
        p += 4;

    if( type == INT )
    {
        int64_t ival = *(const int64_t*)value;
        writeInt(p, ival);
    }
    else if( type == REAL )
    {
        double dbval = *(const double*)value;
        writeReal(p, dbval);
    }
    else if( type == STRING )
    {
        const char* str = (const char*)value;
        writeInt(p, len + 1);
        memcpy(p + 4, str, len);
        p[4 + len] = (uchar)'\0';
    }
}

FileNodeIterator::FileNodeIterator()
{
    fs = 0;
    blockIdx = 0;
    ofs = 0;
    blockSize = 0;
    nodeNElems = 0;
    idx = 0;
}

FileNodeIterator::FileNodeIterator( const FileNode& node, bool seekEnd )
{
    fs = node.fs;
    idx = 0;
    if( !fs )
        blockIdx = ofs = blockSize = nodeNElems = 0;
    else
    {
        blockIdx = node.blockIdx;
        ofs = node.ofs;

        bool collection = node.isSeq() || node.isMap();
        if( node.isNone() )
        {
            nodeNElems = 0;
        }
        else if( !collection )
        {
            nodeNElems = 1;
            if( seekEnd )
            {
                idx = 1;
                ofs += node.rawSize();
            }
        }
        else
        {
            nodeNElems = node.size();
            const uchar* p0 = node.ptr(), *p = p0 + 1;
            if(*p0 & FileNode::NAMED )
                p += 4;
            if( !seekEnd )
                ofs += (p - p0) + 8;
            else
            {
                size_t rawsz = (size_t)(unsigned)readInt(p);
                ofs += (p - p0) + 4 + rawsz;
                idx = nodeNElems;
            }
        }
        fs->normalizeNodeOfs(blockIdx, ofs);
        blockSize = fs->fs_data_blksz[blockIdx];
    }
}

FileNodeIterator::FileNodeIterator(const FileNodeIterator& it)
{
    fs = it.fs;
    blockIdx = it.blockIdx;
    ofs = it.ofs;
    blockSize = it.blockSize;
    nodeNElems = it.nodeNElems;
    idx = it.idx;
}

FileNodeIterator& FileNodeIterator::operator=(const FileNodeIterator& it)
{
    fs = it.fs;
    blockIdx = it.blockIdx;
    ofs = it.ofs;
    blockSize = it.blockSize;
    nodeNElems = it.nodeNElems;
    idx = it.idx;
    return *this;
}

FileNode FileNodeIterator::operator *() const
{
    return FileNode(idx < nodeNElems ? fs : NULL, blockIdx, ofs);
}

FileNodeIterator& FileNodeIterator::operator ++ ()
{
    if( idx == nodeNElems || !fs )
        return *this;
    idx++;
    FileNode n(fs, blockIdx, ofs);
    ofs += n.rawSize();
    if( ofs >= blockSize )
    {
        fs->normalizeNodeOfs(blockIdx, ofs);
        blockSize = fs->fs_data_blksz[blockIdx];
    }
    return *this;
}

FileNodeIterator FileNodeIterator::operator ++ (int)
{
    FileNodeIterator it = *this;
    ++(*this);
    return it;
}

FileNodeIterator& FileNodeIterator::operator += (int _ofs)
{
    CV_Assert( _ofs >= 0 );
    for( ; _ofs > 0; _ofs-- )
        this->operator ++();
    return *this;
}

FileNodeIterator& FileNodeIterator::readRaw( const String& fmt, void* _data0, size_t maxsz)
{
    if( fs && idx < nodeNElems )
    {
        uchar* data0 = (uchar*)_data0;
        int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2];
        int fmt_pair_count = fs::decodeFormat( fmt.c_str(), fmt_pairs, CV_FS_MAX_FMT_PAIRS );
        size_t esz = fs::calcStructSize( fmt.c_str(), 0 );

        CV_Assert( maxsz % esz == 0 );
        maxsz /= esz;

        for( ; maxsz > 0; maxsz--, data0 += esz )
        {
            size_t offset = 0;
            for( int k = 0; k < fmt_pair_count; k++ )
            {
                int elem_type = fmt_pairs[k*2+1];
                int elem_size = CV_ELEM_SIZE(elem_type);

                int count = fmt_pairs[k*2];
                offset = alignSize( offset, elem_size );
                uchar* data = data0 + offset;

                for( int i = 0; i < count; i++, ++(*this) )
                {
                    FileNode node = *(*this);
                    if( node.isInt() )
                    {
                        int64_t ival = static_cast<int64_t>(elem_size == 8 ? (int64_t)node : (int)node);
                        switch( elem_type )
                        {
                        case CV_8U:
                            *(uchar*)data = saturate_cast<uchar>(ival);
                            data++;
                            break;
                        case CV_8S:
                            *(char*)data = saturate_cast<schar>(ival);
                            data++;
                            break;
                        case CV_Bool:
                            *(bool*)data = ival != 0;
                            data++;
                            break;
                        case CV_16U:
                            *(ushort*)data = saturate_cast<ushort>(ival);
                            data += sizeof(ushort);
                            break;
                        case CV_16S:
                            *(short*)data = saturate_cast<short>(ival);
                            data += sizeof(short);
                            break;
                        case CV_32U:
                            *(unsigned*)data = (unsigned)std::max(ival, (int64_t)0);
                            data += sizeof(unsigned);
                            break;
                        case CV_32S:
                            *(int*)data = (int)ival;
                            data += sizeof(int);
                            break;
                        case CV_32F:
                            *(float*)data = (float)ival;
                            data += sizeof(float);
                            break;
                        case CV_64U:
                            *(uint64_t*)data = (uint64_t)ival;
                            data += sizeof(uint64_t);
                            break;
                        case CV_64S:
                            *(int64_t*)data = (int64_t)ival;
                            data += sizeof(int64_t);
                            break;
                        case CV_64F:
                            *(double*)data = (double)ival;
                            data += sizeof(double);
                            break;
                        case CV_16F:
                            *(hfloat*)data = hfloat((float)ival);
                            data += sizeof(hfloat);
                            break;
                        case CV_16BF:
                            *(bfloat*)data = bfloat((float)ival);
                            data += sizeof(bfloat);
                            break;
                        default:
                            CV_Error( Error::StsUnsupportedFormat, "Unsupported type" );
                        }
                    }
                    else if( node.isReal() )
                    {
                        double fval = (double)node;

                        switch( elem_type )
                        {
                        case CV_8U:
                            *(uchar*)data = saturate_cast<uchar>(fval);
                            data++;
                            break;
                        case CV_8S:
                            *(char*)data = saturate_cast<schar>(fval);
                            data++;
                            break;
                        case CV_16U:
                            *(ushort*)data = saturate_cast<ushort>(fval);
                            data += sizeof(ushort);
                            break;
                        case CV_16S:
                            *(short*)data = saturate_cast<short>(fval);
                            data += sizeof(short);
                            break;
                        case CV_32U:
                            *(int*)data = saturate_cast<unsigned>(fval);
                            data += sizeof(int);
                            break;
                        case CV_32S:
                            *(int*)data = saturate_cast<int>(fval);
                            data += sizeof(int);
                            break;
                        case CV_32F:
                            *(float*)data = (float)fval;
                            data += sizeof(float);
                            break;
                        case CV_64U:
                            *(uint64_t*)data = (uint64_t)round(std::max(fval, 0.));
                            data += sizeof(uint64_t);
                            break;
                        case CV_64S:
                            *(int64_t*)data = (int64_t)round(std::max(fval, 0.));
                            data += sizeof(int64_t);
                            break;
                        case CV_64F:
                            *(double*)data = fval;
                            data += sizeof(double);
                            break;
                        case CV_16F:
                            *(hfloat*)data = hfloat((float)fval);
                            data += sizeof(hfloat);
                            break;
                        case CV_16BF:
                            *(bfloat*)data = bfloat((float)fval);
                            data += sizeof(bfloat);
                            break;
                        default:
                            CV_Error( Error::StsUnsupportedFormat, "Unsupported type" );
                        }
                    }
                    else
                        CV_Error( Error::StsError, "readRawData can only be used to read plain sequences of numbers" );
                }
                offset = (int)(data - data0);
            }
        }
    }

    return *this;
}

bool FileNodeIterator::equalTo(const FileNodeIterator& it) const
{
    return fs == it.fs && blockIdx == it.blockIdx && ofs == it.ofs &&
           idx == it.idx && nodeNElems == it.nodeNElems;
}

size_t FileNodeIterator::remaining() const
{
    return nodeNElems - idx;
}

bool operator == ( const FileNodeIterator& it1, const FileNodeIterator& it2 )
{
    return it1.equalTo(it2);
}

bool operator != ( const FileNodeIterator& it1, const FileNodeIterator& it2 )
{
    return !it1.equalTo(it2);
}

void read(const FileNode& node, int& val, int default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (int)node;
    }
}

void read(const FileNode& node, int64_t& val, int64_t default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (int64_t)node;
    }
}

void read(const FileNode& node, double& val, double default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (double)node;
    }
}

void read(const FileNode& node, float& val, float default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (float)node;
    }
}

void read(const FileNode& node, std::string& val, const std::string& default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (std::string)node;
    }
}

FileStorage_API::~FileStorage_API() {}

namespace internal
{

WriteStructContext::WriteStructContext(FileStorage& _fs, const std::string& name,
                                       int flags, const std::string& typeName)
{
    fs = &_fs;
    fs->startWriteStruct(name, flags, typeName);
}

WriteStructContext::~WriteStructContext()
{
    fs->endWriteStruct();
}

}

}
