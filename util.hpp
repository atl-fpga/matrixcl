//------------------------------------------------------------------------------
//
//  LICENSE: This work is licensed under the Creative Commons
//           Attribution 4.0 International License.
//           To view a copy of this license, visit
//           http://creativecommons.org/licenses/by/4.0/
//           or send a letter to:
//              Creative Commons,
//              444 Castro Street, Suite 900,
//              Mountain View, California, 94041, USA.
//
//------------------------------------------------------------------------------

#ifndef __UTIL_HDR
#define __UTIL_HDR

#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <string>

#include <cstdlib>

namespace util {

inline std::string loadProgram(std::string input)
{
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }

     return std::string(
        std::istreambuf_iterator<char>(stream),
        (std::istreambuf_iterator<char>()));
}

inline std::pair<const void *, ::size_t> loadProgramBinary(std::string input) {
    std::ifstream stream(input.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }

    const size_t size = stream.tellg();
    char *const buf = new char[size];

    stream.seekg(std::ios::beg);
    stream.read(buf, size);
    stream.close();

    return std::make_pair(buf, size);
}

#if 1
class Timer
{
private:
    struct timeval startTime_;

    template <typename T>
    T _max(T a,T b)
    {
        return (a > b ? a : b);
    }

    uint64_t getTime(unsigned long long scale)
    {
        uint64_t ticks;
        // WARNING: THIS IS PROBABLY BROKEN
        struct timeval tv;
        gettimeofday(&tv, 0);
        // check for overflow
        if ((tv.tv_usec - startTime_.tv_usec) < 0)
        {
            // Remove a second from the second field and add it to the
            // microseconds fields to prevent overflow.
            // Then scale.
            ticks = (uint64_t) (tv.tv_sec - startTime_.tv_sec - 1) * scale
                    + (uint64_t) ((1000ULL * 1000ULL) + tv.tv_usec - startTime_.tv_usec)
                                    * scale / (1000ULL * 1000ULL);
        }
        else
        {
            ticks = (uint64_t) (tv.tv_sec - startTime_.tv_sec) * scale
                    + (uint64_t) (tv.tv_usec - startTime_.tv_usec) * scale / (1000ULL * 1000ULL);
        }

        return ticks;
    }

public:
    //! Constructor
    Timer()
    {
        reset();
    }

    //! Destructor
    ~Timer()
    {
    }

    /*!
     * \brief Resets timer such that in essence the elapsed time is zero
     * from this point.
     */
    void reset()
    {
        gettimeofday(&startTime_, 0);
    }

    /*!
     * \brief Calculates the time since the last reset.
     * \returns The time in milli seconds since the last reset.
     */
    uint64_t getTimeMilliseconds(void)
    {
        return getTime(1000ULL);
    }

    /*!
     * \brief Calculates the time since the last reset.
     * \returns The time in nano seconds since the last reset.
     */
    uint64_t getTimeNanoseconds(void)
    {
        return getTime(1000ULL * 1000ULL * 1000ULL);
    }

    /*!
     * \brief Calculates the time since the last reset.
     * \returns The time in micro seconds since the last reset.
     */
    uint64_t getTimeMicroseconds(void)
    {
        return getTime(1000ULL * 1000ULL);
    }

    /*!
     * \brief Calculates the tick rate for millisecond counter.
     */
    float getMillisecondsTickRate(void)
    {
        return 1000.f;
    }

    /*!
     * \brief Calculates the tick rate for nanosecond counter.
     */
    float getNanosecondsTickRate(void)
    {
        return (float) (1000ULL * 1000ULL * 1000ULL);
    }

    /*!
     * \brief Calculates the tick rate for microsecond counter.
     */
    float getMicrosecondsTickRate(void)
    {
        return (float) (1000ULL * 1000ULL);
    }
};
#endif

} // namespace util

#endif // __UTIL_HDR
