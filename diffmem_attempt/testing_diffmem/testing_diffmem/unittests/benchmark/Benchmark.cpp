#include "Benchmark.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <climits>
#include <cmath>

namespace internal {

#if __MACH__
#include <errno.h>
#include <mach/mach_time.h>

	static mach_timebase_info_data_t tb_info;
	static bool tb_init = mach_timebase_info(&tb_info) == KERN_SUCCESS;

	int clock_gettime(clockid_t clk_id, struct timespec* ts) {
		if (!tb_init) {
			errno = EINVAL;
			return -1;
		}

		uint64_t now_ticks = mach_absolute_time();
		uint64_t now_ns = (now_ticks * tb_info.numer) / tb_info.denom;
		ts->tv_sec = now_ns / 1000000000;
		ts->tv_nsec = now_ns % 1000000000;

		return 0;
	}

	int clock_getres(clockid_t clk_id, struct timespec* ts) {
		if (!tb_init) {
			errno = EINVAL;
			return -1;
		}

		ts->tv_sec = 0;
		ts->tv_nsec = tb_info.numer / tb_info.denom;

		return 0;
	}
#elif defined(_WIN32) || defined(_WIN64) || defined(_WINNT)
	// The MSVC version has been extracted from the pthreads implemenation here:
	// https://github.com/songdongsheng/libpthread
	// Copyright(c) 2011, Dongsheng Song <songdongsheng@live.cn>
	//
	// It is under the Apache License Version 2.0, just as the rest of the file is.
	// It has been mostly stripped down to what we have.

#include <windows.h>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#define DELTA_EPOCH_IN_100NS    INT64_C(116444736000000000)
#define POW10_7     INT64_C(10000000)
#define POW10_9     INT64_C(1000000000)

typedef uint8_t clockid_t;
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID 3

	int clock_getres(clockid_t clock_id, struct timespec *res)
	{
		switch (clock_id) {
		case CLOCK_MONOTONIC:
		{
			LARGE_INTEGER pf;

			if (QueryPerformanceFrequency(&pf) == 0)
				return -1;

			res->tv_sec = 0;
			res->tv_nsec = (int)((POW10_9 + (pf.QuadPart >> 1)) / pf.QuadPart);
			if (res->tv_nsec < 1)
				res->tv_nsec = 1;

			return 0;
		}

		case CLOCK_REALTIME:
		case CLOCK_PROCESS_CPUTIME_ID:
		case CLOCK_THREAD_CPUTIME_ID:
		{
			DWORD   timeAdjustment, timeIncrement;
			BOOL    isTimeAdjustmentDisabled;

			(void)GetSystemTimeAdjustment(&timeAdjustment, &timeIncrement, &isTimeAdjustmentDisabled);
			res->tv_sec = 0;
			res->tv_nsec = timeIncrement * 100;

			return 0;
		}
		default:
			break;
		}

		return -1;
	}

	int clock_gettime(clockid_t clock_id, struct timespec *tp)
	{
		unsigned __int64 t;
		LARGE_INTEGER pf, pc;
		union {
			unsigned __int64 u64;
			FILETIME ft;
		}  ct, et, kt, ut;

		switch (clock_id) {
		case CLOCK_REALTIME:
		{
			GetSystemTimeAsFileTime(&ct.ft);
			t = ct.u64 - DELTA_EPOCH_IN_100NS;
			tp->tv_sec = t / POW10_7;
			tp->tv_nsec = ((int)(t % POW10_7)) * 100;

			return 0;
		}

		case CLOCK_MONOTONIC:
		{
			if (QueryPerformanceFrequency(&pf) == 0)
				return -1;

			if (QueryPerformanceCounter(&pc) == 0)
				return -1;

			tp->tv_sec = pc.QuadPart / pf.QuadPart;
			tp->tv_nsec = (int)(((pc.QuadPart % pf.QuadPart) * POW10_9 + (pf.QuadPart >> 1)) / pf.QuadPart);
			if (tp->tv_nsec >= POW10_9) {
				tp->tv_sec++;
				tp->tv_nsec -= POW10_9;
			}

			return 0;
		}

		case CLOCK_PROCESS_CPUTIME_ID:
		{
			if (0 == GetProcessTimes(GetCurrentProcess(), &ct.ft, &et.ft, &kt.ft, &ut.ft))
				return -1;
			t = kt.u64 + ut.u64;
			tp->tv_sec = t / POW10_7;
			tp->tv_nsec = ((int)(t % POW10_7)) * 100;

			return 0;
		}

		case CLOCK_THREAD_CPUTIME_ID:
		{
			if (0 == GetThreadTimes(GetCurrentThread(), &ct.ft, &et.ft, &kt.ft, &ut.ft))
				return -1;
			t = kt.u64 + ut.u64;
			tp->tv_sec = t / POW10_7;
			tp->tv_nsec = ((int)(t % POW10_7)) * 100;

			return 0;
		}

		default:
			break;
		}

		return -1;
	}
#endif

	inline void safe_clock_getres(struct timespec *tp) {
		int ret = clock_getres(CLOCK_MONOTONIC, tp);
		if( ret != 0 ) {
			fprintf(stderr, "clock_getres failed");
			assert(ret != 0);
		}
	}

	inline void safe_clock_gettime(struct timespec *tp) {
		int ret = clock_gettime(CLOCK_MONOTONIC, tp);
		if( ret != 0 ) {
			fprintf(stderr, "clock_gettime failed");
			assert(ret != 0);
		}
	}

	#define BENCHMARK_PREFIX "bench"
	// A test filter that matches everything.
	constexpr char kUniversalFilter[] = "*";
	constexpr char kDefaultOutput[] = "benchmark.xml";

	static std::vector<double> p = { .975, .95, .5};

	uint32_t max_secs = 1;
	uint64_t min_usecs = 100;
	uint32_t min_iters = 1;
	uint32_t warmup_rounds = 5;
	std::string filter = kUniversalFilter;
	std::string output = kDefaultOutput;

	std::vector<BenchmarkInfo*> & benchmarks() {
	  static std::vector<BenchmarkInfo*> benchmarks;
	  return benchmarks;
	}

	inline uint64_t timespecDiff(timespec end, timespec start) {
	  auto diff = uint64_t(end.tv_sec - start.tv_sec);
	  return diff * 1000000000UL + end.tv_nsec - start.tv_nsec;
	}

	// Taken from google-test
	// Returns true iff the wildcard pattern matches the string.  The
	// first ':' or '\0' character in pattern marks the end of it.
	//
	// This recursive algorithm isn't very efficient, but is clear and
	// works well enough for matching test names, which are short.
	bool PatternMatchesString(const char *pattern, const char *str) {
		switch( *pattern) {
			case '\0':
			case ':':  // Either ':' or '\0' marks the end of the pattern.
				return *str == '\0';
			case '?':  // Matches any single character.
				return *str != '\0' && PatternMatchesString(pattern + 1, str + 1);
			case '*':  // Matches any string (possibly empty) of characters.
				return (*str != '\0' && PatternMatchesString(pattern, str + 1))
						|| PatternMatchesString(pattern + 1, str);
			default:  // Non-special character.  Matches itself.
				return *pattern == *str && PatternMatchesString(pattern + 1, str + 1);
		}
	}

	bool MatchesFilter(const std::string & name, const std::string & filter) {
		const char *cur_pattern = filter.c_str();
		for(;;) {
			if( PatternMatchesString(cur_pattern, name.c_str()) ) {
				return true;
			}

			// Finds the next pattern in the filter.
			cur_pattern = strchr(cur_pattern, ':');

			// Returns if no more pattern can be found.
			if( cur_pattern == nullptr )
				break;

			// Skips the pattern separator (the ':' character).
			cur_pattern++;
		}

		return false;
	}

	BenchmarkInfo addBenchmark(BenchmarkInfo *info, const char *case_name, const char *name, FactoryBase *const factory) {
		benchmarks().push_back(info);
		return BenchmarkInfo(case_name, name, factory);
	}

	template <typename IT>
	std::vector<double> quantile(IT first, IT last, std::vector<double> & ps) {
		std::vector<double> qs;
		qs.reserve(ps.size());

		const unsigned int n = last - first;
		std::sort(first, last);

		for(auto p : ps) {
			double q;
			double pn = n * p;
			unsigned int j = static_cast<int>(std::floor(pn));

			if( j == 0 ) {
				q = *first;
			} else if( j == n ) {
				q = *last;
			} else {
				double g = pn - j;
				q = (1.0 - g) * *(first + j - 1) + g * *(first + j);
			}

			qs.push_back(q);
		}
		return qs;
	}

	static double estimateTime(double *begin, double *end) {
		return *std::min_element(begin, end);
	}

	void BenchmarkInfo::run() {
		static uint64_t resolutionInNs = 0;
		if(resolutionInNs == 0u) {
			timespec ts;
			safe_clock_getres(&ts);
			resolutionInNs = ts.tv_nsec;
		}

		// We choose a minimum of 100,000 nanoseconds, but if
		// the clock resolution is worse than that, it will be larger. In
		// essence we're aiming at making the quantization noise 0.01%.
		static const auto minNanoseconds =
				std::max<uint64_t>(min_usecs * 1000UL,
						std::min<uint64_t>(resolutionInNs, 1000000000ULL));

		Benchmark *benchmark = factory_->create();
		// Warmup and figure out amount of amortizing
		unsigned int n = min_iters;
		for(unsigned int k = 0; k < warmup_rounds; ++k) {
			for(; n < (1UL << 30); n *= 2) {
				auto const nsecsAndIter = benchmark->run(n);
				if( nsecsAndIter.first < minNanoseconds )
					continue;

				break;
			}
		}

		// We do measurements in several epochs and take the minimum, to account for jitter.
		static const unsigned int epochs = 1000;
		// We establish a total time budget as we don't want a measurement
		// to take too long. This will curtail the number of actual epochs.
		const uint64_t timeBudgetInNs = max_secs * 1000000000ULL;
		timespec global;
		safe_clock_gettime(&global);

		double epochResults[epochs] = { 0 };
		size_t actualEpochs;

		for(actualEpochs = 0; actualEpochs < epochs; ++actualEpochs) {
			auto const nsecsAndIter = benchmark->run(n);
			epochResults[actualEpochs] = std::max(0.0, double(nsecsAndIter.first) / nsecsAndIter.second);

			timespec now;
			safe_clock_gettime(&now);
			if( timespecDiff(now, global) >= timeBudgetInNs ) {
				// No more time budget available.
				++actualEpochs;
				break;
			}
		}

		delete benchmark;

		// If the benchmark was basically drowned in baseline noise, it's
		// possible it became negative.
		time_ = std::max(0.0, estimateTime(epochResults, epochResults + actualEpochs));
		quantiles_ = quantile(epochResults, epochResults + actualEpochs, p );
		nsamples_ = actualEpochs;
	}

	void BenchmarkInfo::filter(const std::string & positive, const std::string & negative) {
		const std::string & full_name = std::string(caseName_) + "." + name_;
		shouldRun_ = (MatchesFilter(full_name, positive) && ! MatchesFilter(full_name, negative));
	}

	Benchmark::Benchmark() {
	}

	Benchmark::~Benchmark() {
	}

	void Benchmark::setUp() {
	}

	void Benchmark::tearDown() {
	}

	TimeIterPair Benchmark::run(unsigned int ntimes) {
	    BenchmarkSuspender::nsSpent = 0;
	    timespec start, end;
	    unsigned int niters = 0;

	    setUp();

	    safe_clock_gettime(&start);
	    niters = body(ntimes);
	    safe_clock_gettime(&end);

	    tearDown();
	    return internal::TimeIterPair(timespecDiff(end, start) - BenchmarkSuspender::nsSpent, niters);
	}

}

using namespace internal;

BenchmarkSuspender::NanosecondsSpent BenchmarkSuspender::nsSpent;

BenchmarkSuspender::BenchmarkSuspender() {
	safe_clock_gettime(&start);
}

void BenchmarkSuspender::tally() {
	timespec end;
	safe_clock_gettime(&end);
	nsSpent += timespecDiff(end, start);
	start = end;
}

// Returns true iff the user-specified filter matches the test case
// name and the test name.
static void FilterTests(const std::string & filter) {
	// Split filter at '-', if there is one, to separate into
	// positive filter and negative filter portions
	const char* const p = filter.c_str();
	const char* const dash = strchr(p, '-');
	std::string positive;
	std::string negative;
	if( dash == NULL ) {
		positive = filter;  // Whole string is a positive filter
		negative = "";
	} else {
		positive = std::string(p, dash);   // Everything up to the dash
		negative = std::string(dash + 1);  // Everything after the dash
		if( positive.empty() )
			positive = kUniversalFilter;
	}

	std::for_each(benchmarks().begin(), benchmarks().end(),
		[&positive, &negative](BenchmarkInfo *info) {
			info->filter(positive, negative);
		}
	);
}

static void WriteToCSV(const std::string & output) {
	if( output.empty() )
		return;

	std::ofstream f( output.c_str() );
	std::for_each(benchmarks().begin(), benchmarks().end(),
		[&](BenchmarkInfo *info) {
			f << info->caseName() << ", " << info->name() << ", ";
			if( info->shouldRun() ) {
				auto nsPerIter = info->timePerIter();
				auto secPerIter = nsPerIter / 1e9;
				auto itersPerSec = (secPerIter == 0) ? std::numeric_limits<double>::infinity() : (1 / secPerIter);
				f << secPerIter << ", " << itersPerSec << ", " << info->numSamples();

				auto qs = info->quantiles();
				for(auto q : qs)
					f << ", " << (q / 1e9);
			} else {
				f << "NA, NA";
				for(auto x : p) {
					(void)(x);
					f << ", " << "NA";
				}
			}

			 f << std::endl;
		}
	);
}

void runAllBenchmarks() {
	FilterTests(filter);

	std::for_each(benchmarks().begin(), benchmarks().end(),
		[](BenchmarkInfo *info) {
			if( info->shouldRun() )
				info->run();
		}
	);

	WriteToCSV(output);
}

static constexpr char kHelpMessage[] =
"This program contains tests written using Benchmarker. You can use the\n"
"following command line flags to control its behavior:\n"
"\n"
"Benchmark Selection:\n"
"  --" BENCHMARK_PREFIX "_list\n"
"      List the names of all tests instead of running them.\n"
"  --" BENCHMARK_PREFIX "_filter=POSTIVE_PATTERNS"
    "[-NEGATIVE_PATTERNS]\n"
"      Run only the tests whose name matches one of the positive patterns but\n"
"      none of the negative patterns. '?' matches any single character; '*'\n"
"      matches any substring; ':' separates two patterns.\n"
"\n"
"Benchmark Execution:\n"
"  --" BENCHMARK_PREFIX "_max_secs\n"
"      Maximum amount of time (in seconds) for benchmarks to run.\n"
"  --" BENCHMARK_PREFIX "_min_usecs\n"
"      Minimum amount of time (in microseconds) for benchmarks to run.\n"
"  --" BENCHMARK_PREFIX "_min_iters\n"
"      Minimum amount of iterations for benchmarks to run.\n"
"\n"
"Test Output:\n"
"  --" BENCHMARK_PREFIX "_output=csv[FILE_PATH]\n"
"      Generate an CSV report in the given directory or with the given file\n"
"      name.\n"
"\n";

// Parses a string as a command line flag.  The string should have
// the format "--flag=value".  When def_optional is true, the "=value"
// part can be omitted.
//
// Returns the value of the flag, or NULL if the parsing failed.
static const char *ParseFlagValue(const char* str, const char* flag, bool def_optional) {
	// str and flag must not be NULL.
	if( str == NULL || flag == NULL )
		return NULL;

	const size_t flag_len = strlen(flag);
	if( strncmp(str, flag, flag_len) != 0 )
		return NULL;

	// Skips the flag name.
	const char* flag_end = str + flag_len;

	// When def_optional is true, it's OK to not have a "=value" part.
	if( def_optional && (flag_end[0] == '\0') ) {
		return flag_end;
	}

	// If def_optional is true and there are more characters after the
	// flag name, or if def_optional is false, there must be a '=' after
	// the flag name.
	if( flag_end[0] != '=' )
		return NULL;

	// Returns the string after "=".
	return flag_end + 1;
}

static bool ParseBoolFlag(const char* str, const char* flag, bool* value) {
	// Gets the value of the flag as a string.
	const char* const value_str = ParseFlagValue(str, flag, true);

	// Aborts if the parsing failed.
	if( value_str == NULL )
		return false;

	// Converts the string value to a bool.
	*value = !(*value_str == '0' || *value_str == 'f' || *value_str == 'F');
	return true;
}

static bool ParseStringFlag(const char* str, const char* flag, std::string* value) {
	// Gets the value of the flag as a string.
	const char* const value_str = ParseFlagValue(str, flag, false);

	// Aborts if the parsing failed.
	if( value_str == NULL )
		return false;

	// Sets *value to the value of the flag.
	*value = value_str;
	return true;
}

static bool ParseUInt64Flag(const char *str, const char *flag, uint64_t *value) {
	// Gets the value of the flag as a string.
	const char* const value_str = ParseFlagValue(str, flag, false);

	// Aborts if the parsing failed.
	if( value_str == NULL )
		return false;

	char* end = NULL;
	const unsigned long long long_value = strtoull(value_str, &end, 10);

	// Has strtoull() consumed all characters in the string?
	if( *end != '\0' ) {
		// No - an invalid character was encountered.
		printf("WARNING: %s is expected to be a 64-bit integer\n", value_str);
		fflush(stdout);
		return false;
	}

	const uint64_t result = static_cast<uint64_t>(long_value);
	if( long_value == ULLONG_MAX || // The parsed value overflows. (strtoull() returns ULLONG_MAX when the input overflows.)
			result != long_value
			// The parsed value overflows as a uint64_t.
					) {
		printf("WARNING: %s is expected to be a 64-bit integer, but actually overflows.\n", value_str);
		fflush(stdout);
		return false;
	}

	// Sets *value to the value of the flag.
	*value = result;
	return true;
}

static bool ParseUInt32Flag(const char *str, const char *flag, uint32_t *value) {
	// Gets the value of the flag as a string.
	const char* const value_str = ParseFlagValue(str, flag, false);

	// Aborts if the parsing failed.
	if( value_str == NULL )
		return false;

	char* end = NULL;
	const unsigned long long_value = strtoul(value_str, &end, 10);

	// Has strtoul() consumed all characters in the string?
	if( *end != '\0' ) {
		// No - an invalid character was encountered.
		printf("WARNING: %s is expected to be a 32-bit integer\n", value_str);
		fflush(stdout);
		return false;
	}

	const uint32_t result = static_cast<uint32_t>(long_value);
	if( long_value == ULONG_MAX || // The parsed value overflows. (strtoul() returns ULONG_MAX when the input overflows.)
			result != long_value // The parsed value overflows as a uint32_t.
					) {
		printf("WARNING: %s is expected to be a 32-bit integer, but actually overflows.\n", value_str);
		fflush(stdout);
		return false;
	}

	// Sets *value to the value of the flag.
	*value = result;
	return true;
}

static bool SkipPrefix(const char* prefix, const char** pstr) {
  const size_t prefix_len = strlen(prefix);
  if (strncmp(*pstr, prefix, prefix_len) == 0) {
    *pstr += prefix_len;
    return true;
  }
  return false;
}

static bool HasFlagPrefix(const char **str) {
  return (SkipPrefix("--", str) ||
          SkipPrefix("-", str)) &&
		  (SkipPrefix(BENCHMARK_PREFIX"-", str) || SkipPrefix(BENCHMARK_PREFIX"_", str));
}

bool parseBenchmarkOptions(int *argc, char *argv[]) {
	bool list_benchmarks = false;
	bool help_flag = false;

	for(int i = 1; i < *argc; i++) {
	    bool remove_flag = false;
	    const char *arg = argv[i];
	    bool hasFlagPrefix = HasFlagPrefix(&arg);
	    if( hasFlagPrefix && (ParseBoolFlag(arg, "list", &list_benchmarks) ||
	    		ParseStringFlag(arg, "output", &output) ||
				ParseStringFlag(arg, "filter", &filter) ||
	    		ParseUInt64Flag(arg, "min_usecs", &min_usecs) ||
				ParseUInt32Flag(arg, "max_secs", &max_secs) ||
				ParseUInt32Flag(arg, "min_iters", &min_iters)) ) {
	      remove_flag = true;
	    } else if( strcmp(argv[i], "--help") == 0 ||
	    		strcmp(argv[i], "-h") == 0 ||
				hasFlagPrefix ) {
	      // Both help flag and unrecognized Google Test flags (excluding
	      // internal ones) trigger help display.
	      help_flag = true;
	    }

	    if( remove_flag ) {
	      // Shift the remainder of the argv list left by one.  Note
	      // that argv has (*argc + 1) elements, the last one always being
	      // NULL.  The following loop moves the trailing NULL element as
	      // well.
	      for (int j = i; j != *argc; j++) {
	        argv[j] = argv[j + 1];
	      }

	      // Decrements the argument count.
	      (*argc)--;

	      // We also need to decrement the iterator as we just removed
	      // an element.
	      i--;
	    }
	}

	if( help_flag ) {
		std::cout << kHelpMessage;
		return true;
	}

	if( list_benchmarks ) {
		std::for_each(benchmarks().begin(), benchmarks().end(),
			[](BenchmarkInfo *info) {
				printf("%s.%s\n", info->caseName(), info->name());
			}
		);

		return true;
	}

	return false;
}

struct ScaleInfo {
  double boundary;
  const char* suffix;
};

static const ScaleInfo kTimeSuffixes[] {
  { 365.25 * 24 * 3600, "years" },
  { 24 * 3600, "days" },
  { 3600, "hr" },
  { 60, "min" },
  { 1, "s" },
  { 1E-3, "ms" },
  { 1E-6, "us" },
  { 1E-9, "ns" },
  { 1E-12, "ps" },
  { 1E-15, "fs" },
  { 0, nullptr },
};

static const ScaleInfo kMetricSuffixes[] {
  { 1E24, "Y" },  // yotta
  { 1E21, "Z" },  // zetta
  { 1E18, "X" },  // "exa" written with suffix 'X' so as to not create
                  //   confusion with scientific notation
  { 1E15, "P" },  // peta
  { 1E12, "T" },  // terra
  { 1E9, "G" },   // giga
  { 1E6, "M" },   // mega
  { 1E3, "K" },   // kilo
  { 1, "" },
  { 1E-3, "m" },  // milli
  { 1E-6, "u" },  // micro
  { 1E-9, "n" },  // nano
  { 1E-12, "p" }, // pico
  { 1E-15, "f" }, // femto
  { 1E-18, "a" }, // atto
  { 1E-21, "z" }, // zepto
  { 1E-24, "y" }, // yocto
  { 0, nullptr },
};

static void humanReadable(char *buf, double n, unsigned int decimals, const ScaleInfo* scales) {
	if( std::isinf(n) || std::isnan(n) ) {
		sprintf(buf, "%f", n);
	} else {
		const double absValue = fabs(n);
		const ScaleInfo* scale = scales;
		while( absValue < scale[0].boundary && scale[1].suffix != nullptr )
			++scale;

		const double scaledValue = n / scale->boundary;
		sprintf(buf, "%3.*f%s", decimals, scaledValue, scale->suffix);
	}
}

static const char *readableTime(double n, unsigned int decimals) {
	static char temp[32];
	humanReadable(temp, n, decimals, kTimeSuffixes);
	return temp;
}

static const char *metricReadable(double n, unsigned int decimals) {
	static char temp[32];
	humanReadable(temp, n, decimals, kMetricSuffixes);
	return temp;
}

void printBenchmarkResults() {
	// Width available
	static const unsigned int columns = 80;
	const int roomNeeded = 20 + static_cast<int>(p.size()) * 10 + 6;

	// Compute the longest benchmark name
	size_t longestName = 0;
	std::for_each(benchmarks().begin(), benchmarks().end(),
		[&longestName](BenchmarkInfo *info) {
			longestName = std::max(longestName, strlen(info->name()));
		}
	);

	// Print a horizontal rule
	auto separator = [&](char pad) {
		puts(std::string(columns, pad).c_str());
	};

	// Print header for a file
	auto header = [&](const char *case_name) {
		separator('=');
		printf("%-*s  time/iter  iters/s     n", columns - roomNeeded, case_name);
		for(auto x : p)
			printf("    q%1.3f", x);
		printf("\n");
		separator('=');
	};

	const char *lastCaseName = nullptr;
	std::for_each(benchmarks().begin(), benchmarks().end(),
		[&](BenchmarkInfo *info) {
			if( info->shouldRun() ) {
				const char *caseName = info->caseName();
				if( (lastCaseName == nullptr) || strcmp(caseName, lastCaseName) != 0 ) {
					// New file starting
					header(caseName);
					lastCaseName = caseName;
				}

				std::string s( info->name() );
				s.resize(columns - roomNeeded, ' ');

				auto nsPerIter = info->timePerIter();
				auto secPerIter = nsPerIter / 1e9;
				auto itersPerSec = (secPerIter == 0) ? std::numeric_limits<double>::infinity() : (1 / secPerIter);

				printf("%*s  %9s  %7s  %4zd",
						static_cast<int>(s.size()), s.c_str(),
						readableTime(secPerIter, 2),
						metricReadable(itersPerSec, 2),
						info->numSamples());

				for(auto q : info->quantiles())
					printf(" %9s", readableTime(q/1e9, 2));
				printf("\n");
			}
		}
	);

	separator('=');
}
