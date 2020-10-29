#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <time.h>
#include <functional>
#include <vector>
#include <cstdint>

namespace internal {

	typedef std::pair<uint64_t, unsigned int> TimeIterPair;

	class Benchmark {
		public:
			virtual ~Benchmark();

		public:
			virtual void setUp();
			virtual unsigned int body(unsigned int ntimes) = 0;
			virtual void tearDown();
			TimeIterPair run(unsigned int ntimes);

		protected:
			Benchmark();
	};

	class FactoryBase {
		public:
			virtual ~FactoryBase() {}
			virtual Benchmark *create() = 0;

		protected:
			FactoryBase() {}
	};

	template<class BenchmarkClass>
	class FactoryImpl: public FactoryBase {
		public:
			virtual Benchmark *create() {
				return new BenchmarkClass;
			}
	};

	class BenchmarkInfo {
		public:
			BenchmarkInfo(const char *case_name, const char *name, FactoryBase *const factory) :
				caseName_(case_name), name_(name), factory_(factory), shouldRun_(true), nsamples_(0), time_(0.0) {}

		public:
			void run();
			const char *name() const { return name_; }
			const char *caseName() const { return caseName_; }
			bool shouldRun() const { return shouldRun_; }
			double timePerIter() const { return time_; }
			const std::vector<double> & quantiles() const { return quantiles_; }
			size_t numSamples() const { return nsamples_; }
			void filter(const std::string & positive, const std::string & negative);

		private:
			const char *caseName_;
			const char *name_;
			internal::FactoryBase *const factory_;
			bool shouldRun_;

			size_t nsamples_;
			double time_;
			std::vector<double> quantiles_;
	};

	BenchmarkInfo addBenchmark(BenchmarkInfo *info, const char *case_name, const char *name, FactoryBase *const factory);
}

#if _MSC_VER == 1800
#ifndef noexcept
#define noexcept
#endif
#ifndef constexpr
#define constexpr
#endif
struct timespec {
	time_t   tv_sec;        /* seconds */
	long     tv_nsec;       /* nanoseconds */
};
#endif

class BenchmarkSuspender {
	public:
		BenchmarkSuspender();
		BenchmarkSuspender(const BenchmarkSuspender &) = delete;
		BenchmarkSuspender(BenchmarkSuspender && rhs) noexcept {
			start = rhs.start;
			rhs.start.tv_nsec = 0;
			rhs.start.tv_sec = 0;
		}

		BenchmarkSuspender& operator=(const BenchmarkSuspender &) = delete;
		BenchmarkSuspender& operator=(BenchmarkSuspender && rhs) {
			if( start.tv_nsec > 0 || start.tv_sec > 0 ) {
				tally();
			}
			start = rhs.start;
			rhs.start.tv_nsec = 0;
			rhs.start.tv_sec = 0;
			return *this;
		}

		~BenchmarkSuspender() {
			if( start.tv_nsec > 0 || start.tv_sec > 0 ) {
				tally();
			}
		}

	public:
		/*
		 * Accumulates nanoseconds spent outside benchmark.
		 */
		typedef uint64_t NanosecondsSpent;
		static NanosecondsSpent nsSpent;

		void dismiss() {
			if( start.tv_nsec > 0 || start.tv_sec > 0 ) {
				tally();
			}
			start.tv_nsec = 0;
			start.tv_sec = 0;
		}

	private:
		void tally();
		timespec start;
};

bool parseBenchmarkOptions(int *argc, char *argv[]);
void runAllBenchmarks();
void printBenchmarkResults();

/**
 * Call doNotOptimizeAway(var) against variables that you use for
 * benchmarking but otherwise are useless. The compiler tends to do a
 * good job at eliminating unused variables, and this function fools
 * it into thinking var is in fact needed.
 */
#ifdef _MSC_VER

#pragma optimize("", off)

template <class T>
void doNotOptimizeAway(T&& datum) {
  datum = datum;
}

#pragma optimize("", on)

#elif defined(__clang__)

template <class T>
__attribute__((__optnone__)) void doNotOptimizeAway(T&& /* datum */) {}

#else
template <class T>
void doNotOptimizeAway(T&& datum) {
  asm volatile("" : "+r" (datum));
}
#endif

#define BENCHMARK(case_name, name, param_name)\
	BENCHMARK_(case_name, name, ::internal::Benchmark, param_name)

#define BENCHMARK_MULTI(case_name, name)\
	BENCHMARK_MULTI_(case_name, name, ::internal::Benchmark)

#define BENCHMARK_CLASS_NAME_(bench_case_name, bench_name) \
		bench_case_name##_##bench_name##_Benchmark

#define BENCHMARK_(bench_case_name, bench_name, parent_class, param_name)\
class BENCHMARK_CLASS_NAME_(bench_case_name, bench_name) : public parent_class {\
 public:\
  BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)() {}\
 private:\
  virtual unsigned int body(unsigned int);\
  void body_helper(unsigned int);\
  static ::internal::BenchmarkInfo info_;\
};\
\
::internal::BenchmarkInfo BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)\
  ::info_ =\
    internal::addBenchmark(& BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)::info_,\
        #bench_case_name, #bench_name, \
        new internal::FactoryImpl<\
            BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)>);\
unsigned int BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)::body(unsigned int n) {\
	body_helper(n);\
    return n;\
}\
inline void BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)::body_helper(unsigned int param_name)

#define BENCHMARK_MULTI_(bench_case_name, bench_name, parent_class)\
class BENCHMARK_CLASS_NAME_(bench_case_name, bench_name) : public parent_class {\
 public:\
  BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)() {}\
 private:\
  virtual unsigned int body(unsigned int);\
  unsigned int body_helper();\
  static ::internal::BenchmarkInfo info_;\
};\
\
::internal::BenchmarkInfo BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)\
  ::info_ =\
    internal::addBenchmark(& BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)::info_,\
        #bench_case_name, #bench_name, \
        new internal::FactoryImpl<\
            BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)>);\
unsigned int BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)::body(unsigned int n) {\
	unsigned int niters = 0;\
	while(niters < n) \
	  niters += body_helper();\
	return niters;\
}\
NOINLINE unsigned int BENCHMARK_CLASS_NAME_(bench_case_name, bench_name)::body_helper()

#ifndef ANONYMOUS_VARIABLE
#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __COUNTER__)
#else
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#endif
#endif

#define BENCHMARK_SUSPEND                               \
  if (auto ANONYMOUS_VARIABLE(BENCHMARK_SUSPEND) =   \
      BenchmarkSuspender()) {}                 \
  else
#endif
