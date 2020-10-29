/*
 * Copyright (c) 2016, Hasselt University
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Benchmark.hpp>
#include "apply.hpp"
#include "stats/mvnormal.hpp"
#include "stats/normal.hpp"
#include "stats/gamma.hpp"

template <typename... Args>
struct Settings {
	Settings(Args... vals) { values = std::tuple<Args...>(vals...); }
	std::tuple<Args...> values;
};

template <typename Func, typename T>
void applyMany(Func && f, int N, T params[]) {
	for(int i = 0; i < N; ++i) {
		double val = apply(f, params[i]);
		doNotOptimizeAway(val);
	}
}

BENCHMARK_MULTI(Statistics, gamma) {
	std::tuple<double, double, double> params[] = {
		std::make_tuple(-1.0, 0.0, 1.0),
		std::make_tuple(0.0, 1.0, 1.0),
		std::make_tuple(0.5, 5.0, 1.0),
		std::make_tuple(1.0, 0.5, 0.5)
	};
	const int N = sizeof(params)/sizeof(params[0]);

	applyMany(stats::Gamma::logpdf<false>, N, params);
	return N;
}

BENCHMARK_MULTI(Statistics, gammaProp) {
	std::tuple<double, double, double> params[] = {
		std::make_tuple(-1.0, 0.0, 1.0),
		std::make_tuple(0.0, 1.0, 1.0),
		std::make_tuple(0.5, 5.0, 1.0),
		std::make_tuple(1.0, 0.5, 0.5)
	};
	const int N = sizeof(params)/sizeof(params[0]);

	applyMany(stats::Gamma::logpdf<true>, N, params);
	return N;
}

BENCHMARK_MULTI(Statistics, gammaGrad) {
	std::tuple<double, double, double> params[] = {
		std::make_tuple(0.0, 1.0, 1.0),
		std::make_tuple(0.5, 5.0, 1.0),
		std::make_tuple(1.0, 0.5, 0.5)
	};

	const int N = sizeof(params)/sizeof(params[0]);
	applyMany(stats::Gamma::logpdf_dx, N, params);
	return N;
}

BENCHMARK_MULTI(Statistics, normal) {
	std::tuple<double, double, double> params[] = {
		std::make_tuple(-1.0, 0.0, 1.0),
		std::make_tuple(0.0, 1.0, 1.0),
		std::make_tuple(0.5, 5.0, 1.0),
		std::make_tuple(1.0, 0.5, 0.5)
	};
	const int N = sizeof(params)/sizeof(params[0]);
	applyMany(stats::Normal::logpdf<false>, N, params);
	return N;
}

BENCHMARK_MULTI(Statistics, normalProp) {
	std::tuple<double, double, double> params[] = {
		std::make_tuple(-1.0, 0.0, 1.0),
		std::make_tuple(0.0, 1.0, 1.0),
		std::make_tuple(0.5, 5.0, 1.0),
		std::make_tuple(1.0, 0.5, 0.5)
	};
	const int N = sizeof(params)/sizeof(params[0]);
	applyMany(stats::Normal::logpdf<true>, N, params);
	return N;
}

BENCHMARK(Statistics, mvnormal, n) {
	BenchmarkSuspender braces;
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;
	Cholesky<Matrixd> chol_omega( omega );

	Vecd x(5);
	x << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	Vecd mu = Vecd::Zero(5);
	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway( stats::MVNormal::logpdf<false>(x, mu, chol_omega) );
}
