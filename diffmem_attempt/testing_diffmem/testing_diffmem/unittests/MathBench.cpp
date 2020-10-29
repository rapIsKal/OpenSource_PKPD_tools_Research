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
#include <cmath>
#include "math/digamma.hpp"
#include "math/inv_phi.hpp"
#include "math/phi.hpp"
#include "math/inv_erf.hpp"

template <typename... Args>
struct Settings {
	Settings(Args... vals) { values = std::tuple<Args...>(vals...); }
	std::tuple<Args...> values;
};

template <typename Func, typename T>
NOINLINE void applyMany(Func && f, int N, T params[]) {
	for(int i = 0; i < N; ++i) {
		double val = apply(f, params[i]);
		doNotOptimizeAway(val);
	}
}

BENCHMARK_MULTI(Math, digamma) {
	std::tuple<double> params[] = {
		std::make_tuple(23),
		std::make_tuple(23.5),
		std::make_tuple(24),
		std::make_tuple(24.5),
		std::make_tuple(25)
	};
	const int N = sizeof(params)/sizeof(params[0]);

	applyMany(math::digamma, N, params);
	return N;
}

BENCHMARK_MULTI(Math, inv_Phi) {
	std::tuple<double> params[] = {
		std::make_tuple(0.5),
		std::make_tuple(0.123456789),
		std::make_tuple(0.02425),
		std::make_tuple(0.97575),
		std::make_tuple(0.99),
	};
	const int N = sizeof(params)/sizeof(params[0]);

	applyMany(math::inv_Phi, N, params);
	return N;
}

BENCHMARK_MULTI(Math, Phi) {
	std::tuple<double> params[] = {
		std::make_tuple(0.0),
		std::make_tuple(-1.157878609150208),
		std::make_tuple(-37.668980431720044),
		std::make_tuple(2.326347874040841),
		std::make_tuple(0.5)
	};

	const int N = sizeof(params)/sizeof(params[0]);
	applyMany(math::Phi, N, params);
	return N;
}

BENCHMARK_MULTI(Math, erf) {
	std::tuple<double> params[] = {
		std::make_tuple(0.0),
		std::make_tuple(-1.0),
		std::make_tuple(1.0),
		std::make_tuple(-0.5),
		std::make_tuple(0.5)
	};

	const int N = sizeof(params)/sizeof(params[0]);
	applyMany([](double x) { return std::erf(x); }, N, params);
	return N;
}

BENCHMARK_MULTI(Math, inv_erf) {
	std::tuple<double> params[] = {
		std::make_tuple(0.0),
		std::make_tuple(-0.8427008),
		std::make_tuple(0.8427008),
		std::make_tuple(-0.5204999),
		std::make_tuple(0.5204999)
	};

	const int N = sizeof(params)/sizeof(params[0]);
	applyMany(math::inv_erf, N, params);
	return N;
}
