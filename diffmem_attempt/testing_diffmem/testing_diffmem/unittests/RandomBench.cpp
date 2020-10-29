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

#include <algorithm>
#include <thread>
#include <vector>
#include <random>
#include <Random.hpp>
#include <Benchmark.hpp>
#include <stats/uniform.hpp>
#include <stats/normal.hpp>
#include <stats/mvnormal.hpp>
#include <stats/gamma.hpp>
#include <stats/trunc_normal.hpp>

using U = stats::Uniform;
using N = stats::Normal;
using G = stats::Gamma;
using MVN = stats::MVNormal;

class fake_rng {
public:
	typedef int result_type;

	fake_rng()
			: _x(0) {}
	result_type operator()() { return ++_x; }
	result_type min() const { return 1; }
	result_type max() const { return std::numeric_limits<result_type>::max(); }

private:
	int _x;
};

BENCHMARK(Random, fake_rng, n) {
	BenchmarkSuspender braces;
	fake_rng rng;
	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(rng());
}

BENCHMARK(Random, mt19937, n) {
	BenchmarkSuspender braces;
	std::mt19937 rng;
	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(rng());
}

BENCHMARK(Random, minstdrand_uniform, n) {
	BenchmarkSuspender braces;
	std::random_device rd;
	std::minstd_rand rng(rd());
	std::uniform_real_distribution<double> dist(0, 1);
	auto u = std::bind(dist, rng);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(u());
}

BENCHMARK(Random, minstdrand_normal, n) {
	BenchmarkSuspender braces;
	std::random_device rd;
	std::minstd_rand rng(rd());
	std::normal_distribution<double> dist;
	auto u = std::bind(dist, rng);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(u());
}

BENCHMARK(Random, mt19937_uniform, n) {
	BenchmarkSuspender braces;
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<double> dist(0, 1);
	auto u = std::bind(dist, rng);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(u());
}

BENCHMARK(Random, mt19937_normal, n) {
	BenchmarkSuspender braces;
	std::random_device rd;
	std::mt19937 rng(rd());
	std::normal_distribution<double> dist;
	auto u = std::bind(dist, rng);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(u());
}

BENCHMARK(Random, uniform, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	auto runif = U::generator(rng);
	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(runif());
}

BENCHMARK(Random, uniform2, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(U::sample(rng));
}

BENCHMARK(Random, normal, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	auto rnorm = N::generator(rng);
	for(unsigned int i = 0; i < n; ++i) {
		doNotOptimizeAway(rnorm());
	}
}

BENCHMARK(Random, normal2, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i) {
		doNotOptimizeAway(N::sample(rng));
	}
}

BENCHMARK(Random, mt19937_gamma, n) {
	BenchmarkSuspender braces;
	std::random_device rd;
	std::mt19937 rng(rd());
	std::gamma_distribution<double> dist(2.0, 2.0);
	auto u = std::bind(dist, rng);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(u());
}

template <typename T>
void doNotOptimizeAwayEigen(Eigen::MatrixBase<T> &v) {
	for(int i = 0; i < v.size(); ++i)
		doNotOptimizeAway(v[i]);
}

BENCHMARK(Random, gamma, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		doNotOptimizeAway(G::sample(rng, 2.0, 2.0));
}

BENCHMARK(Random, normalNullary, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	Vecd v(500);
	for(unsigned int i = 0; i < n; ++i) {
		v = generate_random<Vecd>(N::generator(rng), 500);
		doNotOptimizeAwayEigen(v);
	}
}

BENCHMARK(Random, mvnormal, n) {
	BenchmarkSuspender braces;
	Random rng;

	Vecd mu = Vecd::LinSpaced(5, 0.0, 1.0);
	braces.dismiss();

	Vecd v(5);
	for(unsigned int i = 0; i < n; ++i) {
		v = MVN::sample(rng, mu);
		doNotOptimizeAwayEigen(v);
	}
}

BENCHMARK(Random, mvnormal_omega, n) {
	BenchmarkSuspender braces;
	Random rng;

	Vecd mu = Vecd::LinSpaced(5, 0.0, 1.0);
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;
	Cholesky<Matrixd> chol_omega(omega);
	braces.dismiss();

	Vecd v(5);
	for(unsigned int i = 0; i < n; ++i) {
		v = MVN::sample(rng, Vecd::Zero(5), chol_omega);
		doNotOptimizeAwayEigen(v);
	}
}

BENCHMARK(Random, mvnormal_omega4, n) {
	BenchmarkSuspender braces;
	Random rng;

	Vecd mu = Vecd::LinSpaced(5, 0.0, 1.0);
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;
	Cholesky<Matrixd> chol_omega(omega);
	braces.dismiss();

	Vecd v(5);
	for(unsigned int i = 0; i < n; ++i) {
		MVN::sample(rng, Vecd::Zero(5), chol_omega, v);
		doNotOptimizeAwayEigen(v);
	}
}

BENCHMARK(Random, mvnormal_omega3, n) {
	BenchmarkSuspender braces;
	Random rng;

	Vecd mu = Vecd::LinSpaced(5, 0.0, 1.0);
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;
	braces.dismiss();


	auto rmvnorm = MVN::generator<Vecd>(rng, mu, omega);

	Vecd v(5);
	for(unsigned int i = 0; i < n; ++i) {
		v = rmvnorm();
		doNotOptimizeAwayEigen(v);
	}
}

BENCHMARK(Random, truncnormal, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	auto rnorm = stats::Truncnormal::generator(rng, 0, 1, 4, math::posInf());
	for(unsigned int i = 0; i < n; ++i) {
		doNotOptimizeAway(rnorm());
	}
}

BENCHMARK(Random, truncnormal2, n) {
	BenchmarkSuspender braces;
	Random rng;

	braces.dismiss();

	auto rnorm = stats::Truncnormal::generator(rng, 0, 1, -1, 1);
	for(unsigned int i = 0; i < n; ++i) {
		doNotOptimizeAway(rnorm());
	}
}