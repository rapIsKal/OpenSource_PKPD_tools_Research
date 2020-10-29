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

#include <gtest/gtest.h>
#include "Random.hpp"
#include "math/integration.hpp"
#include "math/constants.hpp"

TEST(Integration, testCubatureSin) {
	auto f = [](double x) { return std::sin(x); };
	double exact = 2.0;

	const double approx = math::cubature(f, 0, math::pi());
	const double error = std::abs( approx - exact );
	EXPECT_LT( error, 1e-8 );
}

TEST(Integration, testQuadratureSin) {
	auto f = [](double x) { return std::sin(x); };
	double exact = 2.0;

	for(int n = 6; n <= 20; n++) {
		const double approx = math::quadrature(f, 0, math::pi(), n);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

TEST(Integration, testQuadraturePolynomial) {
	auto f = [](double x) { return 5.0 * x*x - 2.0 * x + 3.0; };
	double exact = 45.0;

	for(int n = 2; n <= 20; n++) {
		const double approx = math::quadrature(f, 0, 3.0, n);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

/* test integrand from W. J. Morokoff and R. E. Caflisch, "Quasi=
   Monte Carlo integration," J. Comput. Phys 122, 218-230 (1995).
   Designed for integration on [0,1]^dim, integral = 1. */
static double morokoff(unsigned dim, const double *x) {
	double p = 1.0 / dim;
	double prod = pow(1 + p, dim);
	unsigned int i;
	for(i = 0; i < dim; i++)
		prod *= pow(x[i], p);
	return prod;
}

#define K_2_SQRTPI 1.12837916709551257390

/* Simple product function */
static double f0(unsigned dim, const double *x) {
	double prod = 1.0;
	unsigned int i;
	for(i = 0; i < dim; ++i)
		prod *= 2.0 * x[i];
	return prod;
}

/* Gaussian centered at 1/2. */
static double f1(unsigned dim, const double *x, double a = 0.1) {
	double sum = 0.;
	unsigned int i;
	for(i = 0; i < dim; i++) {
		double dx = x[i] - 0.5;
		sum += dx * dx;
	}
	return (pow(K_2_SQRTPI / (2. * a), (double)dim) * exp(-sum / (a * a)));
}

/* Tsuda's example */
static double f3(unsigned dim, const double *x, double c = (1.0+sqrt (10.0))/9.0) {
	double prod = 1.;
	unsigned int i;
	for(i = 0; i < dim; i++)
		prod *= c / (c + 1) * pow((c + 1) / (c + x[i]), 2.0);
	return prod;
}

TEST(Integration, testCubatureMorokoff) {
	auto f = [](ConstRefVecd &x) { return morokoff(x.size(), x.data()); };
	double exact = 1.0;

	for(int n = 2; n <= 3; n++) {
		Vecd lb = Vecd::Zero(n);
		Vecd ub = Vecd::Ones(n);
		const double approx = math::cubature(f, lb, ub);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

TEST(Integration, testQuadratureTsuda) {
	auto f = [](ConstRefVecd &x) { return f3(x.size(), x.data()); };
	double exact = 1.0;

	for(int n = 2; n <= 3; n++) {
		Vecd lb = Vecd::Zero(n);
		Vecd ub = Vecd::Ones(n);
		const double approx = math::quadrature(f, lb, ub, 10);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

TEST(Integration, testCubatureMultidim) {
	auto f = [](ConstRefVecd &x, RefVecd val) {
		val[0] = f0(x.size(), x.data());
		val[1] = f1(x.size(), x.data());
		val[2] = f3(x.size(), x.data());
	};
	double exact = 1.0;

	for(int n = 2; n <= 3; n++) {
		Vecd lb = Vecd::Zero(n);
		Vecd ub = Vecd::Ones(n);
		Vecd res(3);
		math::cubature(f, lb, ub, res);

		for(int j = 0; j < 3; ++j)
			EXPECT_LT( std::abs( res[j] - exact ), 1e-8 );
	}
}

TEST(Integration, testQuadratureMultidim) {
	auto f = [](ConstRefVecd &x, RefVecd val) {
		val[0] = f0(x.size(), x.data());
		val[1] = f1(x.size(), x.data());
		val[2] = f3(x.size(), x.data());
	};
	double exact = 1.0;

	for(int n = 2; n <= 3; n++) {
		Vecd lb = Vecd::Zero(n);
		Vecd ub = Vecd::Ones(n);
		Vecd res(3);
		math::quadrature(f, lb, ub, 32, res);

		for(int j = 0; j < 3; ++j)
			EXPECT_LT(std::abs(res[j] - exact), 1e-8);
	}
}

TEST(Integration, testCubatureNormal) {
	auto f = [](double x) { return x*x; };
	double exact = 1.0;

	const double approx = math::cubature_normal(f, false);
	const double error = std::abs(approx - exact);
	EXPECT_LT(error, 0.008);
}

TEST(Integration, testQuadratureNormal) {
	auto f = [](double x) { return x*x; };
	double exact = 1.0;

	for(int n = 3; n <= 25; n++) {
		const double approx = math::quadrature_normal(f, false, n);
		const double error = std::abs(approx - exact);
		EXPECT_LT(error, 1e-8);
	}
}

TEST(Integration, testMCNormal) {
	auto f = [](double x) { return std::cos(x); };
	double exact = std::sqrt(1.0 / math::e());

	Random rng;
	const double approx = math::mc_normal(f, false, rng, 10000);
	const double error = std::abs(approx - exact);
	EXPECT_LT(error, 0.04);
}

TEST(Integration, testCubatureNormalMulti) {
	auto f = [](ConstRefVecd &x, RefVecd val) {
		val[0] = 1;
		val[1] = sin(x[0])+cos(x[1]);
	};
	VecNd<2> exact;
	exact << 1.0, (1.0 / std::sqrt(math::e()));

	Vecd approx(2);
	math::cubature_normal(f, 2, false, approx);

	for(int j = 0; j < 2; ++j)
		EXPECT_LT(std::abs(approx[j] - exact[j]), 0.008);
}

TEST(Integration, testQuadratureNormalMulti) {
	auto f = [](ConstRefVecd &x, RefVecd val) {
		val[0] = 1;
		val[1] = sin(x[0])+cos(x[1]);
	};
	VecNd<2> exact;
	exact << 1.0, (1.0 / std::sqrt(math::e()));

	for(int n = 8; n <= 25; n++) {
		Vecd approx(2);
		math::quadrature_normal(f, 2, false, n, approx);

		for(int j = 0; j < 2; ++j)
			EXPECT_LT(std::abs(approx[j] - exact[j]), 1e-8);
	}
}

TEST(Integration, testMCNormalMulti) {
	auto f = [](ConstRefVecd &x, RefVecd val) {
		val[0] = 1;
		val[1] = sin(x[0])+cos(x[1]);
	};
	VecNd<2> exact;
	exact << 1.0, (1.0 / std::sqrt(math::e()));

	Vecd approx(2);
	Random rng;
	math::mc_normal(f, 2, false, rng, 10000, approx);

	for(int j = 0; j < 2; ++j)
		EXPECT_LT(std::abs(approx[j] - exact[j]), 0.05);
}
