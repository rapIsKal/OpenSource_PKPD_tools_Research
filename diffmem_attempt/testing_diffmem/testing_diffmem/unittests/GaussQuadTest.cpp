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
#include "GaussianQuadrature.hpp"
#include "math/constants.hpp"

TEST(GaussQuadTest, testSin) {
	auto f = [](double x) { return std::sin(x); };
	double exact = 2.0;

	for(int n = 6; n <= 20; n++) {
		const double approx = GaussianQuadrature::int1d(n, f, 0, math::pi());
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

TEST(GaussQuadTest, testPolynomial) {
	auto f = [](double x) { return 5.0 * x*x - 2.0 * x + 3.0; };
	double exact = 45.0;

	for(int n = 2; n <= 20; n++) {
		const double approx = GaussianQuadrature::int1d(n, f, 0, 3.0);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

TEST(GaussQuadTest, testHermite) {
	auto f = [](double x) { return x*x; };
	double exact = math::sqrtPi() / 2.0;

	for(int n = 3; n <= 25; n++) {
		const double approx = GaussianQuadrature::int1d_hermite(n, f);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}

TEST(GaussQuadTest, test2D) {
	auto f = [](const VecNd<2> & x) { return std::sin(x[0]) + std::cos(x[1]); };
	double exact = math::pi()/std::pow(math::e(), 0.25);

	for(int n = 7; n <= 25; n++) {
		const double approx = GaussianQuadrature::intnd_hermite<2>(n, f);
		const double error = std::abs( approx - exact );
		EXPECT_LT( error, 1e-8 );
	}
}
