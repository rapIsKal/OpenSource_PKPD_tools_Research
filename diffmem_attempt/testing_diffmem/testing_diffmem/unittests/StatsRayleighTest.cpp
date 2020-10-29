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
#include "TestCommon.hpp"
#include "stats/rayleigh.hpp"
#include "chisq_test.hpp"

TEST(Stats, testRayleigh) {
	TestValues<double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 1.0), math::negInf() },
		{ std::make_tuple(1.0, -1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.0), math::NaN() },
		{ std::make_tuple(0.1, 0.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0), math::negInf() },
		{ std::make_tuple(1.0, 0.5), -0.6137056388801094 },
		{ std::make_tuple(1.0, 1.5), -1.033152438438551 },
		{ std::make_tuple(3.0, 5.0), -2.300263536200091 },
		{ std::make_tuple(4.0, 5.0), -2.15258146374831 },
		{ std::make_tuple(1.0, 0.1), -45.3948298140119 },
	};
	testDistribution( vals, stats::Rayleigh::logpdf<false>, stats::Rayleigh::logpdf<true> );
}

TEST(Stats, testRayleighDx) {
	TestValues<double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.5), -3.0 },
		{ std::make_tuple(1.0, 1.5), 0.5555555555555556 },
		{ std::make_tuple(3.0, 5.0), 0.2133333333333333 },
		{ std::make_tuple(4.0, 5.0), 0.09 },
		{ std::make_tuple(1.0, 0.1), -99.0 },
	};
	testFunction( vals_dx, stats::Rayleigh::logpdf_dx );
}

TEST(Stats, testRayleighGrad) {
	static_assert( std::is_same<stats::Rayleigh::GradientType, math::Gradients<1>>::value, "Rayleigh::GradientType != Gradients<1>");

	TestValues<math::Gradients<1>, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 1.0),{ math::NaN() } },
		{ std::make_tuple(1.0, 0.5),{ 4.0 } },
		{ std::make_tuple(1.0, 1.5),{ -1.037037037037037 } },
		{ std::make_tuple(3.0, 5.0),{ -0.328 } },
		{ std::make_tuple(4.0, 5.0),{ -0.272 } },
		{ std::make_tuple(1.0, 0.1),{ 980.0 } },
	};
	testFunction( vals_grad, stats::Rayleigh::logpdf_grad );
}

TEST(Stats, testRayleighCdf) {
	TestValues<double, double, double> cdf_vals[] = {
		{ std::make_tuple(-1.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.5), 0.8646647167633873 },
		{ std::make_tuple(1.0, 1.5), 0.1992625970831919 },
		{ std::make_tuple(3.0, 5.0), 0.164729788588728 },
		{ std::make_tuple(4.0, 5.0), 0.2738509629263091 },
		{ std::make_tuple(1.0, 0.1), 1 },
	};

	testFunction( cdf_vals, stats::Rayleigh::cdf );
}

TEST(Stats, testRayleighRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Rayleigh>(rng, 1.0), 0.01);
	EXPECT_LT(chisq_test<stats::Rayleigh>(rng, 5.0), 0.01);
}
