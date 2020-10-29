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
#include "stats/normal.hpp"
#include "chisq_test.hpp"

TEST(Stats, testNormal) {
	TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), -1.418938533204673 },
		{ std::make_tuple(0.0, 1.0, 1.0), -1.418938533204673 },
		{ std::make_tuple(0.5, 5.0, 1.0), -11.04393853320467 },
		{ std::make_tuple(1.0, 0.5, 0.5), -0.7257913526447274 },
		{ std::make_tuple(0.1, 2.3, 3.4), -2.35205652538042 },
		{ std::make_tuple(17.000001,17.0, 0.0), math::negInf() },
		{ std::make_tuple(17.0,17.0, 0.0), math::posInf() },
	};
	testDistribution( vals, stats::Normal::logpdf<false>, stats::Normal::logpdf<true> );
}

TEST(Stats, testNormalDx) {
	TestValues<double, double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), 1.0 },
		{ std::make_tuple(0.0, 1.0, 1.0), 1.0 },
		{ std::make_tuple(0.5, 5.0, 1.0), 4.5 },
		{ std::make_tuple(1.0, 0.5, 0.5), -2.0 }
	};
	testFunction( vals_dx, stats::Normal::logpdf_dx );
}

TEST(Stats, testNormalGrad) {
	static_assert( std::is_same<stats::Normal::GradientType, math::Gradients<2>>::value, "Normal::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), {math::NaN()} },
		{ std::make_tuple(-1.0, 0.0, 1.0), {-1.0, 0.00} },
		{ std::make_tuple(0.0, 1.0, 1.0), {-1.0, 0.00} },
		{ std::make_tuple(0.5, 5.0, 1.0), {-4.5, 19.25} },
		{ std::make_tuple(1.0, 0.5, 0.5), {2.0, 0.0} }
	};
	testFunction( vals_grad, stats::Normal::logpdf_grad );
}

TEST(Stats, testNormalCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), 0.158655253931457 },
		{ std::make_tuple(0.0, 1.0, 1.0), 0.158655253931457 },
		{ std::make_tuple(0.5, 5.0, 1.0), 3.39767312473006e-06 },
		{ std::make_tuple(1.0, 0.5, 0.5), 0.8413447460685429 }
	};

	testFunction( cdf_vals, stats::Normal::cdf );
}

TEST(Stats, testNormalCdfGrad) {
	TestValues<math::Gradients<2>, double, double, double> cdf_vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), {math::NaN()} },
		{ std::make_tuple(-1.0, 0.0, 1.0), {-0.24197072451914337, 0.24197072451914337} },
		{ std::make_tuple(0.0, 1.0, 1.0), {-0.24197072451914337, 0.24197072451914337} },
		{ std::make_tuple(0.5, 5.0, 1.0), {-1.598374110703489e-05, 7.192683498059893e-05} },
		{ std::make_tuple(1.0, 0.5, 0.5), {-0.48394144903828673, -0.48394144903828673} }
	};

	testFunction( cdf_vals, stats::Normal::cdf_grad );
}

TEST(Stats, testNormalIcdf) {
	TestValues<double, double, double, double> icdf_vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(0.158655253931457, 0.0, 1.0), -1.0 },
		{ std::make_tuple(0.158655253931457, 1.0, 1.0), 0.0 },
		{ std::make_tuple(3.39767312473006e-06, 5.0, 1.0), 0.5 },
		{ std::make_tuple(0.8413447460685429, 0.5, 0.5), 1.0 },
		{ std::make_tuple(0.8413447460685429, 5.0, 0.0), 5.0 }
	};

	testFunction( icdf_vals, stats::Normal::icdf );
}

TEST(Stats, testNormalRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Normal>(rng, 0.0, 1.0), 0.01);
	EXPECT_LT(chisq_test<stats::Normal>(rng, 5.0, 0.5), 0.01);
}