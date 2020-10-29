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
#include "stats/uniform_discrete.hpp"
#include "chisq_test.hpp"

TEST(Stats, testUniformDiscrete) {
	static TestValues<double, int, int, int> vals[] = {
		{ std::make_tuple(0, 1, 0), math::NaN() },
		{ std::make_tuple(0, 1, 1), math::negInf() },
		{ std::make_tuple(1, 1, 1), 0.0 },
		{ std::make_tuple(-1, 0, 1), math::negInf() },
		{ std::make_tuple(0, 0, 1), -0.6931471805599453 },
		{ std::make_tuple(1, 0, 1), -0.6931471805599453 },
		{ std::make_tuple(1, 0, 2), -1.09861228866811 },
		{ std::make_tuple(0, 0, 2), -1.09861228866811 },
		{ std::make_tuple(2, 0, 2), -1.09861228866811 },
		{ std::make_tuple(10, 0, 100),  -4.61512051684126},
	};

	testDistribution( vals, stats::UniformDiscrete::logpdf<false>, stats::UniformDiscrete::logpdf<true> );
}

TEST(Stats, testUniformDiscreteCdf) {
	TestValues<double, int, int, int> cdf_vals[] = {
		{ std::make_tuple(-1, 0, -1), math::NaN() },
		{ std::make_tuple(-1, 0, 1), 0 },
		{ std::make_tuple(0, 0, 1), 0.5 },
		{ std::make_tuple(1, 0, 1), 1 },
		{ std::make_tuple(1, 0, 2), 0.6666666666666666 }
	};

	testFunction( cdf_vals, stats::UniformDiscrete::cdf );
}

TEST(Stats, testUniformDiscreteIcdf) {
	TestValues<double, double, int, int> icdf_vals[] = {
		{ std::make_tuple(0, 0, -1), math::NaN() },
		{ std::make_tuple(-1, 0, 1), math::NaN() },
		{ std::make_tuple(2, 0, 1), math::NaN() },
		{ std::make_tuple(0, 0, 1), 0 },
		{ std::make_tuple(1, 0, 1), 1 },
		{ std::make_tuple(0.5, 0, 4), 2 }
	};

	testFunction( icdf_vals, stats::UniformDiscrete::icdf );
}

TEST(Stats, testUniformDiscreteRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::UniformDiscrete>(rng, 0, 1), .005);
	EXPECT_LT(chisq_test<stats::UniformDiscrete>(rng, -2, 2), .005);
}