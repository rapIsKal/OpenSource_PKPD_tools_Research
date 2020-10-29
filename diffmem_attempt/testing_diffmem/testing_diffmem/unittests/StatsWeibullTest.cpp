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
#include "stats/weibull.hpp"
#include "chisq_test.hpp"

TEST(Stats, testWeibull) {
	static TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(0.25, 2.9, 1.8), -3.277093769205283724233 },
		{ std::make_tuple(3.9, 1.7, 0.25), -102.89620743927044 },
		{ std::make_tuple(2.0, 1.0, 0.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 0.0), math::posInf() },
		{ std::make_tuple(2.0, 0.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 0.0, 1.0), math::posInf() },
		{ std::make_tuple(2.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.5, 1.0), math::posInf() },
		{ std::make_tuple(0.0, 2.5, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 1.0), 0.0 },
		{ std::make_tuple(0.0, 1.0, 2.0), -0.693147180559945 }
	};

	testDistribution( vals, stats::Weibull::logpdf<false>, stats::Weibull::logpdf<true> );
}

TEST(Stats, testWeibullDx) {
	static TestValues<double, double, double, double> vals_dx[] = {
		{ std::make_tuple(0.25, 2.9, 1.8), 7.5621388544962098 },
		{ std::make_tuple(3.9, 1.7, 0.25), -46.346557301393148 },
		{ std::make_tuple(2.0, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(2.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.5, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 2.5, 1.0), math::posInf() },
		{ std::make_tuple(0.0, 1.0, 1.0), -1.0 },
		{ std::make_tuple(0.0, 1.0, 2.0), -0.5 }
	};

	testFunction( vals_dx, stats::Weibull::logpdf_dx );
}

TEST(Stats, testWeibullGrad) {
	static_assert( std::is_same<stats::Weibull::GradientType, math::Gradients<2>>::value, "Weibull::GradientType != Gradients<2>");

	static TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(0.25, 2.9, 1.8), {-1.622810252835588, -1.605852618680029} },
		{ std::make_tuple(3.9, 1.7, 0.25), {-289.89780564748042, 719.0062939017331} },
		{ std::make_tuple(2.0, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(2.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.5, 1.0), {math::negInf(), -.5} },
		{ std::make_tuple(0.0, 2.5, 1.0), {math::negInf(), -2.5} },
		{ std::make_tuple(0.0, 1.0, 1.0), {math::negInf(), -1.0} },
		{ std::make_tuple(0.0, 1.0, 2.0), {math::negInf(), -0.5} }
	};

	testFunction( vals_grad, stats::Weibull::logpdf_grad );
}

TEST(Stats, testWeibullCdf) {
	TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(0.25, 2.9, 1.8), 0.003258571149067424 },
		{ std::make_tuple(3.9, 1.7, 0.25), 1 },
		{ std::make_tuple(0.5, 1.0, 1.0), 0.3934693402873666 },
		{ std::make_tuple(0.0, 1.0, 2.0), 0.0 }
	};

	testFunction(vals, stats::Weibull::cdf);
}

TEST(Stats, testWeilbullRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Weibull>(rng, 2.9, 1.8), 0.01);
	EXPECT_LT(chisq_test<stats::Weibull>(rng, 1.0, 1.0), 0.01);
}