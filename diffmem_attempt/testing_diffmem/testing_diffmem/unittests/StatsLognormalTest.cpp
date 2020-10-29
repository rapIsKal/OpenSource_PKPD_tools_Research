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
#include "stats/lognormal.hpp"
#include "chisq_test.hpp"

TEST(Stats, testLognormal) {
	TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(0.5, 5.0, 1.0), -16.43175376240356 },
		{ std::make_tuple(1.0, 0.5, 0.5), -0.7257913526447274 },
		{ std::make_tuple(2.0, 2.0, 4.0), -3.051750833999459 },
		{ std::make_tuple(1.0, 0.0, 2.0), -1.612085713764618 },
		{ std::make_tuple(2.0, 2.0, 0.0), math::negInf() },
		{ std::make_tuple(1.0, 0.0, 0.0), math::posInf() },
		{ std::make_tuple(2.0, math::log2(), 0.0), math::posInf() },
		{ std::make_tuple(1.0, 2.0, 0.0), math::negInf() },
		{ std::make_tuple(1.0, 1.0, 0.0), math::negInf() },
	};
	testDistribution( vals, stats::Lognormal::logpdf<false>, stats::Lognormal::logpdf<true> );
}

TEST(Stats, testLognormalDx) {
	TestValues<double, double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0), math::posInf() },
		{ std::make_tuple(0.5, 5.0, 1.0), 9.3862943611198908 },
		{ std::make_tuple(1.0, 0.5, 0.5), 1.0 },
		{ std::make_tuple(2.0, 2.0, 4.0), -0.4591608493924983 },
		{ std::make_tuple(1.0, 0.0, 2.0), -1.0 },
	};
	testFunction( vals_dx, stats::Lognormal::logpdf_dx );
}

TEST(Stats, testLognormalGrad) {
	static_assert( std::is_same<stats::Lognormal::GradientType, math::Gradients<2>>::value, "Lognormal::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), {math::NaN()} },
		{ std::make_tuple(1.0, 0.0, -1.0), {math::NaN()} },
		{ std::make_tuple(-1.0, 0.0, 1.0), {math::NaN()} },
		{ std::make_tuple(0.0, 1.0, 1.0), {math::NaN()} },
		{ std::make_tuple(0.5, 5.0, 1.0), {-5.6931471805599454, 31.411924819517658} },
		{ std::make_tuple(1.0, 0.5, 0.5), {-2.0, 0.0} },
		{ std::make_tuple(2.0, 2.0, 4.0),{ -0.0816783012150034, -0.2233146204425247} },
		{ std::make_tuple(1.0, 0.0, 2.0),{ 0.0, -0.5 } },
	};
	testFunction( vals_grad, stats::Lognormal::logpdf_grad );
}

TEST(Stats, testLognormalCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0), 0 },
		{ std::make_tuple(0.5, 5.0, 1.0), 6.235939346711529e-09 },
		{ std::make_tuple(1.0, 0.5, 0.5), 0.158655253931457 }
	};

	testFunction( cdf_vals, stats::Lognormal::cdf );
}

TEST(Stats, testLognormalRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Lognormal>(rng, 0.0, 1.0), 0.01);
	EXPECT_LT(chisq_test<stats::Lognormal>(rng, 5.0, 0.5), 0.01);
}