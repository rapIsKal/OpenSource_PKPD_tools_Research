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
#include "stats/beta.hpp"
#include "chisq_test.hpp"

TEST(Stats, testBeta) {
	static TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(-1.0, 2.0, 2.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 0.1), -2.302585092994045 },
		{ std::make_tuple(0.0, 1.0, 2.0), 0.6931471805599453 },
		{ std::make_tuple(0.5, 5.0, 1.0), -1.163150809805681 },
		{ std::make_tuple(1.0, 0.0, 0.5), math::NaN() },
		{ std::make_tuple(1.0, 0.0, -0.5), math::NaN() },
		{ std::make_tuple(1.0, 0.5, 0.0), math::NaN() },
		{ std::make_tuple(1.0, -0.5, 0.0), math::NaN() },
		{ std::make_tuple(0.0, 0.1, 0.5), math::posInf() },
		{ std::make_tuple(0.0, 2.5, 0.5), math::negInf() },
		{ std::make_tuple(0.5, 0.1, 0.5), -1.456437683805748 },
		{ std::make_tuple(2.0, 0.1, 0.5), math::negInf() },
		{ std::make_tuple(1.0, 0.1, 1.0), -2.302585092994045 },
		{ std::make_tuple(1.0, 0.1, 0.5), math::posInf() },
		{ std::make_tuple(1.0, 0.1, 1.5), math::negInf() },
	};

	testDistribution( vals, stats::Beta::logpdf<false>, stats::Beta::logpdf<true> );
}

TEST(Stats, testBetaDx) {
	TestValues<double, double, double, double> dx_vals[] = {
		{ std::make_tuple(0.0, 1.0, 1.0), 0.0 },
		{ std::make_tuple(0.0, 1.0, 0.1), 0.9 },
		{ std::make_tuple(0.0, 1.0, 2.0), -1.0 },
		{ std::make_tuple(0.5, 5.0, 1.0), 8.0 },
		{ std::make_tuple(0.0, 0.1, 0.5), math::negInf() },
		{ std::make_tuple(0.0, 1.1, 0.5), math::posInf() },
		{ std::make_tuple(1.0, 0.1, 0.5), math::negInf() },
		{ std::make_tuple(1.0, 0.1, 1.5), math::posInf() },
		{ std::make_tuple(0.5, 0.1, 0.5), -0.8 },
		{ std::make_tuple(0.9, 1.0, 0.4), 6.0 },
		{ std::make_tuple(0.8, 1.0, 1.4), -2.0 },
		{ std::make_tuple(0.9, 0.4, 1.0), -0.6666666666666667 },
		{ std::make_tuple(0.8, 1.4, 1.0), 0.5 },
		{ std::make_tuple(2.0, 0.1, 0.5), math::NaN() },
		{ std::make_tuple(1.0, 0.1, 1.0), -0.9 }
	};

	testFunction( dx_vals, stats::Beta::logpdf_dx );
}

TEST(Stats, testBetaGrad) {
	static_assert( std::is_same<stats::Beta::GradientType, math::Gradients<2>>::value, "Beta::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(0.0, 1.0, 1.0), {math::negInf(), 1} },
		{ std::make_tuple(0.0, 1.0, 0.1), {math::negInf(), 10.0} },
		{ std::make_tuple(0.0, 1.0, 2.0), {math::negInf(),  0.5} },
		{ std::make_tuple(0.5, 5.0, 1.0), {-0.49314718055994522, 1.5901861527733878} },
		{ std::make_tuple(0.0, 0.1, 0.5), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.5, 0.1, 0.5), {8.1899885459579398, -0.27025636843171208} },
		{ std::make_tuple(2.0, 0.1, 0.5), {math::NaN(), math::NaN()} },
		{ std::make_tuple(-0.1, 0.1, 0.5),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.1, 1.0), {10.0, math::negInf()} }
	};

	testFunction( vals_grad, stats::Beta::logpdf_grad );
}

TEST(Stats, testBetaCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(0.5, 2.0, 5.0), 0.890625 },
	};

	auto ftol = [](double expected) { return epsilon(expected, 1e-9, 1e-9); };
	testFunction(cdf_vals, stats::Beta::cdf, ftol);
}

TEST(Stats, testBetaRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Beta>(rng, 1.0, 1.0), 0.01);
	EXPECT_LT(chisq_test<stats::Beta>(rng, 0.1, 0.5), 0.01);
}