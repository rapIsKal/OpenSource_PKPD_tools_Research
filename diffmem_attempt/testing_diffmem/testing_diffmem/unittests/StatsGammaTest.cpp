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
#include "stats/gamma.hpp"
#include "functional.hpp"
#include "chisq_test.hpp"

TEST(Stats, testGamma) {
	static TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), math::negInf() },
		{ std::make_tuple(1.0, -0.5, 1.0), math::NaN() },
		{ std::make_tuple(4.0, 0.0, 0.0), math::negInf() },
		{ std::make_tuple(4.0, 0.0, 11.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 1.23), 0.207014169384326},
		{ std::make_tuple(0.0, 0.5, 1.23), math::posInf() },
		{ std::make_tuple(0.0, 2.0, 1.23), math::negInf() },
		{ std::make_tuple(0.0, 0.0, 1.23), math::posInf() },
		{ std::make_tuple(0.0, 2.0, 0.0), math::posInf() },
		{ std::make_tuple(0.0, 1.0, 1.0), 0.000000 },
		{ std::make_tuple(0.5, 5.0, 1.0), -6.450642552587727 },
		{ std::make_tuple(1.0, 0.5, 0.5), -1.418938533204673 }
	};

	testDistribution( vals, stats::Gamma::logpdf<false>, stats::Gamma::logpdf<true> );
}

TEST(Stats, testGammaDx) {
	TestValues<double, double, double, double> dx_vals[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, -0.5, 1.0), math::NaN() },
		{ std::make_tuple(4.0, 0.0, 11.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.23), -1.23 },
		{ std::make_tuple(0.0, 0.5, 1.23), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 1.23), math::NaN() },
		{ std::make_tuple(0.0, 0.0, 1.23), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 0.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0), -1.0 },
		{ std::make_tuple(0.0, 1.0, 2.0), -2.0 },
		{ std::make_tuple(0.5, 5.0, 1.0), 7.0 },
		{ std::make_tuple(1.0, 0.5, 0.5), -1.0 }
	};

	testFunction( dx_vals, stats::Gamma::logpdf_dx );
}

TEST(Stats, testGammaGrad) {
	static_assert( std::is_same<stats::Gamma::GradientType, math::Gradients<2>>::value, "Gamma::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), {math::NaN(), math::NaN()} },
		{ std::make_tuple(1.0, -0.5, 1.0), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.0, 1.0, 1.0), {math::NaN(), 1.0} },
		{ std::make_tuple(4.0, 0.0, 0.0), {math::NaN(), math::NaN()} },
		{ std::make_tuple(4.0, 0.0, 11.0), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.0, 1.0, 1.23), {math::NaN(), 0.8130081300813008} },
		{ std::make_tuple(0.0, 0.5, 1.23), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.0, 2.0, 1.23), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.0, 0.0, 1.23), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.0, 2.0, 0.0), {math::NaN(), math::NaN()} },
		{ std::make_tuple(0.5, 5.0, 1.0), {-2.1992648489917457, 4.5} },
		{ std::make_tuple(1.0, 0.5, 0.5), {1.2703628454614782, 0.0} }
	};

	testFunction( vals_grad, stats::Gamma::logpdf_grad );
}

TEST(Stats, testGammaCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(0.0, 0.0, 1.0), 0.0 },
		{ std::make_tuple(1.0, 1.0, 2.0), 0.8646647167633873 },
		{ std::make_tuple(1.0, 2.0, 2.0), 0.5939941502901618930466 },
		{ std::make_tuple(2.0, 0.25, 0.75), 0.9665835558410209582547 },
		{ std::make_tuple(1.0, 1.0, 1.0), 0.6321205588285576659757 },
//		{ std::make_tuple(195.0, 100, 0.5), 0.4134406976650642 }
	};

	testFunction(cdf_vals, stats::Gamma::cdf, constant(1e-9));
}

TEST(Stats, testGammaRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Gamma>(rng, 1.0, 1.0), 0.05);
	EXPECT_LT(chisq_test<stats::Gamma>(rng, 1.0, 2.0), 0.05);
}
