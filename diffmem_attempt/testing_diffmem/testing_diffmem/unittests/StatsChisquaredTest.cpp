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
#include "stats/chisquared.hpp"
#include "functional.hpp"
#include "chisq_test.hpp"

TEST(Stats, testChisquared) {
	TestValues<double, double, double> vals[] = {
		{ std::make_tuple(2.0, 1), -2.2655121234846453682 },
		{ std::make_tuple(0.25, 2), -0.81814718055994528623 },
		{ std::make_tuple(3.9, 3), -2.1884502566368722043 },
		{ std::make_tuple(2.0, 0), math::NaN() },
		{ std::make_tuple(2.0, -1), math::NaN() },
		{ std::make_tuple(-2.0, 1), math::negInf() },
		{ std::make_tuple(0.0, 1), math::posInf() },
		{ std::make_tuple(0.0, 2), -0.6931471805599453 },
		{ std::make_tuple(0.0, 4), math::negInf() }
	};
	testDistribution( vals, stats::Chisquared::logpdf<false>, stats::Chisquared::logpdf<true> );
}

TEST(Stats, testChisquaredDx) {
	TestValues<double, double, double> vals_dx[] = {
		{ std::make_tuple(2.0, 1), -0.75 },
		{ std::make_tuple(0.25, 2), -0.5 },
		{ std::make_tuple(3.9, 3), -0.37179487179487180626 },
		{ std::make_tuple(2.0, -1), math::NaN() },
		{ std::make_tuple(-2.0, 1), math::NaN() },
		{ std::make_tuple(0.0, 1), math::negInf() },
		{ std::make_tuple(0.0, 2), -0.5 },
		{ std::make_tuple(0.0, 4), math::posInf() }
	};
	testFunction( vals_dx, stats::Chisquared::logpdf_dx );
}

TEST(Stats, testChisquaredGrad) {
	static_assert( std::is_same<stats::Chisquared::GradientType, math::Gradients<1>>::value, "Chisquared::GradientType != Gradients<1>");

	TestValues<math::Gradients<1>, double, double> vals_grad[] = {
		{ std::make_tuple(2.0, 1), 0.98175501301071177 },
		{ std::make_tuple(0.25, 2), -0.7511129383891515 },
		{ std::make_tuple(3.9, 3), 0.31566969929853939 },
		{ std::make_tuple(2.0, -1), math::NaN() },
		{ std::make_tuple(-2.0, 1), math::NaN() },
		{ std::make_tuple(0.0, 1), math::negInf() },
		{ std::make_tuple(0.0, 2), math::negInf() },
		{ std::make_tuple(0.0, 4), math::negInf() }
	};
	testFunction( vals_grad, stats::Chisquared::logpdf_grad );
}

TEST(Stats, testChiSquaredCdf) {
	TestValues<double, double, double> cdf_vals[] = {
		{ std::make_tuple(7.9, 3.0), 0.951875748155839862541 },
		{ std::make_tuple(1.9, 0.5), 0.9267752080547182469417 },
//		{ std::make_tuple(195.0, 200), 0.4134406976650642 }
	};

	testFunction(cdf_vals, stats::Chisquared::cdf, constant(1e-9));
}

TEST(Stats, testChiSquaredRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Chisquared>(rng, 2.0), 0.05);
}
