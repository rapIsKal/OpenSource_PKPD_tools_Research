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
#include "stats/exponential.hpp"
#include "chisq_test.hpp"

TEST(Stats, testExponential) {
	TestValues < double, double, double > vals[] = {
		{ std::make_tuple(-2.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 5.0), 1.609437912434100 },
		{ std::make_tuple(0.0, 1.0), 0.0 },
		{ std::make_tuple(3.0, 4.0), -10.61370563888011},
		{ std::make_tuple(0.0, 0.0), math::posInf() },
		{ std::make_tuple(-0.1, 0.0), math::negInf() },
		{ std::make_tuple(2.0, 0.0), math::negInf() },
		{ std::make_tuple(0.1, 0.5), -0.743147180559945 },
	};
	testDistribution(vals, stats::Exponential::logpdf<false>, stats::Exponential::logpdf<true>);
}

TEST(Stats, testExponentialDx) {
	TestValues < double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0), math::NaN() },
		{ std::make_tuple(3.0, 4.0), -4.0 },
		{ std::make_tuple(0.1, 0.5), -0.5 },
	};
	testFunction(vals_dx, stats::Exponential::logpdf_dx);
}

TEST(Stats, testExponentialGrad) {
	static_assert(std::is_same<stats::Exponential::GradientType, math::Gradients<1>>::value, "Exponential::GradientType != math::Gradients<1>");
	TestValues < math::Gradients<1>, double, double > vals_grad[] = {
		{ std::make_tuple(-2.0, 1.0), { math::NaN() } },
		{ std::make_tuple(0.0, 5.0), { math::NaN() } },
		{ std::make_tuple(1.0, 0.0), { math::NaN() } },
		{ std::make_tuple(0.0, 1.0), { math::NaN() } },
		{ std::make_tuple(3.0, 4.0), { -2.75 } },
		{ std::make_tuple(0.1, 0.5), { 1.9 } },
	};
	testFunction(vals_grad, stats::Exponential::logpdf_grad);
}

TEST(Stats, testExponentialCdf) {
	TestValues<double, double, double> cdf_vals[] = {
		{ std::make_tuple(0.0, 5.0), 0 },
		{ std::make_tuple(3.0, 4.0), 0.9999938557876467 },
		{ std::make_tuple(0.1, 0.5), 0.04877057549928599 }
	};

	testFunction( cdf_vals, stats::Exponential::cdf );
}

TEST(Stats, testExponentialIcdf) {
	TestValues<double, double, double> icdf_vals[] = {
		{ std::make_tuple(0.0, 5.0), 0 },
		{ std::make_tuple(0.05, 1.0), 0.05129329438755054 },
		{ std::make_tuple(0.9, 4.0), 0.5756462732485115 },
		{ std::make_tuple(0.1, 0.5), 0.2107210313156526 }
	};

	testFunction( icdf_vals, stats::Exponential::icdf );
}

TEST(Stats, testExponentialRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Exponential>(rng, 4.0), 0.01);
	//EXPECT_LT(chisq_test<stats::Exponential>(rng, 0.5), 0.01);
}