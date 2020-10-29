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
#include "stats/uniform.hpp"
#include "chisq_test.hpp"

TEST(Stats, testUniform) {
	static TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(0.5, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(1.0, 1.0, 1.0), math::posInf() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 0.0, 1.0), 0.0 },
		{ std::make_tuple(1.0, 0.0, 1.0), 0.0 },
		{ std::make_tuple(0.5, 0.0, 1.0), 0.0 },
		{ std::make_tuple(1.0, 0.0, 2.0), -0.6931471805599453 },
		{ std::make_tuple(0.0, 0.0, 2.0), -0.6931471805599453 },
		{ std::make_tuple(2.0, 0.0, 2.0), -0.6931471805599453 },
		{ std::make_tuple(-1.0, math::negInf(), 0.0), 0.0 },
		{ std::make_tuple(1.0, math::negInf(), 0.0), math::negInf() },
		{ std::make_tuple(1.0, 0.0, math::posInf()), 0.0 },
		{ std::make_tuple(0.0, math::negInf(), math::posInf()), 0.0 },
		{ std::make_tuple(math::negInf(), math::negInf(), math::posInf()), 0.0 }
	};

	testDistribution( vals, stats::Uniform::logpdf<false>, stats::Uniform::logpdf<true> );
}

TEST(Stats, testUniformDx) {
	TestValues<double, double, double, double> dx_vals[] = {
		{ std::make_tuple(0.5, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.5, 0.0, 1.0), 0.0 },
		{ std::make_tuple(1.0, 0.0, 2.0), 0.0 },
		{ std::make_tuple(0.0, 0.0, 2.0), math::NaN() },
		{ std::make_tuple(2.0, 0.0, 2.0), math::NaN() },
		{ std::make_tuple(-1.0, math::negInf(), 0.0), 0.0 },
		{ std::make_tuple(1.0, math::negInf(), 0.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, math::posInf()), 0.0 },
		{ std::make_tuple(0.0, math::negInf(), math::posInf()), 0.0 },
		{ std::make_tuple(math::negInf(), math::negInf(), math::posInf()), math::NaN() }
	};

	testFunction( dx_vals, stats::Uniform::logpdf_dx );
}

TEST(Stats, testUniformGrad) {
	static_assert( std::is_same<stats::Uniform::GradientType, math::Gradients<2>>::value, "Uniform::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(0.5, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.5, 0.0, 1.0), { 1.0, -1.0 } },
		{ std::make_tuple(1.0, 0.0, 2.0), { 0.5, -0.5 } },
		{ std::make_tuple(0.0, 0.0, 2.0), math::NaN() },
		{ std::make_tuple(2.0, 0.0, 2.0), math::NaN() },
		{ std::make_tuple(-1.0, math::negInf(), 0.0), { 0.0, 0.0 } },
		{ std::make_tuple(1.0, math::negInf(), 0.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, math::posInf()), { 0.0, 0.0 } },
		{ std::make_tuple(0.0, math::negInf(), math::posInf()), { 0.0, 0.0 } },
		{ std::make_tuple(math::negInf(), math::negInf(), math::posInf()), math::NaN() }
	};

	testFunction( vals_grad, stats::Uniform::logpdf_grad );
}

TEST(Stats, testUniformCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), 0.0 },
		{ std::make_tuple(0.0, 0.0, 1.0), 0.0 },
		{ std::make_tuple(0.5, 0.0, 1.0), 0.5 },
		{ std::make_tuple(1.0, 0.0, 1.0), 1.0 },
		{ std::make_tuple(1.0, 0.0, 2.0), 0.5 }
	};

	testFunction( cdf_vals, stats::Uniform::cdf );
}

TEST(Stats, testUniformIcdf) {
	TestValues<double, double, double, double> icdf_vals[] = {
		{ std::make_tuple(0.5, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(2.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 0.0, 1.0), 0.0 },
		{ std::make_tuple(0.5, 0.0, 1.0), 0.5 },
		{ std::make_tuple(1.0, 0.0, 1.0), 1.0 },
		{ std::make_tuple(0.5, 0.0, 2.0), 1.0 }
	};

	testFunction( icdf_vals, stats::Uniform::icdf );
}

TEST(Stats, testUniformRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Uniform>(rng, 0.0, 1.0), 0.05);
	EXPECT_LT(chisq_test<stats::Uniform>(rng, -1.0, 2.0), 0.05);
}