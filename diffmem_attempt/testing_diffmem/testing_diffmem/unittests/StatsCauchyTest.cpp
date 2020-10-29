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
#include "stats/cauchy.hpp"
#include "chisq_test.hpp"

TEST(Stats, testCauchy) {
	static TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), -1.837877066409345339082 },
		{ std::make_tuple(-1.5, 0.0, 1.0), -2.323384882191046330036 },
		{ std::make_tuple(-1.5, -1.0, 1.0), -1.367873437163609873224 },
		{ std::make_tuple(-1.5, -1.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 0.0), math::NaN() },
		{ std::make_tuple(math::negInf(), 1.0, 1.0), math::negInf() },
		{ std::make_tuple(-1.5, 0.0, 0.5), -2.7541677982835004932 },
		{ std::make_tuple(-10.5, 1.0, 2.0), -5.3660741388328082654 },
	};

	testDistribution( vals, stats::Cauchy::logpdf<false>, stats::Cauchy::logpdf<true> );
}

TEST(Stats, testCauchyDx) {
	static TestValues<double, double, double, double> dx_vals[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), 1 },
		{ std::make_tuple(-1.5, 0.0, 1.0), 0.92307692307692313 },
		{ std::make_tuple(-1.5, -1.0, 1.0), 0.8 },
		{ std::make_tuple(-1.5, 0.0, 0.5), 1.2 },
		{ std::make_tuple(-10.5, 1.0, 2.0), 0.16880733944954129 },
		{ std::make_tuple(1.0, 0.0, 0.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, -1.0), math::NaN() },
	};

	testFunction( dx_vals, stats::Cauchy::logpdf_dx );
}

TEST(Stats, testCauchyGrad) {
	static_assert( std::is_same<stats::Cauchy::GradientType, math::Gradients<2>>::value, "Cauchy::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), {-1.0, 0.0} },
		{ std::make_tuple(-1.5, 0.0, 1.0), {-0.92307692307692313, 0.38461538461538464} },
		{ std::make_tuple(-1.5, -1.0, 1.0), {-0.8, -0.6} },
		{ std::make_tuple(-1.5, 0.0, 0.5), {-1.2, 1.6} },
		{ std::make_tuple(-10.5, 1.0, 2.0), {-0.16880733944954129, 0.47064220183486238} },
		{ std::make_tuple(1.0, 0.0, 0.0), {math::NaN(), math::NaN()} },
		{ std::make_tuple(1.0, 0.0, -1.0), {math::NaN(), math::NaN() } },
	};

	testFunction( vals_grad, stats::Cauchy::logpdf_grad );
}

TEST(Stats, testCauchyCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(1.0, 0.0, 1.0), 0.75 },
		{ std::make_tuple(-1.5, 0.0, 1.0), 0.1871670418109988 },
		{ std::make_tuple(-2.5, 0.0, 1.0), 0.1211189415908434 },
		{ std::make_tuple(1000.0, 0.0, 1.0), 0.9996816902199195 },
		{ std::make_tuple(-1000.0, 0.0, 1.0), 0.000318309780080559 },
	};

	testFunction(cdf_vals, stats::Cauchy::cdf);
}

TEST(Stats, testCauchyRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Cauchy>(rng, 0.0, 1.0), 0.05);
	EXPECT_LT(chisq_test<stats::Cauchy>(rng, -2.0, 0.5), 0.05);
}