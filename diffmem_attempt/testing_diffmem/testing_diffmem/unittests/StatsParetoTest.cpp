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
#include "stats/pareto.hpp"
#include "chisq_test.hpp"

TEST(Stats, testPareto) {
	TestValues < double, double, double, double > vals[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::negInf() },
		{ std::make_tuple(1.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), 0 },
		{ std::make_tuple(2.0, 2.0, 2.0), 0 },
		{ std::make_tuple(5.0, 2.0, 20.0), -16.9395202763632 },
		{ std::make_tuple(2.0, 2.0, math::posInf()), math::posInf() },
		{ std::make_tuple(5.0, 2.0, math::posInf()), math::negInf() },
	};
	testDistribution(vals, stats::Pareto::logpdf < false >, stats::Pareto::logpdf < true > );
}

TEST(Stats, testParetoDx) {
	TestValues < double, double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), -2.0 },
		{ std::make_tuple(2.0, 2.0, 2.0), -1.5 },
		{ std::make_tuple(5.0, 2.0, 20.0), -4.2 },
	};
	testFunction(vals_dx, stats::Pareto::logpdf_dx);
}

TEST(Stats, testParetoGrad) {
	static_assert( std::is_same<stats::Pareto::GradientType, math::Gradients<2>>::value, "Pareto::GradientType != math::Gradients<2>");
	TestValues < math::Gradients<2>, double, double, double > vals_grad[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), { math::NaN(), math::NaN() } },
		{ std::make_tuple(0.0, 2.0, 5.0), { math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, 1.0), { 1.0, 1.0 } },
		{ std::make_tuple(2.0, 2.0, 2.0), { 1.0, 0.5 } },
		{ std::make_tuple(5.0, 2.0, 20.0), { 10.0, -0.866290731874155 } },
	};
	testFunction(vals_grad, stats::Pareto::logpdf_grad);
}

TEST(Stats, testParetoCdf) {
	TestValues<double, double, double, double> cdf_vals[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), 0.0 },
		{ std::make_tuple(1.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), 0.0 },
		{ std::make_tuple(5.0, 2.0, 1.0), 0.6 },
		{ std::make_tuple(5.0, 2.0, 20.0), 0.9999999890048837 }
	};

	testFunction( cdf_vals, stats::Pareto::cdf );
}

TEST(Stats, testParetoRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Pareto>(rng, 2.0, 1.0), 0.01);
	EXPECT_LT(chisq_test<stats::Pareto>(rng, 2.0, 20.0), 0.01);
}