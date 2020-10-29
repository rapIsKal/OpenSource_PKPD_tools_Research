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
#include "stats/betaprime.hpp"

TEST(Stats, testBetaprime) {
	static TestValues<double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(-1.0, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 3.0), 1.09861228866811 },
		{ std::make_tuple(0.0, 2.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 0.5, 1.0), math::posInf() },
		{ std::make_tuple(1.0, 1.0, 1.0), -1.386294361119891 },
		{ std::make_tuple(2.0, 2.0, 2.0), -1.909542504884438 },
		{ std::make_tuple(0.5, 0.5, 1.5), -0.915939331225811 },
		{ std::make_tuple(0.5, 1.5, 0.5), -1.60908651178576 },
	};

	testDistribution( vals, stats::Betaprime::logpdf<false>, stats::Betaprime::logpdf<true> );
}

TEST(Stats, testBetaprimeDx) {
	TestValues<double, double, double, double> dx_vals[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), -1.0 },
		{ std::make_tuple(2.0, 2.0, 2.0), -0.8333333333333333 },
		{ std::make_tuple(5.0, 2.0, 20.0), -3.466666666666667 },
		{ std::make_tuple(0.5, 0.5, 1.5), -2.333333333333333 },
		{ std::make_tuple(0.5, 1.5, 0.5), -0.333333333333333 },
	};

	testFunction( dx_vals, stats::Betaprime::logpdf_dx );
}

TEST(Stats, testBetaprimeGrad) {
	static_assert( std::is_same<stats::Betaprime::GradientType, math::Gradients<2>>::value, "Betaprime::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(0.0, 2.0, 5.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, 1.0),{ 0.3068528194400547, 0.3068528194400547 } },
		{ std::make_tuple(2.0, 2.0, 2.0),{ 0.427868225225169, -0.2652789553347764 } },
		{ std::make_tuple(5.0, 2.0, 20.0),{ 2.463037147968775, -1.694140421609007 } },
		{ std::make_tuple(0.5, 0.5, 1.5),{ 1.28768207245178, -0.019170746988274 } },
		{ std::make_tuple(0.5, 1.5, 0.5),{ -0.712317927548219, 1.98082925301173 } },
	};

	testFunction( vals_grad, stats::Betaprime::logpdf_grad );
}
