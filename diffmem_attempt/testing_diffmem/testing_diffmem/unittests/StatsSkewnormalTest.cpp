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
#include "stats/skewnormal.hpp"

TEST(Stats, testSkewnormal) {
	TestValues<double, double, double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0, 0.0), -1.418938533204673 },
		{ std::make_tuple(0.0, 1.0, 1.0, 0.0), -1.418938533204673 },
		{ std::make_tuple(0.5, 5.0, 1.0, 0.0), -11.04393853320467 },
		{ std::make_tuple(1.0, 0.5, 0.5, 0.0), -0.7257913526447274 },
		{ std::make_tuple(0.1, 2.3, 3.4, 0.0), -2.35205652538042 },
		{ std::make_tuple(17.000001, 17.0, 0.0, 2.0), math::negInf() },
		{ std::make_tuple(17.0, 17.0, 0.0, 0.0), math::posInf() },
		{ std::make_tuple(1.0, 0.5, 0.5, -0.7), -1.45161193361631 },
		{ std::make_tuple(3.0, 4.0, 5.0, 7.0), -4.371544118067157 },
		{ std::make_tuple(4.0, 3.0, 5.0, 6.0), -1.977475625566246 },
	};
	testDistribution( vals, stats::Skewnormal::logpdf<false>, stats::Skewnormal::logpdf<true> );
}

TEST(Stats, testSkewnormalDx) {
	TestValues<double, double, double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0, 0.0), 1.0 },
		{ std::make_tuple(0.0, 1.0, 1.0, 0.0), 1.0 },
		{ std::make_tuple(0.5, 5.0, 1.0, 0.0), 4.5 },
		{ std::make_tuple(1.0, 0.5, 0.5, 0.0), -2.0 },
		{ std::make_tuple(1.0, 0.5, 0.5, -0.7), -3.806699075241433 },
		{ std::make_tuple(3.0, 4.0, 5.0, 7.0), 2.635680082347735 },
		{ std::make_tuple(4.0, 3.0, 5.0, 6.0), 0.2233238551539268 },
	};
	testFunction( vals_dx, stats::Skewnormal::logpdf_dx );
}

TEST(Stats, testSkewnormalGrad) {
	static_assert( std::is_same<stats::Skewnormal::GradientType, math::Gradients<3>>::value, "Skewnormal::GradientType != Gradients<3>");

	TestValues<math::Gradients<3>, double, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, 0.0), {math::NaN()} },
		{ std::make_tuple(-1.0, 0.0, 1.0, 0.0),{ -1.0, 0.0, -0.7978845608028654 } },
		{ std::make_tuple(1.0, 0.5, 0.5, 0.0),{ 2.0, 0.0, 0.7978845608028654 } },
		{ std::make_tuple(1.0, 0.5, 0.5, -0.7),{ 3.806699075241433, 1.806699075241433, 1.290499339458166 } },
		{ std::make_tuple(3.0, 4.0, 5.0, 7.0),{ -2.635680082347735, 0.327136016469547, -0.3708114403353907 } },
		{ std::make_tuple(4.0, 3.0, 5.0, 6.0),{ -0.2233238551539268, -0.2446647710307853, 0.04388730919232113 } },
	};
	testFunction( vals_grad, stats::Skewnormal::logpdf_grad );
}
