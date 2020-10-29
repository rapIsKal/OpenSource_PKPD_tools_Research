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
#include "stats/invgaussian.hpp"

TEST(Stats, testInvgaussian) {
	TestValues < double, double, double, double > vals[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::negInf() },
		{ std::make_tuple(1.0, -1.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), -0.9189385332046727 },
		{ std::make_tuple(2.0, 2.0, 2.0), -1.612085713764618 },
		{ std::make_tuple(5.0, 2.0, 20.0), -6.335229265078828 },
		{ std::make_tuple(0.001, 2.0, 10.0), -4984.407263068235 },
		{ std::make_tuple(3.0, 100.0, 0.1), -3.733831179370527 },
		{ std::make_tuple(1.0, 1.5, 150.0), -6.746954219489882 },
	};
	testDistribution(vals, stats::Invgaussian::logpdf < false >, stats::Invgaussian::logpdf < true >);
}

TEST(Stats, testInvgaussianDx) {
	TestValues<double, double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), -1.5 },
		{ std::make_tuple(2.0, 2.0, 2.0), -0.75 },
		{ std::make_tuple(5.0, 2.0, 20.0), -2.4 },
		{ std::make_tuple(0.001, 2.0, 10.0), 4998498.75 },
		{ std::make_tuple(3.0, 100.0, 0.1), -0.4944494444444444 },
		{ std::make_tuple(1.0, 1.5, 150.0), 40.16666666666667 },
	};
	testFunction(vals_dx, stats::Invgaussian::logpdf_dx);
}

TEST(Stats, testInvgaussianGrad) {
	static_assert( std::is_same<stats::Invgaussian::GradientType, math::Gradients<2>>::value, "Invgaussian::GradientType != Gradients<2>");

	TestValues<math::Gradients<2>, double, double, double> vals_grad[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(0.0, 2.0, 5.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, 1.0),{ 0.0, 0.5 } },
		{ std::make_tuple(2.0, 2.0, 2.0),{ 0.0, 0.25 } },
		{ std::make_tuple(5.0, 2.0, 20.0),{ 7.5, -0.2 } },
		{ std::make_tuple(0.001, 2.0, 10.0),{ -2.49875, -499.450125 } },
		{ std::make_tuple(3.0, 100.0, 0.1),{ -9.7e-06, 4.843183333333333 } },
		{ std::make_tuple(1.0, 1.5, 150.0),{ -22.22222222222222, -0.05222222222222225 } },
	};
	testFunction( vals_grad, stats::Invgaussian::logpdf_grad );
}
