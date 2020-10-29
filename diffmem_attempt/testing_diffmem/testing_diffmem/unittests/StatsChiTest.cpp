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
#include "stats/chi.hpp"

TEST(Stats, testChi) {
	TestValues<double, double, double> vals[] = {
		{ std::make_tuple(2.0, 1), -2.2257913526447276725 },
		{ std::make_tuple(0.25, 2), -1.4175443611198905956 },
		{ std::make_tuple(3.9, 3), -5.1088382463735254394 },
		{ std::make_tuple(2.0, -1), math::NaN() },
		{ std::make_tuple(-2.0, 1), math::negInf() },
		{ std::make_tuple(0.0, 1), -0.22579135264472738 },
		{ std::make_tuple(0.0, 2), math::negInf() },
		{ std::make_tuple(0.0, 4), math::negInf() },
		{ std::make_tuple(0.0, 0.5), math::posInf() },
		{ std::make_tuple(2.0, 0.5), -3.1147357295580913 },
	};
	testDistribution( vals, stats::Chi::logpdf<false>, stats::Chi::logpdf<true> );
}

TEST(Stats, testChiDx) {
	TestValues<double, double, double> vals_dx[] = {
		{ std::make_tuple(2.0, 1), -2.0 },
		{ std::make_tuple(0.25, 2), 3.75 },
		{ std::make_tuple(3.9, 3), -3.3871794871794875803 },
		{ std::make_tuple(2.0, -1), math::NaN() },
		{ std::make_tuple(-2.0, 1), math::NaN() },
		{ std::make_tuple(-2.0, 2), math::NaN() },
		{ std::make_tuple(0.0, 1), 0.0 },
		{ std::make_tuple(0.0, 2), math::posInf() },
		{ std::make_tuple(0.0, 4), math::posInf() },
		{ std::make_tuple(0.0, 0.5), math::negInf() },
		{ std::make_tuple(2.0, 0.5), -2.25 },
	};
	testFunction( vals_dx, stats::Chi::logpdf_dx );
}

TEST(Stats, testChiGrad) {
	static_assert(std::is_same<stats::Chi::GradientType, math::Gradients<1>>::value, "Chi::GradientType != math::Gradients<1>");
	TestValues < math::Gradients<1>, double, double > vals_grad[] = {
		{ std::make_tuple(2.0, 1.0),{ 1.328328603290684 } },
		{ std::make_tuple(0.25, 2.0),{ -1.444260118949097 } },
		{ std::make_tuple(3.9, 3.0),{ 0.9961579758663397 } },
		{ std::make_tuple(2.0, -1.0),{ math::NaN() } },
		{ std::make_tuple(-2.0, 1.0),{ math::NaN() } },
		{ std::make_tuple(-2.0, 2.0),{ math::NaN() } },
		{ std::make_tuple(0.0, 1.0),{ math::negInf() } },
		{ std::make_tuple(0.0, 2.0),{ math::negInf() } },
		{ std::make_tuple(0.0, 4.0),{ math::negInf() } },
		{ std::make_tuple(0.0, 0.5),{ math::negInf() } },
		{ std::make_tuple(2.0, 0.5),{ 2.460300356968105 } },
	};
	testFunction(vals_grad, stats::Chi::logpdf_grad);
}
