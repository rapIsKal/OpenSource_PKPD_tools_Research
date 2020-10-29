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
#include "stats/halfnormal.hpp"

TEST(Stats, testHalfnormal) {
	TestValues < double, double, double > vals[] = {
		{ std::make_tuple(-1.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 0.0), math::posInf() },
		{ std::make_tuple(0.1, 0.0), math::negInf() },
		{ std::make_tuple(1.0, 0.5), -1.532644172084782 },
		{ std::make_tuple(1.0, 1.5), -0.853478682975114 },
		{ std::make_tuple(3.0, 5.0), -2.015229265078828 },
		{ std::make_tuple(4.0, 5.0), -2.155229265078828 },
		{ std::make_tuple(1.0, 0.1), -47.92320625965068 },
	};
	testDistribution(vals, stats::Halfnormal::logpdf < false >, stats::Halfnormal::logpdf < true > );
}

TEST(Stats, testHalfnormalDx) {
	TestValues < double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.5), -4.0 },
		{ std::make_tuple(1.0, 1.5), -0.4444444444444444 },
		{ std::make_tuple(3.0, 5.0), -0.12 },
		{ std::make_tuple(4.0, 5.0), -0.16 },
		{ std::make_tuple(1.0, 0.1), -99.99999999999999 },
	};
	testFunction(vals_dx, stats::Halfnormal::logpdf_dx);
}

TEST(Stats, testHalfnormalGrad) {
	static_assert( std::is_same<stats::Halfnormal::GradientType, math::Gradients<1>>::value, "Halfnormal::GradientType != math::Gradients<1>");
	TestValues < math::Gradients<1>, double, double > vals_grad[] = {
		{ std::make_tuple(-1.0, 1.0),{ math::NaN() } },
		{ std::make_tuple(1.0, -1.0),{ math::NaN() } },
		{ std::make_tuple(1.0, 0.5),{ 6.0 } },
		{ std::make_tuple(1.0, 1.5),{ -0.3703703703703703 } },
		{ std::make_tuple(3.0, 5.0),{ -0.128 } },
		{ std::make_tuple(4.0, 5.0),{ -0.072 } },
		{ std::make_tuple(1.0, 0.1),{ 989.9999999999999 } },
	};
	testFunction(vals_grad, stats::Halfnormal::logpdf_grad);
}
