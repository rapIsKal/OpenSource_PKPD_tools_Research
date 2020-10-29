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
#include "stats/levy.hpp"

TEST(Stats, testLevy) {
	TestValues < double, double, double, double > vals[] = {
		{ std::make_tuple(2.0, 0.0, 0.0), math::NaN() },
		{ std::make_tuple(2.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(-1.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(1.0, 2.0, 1.0), math::negInf() },
		{ std::make_tuple(2.0, 2.0, 1.0), math::negInf() },
		{ std::make_tuple(1.0, 0.0, 5.0), -2.614219576987623 },
		{ std::make_tuple(1.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(2.0, 2.0, 2.0), math::negInf() },
		{ std::make_tuple(2.0, 0.0, 1.0), -2.20865930404459 },
		{ std::make_tuple(0.0, -1.0, 5.0), -2.614219576987623 },
		{ std::make_tuple(5.0, 2.0, 1.0), -2.7335236328735 },
		{ std::make_tuple(5.01, 5.0, 0.001), 2.48493910628643 },
	};
	testDistribution(vals, stats::Levy::logpdf < false >, stats::Levy::logpdf < true > );
}

TEST(Stats, testLevyDx) {
	TestValues < double, double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, 5.0), 1.0 },
		{ std::make_tuple(1.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(2.0, 2.0, 2.0), math::NaN() },
		{ std::make_tuple(0.0, -1.0, 5.0), 1.0 },
		{ std::make_tuple(2.0, 0.0, 1.0), -0.625 },
		{ std::make_tuple(0.0, 2.0, 10.0), math::NaN() },
		{ std::make_tuple(5.0, 2.0, 1.0), -0.444444444444444 },
		{ std::make_tuple(5.01, 5.0, 0.001), -145.0000000000030 },
	};
	testFunction(vals_dx, stats::Levy::logpdf_dx);
}

TEST(Stats, testLevyGrad) {
	static_assert( std::is_same<stats::Levy::GradientType, math::Gradients<2>>::value, "Levy::GradientType != math::Gradients<2>");
	TestValues < math::Gradients<2>, double, double, double > vals_grad[] = {
		{ std::make_tuple(-1.0, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.0, 0.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.0, -1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.0, 5.0),{ -1.0, -0.4 } },
		{ std::make_tuple(1.0, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(2.0, 2.0, 2.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(0.0, -1.0, 5.0),{ -1.0, -0.4 } },
		{ std::make_tuple(2.0, 0.0, 1.0),{ 0.625, 0.25 } },
		{ std::make_tuple(0.0, 2.0, 10.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(5.0, 2.0, 1.0),{ 0.444444444444444, 0.333333333333333 } },
		{ std::make_tuple(5.01, 5.0, 0.001),{ 145.0000000000030, 450.0 } },
	};
	testFunction(vals_grad, stats::Levy::logpdf_grad);
}
