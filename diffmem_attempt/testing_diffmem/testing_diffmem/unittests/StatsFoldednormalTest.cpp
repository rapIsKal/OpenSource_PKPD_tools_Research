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
#include "stats/foldednormal.hpp"

TEST(Stats, testFoldednormal) {
	TestValues < double, double, double, double > vals[] = {
		{ std::make_tuple(-1.0, -2.0, 1.0), math::negInf() },
		{ std::make_tuple(2.0, 2.0, 0.0), math::posInf() },
		{ std::make_tuple(2.1, 2.0, 0.0), math::negInf() },
		{ std::make_tuple(1.0, 1.0, 1.0), -0.7920105221617 },
		{ std::make_tuple(1.0, -1.0, 1.0), -0.7920105221617 },
		{ std::make_tuple(1.0, 0.5, 0.5), -0.7076414247269178 },
		{ std::make_tuple(1.0, 1.5, 0.5), -0.7257852084512497 },
		{ std::make_tuple(3.0, 4.0, 5.0), -2.224198846443584 },
		{ std::make_tuple(4.0, 3.0, 5.0), -2.224198846443584 },
		{ std::make_tuple(1.0, 1.1, 0.1), 0.883646559789372 },
		{ std::make_tuple(1.0, 1.1, 0.2), 0.565499379229427 },
		{ std::make_tuple(1.0, -0.5, 1.0), -0.73067684568645 },
		{ std::make_tuple(3.0, 0.0, 5.0), -2.015229265078828 },
		{ std::make_tuple(4.0, 0.0, 5.0), -2.155229265078828 },
		{ std::make_tuple(1.0, 0.0, 0.1), -47.92320625965068 },
	};
	testDistribution(vals, stats::Foldednormal::logpdf < false >, stats::Foldednormal::logpdf < true > );
}

TEST(Stats, testFoldednormalDx) {
	TestValues < double, double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, -1.0), math::NaN() },
		{ std::make_tuple(1.0, 0.5, 0.5), -2.071944839848366 },
		{ std::make_tuple(1.0, 1.0, 1.0), -0.238405844044235 },
		{ std::make_tuple(1.0, -1.0, 1.0), -0.238405844044235 },
		{ std::make_tuple(1.0, 1.5, 0.5), 1.999926269904774 },
		{ std::make_tuple(3.0, 4.0, 5.0), -0.04860102236019525 },
		{ std::make_tuple(4.0, 3.0, 5.0), -0.1064507667701464 },
		{ std::make_tuple(1.0, 1.1, 0.1), 10.0 },
		{ std::make_tuple(1.0, 1.1, 0.2), 2.5 },
		{ std::make_tuple(1.0, -0.5, 1.0), -0.7689414213700 },
		{ std::make_tuple(3.0, 0.0, 5.0), -0.12 },
		{ std::make_tuple(4.0, 0.0, 5.0), -0.16 },
		{ std::make_tuple(1.0, 0.0, 0.1), -99.99999999999999 },

	};
	testFunction(vals_dx, stats::Foldednormal::logpdf_dx);
}

TEST(Stats, testFoldednormalGrad) {
	static_assert( std::is_same<stats::Foldednormal::GradientType, math::Gradients<2>>::value, "Foldednormal::GradientType != math::Gradients<2>");
	TestValues < math::Gradients<2>, double, double, double > vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, 1.0),{ -0.238405844044235, -0.523188311911530 } },
		{ std::make_tuple(1.0, -1.0, 1.0),{ 0.238405844044235, -0.523188311911530 } },
		{ std::make_tuple(1.0, 0.5, 0.5),{ 1.856110320303268, 0.2877793593934651 } },
		{ std::make_tuple(1.0, 1.5, 0.5),{ -2.00004915339682, 0.0002949203809063889 } },
		{ std::make_tuple(3.0, 4.0, 5.0),{ -0.1064507667701464, -0.08567877316776569 } },
		{ std::make_tuple(4.0, 3.0, 5.0),{ -0.04860102236019525, -0.08567877316776569 } },
		{ std::make_tuple(1.0, 1.1, 0.2),{ -2.5, -3.75 } },
		{ std::make_tuple(1.0, -0.5, 1.0),{ 0.0378828427399903, -0.212117157260010 } },
		{ std::make_tuple(1.0, 0.0, 1.5),{ 0.0, -0.3703703703703703 } },
		{ std::make_tuple(3.0, 0.0, 5.0),{ 0.0, -0.128 } },
		{ std::make_tuple(4.0, 0.0, 5.0),{ 0.0, -0.072 } },



	};
	testFunction(vals_grad, stats::Foldednormal::logpdf_grad);
}
