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
#include "stats/fdistribution.hpp"

TEST(Stats, testFdistribution) {
	TestValues < double, double, double, double > vals[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(0.0, 1.0, 5.0), math::posInf() },
		{ std::make_tuple(0.0, 2.0, 5.0), 0 },
		{ std::make_tuple(0.0, 3.0, 5.0), math::negInf() },
		{ std::make_tuple(1.0, 0.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, -0.1), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), -1.837877066409345 },
		{ std::make_tuple(2.0, 2.0, 2.0), -2.197224577336219 },
		{ std::make_tuple(0.001, 2.0, 10.0), -0.001199880015997468 },
		{ std::make_tuple(3.0, 100.0, 0.1), -4.28936569465731 },
		{ std::make_tuple(1.0, 1.5, 150.0), -1.174028672103134 },
		{ std::make_tuple(0.0, 1.5, math::posInf()), math::posInf() },
		{ std::make_tuple(0.0, 2.0, math::posInf()), 0.0 },
		{ std::make_tuple(0.0, 2.5, math::posInf()), math::negInf() },
		{ std::make_tuple(2.0, 1.5, math::posInf()), -2.09232930091012 },
		{ std::make_tuple(2.0, 2.0, math::posInf()), -2.0 },
		{ std::make_tuple(0.0, math::posInf(), 150.0), math::negInf() },
		{ std::make_tuple(1.1, math::posInf(), 150.0), 0.813302572089555 },
		{ std::make_tuple(0.0, math::posInf(), math::posInf()), math::NaN() },
		{ std::make_tuple(0.5, math::posInf(), math::posInf()), math::NaN() },
	};
	testDistribution(vals, stats::Fdistribution::logpdf<false>, stats::Fdistribution::logpdf<true>);
}

TEST(Stats, testFdistributionDx) {
	TestValues < double, double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 1.0, 1.0), -1.0 },
		{ std::make_tuple(2.0, 2.0, 2.0), -0.6666666666666667 },
		{ std::make_tuple(0.0, 1.0, 5.0), math::NaN() },
		{ std::make_tuple(0.001, 2.0, 10.0), -1.199760047990402 },
		{ std::make_tuple(3.0, 100.0, 0.1), -0.3444407419748998 },
		{ std::make_tuple(1.0, 1.5, 150.0), -1.0 },
		{ std::make_tuple(2.0, 1.5, math::posInf()), -0.875 },
		{ std::make_tuple(0.5, math::posInf(), 150.0), 148.0 },
		{ std::make_tuple(0.5, math::posInf(), math::posInf()), math::NaN() },
	};
	testFunction(vals_dx, stats::Fdistribution::logpdf_dx);
}

TEST(Stats, testFdistributionGrad) {
	static_assert( std::is_same<stats::Fdistribution::GradientType, math::Gradients<2>>::value, "Fdistribution::GradientType != math::Gradients<2>");
	TestValues < math::Gradients<2>, double, double, double > vals_grad[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(0.0, 2.0, 5.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.0, 5.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, 1.0),{ 0.3465735902799727, 0.3465735902799727 } },
		{ std::make_tuple(2.0, 2.0, 2.0),{ 0.1306007792792511, 0.1173605223326118 } },
		{ std::make_tuple(0.001, 2.0, 10.0),{ -2.61762979906678, 1.998600346575508e-05 } },
		{ std::make_tuple(3.0, 100.0, 0.1),{ 4.917297969935776e-06, 8.53029394734283 } },
		{ std::make_tuple(1.0, 1.5, 150.0),{ 0.3957818123086628, 3.314925038195565e-05 } },
		{ std::make_tuple(2.0, 1.5, math::posInf()),{ 0.245662993947318, math::NaN() } },
		{ std::make_tuple(0.5, math::posInf(), 150.0),{ math::NaN(), -0.15008566911096 } },
		{ std::make_tuple(0.5, math::posInf(), math::posInf()),{ math::NaN(), math::NaN() } },
	};
	testFunction(vals_grad, stats::Fdistribution::logpdf_grad);
}
