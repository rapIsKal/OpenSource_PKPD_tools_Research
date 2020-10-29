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
#include "stats/geometric.hpp"

TEST(Stats, testGeometric) {
	TestValues<double, int, double> vals[] = {
		{ std::make_tuple(-1, 0.5), math::negInf() },
		{ std::make_tuple(0, 1.0), 0.0 },
		{ std::make_tuple(1, 1.0), math::negInf() },
		{ std::make_tuple(0, 0.9), -0.105360515657826 },
		{ std::make_tuple(0, 0.001), -6.90775527898214 },
		{ std::make_tuple(1, 0.5), -1.38629436111989 },
		{ std::make_tuple(2, 0.5), -2.07944154167984 },
		{ std::make_tuple(1, -0.1), math::NaN() },
		{ std::make_tuple(1, 0.0), math::NaN() },
		{ std::make_tuple(1, 1.1), math::NaN() },
		{ std::make_tuple(1, 1.0), math::negInf() },
		{ std::make_tuple(5, 0.1), -2.829387671283177 },
		{ std::make_tuple(17, 0.2), -5.402878284775666 },
	};
	testFunction(vals, stats::Geometric::logpdf<false>);
}

TEST(Stats, testGeometricGrad) {
	static_assert(std::is_same<stats::Geometric::GradientType, math::Gradients<1>>::value, "Geometric::GradientType != math::Gradients<1>");
	TestValues<math::Gradients<1>, int, double> vals_grad[] = {
		{ std::make_tuple(-1, 0.5),{ math::NaN() } },
		{ std::make_tuple(1, 1.0),{ math::NaN() } },
		{ std::make_tuple(0, 0.9),{ 1.111111111111111 } },
		{ std::make_tuple(0, 0.001),{ 1000.0} },
		{ std::make_tuple(1, 0.5),{ 0.0 } },
		{ std::make_tuple(2, 0.5),{ -2.0 } },
		{ std::make_tuple(5, 0.1),{ 4.444444444444444 } },
		{ std::make_tuple(17, 0.2),{ -16.25 } },
	};
	testFunction(vals_grad, stats::Geometric::logpdf_grad);
}
