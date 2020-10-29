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
#include "stats/binomial.hpp"
#include "chisq_test.hpp"

TEST(Stats, testBinomial) {
	TestValues<double, int, int, double> vals[] = {
		{ std::make_tuple(-1, 1, 0.5), math::negInf() },
		{ std::make_tuple(10, 2, 0.3), math::negInf() },
		{ std::make_tuple(1, 2, -0.1), math::NaN() },
		{ std::make_tuple(1, 2, 1.1), math::NaN() },
		{ std::make_tuple(0, 0, 0.1), 0 },
		{ std::make_tuple(1, 0, 0.1), math::negInf() },
		{ std::make_tuple(0, 1, 0.5), -0.693147180559945 },
		{ std::make_tuple(0, 2, 0.0), 0 },
		{ std::make_tuple(1, 2, 0.0), math::negInf() },
		{ std::make_tuple(1, 2, 1.0), math::negInf() },
		{ std::make_tuple(2, 2, 1.0), 0 },
		{ std::make_tuple(5, 20, 0.1), -3.444479865707075 },
		{ std::make_tuple(17, 2000, 0.01), -2.57562857170888 },
	};
	testFunction(vals, stats::Binomial::logpdf<false>);
}

TEST(Stats, testBinomialGrad) {
	static_assert(std::is_same<stats::Binomial::GradientType, math::Gradients<1>>::value, "Binomial::GradientType != math::Gradients<1>");
	TestValues<math::Gradients<1>, int, int, double> vals_grad[] = {
		{ std::make_tuple(1, 2, 0.5), { 0 } },
		{ std::make_tuple(1, 2, 0.99), { -98.98989898989889 } },
		{ std::make_tuple(11, 12, 0.99), { -88.888888888888843 } },
		{ std::make_tuple(5, 20, 0.1), { 33.33333333333333 } },
		{ std::make_tuple(17, 2000, 0.01), { -303.03030303030306 } },
	};
	testFunction(vals_grad, stats::Binomial::logpdf_grad);
}

TEST(Stats, testBinomialCdf) {
	TestValues<double, int, int, double> cdf_vals[] = {
		{ std::make_tuple(17, 45, 0.5), 0.06757822542283530020679 },
	};

	auto ftol = [](double expected) { return epsilon(expected, 1e-9, 1e-9); };
	testFunction(cdf_vals, stats::Binomial::cdf, ftol);
}

TEST(Stats, testBinomialRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Binomial>(rng, 45, 0.5), 0.01);
}