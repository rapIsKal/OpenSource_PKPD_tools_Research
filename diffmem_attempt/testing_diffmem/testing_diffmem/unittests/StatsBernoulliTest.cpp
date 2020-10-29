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
#include "stats/bernoulli.hpp"
#include "chisq_test.hpp"

TEST(Stats, testBernoulli) {
	TestValues < double, int, double > vals[] = {
		{ std::make_tuple(-1, 0.5), math::negInf() },
		{ std::make_tuple(0, 0.5), -0.693147180559945 },
		{ std::make_tuple(1, 0.5), -0.693147180559945 },
		{ std::make_tuple(2, 0.5), math::negInf() },
		{ std::make_tuple(0, 0.3), -0.3566749439387324 },
		{ std::make_tuple(1, 0.3), -1.203972804325936 },
		{ std::make_tuple(1, -0.001), math::NaN() },
		{ std::make_tuple(1, 1.001), math::NaN() },
		{ std::make_tuple(0, 0.0), 0.0 },
		{ std::make_tuple(0, 1.0), math::negInf() },
		{ std::make_tuple(1, 0.0), math::negInf() },
		{ std::make_tuple(1, 1.0), 0.0 },
	};
	testFunction(vals, stats::Bernoulli::logpdf < false >);
}

TEST(Stats, testBernoulliGrad) {
	static_assert(std::is_same<stats::Bernoulli::GradientType, math::Gradients<1>>::value, "Bernoulli::GradientType != math::Gradients<1>");
	TestValues < math::Gradients<1>, int, double > vals_grad[] = {
		{ std::make_tuple(-1, 0.5),{ math::NaN() } },
		{ std::make_tuple(0, 0.5),{ -2 } },
		{ std::make_tuple(1, 0.5),{ 2 } },
		{ std::make_tuple(2, 0.5),{ math::NaN() } },
	};
	testFunction(vals_grad, stats::Bernoulli::logpdf_grad);
}

TEST(Stats, testBernoulliRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Bernoulli>(rng, 0.5), 0.01);
	EXPECT_LT(chisq_test<stats::Bernoulli>(rng, 0.3), 0.01);
}