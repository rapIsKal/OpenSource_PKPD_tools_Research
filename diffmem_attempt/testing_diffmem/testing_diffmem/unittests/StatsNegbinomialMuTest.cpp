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
#include "stats/negbinomial_mu.hpp"
#include "chisq_test.hpp"

TEST(Stats, testNegbinomialMu) {
	TestValues < double, int, double, double > vals[] = {
		{ std::make_tuple(100000,9999,1.0), -19.210840405352918 },
		{ std::make_tuple(0, 0, 0), 0.0 },
		{ std::make_tuple(1, 0, 0), math::negInf() },
		{ std::make_tuple(1, 0.4, .1), -2.68667243555167 },
		{ std::make_tuple(1, 0.1, 1.0), -2.493205452602695 },
		{ std::make_tuple(-1, 0.1, 1.0), math::negInf() },
		{ std::make_tuple(1, -0.1, 2.0), math::NaN() },
		{ std::make_tuple(1, 0.0, 2.0), math::negInf() },
		{ std::make_tuple(0, 0.0, 2.0), 0 },
		{ std::make_tuple(5, 85.5, 9.5), -15.01542028362279 },
		{ std::make_tuple(17, 80, 20.), -13.10746171621675 },
		{ std::make_tuple(7, 22, 5.5), -4.1625411943347 },
		{ std::make_tuple(7, .5, math::posInf()), -13.87719162498503 },
	};
	testFunction(vals, stats::NegbinomialMu::logpdf < false >);
}

TEST(Stats, testNegbinomialMuGrad) {
	static_assert(std::is_same<stats::Negbinomial::GradientType, math::Gradients<2>>::value, "Negbinomial::GradientType != math::Gradients<2>");
	TestValues < math::Gradients<2>, int, double, double > vals_grad[] = {
		{ std::make_tuple(-1, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1, 1.1, math::posInf()),{ -0.0909090909090909, 0.0 } },
		{ std::make_tuple(0, 0.0, 0.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(5, 85.5, 9.5),{ -1.861053244048009, 89.44444444444444 } },
		{ std::make_tuple(17, 80, 20.0),{ -0.9826183727831431, 78.75 } },
		{ std::make_tuple(7, 22, 5.5),{ -0.73533540974093992, 18.75 } },
	};
	testFunction(vals_grad, stats::NegbinomialMu::logpdf_grad);
}

TEST(Stats, testNegbinomialMuCdf) {
	TestValues<double, int, double, double> cdf_vals[] = {
		{ std::make_tuple(10, 0.0, 0.3), 1.0 },
		{ std::make_tuple(0, 0.0, 0.0), 1.0 },
		{ std::make_tuple(5, 85.5, 9.5), 4.850504723768924e-07 },
		{ std::make_tuple(17, 80, 20.0), 4.680090701535518e-06 },
		{ std::make_tuple(7, 22, 5.5), 0.04712456960893529 },
		{ std::make_tuple(7, .5, math::posInf()), 0.9999999378030913 },
	};

	auto ftol = [](double expected) { return epsilon(expected, 1e-9, 1e-9); };
	testFunction(cdf_vals, stats::NegbinomialMu::cdf, ftol);
}

TEST(Stats, testNegbinomialMuRng) {
	Random rng;
	//EXPECT_LT(chisq_test<stats::NegbinomialMu>(rng, 45, 0), 0.01);
	EXPECT_LT(chisq_test<stats::NegbinomialMu>(rng, 20, 5), 0.01);
}
