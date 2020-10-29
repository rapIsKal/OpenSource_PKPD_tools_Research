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
#include "stats/studentt.hpp"
#include "stats/normal.hpp"
#include "chisq_test.hpp"

TEST(Stats, testStudentt) {
	static TestValues<double, double, double> vals[] = {
		{ std::make_tuple(-1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.5), -2.13449214241259 },
		{ std::make_tuple(-1.0, 1.0), -1.83787706640935 },
		{ std::make_tuple(-1.0, 2.0), -1.64791843300216 },
		{ std::make_tuple(-1.0, 30.0), -1.43551257913520 },
		{ std::make_tuple(0.0, 3.0), -1.00088884962351 },
		{ std::make_tuple(1.0, 5.0), -1.51558425943659 },
		{ std::make_tuple(-1.0, math::posInf()), stats::Normal::logpdf<false>(-1.0, 0, 1) },
		{ std::make_tuple(0.0, math::posInf()), stats::Normal::logpdf<false>(0.0, 0, 1) },
		{ std::make_tuple(7.0, math::posInf()), stats::Normal::logpdf<false>(7.0, 0, 1) },
	};

	testDistribution( vals, stats::Studentt::logpdf<false>, stats::Studentt::logpdf<true> );
}

TEST(Stats, testStudenttDx) {
	static TestValues<double, double, double> dx_vals[] = {
		{ std::make_tuple(-1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.5), 1.00000000000000 },
		{ std::make_tuple(-2.0, 1.0), 0.800000000000000 },
		{ std::make_tuple(-3.0, 2.0), 0.818181818181818 },
		{ std::make_tuple(-4.0, 30.0), 2.69565217391304 },
		{ std::make_tuple(0.0, 3.0), 0 },
		{ std::make_tuple(1.0, 5.0), -1 },
	};

	testFunction( dx_vals, stats::Studentt::logpdf_dx );
}

TEST(Stats, testStudenttGrad) {
	static_assert( std::is_same<stats::Studentt::GradientType, math::Gradients<1>>::value, "Studentt::GradientType != Gradients<1>");

	TestValues<math::Gradients<1>, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.5), 1.02149018246084 },
		{ std::make_tuple(-2.0, 1.0), 0.188428224342895 },
		{ std::make_tuple(-3.0, 2.0), -0.181884863042794 },
		{ std::make_tuple(-4.0, 30.0), -0.0337342386878144 },
		{ std::make_tuple(0.0, 3.0), 0.0264805138932786 },
		{ std::make_tuple(1.0, 5.0), 0.0186530688296347 },
	};

	testFunction( vals_grad, stats::Studentt::logpdf_grad );
}

TEST(Stats, testStudenttCdf) {
	TestValues<double, double, double> cdf_vals[] = {
		{ std::make_tuple(-1.0, 0.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.5), 0.3011216108413221 },
		{ std::make_tuple(-2.0, 1.0), 0.1475836176504332 },
		{ std::make_tuple(-3.0, 2.0), 0.04773298313335456 },
		{ std::make_tuple(-4.0, 30.0), 0.0001909228180418782 },
//		{ std::make_tuple(0.0, 3.0), 0.5 },
//		{ std::make_tuple(1.0, 5.0), 0.8183912661754387 },
	};

	auto ftol = [](double expected) { return epsilon(expected, 1e-9, 1e-9); };
	testFunction(cdf_vals, stats::Studentt::cdf, ftol);
}

TEST(Stats, testStudenttRng) {
	Random rng;
//	EXPECT_LT(chisq_test<stats::Studentt>(rng, 1.0), 0.01);
//	EXPECT_LT(chisq_test<stats::Studentt>(rng, 30), 0.01);
}
