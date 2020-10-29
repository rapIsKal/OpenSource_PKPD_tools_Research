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
#include "stats/trunc_normal.hpp"
#include "stats/trunc_normalp.hpp"
#include "chisq_test.hpp"

TEST(Stats, testTruncNormal) {
	TestValues<double, double, double, double, double, double> vals[] = {
		{ std::make_tuple(1.0, 0.0, 1.0, 0.0, math::posInf()), -0.7257913526447274 },
		{ std::make_tuple(0.0, 1.0, 1.0, -1.0, 1.0), -0.6792234403523372 },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), 1.0), -0.6838370466773807 },
		{ std::make_tuple(1.0, 0.5, 0.5, math::negInf(), math::posInf()), -0.7257913526447274 },
		{ std::make_tuple(6.0, 1.0, 0.7, 5.0, 7.0), -7.055455790352465 },
		{ std::make_tuple(5.0, 1.0, 0.7, 5.0, 7.0), 2.128217679035291 },
		{ std::make_tuple(5.01, 1.0, 0.7, 5.0, 7.0), 2.0464829851577413 },
		{ std::make_tuple(5.01, 1.0, 0.7, 5.0, math::posInf()), 2.0464829851577413 },
		{ std::make_tuple(6.99, 7.5, 3.7, 5.0, 7.0), -0.6103355529690405 },
		{ std::make_tuple(6.99, 7.5, 3.7, math::negInf(), 7.0), -1.42990067884692 },
		{ std::make_tuple(6.0, 5.5, 7.0, 5.0, 7.0), -0.6897677513609832 },
		{ std::make_tuple(5.01, 5.5, 7.0, 5.0, 7.0), -0.68966673095282 },
		{ std::make_tuple(5.01, 5.5, 7.0, 5.0, math::posInf()), -2.229532592234024 },
	};
	testDistribution( vals, stats::Truncnormal::logpdf<false>, stats::Truncnormal::logpdf<true> );
}

TEST(Stats, testTruncNormalDx) {
	TestValues<double, double, double, double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0, -25.0, 17.0), 1.0 },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), 2.0), 4.5 },
		{ std::make_tuple(1.0, 0.5, 0.5, 0.5, math::posInf()), -2.0 },
		{ std::make_tuple(6.0, 1.0, 0.7, 5.0, 7.0), -10.20408163265306 },
		{ std::make_tuple(5.01, 1.0, 0.7, 5.0, 7.0), -8.183673469387754 },
		{ std::make_tuple(5.01, 1.0, 0.7, 5.0, math::posInf()), -8.183673469387754 },
		{ std::make_tuple(6.99, 7.5, 3.7, 5.0, 7.0), 0.0372534696859021 },
		{ std::make_tuple(6.99, 7.5, 3.7, math::negInf(), 7.0), 0.0372534696859021 },
	};
	testFunction(vals_dx, stats::Truncnormal::logpdf_dx);
}

TEST(Stats, testTruncNormalGrad) {
	static_assert(std::is_same<stats::Truncnormal::GradientType, math::Gradients<4>>::value, "Truncnormal::GradientType != Gradients<4>");

	TestValues<math::Gradients<4>, double, double, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, math::negInf(), math::posInf()),{ math::NaN() } },
		{ std::make_tuple(-1.0, 0.0, 1.0, math::negInf(), math::posInf()),{ -1.0, 0.00, math::NaN(), math::NaN() } },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), math::posInf()),{ -4.5, 19.25, math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.5, 0.5, math::negInf(), math::posInf()),{ 2.0, 0.0, math::NaN(), math::NaN() } },
		{ std::make_tuple(6.0, 5.5, 7.0, 5.0, 7.0),{ 6.922634123765223e-05, -0.0009592994590079101, 0.5016925729538706, -0.4915577176624551 } },
		{ std::make_tuple(5.01, 5.5, 7.0, 5.0, 7.0),{ -0.02013485529141541, -0.0009881624327688337, 0.5016925729538706, -0.4915577176624551 } },
		{ std::make_tuple(5.01, 5.5, 7.0, 5.0, math::posInf()),{ -0.1175678344113756, -0.13447372611347, 0.1075678344113755, math::NaN() } },
		{ std::make_tuple(6.99, 7.5, 3.7, 5.0, 7.0),{ 0.06967566522690576, -0.0436717864441925, 0.436439844036541, -0.5433689789493489 } },
		{ std::make_tuple(6.99, 7.5, 3.7, math::negInf(), 7.0),{ 0.2021675240786463, -0.297489520903369, math::NaN(), -0.2394209937645484 } },
	};
	testFunction(vals_grad, stats::Truncnormal::logpdf_grad);
}

TEST(Stats, testTruncNormalp) {
	TestValues<double, double, double, double, double, double> vals[] = {
		{ std::make_tuple(1.0, 0.0, 1.0, 0.0, math::posInf()), -0.7257913526447274 },
		{ std::make_tuple(0.0, 1.0, 1.0, -1.0, 1.0), -0.6792234403523372 },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), 1.0), -0.6838370466773807 },
		{ std::make_tuple(1.0, 0.5, 4.0, math::negInf(), math::posInf()), -0.7257913526447274 },
		{ std::make_tuple(0.0, 0.0, 2.0, math::negInf(), math::posInf()), -0.5723649429247001 },
		{ std::make_tuple(6.0, 5.5, 0.0204081632653061, 5.0, 7.0), -0.6897677513609834 },
		{ std::make_tuple(5.01, 5.5, 0.0204081632653061, 5.0, 7.0), -0.6896667309528202 },
		{ std::make_tuple(5.01, 5.5, 0.0204081632653061, 5.0, math::posInf()), -2.229532592234024 },
		{ std::make_tuple(6.99, 7.5, 0.0730460189919649, 5.0, 7.0), -0.6103355529690409 },
		{ std::make_tuple(6.99, 7.5, 0.0730460189919649, math::negInf(), 7.0), -1.42990067884692 },
	};
	testDistribution( vals, stats::Truncnormalp::logpdf<false>, stats::Truncnormalp::logpdf<true> );
}

TEST(Stats, testTruncNormalpDx) {
	TestValues<double, double, double, double, double, double> vals_dx[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(-1.0, 0.0, 1.0, 0.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 1.0, 1.0, -25.0, 17.0), 1.0 },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), 2.0), 4.5 },
		{ std::make_tuple(6.0, 5.5, 0.0204081632653061, 5.0, 7.0), -0.01020408163265306 },
		{ std::make_tuple(5.01, 5.5, 0.0204081632653061, 5.0, 7.0), 0.01 },
		{ std::make_tuple(5.01, 5.5, 0.0204081632653061, 5.0, math::posInf()), 0.01 },
		{ std::make_tuple(6.99, 7.5, 0.0730460189919649, 5.0, 7.0), 0.0372534696859021 },
		{ std::make_tuple(6.99, 7.5, 0.0730460189919649, math::negInf(), 7.0), 0.0372534696859021 },
	};
	testFunction(vals_dx, stats::Truncnormalp::logpdf_dx);
}

TEST(Stats, testTruncNormalpGrad) {
	static_assert(std::is_same<stats::Truncnormalp::GradientType, math::Gradients<4>>::value, "Truncnormalp::GradientType != Gradients<4>");

	TestValues<math::Gradients<4>, double, double, double, double, double> vals_grad[] = {
		{ std::make_tuple(-1.0, 0.0, -1.0, math::negInf(), math::posInf()),{ math::NaN() } },
		{ std::make_tuple(-1.0, 0.0, 1.0, math::negInf(), math::posInf()),{ -1.0, 0.00, math::NaN(), math::NaN() } },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), math::posInf()),{ -4.5, -9.625, math::NaN(), math::NaN() } },
		{ std::make_tuple(6.0, 5.5, 0.0204081632653061, 5.0, 7.0),{ 6.922634123765223e-05, 0.164519857219847, 0.5016925729538706, -0.4915577176624551 } },
		{ std::make_tuple(5.01, 5.5, 0.0204081632653061, 5.0, 7.0),{ -0.02013485529141541, 0.169469857219848, 0.5016925729538706, -0.4915577176624551 } },
		{ std::make_tuple(5.01, 5.5, 0.0204081632653061, 5.0, math::posInf()),{ -0.1175678344113756, 23.062244028460675, 0.10756783441138, math::NaN() } },
		{ std::make_tuple(6.99, 7.5, 0.0730460189919649, 5.0, 7.0),{ 0.06967566522690578, 1.106053499378842, 0.4364398440365409, -0.5433689789493488 } },
		{ std::make_tuple(6.99, 7.5, 0.0730460189919649, math::negInf(), 7.0),{ 0.2021675240786463, 7.53436835115917, math::NaN(), -0.2394209937645484 } },
	};
	testFunction(vals_grad, stats::Truncnormalp::logpdf_grad);
}

TEST(Stats, testTruncNormalCdf) {
	TestValues<double, double, double, double, double, double> vals[] = {
		{ std::make_tuple(1.0, 0.0, 1.0, 0.0, math::posInf()), 0.6826894921370859 },
		{ std::make_tuple(0.0, 1.0, 1.0, -1.0, 1.0), 0.2847672279890939 },
		{ std::make_tuple(0.5, 5.0, 1.0, math::negInf(), 1.0), 0.10727944116121195 },
		{ std::make_tuple(1.0, 0.5, 0.5, math::negInf(), math::posInf()), 0.8413447460685429 },
		{ std::make_tuple(6.0, 1.0, 0.7, 5.0, 7.0), 0.99991701981116232 },
		{ std::make_tuple(5.0, 1.0, 0.7, 5.0, 7.0), 0 },
		{ std::make_tuple(6.99, 7.5, 3.7, 5.0, 7.0), 0.9945673089765716 },
		{ std::make_tuple(6.99, 7.5, 3.7, math::negInf(), 7.0), 0.9976062301418827 },
		{ std::make_tuple(5.01, 5.5, 7.0, 5.0, 7.0), 0.005017179997276218 },
		{ std::make_tuple(5.01, 5.5, 7.0, 5.0, math::posInf()), 0.001075732861623799 },
	};

	testFunction( vals, stats::Truncnormal::cdf );
}

TEST(Stats, testTruncNormalIcdf) {
	TestValues<double, double, double, double, double, double> vals[] = {
		{ std::make_tuple(0.68, 0.0, 1.0, 0.0, math::posInf()), 0.9944578832097535 },
		{ std::make_tuple(0.28, 1.0, 1.0, -1.0, 1.0), -0.009447245198912269 },
		{ std::make_tuple(0.1, 5.0, 1.0, math::negInf(), 1.0), 0.4850860608243073 },
		{ std::make_tuple(0.8, 0.5, 0.5, math::negInf(), math::posInf()), 0.9208106167864571 },
		{ std::make_tuple(0.99, 1.0, 0.7, 5.0, 7.0), 5.516609368943306 },
		{ std::make_tuple(0, 1.0, 2, 5.0, 7.0), 5 },
		{ std::make_tuple(0.99, 7.5, 3.7, 5.0, 7.0), 6.981590035556193 },
		{ std::make_tuple(0.99, 7.5, 3.7, math::negInf(), 7.0), 6.958199788869407 },
		{ std::make_tuple(0.005, 5.5, 7.0, 5.0, 7.0), 5.009965759379945 },
		{ std::make_tuple(0.005, 5.5, 7.0, 5.0, math::posInf()), 5.046471618066342 },
	};

	testFunction( vals, stats::Truncnormal::icdf );
}

TEST(Stats, testTruncNormalRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Truncnormal>(rng, 0.0, 1.0, -1, 1), 0.01);
	EXPECT_LT(chisq_test<stats::Truncnormal>(rng, 0.0, 1.0, 4.0, 6.0), 0.01);
	EXPECT_LT(chisq_test<stats::Truncnormal>(rng, 0.0, 1.0, -3.0, 6.0), 0.01);
	EXPECT_LT(chisq_test<stats::Truncnormal>(rng, 5.0, 0.5, 4.0), 0.01);
}