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
#include "stats/gompertz.hpp"
#include "chisq_test.hpp"

TEST(Stats, testGompertz) {
	TestValues < double, double, double, double > vals[] = {
		{ std::make_tuple(-1.0, 1.0, 1.0), math::negInf() },
		{ std::make_tuple(-2.0, 1.0, 0.0), math::NaN() },
		{ std::make_tuple(-2.0, 1.0, -1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), 1.6094379124341 },
		{ std::make_tuple(1.0, 0.0, 5.0), -3.3905620875658995 },
		{ std::make_tuple(1.0, 1.0, 1.0), -0.7182818284590452 },
		{ std::make_tuple(2.0, 2.0, 2.0), -48.90500285258429 },
		{ std::make_tuple(0.0, -1.0, 5.0), 1.6094379124341 },
		{ std::make_tuple(0.001, 2.0, 10.0), 2.294575086324044 },
		{ std::make_tuple(1.0, 1.5, 0.1), -1.03469769768325 },
	};
	testDistribution(vals, stats::Gompertz::logpdf < false >, stats::Gompertz::logpdf < true > );
}

TEST(Stats, testGompertzDx) {
	TestValues < double, double, double, double > vals_dx[] = {
		{ std::make_tuple(-2.0, 1.0, 1.0), math::NaN() },
		{ std::make_tuple(0.0, 2.0, 5.0), math::NaN() },
		{ std::make_tuple(1.0, 0.0, 5.0), -5.0 },
		{ std::make_tuple(1.0, 1.0, 1.0), -1.718281828459045 },
		{ std::make_tuple(2.0, 2.0, 2.0), -107.1963000662885 },
		{ std::make_tuple(1.0, -1.0, 5.0), -2.83939720585721 },
		{ std::make_tuple(0.001, 2.0, 10.0), -8.020020013340003 },
		{ std::make_tuple(3.0, 100.0, 0.1), -1.942426395241256e+129 },
		{ std::make_tuple(1.0, 1.5, 150.0), -670.7533605507097 },
	};
	testFunction(vals_dx, stats::Gompertz::logpdf_dx);
}

TEST(Stats, testGompertzGrad) {
	static_assert( std::is_same<stats::Gompertz::GradientType, math::Gradients<2>>::value, "Gompertz::GradientType != math::Gradients<2>");
	TestValues < math::Gradients<2>, double, double, double > vals_grad[] = {
		{ std::make_tuple(-1.0, 1.0, 1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, 0.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 1.0, -1.0),{ math::NaN(), math::NaN() } },
		{ std::make_tuple(1.0, 0.0, 5.0),{ -1.5, -0.8 } },
		{ std::make_tuple(1.0, 1.0, 1.0),{ 0.0, -0.7182818284590452 } },
		{ std::make_tuple(2.0, 2.0, 2.0),{ -80.39722504971635, -26.29907501657212 } },
		{ std::make_tuple(1.0, -1.0, 5.0),{ -0.321205588285577, -0.432120558828558 } },
		{ std::make_tuple(0.001, 2.0, 10.0),{ 0.0009949933283307558, 0.09899899933299985 } },
		{ std::make_tuple(1.0, 1.5, 0.1),{ 0.855962465103599, 7.67887395310796 } },
	};
	testFunction(vals_grad, stats::Gompertz::logpdf_grad);
}

TEST(Stats, testGompertzCdf) {
	TestValues<double, double, double, double> vals[] = {
		{std::make_tuple(-1.0, 1.0, 1.0), math::NaN()},
		{std::make_tuple(2.0, 1.0, -1.0), math::NaN()},
		{std::make_tuple(0.0, 2.0, 5.0), 0.0},
		{std::make_tuple(1.0, 0.1, 5.0), 0.9947971353132254},
		{std::make_tuple(1.0, 1.0, 1.0), 0.8206259212659828},
		{std::make_tuple(2.0, 2.0, 2.0), 1},
		{std::make_tuple(0.0, -1.0, 5.0), 0.0},
		{std::make_tuple(0.001, 2.0, 10.0), 0.009960073303234765},
		{std::make_tuple(1.0, 1.5, 0.1), 0.2071431611180702},
	};
	testFunction(vals, stats::Gompertz::cdf);
}

TEST(Stats, testGompertzIcdf) {
	TestValues<double, double, double, double> vals[] = {
		{std::make_tuple(0.0, 1.0, 1.0), 0.0},
		{std::make_tuple(-2.0, 1.0, -1.0), math::NaN()},
		{std::make_tuple(0.9, 0.0, 5.0), 0.4605170185988092},
		{std::make_tuple(0.8, 1.0, 1.0), 0.959134838920824},
		{std::make_tuple(1.0, 2.0, 2.0), math::posInf()},
		{std::make_tuple(0.7, -2.0, 2.0), math::posInf()},
		{std::make_tuple(0.01, 2.0, 10.0), 0.001004024844374357},
		{std::make_tuple(0.2, 1.5, 0.1), 0.9796808067455089},
	};
	testFunction(vals, stats::Gompertz::icdf);
}

TEST(Stats, testGompertzRng) {
	Random rng;
	EXPECT_LT(chisq_test<stats::Gompertz>(rng, 1.0, 1.0), 0.01);
	EXPECT_LT(chisq_test<stats::Gompertz>(rng, 2.0, 5.0), 0.01);
	EXPECT_LT(chisq_test<stats::Gompertz>(rng, 100.0, 0.1), 0.01);
}