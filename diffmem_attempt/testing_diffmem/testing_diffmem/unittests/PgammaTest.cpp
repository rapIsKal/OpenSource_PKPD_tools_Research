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
#include "math/number.hpp"
#include "functional.hpp"
#include "math/pqgamma.hpp"

TEST(Math, testPgamma) {
	TestValues<double, double, double> vals[] = {
		{ std::make_tuple(0.03, 0.1), 0.7382350532339351 },
		{ std::make_tuple(0.3, 0.1), 0.9083579897300343 },
		{ std::make_tuple(1.5, 0.1), 0.9886559833621947 },
		{ std::make_tuple(0.075, 0.5), 0.3014646416966613 },
		{ std::make_tuple(0.75, 0.5), 0.7793286380801532 },
		{ std::make_tuple(3.5, 0.5), 0.9918490284064972 },
		{ std::make_tuple(0.1, 1), 0.09516258196404043 },
		{ std::make_tuple(1, 1), 0.6321205588285577 },
		{ std::make_tuple(5, 1), 0.9932620530009145 },
		{ std::make_tuple(0.1, 1.1), 0.07205974576054322 },
		{ std::make_tuple(1, 1.1), 0.5891809618706485 },
		{ std::make_tuple(5, 1.1), 0.9915368159845525 },
		{ std::make_tuple(0.15, 2), 0.01018582711118352 },
		{ std::make_tuple(1.5, 2), 0.4421745996289254 },
		{ std::make_tuple(7, 2), 0.9927049442755639 },
		{ std::make_tuple(2.5, 6), 0.04202103819530612 },
		{ std::make_tuple(12, 6), 0.9796589705830716 },
		{ std::make_tuple(16, 11), 0.9226039842296428 },
		{ std::make_tuple(25, 26), 0.4470785799755852 },
		{ std::make_tuple(45, 41), 0.7444549220718699 },
		{ std::make_tuple(0.0, 0.0), 0.0 },
		{ std::make_tuple(0.0, -1.0), 0.0 },
		{ std::make_tuple(0.0, 1.0), 0.0 },
		{ std::make_tuple(1.0, 0.0), math::NaN() },
		{ std::make_tuple(-4.0, 3.0), math::NaN() },
		{ std::make_tuple(4.0, -3.0), math::NaN() },
		{ std::make_tuple(4.0, -3.0), math::NaN() },
	};

	testFunction(vals, math::pgamma, constant(1e-8));
}
