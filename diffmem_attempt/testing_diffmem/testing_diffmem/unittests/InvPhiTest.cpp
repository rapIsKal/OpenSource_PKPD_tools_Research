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
#include "math/inv_phi.hpp"
#include "math/phi.hpp"

// inv_Phi returns `NaN` or throws if outside range [0,1]
TEST(Math, testInvPhiOutside) {
	EXPECT_DOMAIN_ERROR( math::inv_Phi(0.0) );
}

// test the inv_Phi function for boundary cases
TEST(Math, testInvPhiBoundary) {
	EXPECT_SIMILAR( math::negInf(), math::inv_Phi(0.0), 1e-15 );
	EXPECT_SIMILAR( math::posInf(), math::inv_Phi(1.0), 1e-15 );
}

// test the inv_Phi function for breakpoint cases
TEST(Math, testInvPhiBreakpoint) {
	EXPECT_NEAR( -1.972961051311884, math::inv_Phi(0.02425), 1e-15 );
	EXPECT_NEAR( 1.972961051311885, math::inv_Phi(0.97575), 1e-15 );
}

TEST(Math, testInvPhi) {
	double params[] = {
		0.5, 0.123456789, 8e-311, 0.99
	};

	double values[] = {
		0.0, -1.157878609150208, -37.668980431720044, 2.326347874040841
	};

	const int N = sizeof(values) / sizeof(values[0]);
	for(int i = 0; i < N; ++i) {
		const double val = math::inv_Phi(params[i]);
		EXPECT_SIMILAR_OR_NAN( values[i], val, 1e-14 );
	}
}

TEST(Math, testInvPhiFwdBwd) {
	double params[] = {
		0.5, 0.123456789, 8e-311, 0.99, 0.02425, 0.97575, 0.0, 1.0
	};

	const int N = sizeof(params) / sizeof(params[0]);
	for(int i = 0; i < N; ++i) {
		const double val = math::Phi( math::inv_Phi(params[i]) );
		EXPECT_SIMILAR_OR_NAN( params[i], val, 1e-15 );
	}
}
