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
#include "NormalDistribution.hpp"

TEST(NormalDistribution, testMVN) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Vecd mu(5);
	mu << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	Vecd x = Vecd::Zero(5);

	MvNDistribution N(mu, omega);
	EXPECT_SIMILAR(-0.8752981582594985, N.logpdf(x), 1e-10 );

	Vecd grad(5), grad_fd(5);
	EXPECT_SIMILAR(-0.8752981582594985, N.logpdf(x, grad), 1e-10);
	EXPECT_SIMILAR(-0.8752981582594985, N.Distribution::logpdf(x, grad_fd), 1e-10);
	AssertEqual(grad, grad_fd, 1e-4);
}
