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
#include "math/nthroot.hpp"

TEST(Math, testRoot) {
	EXPECT_EQ(5.0, math::nthroot<double>(5, 1));
	EXPECT_EQ(1.0, math::nthroot<double>(1, 2));
	EXPECT_EQ(1.0, math::nthroot<double>(1, 3));
	EXPECT_SIMILAR(1.3894954943731376371, math::nthroot<double>(10, 7), 1e-10);
	EXPECT_SIMILAR(0.4641588833612778892, math::nthroot<double>(10, -3), 1e-10);
	EXPECT_SIMILAR(1.000005372712983290144, math::nthroot<double>(10000000, 3000000), 1e-10);
	EXPECT_SIMILAR(1.22578147646429903009, math::nthroot<double>(12.3456, 12.3456), 1e-10);
	EXPECT_SIMILAR(-3, math::nthroot<double>(-27, 3), 1e-10);
	EXPECT_DOMAIN_ERROR( math::nthroot<double>(-27, 2) );
}
