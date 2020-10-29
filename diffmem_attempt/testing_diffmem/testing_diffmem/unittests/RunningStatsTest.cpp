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
#include "RunningStats.hpp"

TEST(RunningStatsTest, testFull) {
	RunningStats<double> stats;
	stats.push(1.025868);
	stats.push(6.455832);
	stats.push(8.367709);
	stats.push(5.946632);
	stats.push(4.701303);
	stats.push(5.017764);
	stats.push(2.184651);
	stats.push(6.828075);
	stats.push(4.986580);
	EXPECT_NEAR(stats.mean(), 5.057157, 1e-6);
	EXPECT_NEAR(stats.variance(), 5.184889, 1e-6);
}

TEST(RunningStatsTest, testPartials) {
	RunningStats<double> stats;
	stats.push(1);
	stats.push(6);
	stats.push(8);
	EXPECT_NEAR(stats.mean(), 5, 1e-6);
	EXPECT_NEAR(stats.variance(), 13, 1e-6);

	stats.push(5);
	stats.push(4);
	stats.push(5);
	EXPECT_NEAR(stats.mean(), 4.833333, 1e-6);
	EXPECT_NEAR(stats.variance(), 5.366667, 1e-6);

	stats.push(2);
	stats.push(6);
	stats.push(4);
	EXPECT_NEAR(stats.mean(), 4.555556, 1e-6);
	EXPECT_NEAR(stats.variance(), 4.527778, 1e-6);
}
