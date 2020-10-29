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
#include "Dosing.hpp"

TEST(DosingTest, testSimple) {
	Dosing dosing;

	dosing.addEvent(1.0, 200, 1);
	dosing.addEvent(4.0, 100, 2);
	dosing.addEvent(2.0, 150, 1);

	double amount;
	int cmt;
	DosingIterator it( dosing );

	EXPECT_NEAR( 1.0, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_NEAR( 2.0, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 150, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_NEAR( 4.0, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 100, amount, 1e-12 );
	EXPECT_EQ( 2, cmt );

	EXPECT_TRUE( math::isPosInf( it.nextEventTime() ) );
}

TEST(DosingTest, testRepeat) {
	Dosing dosing;

	dosing.addEvent(0.5, 200, 2, 10, 1);

	double amount;
	int cmt;
	DosingIterator it( dosing );

	EXPECT_NEAR( 0.5, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_NEAR( 10.5, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_NEAR( 20.5, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_TRUE( math::isPosInf( it.nextEventTime() ) );
}

TEST(DosingTest, testMix) {
	Dosing dosing;

	dosing.addEvent(0.5, 200, 2, 10, 1);
	dosing.addEvent(5.0, 150, 2);

	double amount;
	int cmt;
	DosingIterator it( dosing );

	EXPECT_NEAR( 0.5, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_NEAR( 5.0, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 150, amount, 1e-12 );
	EXPECT_EQ( 2, cmt );

	EXPECT_NEAR( 10.5, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_NEAR( 20.5, it.nextEventTime(), 1e-12 );
	it.popNextEvent(amount, cmt);
	EXPECT_NEAR( 200, amount, 1e-12 );
	EXPECT_EQ( 1, cmt );

	EXPECT_TRUE( math::isPosInf( it.nextEventTime() ) );
}
