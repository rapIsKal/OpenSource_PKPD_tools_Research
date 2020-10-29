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

#include "TestCommon.hpp"

testing::AssertionResult DoubleSimilar(const char* expr1,
                                     const char* expr2,
                                     const char* abs_error_expr,
                                     double val1,
                                     double val2,
                                     double abs_error) {
  if( std::isfinite(val1)  &&  std::isfinite(val2) ) {
	  const double diff = fabs(val1 - val2);
	  if (diff <= abs_error) return testing::AssertionSuccess();

	  return testing::AssertionFailure()
		  << "The difference between " << expr1 << " and " << expr2
		  << " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
		  << expr1 << " evaluates to " << val1 << ",\n"
		  << expr2 << " evaluates to " << val2 << ", and\n"
		  << abs_error_expr << " evaluates to " << abs_error << ".";
  } else {
	  if( std::isnan(val1)  || std::isnan(val2) )
		  return testing::AssertionFailure() << "NaN found "
				  << expr1 << " evaluates to " << val1 << ",\n"
				  << expr2 << " evaluates to " << val2 << ", and\n";
	  else {
		  if( val1 == val2 ) return testing::AssertionSuccess();
		  return testing::AssertionFailure()
			  << "The difference between " << expr1 << " and " << expr2
			  << "exceeds " << abs_error_expr << ", where\n"
			  << expr1 << " evaluates to " << val1 << ",\n"
			  << expr2 << " evaluates to " << val2 << ", and\n"
			  << abs_error_expr << " evaluates to " << abs_error << ".";
	  }
	}
}

testing::AssertionResult DoubleSimilarNaN(const char* expr1,
                                     const char* expr2,
                                     const char* abs_error_expr,
                                     double val1,
                                     double val2,
                                     double abs_error) {
  if( std::isfinite(val1) && std::isfinite(val2) ) {
	  const double diff = fabs(val1 - val2);
	  if (diff <= abs_error) return testing::AssertionSuccess();

	  return testing::AssertionFailure()
		  << "The difference between " << expr1 << " and " << expr2
		  << " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
		  << expr1 << " evaluates to " << val1 << ",\n"
		  << expr2 << " evaluates to " << val2 << ", and\n"
		  << abs_error_expr << " evaluates to " << abs_error << ".";
  } else {
	  if( std::isnan(val1) && std::isnan(val2) ) return testing::AssertionSuccess();

	  if( std::isnan(val1) || std::isnan(val2) )
		  return testing::AssertionFailure() << "NaN found "
				  << expr1 << " evaluates to " << val1 << ",\n"
				  << expr2 << " evaluates to " << val2 << ", and\n";
	  else {
		  if( val1 == val2 ) return testing::AssertionSuccess();
		  return testing::AssertionFailure()
			  << "The difference between " << expr1 << " and " << expr2
			  << "exceeds " << abs_error_expr << ", where\n"
			  << expr1 << " evaluates to " << val1 << ",\n"
			  << expr2 << " evaluates to " << val2 << ", and\n"
			  << abs_error_expr << " evaluates to " << abs_error << ".";
	  }
	}
}
