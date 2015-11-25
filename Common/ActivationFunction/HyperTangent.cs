using System;
//------------------------------------------------------------------------------
// Copyright (c) 2006, Abla Corporation, All Rights Reserved.
//
// This software is the confidential and proprietary information of Abla 
// Corporation.
// You shall use it only in accordance with the terms of the license agreement you 
// entered into with Abla.
//
// The software is provided "AS IS" WITHOUT WARRANTY OF ANY KIND EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTY OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK ARISING OUT OF THE USE OR
// PERFORMANCE OF THIS PROGRAM AND DOCUMENTATION REMAINS WITH YOU. IN NO EVENT
// WILL Abla BE LIABLE FOR ANY LOST PROFITS, LOST SAVINGS, INCIDENTAL
// OR INDIRECT DAMAGES OR OTHER ECONOMIC CONSEQUENTIAL DAMAGES, EVEN IF
// Abla OR ITS AUTHORIZED SUPPLER HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES. IN  ADDITION Abla AND ITS SUPPLIERS WILL NOT BE LIABLE FOR
// ANY DAMAGES CLAIMED BY YOU BASED ON ANY THIRD PARTY CLAIM.
//------------------------------------------------------------------------------

namespace CSPSolutions.CI.NeuralNetworks.Common.ActivationFunction
{
	/// <summary>
	/// Summary description for HyperTangent Activation Function.
	/// </summary>
	public class HyperTangent : AActivationFunction
	{
		/// <summary>
		/// Default constructor
		/// </summary>
		public HyperTangent():base()
		{
            this._type = ActivationType.HyperTangent; 
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		/// Calculates the value using this activation Activation Function
		/// </summary>
		/// <param name="net"> The net of the neurons</param>
		/// <returns> The value of the activation function (value between -1 and 1)</returns>     
		override public double ActivationFunction( double net )
		{
			return ( this._value = ((Math.Exp(net) - Math.Exp(-net)) / (Math.Exp(net) + Math.Exp(-net))) );
		}

	}
}
