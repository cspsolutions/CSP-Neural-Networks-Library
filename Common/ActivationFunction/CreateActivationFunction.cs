using System;
using System.Reflection;
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
	/// Creates Activation Functions.
	/// </summary>
	public class CreateActivationFunction
	{   
		/// <summary>
		/// Default constructor
		/// </summary>
		public CreateActivationFunction()
		{
		}
        
		/// <summary>
		/// Create an activation function object
		/// </summary>
		/// <param name="type"> The type of the activation function </param>
		/// <returns> An instant of an activation function object </returns>
		public static AActivationFunction CreateActivationFnCLass( ActivationType type )
		{
			AActivationFunction _activationFunction = null;

			Assembly assembly = Assembly.GetCallingAssembly(); 
			Type[] types = assembly.GetTypes();
            
			for( int i = 0 ; i < types.Length ; i++ ) 
			{
				if (types[i].IsSubclassOf(typeof(AActivationFunction)) && types[i].Name == type.ToString() )
				{
					_activationFunction = (AActivationFunction)System.Activator.CreateInstance(types[i]);
					break;
				}
			}
           
			if ( _activationFunction == null)
				throw new ArgumentNullException("Activation Function ( Type not found )");

			return _activationFunction;
		}

	}
}
