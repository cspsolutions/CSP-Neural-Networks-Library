using System;
using System.Collections;
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

namespace CSPSolutions.CI.NeuralNetworks.Common
{
	/// <summary>
	/// Node presented in layer
	/// </summary>
	[Serializable()]
	public class Node 
	{
		
		#region Variables

		int _num; //The _number of neuron in Layer
		double[] _weights;
		double _bias , _sigmoud , _output;
	    int _target;

		#endregion

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  
		
		#region Properties
		
		/// <summary>
		/// Set or get the _bias
		/// </summary>
		public double Bias
		{
			set
			{
				_bias += value;
			}	
			get
			{
				return _bias;
			}
		} 

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		///  Set or get the _output value
		/// </summary>
		public double Output
		{
			set
			{
				_output = value;
			}	
			get
			{
				return _output;
			}			
		} 

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		///  Get or set the _target
		/// </summary>
		public int Target
		{
			set
			{
				_target = value;
			}	
			get
			{
				return _target;
			}			
		} 

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		/// Get or set the _sigmoud
		/// </summary>
		public double Sigmoud
		{
			set
			{
				_sigmoud = value;
			}	
			get
			{
				return _sigmoud;
			}			
		} 

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		/// Accesses the _weights matrix
		/// </summary>
		/// <param name="i"> index </param>
		/// <returns></returns>
		public double Weight(int i)
		{				
			return _weights[i];
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		/// Indexer declaration accesses the _weights matrix
		/// </summary>
		public double this [int index]    
		{
			get 
			{
				return _weights[ index ];
			}
			set 
			{				
				_weights[ index ] = value;
			}
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  
        /// <summary>
        /// Get the node number
        /// </summary>
		public int Number
		{
			get
			{
				return this._num;
			}
		} 

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

        /// <summary>
        /// Gets the weights matrix
        /// </summary>
		public double [] Weights 
		{
			get
			{
			 return this._weights;
			}
		}

		#endregion  

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		///  Node constructor 
		/// </summary>
		/// <param name="num"> number of the Node</param>
		/// <param name="weights"> weights intializes to the node</param>
		public Node(int num ,double [] weights )
		{   
			this._num = num; 
			this._weights = weights; 
			this._target = 0; //by default!
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  
				
		/// <summary>
		/// Update weights of the node with the given parameters
		/// </summary>
		/// <param name="learningRate"> learning Rate</param>
		/// <param name="previousLayer"> previous Layer</param>
		public void UpdateWeights( double learningRate ,ArrayList previousLayer)
		{
			for(int i = 0 ; i < this._weights.Length ; i++)
				this._weights[i] += DeltaW(learningRate ,((Node)previousLayer[i])._output);

			this._bias = DeltaW(learningRate, 1); 
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  
        /// <summary>
        ///  Intialize weights to value range 
        /// </summary>
        /// <param name="value"> to initailize with </param>
		public void IntializeWeights(double value)
		{               
			for(int i = 0 ; i < this._weights.Length ; i++)
				this._weights[i] = value;

			this._bias = value;
		}	

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

		/// <summary>
		/// Is the shift of change, to be a add to the _weights
		/// </summary>
		/// <param name="learningRate"> Learning Rate </param>
		/// <param name="output"> output </param>
		/// <returns></returns>
		private double DeltaW(double learningRate ,double output)
		{			 
			return learningRate * this._sigmoud * output;  			
		}

	}
}
