using System;
using System.Collections;
using CSPSolutions.CI.NeuralNetworks.Common;

//------------------------------------------------------------------------------
// Copyright (c) 2006, CSP SOLUTIONS Corporation, All Rights Reserved.
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

namespace CSPSolutions.CI.NeuralNetworks
{
	/// <summary>
	/// Summary description for SOM Self Organizing Map Kohonen.
	/// </summary>
	public class SOM : ANN
	{   
		private int maximumEpochs;
		/// <summary>
		/// Dominant weight
		/// </summary>
	       public double Dominant_weight=0;
	
		/// <summary>
		/// Default constructor
		/// </summary>
		public SOM() : base()
		{
			maximumEpochs = 300;
		}
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	
	       /// <summary>
	       /// Overloaded Constructor
	       /// </summary>
	       /// <param name="ClassId"></param>
	   
		public SOM(int ClassId) : base(ClassId)
		{
			maximumEpochs = 300;
		}
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		
		/// <summary>
		/// Starts Training the network on the given data
		/// </summary>
		public new void TrainNetwork()
		{
			
			_epochs = 0;  
	
			IntializeWeights();                         //1 
	
			do
			{	
				_epochs++;
	
				foreach(Input node in this._inputVector)
				{					
					FeedToNetwork(node);                //2
					
					UpdateNetwork();					//3	
				}
				   
			}while( _epochs < MaximumEpochs);		    //4
		}
	
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		
		/// <summary>
		/// Maximum number of Epochs (Stopping Criteria)
		/// </summary>
		public int MaximumEpochs
		{
			set
			{
				maximumEpochs = value;   
			}
	
			get
			{
				return maximumEpochs;
			}	
		}		
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	
		private void CalculateOutputs()
		{		
			foreach(Node Unit in this._outputLayer)
			{			
				Unit.Output = Distance(this._inputLayer , Unit);			
			} 		  
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Update Network, The weights of the node with the 
		/// minimum distance is adjusted
		/// </summary>
		private void UpdateNetwork()
		{				
	           //Node Winner = ((Node)(_outputLayer[ GetWinner() ]));
	          	int W = GetWinner(),Neighbors = 2;
	
			if ( (W - 1) != 0) // to update the neighbors
	             W--;
	
			int n = this._inputLayer.Count;
			double Xi;
	
			for(int i = 0 ; i < n ; i++)
			{			
			   Xi = ((Node)_inputLayer[i]).Output;
	              ((Node)(_outputLayer[W]))[i] += this._learningRate * (Xi -  ((Node)(_outputLayer[W]))[i]);  // Wji = Wji(old) + learningRate*(Xi - Wji(old)) 
			      
				if( (i == (n - 1)) && (Neighbors > 0)  ){
				   
					W++; //go to next neighbor
	
	                  if (W < _outputLayer.Count) // check if valid
				     i = -1;      //set update again but this time for neighbor W
	                     
	                  Neighbors--;   
				}
			}
	
			//Dominant_weight = Winner[0];
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  
	       /// <summary>
	       /// For each input vector presented, 
	       /// the Euclidean distance to the output node is computed
	       /// </summary>
	       /// <param name="Input">input vector</param>
	       /// <param name="Unit">the output</param>
	       /// <returns>the Euclidean distance</returns>
	   
		protected double Distance(ArrayList Input , Node Unit){
		
			double Euclidean_distance = 0,Xi;
	
			for(int i = 0 ; i < Input.Count ; i++)
			{			
				Xi = ((Node)Input[i]).Output; 
				Euclidean_distance += ((Xi - Unit[i]) * (Xi - Unit[i])) ;			
			}
	
	           return Euclidean_distance;
		}
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	       /// <summary>
	       /// Gets the Winner neuron with the closest distance to the input
	       /// </summary>
	       /// <returns></returns>
		protected int GetWinner(){
	        int WinnerNeuron = 0; 
	        double SValue = ((Node)_outputLayer[0]).Output;
	   
			foreach(Node Unit in this._outputLayer)
			{			
				if (SValue > Unit.Output){
					SValue =  Unit.Output;
	                   WinnerNeuron = Unit.Number;
				}
			} 		  
		
		   return WinnerNeuron;
		}
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// Feed To Network
		/// </summary>
		/// <param name="node"></param>
		protected  new void FeedToNetwork(Input node)
		{		
			FeedToInputLayer(node);   
		    CalculateOutputs();	
		}
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// Testing Phase 
		/// </summary>
		/// <returns></returns>
		public new double[] TestNetwork(){
		
			foreach(Input node in this._inputVector )
			{
				FeedToNetwork(node);                //2
			}
	
			return ((Node)_outputLayer[GetWinner()]).Weights;	
		
		} 
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	  
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
	 
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
	 
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	
	
	}


}
