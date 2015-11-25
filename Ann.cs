using System;
using System.Collections;
using CSPSolutions.CI.NeuralNetworks.Common;
using CSPSolutions.CI.NeuralNetworks.Common.ActivationFunction;

//------------------------------------------------------------------------------
// Copyright (c) 2006, CSP Solutions Corporation, All Rights Reserved.
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

namespace CSPSolutions.CI.NeuralNetworks       //Neural Network namespace
{
	/// <summary>
	/// Ann class is an implementation of the multi-layered artificial neural network
	/// </summary>
	public class ANN
	{   
		#region ANN Declarations 
        /// <summary>
        /// Variables
        /// </summary>
		protected int _class_id , _epochs , _errorFn ;
		/// <summary>
		/// Variables
		/// </summary>
		protected double _learningRate , _momentum , _tolerance , _minimum , _maximum , _threshold , _previousError;
		/// <summary>
		/// Layers
		/// </summary>
		protected ArrayList _inputLayer = null, _hiddenLayers = null, _outputLayer = null , _inputVector = null , _graphPoints = null;
        /// <summary>
        /// The activation function used by this network
        /// </summary>
		protected AActivationFunction _activationFunction = null;
		/// <summary>
		/// Activation function type 
		/// </summary>
		protected ActivationType _activationFnType;

		#endregion

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		#region Properties

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Set or get threshold
		/// </summary>
		public double Threshold 
		{
			get
			{
				return _threshold; 
			}
			set
			{
				_threshold = value;
			}
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		
		/// <summary>
		/// Sets learning rate for the network
		/// </summary>
		public double LearningRate
		{
			set
			{
				_learningRate = value;   
			}

			get
			{
				return _learningRate ;
			}	
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Sets momentum for the network
		/// </summary>
		public double Momentum
		{
			set
			{
				_momentum = value;   
			}

			get
			{
				return _momentum ;
			}	
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Sets tolerance for the network
		/// </summary>
		public double Tolerance
		{
			set
			{
				_tolerance = value;   
			}

			get
			{
				return _tolerance;
			}	
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

        /// <summary>
        /// Get or set the activation function
        /// </summary>
		public AActivationFunction ActivationFunction
		{
			set
			{
				if ( value == null )
					throw new ArgumentNullException("Activation function set to null");

				this._activationFunction = value;
			}
			get
			{
				if ( this._activationFunction == null)
                   this._activationFunction = CreateActivationFunction.CreateActivationFnCLass( this._activationFnType );

			    return this._activationFunction; 
			}
		}
		#endregion

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		
		/// <summary>
		/// Default Constructor of ANN
		/// </summary>
		public ANN()
		{
			
			this._class_id = 0;
			this._previousError = _threshold = 0.1;//double.Epsilon; 
			this._hiddenLayers = new ArrayList();
			this._inputVector = new ArrayList();
			this._graphPoints = new ArrayList();
			this._minimum = -0.5;
			this._maximum = 0.5;
			this._learningRate = 0.9;
			this._momentum = 1;
			this._errorFn = 2;
			this._activationFnType = ActivationType.Sigmoud;
			this._activationFunction = CreateActivationFunction.CreateActivationFnCLass( this._activationFnType );
			
		}


		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
        
		/// /// <summary>
		/// Constructor of ANN
		/// </summary>
		/// <param name="classId">Identification of a class label</param>
		public ANN(int classId)
		{
			this._class_id = classId;
			this._previousError = _threshold = 0.1;//double.Epsilon; 
			this._hiddenLayers = new ArrayList();
			this._inputVector = new ArrayList();
			this._graphPoints = new ArrayList();
			this._minimum = -0.5;
			this._maximum = 0.5;
			this._learningRate = 0.9;
			this._momentum = 1;
			this._errorFn = 2;
			this._activationFnType = ActivationType.Sigmoud;
			this._activationFunction = CreateActivationFunction.CreateActivationFnCLass( this._activationFnType );
			
		}


		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Setup a network with the specified parameters
		/// </summary>
		/// <param name="inputs"> Number of input nodes</param>
		/// <param name="hiddenLayers"> Number of hidden layers</param>
		/// <param name="outputs"> Number of output nodes</param>
		public void SetupNetwork(int inputs , int[] hiddenLayers , int outputs)
		{	
			this._inputLayer = CreateLayer(inputs ,0);
			if( hiddenLayers != null )
			{
				this.CreateHiddenLayers( hiddenLayers , inputs );
				this._outputLayer = CreateLayer(outputs , LastHidden(hiddenLayers) );
			}
			else
			{
				this._outputLayer = CreateLayer( outputs , inputs);
			}
		}


		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Setup a network with the specified parameters
		/// </summary>
		/// <param name="inputs"> Number of input nodes</param>
		/// <param name="hiddenLayers"> An array containing number of hidden nodes for each hidden layer,
		/// the length of the array specifies the number of hidden layers</param>
		/// <param name="outputs"> Number of output nodes</param>
		/// <param name="learningRate"> Learning Rate</param>
		/// <param name="momentum"> Momentum</param>
		/// <param name="tolerance"> Tolerance</param>
		
		public void SetupNetwork( int inputs , int[] hiddenLayers , int outputs , double learningRate , double momentum , double tolerance)
		{		
			this._tolerance = tolerance;
            this._learningRate = learningRate;
            this._momentum = momentum;
                   
			this._inputLayer = CreateLayer( inputs , 0 );
		
			if( hiddenLayers != null )
			{
				this.CreateHiddenLayers( hiddenLayers , inputs );
				this._outputLayer = CreateLayer( outputs , LastHidden( hiddenLayers ) );
			}
			else{
				this._outputLayer = CreateLayer( outputs , inputs );
			}
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Sets the range for which the weights will be intialized to
		/// </summary>
		/// <param name="min"> Beginning of interval</param>
		/// <param name="max"> End of interval</param>
		
		public void SetWeightsRange( double min, double max )
		{		
			this._minimum = min; 
			this._maximum = max;
		}


		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 

		/// <summary>
		/// Range Value
		/// </summary>
		/// <returns></returns>
		private double RangeValue( int seed )
		{  
			Random rV = new Random( seed );
			double director = this._minimum; 
   
			if( rV.Next(-3, 3) > 0 )  
				director = this._maximum;
  
			return  director * rV.NextDouble();
		}
		

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// Intialize Weights for the nodes in a specified range (By default -1 to +1)
		/// </summary>
		public void IntializeWeights()
		{
			IntializeHiddenLayersWeights(); 
			IntializeLayerWeights( this._outputLayer );
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// Intialize hidden Layers Weights
		/// </summary>
		private void IntializeHiddenLayersWeights()
		{
			foreach(ArrayList HiddenLayer in this._hiddenLayers)
			{
			   IntializeLayerWeights(HiddenLayer);			 		    
			} 
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
        /// <summary>
        /// Intialize layer Weights
        /// </summary>
        /// <param name="layer"></param>
		private void IntializeLayerWeights(ArrayList layer)
		{	
			int seed = 0; 					

			foreach( Node unit in layer )
			{
				unit.IntializeWeights( RangeValue(seed) );
			  	seed++;	    
			}
		}

	    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
	    /// 
	    /// </summary>
	    /// <param name="nodes"></param>
	    /// <param name="previousNodes"></param>
	    /// <returns></returns>
		private ArrayList CreateLayer(int nodes , int previousNodes )
		{
			ArrayList layer = new ArrayList();

			for( int i = 0 ; i < nodes ; i++ )
				layer.Add( new Node( i , new double[previousNodes]) );
 			
			return layer;
		}
      
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
	    /// <summary>
	    /// Creates hidden layers 
	    /// </summary>
	    /// <param name="hiddenLayers">Array that contains number of hidden nodes for each hidden layer</param>
	    /// <param name="previousNodes"></param>
 
		private void CreateHiddenLayers(int[] hiddenLayers , int previousNodes )
		{   			 
			for(int i = 0 ; i < hiddenLayers.Length  ; previousNodes = hiddenLayers[i], i++)			
				this._hiddenLayers.Add( CreateLayer( hiddenLayers[i] , previousNodes) ); 									
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Last hidden layer nodes
		/// </summary>
		/// <param name="hiddenLayers"> Contain architecture of hidden layer(s)</param>
		/// <returns>Last hidden layer nodes</returns>
		private int LastHidden(int [] hiddenLayers)
		{				
			return hiddenLayers[ hiddenLayers.Length - 1 ];
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//      

		/// <summary>
		/// Starts Training the network on the given data
		/// </summary>
		public void TrainNetwork()
		{
			 double tError = 0;
			_epochs = 0;  
            _graphPoints.Clear();

			IntializeWeights();                         //1 

			do
			{	
				_epochs++;

				foreach(Input node in this._inputVector)
				{
					SetupTarget(node.Target);           //2
					FeedToNetwork(node);                //3 
					tError += Error();					//4	
					UpdateNetwork();					//5	
				}
				   
			}while( NetworkError(tError) > this._threshold );		//6 stopping condition
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//    
		
		/// <summary>
		/// Starts Testing the network on the given data
		/// </summary>
  
		public double[] TestNetwork()
		{   
			double[] Results = new double[_inputVector.Count];
            int i = 0;
			
			foreach(Input node in this._inputVector)
            {
				FeedToNetwork(node);
                Results[i] = PredictedOuput();
				i++;
			}

			return Results;
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		private int PredictedOuput(){
		  int PredictedNeuron = ((Node)_outputLayer[0]).Number;
          double Biggest = ((Node)_outputLayer[0]).Output;
                     
			foreach(Node Unit in this._outputLayer)
			  {
				if (Unit.Output > Biggest) {
				  Biggest = Unit.Output;
				  PredictedNeuron = Unit.Number; 
				}			
			}
		     
			return PredictedNeuron ;//PredictedNeuron;
		}


		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
        /// <summary>
        /// Feed To Network
        /// </summary>
        /// <param name="node"></param>
		protected void FeedToNetwork(Input node){
		
          FeedToInputLayer(node);   
		  ForwardPropagation();     
   
		}

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// 
		/// </summary>
		/// <param name="node"></param>
		protected void  FeedToInputLayer(Input node)
		{
				foreach(Node Unit in this._inputLayer)
				{
					Unit.Output = node.Value(Unit.Number);
				}
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
	
		/// <summary>
		/// Setup Data for the input layer
		/// </summary>
		/// <param name="InputMatrix"></param>
		public void SetupData(double[][] InputMatrix) 
		{   
			_inputVector.Clear();
  			 
			foreach(double[] InputV in InputMatrix)
			{							
				_inputVector.Add(new Input(InputV));			
			}
		}
       
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		private double Error() 
		{
			double Sigma = 0;
			
			foreach(Node Unit in this._outputLayer)
			{	
				if (this._errorFn == 2)
				 Sigma +=  (Unit.Target - Unit.Output)*(Unit.Target - Unit.Output);
				else if (this._errorFn == 1)
                 Sigma += (Unit.Target * Math.Log(Unit.Output , Math.E) + (1 - Unit.Target) * Math.Log((1 - Unit.Output) , Math.E) ) ;				 
			}
			
			if (this._errorFn == 2)
				Sigma *= 0.5; 

			return  Sigma;	 
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
         /// <summary>
         /// 
         /// </summary>
         /// <param name="Error">Total error for all the cases</param>
         /// <returns></returns>
		private double NetworkError(double Error) 
		{	
			double Nterror = 0;
			
			if(_errorFn == 2)
			 Nterror =   Error/_epochs;	
			else if(_errorFn == 1)
			 Nterror =  -1 * Error/_epochs;
             
			if (double.IsNaN(Nterror))
			 _graphPoints.Add(0); 
			else
			 _graphPoints.Add(Nterror); 

			return Nterror;
		}
       
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Sets Error Function
		/// </summary>
		/// <param name="Type">Two available types "Optimized" or "Squared" or "Average"</param>
		public void SetErrorFunction(string Type){
		 
			if (Type == "Optimized")
			  _errorFn = 1;
		    else if (Type == "Squared")
              _errorFn = 2; 
			else{
			 Exception E = new Exception("SetErrorFunction Two available types 'Optimized' or 'Squared' ");
				throw(E);
			}
		}
	    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  
		/// <summary>
		/// Sets the number of hidden nodes, for a specified hidden layer
		/// </summary>
		/// <param name="HiddenLayer"> A specified hidden layer </param>
		/// <param name="Hidden"> Number of hidden nodes</param>
		public void SetupHiddenLayer(int HiddenLayer , int Hidden) 
		{
 
					  
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Starts calculating neuron outputs, from 1st hidden layer to the output layer 
		/// </summary>
		private void ForwardPropagation(){
		                   
			ArrayList PreviousLayer = _inputLayer;
			
			foreach(ArrayList HiddenLayer in this._hiddenLayers)
			{			
			   CalculateOutputs(HiddenLayer , PreviousLayer);				    
			   PreviousLayer = HiddenLayer;
			}			  				

			CalculateOutputs( _outputLayer , PreviousLayer);				    				
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// 
		/// </summary>
		/// <param name="CurrentLayer"></param>
		/// <param name="PreviousLayer"></param>
		private void CalculateOutputs(ArrayList CurrentLayer , ArrayList PreviousLayer )
		{		
			foreach(Node Unit in CurrentLayer )
			{			
				Unit.Output = this._activationFunction.ActivationFunction( Net(Unit,PreviousLayer) );			
			} 		  
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// 
		/// </summary>
		/// <param name="CurrentUnit"></param>
		/// <param name="PreviousLayer"></param>
		/// <returns></returns>
		private double Net(Node CurrentUnit, ArrayList PreviousLayer )
		{
			double NET = 0;

			foreach(Node Unit in PreviousLayer )
			{
			   NET += Unit.Output * CurrentUnit.Weight(Unit.Number);
			}
										 		  
			return (NET + CurrentUnit.Bias);
		}
				
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
	    /// <summary>
	    /// 
	    /// </summary>
	    /// <param name="Unit"></param>
	    /// <param name="LayerTotheRight"></param>
	    /// <returns></returns>
		private double Sigmoud(Node Unit , ArrayList LayerTotheRight)
		{	   
			if (LayerTotheRight == null) //Neuron in Output Layer
				return (Unit.Output * (Unit.Target - Unit.Output) * ( 1 - Unit.Output));
			else                         //Neuron in Hidden Layer  
				return (Unit.Output * ( 1 - Unit.Output) * SumOF(LayerTotheRight , Unit.Number) );
		
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// 
		/// </summary>
		/// <param name="LayerTotheRight"></param>
		/// <param name="CurrentUnit"></param>
		/// <returns></returns>
		private double SumOF(ArrayList LayerTotheRight , int CurrentUnit) 
		{	
			double Sum = 0;
		
			foreach(Node Unit in LayerTotheRight )
			{			    
				Sum += Unit.Weight(CurrentUnit) * Unit.Sigmoud;
			}		

			return Sum;		
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Update Network, using backward path 
		/// </summary>
		private void UpdateNetwork()
		{	
			int i;
			ArrayList LayerToRight = null;

			if (_hiddenLayers.Count > 0)
			{			  
		     
			  UpdateWeights(_outputLayer ,((ArrayList)_hiddenLayers[_hiddenLayers.Count - 1]), LayerToRight );
			
			  LayerToRight = _outputLayer;
			
				for( i = _hiddenLayers.Count - 1; i > 0 ; i-- )
				{
			
					UpdateWeights(((ArrayList)_hiddenLayers[i]) ,((ArrayList)_hiddenLayers[i-1]), LayerToRight );
			    
					LayerToRight = ((ArrayList)_hiddenLayers[i]);
				}			  				
			
				UpdateWeights(((ArrayList)_hiddenLayers[i]) ,_inputLayer , LayerToRight );			
			}
			else
               UpdateWeights(_outputLayer ,_inputLayer, LayerToRight );
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
        /// <summary>
        /// 
        /// </summary>
        /// <param name="CurrentLayer"></param>
        /// <param name="PreviousLayer"></param>
        /// <param name="LayerToRight"></param>
		private void UpdateWeights(ArrayList CurrentLayer , ArrayList PreviousLayer, ArrayList LayerToRight ){
	
			foreach(Node Unit in CurrentLayer )
			{
			    Unit.Sigmoud  = Sigmoud(Unit , LayerToRight);
				Unit.UpdateWeights(_learningRate , PreviousLayer);
			}
	
	    }
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
      
       /// <summary>
       /// 
       /// </summary>
       /// <param name="TargetUnit"></param>
		private void SetupTarget(int TargetUnit)
		{
			foreach(Node Unit in this._outputLayer )
			{
				if (Unit.Number == TargetUnit)
					Unit.Target = 1;
				else
					Unit.Target = 0;
			}	
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
		/// <summary>
		/// Modifies the network architecture 
		/// </summary>
		/// <param name="Inputs"> Number of input nodes </param>
		/// <param name="_hiddenLayers"> Number of hidden layers</param>
		/// <param name="Outputs"> Number of output nodes </param>
		public void ModifyNetwork(int Inputs , int _hiddenLayers , int Outputs ) 
		{
					 
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
		/// <summary>
		/// Gets the current network properties
		/// </summary>
		/// <returns> A string containing the current network properties</returns>
		public string NetworkProperties()
		{
			string Properties = "";

			return Properties;
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Saves statistics about the training phase to a file
		/// </summary>
		/// <param name="FileName">Path of file to save to</param>
		public void SaveStatisticsToFile(string FileName)
		{
		
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		/// <summary>
		/// Gets statistics about the training phase 
		/// </summary>
		/// <returns> Statistics as a string format</returns>
		public string GetStatistics()
		{
			string Statistics = "";

			return Statistics;
		}
		
	    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public ArrayList GraphPoints()
		{
		   return this._graphPoints; 		
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

	}
}
