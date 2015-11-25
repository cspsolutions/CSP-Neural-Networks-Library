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

namespace CSPSolutions.CI.NeuralNetworks.Common
{
	/// <summary>
	/// Summary description for Input.
	/// </summary>
	public class Input
	{
	  #region Variables

		private int target;
		private double[] InputMatrix;	
	
	  #endregion
      
 	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   

	  #region Properties
		/// <summary>
		/// 
		/// </summary>
			public int Target 
			{
				set
				{
					target = value;
				}
				get
				{
					return target;
				}			
			}
		
		#endregion

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
         
      /// <summary>
      /// 
      /// </summary>
      /// <param name="inputMatrix"></param>
   
	  public Input(double[]inputMatrix  )
	  {
		  this.InputMatrix = inputMatrix;
           
		  if (this.InputMatrix != null)
	  	   Target = (int)InputMatrix[InputMatrix.Length - 1 ];         
	  }

	 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//   
      /// <summary>
      /// 
      /// </summary>
      /// <param name="i"></param>
      /// <returns></returns>
 
	  public double Value(int i)
	  {		
		if ( i > (InputMatrix.Length - 1))
			  return 0;

		return InputMatrix[i];
	  }

    }
}
