#include <iostream>
#include <stdlib.h>
#include <math.h>


// testing testing 123



/*
 * 
 * data preperation/normalization functions
 * 
 *
 * */
template<unsigned R, unsigned C>
void normalize(double (&myArray)[R][C])
{

    double maxVar = 0;
    for(int i = 0; i < R; i++){
       for(int j = 0; j < C; j++){

	   if(myArray[i][j] > maxVar) maxVar = myArray[i][j];}
        
       }
    

    for(int i = 0; i < R; i++){
        for(int j = 0; j < C; j++){
		myArray[i][j] = myArray[i][j] / maxVar;}
       
	}
}








template<unsigned R, unsigned C>
void transpose(double (&original)[R][C], double (&transposed)[C][R])
{


     for(int i = 0; i < R; i++){
         for(int j = 0; j < C; j++){
             transposed[j][i] = original[i][j];}
     }


 }    




/*
 * Activation Functions
 *
 *
 *
 *
 * */




//implamentation of reLu activation function (current strengths: fast | current weaknesses: must take int to do bitwize opps, cannot be scaled to wieghts or biases (yet)

constexpr int reLu(int x){
    return (1+((x&(1<<31)) >> 31))*x;
	  }


//implamentation of the sigmoid function

constexpr double sigmoid(double x){

    return 1.0/(1.0 + pow(2.71828, -x));
	  }



/*
 *
 * Backpropagation
 * Functions
 *
 *
 *
 *
 *
 * */


/*
 * General Use
 *
 * */

//implamentation of the prime sigmoid function
constexpr double sigmoidPrime(double x){

    return pow(2.71828, -x)/pow((1.0 + pow(2.71828, -x)), 2);}


//Apply the derivative of the sigmoid activation function element-wise to a matrix
template<int R, int C>
void applySigmoidPrime(double (&Z)[R][C], double (&result)[R][C]){ 
         for(int i = 0; i < R; i++){
             for(int j = 0; j < C; j++){
                 result[i][j] = sigmoidPrime(Z[i][j]);
	       }
	     }
}


    
/*
 * Output layer back to hidden layer
 *
 * */



//find -(y - yHat)
template<int R, int C>
void delyHat(double (&y)[R][C], double (&yHat)[R][C], double (&result)[R][C]){

    for(int i = 0; i < R; i++){
   
	   result[i][0] = -(y[i][0] - yHat[i][0]);

	   //std::cout << "-y -yHat = " << *result[i] << '\n';
    
    }    

}


//find backprop error delta 3
template<int R, int C>
void backpropError3(double (&negYminYhat)[R][C], double (&primeActiveZ3)[R][C], double (&result)[R][C]){
    
    for(int i = 0; i < R; i++){

	//std::cout << negYminYhat[i][0] << " " << *primeActiveZ3[0] << '\n';
        result[i][0] = negYminYhat[i][0] *  primeActiveZ3[i][0];}

	
}

//multiply backprop error by previous layer activations
template<int R, int C, int len2>
void delCdelW2(double (&a2T)[R][C], double (&backPropError3)[C][len2], double (&result)[C][len2]){

    double product;
   // std::cout << '\n';
   // std::cout << '\n';
    for(int i = 0; i < R; i++){
	    product = 0;
	    for(int j = 0; j < C; j++){
	
		product += a2T[i][j] * backPropError3[0][j];
	        //std::cout << a2T[i][j] << " * " << backPropError3[0][j] << " + ";

	    }

      //std::cout << product << '\n';

      result[i][0] = product;
           
    }


}


/*
 * Hidden Layer back to input layer
 *
 * */


// calculate delta 2 (delta3 * W2T * f'(Z2))

template<int R, int C, int len2>
void delta2(double (&d3)[R][len2], double (&W2T)[len2][R], double (&sigPrimeZ2)[R][C], double (&result) [R][len2]){

  
    for(int i = 0; i < R; i++){
        for(int j = 0; j < len2; j++){   
            //std::cout << " " << d3[i][j] << " * " << W2T[j][i] << " "  << '\n';
	    result[i][j] = d3[i][j] * W2T[j][i];
	}
    }


    double product;
    for(int i = 0; i < R; i++){
	product = 0;
        for(int j = 0; j < C; j++){
	   // std::cout << " " << sigPrimeZ2[i][j] << " *  " << result[i][0] << " ";
	    product += sigPrimeZ2[i][j] * result[i][0];
           }

	result[i][0] = product;
        //std::cout << '\n';
	
    }


}



template<int M, int N, int P>
void matrixMultiplication(double (&matrixOne)[M][N], double (&matrixTwo)[N][P], double (&result)[M][P]){

    double product;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < P; j++){
	    product = 0;
	    for(int v = 0; v < N; v++){
           
            product += matrixOne[i][v] * matrixTwo[v][j];
	        }       

        result[i][j] = product;}
    
    }

 }



template<int R, int C>
void linearTransform(double (&matrix)[R][C], double (&vector)[C][1], double (&result)[R][1]){


	for(int i = 0; i < R; i++){
            double product = 0;
            for(int j = 0; j < C; j++){
                //std::cout << matrix[i][j] << " * " << vector[i][0];
                product += matrix[i][j] * vector[i][0];
            
              // std::cout << product << " ";
	    }
                           //std::cout << product << '\n';
	                   result[i][0] = product;}
}






/*
 * Gradient Decent
 *
 *
 * main function which takes wieghts as arguments
 *
 * */








template<int w1R, int w1C, int w2R, int w2C>

int NeuralNet(double (&wieghts)[w1R][w1C], double (&wieghts2)[w2R][w2C], double (&returnWieghts)[w1R][w1C], double (&returnWieghts2)[w2R][w2C]){

	
	
        //inputs being trained	

	double X[3][2] = {
		            {3, 5},
		             {5,1},
		             {10,2}
       				     };
	


       // Labels which are being checked
        // against outputs
    
         double y[3][1] = {
		           {75}, 
		           {82},
			   {93}
	                        };

        
	

	//Data prep, normalization of inputs and labels
	
	normalize(X);
	normalize(y);






	//Hidden Layer
	
	/*Hyperparameters:
	 *
	 * Input size = 2
	 *
	 *
	 * Hidden Layer size = 3
	 *
	 *
	 * Output size = 1
	 *
	 *
	 * */


        
	/*
	 * Layer 1 synapses, connect the input layer and first hidden layer
	 *
	 * */




	/*
	 * Nuerons in hidden layer 1
	 *
	 * */

	
	//not sure why only one bias per hidden layer nueron yet but that is what 3X1 references, 
	// maybe to do with activation living in the node?
	double biases[3][1] = {
		               {0.7},
	                       {1.1},
			       {2.2}
			            };
        
        //There are only 3 nuerons on the hidden layer each of which can process 3 wieghted sums
	//2 input neuron values will be multiplied by 3 synapse wieghts apiece (3 wieghted sums)
	//and each will be added together into a hidden layer along with that hidden layer's bias
	//the three hidden layers will therefore each have three wieghted sums 	
	double Z2[3][3];


        //Again, three activations per neuron in the hidden layer
	double activation2[3][3];

	//End Hidden Layer
	
	




	std::cout << '\n' << '\n'  << "End Hidden Layer Parameterization" << '\n';
	std::cout << '\n' << "Begin Output Layer Parameterization" << '\n';





	/*
	 *
	 * Output layer
	 *
	 * */
	
	


	/*
	 *  synapses connecting hidden layer to output layer
	 *
	 *
	 * */





        /*
	 * Output layer node
	 *
	 * */


        //one node processing three wieghted summs should be 1 x 3 ... not 3 x 1 ... no?
	//Ultimately its most important that it matches the shape of y
        double Z3[3][1];

        
	double yHat[3][1];
	//end Output layer
	

	
	std::cout << '\n'  << "End Output Layer Parameterization" << '\n';
        std::cout << '\n' << '\n' << '\n' << '\n' << "BEGIN FORWARD PROPOGATION" << '\n' << '\n' << '\n' << '\n';


	
	/*
	 * Forward Propagation
	 *
	 *
	 * forward pass layer 1
	 *
	 * */

	std::cout << '\n' << '\n' << "Forward Pass Layer 1" << '\n' << '\n' << '\n' << '\n';


	for(int i = 0; i < 3; i++){
           std::cout << '\n';
           std::cout << "Hidden Nueron " << i+1 << '\n';
	  

	   for(int j = 0; j < 3; j++){

	       double product = 0;
	       for(int v = 0; v < 2; v++){
	       
	        //(1) Z2 = X*W1  
                product += X[i][v] * wieghts[v][j] + biases[i][0];
	           
	  
           
	   
	   }
         
	   Z2[i][j] = product;
	   std::cout << product << "    ";	
	   }
        
	   std::cout << '\n';
	}



	//print z2
	std::cout << '\n';
	std::cout << '\n';
	std::cout << "Z2" << '\n';
	for(int i = 0; i < 3; i++){

	    for(int j = 0; j < 3; j++){

	    std::cout << Z2[i][j] << " ";

	    //(2) a2 = f(Z2)
	    activation2[i][j] = sigmoid(Z2[i][j]);}
	       
	    std::cout << '\n';
	
	}




        //print a2

	std::cout << '\n';
        std::cout << '\n';
        std::cout << "a2" << '\n';	
        for(int i = 0; i < 3; i++){

            for(int j = 0; j < 3; j++){

            std::cout << activation2[i][j] << " ";}

            std::cout << '\n';}



	std::cout << '\n';
        std::cout << '\n';
        std::cout << '\n';
        std::cout << "Output Nueron " << '\n' << '\n';
        std::cout << '\n';


	/*
	 *  Layer 2 forward Propagation
	 *
	 *
	 *
	 * forward pass to the output layer
	 *
	 **/

	
	std::cout << '\n' << '\n' << '\n' << '\n' << "Forward Pass Layer 2 (output layer)" << '\n' << '\n' << '\n' << '\n'; 

	
	for(int i = 0; i < 3; i++){
            
            double product = 0;
	    for(int j = 0; j < 3; j++){
	   
	    //(3) Z3 = a2*W2
	    product += activation2[i][j] * wieghts2[0][j];
	   

	 
	    }

	    Z3[i][0] = product;
    	 }
	
        
        //print z3
        std::cout << '\n';
        std::cout << "Z3" << '\n';	
       	for(int i = 0; i < 3; i++){
	    
	   //(4) Yhat = a3 = f(Z3)
           yHat[i][0] = sigmoid(Z3[i][0]);
	   std::cout <<  Z3[i][0] << " ";
	}

       

        //print yHat/outputs
	std::cout << '\n';
        std::cout << '\n';
	std::cout << '\n';
	std::cout << "a3/yHat/Outputs" << '\n';
        for(int i = 0; i < 3; i++){

            std::cout <<  yHat[i][0] << " ";}
         



	/*
	 * End forward propagation
	 *
	 *
	 *
	 * */

	
        std::cout << '\n'; 
	std::cout << '\n';
	std::cout << '\n';
	std::cout << "END OF FORWARD PROPOGATION" << '\n' << '\n';
        std::cout << '\n';
	std::cout << '\n';


	//cost function calculation? nope, there is no need to calculate the actual cost function as we will
	//use backpropagation and gradient decent to minimize it. The below print statements let us know where
	//we are generally in the process. (it is important to know however that the cost fuction is 1/2(y-yHat)^2
	
	for(int i = 0; i < 3; i++){
	    std::cout << "Machine prediction number " << i+1 << " is... " << yHat[i][0] << '\n';
	    std::cout << "Correct labeled answer number " << i+1 << " is " << y[i][0] << '\n';
	    std::cout << '\n';
	}
	
      
       /*
	* Backpropigation
	*
	*
	*
	* */


	
	std::cout << '\n';
        std::cout << '\n';
        std::cout << '\n';
        std::cout << "BEGINNING BACKPROPOGATION" << '\n' << '\n';
        std::cout << '\n';
        std::cout << '\n';


	
	/*
	 *
	 * Output Layer Backprop
	 * 
	 * del C / del W2
	 *
	 * change in cost relative to output layer wieghts
	 *
	 * */



   
       std::cout << '\n' << '\n';
       std::cout << "Output Layer Backpropigation" << '\n';
       std::cout << '\n' << '\n';
       

       

       //declares (-y-yHat) which
       //is the partial of the Cost function
       //with respect to outputs yHat
       //After being fed into function
       //this should hold del C / del yHat
       double negYminYhat[3][1];



       //del C / del W2 =  del C/del yHat * del yHat/del W2; 
       //Below function calculates del C/del yHat
       //or the Change in cost with respect to machine outputs
       delyHat(y, yHat, negYminYhat);

       /*for(int i = 0; i < 3; i++){
	       
	       std::cout << "-(y - yHat) = " << " ";
	       std::cout << negYminYhat[i][0] << '\n';}*/


       //del yHat/ del W2 = del yHat/del Z3 * del Z3/del w2
       //The change in wieghted sums in relation to the wieghts is the change in activation
       //times the size of wieghted activation that change affected
       double primeActivez3[3][1]; 
       for(int i = 0; i < 3; i++){
          primeActivez3[i][0] = sigmoidPrime(Z3[i][0]);}

       /*
       for(int i = 0; i < 3; i++){

	  std::cout << '\n';
	  std::cout << '\n';

	  std::cout << "f'(Z3)" << " The change in the activation function times Z3" << '\n';

	  std::cout << primeActivez3[i][0] << " ";
          std::cout << '\n';
	  std::cout << '\n';}
	  */

      
       double outputLayerBackpropError[3][1];
      
     



       //total backpropagation error of the past partials into del3 (del C/del z3)
       backpropError3(negYminYhat, primeActivez3, outputLayerBackpropError);


       //std::cout << '\n';
      

    /*   //del z2/ del w2;
       for(int i = 0; i < 3; i++){
	  std::cout << '\n';
	  std::cout << "Total backpropigation error from the output to hidden layer delta3 is a vector with each spot holding (-y-yHat) * f'(z3) " << '\n';
          std::cout << outputLayerBackpropError[i][0] << '\n';}  */ 

        

       //transposing a2

        double a2T[3][3];

	transpose(activation2, a2T);




	//initializing output layer result
	
        double outputLayerBp[3][1];



        //multiplying output bpError times activations of hidden layer
	//movig back into hidden layer
        delCdelW2(a2T, outputLayerBackpropError, outputLayerBp);

	

/*


        //printing output layer results
	for(int i = 0; i < 3; i++){
		std::cout << '\n';
		std::cout << "Output Layer Backprop" << " " << i+1 << '\n';
		std::cout << outputLayerBp[i][0] << '\n';}

		*/




	/*
	 * End Output layer to hidden layer backpropagation
	 *
	 *
	 * Begin Hidden layer to input layer backpropagation
	 *
	 *
	 *
	 * */





	std::cout << '\n';
	std::cout << '\n';
        std::cout << "END Output layer to Hidden Layer Backpropagaton"  <<'\n';
        std::cout << "BEGIN Hidden Layer to Input Layer Backpropagation" << '\n';
        std::cout << '\n';
        std::cout << '\n';



    
	double primeActivez2[3][3];
	
       applySigmoidPrime(Z2, primeActivez2);


       //f'(Z2) change in activation times Z2
       for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
               
                primeActivez2[i][j] = sigmoidPrime(Z2[i][j]);}

	}



	//transpose w2
	double W2Transpose[1][3];
	transpose(wieghts2, W2Transpose);

	/*
	for(int i = 0; i < 1; i++){
	    for(int j = 0; j < 3; j++){
	        std::cout << W2Transpose[i][j] << " ";
	    }
	    
	std::cout << '\n';   
	}*/
	



       
        //delta 3 dot wieghts 2 transpose	
	
	double backpropErrorDelta2[3][1];

	delta2(outputLayerBp, W2Transpose, primeActivez2, backpropErrorDelta2);


	
	for(int i = 0; i < 3; i++){
	   std::cout << backpropErrorDelta2[i][0] << '\n';}
	


	//transpose X
	double XT[2][3];
	transpose(X, XT);

        //done
	double finalBackpropLayerOne [2][1];
	linearTransform(XT, backpropErrorDelta2, finalBackpropLayerOne);

         for(int i = 0; i < 2; i++){
	     std::cout << finalBackpropLayerOne[i][0] << '\n';}
       



	/*
	 *
	 * Gradient decent
	 *
	 *
	 *
	 *
	 *
	 *
	 *
	 * */



        std::cout << '\n';
        std::cout << '\n';
        std::cout << '\n';
        std::cout << "Gradient decent" << '\n';
        std::cout << '\n';
        std::cout << '\n';



	//output layer (typically would include a learning rate which would scale the backpropigation error but in this case it's 1)
	
	for(int i = 0; i < 3; i++){
	    
	  //std::cout << wieghts2[i][0] << '\n';
	  returnWieghts2[i][0] = wieghts2[i][0] - 12*outputLayerBp[i][0];
	    }


	std::cout << "New Wieghts 2" << '\n';

	//new Wieghts after decent
      	for(int i = 0; i < 3; i++){
	      for(int j = 0; j < 1; j++){	
	          std::cout << returnWieghts2[i][0] << " ";}
	          std::cout << '\n';}

        
	std::cout << '\n';
        std::cout << '\n';


	//hidden to input layer
	
	for(int i = 0; i < 2; i++){
	    for(int j = 0; j < 3; j++){
	        //std::cout << wieghts[i][j] << " ";
	        returnWieghts[i][j] = wieghts[i][j] - 12*finalBackpropLayerOne[i][0];}
	   //std::cout << '\n';
	}

	
	std::cout << "New Wieghts" << '\n';


	//new Wieghts after decent
	
	for(int i = 0; i < 2; i++){
	    for(int j = 0; j < 3; j++){
	        std::cout << returnWieghts[i][j] << " ";}
	    
	    std::cout << '\n';
	}


  
	return 0;
}






int main(){

  
    //first layer wieghts are 2x3 because they are connecting the inputs to the hidden layer.
    // 3 connections (wieghts) leaving each input neuron	
    double wieghts[2][3] = {
                                 {1.2, 0.3, 2.4},
                                 {0.5, 1.1, 0.1},

                                        };    

   
     //3 hidden layer nodes connect to 1 output node, therefore there are 3x1 synapses/wieghts
        double wieghts2[3][1]= {
                                {1.3},
                                {0.2},
                                {1.4}
                                      };


	


    //update wieghts
    double updateWieghts[2][3];

    //update wieghts2
    double updateWieghts2[3][1];



    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 1 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 2 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);
   
    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 3 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 4 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 5 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 6 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 7 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 8 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 9 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 10 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 11 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 12 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 13 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 14 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 15 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 16 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 17 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 18 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 19 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 20 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 21 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 22 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(updateWieghts, updateWieghts2, wieghts, wieghts2);

    std::cout << '\n' << '\n' << '\n' << '\n' << "####PASS 23 THROUGH MODEL####" << '\n' << '\n' << '\n' << '\n';

    NeuralNet(wieghts, wieghts2, updateWieghts, updateWieghts2);

    




    return 0; 
}
