# nengo-sctn_integration
 for now , since we dont want to break nengo conventions , the summation of the spikes multiplied by the weights would be taken care of by the nengo.connection class and inputed into the neuron as the current "J"  ( though we would carefully need to choose and define its parameters to correspond with the SCTN  algorithm.
 the actuall mathematical computation that would be performed by the " neuron " itself would be the decay ( the synapsic behavior  , therefore synpase=none in connection )  the  change of the membrane voltage ( voltage in nengos syntax )  ,  the rng  module, the spike decision , and perhaps the voltage reset if the atribute is true  )

29.5 : seems like the integration works ( as long as we pass weight by nengo.connection ) , i have managed to almost decode a sin wave with 5 neurons . my next mission is to implement the rest of the activation functions and to finetune my  simulation parameters in order to modulate neuron behaviors .
1.6  i was able to create a pdm encoder using a single neuron and the identity activation function
3.6 : the function gain bias works for the identity and the binary activation functions , though the tuning curves are sparse and not always accurate , need to further check how to make it work on the sigmoid aswell

