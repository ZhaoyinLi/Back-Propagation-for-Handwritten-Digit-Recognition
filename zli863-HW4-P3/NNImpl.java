import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        // TODO: add code here
    	
    	 ArrayList<Double>  attr= instance.attributes;
         ArrayList<Integer> calVal = instance.classValues;
         double best = 0;//initial the best score
         int index = 0;
         
         forwardPass(attr);

         for (int i = 0; i < calVal.size(); i++) {
             if (outputNodes.get(i).getOutput() > best) {
                 best = outputNodes.get(i).getOutput();
                 index = i;//find the label index
             }
         }

         return index;
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
    	
        // TODO: add code here
    	
    	 for (int i = 0; i < maxEpoch; i++) {
             Collections.shuffle(trainingSet, random);
             
             for (Instance j:trainingSet) {
            	 List<Double> attr = j.attributes;
                 List<Integer> calVal = j.classValues;

                 forwardPass(attr);
                 backwardPass(calVal);
             }
 
             double loss = trainingSet.stream().mapToDouble(this::loss).average().orElse(Double.NaN);
 			 double totl = 0.0;
 			 for (Instance k:trainingSet) {
 				totl += loss(k);
 			 } 
 			 totl/= (1.0 * trainingSet.size());
             System.out.printf("Epoch: %s, Loss: %.3e\n", i, totl);
         }
    }

    private void forwardPass(List<Double> fP) {
        for (int i = 0; i < fP.size(); i++)
            inputNodes.get(i).setInput(fP.get(i));
        
        for(Node n : hiddenNodes) {  
        	n.calculateOutput();
        }
        
        for(Node n : outputNodes) {  
        	n.calculateOutput();
        }
        
		double sum = 0.0;	
		for (Node n:outputNodes) {
			sum += n.getOutput();
		}
		
		for (Node n:outputNodes) {
			n.setOutput((n.getOutput() * 1.0) / sum);
		}
    }

    private void backwardPass(List<Integer> bP) {
				// add gradient(actually y) to nodes at output layer
        for (int i = 0; i < bP.size(); i++)
            outputNodes.get(i).setDelta(bP.get(i) - outputNodes.get(i).getOutput());
        
        for(Node n : outputNodes) {  
        	n.calculateDelta();
        }
        
        for(Node n : hiddenNodes) {  
        	n.calculateDelta();
        }
        
        for (Node n : outputNodes) {
        	n.updateWeight(learningRate);
        }

        for (Node n : hiddenNodes) {
        	n.updateWeight(learningRate);
        }
        
    }
    
    
    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // TODO: add code here
        ArrayList<Double> attr = instance.attributes;
        ArrayList<Integer> calVal= instance.classValues;

        forwardPass(attr);

				// only need the y w/ value 1, and multiply it w/ the corresponding 
				// predicted val
        double loss = outputNodes.get(calVal.indexOf(1)).getOutput();
        double ans = -Math.log(loss);
        return ans;
 
    }
}
