import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    
    public void setInput(double inputValue) {
        if (type == 0) {    
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {   //Not an input or bias node
            // TODO: add code here
        	inputValue = 0.0;
        	for(NodeWeightPair n:parents) {
        		inputValue=inputValue+n.node.getOutput() * n.weight;
        	}
        	if(type==2) {
        		outputValue = Math.max(0, inputValue);
        	}
        	else  {
        		outputValue = Math.exp(inputValue);
        	}
        }
    }
    
    public void setOutput(double output) {
		this.outputValue = output;
}

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }
    
    public void setDelta(double delta) {
        if (type == 2 || type == 4) {
            this.delta = delta;
        }
    }
    
    public double getDelta() {
		return this.delta;
	}

    //Calculate the delta value of a node.
    public void calculateDelta() {
        if (type == 2 || type == 4)  {
            // TODO: add code here
        	if(type == 2) {
        		if(inputValue > 0) {
        			delta = delta;
        		}
        		else delta =0;
        	}
        	else {
        		for (NodeWeightPair n:parents) {
					n.node.setDelta(n.node.getDelta() + n.weight * delta);
        		}
        	}
        	
        }
    }


    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            // TODO: add code here
        	for(NodeWeightPair n:parents) {
        		n.weight+=learningRate * n.node.getOutput() * delta;
        	}
        }
        this.delta=0;
    }
}


