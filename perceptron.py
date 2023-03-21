
class perceptron:

    def __init__(self, init_weights, init_threshold, l_r):
        self.weights = init_weights
        self.threshold = init_threshold
        self.learning_rate = l_r
        self.error = 0

        self.number_of_training_steps = 0

        self.weights_history =[[w for w in self.weights]]
        self.weights_history[0].append(self.threshold)

        self.initialized_with_learning_rate = False

    # activation function
    # for Heaviside there has to be a >=
    def activation_function(self, scalarProd, threshold):
        if scalarProd >= threshold:
            return 1
        else:
            return 0

    def identity_function(self, scalarProd):
        return scalarProd

    # trains perceptron for a single input/point and corresponding target/label (dataset [x,y,t])
    # ONLY used for classification
    # For regression we train only a complete set, because we update with the complete error of the dataset,
    # not only of one data point
    def train_single_dataset(self, dataset):

        # calculate output
        # converts inputs list to 2d array (inputs list is one input vector)
        inputs = dataset[0:-1]
        target = dataset[-1]

        calculated_output = self.calc_output_perceptron_classification(inputs)

        # calculate error delta
        delta = target- calculated_output

        # update weights (if error zero, no effect happens)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + delta * self.learning_rate * inputs[i]

        # updates bias (if error zero, no effect happens)
        self.threshold = self.threshold - delta * self.learning_rate * 1

        self.number_of_training_steps += 1
        #Hinzufügen der akutellen Gewichte zum Verlauf der bisherigen Gewichte, wenn diese geändert wurden
        if delta != 0:
          self.weights_history.append([self.weights[0],self.weights[1], self.threshold])


    # calculates the output of the perceptron for a given input vector as list(without target!)
    def calc_output_perceptron_classification(self, inputs_list):
        # calculates scalar product
        output = 0
        for i in range(len(self.weights)):
            output += (self.weights[i] * inputs_list[i])

        final_output = self.activation_function(output, self.threshold)
        return final_output

    # calculates the output of the perceptron for a given input vector as list (without target!)
    # Hier ist der Threshold eigentlich ein Bias!
    def calc_output_perceptron_regression(self, inputs_list):
        output = 0
        for i in range(len(self.weights)):
            output += (self.weights[i] * inputs_list[i])
        output += self.threshold
        return output

    # Trains the perceptron one time with the complete training data
    def train_all_classification(self, read_in_training_data):
        # Trains the Perceptron with the read in training data
        for dataset in read_in_training_data:
            self.train_single_dataset(dataset)

    #Hier ist der Threshold eigentlich ein Bias
    def train_linear_regession(self, read_in_training_data, epochs):
        for x in range(epochs):
            error_w_x = 0
            error_w_0 = 0

            # calculate loss for updating weights
            for dataset in read_in_training_data:
                error_w_0 += dataset[1] - (self.weights[0] * dataset[0] + self.threshold)
                error_w_x += (dataset[1] - (self.weights[0] * dataset[0] + self.threshold)) * dataset[0]

            # update weights
            self.weights[0] = (self.weights[0] + error_w_x * self.learning_rate)
            self.threshold = (self.threshold + error_w_0 * self.learning_rate)

            self.number_of_training_steps += 1
            if error_w_0 != 0 and error_w_x != 0:
                self.weights_history.append([self.weights[0], self.threshold])


    def number_of_correct_classified(self, read_in_training_data):
        number_of_correct = 0
        for dataset in read_in_training_data:
            if self.calc_output_perceptron_classification(dataset[0:len(dataset) - 1]) == dataset[2]:
                number_of_correct += 1

        return number_of_correct
