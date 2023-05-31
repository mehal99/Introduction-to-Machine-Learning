import pandas as pd
import sys

class MajorityVoteClassifier():

    def __init__(self, train_input, test_input, train_output, test_output, metrics_output):
        self.train_input =  pd.read_csv(train_input, sep='\t')
        self.test_input = pd.read_csv(test_input, sep='\t')
        self.train_output = train_output
        self.test_output = test_output
        self.metrics_output = metrics_output
        self.majority_label = None
        
        num_columns = self.train_input.shape[1]
        self.train_x = self.train_input.iloc[:,:num_columns-1]
        self.train_y = self.train_input.iloc[:,-1]
        self.test_x = self.test_input.iloc[:,:num_columns-1]
        self.test_y = self.test_input.iloc[:,-1]

    def majority_vote(self):
       return self.train_input.iloc[:,-1].value_counts().idxmax()
    
    def train(self):
        self.majority_label = self.majority_vote()

    def h_x(self, x):
        return self.majority_label

    def predict(self, data):
        predictions = []
        for x in data.iterrows():
            predictions.append(self.h_x(x))
        return predictions
    
    def error_rate(self, data, predictions):
        sum = 0
        total = len(data)
        print(total)
        for y, y_hat in zip(data, predictions):
            if y != y_hat:
                sum +=1
        return sum/total
    
    def run(self):
        self.train()
        # predict the training labels and write them to the output file
        train_predictions = self.predict(self.train_x)
        with open(self.train_output, "w") as file:
            for p in train_predictions:
                file.write(str(p)+"\n")

        #predict th test labels and write them to the output file
        test_predictions = self.predict(self.test_x)
        with open(self.test_output, "w") as file:
            for p in test_predictions:
                file.write(str(p)+"\n")
        
        #calculate training and testing error rates and write then to the metrics output file
        train_error_rate = self.error_rate(self.train_y, train_predictions)
        test_error_rate = self.error_rate(self.test_y, test_predictions)

        with open(self.metrics_output, "w") as file:
            file.write(f'error(train): {train_error_rate:.6f}\n')
            file.write(f'error(test): {test_error_rate:.6f}\n')

if __name__=="__main__":
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_output = sys.argv[3]
    test_output = sys.argv[4]
    metrics_out = sys.argv[5]
    print(train_input)
    classifier = MajorityVoteClassifier(train_input, test_input, train_output, test_output, metrics_out)
    classifier.run()