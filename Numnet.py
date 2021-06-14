# THIS IS MY OWN NEURONET!!!!
# All rights reserved / removed
# Jonny's code
import numpy as np
import scipy.special
import tkinter as tk
from tkinter import filedialog as fd

# Setting the correct numbers showing:
np.set_printoptions(precision=7, suppress = True)


"""GRAPHIC INTERFACE"""
class Paint(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.color = "black"

        self.pixels = np.zeros((28, 28))
        self.setUI()

    def get_pixels(self):
        return self.pixels

    def clear(self):
        self.canv.delete("all")
        self.pixels = np.zeros((28, 28))

    def switch_color(self):
        if self.color == "white":
            self.color = "black"
        else:
            self.color = "white"
        self.switch_btn['text'] = "Color: " + self.color

    def draw(self, event):
        if self.color == "black":
            self.pixels[event.y // 10, event.x // 10] = 1
        else:
            self.pixels[event.y // 10, event.x // 10] = 0
        self.canv.create_rectangle(event.x // 10 * 10,
                                   event.y // 10 * 10,
                                   event.x // 10 * 10 + 10,
                                   event.y // 10 * 10 + 10,
                                   fill=self.color, outline=self.color)

    def setUI(self):
        self.parent.title("Paint")
        self.pack(fill=tk.BOTH, expand=1)

        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)

        self.canv = tk.Canvas(self, bg="white", width=280, height=280)
        self.canv.pack()
        
        self.canv.bind("<B1-Motion>", self.draw)
        self.clear_btn = tk.Button(self, text="Clear all", width=10,
                                command=lambda: self.clear())
        self.switch_btn = tk.Button(self, text="Color: black", width=10,
                                 command=lambda: self.switch_color())
        self.dump_btn = tk.Button(self, text="dump pixels", width=10,
                                    command=lambda: print(self.pixels))
        self.clear_btn.pack()
        self.switch_btn.pack()
        self.dump_btn.pack()





"""THE NETWORK"""
class neuralNetwork:
    # Inicialization
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learningrate

        # Matrixes of weight ratio
        # "wih" - between input and hiddan layers
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0, 5), (self.hnodes, self.inodes))
        # "who" - between hidden and output layers
        self.who = np.random.normal(0.0, pow(self.onodes, -0, 5), (self.onodes, self.hnodes))
        # Using the sigmoid as the activation function
        self.activation_function = lambda x: scipy.special.expit(x)



    def train(self, inputs_list, targets_list):
        # Making from the lists two-dimensional arrays 
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # Inputs and outputs for the HIDDEN layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # inputs and ouputs for the OUTPUT layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Error = target value - actual value
        output_errors = targets - final_outputs
        # Output_errors of the hidden layer, distributed proportionally
        # by the link weights and recombined on the hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights (between the hidden and OUTput layers)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # Update the weights (between the INPUT and hidden layers)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs





"""EXECUTOR"""
class Laboratory:
    def __init__(self):
        # Settings:
        input_nodes = 784
        hidden_nodes = 100  # Ideally 200, but it takes more time to train
        output_nodes = 10
        
        learning_rate = 0.2  # For 200 better set 0.1
        # Creating the network
        self.network = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
        self.pixels = np.zeros((28, 28))
        self.root = tk.Tk()
        self.root.geometry("300x480")

        def callback():
            self.train_path = fd.askopenfilename()
            return self.train_path


        tk.Button(self.root, text="Train network", command=lambda: self.train_network(callback())).pack()
        tk.Button(self.root, text="Test network by file", command=lambda: self.test_by_file(callback())).pack()
        tk.Button(self.root, text="Test network by picture", command=lambda: self.test_by_window()).pack()
        self.app = Paint(self.root)
        self.root.mainloop()


    def train_network(self, path_1):
        print("Started training:", "....", sep="\n", flush=True)
        training_data_file = open(path_1, mode="r")
        training_data_list = training_data_file.readlines()
        
        # Here you can change the cut to whatever you want. There are 60 000 values in that list.
        for record in training_data_list[::]:
            all_values = record.strip().split(',')
            targets = np.zeros(self.network.onodes) + 0.01
            # all_values[0] - answer for that test
            targets[int(all_values[0])] = 0.99
            
            # Translating the color codes from the range 0-255 to the range 0.01-1.0
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            self.network.train(inputs, targets)

        print("Ok", flush=True)
        training_data_file.close()

    
    def test_by_file(self, path_2):
        print("Started testing:", "....", sep="\n", flush=True)
        test_data_file = open(path_2, "r")
        test_data_list = test_data_file.readlines()

        # Log of network answers
        scorecard = list()

        # Here you can change the cut to whatever you want. There are 20 000 values in that list.
        for record1 in test_data_list[:101]:
            all_values = record1.split(',')
            correct_label = int(all_values[0])
            
            # Translating the color codes from the range 0-255 to the range 0.01-1.0           
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = self.network.query(inputs)
            # The index of the maximum value must be the right answer
            label = np.argmax(outputs)

            if label != correct_label:
                scorecard.append(0)
                print("A mistake: ", flush=True)
                print(label, " - Network answer", flush=True)
                print(correct_label, " - Correct answer", flush=True)                
                
                targets1 = np.zeros(self.network.onodes) + 0.01
                targets1[int(all_values[0])] = 0.99
                
                print("Retraining...", flush=True)
                self.network.train(inputs, targets1)
                print("Ok\n", flush=True)
            else:
                scorecard.append(1)


        print("Log (1 = correct, 0 = mistake):", scorecard, flush=True)
        print("Percentage of efficiency:", round(sum(scorecard) / len(scorecard), 7), flush=True)
        
        test_data_file.close()



    def test_by_window(self):
        test_data_file = self.app.get_pixels()       
        
        test_data_list = []
        for str_1 in test_data_file:
            test_data_list.extend(str_1)

        correct_label = int(input("Input a correct answer: "))
        inputs = np.asfarray(test_data_list)
        outputs = self.network.query(inputs)
        print(outputs, flush=True)

        # The index of the maximum value must be the right answer
        label = np.argmax(outputs)
        print("Network thinks that is the", label, flush=True)

        if label == correct_label:
            print()
            print("Correct!!!", flush=True)
            print()
        else:
            print("No, wrong anwer", flush=True)
            print("Retraining...", flush=True)
            
            targets1 = np.zeros(self.network.onodes) + 0.01
            targets1[int(correct_label)] = 0.99
            
            self.network.train(inputs, targets1)
            print("Ok\n", flush=True)
            print("Write something one more time", flush=True)
        return None



def main():
    lab = Laboratory()


if __name__ == "__main__":
    main()