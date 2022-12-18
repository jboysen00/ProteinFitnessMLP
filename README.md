# ProteinFitnessMLP

Protein engineering holds great promise for a wide range of human endeavors, such as the development of therapeutics drugs and gene editing, through producing protein variants that enhance the original function or are entirely novel [1]. One traditionally successful approach to protein engineering is laboratory-based directed evolution, however, this method is constrained by the limited protein sequence space that can be sampled and requires significant experimental effort [2]. Protein language models (PLM) have been found to generate state-of-the-art representations of biological properties and achieve impressive prediction performance in protein prediction related tasks [3]. However, current methods employ a classical model predictor to unlock the information encoded in the learned representations and predict a protein fitness score. Here I show the accuracy of using a novel deep neural network to generate fitness predictions. I found that using a Multilayer Perceptron with three hidden layers taking as input a PLM embedding of a mutated amino acid with a protein sequence does not accurately predict protein function. These results demonstrate that this architecture cannot accurately capture sufficient information from the sequence to yield an accurate fitness prediction. 

## Download ProteinGym Substitution Benchmark
ProteinGym is a set of Deep Mutational Scanning (DMS) assays that provides insight on various mutational effects on fitness [4]. I focused on the substitution benchmark, which contains 87 DMS assays and the experimental characterization of 1.5 million missense variants. The DMS fitness scores correspond to each mutation’s respective experimental measurement in the DMS assay. The higher the fitness score, the higher the fitness of the mutated protein. 

Download CSV files within a file named 'ProteinGym_substitutions'
```
curl -o ProteinGym_substitutions.zip https://marks.hms.harvard.edu/tranception/ProteinGym_substitutions.zip 
unzip ProteinGym_substitutions.zip
rm ProteinGym_substitutions.zip
```
Two of the DMS assay files are a different format and must be removed.
```
rm F7YBW8_MESOW_Aakre_2015.csv
rm GCN4_YEAST_Staller_induction_2018.csv
```

## Generate Mutation PLM Token Representations

As a prerequisite, you must have PyTorch installed. The Facebook ESM-2 model [5] can be downloaded via:
```
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```
Once the model has been downloaded, the mutation PLM token representations can be generated.
```
./Boysen_PLMrepresentations.py
```

## PLM-Based Multilayer Perceptron
The Multilayer Perceptron was chosen as it is fully connected and is thus structurally agnostic developed to output a predicted protein fitness score given the mutation PLM token representation. The model contains 3 hidden layers, each of which with a ReLU activation function. As this is a regression model that outputs continuous and real numbers, I used the mean squared error (MSE) loss function. The current network architecture starts with the 320 initial input features and is reduced to 200, 75, 10, and finally 1. This model is trained over the course of 100 epochs with a stochastic gradient descent optimizer learning rate of 0.01 and a momentum parameter value on 0.9. To maximize the number of samples used to train the model, I implemented an 80 to 20 test train split. 

Train the model and predict the fitness scores for the test sequences.
```
./Boysen_ProteinNN2.py
```
