# Modeling Immunogenicity of De Novo Proteins
Welcome to our project repository! This project is designed to predict the immunogenicity of *de novo* proteins by integrating various data sources and leveraging existing immunogenicity prediction models. 

Our strategy begins with a preprocessing of peptide sequences to ensure that all data is cleaned and standardized for accurate analysis. We then evaluate the immunogenicity of various peptides while also assessing the predictive accuracy of each model. A key component of this project is the examination of important features that play a significant role in determining immunogenicity, especially as these features can differ noticeably between natural proteins and de novo proteins. The primary features we focus on are melting temperature and peptide length (Quijano *et al*., 2020). We compare these attributes between the peptides being evaluated and the training dataset to understand how they relate to the model's predictive capabilities, providing insights into what influences immunogenicity predictions.

# Table of Content
- Data Input Requirements
- Data Input Cleanup
- Melting Temperature Prediction
- Visualization of Input Datasets
- Immunogenicity prediction
- Evaluating Immunogenicity Prediction
- Euclidean Distance Calculation
- References


## Data Input Requirements
The data input for this project consists of two types of datasets: Test dataset and the Training dataset from the models evaluated. Although they differ in content, both datasets must adhere to the same structure. Below is the required format for each input file:

### CSV File Structure
Each CSV file should contain the following columns in the specified order:

*   Sequence: The protein sequence of interest
*   Length: The length of the sequence.
*   Tm: The melting temperature of the sequence.

#### Additional columns
The Test dataset contains two extra columns that are important to include for following analysis:

*   Category: The immunogenic classification of the sequence.
*   Immunogenic Score: The known immunogenic score for controls.

#### Additional Notes
The CSV file must contain a header row with the column names as specified above.

Ensure each column is correctly populated as missing or malformed data may lead to processing errors.

Once data matches the structure outlined above, place the CSV file in the designated directory within the repository.

The category labels used in this project are: clinically approved /native, non-self / de-immunized mutant, non-self / cancer-associated neoantigens / de novo, not tested in humans.

#### Example Data
An example of how the data should be structured is shown below:

```
PTSSST,133,52.9069,clinically approved,0
GPEEEG,15,59.10659,de-immunized mutant, non-self,0
MCDLPQ,166,54.9079,clinically approved,0
```
# Data Input Cleanup
In this section, we outline the process for cleaning the datasets to ensure it is suitable for further analysis and modeling in the context of immunogenicity prediction. The cleaning process includes several critical steps that enhance the data quality by removing duplicate entries, filtering out unsuitable sequences, and preparing the dataset for effective utilization in immunogenicity prediction models.

### Key Steps:
*  Remove Duplicates: The first step involves deduplicating the dataset to guarantee that each peptide sequence is unique. This helps prevent redundancy in the analysis and ensures that all data points contribute meaningfully to model training and evaluation.

*  Filter Peptides by Length: Peptides shorter than 15 amino acids are excluded from the dataset, as such sequences are incompatible with many CD4 T cell prediction models.

*  Handle Missing Values: To enable accurate data analysis, entries marked as "N/A" in the 'Immunogenic Score' column are converted to true NaN values. This step ensures that missing values are properly recognized by analytical tools and do not introduce biases in the model predictions.

*  Prepare Peptide List: A list of unique, filtered peptide sequences is generated for integration into the immunogenicity prediction models. This final list is essential for ensuring that the data fed into the models is clean, well-structured, and ready for predictive analysis. In some cases, models require data inputs to go in batches of 100 datapoints. The code to split the dataset in batches of 100 is also provided.

### Usage Instructions
*  Input File Preparation: Ensure that your input files are properly formatted and placed in an accessible directory.

*  Modify Paths: In the script, adjust the input_file path and file_type argument within the deduplicate_and_save function calls in the main execution block as needed to point to your specific data files.

*  Run the Script: Execute the script to perform the data cleaning process, which will generate a deduplicated output CSV file that can be used in subsequent analysis and modeling tasks.

# Melting Temperature Prediction
## Overview
deepSTABp is a deep learning tool developed to predict the melting temperature (Tm) of proteins, utilizing a transformer-based model for sequence embedding and feature extraction (Jung *et al.*, 2023). Unlike traditional experimental methods, which are costly and limited in scalability, deepSTABp offers a computational alternative that can handle a wider proteome and species coverage.

## Predicting Melting Temperature

### Online
Web Interface: Directly predict melting temperatures by visiting the deepSTABp web https://csb-deepstabp.bio.rptu.de/. It is important to mention that the highest number of sequences allowed in one file is 1000.

If a couple of 1000 sequences require Tm prediction, a code to split the batches of 1000 is provided and recommended.

### Local
If more than 1000 sequences require Tm prediction, a local usage is recommended.

Local Installation: Clone the ARC repository to run deepSTABp locally, providing flexibility and control over your predictions.

#### Setup Environment
To run deepSTABp locally, follow these steps:

```
# Clone the Repository
# Ensure git and git lfs are installed, then clone the repository
git clone https://github.com/your-username/deepSTABp.git
cd deepSTABp

# Create the Conda Environment
# In the /workflows directory of the cloned repository, create a conda environment using the provided YAML configuration file
conda create --name deepstabp --file environment.yml

# Note: The complete ARC is large (approximately 67 GB), so ensure you have sufficient disk space
```
### Prepare input file
To convert the .csv file to .fasta format for proper analysis in the deepSTABp environment, a utility from the repository [CSV-to-FASTA-Converter](https://github.com/e18rayan/CSV-to-FASTA-Converter) was used.



```
python csv2fasta.py Sequences.csv
Sequences.fasta file successfully created
```

### Running deepSTABp



```
# Activate the Conda Environment
conda create --name deepstabp --file environment.yml
conda env update --file environment.yml --name deepstabp
conda activate deepstabp

# Navigate to the Prediction Model Directory
# Change directory to where the prediction scripts are located
cd workflows/prediction_model

# Run the Prediction Script
# Choose between CPU and GPU versions
# For CPU
python predict_cpu.py <filepath>
# For GPU
python predict_cuda.py <filepath>

# Specify Additional Parameters
# You can set the growth temperature and protein environment using flags
python predict_cpu.py <filepath> -g growthtemp<int> -m measurementcondition <str>
# Measurement Conditions: "Cell" and 37°C are the recommended conditions to test for immunogenicity

#Save Output:
#To store the output in a specified location, use the -s flag:
python predict_cpu.py <filepath> -g growthtemp<int> -m measurementcondition<str> -s <savepath>

# Example
# python predict_cpu.py Sequences.fasta -g 37 -s Data_Output
# Cell is default option
```

# Visualization of Input Datasets
This section includes visualizations for the distribution of peptide length and melting temperature (Tm) for the input datasets. These plots help in understanding the data distribution and characteristics.

## Temperature Melting (Tm) Distribution
To visualize the distribution of melting temperatures (Tm) from the dataset, the code provided could be use for this purpose.

## Peptide Length Distribution
When analyzing peptide datasets, it is common to encounter certain lengths that are predominantly more frequent than others. This disparity can make it difficult to interpret the distribution of less common lengths, as the presence of a large peak can obscure lower-frequency values.

To address this challenge, we present two types of visualizations:

*   Linear Frequency Plot: This plot displays the length distribution on a standard linear scale. It provides a general overview of the data, highlighting the most common peptide lengths.
*   Axis-Break Plot: This plot features an axis break to allow for a clearer visualization of both high-frequency and low-frequency peptide lengths. By breaking the axis, we can more effectively show the distribution of less common lengths while still retaining the information from the more frequently occurring lengths.

We recommend running both plots. The linear frequency plot serves as a guide to identify which ranges of lengths are particularly prominent, while the axis-break plot allows for a focused examination of both the common and rare peptide lengths in the dataset.

## Visualization of *De Novo* Proteins' Tm Distribution
The heat-stability of *de novo* proteins is greatly increased compared to normal proteins. In order to highlight this difference within the test dataset, the following code specifically visualize the melting temperature (Tm) distribution of de novo proteins within the test dataset.

# Immunogenicity prediction
An artificial neural network (ANN) from Dhanda *et al.* (2018) predict the immunogenicity of peptide sequences is performed in this section. Unlike traditional methods that rely on HLA binding affinity, this model predicts CD4+ T cell immunogenicity at the population level without needing HLA typing data. By training on validated datasets, it identifies key features that differentiate immunogenic peptides from non-immunogenic ones, resulting in an HLA-agnostic immunogenicity score. 

### Output 
For classification, we apply an immunogenicity score threshold of 70, above which peptides are excluded as
candidates that are predicted unlikely to be immunogenic. This cutoff was empirically
derived from immunogenicity prediction benchmarks, where it achieves >99% specificity
with minimized false positives.

The function returns a binary list representing the immunogenicity status of each peptide, where a value of '1' indicates immunogenic and '0' indicates non-immunogenic. This information is subsequently added as a new column to the cleaned dataset and exported to an Excel file for further analysis.

### Usage Instructions
*   Input the peptides into the webpage platform (http://tools.iedb.org/deimmunization/) following the structure: "Model{model_number}_peptide_batch_{batch_number}.txt" to run the Model 1.
*   The output file should be in the format of "CD4_Prediction_{file_num}.csv"
*   Those files should be put in the Data folder; and directory path should be re-run to include those new files that include immunogenicity scores of the input peptide sequences.
*   The output from model 1 is then processed. It iterates over the specified input files (named according to the base_filename), reading the unique protein numbers from each file. The unique protein numbers are adjusted based on their corresponding file number, and each immunogenic sequence is marked as '1' in the output list.

  ## Immunogenic Prediction of *De Novo* Proteins
In this section, the predicted immunogenicity of *de novo* proteins is visualized as immunogenic and non-immunogenic, based on the model’s output. This distribution helps in the evaluation of trends and patterns in the predictive models.

### Key Steps
The function takes in a filtered_df DataFrame, which contains the immunogenicity predictions for the peptides, and an optional parameter model_name to label the plot appropriately.

*   Identify Missing Scores: The function first identifies which peptides have missing immunogenicity scores. This is important to ensure that only complete data is used for visualization.

*   Count Predictions: It counts how many peptides are classified as immunogenic (marked as 1) and how many are classified as non-immunogenic (marked as 0). This helps in understanding the model's predictions and the distribution of these classes.

*   Create Bar Plot: A bar plot is generated to visually represent the counts of immunogenic and non-immunogenic peptides. Each category is displayed in a different color, making it easy to distinguish between the two.

### Usage Instructions 
*   Prepare the DataFrame: Ensure that the filtered_df DataFrame is populated with the immunogenicity predictions from your model, including a column for the 'Immunogenic Score' and a unique identifier for each peptide.
*   Execute the Visualization Function: Run the de_novo_protein_predictions function, providing the prepared DataFrame and display the bar plot that illustrates the distribution of immunogenicity predictions.

# Evaluation of Immunogenicity Prediction
To assess the predictive power of the immunogenic predictor model, the predicted immunogenicity scores can be compared against experimental results of the control peptides included in the test dataset. This evaluation employs a confusion matrix to visualize the classifications of true immunogenic and non-immunogenic peptides, as well as a calculation of the balanced accuracy and the F1 score.

### Performance Metrics

*   Balanced Accuracy: This metric accounts for class imbalances by averaging the sensitivity and specificity, providing a more accurate representation of model performance in datasets where the classes are not evenly distributed.

*   F1 Score: This metric balances precision and recall, offering a single value that reflects the model's ability to correctly identify both immunogenic and non-immunogenic peptides.

*   Confusion Matrix: The confusion matrix visually summarizes the model's classification results, allowing us to identify specific areas where the model overpredicts or underpredicts immunogenicity.

### Usage Instructions
To run the evaluation, the true immunogenicity scores and the predicted scores are extracted from the filtered_df DataFrame. The metrics are then computed by calling the immunogenicity_model_metrics function, which generates visualizations of the performance metrics and confusion matrix.

# Euclidean Distance Calculation
A critical aspect of this project is the examination of key features that significantly influence immunogenicity predictions. Understanding how these features differ between natural and *de novo proteins* is essential for enhancing the accuracy of predictive models. The two primary features of this analysis are melting temperature (Tm) and peptide length, as highlighted by Quijano et al. (2020). This section evaluates the similarity between peptide sizes in the training and test datasets by calculating the Euclidean distance. This analysis is crucial for comparing top-performing immunogenic prediction models, particularly considering the variability in training strategies and their relevance in immunogenic determination. By assessing the distances between peptide sizes used in the training datasets and those present in the test dataset, we can gain insights into how well the models are likely to perform.

### Key steps
The compute_pairwise_distances function calculates the Euclidean distances between standardized features (specifically peptide length and melting temperature) in the training and test datasets. The function outputs an array of distances, where each value represents the distance between a test peptide and the closest training peptide. The function also generates a scatter plot visualizing the training data and the test data color-coded by distance.

### Usage Instructions
*   Prepare Your Datasets: Ensure that both the training and test datasets are in the appropriate format, with relevant features such as 'Length' and 'Tm' included as columns.

*   Run the Distance Calculation: Call the compute_pairwise_distances function using your training and test DataFrames. 
*   Visualize the Results: The function automatically generates a scatter plot that displays the training data and test data, with the test data points colored according to their distances to the training data.

# References

Dhanda, S.K., Grifoni, A., Pham, J., Vaughan, K., Sidney, J., Peters, B. and Sette, A.
(2018), Development of a strategy and computational application to select candidate
protein analogues with reduced HLA binding and immunogenicity. Immunology, 153,
118-132. https://doi.org/10.1111/imm.12816

Jung, F., Frey, K., Zimmer, D., & Mühlhaus, T. (2023). DeepSTABp: A Deep Learning Approach for the Prediction of Thermal Protein Stability. International Journal of Molecular Sciences, 24(8), 7444. https://doi.org/10.3390/ijms24087444

Quijano-Rubio, A., Ulge, U.Y., Walkey, C.D. & Silva, D.A. (2020). The advent of de
novo proteins for cancer immunotherapy. Current Opinion in Chemical Biology.
https://doi.org/10.1016/j.cbpa.2020.02.002.
