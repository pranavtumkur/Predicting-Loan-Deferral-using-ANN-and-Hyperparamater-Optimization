# Predicting-Loan-Deferral-using-ANN-and-Hyperparamater-Optimization

![232](https://user-images.githubusercontent.com/65482013/84423831-b604bb00-ac3c-11ea-8133-404a72e2a6e7.jpg)
  
In this project, we build and train a model to predict if a customer will defer on a particular loan on an imbalanced dataset. We'll build a layered ANN for this and try to make our model better using Hyperparamater Optimization

## The Data

We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

## Our Goal

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model thatcan predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. Keep in mind classification metrics when evaluating the performance of your model!

The "loan_status" column contains our label.

## Data Overview

There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>

## EDA takeouts

1. There is a large mismatch in the samples for the parameter we are predicting (loan_status) - Compared to 318357 records of loan repaid, just 77673 are available for Charged Off Loans
2. Higher the loan amount, slightly higher the chance of the loan being Charged Off
3. Customers of F and G subgrades don't get paid back that often.
4. Repayment of loan is not dependent on employment status or length of employment
5. People who want to repay loan in 36 EMIs is 320% higher than those who want to repay it in 60 EMIs
6. 90% of people who have taken a loan live in houses which have been mortgaged or rented. Just 9.5% of the loan-takers hold ownership of their houses.

## Creating model

We make an ANN with the following properties-

* 1 input layer with neurons = independent param
* 1st hidden layer with dropout and neurons = independent param/2 and activation function as relu
* 1st hidden layer with dropout and neurons = independent param/4 and activation function as relu
* 1 output layer with 1 neuron and loss = binary crossentropy and optimizer as adam
          
 ## Model Evaluation
 
 1. No separation of loss (on training data) and validation loss (on testing data) graphs. Therefpre, no overfitting.
 2. 88% accuracy (This is moderately good, since the data is imbalanced 80-20)
 3. 85% and 88% precision for loans charged off and repaid resp.
 4. **Now comes the problematic one** - 48% recall for loans charged off (This is mainly due to the imbalanced dataset)

## Hyperparameter Optimization

Using GridSearchCV, we iterate through all combinations of the below configurations of the ANN

* Hidden Layers
  * 50
  * 25
  * 50,25
  * 50,25,10
  * 60,45,30,15
  * 60,45,30,15,5
* Activation Functions
  * sigmoid
  * relu
* batch_size
  * 500
  * 256
  * 128
* Training Epochs
  * 25
  * 30

## Result of Hyperparameter Optimization

We find that the best ANN (88.88% accuracy) for our data is with the following configuration-
* Hidden Layers- 2
* No. of neurons in hidden layers- 50, 25
* Activation Function- relu for all hidden layers
* Training Batch size- 128
* Training epochs- 20

We achieve a 4% increase in recall while maintaining the same 89% accuracy after using GridSearchCV to determing the best ANN

**But this result of recall 47% for loan deferral cases is still not acceptable since it is quite low!** We already know that the reason for low recall of cases where loan was deferred, is due to the imbalance in the dataset for samples with loan repaid (very high) vs loan deferred cases (very low).

So we try to feed our ML model better and more balanced data using- **oversampling**

## Oversampling

We use SMOTTomek class from the imblearn library, to reshape our data to create equal no. of records/samples where loan was deferred and where it was repaid.

## Training and Evaluation of the model after Oversampling

We use the optimal ANN obtained from GridSearchCV, to not train our model with the above reshaped data (after oversampling transformation).

### This gives us wonderful results! Earlier we had 47% recall for loan deferred cases with 89% model accuracy.

# But now, we have 87% recall for loan deferred cases with 93% model accuracy!

This is a model good enough to deploy for all practical purposes!


