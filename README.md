APPROACH AND FINDINGS:

BUSINESS UNDERSTANDING:

In the context of the CRISP-DM framework, the task of identifying key drivers for used vehicle prices can be reframed as a data 
analysis problem as follows:

The primary objective is to conduct an exploratory data analysis (EDA) on the dataset of 426K used cars to uncover trends, 
patterns, and relationships. This involves statistical analysis and visualizations to understand the distribution and 
characteristics of various features in the dataset, such as make, model, year, mileage, condition, and any other relevant
attributes.

In the data preparation phase of the CRISP-DM process, the focus is on cleaning, pre-processing, and transforming the dataset
to make it suitable for analysis and modeling. This phase is crucial as the quality of data significantly affects the outcome
of the analysis. The initial step involves cleaning the data, which includes identifying and handling missing or inconsistent
entries, removing duplicates, and correcting any errors or anomalies in the dataset. This may also involve standardizing units
of measurement and ensuring that categorical data is consistently labeled.

In the modeling phase, the task progresses to predictive modeling, where the goal is to develop a regression model (or models)
that can accurately predict the price of a used car based on its attributes. This involves selecting relevant features, 
handling missing data, encoding categorical variables, and choosing an appropriate regression algorithm and associated
hyper parameter.

The final step is to interpret the results of the model, focusing on identifying which features are most influential in 
determining used car prices. This will involve analyzing the model coefficients or feature importances to understand the 
impact of different car attributes on their resale value.

In the deployement phase, insights from the modeling are used to implement strategic used car dealer decisions.

DATA UNDERSTANDING:

This part involves exploring the dataset to understand its structure, quality, and the types of information it contains.

STEP 1: Obtain a general sense of the size and content of the dataset. Determine the column that will serve as the target
variable, and which columns will serve as inputs to the analysis.

STEP 2: Obtain the data type of each column. Determine which columns are numerical and which are categorical. Dtermine the
acceptable ranges for numerical values. Determine the number of unique values in each categorical column as well as the
acceptable values for each column. Identify derived columns and/or data transformations that might be useful for the analysis.

STEP 3: Assess the data quality. Determine if there are any duplicate entries, missing values, and inconsistencies between
columns. Develop strategies for dealing with them. For example, misclassifications in one column can often be determined and
corrected based on information in another column.

STEP 4: Use scatter plots, histograms, and heat maps to get a sense of relationships between the columns. Develop strategies
for transforming non-normal distributions, specifically with respect to the target variable. Identify opportunities for model
simplification, particularly with respect to the numerical columns, which may have interdependencies as identified by principal
componenents analysis.

DATA CLEANING AND PREPARATION:

STEP 1: 'price', 'year', 'manufacturer', and 'VIN' were deemed crucial for subsequent cleaning of the dataset. Therefore, all
rows where any of these values were missing were removed from the dataset.

STEP 2: 147,592 rows with duplicate VINs were identified and removed. VINs are unique identifiers for specific vehicles, so
duplicates are not allowed.

STEP 3: A large number of NaNs existed in the dataset. In some cases, the percentage of missing values in the columns was as
high as 72%. Therefore, the NaN's were replaced with the median value for the numerical columns and 'UNKNOWN' for the cate-
gorical columns.

STEP 4: A lot of hybrid vehicles were mis-classified as electric and vice-versa. Therefore, information in the 'model' column
was used to correct these designations where possible.

STEP 5: The 'id', 'VIN', 'region', 'state' and 'model columns were dropped from the dataset before modeling fits were applied.
The 'id', 'VIN', 'region', and 'state' columns contained information that was too specific for the purposes of this analysis.
Also, the 'model' column was too cluttered for practical use in this analysis. There were 29649 unique values in the 'model'
column although the number of actual vehicle models is far fewer. The proliferation of unique values was caused by non-standard
designations for the vehicle models. In theory, this could be cleaned up, but it was not practical for this analysis.

STEP 6: All sparse occurences of categorical classes were removed from the dataset using a threshold of 150 values. In other
words, if a categorical class had less than 150 entries in the dataset, all rows for that class were removed. This resulted
in the total removal of the following classes for the following five columns:

   COLUMN         REMOVED CLASSES
   manufacturer : 'harley-davidson', 'ferrari', 'datsun', 'aston-martin', 'land rover'
   condition    : 'salvage'
   cylinders    : '12 cylinders'
   title_status : 'missing', 'parts only'
   type         : 'offroad', 'bus'

STEP 7: The 'year' column was replaced with the 'age' column, which was more usefaul for the analysis. Also, the dataset
was augmented with the brand 'nationality' based on the identified vehicle manufacturer.

STEP 8: The 'age', 'odometer', and 'price' columns were limited to ranges where the vehicle pricing dynamics were expected
to be more uniform. Visual inspection of scatter plots indicated that different pricing dynamics were at play outside these
ranges:

   age:        0 - 30   years
   odometer: 12K - 325K miles
   price:    $2K - $80K
      
STEP 9: All categorical columns were transformed using onehotencoding, while numerical columns were scaled and augmented
with polynomial expansions to allow the model to capture non-linear relationships.

STEP 10: The Price column was identified as the target variable and was transformed to log(Price) to produce a normalized
distrbution for the target variable.

After completion of the dataset cleaning steps, the original dataset was reduced from 426,880 rows to 111,376 rows.

MODELING:

In this stage, machine learning models were developed to understand the relationship between different variables and vehicle
prices. I implemented five different modeling approaches in conjunction with GridSearchCV to optimize a set of hyper-parameters.
The models, hyper-parameters, and associated ranges that were used are as follows:

1) Ridge Regression
      alpha:                [0.001, 0.01, 0.1, 1, 10, 100]
      ploynomial degree:    [1, 2, 3]
      polynomial bias:      [True, False]
2) Lasso Regression
      alpha:                [0.001, 0.01, 0.1, 1, 10, 100]
      ploynomial degree:    [1, 2, 3]
      polynomial bias:      [True, False]
3) ElasticNet Regression
      alpha:                [0.001, 0.01, 0.1, 1, 10, 100]
	  L1 ratio:             [1, 1, 5]
4) SGD Regression
	  penalty:              [L1, L2, elasticnet]
      ploynomial degree:    [1, 2, 3]
      polynomial bias:      [True, False]
5) Random Forest Regression
      number of estimators: [50, 100, 200]
      max features:         ['auto', 'sqrt', 'log2']
	  
The cleaned vehicle data was split into training and test sets using a 70%/30% split. The training set was used with 
GridSearchCV to optimize the hyper-parameters of the model. A developement data set was not explicitly split out since the
GridSearchCV method internally splits out a development set from the training set that is provided to it. Once the optimized
models were determined via the GridSearchCV method, the test data set was used to cross validate each model and generate
MSE and R2 metrics. The results were as follows:

1) Ridge Regression
      MSE               = 0.09
      R2                = 0.80
      Best Alpha        = 0.001
      Best Poly Degree  = 3
      Best Poly Bias    = True
2) Lasso Regression
      MSE               = 0.10
      R2                = 0.79
      Best Alpha        = 0.001
      Best Poly Degree  = 3
      Best Poly Bias    = True
3) ElasticNet Regression
      MSE               = 0.09
      R2                = 0.80
      Best Alpha        = 0.001
      Best L1 Ratio     = 0.0
4) SGD Regression
      MSE               = 0.10
      R2                = 0.80
      Best Alpha        = 0.001
      Best Poly Degree  = 3
      Best Poly Bias    = True
      Best Penalty      = elasticnet
5) Random Forest Regression
      MSE               = 0.08
      R2                = 0.84
      Best N Estimators = 200
      Best Max Features = sqrt

Based on these results, Random Forest Regression was determined to be the best predictor of vehicle price because it 
had the lowest MSE and the highest R2 values. With an R2 = 0.84, it accounts for 84% of the variability in the inputs
features.

TEST SET MSE: Provides a clear indication of how well the models perform in terms of numerical predictions. A lower test set 
MSE indicates a better fit of the model to the data. With MSE, larger errors are given more weight. This is particularly use-
ful in scenarios where we want to avoid large deviations from the actual values.

TEST SET R2: Measures the proportion of the variance in the dependent variable that is predictable from the independent 
variables in the dataset. It provides an indication of the goodness of fit of a model. The higher the score, the better the 
fit, and the more the model explains the target values based on the independent variables.

DEPLOYMENT RECOMMENDATIONS:

The best model resulted from Random Forest Regression, which accounts for 84% of price variance. It is only valid for vehicles
1) 0 - 30 years of age, 2) 12K - 325K miles, and 3) Vehicle Prices in the range of $2k - $80k. The conclusions provided below
are not valid for vehicles outside these ranges.

The sweet spot for maximizing total sales revenue is with vehicles valued between $18k - $80K. Because of market availability,
most sales will involve vehicles near the lower end of this range. However, higher valued vehicles generally will contribute
more to profitability and should be prioritized.

The Top10 traded brands in order from greatest to least include: Ford, Chevrolet, Toyota, Honda, Nissan, Jeep, RAM, BMW, GMC,
and Dodge. However, in order of retained value from greatest to least, they are: Toyota, GMC, RAM, Honda, Jeep, Ford, 
Chevrolet, BMW, Dodge, and Nissan. The regression analysis identifies all of these makes as better than average in their
ability to retain resale value, so they should be prioritized.

Prioritize features in the following order:  1) age,                2) odometer,       3) drive,         4) type (body style), 
                                             5) cylinders,          6) manufacturer,   7) fuel,          8) paint color,
                                             9) nationality,       10) condition,     11) vehicle size, 12) transmission,
                                            13) title status

 1. AGE:             Understand that vehicle resale value diminishes with age up to 30 years. Vehicles older than 30 years
                     of age work off a different pricing model, so do not automatically discount vehicles greater than 30 years
                     of age. Also, the relationship between age and price is non-linear. Vehicle value initially drops off
                     steeply and levels off over time. 

 2. ODOMETER:        Vehicle value diminishes with odometer mileage up to 325k miles. Also, the relationship between odometer
                     mileage and price is non-linear. Vehicle value initially drops off steeply and levels off over time. 

 3. DRIVE TRAIN:     Prioritize (from best to worste): FWD, AWD, and then RWD

 4. TYPE:            Focus on popular vehicle types (body styles)

                     Prioritize:    Sedans, Pick-ups, SUVs, and Hatchbacks
                     De-Prioritize: Mini Vans, Convertibles, Vans, Wagons, and Coups

 5. CYLINDER COUNT:  For gas powered vehicles, prioritize cylinder counts in the mid-range (e.g. 4, 6, 8). Avoid vehicles
                     with odd cylinder counts (e.g. 3, 5).

 6. MANUFACTURER:    Focus on valued brands and models: Prioritize vehicles known for reliability and retained resale value.

                     Prioritize: Ford, Chevrolet, Toyota, Honda, Nissan, Jeep, RAM, BMW, GMC, and Dodge
                     Avoid:      Alpha-Romeo, Jaguar, Fiat, Volvo, Saturn, Infiniti, Mercury, Tesla*, Acura, Lincoln

                     *NOTE: Tesla is a relatively young brand. Although the analysis identifies it as a make to avoid, there
                     are likely other dynamics at play that require further study. Vehicles depreciate more rapidly in their
                     first few years. Given the likely lack of older Tesla vehicles on the road, the analysis is likely
                     penalizing Tesla for retained value becuase it is a newer brand.

 7. FUEL TYPE:       Prioritize (from best to worste): Diesel, Gas, Hybrid, Electric

 8. PAINT COLOR:     Prioritize neutral paint colors.
                     Colors (from best to worst): White, Black, Silver, blue, Red, Grey, Green, Brown, Yellow, Orange, Purple

 9. NATIONALITY:     Prioritize: American, Japanese, German, and South Korean brands.
                     De-Prioritize: Italian, Swedish, and British brands.

 10. CONDITION:      Prioritize vehicles in better condition, but realize that 'new' or 'like new' vehicles that are younger
                     will still be steeply discounted because they've been used.

 11. SIZE:           Avoid sub-compact vehicles. All other sizes similarly retain value.

 12. TRANSMISSION:   Manual is preferred over automatic.

 13. TITLE STATUS:   Stick with 'Clean' titles.

 NEXT STEPS:

   1) Investigate the vehicle sales price round-down effect. Sales revenue data indicates a common practice of rounding
      down final sales prices to the nearest $1K increment and even more so to the nearest $5K increment. This may represent
      buyer psychology where sales prices are rounded down to close deals more quickly. However, the pattern could represent
      dealership biases in vehicle pricing that are leading to lost revenue.
   2) Complete a residuals analysis to determine if there are any remaining patterns in the data that the model doesn't
      account for
   3) Compare these findings with industry benchmarks or trends to validate the recommendations.
   4) Implement more extensive cleaning of the dataset based on the content provided in the 'model' column.
   5) Explore the impact of modern features like advanced safety technology (e.g. abs braking, air bags), infotainment
      systems, or electric/hybrid engines and drive trains on resale value.
   6) Explore how younger brands like Tesla are prioritized by the model to ensure their resale values are accurately
      accounted for.
   7) Look at how consumer knowledge and understanding of new technologies, such as hybrid and electric vehicles, impacts
      retained value.
