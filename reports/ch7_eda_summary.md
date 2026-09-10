## Integrity Checks

train_transaction shape: (590540, 394)

train_identity shape: (144233, 41)

Duplicate transactionid in tx: 0

Duplicate transactionid in idn: 0

isfraud unique values: [0 1]

transactionamt min/max: 0.251 / 31937.391

transactionamt <= 0 count: 0

Identity coverage: 24.42%

productcd value counts:
productcd
W    439670
C     68519
R     37699
H     33024
S     11628
Name: count, dtype: int64

card4 value counts:
card4
visa                384767
mastercard          189217
american express      8328
discover              6651
NaN                   1577
Name: count, dtype: int64

card6 value counts:
card6
debit              439938
credit             148986
NaN                  1571
debit or credit        30
charge card            15
Name: count, dtype: int64

Class balance: {0: 96.50099908558268, 1: 3.4990009144173126}

            count        mean         std    min     25%   50%    75%        max
isfraud                                                                         
0        569877.0  134.511665  239.395078  0.251  43.970  68.5  120.0  31937.391
1         20663.0  149.244779  232.212163  0.292  35.044  75.0  161.0   5191.000

Dataset spans approximately 183 days

### Fraud rate by productcd
           transaction_count  fraud_rate
productcd                               
C                      68519    0.116873
S                      11628    0.058996
H                      33024    0.047662
R                      37699    0.037826
W                     439670    0.020399

### Fraud rate by card4
                  transaction_count  fraud_rate
card4                                          
discover                       6651    0.077282
visa                         384767    0.034756
mastercard                   189217    0.034331
american express               8328    0.028698

### Fraud rate by card6
                 transaction_count  fraud_rate
card6                                         
credit                      148986    0.066785
debit                       439938    0.024263
charge card                     15    0.000000
debit or credit                 30    0.000000

Mean missingness by column group:
D-columns                 58.151263
M-columns                 49.923328
V-columns (anonymized)    43.038469
Named/business fields     15.072890
C-columns                  0.000000

Top 10 most-missing identity columns:
id_24    96.708798
id_25    96.441868
id_07    96.425922
id_08    96.425922
id_21    96.423149
id_26    96.420375
id_23    96.416215
id_27    96.416215
id_22    96.416215
id_18    68.722137

### Fraud rate by identity linkage
              transaction_count  fraud_rate
has_identity                               
False                    446307    0.020939
True                     144233    0.078470

### Fraud rate by DeviceType (identity-linked transactions only)
            transaction_count  fraud_rate
devicetype                               
mobile                  55645    0.101662
desktop                 85165    0.065215