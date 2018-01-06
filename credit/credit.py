import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import math

df = pd.read_excel('./train.xlsx')

#remove disbursed null part
df = df[df.isnull()['Disbursed'] == False]

#remove the DOB is greater than the lead date
df = df[df['DOB'] < df['Lead_Creation_Date']]

#city and reduce the category
df['City'].fillna(df['City'].mode()[0], inplace=True)
df['Salary_Account'].fillna('Other', inplace=True)

city_counter = Counter(df['City'])
city_arr = [ (k, city_counter[k]) for k in city_counter.keys()]
city_arr = sorted(city_arr, key=lambda x: x[1], reverse=True)
major_city = [k for k, c in city_arr[:10]]

df['City'] = df['City'].map(lambda x: x if x in major_city else 'Other')

#calcualte the age
def calcualte_age(row):
	dob = row['DOB']
	lead_date = row['Lead_Creation_Date']
	return (int)((lead_date - dob).days / 365)
df['Age'] = df.apply(calcualte_age, axis=1)

#calcualte the top account
def get_top_freq_items(ser, n):
	c = Counter(ser)
	c_arr = [(k, c[k]) for k in c.keys()]
	c_arr = sorted(c_arr, key=lambda x:x[1], reverse=True)
	cnt = 0
	ret = []
	for i in range(n):
		cnt += c_arr[i][1]
		ret.append(c_arr[i][0])
	print(cnt)
	return ret

top_account = get_top_freq_items(df['Salary_Account'], 10)
df['Salary_Account'] = df['Salary_Account'].map(lambda x: x if x in top_account else 'Other')


#calculate the loan
def calculate_loan_amount(row):
	submit = row['Loan_Amount_Submitted']
	applied = row['Loan_Amount_Applied']
	if not math.isnan(submit):
		return submit
	if not math.isnan(applied):
		return applied
	return None

df['Loan_Amount'] = df.apply(calculate_loan_amount, axis=1)
#remove the loan is equal to 0
df = df[df['Loan_Amount'] > 0]


#calculate the tenure
def calculate_loan_tenure(row):
	submit = row['Loan_Tenure_Applied']
	applied= row['Loan_Tenure_Submitted']
	if not math.isnan(submit):
		return submit
	if not math.isnan(applied):
		return applied
	return None

df['Loan_Tenure'] = df.apply(calculate_loan_tenure, axis=1)

#fill na for other column
df['Interest_Rate'].fillna(df['Interest_Rate'].mean(), inplace=True)
df['Existing_EMI'].fillna(df['Existing_EMI'].mode()[0], inplace=True)

#calculate processing fee
def process_fee_avg(row):
	loan = row['Loan_Amount']
	process_fee = row['Processing_Fee']
	if not math.isnan(process_fee):
		return loan / process_fee
	return None

df['avg_pro_fee'] = df.apply(process_fee_avg, axis=1)
avg_pro_fee = df['avg_pro_fee'].mean()

def process_fee(row):
	orig = row['Processing_Fee']
	loan = row['Loan_Amount']
	if not math.isnan(orig):
		return orig
	return loan * avg_pro_fee

df['Processing_Fee'] = df.apply(process_fee, axis=1)
del df['avg_pro_fee']

#calculate the Emi
def calculate_emi(row):
	existing_emi = row['Existing_EMI']
	rate = row['Interest_Rate'] / 1200
	loan = row['Loan_Amount']
	dur = row['Loan_Tenure'] * 12
	emi_submit = row['EMI_Loan_Submitted']
	if not math.isnan(emi_submit):
		return emi_submit + existing_emi
	elif dur == 0:
		return None
	else:
		emi = loan * rate * pow(1 + rate, dur) / (pow(1+ rate, dur) - 1)
		return emi + existing_emi


df['Emi'] = df.apply(calculate_emi, axis=1)
#filter no emi part
df = df[df.isnull()['Emi'] == False]

#filter the loan is less than the 10000
df = df[df['Loan_Amount'] > 10000]

df['Ratio_Income_EMI'] = df['Monthly_Income'] / df['Emi']
#filter the ration greater than 50, only have 1 negative 
df = df[df['Ratio_Income_EMI'] < 50]

#del the columns not used any more
del df['DOB']
del df['Lead_Creation_Date']
del df['Loan_Amount_Applied']
del df['Loan_Tenure_Applied']
del df['Employer_Name']
del df['Loan_Amount_Submitted']
del df['Loan_Tenure_Submitted']
del df['EMI_Loan_Submitted']
del df['Existing_EMI']
del df['Interest_Rate']
del df['ID']
del df['Loan_Amount']
del df['Emi']
del df['Monthly_Income']
del df['Processing_Fee']

norm_col_list = ['Var5', 'Var4', 'Age', 'Loan_Tenure', 'Ratio_Income_EMI']
df_to_norm = df[norm_col_list]
df_norm = (df_to_norm - df_to_norm.mean()) / (df_to_norm.max() - df_to_norm.min())
df[norm_col_list] = df_norm


#sample
df_0 = df[df['Disbursed'] == 0]

df_0_sample = df_0[:5000]

df_1 = df[df['Disbursed'] == 1]

df_xy = pd.merge(df_0_sample, df_1, how='outer')
target = df_xy['Disbursed']
del df_xy['Disbursed']

df_xy_dummy = pd.get_dummies(df_xy)

from sklearn import svm
from sklearn import model_selection
import random

train_x, test_x, train_y, test_y = model_selection.train_test_split(df_xy_dummy, target, test_size=0.2)

#xgboost
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(train_x, train_y)
pred = xg.predict(test_x)
accu = np.sum(pred == test_y) / len(test_y)
print('accu for xgboost is ', accu)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import scipy

#use grid cv to search the hyper parameter
tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}]

clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy', n_jobs = 6)
clf.fit(train_x, train_y)

print(clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

y_true, y_pred = test_y, clf.predict(test_x)
print(classification_report(y_true, y_pred))

#random cv search
tuned_parameters = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1)}
clf = RandomizedSearchCV(svm.SVC(), param_distributions=tuned_parameters, cv=5, n_jobs=6, scoring='accuracy')
clf.fit(train_x, train_y)

print(clf.best_params_)
y_true, y_pred = test_y, clf.predict(test_x)
print(classification_report(y_true, y_pred))


#bayes search
print('bayes optimization')
#bayes optimization
def calculate_test_accurency(arr):
	c = arr[0]
	gamma = arr[1]
	model = svm.SVC(C=c, gamma=gamma)
	model.fit(train_x, train_y)
	pred_y = model.predict(test_x)
	accu = np.sum(pred_y == test_y) / len(test_y)
	print(accu)
	return 1 - accu


from skopt import gp_minimize

res = gp_minimize(calculate_test_accurency, [(0.01, 100), (0.01,100)], n_calls=50, acq_func="EI")


def calculate_test_accurency(c, gamma):
	model = svm.SVC(C=c, gamma=gamma)
	model.fit(train_x, train_y)
	pred_y = model.predict(test_x)
	accu = np.sum(pred_y == test_y) / len(test_y)
	print(accu)
	return accu

print('grid search')
#grid search
c_options = [0.01, 0.1, 1, 10, 100]
gamma_options = [0.001, 0.01, 0.1, 1, 10, 100]

best_accu = 0
best_c = 0
best_g = 0
for c in c_options:
	for g in gamma_options:
		res = calculate_test_accurency(c, g)
		if res > best_accu:
			best_accu = res
			best_c = c
			best_g = g

best_accu = 0


print('random search')
#random search
for i in range(50):
	c = random.random() * 100
	g = random.random() * 100
	res = calculate_test_accurency(c, g)
	if res > best_accu:
		best_accu = res
		best_c = c
		best_g = g

print('bayes optimization')
#bayes optimization
def calculate_test_accurency(arr):
	c = arr[0]
	gamma = arr[1]
	model = svm.SVC(C=c, gamma=gamma)
	model.fit(train_x, train_y)
	pred_y = model.predict(test_x)
	accu = np.sum(pred_y == test_y) / len(test_y)
	print(accu)
	return 1 - accu


from skopt import gp_minimize

res = gp_minimize(calculate_test_accurency, [(0.01, 100), (0.01,100)], n_calls=50, acq_func="EI")



#useful links
#https://thuijskens.github.io/2016/12/29/bayesian-optimisation/#parameter-selection-of-a-support-vector-machine

#https://scikit-optimize.github.io/notebooks/bayesian-optimization.html
