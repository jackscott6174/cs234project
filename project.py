
# coding: utf-8

# In[1]:


import pandas
import numpy as np
from enum import Enum
import re
from scipy import stats


# In[2]:


header = list(pandas.read_csv("data/warfarin.csv", nrows=1).columns)

class InputType(Enum):
    ENUM = 0
    REAL = 1
    LIST = 2
    RANGE = 3
    
input_types = dict(zip(header, [InputType.ENUM] * len(header)))
input_types.update({
    'Height (cm)': InputType.REAL,
    'Weight (kg)': InputType.REAL,
    'Indication for Warfarin Treatment': InputType.LIST,
    'Comorbidities': InputType.LIST,
    'Medications': InputType.LIST,
    'Target INR': InputType.REAL,
    'Estimated Target INR Range Based on Indication': InputType.RANGE,
    'Therapeutic Dose of Warfarin': InputType.REAL,
    'INR on Reported Therapeutic Dose of Warfarin': InputType.REAL,
})

panda_types = {}
for feature, f_type in input_types.items():
    if f_type == InputType.ENUM:
        panda_types[feature] = 'category'
    elif f_type == InputType.REAL:
        panda_types[feature] = np.float32
    else:
        panda_types[feature] = str
        
output_feature = 'Therapeutic Dose of Warfarin'

basic_features = [
    'Age',
    'Gender',
    'Race',
    'Ethnicity',
    'Height (cm)',
    'Weight (kg)'
]

medical_features = header[header.index('Medications') : header.index('Target INR')] + ['Current Smoker']

genome_features = header[header.index('Cyp2C9 genotypes') : header.index('Unnamed: 63')]


# In[3]:


raw = pandas.read_csv('data/warfarin.csv', dtype=panda_types)
raw = raw.drop(np.where(pandas.isnull(raw[output_feature]))[0]).reset_index()
m = raw.shape[0]

patients = None
d = None

def enzyme_inducer_status(meds_str):
    return 1 if re.search('carbamazepine|phenytoin|rifampin|rifampicin', meds_str) else 0

def amiodarone_status(meds_str):
    return 1 if re.search('amiodarone', meds_str) else 0

def set_patients(use_features):
    def list_cols(feature):
        if feature == 'Medications':
            return [
                [enzyme_inducer_status(meds_str) if pandas.notnull(meds_str) else 0 for meds_str in raw['Medications']],
                [amiodarone_status(meds_str) if pandas.notnull(meds_str) else 0 for meds_str in raw['Medications']]
            ]

    columns = []
    LIST_DELIMS = ';|and\/or| and |/'
    for feature, f_type in input_types.items():
        if feature not in use_features:
            continue
        if f_type == InputType.ENUM:
            new_cols = [np.zeros(m) for _ in range(len(set(list(raw[feature]) + [np.nan])))]
            for t, v in enumerate(raw[feature].cat.codes):
                new_cols[v + 1][t] = 1
            columns += new_cols
        elif f_type == InputType.REAL:
            avg = np.mean(raw[feature][pandas.notnull(raw[feature])])
            columns.append([x if pandas.notnull(x) else avg for x in raw[feature]])
        elif f_type == InputType.LIST:
            columns += list_cols(feature)
        elif f_type == InputType.RANGE:
            valid_col = np.zeros(m)
            start_col = np.zeros(m)
            end_col = np.zeros(m)
            for R in raw[feature]:
                if not pandas.isnull(R):
                    start, end = [np.float32(x) for x in R.split('-')]
            columns += [valid_col, start_col, end_col]
    columns.append(np.ones(m))
    global patients
    patients = np.column_stack(columns)
    global d
    d = patients.shape[1]


# In[4]:


class DoseLevel(Enum):
    LOW = 0
    MED = 1
    HIGH = 2

def weekly_dose_to_level(dose):
    if dose < 21:
        return DoseLevel.LOW.value
    elif dose <= 49:
        return DoseLevel.MED.value
    else:
        return DoseLevel.HIGH.value
    
true_data = [weekly_dose_to_level(x) for x in raw[output_feature]]


# In[5]:


class Bandit:
    def pull(self, patient):
        pass
    
    def learn(self, patient, arm, reward):
        pass

class Fixed(Bandit):
    name = 'Fixed'
    
    def pull(self, patient):
        return DoseLevel.MED.value
        
def get_decade(age_str):
    return int(re.split(' - |\+', age_str)[0]) / 10

def non_null_values(feature):
    return raw[feature][pandas.notnull(raw[feature])]

mode_age = get_decade(stats.mode(non_null_values('Age'))[0][0])
avg_height = np.mean(non_null_values('Height (cm)'))
avg_weight = np.mean(non_null_values('Weight (kg)'))

def impute_age(age_str):
    return mode_age if pandas.isnull(age_str) else get_decade(age_str)

class Clinical(Bandit):
    name = 'Clinical'
    
    def pull(self, patient):
        sqrt_weekly_dose = 4.0376
        
        sqrt_weekly_dose -= 0.2546 * impute_age(raw['Age'][patient])
        
        height = raw['Height (cm)'][patient]
        sqrt_weekly_dose += 0.0118 * (avg_height if pandas.isnull(height) else height)
        
        weight = raw['Weight (kg)'][patient]
        sqrt_weekly_dose += 0.0134 * (avg_weight if pandas.isnull(weight) else weight)
        
        race = raw['Race'][patient]
        if (race == 'Asian'):
            sqrt_weekly_dose -= 0.6752
        elif (race == 'Black or African American'):
            sqrt_weekly_dose += 0.4060
        elif (race == 'Unknown'):
            sqrt_weekly_dose += 0.0443
        
        meds_str = raw['Medications'][patient]
        if pandas.notnull(meds_str):
            if enzyme_inducer_status(meds_str):
                sqrt_weekly_dose += 1.2799
            if amiodarone_status(meds_str):
                sqrt_weekly_dose -= 0.5695
        return weekly_dose_to_level(pow(sqrt_weekly_dose, 2))
    
class Pharmacogenetic(Bandit):
    name = 'Pharmacogenetic'
    
    def pull(self, patient):
        sqrt_weekly_dose = 5.6044
        
        sqrt_weekly_dose -= 0.2614 * impute_age(raw['Age'][patient])
        
        height = raw['Height (cm)'][patient]
        sqrt_weekly_dose += 0.0087 * (avg_height if pandas.isnull(height) else height)
        
        weight = raw['Weight (kg)'][patient]
        sqrt_weekly_dose += 0.0128 * (avg_weight if pandas.isnull(weight) else weight)
        
        vkor = raw['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'][patient]
        if pandas.notnull(vkor):
            if vkor == 'A/G': 
                sqrt_weekly_dose -= 0.8677
            elif vkor == 'A/A': 
                sqrt_weekly_dose -= 1.6974
        else:
            sqrt_weekly_dose -= 0.4854
            
        cyp = raw['Cyp2C9 genotypes'][patient]
        if pandas.notnull(cyp):
            if cyp == '*1/*2':
                sqrt_weekly_dose -= 0.5211
            elif cyp == '*1/*3':
                sqrt_weekly_dose -= 0.9357
            elif cyp == '*2/*2':
                sqrt_weekly_dose -= 1.0616
            elif cyp == '*2/*3':
                sqrt_weekly_dose -= 1.9206
            elif cyp == '*3/*3':
                sqrt_weekly_dose -= 2.3312    
        else:
            sqrt_weekly_dose -= 0.2188
            
        race = raw['Race'][patient]
        if (race == 'Asian'):
            sqrt_weekly_dose -= 0.1092
        elif (race == 'Black or African American'):
            sqrt_weekly_dose -= 0.2760
        elif (race == 'Unknown'):
            sqrt_weekly_dose -= 0.1032    
            
        meds_str = raw['Medications'][patient]
        if pandas.notnull(meds_str):
            if enzyme_inducer_status(meds_str):
                sqrt_weekly_dose += 1.1816
            if amiodarone_status(meds_str):
                sqrt_weekly_dose -= 0.5503
            
        return weekly_dose_to_level(pow(sqrt_weekly_dose, 2))


# In[6]:


K = 3

def get_features(p):
    return patients[p].reshape(d, 1)

class LinUCB(Bandit):
    name = 'LinUCB'
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.A = np.tile(np.identity(d), (K, 1, 1))
        self.b = np.zeros((K, d, 1))
    
    def pull(self, patient):
        max_p = None
        best_a = None
        x = get_features(patient)
        for a in range(K):
            theta = np.linalg.inv(self.A[a]).dot(self.b[a])
            p = theta.transpose().dot(x) + self.alpha * np.sqrt(x.transpose().dot(np.linalg.inv(self.A[a]).dot(x)))
            if best_a is None or p > max_p:
                max_p = p
                best_a = a
        return best_a
    
    def learn(self, patient, arm, reward):
        x = get_features(patient)
        self.A[arm] += x.dot(x.transpose())
        self.b[arm] += reward * x


# In[7]:


from sklearn import linear_model

class Lasso(Bandit):
    name = 'Lasso'
    
    def __init__(self, q, h, l1, l20):
        self.h = h
        self.l1 = l1
        self.l20 = l20
        
        self.t = 0
        self.forced_times = {}
        for n in range(np.int(np.log2(m / K / q)) + 2):
            for i in range(K):
                for j in range(q * i + 1, q * (i + 1) + 1):
                    t = (2 ** n - 1) * K * q + j
                    self.forced_times[t] = i
                    
        self.forced_sample_patients = [[] for _ in range(K)]
        self.forced_sample_rewards = [[] for _ in range(K)]
        self.all_sample_patients = [[] for _ in range(K)]
        self.all_sample_rewards = [[] for _ in range(K)]
    
    def pull(self, patient):
        if self.t == 0:
            return np.random.randint(K)
        x = patients[patient]
        if self.t in self.forced_times:
            return self.forced_times[self.t]
        estimates = np.empty(K)
        for a in range(K):
            model = linear_model.Lasso(alpha=self.l1, fit_intercept=False, max_iter=100000)
            model.fit(self.forced_sample_patients[a], self.forced_sample_rewards[a])
            estimates[a] = x.dot(model.coef_)
        max_est = np.max(estimates)
        best_a = None
        max_all_est = None
        for a in range(K):
            if estimates[a] >= max_est - self.h / 2:
                model = linear_model.Lasso(alpha=self.l20 * np.sqrt((np.log(self.t) + np.log(d)) / self.t), fit_intercept=False, max_iter=100000)
                model.fit(self.all_sample_patients[a], self.all_sample_rewards[a])
                all_est = x.dot(model.coef_)
                if best_a is None or all_est > max_all_est:
                    max_all_est = all_est
                    best_a = a
        return best_a
        
    def learn(self, patient, arm, reward):
        x = patients[patient]
        if self.t in self.forced_times:
            self.forced_sample_patients[arm].append(x)
            self.forced_sample_rewards[arm].append(reward)
        self.all_sample_patients[arm].append(x)
        self.all_sample_rewards[arm].append(reward)
        self.t += 1


# In[8]:


PRINT_FREQ = 250

def evaluate(bandit, verbose=True):
    h = []
    t = 0
    for p in np.random.permutation(m):
        if (verbose and t % PRINT_FREQ == 0):
            print(t, '/', m)
        dose = bandit.pull(p)
        reward = 0 if (dose == true_data[p]) else -1
        h.append(reward)
        bandit.learn(p, dose, reward)
        t += 1
    return h


# In[12]:


np.random.seed(100)

n_trials = 12

curr_features = []
all_histories = []
for new_features in [basic_features, medical_features, genome_features]:
    feature_hist = []
    curr_features += new_features
    set_patients(curr_features)
    for _ in range(n_trials):
        trial_hist = []
        for bandit in [Fixed(), Clinical(), Pharmacogenetic(), LinUCB(0.2), Lasso(1, 5, 0.05, 0.05)]:
            print(bandit.name)
            trial_hist.append(evaluate(bandit))
        feature_hist.append(trial_hist)
    all_histories.append(feature_hist)
np.save('all_hist6', all_histories)

