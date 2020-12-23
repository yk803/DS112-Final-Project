#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:15:59 2020

@author: Yukai Yang   yy2949
"""
#%%Q0: Preparation works
# Import all the packages we need.
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from simple_linear_regress_func import simple_linear_regress_func; # this one is from previous homework
from scipy import stats;
from sklearn.decomposition import PCA;
from sklearn.metrics import silhouette_samples;
import math;

# Open the file first
df = pd.read_csv("middleSchoolData.csv");


#%%Q1: find correlation between applications and acceptances
# Here we do a linear regression and return the r value.
ans1 = np.corrcoef(df["applications"],df["acceptances"])[1,0];
np_q1 = np.array([df["applications"],df["acceptances"]]);
np_q1 = np.transpose(np_q1);
# Find the regression line
para_q1 = simple_linear_regress_func(np_q1); # we can also get R-value from here, as para_q1[2] == R^2.
y_hat = df["applications"] * para_q1[0] + para_q1[1];

# Draw the graph
plt.plot(df["applications"],df["acceptances"],'o',markersize = 1.25);
plt.plot(df["applications"],y_hat,color = "orange",linewidth = 0.5);
plt.xlabel("applications nums");
plt.ylabel("acceptances nums");
plt.title("Q1: R={:.3f}".format(ans1));


#%%Q2: determine whether nums of applications or rate of applications better 
# predicts the admissions (acceptances).
# here we do two linear regression and compare the r^2 value (COD).
# the raw num one, we have done it in Q1.
# For the one asking for rate, there are two schools' data missing. 
# Since there are 594 schools in total, we just remove these two schools
raw_num_predict = ans1 ** 2; ## r^2 for nums of applications predict admissions

application_rates, acceptances = [],[];
for i in range(len(df)):
    application_rates.append(df["applications"][i]/df["school_size"][i]);
    acceptances.append(df["acceptances"][i]);
df_q2 = pd.DataFrame({"application_rates":application_rates,"acceptances":acceptances});
df_q2 = df_q2.dropna(); # remove the two NaN data only in this question
ans2 = np.corrcoef(df_q2["application_rates"],df_q2["acceptances"])[1,0];
rate_predict = ans2 ** 2; # again, use the r^2 value to compare
print("Q2: The r^2(COD) values here are {:.3f}, {:.3f}.".format(raw_num_predict,rate_predict));
print();


#%%Q3: sort the acceptance rates and return the highest one. Rates = acceptances / schoolsize
# Again, for the two schools whose school size data are missing, we remove them from our calculations.
per_odds = [];
for i in range(len(df)):
    per_odds.append([df["acceptances"][i]/df["school_size"][i],df["school_name"][i],i]);
cleaned_per_odds = [elem for elem in per_odds if str(elem[0]) != 'nan']; # we simply remove the 2 incomplete data
cleaned_per_odds.sort(); # and then sort it to find the "best" school
ans3 = cleaned_per_odds[-1][1];
print("Q3: {} has the best *per student* odds of sending someone to HSPHS.".format(ans3));
print();


#%%Q4:data here are too complicated to analyze directly, so a dimension reduction method, namely,
# PCA, must be applied first.
# Considering PCA method might be used later again, here we define a pca helper function
# so that we don't have to rewrite the codes again.
def pca_func(data):
    def remove_nan_np(data):
        i = 0; 
        while i < len(data):
            for elem in data[i]:
                if str(elem) == "nan":
                    data = np.delete(data,i,0);
                    i -= 1;
                    break
            i += 1;
        return data
    cleaned_data = remove_nan_np(data);
    zscored_data = stats.zscore(cleaned_data);
# do the calculation;
    pca = PCA();
    pca.fit(zscored_data);
# return the outputs;
    eig_vals = pca.explained_variance_;
    loadings = pca.components_;
    rotated_data = pca.fit_transform(zscored_data);
# covar_explained = eig_vals/sum(eig_vals)*100;
# scree plot:
    num_elems = len(eig_vals);
    plt.figure();
    plt.bar(np.linspace(1,num_elems,num_elems),eig_vals);
    plt.xlabel('Principal component');
    plt.ylabel('Eigenvalue');
# loadings index needs to be adjusted: what if there is more than 1 component that matters
    plt.figure();
    plt.bar(np.linspace(1,num_elems,num_elems),loadings[:,0]);
    plt.xlabel('Variable');
    plt.ylabel('Loading');
    
# return the reducted data
    #return np.dot(cleaned_data,loadings[:,0]);
    return rotated_data[:,0]

np_q4_1 = np.array([df["rigorous_instruction"],df["collaborative_teachers"],
                    df["supportive_environment"],df["effective_school_leadership"],
                    df["strong_family_community_ties"],df["trust"]]);
np_q4_2 = np.array([df["student_achievement"],df["reading_scores_exceed"],df["math_scores_exceed"]]);

np_q4_1 = np.transpose(np_q4_1);
np_q4_2 = np.transpose(np_q4_2);

pca_np_q4_1 = pca_func(np_q4_1);
pca_np_q4_2 = pca_func(np_q4_2);
# To make sure the cleaned data's sizes are the same, we unify their sizes once again.
def len_adjust_helper(nparray1,nparray2):
    if len(nparray1) == len(nparray2):
        return
    return nparray1[:min(len(nparray1),len(nparray2))],nparray2[:min(len(nparray1),len(nparray2))];

pca_np_q4_1,pca_np_q4_2 = len_adjust_helper(pca_np_q4_1,pca_np_q4_2);

ans4 = np.corrcoef(pca_np_q4_1,pca_np_q4_2)[1,0];
print("Q4: The r-value here is {:.3f}.".format(ans4));
print();


#%%Q5:This questions is a hypothesis testing one. 
# Here we test whether school size(large/small) has a significant effect on acceptance rates(high/low).
# That means, we need to change the numerical data in the file into categorical data
# so that we can run a chi-square test to test their independence.
school_size = np.array(df.school_size);
school_size = school_size[~np.isnan(school_size)];
plt.figure();
plt.hist(school_size);
plt.xlabel("school size");
acceptance_rates = per_odds;
acceptance_rates = [elem[0] for elem in per_odds if str(elem[0]) != 'nan']; #clean the data
plt.figure();
plt.hist(acceptance_rates);
plt.xlabel("rates");
# here by taking a look at the (frequency) distribution graph first, we can better know
# how to classify our numerical data into categorical groups. 
# We choose school_size 500 and accpetance_rates 0.05 here as the boundary.
np_q5 = np.array([school_size,acceptance_rates]);
np_q5 = np.transpose(np_q5);
obs_q5 = np.zeros((3,3));
for i in range(len(np_q5)):
    if np_q5[i][0] >= 500 and np_q5[i][1] >= 0.05:
        obs_q5[0][0] += 1;
    elif np_q5[i][0] >= 500 and np_q5[i][1] < 0.05:
        obs_q5[1][0] += 1;
    elif np_q5[i][0] < 500 and np_q5[i][1] >= 0.05:
        obs_q5[0][1] += 1;
    else:
        obs_q5[1][1] += 1;
for i in range(2):
    obs_q5[i][2] = obs_q5[i][0] + obs_q5[i][1];
    obs_q5[2][i] = obs_q5[0][i] + obs_q5[1][i];
obs_q5[2][2] = obs_q5[2][0]+obs_q5[2][1];

# Now we have got the obs table. Just need to get the exp table and then we can run the chi-square test.
exp_q5 = np.zeros((2,2));
for i in range(2):
    for j in range(2):
        exp_q5[i][j] = obs_q5[2,j]*obs_q5[i,2]/obs_q5[2,2];
lst_obs_q5 = [obs_q5[i][j] for i in range(2) for j in range(2)];
lst_exp_q5 = [exp_q5[i][j] for i in range(2) for j in range(2)];
print("Q5: {}".format(stats.chisquare(lst_obs_q5,f_exp=lst_exp_q5,ddof=2))); # RK: k-1-ddof= df! DDOF != D.F.!
print();


#%%Q6:
# It could be highly biased if we simply take all values from public school to infer
# data in charter schools. So (only) for this question, we will solely focus on public school
# data and completely ignore the charter school, to prevent greater biases.
        
np_q6 = np.array([df.per_pupil_spending,df.student_achievement,df.reading_scores_exceed,df.math_scores_exceed]);
np_q6 = np.transpose(np_q6);
np_q6 = np_q6[np_q6[:,0].argsort()];
# Notice that the per_pupil_spending col has numerical values until row 472. So we only keep them.
np_q6 = np_q6[:473];
# Now, only the student_achievement column have some missing values. We can draw the histogram
# and see that student_achievement's distribution is suitable for using mean imputation
for i in range(len(np_q6[:,1])):
    if str(np_q6[:,1][i]) == 'nan':
        np_q6[:,1][i] = np.mean(np_q6[:,1][~np.isnan(np_q6[:,1])]); #impute with mean

# Do the PCA for objective performances
pca_np_q6 = pca_func(np_q6[:,1:]);
spending = np_q6[:,0];

# Cut the result into 2 groups, so that one spends more than the other. We then test
# whether that has a significant effect on objective achievement
group1_q6 = np.array(pca_np_q6[:len(np_q6)//2]).transpose();
group2_q6 = np.array(pca_np_q6[len(np_q6)//2:]).transpose();

# and then we do a hypothesis test to see if there is an impact
t,p = stats.ttest_ind(group1_q6,group2_q6);
plt.plot(spending,pca_np_q6,'o');
plt.xlabel("per pupil spending");
plt.ylabel("objective achievement");
plt.title("Q6: p-value={:.3f}, R={:.3f}".format(p,np.corrcoef(spending,pca_np_q6)[0,1]));
print("Q6: t-statistics = {}, p-value = {}".format(t,p));
print();


#%%Q7:
np_q7 = np.array(df.acceptances).transpose();
np_q7 = np.sort(np_q7);
school_accounted = 0;
school_num = 0;
school_total = sum(elem for elem in np_q7);
for i in range(-1,-len(np_q7)-1,-1):
    if (np_q7[i] + school_num) / school_total < 0.9:
        school_num += np_q7[i];
        school_accounted += 1;
    else:
        break
ans_q7 = (school_accounted + 1)/ len(df); #need to add the last one to exceed 90%
print("Q7:", ans_q7);
print();


#%%Q8(0): Helper Functions and Preparations
from sklearn.experimental import enable_iterative_imputer;
from sklearn.impute import IterativeImputer;

# We want to construct our new df for deeper analysis. To do that, we need to create a 
# new dataframe to clean the data thoroughly.
# But before we really get started, it is necessary to build some helper functions to make
# our goals clearer.
def check_distribution():
    col_name = input("which column u wanna check? ");
    plt.figure();
    print(len(df[col_name].dropna()));
    plt.hist(df[col_name].dropna());

def imp_func(np1,np2):
    imp = IterativeImputer(max_iter=10,random_state=0);
    imp.fit([[np1[i],np2[i]] for i in range(len(np1))]);
    X_test = [[np1[i],np2[i]] for i in range(len(np1))];
    temp  = imp.transform(X_test);
    np2 = np.array([elem[1] for elem in temp]);
    return np2

def ethnicity_func(dfseries):
    new = np.array(dfseries);
    for i in range(len(dfseries)):
        if str(new[i]) == "nan":
            new[i] = np.median(dfseries.dropna());
    return new

# Compute correlation between each measure across all courses:
np_q8 = np.array(df.iloc[:,2:].dropna());
r = np.corrcoef(np_q8,rowvar=False);
# Plot the data:
plt.figure();
plt.imshow(r,cmap="seismic"); # Choose this color as it looks more explicit than others
plt.colorbar();


#%%Q8(1): Impute + no PCA at all
# Let's try no using PCA first.

df_q8 = pd.DataFrame();
df_q8["school_name"] = df.school_name;
df_q8["applications"] = df.applications;
df_q8["acceptances"] = df.acceptances;
df_q8["school_size"] = imp_func(df.applications,df.school_size);
df_q8["acceptance_rates"] = np.array([df.acceptances[i]/df_q8.school_size[i] for i in range(len(df.acceptances))]);
# From now on we cannot simply copy the data from the original df. 
# More processes, regression, PCA, etc. is needed to simplify the dataframe.
# Ethnicity
df_q8["asian_percent"] = ethnicity_func(df.asian_percent);
df_q8["black_percent"] = ethnicity_func(df.black_percent);
df_q8["hispanic_percent"] = ethnicity_func(df.hispanic_percent);
df_q8["multiple_percent"] = ethnicity_func(df.multiple_percent);
df_q8["white_percent"] = ethnicity_func(df.white_percent);
# School environment: We impute data with IterativeImputer, but without doing PCA.
# With that said, here we still need to find a column that is relatively highly correlated
# with these factors to use imp_func. Obviously the best choice is among these six columns. 
# So we sum the r in each column to see which one is better.
# Obviously the best choice is among these six columns. So we sum the r in each column to see which one is better.
r_sum = [(sum(r[i][j] for i in range(9,15)) - 1)for j in range(9,15)];
# We find that the second factor -- collaborative_teachers is the best choice.
np_q8 = np_q4_1;
for i in range(int(np.size(np_q4_1)/len(np_q4_1))):
    np_q8[:,i] = imp_func(df.collaborative_teachers,np_q8[:,i]);
    df_q8[df.columns[11+i]] = np_q8[:,i];

# Student performance and "negative" factors
np_q8 = np.array([df.student_achievement,df.reading_scores_exceed,df.math_scores_exceed]);
np_q8 = np.transpose(np_q8);
np_q8[:,0] = imp_func(df.poverty_percent,np_q8[:,0]);
np_q8[:,1] = imp_func(df.poverty_percent,np_q8[:,1]);
np_q8[:,2] = imp_func(df.poverty_percent,np_q8[:,2]);
df_q8["student_achievement"] = np_q8[:,0];
df_q8["reading_scores_exceed"] = np_q8[:,1];
df_q8["maths_scores_exceed"] = np_q8[:,2];

np_q8 = np.array([df.disability_percent,df.poverty_percent,df.ESL_percent]);
np_q8 = np.transpose(np_q8);
np_q8[:,0] = imp_func(df_q8.reading_scores_exceed,np_q8[:,0]);
np_q8[:,1] = imp_func(df_q8.reading_scores_exceed,np_q8[:,1]);
np_q8[:,2] = imp_func(df_q8.reading_scores_exceed,np_q8[:,2]);
df_q8["disability_percent"] = np_q8[:,0];
df_q8["poverty_percent"] = np_q8[:,1];
df_q8["ESL_percent"] = np_q8[:,2];

# Now we can draw a new bar graph. 
# Acceptance rates is in column 3(index 3 not third column)
# and student achievement is in column 9.
r = np.corrcoef(df_q8.iloc[:,1:],rowvar=False);
# Plot the data:
plt.figure();
plt.imshow(r,cmap='seismic'); 
plt.colorbar();


#%%Q8(2): Impute + PCA

# Now we are ready to build our new dataframe.
df_q8_2 = pd.DataFrame();
df_q8_2["school_name"] = df_q8.school_name;
df_q8_2["applications"] = df_q8.applications;
df_q8_2["acceptances"] = df_q8.acceptances;
df_q8_2["school_size"] = df_q8.school_size;
df_q8_2["acceptance_rates"] = df_q8.acceptance_rates;
# From now on we cannot simply copy the data from the original df. 
# More processes, regression, PCA, etc. is needed to simplify the dataframe.
# Ethnicity
df_q8_2["asian_percent"] = df_q8.asian_percent;
df_q8_2["black_percent"] = df_q8.black_percent;
df_q8_2["hispanic_percent"] = df_q8.hispanic_percent;
df_q8_2["multiple_percent"] = df_q8.multiple_percent;
df_q8_2["white_percent"] = df_q8.white_percent;
# Student performance and "negative" factors
np_q8 = np.array([df.student_achievement,df.reading_scores_exceed,df.math_scores_exceed]);
np_q8 = np.transpose(np_q8);
np_q8[:,0] = imp_func(df.poverty_percent,np_q8[:,0]);
np_q8[:,1] = imp_func(df.poverty_percent,np_q8[:,1]); #### ?
np_q8[:,2] = imp_func(df.poverty_percent,np_q8[:,2]); #### which col to iteratively impute with???
pca_np_q8 = pca_func(np_q8);
df_q8_2["achievement"] = pca_np_q8;
np_q8 = np.array([df.disability_percent,df.poverty_percent,df.ESL_percent]);
np_q8 = np.transpose(np_q8);
np_q8[:,0] = imp_func(df_q8_2.achievement,np_q8[:,0]);
np_q8[:,1] = imp_func(df_q8_2.achievement,np_q8[:,1]);
np_q8[:,2] = imp_func(df_q8_2.achievement,np_q8[:,2]);
pca_np_q8 = pca_func(np_q8);
df_q8_2["negative factors"] = pca_np_q8;
# School environment
np_q8 = np_q4_1;
for i in range(int(np.size(np_q4_1)/len(np_q4_1))):
    np_q8[:,i] = imp_func(df.collaborative_teachers,np_q8[:,i]);
pca_np_q8 = pca_func(np_q8);
df_q8_2["environment"] = pca_np_q8;

# Now we can draw a new bar graph. 
# Acceptance rates is in column 3(index 3 not third column)
# and student achievement is in column 9.
r = np.corrcoef(df_q8_2.iloc[:,1:],rowvar=False);
# Plot the data:
plt.figure();
plt.imshow(r,cmap='seismic'); 
plt.colorbar();


#%%Q8(3): Impute + PCA(partly) 
# From previous results, we see that disability,poverty, and esl really matter
# while school environment does not. So here we copy most of our data from previous works,
# just one difference: we don't do PCA to disability, poverty, and esl. Let's see if that
# would differentiate their effects to our objectives: acceptance_rates and achievements
df_q8_3 = pd.DataFrame();
for i in range(0,11):
    df_q8_3[df_q8_2.columns[i]] = df_q8_2[df_q8_2.columns[i]];
df_q8_3["disability_percent"] = df_q8["disability_percent"];
df_q8_3["poverty_percent"] = df_q8["poverty_percent"];
df_q8_3["ESL_percent"] = df_q8["ESL_percent"];

r = np.corrcoef(df_q8_3.iloc[:,1:],rowvar=False);
# Plot the data:
plt.figure();
plt.imshow(r,cmap='seismic'); 
plt.colorbar();

#%%Q8(4): Clustering
from kmeans_clustering import kmeans_clustering; 
# a clustering function. It's a bit long so we write it in another file

temp_data = np.array([df_q8.asian_percent,df_q8.acceptance_rates]).transpose();
temp_result = kmeans_clustering(temp_data);




#%%
# For acceptance rates, number of acceptances, asian percent, and negative factor precent matter the most.
# For student achievement, "negative" factors matter the most.

# plt.figure();
# plt.plot(pca_np_q4_1,pca_np_q4_2,'o',markersize=5);
# plt.xlabel("school_values");
# plt.ylabel("student_achievements");
#np_q8 = np.array([pca_np_q4_1,pca_np_q4_2]);
#np_q8 = np.transpose(np_q8);
#kmeans = KMeans(n_clusters=2,random_state=0).fit(np_q8);
#label = kmeans.fit_predict(np_q8);
#centeroids = kmeans.cluster_centers_;
#u_labels = np.unique(label);
#plt.figure();
#for i in u_labels:
#    plt.scatter(np_q8[label==i,0],np_q8[label==i,1],label=i);
#plt.legend();
#plt.show();
#xyz_q8 = silhouette_samples(np_q8,label);
#print(sum(elem for elem in xyz_q8));