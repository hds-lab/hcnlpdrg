import pandas as pd
import numpy as np

#function that count the number of agreements of coder
#df: dataframe of data
#n_codes: number of avalible codes
def calc_agrement(df, n_codes):
    #group by coded text by text_d
    text_grouped = df.groupby("text_id")
    #array that count the agrements for each code
    codes_agrement = np.zeros(n_codes)

    for name, text_group in text_grouped:
        #count the number of each code for the text regardless the coder and the number of coders
        code_counts = text_group["code_id"].value_counts()
        number_raters = len(text_group["coder_id"].unique())
        #for each code if are agrement
        for i in range(1,n_codes+1):
            #check if the code was ues
            if i in code_counts.index:
                #check if more than half agreed or all agreed
                if np.ceil(number_raters/2)  < code_counts[i] or number_raters == code_counts[i]:
                    codes_agrement[i-1] += 1
            # the code was not used by any coder, therfore it's agreement
            else:
                codes_agrement[i-1] += 1
                
    return codes_agrement.sum()

#function that generate simulation on codings
#prob_matrix: array of matrix of relative probability of a code is selected when a certain numbers of code are used by each coder
#n_coding: array of arrays that count how many times coder select a certain number of codes for a text
#n_text: number total of text
#n_codes: number or codes avalible
def simulate_coding(prob_matrix, n_coding, n_text, n_codes):
    #transform the counters to probabilites
    n_codes_prob = n_coding/n_text
    #array that save the agreement in the fake text
    sum_text_codes = np.zeros(n_codes)
    #counter for each code avalible
    agrements = np.zeros(n_codes)
    n_coders = 0
    for coder in range(prob_matrix.shape[0]):
        n_coders += 1
        #select randomly the number of code that will use the coder
        n_fake_codes = np.random.choice(n_codes+1,p = n_codes_prob[coder])
        
        if n_fake_codes == 0:
            continue
        #calculate the probabilty for each code when a n_fakes_codes are used
        prob_fake_codes = prob_matrix[coder][n_fake_codes-1]/(n_coding[coder][n_fake_codes] * n_fake_codes)
        #select n_fakes_codes randomly
        fake_codings = np.random.choice(n_codes,n_fake_codes, replace=False,p=prob_fake_codes)
        #add the fake_codes to the counter
        for code in fake_codings:
            sum_text_codes[code] += 1
    #checj if there is agreement of the fake codes
    for i, code in enumerate(sum_text_codes):
        if code > np.ceil(n_coders/2) or code == n_coders or code == 0:
            agrements[i] = 1
    return agrements




#load dataset
#is asumed that the dataset have the next headers, text_id, coder_id, code_id 
dataset = pd.read_csv('topic_codings.csv', sep=",")

#group the data by coder
coder_grouped = dataset.groupby("coder_id" )
#count the total of text
n_text = len(dataset["text_id"].unique())

#number of posibles codes
n_codes = 13

#array of matrix of relative probability of a code is selected when a certain numbers of code are used by each coder
prob_matrix = None
#array of arrays that count how many times coder select a certain number of codes for a text
n_coding = None

for name, coder_group in coder_grouped:
    # create matrix and array for each coder
    prob_matrix_coder = np.zeros((n_codes, n_codes))
    n_coding_coder = np.zeros((1, n_codes+1))
    #group the text of each coder by text_id
    text_grouped = coder_group.groupby("text_id")
    
    for _, text_group in text_grouped:
        #count the number of diferent codes used in a text and register it in the array
        n_codes_coder = text_group["coder_id"].size
        n_coding_coder[0][n_codes_coder] += 1
        # add to the matrix the codes
        for code_id in text_group["code_id"]:
            prob_matrix_coder[n_codes_coder-1][code_id-1] +=1
    # add the number of times that the user didn't add a code to a text
    n_coding_coder[0][0] = n_text - n_coding_coder[0].sum()
    
    #save the matrix and array of the user in their respective arrays
    if prob_matrix is None:
        prob_matrix = prob_matrix_coder.reshape((1,n_codes,n_codes))
        n_coding = n_coding_coder
    else:
        prob_matrix = np.concatenate((prob_matrix, prob_matrix_coder.reshape(1,n_codes,n_codes)))
        n_coding = np.concatenate((n_coding, n_coding_coder))

#auxliar variables
prev_p_agrement_chance = 0.
new_p_agrement_chance = 3.
#counter of fake_texts
n_fake_texts = 0
#counter the number of agrement by code
agrements = np.zeros(n_codes)
#generate simulation until the agrement chance variation ins less than 0.0001 
while(np.abs(prev_p_agrement_chance-new_p_agrement_chance) > 0.0001):
    #save the number of agreement in the simulation
    agrements += simulate_coding(prob_matrix, n_coding, n_text, n_codes)
    n_fake_texts += 1
    #update the aux variable
    prev_p_agrement_chance = new_p_agrement_chance
    new_p_agrement_chance = agrements.sum()/(n_codes * n_fake_texts)


P_o = calc_agrement(dataset,n_codes)/(n_codes * n_text)
P_e = new_p_agrement_chance

print("Agreement Observed (%) (P_o):", P_o)
print("Probability of Agreement Chance (P_e):", P_e)
print("Nro of simulations:", n_fake_texts)
print("Extended Kappa:", 1-(1-P_o)/(1.-P_e))
