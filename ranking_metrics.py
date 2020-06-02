#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import json
import numpy as np
from scipy import spatial


# In[8]:


val = pd.read_csv('/srv/share/sameer/pythia_results/model_0.33_0.75/csv_results/val_three_similarities.csv')


# In[9]:


val = val.sort_values(['image_id', 'reasoning_question_id', 'similarity_1'], ascending=(True, True, False))


# In[10]:


groups = val.groupby(['image_id', 'reasoning_question_id'])


# In[11]:


print("Computing metrics with respect to Similarity 1")


# In[12]:


recall_1 = 0
for name, group in groups:
    #sq_count = 0
    #print(name)
    #print(group['question_type'][0])
    #for i in group.index.values.tolist():
    first_index = group.index.values.tolist()[0]
    if group['question_type'][first_index] == 'sub_question':
        recall_1 += 1
        
            #sq_count += 1
#print(sq_count)
#print("Total count : {}".format(recall_1))
recall_1 = recall_1/len(groups)
print("Recall 1 value : {}".format(recall_1))


# In[13]:


recall_k = 0
for name, group in groups:
    #print(len(group))
    sq_count = 0
    sq_order = 0
    first_index = group.index.values.tolist()[0]
    for i in group.index.values.tolist():
        if group['question_type'][i] == 'sub_question':
            sq_count += 1
            
    #for j in range(first_index, (first_index+sq_count)):
    for j in group.index.values.tolist()[:sq_count]:
        if group['question_type'][j] == 'sub_question':
            sq_order += 1
        else:
            break
    #print("SQ count : {}".format(sq_count))
    #print("J: {}".format(j))
    #print("First index : {}".format(first_index))
    if sq_order == sq_count:
        recall_k += 1
        
#print(recall_k)
#print("Total count : {}".format(recall_k))
recall_k = recall_k/len(groups)
print("Recall k value : {}".format(recall_k))


# In[14]:


sum_reciprocal_rank = 0
mean_reciprocal_rank = 0
for name, group in groups:
    #print(len(group))
    reciprocal_rank = 0
    for idx, index in enumerate(group.index.values.tolist()):
        if group['question_type'][index] == 'sub_question':
            reciprocal_rank = 1/(idx+1)
            sum_reciprocal_rank += reciprocal_rank
            break

mean_reciprocal_rank = sum_reciprocal_rank/len(groups)
#print("Sum reciprocal rank : {}".format(sum_reciprocal_rank))
print("Mean reciprocal rank : {}".format(mean_reciprocal_rank))


# In[15]:


mean_average_precision = 0
average_precisions = []
for name, group in groups:
    #print(len(group))
    reciprocal_ranks = []
    for idx, index in enumerate(group.index.values.tolist()):
        correct_answers = 0
        if group['question_type'][index] == 'sub_question':
            correct_answers += 1
            reciprocal_ranks.append(correct_answers/(idx+1))
    group_average_precision = sum(reciprocal_ranks)/len(reciprocal_ranks)
    average_precisions.append(group_average_precision)

mean_average_precision = sum(average_precisions)/len(average_precisions)
print("Mean average precision : {}".format(mean_average_precision))


# In[16]:


warp = 0
group_warp = []
for name, group in groups:
    #print(len(group))
    ground_truth_order_sub_questions = []
    ground_truth_order_other_questions = []
    ground_truth_order = []
    group['question_type_index'] = 0
    for i in group.index.values.tolist():
        if group['question_type'][i] == 'sub_question':
            group['question_type_index'][i] = 1
            ground_truth_order_sub_questions.append((1, group['similarity_1'][i]))
        else:
            ground_truth_order_other_questions.append((0, group['similarity_1'][i]))

    ground_truth_order = ground_truth_order_sub_questions + ground_truth_order_other_questions
    #print(ground_truth_order)

    differences = []

    for idx, index in enumerate(group.index.values.tolist()):
        if group['question_type_index'][index] != ground_truth_order[idx][0]:
            differences.append(abs(ground_truth_order[idx][1] - group['similarity_1'][index]))
    if len(differences) > 0:
        group_warp.append(sum(differences))
        #group_warp.append(sum(differences)/len(differences))

warp = sum(group_warp)/len(group_warp)
print("WARP loss : {}".format(warp))


# In[17]:


val = val.sort_values(['image_id', 'reasoning_question_id', 'similarity_2'], ascending=(True, True, False))


# In[18]:


groups = val.groupby(['image_id', 'reasoning_question_id'])


# In[19]:


print("Computing metrics with respect to Similarity 2")


# In[20]:


recall_1 = 0
for name, group in groups:
    #sq_count = 0
    #print(name)
    #print(group['question_type'][0])
    #for i in group.index.values.tolist():
    first_index = group.index.values.tolist()[0]
    if group['question_type'][first_index] == 'sub_question':
        recall_1 += 1
        
            #sq_count += 1
#print(sq_count)
#print("Total count : {}".format(recall_1))
recall_1 = recall_1/len(groups)
print("Recall 1 value : {}".format(recall_1))


# In[21]:


recall_k = 0
for name, group in groups:
    #print(len(group))
    sq_count = 0
    sq_order = 0
    first_index = group.index.values.tolist()[0]
    for i in group.index.values.tolist():
        if group['question_type'][i] == 'sub_question':
            sq_count += 1
            
    #for j in range(first_index, (first_index+sq_count)):
    for j in group.index.values.tolist()[:sq_count]:
        if group['question_type'][j] == 'sub_question':
            sq_order += 1
        else:
            break
    #print("SQ count : {}".format(sq_count))
    #print("J: {}".format(j))
    #print("First index : {}".format(first_index))
    if sq_order == sq_count:
        recall_k += 1
        
#print(recall_k)
#print("Total count : {}".format(recall_k))
recall_k = recall_k/len(groups)
print("Recall k value : {}".format(recall_k))


# In[22]:


sum_reciprocal_rank = 0
mean_reciprocal_rank = 0
for name, group in groups:
    #print(len(group))
    reciprocal_rank = 0
    for idx, index in enumerate(group.index.values.tolist()):
        if group['question_type'][index] == 'sub_question':
            reciprocal_rank = 1/(idx+1)
            sum_reciprocal_rank += reciprocal_rank
            break

mean_reciprocal_rank = sum_reciprocal_rank/len(groups)
#print("Sum reciprocal rank : {}".format(sum_reciprocal_rank))
print("Mean reciprocal rank : {}".format(mean_reciprocal_rank))


# In[23]:


mean_average_precision = 0
average_precisions = []
for name, group in groups:
    #print(len(group))
    reciprocal_ranks = []
    for idx, index in enumerate(group.index.values.tolist()):
        correct_answers = 0
        if group['question_type'][index] == 'sub_question':
            correct_answers += 1
            reciprocal_ranks.append(correct_answers/(idx+1))
    group_average_precision = sum(reciprocal_ranks)/len(reciprocal_ranks)
    average_precisions.append(group_average_precision)

mean_average_precision = sum(average_precisions)/len(average_precisions)
print("Mean average precision : {}".format(mean_average_precision))


# In[24]:


warp = 0
group_warp = []
for name, group in groups:
    #print(len(group))
    ground_truth_order_sub_questions = []
    ground_truth_order_other_questions = []
    ground_truth_order = []
    group['question_type_index'] = 0
    for i in group.index.values.tolist():
        if group['question_type'][i] == 'sub_question':
            group['question_type_index'][i] = 1
            ground_truth_order_sub_questions.append((1, group['similarity_2'][i]))
        else:
            ground_truth_order_other_questions.append((0, group['similarity_2'][i]))

    ground_truth_order = ground_truth_order_sub_questions + ground_truth_order_other_questions
    #print(ground_truth_order)

    differences = []

    for idx, index in enumerate(group.index.values.tolist()):
        if group['question_type_index'][index] != ground_truth_order[idx][0]:
            differences.append(abs(ground_truth_order[idx][1] - group['similarity_2'][index]))
    if len(differences) > 0:
        group_warp.append(sum(differences))
        #group_warp.append(sum(differences)/len(differences))

warp = sum(group_warp)/len(group_warp)
print("WARP loss : {}".format(warp))


# In[25]:


val = val.sort_values(['image_id', 'reasoning_question_id', 'similarity_3'], ascending=(True, True, False))


# In[26]:


groups = val.groupby(['image_id', 'reasoning_question_id'])


# In[27]:


print("Computing metrics with respect to Similarity 3")


# In[28]:


recall_1 = 0
for name, group in groups:
    #sq_count = 0
    #print(name)
    #print(group['question_type'][0])
    #for i in group.index.values.tolist():
    first_index = group.index.values.tolist()[0]
    if group['question_type'][first_index] == 'sub_question':
        recall_1 += 1
        
            #sq_count += 1
#print(sq_count)
#print("Total count : {}".format(recall_1))
recall_1 = recall_1/len(groups)
print("Recall 1 value : {}".format(recall_1))


# In[29]:


recall_k = 0
for name, group in groups:
    #print(len(group))
    sq_count = 0
    sq_order = 0
    first_index = group.index.values.tolist()[0]
    for i in group.index.values.tolist():
        if group['question_type'][i] == 'sub_question':
            sq_count += 1
            
    #for j in range(first_index, (first_index+sq_count)):
    for j in group.index.values.tolist()[:sq_count]:
        if group['question_type'][j] == 'sub_question':
            sq_order += 1
        else:
            break
    #print("SQ count : {}".format(sq_count))
    #print("J: {}".format(j))
    #print("First index : {}".format(first_index))
    if sq_order == sq_count:
        recall_k += 1
        
#print(recall_k)
#print("Total count : {}".format(recall_k))
recall_k = recall_k/len(groups)
print("Recall k value : {}".format(recall_k))


# In[30]:


sum_reciprocal_rank = 0
mean_reciprocal_rank = 0
for name, group in groups:
    #print(len(group))
    reciprocal_rank = 0
    for idx, index in enumerate(group.index.values.tolist()):
        if group['question_type'][index] == 'sub_question':
            reciprocal_rank = 1/(idx+1)
            sum_reciprocal_rank += reciprocal_rank
            break

mean_reciprocal_rank = sum_reciprocal_rank/len(groups)
#print("Sum reciprocal rank : {}".format(sum_reciprocal_rank))
print("Mean reciprocal rank : {}".format(mean_reciprocal_rank))


# In[31]:


mean_average_precision = 0
average_precisions = []
for name, group in groups:
    #print(len(group))
    reciprocal_ranks = []
    for idx, index in enumerate(group.index.values.tolist()):
        correct_answers = 0
        if group['question_type'][index] == 'sub_question':
            correct_answers += 1
            reciprocal_ranks.append(correct_answers/(idx+1))
    group_average_precision = sum(reciprocal_ranks)/len(reciprocal_ranks)
    average_precisions.append(group_average_precision)

mean_average_precision = sum(average_precisions)/len(average_precisions)
print("Mean average precision : {}".format(mean_average_precision))


# In[32]:


warp = 0
group_warp = []
for name, group in groups:
    #print(len(group))
    ground_truth_order_sub_questions = []
    ground_truth_order_other_questions = []
    ground_truth_order = []
    group['question_type_index'] = 0
    for i in group.index.values.tolist():
        if group['question_type'][i] == 'sub_question':
            group['question_type_index'][i] = 1
            ground_truth_order_sub_questions.append((1, group['similarity_3'][i]))
        else:
            ground_truth_order_other_questions.append((0, group['similarity_3'][i]))

    ground_truth_order = ground_truth_order_sub_questions + ground_truth_order_other_questions
    #print(ground_truth_order)

    differences = []

    for idx, index in enumerate(group.index.values.tolist()):
        if group['question_type_index'][index] != ground_truth_order[idx][0]:
            differences.append(abs(ground_truth_order[idx][1] - group['similarity_3'][index]))
    if len(differences) > 0:
        group_warp.append(sum(differences))
        #group_warp.append(sum(differences)/len(differences))

warp = sum(group_warp)/len(group_warp)
print("WARP loss : {}".format(warp))

