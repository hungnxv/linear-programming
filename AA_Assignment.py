from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import os.path as osp
from pulp import GLPK
import pandas as pd
import math
import glob

def dataPath(week_name = "Week1"):
    root = "./data/"
    target_path_list = osp.join(root + week_name + "/*.csv")
    path_list = []
    for path in glob.glob(target_path_list):
        path_list.append(path)
    return sorted(path_list)

def extractScoresFromData(file_name):
    dataFrames = pd.read_csv(file_name, sep = ',')
    scores = dataFrames[:-1].loc[:, 'Q. 1 /1,00':'Q. 10 /1,00']
    scores = scores.replace('-', '0').apply(lambda x: x.str.replace(',','.'))[:].astype(float)
    scores.columns = [str(i) for i in range(1, 11)]
    return scores

def getDecisionVariable(file_name, k_partition = 4):
    scoreOfStudents = extractScoresFromData(file_name)
    numberOfStudents = scoreOfStudents.count()['1']
    
    #number of group
    numberOfGroups = math.ceil(numberOfStudents / k_partition)
    x = {}
    for i in range(0, numberOfStudents):
        x[i] = {}
        for j in range(0, numberOfGroups):
            x[i][j] = LpVariable('x_'+str(i)+','+str(j),cat="Binary")
    c = {}
    for j in range(0, numberOfGroups):
        c[j]={}
        for t in [str(i) for i in range(1,11)]:
            c[j][t]=LpVariable('c_'+str(j)+','+t,lowBound=0)
    return x, c, numberOfGroups, numberOfStudents, scoreOfStudents

def optimizationLinearProgramming(file_name, k_partition = 4):
    x, c, groupNumber, studentNumber, scores = getDecisionVariable(file_name, k_partition)
    prob = LpProblem("Mixed_Problem", LpMaximize)

    for j in range(0, groupNumber):
        # Excercise related constraints:
        for t in [str(i) for i in range(1,11)]:
            # c[j,t]=min(10, sum(x[i,j]*score[i,t]))
            prob += c[j][t] <= 10 
            # c[j,t]=min(10, sum(x[i,j]*score[i,t]))
            prob += c[j][t] <= lpSum([x[i][j] * scores.loc[i][t] for i in range(0, studentNumber)])

        # Each group has exactly k students or lastGroupNumber if it is the last group
        if j < groupNumber - 1:
            prob += lpSum([x[i][j] for i in range(0, studentNumber)]) == k_partition 
        else:
            lastGroupNumber = studentNumber - k_partition * (groupNumber - 1)
            prob += lpSum([x[i][j] for i in range(0,studentNumber)]) == lastGroupNumber 
        
    for i in range(0,studentNumber):
        # Each student belongs to only one group
        prob += lpSum([x[i][j] for j in range(0, groupNumber)]) == 1
       
    prob += lpSum([cjt for cj in c.values() for cjt in cj.values()])

    prob.solve(solver=GLPK(msg=False))
    return x, prob.objective.value(), groupNumber, studentNumber

def show_optimal_solution(file_name, k_partition = 5):
    dataFrames = pd.read_csv(file_name, sep=',')
    students = dataFrames[:-1].loc[:,'Tên': 'Họ']
    x, _, numberOfGroups, numberOfStudents = optimizationLinearProgramming(file_name, k_partition)
    solution = {}
    solutionFrame = {}
    for i in range(0, numberOfStudents):
        for j in range(0, numberOfGroups):
            if not(j in solution):
                solution[j]=[]
                solutionFrame[j] = []
            if x[i][j].value() == 1:
                solution[j].append((i, students.loc[i]['Họ']+ ' ' + students.loc[i]['Tên']))
                solutionFrame[j].append(students.loc[i]['Họ']+ ' ' + students.loc[i]['Tên'])
                break
    dfObj = pd.DataFrame.from_dict(solutionFrame, orient = 'index')
    print(dfObj)

def assignmentResult(week_name):
    for i in range(len(dataPath(week_name))):
        print("Class: " + str(i + 1))
        show_optimal_solution(dataPath(week_name)[i])
        print()
        
if __name__ == "__main__":
    assignmentResult("Week1")
    


    


