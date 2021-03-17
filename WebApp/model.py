import pandas as pd 
import pickle

data = pd.read_csv("salarydata.csv")

print(data.head)
print(data.columns)

#missing values
data['workclass']=data['workclass'].replace('?','Private')
data['occupation']=data['occupation'].replace('?','Prof-specialty')
data['native-country']=data['native-country'].replace('?','United-States')

print(data.head)
print(data.columns)

#Dropping unnecessary columns
data=data.drop(['education'],axis=1)

#Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in ['workclass', 'marital-status','occupation', 'relationship', 'race','native-country', 'salary','sex']:
    data[i] = label_encoder.fit_transform(data[i])



#splitting

X=data.drop(['salary'],axis=1)
y=pd.DataFrame(data['salary'])

print('X')
print(X.head())

print('y')
print(y.head())


#Std Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

print(X)


#modeling
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(max_depth=5,n_estimators=130,random_state=10,max_features='auto',min_samples_split=2,learning_rate=0.1,
                              min_samples_leaf=1,subsample=1.0)


model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))