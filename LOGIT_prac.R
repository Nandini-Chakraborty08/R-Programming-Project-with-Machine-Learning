#------------------->Basic Functional Form:
#P(Y=1)=e^Z/(1+e^Z), e refers to exponential
#where Z=B0+B1X1+B2X2+..........+BNXN


#Problem Statement: 
#To predict which customers are more probable to churn (Y=1), based on the attributes to the customer

#packages required for LOGISTIC REGRASSION
install.packages("mlogit", dependencies = TRUE )
library(mlogit) #LOGISTIC MODEL
library(caret)# LOGISTIC MODEL
library(ggplot2)# VISUALIZATION
library(MASS)# VIF CALCULATION
library(car) #VIF CALCULATION
install.packages("sqldf", dependencies = TRUE )
library(sqldf) #WOE & IV
install.packages("Hmisc", dependencies = TRUE )
library(Hmisc)#WOE & IV
library(caTools)#TRAIN AND TEST SPLIT
install.packages("aod", dependencies = TRUE )
library(aod) #WALD TEST
install.packages("BaylorEdPsych", dependencies = TRUE )
library(BaylorEdPsych) #R square
install.packages("ResourceSelection", dependencies = TRUE )
library(ResourceSelection) #HOSMER LEMESHOW TEST
library(pROC) #ROC curve 
install.packages("ROCR", dependencies = TRUE )
library(ROCR)


##setting working directory 
path<-setwd("E:/IVY 2021-22/STAT + R SESSIONS/DATA/LOGIT")
getwd
data = read.csv("Fn-UseC_-Telco-Customer-Churn.csv",header = TRUE)
data1=data  #To create a backup of original data
head(data1) #creating the head of the data


#BASIC EXPLORATION OF THE DATA
str(data1) #this shows most of the variables are character in nature which should be categorical
summary(data1) #here for the most variables we can't check the actual levels of the charterer variables
dim(data1)
#Changing into factor from the character datatype columns 
cols_cat<-c("gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","TechSupport")
data1[cols_cat]<-lapply(data1[cols_cat],factor)# all columns mentioned in the previous code will get converted into factors
str(data1)
summary(data1) #now can check frequency distribution for all categorical variables, 

#MISSING VALUE TREATNENT IF ANY
data.frame(colSums(is.na(data1))) ##total charges numerical variable has 11 missing value and will impute with mean. 

#Substituting missing values with mean
data1[is.na(data1$TotalCharges),19]=mean(data1$TotalCharges,na.rm=T)
data.frame(colSums(is.na(data1))) #no missing values now

#Splitting the data into training and test data set, build the model on training and validate in test data
set.seed(144) #This is used to produce reproducible results, every time we run the model

spl = sample.split(data1$Churn, 0.7) #keeping 70% of the data into the training portion
data.train = subset(data1, spl == TRUE)
str(data.train)
dim(data.train) #4930 observations are in my training 

data.test = subset(data1, spl == FALSE)
str(data.test)
dim(data.test) #2113 observations in the test data

#Logistic Regression Model Building
model <- glm(Churn~., data=data.train, family=binomial()) #building the logit model
summary(model)


## Remove the insignificant variable
#iteration 1
model <- glm(Churn~ gender +	SeniorCitizen + Partner + Dependents + tenure +	PhoneService + MultipleLines
             + InternetService + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV 
             + StreamingMovies + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges 
             + TotalCharges, data=data.train, family=binomial())
summary(model)

#iteration 2
model <- glm(Churn~ gender +	SeniorCitizen + Partner + Dependents + tenure +	PhoneService 
             + I(MultipleLines=="Yes")+ InternetService + I(OnlineSecurity=="Yes")+ I(OnlineBackup=="Yes") 
             + I(DeviceProtection=="Yes") + I(TechSupport=="Yes") + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges +	TotalCharges 
             , data=data.train, family=binomial())
summary(model)

#iteration 3
model <- glm(Churn~ 	SeniorCitizen  + tenure + Dependents
             + I(MultipleLines=="Yes")+ InternetService + I(OnlineSecurity=="Yes")+ I(OnlineBackup=="Yes") 
             + I(DeviceProtection=="Yes") + I(TechSupport=="Yes") + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges +	TotalCharges 
             , data=data.train, family=binomial())
summary(model)

#iteration 4
model <- glm(Churn~ 	SeniorCitizen  + tenure 
             + I(MultipleLines=="Yes")+ InternetService   +  I(TechSupport=="Yes") + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + PaymentMethod +	MonthlyCharges +	TotalCharges 
             , data=data.train, family=binomial())
summary(model)

#iteration 5
model <- glm(Churn~ 	SeniorCitizen  + tenure 
             + I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
             + I(StreamingMovies=="Yes") + Contract + PaperlessBilling + I(PaymentMethod=="Electronic check")+	tenure , data=data.train, family=binomial())
summary(model) #####FINAL MODEL WITH 12 VARIABLES#####

##VALIDATION CHECKS:
vif(model) #checks multicolinearity (there is not in the model)


#Checking the overall fitness of the model
#using Wald Test
wald.test(b=coef(model), Sigma= vcov(model), Terms=1:12) #Here Terms, no. of independent variables in your final train model is 12
#Since, p-value is less then 0.001, hence we reject Ho that the all Bi=0

#Lagrange Multiplier or Score Test (Assess whether the current variable significantly improves the model fit or not)
modelChi <- model$null.deviance - model$deviance
modelChi
#Finding the degree of freedom for Null model and model with variables
chidf <- model$df.null - model$df.residual
chidf

# With more decimal places
# If p value is less than .05 then we reject the null hypothesis that the model is no better than chance.
chisq.prob <- 1 - pchisq(modelChi, chidf)
format(round(chisq.prob, 2), nsmall = 5)

#Predicting power of the model using R2 (it checks the how good your model is in terms of predicting output)
#Overall R-Square is between 0 -0.5
#Good R-Square is between 0.2 - 0.4

# Hosmer and Lemeshow given by the McFadden 6R square
R2.hl<-modelChi/model$null.deviance
R2.hl
# Cox and Snell R Square 

R.cs <- 1 - exp ((model$deviance - model$null.deviance) /nrow(data.train))
R.cs

# Max rescaled R square (Nagelkarke) 

R.n <- R.cs /(1-(exp(-(model$null.deviance/(nrow(data.train))))))
R.n

#Lackfit Deviance
##Ho: Observed Frequencies/probabilities =Expected Frequencies/probabilities
1-pchisq(deviance(model), df.residual(model)) ##Thus, we accept the Null Hypothesis Ho that Observed Frequencies = Expected Frequencies


#NEED TO INTERPRET THE COEFFICIENTS BY TAKING THE EXPONENTIAL:
# Coefficients (Odds)
modelt$coefficients #taking exponential
# Coefficients (Odds Ratio)
exp(modelt$coefficients)  #Interpret 

# Variable Importance of the model
varImp(modelt)

# Predicted Probabilities
prediction <- predict(model,newdata = data.train,type="response")
prediction #generates probabilities and can convert them into 0 & 1 format using threshold, output of logit model is always probabilities
write.csv(prediction,"pred.csv")

##accuracy  checks of the model
#DERIVING THE ROC Curve
data.train$Churn <- as.factor(data.train$Churn) #define this first this devides the data in two levels 0 & 1
rocCurve<- roc(response= data.train$Churn, predictor = factor(prediction, ordered=TRUE), 
               levels =rev(levels(data.train$Churn)))
threshold<-as.numeric(coords(rocCurve,"best")[1]) #converting probabilities in 1 and 0 format
threshold #its 0.29
predclass <-ifelse(prediction>threshold,1,0)
predclass
Confusion <- table(Predicted = predclass,Actual = data.train$Churn)
Confusion #generating confusion matrix
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
AccuracyRate #overall accuracy of model checking with confusion matrix which is 0.75
Gini <-2*auc(rocCurve)-1
Gini #its 0.67
#Gini Coefficient is the area under the Lorenz Curve (Similiar ROC Curve where final model compared to baseline model)
#Range 0.5 - 0.8
auc(rocCurve) #to calculate the area under the ROC Curve which is 0.84
plot(rocCurve)
#another accuracy check
### KS statistics calculation
data.train$m1.yhat <- predict(model, data.train, type = "response")
m1.scores <- prediction(data.train$m1.yhat, data.train$Churn)

plot(performance(m1.scores, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")

m1.perf <- performance(m1.scores, "tpr", "fpr")
ks1.logit <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit # Thumb rule : should lie between 0.4 - 0.7, here it is 0.52



##TESTING ON TEST DATA SET
#BUILDING LOGISTIC MODEL ON TEST DATA SET
modelt <- glm(Churn~., data=data.test, family=binomial()) #building the logit model
summary(model)

###REMOVING INSIGNIFICANT VARIABLES
modelt <- glm(Churn~ 	SeniorCitizen  
              + I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
              + I(StreamingMovies=="Yes") + Contract + PaperlessBilling
              + I(PaymentMethod=="Electronic check")+	TotalCharges , data=data.test, family=binomial())
summary(modelt)

modelt <- glm(Churn~ I(MultipleLines=="Yes")+ InternetService    + I(StreamingTV=="Yes") 
              + I(StreamingMovies=="Yes") + Contract + PaperlessBilling
              + I(PaymentMethod=="Electronic check")+	TotalCharges , data=data.test, family=binomial())
summary(modelt)

#MULTICOLIEARITY CHECK
vif(modelt)

#Checking the overall fitness of the model
#using Wald Test
wald.test(b=coef(modelt), Sigma= vcov(modelt), Terms=1:12) #Here Terms, no. of independent variables in your final train model is 12
#Since, p-value is less then 0.001, hence we reject Ho that the all Bi=0

#Lagrange Multiplier or Score Test (Assess whether the current variable significantly improves the model fit or not)

# Difference between -2LL of Null model and model with variables
modelChi <- modelt$null.deviance - modelt$deviance
modelChi

#Finding the degree of freedom for Null model and model with variables
chidf <- modelt$df.null - modelt$df.residual
chidf #its 10 here

# With more decimal places
# If p value is less than .05 then we reject the null hypothesis that the model is no better than chance.
chisq.prob <- 1 - pchisq(modelChi, chidf)
format(round(chisq.prob, 2), nsmall = 5)

# Hosmer and Lemeshow R square
R2.hl<-modelChi/modelt$null.deviance
R2.hl #its 0.28 here


# Cox and Snell R Square (the last number; here is 2000 should be total no. of ovservation)

R.cs <- 1 - exp ((modelt$deviance - modelt$null.deviance) /2000)
R.cs #its 0.29 here

# Max rescaled R square (Nagelkarke) (the last number; here is 2000 should be total no. of ovservation)

R.n <- R.cs /(1-(exp(-(modelt$null.deviance/2000))))
R.n #0.42 here


# Lackfit Deviance 
####Ho: Observed Frequencies/probabilties =Expected FRequencies/probabilties
1-pchisq(deviance(modelt), df.residual(modelt))

# Coefficients (Odds)
modelt$coefficients
# Coefficients (Odds Ratio)
exp(modelt$coefficients)

# Predicted Probabilities
prediction <- predict(modelt,newdata = data.test,type="response")
prediction

##accuracy  checks of the model
#DERIVING THE ROC Curve
data.test$Churn <- as.factor(data.test$Churn) #define this first this devides the data in two levels 0 & 1
rocCurvet<- roc(response= data.test$Churn, predictor = factor(prediction, ordered=TRUE), 
               levels =rev(levels(data.test$Churn)))
threshold<-as.numeric(coords(rocCurvet,"best")[1]) #converting probabilities in 1 and 0 format
threshold #its 0.27
predclasst <-ifelse(prediction>threshold,1,0)
predclasst
write.csv(predclasst,"predtest.csv")
Confusiont <- table(Predicted = predclasst,Actual = data.test$Churn)
Confusiont #generating confusion matrix
AccuracyRatet <- sum(diag(Confusiont))/sum(Confusiont)
AccuracyRatet #overall accuracy of model checking with confusion matrix which is 0.76
Ginit <-2*auc(rocCurvet)-1
Ginit #its 0.69
#Gini Coefficient is the area under the Lorenz Curve (Similiar ROC Curve where final model compared to baseline model)
#Range 0.5 - 0.8
auc(rocCurvet) #to calculate the area under the ROC Curve which is 0.85
plot(rocCurvet) #for plotting roc curve


#another accuracy check
### KS statistics calculation
data.test$m1.yhat <- predict(model, data.test, type = "response")
m1.scorest <- prediction(data.train$m1.yhat, data.train$Churn)

plot(performance(m1.scorest, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")

m1.perft <- performance(m1.scorest, "tpr", "fpr")
ks1.logit_t <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit_t # Thumb rule : should lie between 0.4 - 0.7, here it is 0.52



