library(dplyr)
library(car)

test = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Real Estate\\housing_test.csv")
train = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Real Estate\\housing_train.csv")


head(train)
test$Price= NA

train$data = 'train'
test$data = 'test'

all= rbind(train,test)

glimpse(all)
all$Postcode=as.character(all$Postcode)

apply(all,2,function(x) length(unique(x)))

CreateDummies=function(data,var,freq_cutoff=100){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    name=gsub("/","_",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
} 

all=all %>% 
  select(-SellerG,-Address,-Suburb)

head(all)
for_dummy_vars=c('Postcode','CouncilArea','Method','Type')

for(var in for_dummy_vars){
  all=CreateDummies(all,var,100)
}

for(col in names(all)){
  
  if(sum(is.na(all[,col]))>0 & !(col %in% c("data","Price"))){
    
    all[is.na(all[,col]),col]=mean(all[all$data=='train',col],na.rm=T)
  }
  
}

head(all)

# Impute the NA values with Mean in the dataset

trainf = all %>% filter(data == 'train') %>% select(-data) 
testf= all %>% filter(data == 'test') %>% select(-Price, -data) 

any(is.na(trainf))
any(is.na(testf))

library(randomForest)
fit = randomForest(Price ~ ., data = trainf) 

train.predictions = predict(fit, newdata = trainf)

### Make predictions on test and submit 

test.predictions = predict(fit, newdata = testf)

write.csv(test.predictions,file = "submission.csv", row.names = F)

