#Machine Learning Final Project - Sentiment Analysis and Corn Price Movement 
#Name: Yu-Chu Alice Chen(yc3178), Hsiao Hsien Tsao(ht2435)
#Date: 8/13/2017
#Document includes data preprocessing and machine learning models explored 

#Set directory and read files 
setwd("~/Desktop/R/Machine learning")
tweets<-read.csv('tweets_data.csv',header = TRUE)
head(tweets)
corn<-read.csv('corn_price_data.csv',header = TRUE)
head(corn)
summary(corn)
summary(tweets)


#Calculate percentage price for settle
library(quantmod)
#calculate standard deviation of corn price and price percentage change of price
corn$pctchange <- Delt(corn$settle)
#set var column to numeric
corn$pctchange<-as.numeric(corn$pctchange)
#calculate difference in price
library(dplyr)
corn$var<-c(NA,diff(corn$settle))

#corn%>%
#mutate(var=diff(log(corn$settle)))

#corn$diff<-percent_change(corn$settle)
#corn$stdvarprice<-sd(corn$settle)

#Change date format to the same as corn price
date <- format(as.POSIXct(strptime(tweets$date,"%m/%d/%Y %H:%M",tz="")) ,format = "%m/%d/%Y")
tweets$date<-date

#create the wordcloud graph
r_stats_text_corpus <- Corpus(VectorSource(tweets$text))
r_stats_text_corpus <- tm_map(r_stats_text_corpus,
                              content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')))
require("twitteR")
require("wordcloud")
require("tm")
require("dplyr")
r_stats_text_corpus <- tm_map(r_stats_text_corpus, content_transformer(tolower))
r_stats_text_corpus <- tm_map(r_stats_text_corpus, removePunctuation) #remove all punctuation 
r_stats_text_corpus <- tm_map(r_stats_text_corpus, function(x)removeWords(x,stopwords()))
wordcloud(r_stats_text_corpus, min.freq = 10, max.words = 150, colors=brewer.pal(8, "Dark2"))

#connect all libraries
library(twitteR)
install.packages("ROAuth")
library(ROAuth)
library(plyr)
library(dplyr)
library(stringr)
library(ggplot2)
install.packages("NLP")
install.packages("stringi")
library(tm)
library(NLP)
library(stringi)
install.packages("qdap")
library(qdap)
#change tweets name to tweetpro
twepro<-tweets
#data preprocessing
twepro$text <- tolower(twepro$text)
twepro$text <- removePunctuation(twepro$text)
twepro$text <- removeNumbers(twepro$text)
twepro$text <- stripWhitespace(twepro$text)

#data.frame manipulation for analysis
twepro$location <- NULL
twepro$language <- NULL
twepro$favorites <- NULL

#tweets evaluation function
score.sentiment <- function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  scores <- laply(sentences, function(sentence, pos.words, neg.words){
    sentence <- gsub('[[:punct:]]', "", sentence)
    sentence <- gsub('[[:cntrl:]]', "", sentence)
    sentence <- gsub('\\d+', "", sentence)
    sentence <- tolower(sentence)
    word.list <- str_split(sentence, '\\s+')
    words <- unlist(word.list)
    pos.matches <- match(words, pos.words)
    neg.matches <- match(words, neg.words)
    pos.matches <- !is.na(pos.matches)
    neg.matches <- !is.na(neg.matches)
    score <- sum(pos.matches) - sum(neg.matches)
    return(score)
  }, pos.words, neg.words, .progress=.progress)
  scores.df <- data.frame(score=scores, text=sentences)
  return(scores.df)
}

#lexicon
neg = scan("negative-words.txt", what="character", comment.char=";")
pos = scan("positive-words.txt", what="character", comment.char=";")


twepro$text <- as.factor(twepro$text)
score <- score.sentiment(twepro$text, pos.words, neg.words, .progress='text')
combinescore <- cbind(tweets,score)
write.csv(score, file = "scores.csv")
write.csv(combinescore, file = "combinescore.csv")

#drop unimportant columns
combinescore<-combinescore[,-3:-6]

#Join corn and score together
library(plyr)
stockscore<-join(combinescore, corn[c('date','commodity', 'settle','pctchange','var')], by='date', type='full',match='first')
#create pos, neg and neutral
stockscore$pos <- as.numeric(stockscore$score >= 1)
stockscore$neg <- as.numeric(stockscore$score <= -1)
stockscore$neu <- as.numeric(stockscore$score == 0)

#Make one row per day
tweet_stock <- ddply(stockscore, c('date','settle','pctchange','var'), plyr::summarise, pos.count = sum(pos), neg.count = sum(neg), neu.count = sum(neu),mean.score=mean(score),sum.retweets=sum(retweets),sum.score=sum(score))

tweet_stock$all.count <- tweet_stock$pos.count + tweet_stock$neg.count + tweet_stock$neu.count

tweet_stock$percent.pos <- round((tweet_stock$pos.count / tweet_stock$all.count) * 100)
cor(tweet_stock$percent.pos, tweet_stock$pctchange, use = "complete")

#tweet_stock_df<- subset(tweet_stock_df,  !is.na(var))

#Remove rows with no settle price
tweet_stock<- subset(tweet_stock, !is.na(settle))
#change to matrix
tweet_stock_t<-as.matrix(tweet_stock)


#create features using lag function
library(zoo)
tweet_stock_t = tweet_stock %>%
  mutate(var.lag5 = rollapply(data = var, 
                                     width = 5, 
                                     FUN = mean, 
                                     align = "right", 
                                     fill = NA, 
                                     na.rm = T))%>%
  mutate(var.lag10 = rollapply(data = var, 
                                     width = 10, 
                                     FUN = mean, 
                                     align = "right", 
                                     fill = NA, 
                                     na.rm = T))%>%
  mutate(mean.score.lag5 = rollapply(data = mean.score, 
                                 width = 5, 
                                 FUN = mean, 
                                 align = "right", 
                                 fill = NA, 
                                 na.rm = T))%>%
  mutate(mean.score.lag10 = rollapply(data = mean.score, 
                                   width = 10, 
                                   FUN = mean, 
                                   align = "right", 
                                   fill = NA, 
                                   na.rm = T))
 
#drop mean.score with NA value
tweet_stock_t<- subset(tweet_stock_t,  !is.na(mean.score))
#set dummy variable for price change up down and neutral
tweet_stock_t$change <- ifelse(tweet_stock_t$var > 5, 1,-1)


#Split train and test set
n = nrow(tweet_stock_t)
trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
tweetstocktrain = tweet_stock_t[trainIndex ,]
tweetstocktest = tweet_stock_t[-trainIndex ,]

write.csv(tweetstocktrain,"train5-10.csv")
write.csv(tweetstocktest,"test5-10.csv")

#delete date column
train<-tweetstocktrain[,-1]
test<-tweetstocktest[,-1]

#correlation graph
install.packages("corrplot")
library(corrplot)
str(train)
train<-sapply(train, as.numeric)
corr_mat=cor(train, method="s")
corrplot(corr_mat)

##Machine Learning Models:
#load library
library("e1071")
library(caret)

#train set 
train<-read.csv("train5-10.csv",header = T)
train<-train[,-1] #remove x col
train$date<-as.Date(train$date, format="%m/%d/%Y")
train$change<-as.factor(train$change)
train<-na.omit(train)

#test set
test<-read.csv("test5-10.csv",header = T)
test<-test[,-1] #remove x col
test$date<-as.Date(test$date, format="%m/%d/%Y")
test$change<-as.factor(test$change)
test<-na.omit(test)

#Multi Linear Regression
linermodel = lm(pctchange ~mean.score+var+pos.count, data = train)
summary(linermodel)

#annual return
lin.pred<-predict(linermodel,train)
newdata<-data.frame(pos.count=10, mean.score=1, var=50) 
lin.pred2<-predict(linermodel, test)

#change y column to factor
tweetstocktrain$change<-as.factor(tweetstocktrain$change)
tweetstocktest$change<-as.factor(tweetstocktest$change)

#SVM Model 
svm.model<-svm(change~.,data=train)

svm.pred<-predict(svm.model, train)
tab<-table(pred=svm.pred, true=train$change)
confusionMatrix(svm.pred,train$change)
confusionMatrix(tab)

svm.testpred<-predict(svm.model,newdata=test)
confusionMatrix(svm.testpred,test$change)

#SVM ROC and AURPC 
#train SVM
svm.pred<-as.numeric(svm.pred)
y2<-as.numeric(train$change)
pr.train<-pr.curve(scores.class0 = svm.pred, scores.class1 = y2, curve = T)
plot(pr.train)
roc.train<-roc(response=svm.pred, predictor=y2)
plot(roc.train)

#test SVM
svm.testpred<-as.numeric(svm.testpred) #first change to numeric
y<-as.numeric(test$change) 
roc<-roc(response=svm.testpred,predictor = y)
pr<-pr.curve(scores.class0 = svm.testpred, scores.class1 = y, curve=T)
plot(roc)

#Kernel SVM - Radial
kr.svm.model<-svm(change~., data=train, kernel="radial", cost=10, scale=FALSE)  
kr.svm.pred<-predict(kr.svm.model, train)
kr.test.tab<-table(pred=kr.svm.pred, true=train$change)
confusionMatrix(kr.test.tab)

kr.svm.testpred<-predict(kr.svm.model,newdata=test)
confusionMatrix(kr.svm.testpred,test$change)

svm.error.rate <- sum(test$change != kr.svm.pred)/nrow(test)
print(paste0("Accuracy (Precision): ", 1 - svm.error.rate))

#SVM ROC and AURPC 
#train SVM
kr.svm.pred<-as.numeric(kr.svm.pred)
y2<-as.numeric(train$change)
pr.train<-pr.curve(scores.class0 = kr.svm.pred, scores.class1 = y2, curve = T)
plot(pr.train)
roc.train<-roc(response=svm.pred, predictor=y2)
plot(roc.train)

#test SVM
kr.svm.testpred<-as.numeric(kr.svm.testpred) #first change to numeric
y<-as.numeric(test$change) 
roc<-roc(response=kr.svm.testpred,predictor = y)
pr<-pr.curve(scores.class0 = svm.testpred, scores.class1 = y, curve=T)
plot(roc)

#Naive Bayes
train<-train[,-1]
nbmodel<-naiveBayes(change~., data=train)
pred<-predict(nbmodel, train)
nbtab<-table(pred,train$change)
confusionMatrix(nbtab)

test<-test[,-1]
newdata<-test[,-17]
nb.testpred<-predict(nbmodel,test,type="class")
confusionMatrix(nb.testpred,test$change)

#Naive Bayes ROC and AURPC 
nb.testpred<-as.numeric(nb.testpred) #first change to numeric
y<-as.numeric(test$change) 
roc<-roc(response=nb.testpred,predictor = y)
pr<-pr.curve(scores.class0 = nb.testpred, scores.class1 = y, curve=T)
plot(roc)

pred<-as.numeric(pred)
y2<-as.numeric(train$change)
pr.train<-pr.curve(scores.class0 = pred, scores.class1 = y2, curve = T)
plot(pr.train)
roc.train<-roc(response=pred, predictor=y2)
plot(roc.train)


#plots
library(ggplot2)
pctgraph<-ggplot(train,aes(date,pctchange))+geom_point()+geom_smooth()

poscount.graph<-ggplot(train,aes(pctchange,pos.count))+geom_point()+geom_smooth()
var.graph<-ggplot(train,aes(date,var))+geom_line()+geom_smooth()

#ggplot
ggplot(train, aes(mean.score,pctchange))+geom_point()+geom_smooth() 
