#---------------------Convert data to the Rates-----------------------


library(e1071)
mydf<-read.csv("WPI_OCTOBER_2015.csv", header=TRUE, stringsAsFactors=FALSE)
mymat<-as.data.frame(t(mydf[,-1]))
gdp<-read.csv("eco_growth.csv", header=TRUE, stringsAsFactors=FALSE)
tgdp<-as.data.frame(t(gdp[,6:15]))

year_avg<-function(V_ts) {
  time_s<-ts(V_ts,frequency = 12,start=c(2004,4))
  avg<-c()
  for(i in 2004:2014) {
    avg<-rbind(avg,mean(window(time_s,start=c(i,4),end=c(i+1,4) ) ) )  
  }
  avg
}

to_rate<-function(arg) {
  #print(arg)
  gdp_rate<-tgdp$V1
  gdp_list<-c(100)
  for(i in 1:length(gdp_rate)) {
    gdp_list<-c(gdp_list,(gdp_rate[i]*.01+1)*gdp_list[length(gdp_list)])
  }
  arg_val<-arg*.01*gdp_list
  arg_rate<-100*diff(arg_val)/arg_val[-length(arg_val)]
  #print(arg_val)
  as.array(arg_rate)
}

rates<-function() {
  rate<-c(1,2,3,4,5,6,7,8,13,14,18,19,21,23)
  rate_to_conv<-c(9,10,11,12,15,16,17,27,29)
  rate_bill<-c(20,22,24,25,26,28,30,31,32,33)
  #print(rate_conv_prev)
  gdp_ts_V<-c()
  for(i in c(1:33)) {
    if(i %in% rate) {
      #print(ts(tgdp[i],start = c(2004)))
      gdp_ts_V<-c(gdp_ts_V,ts(tgdp[i],start = c(2004)))
    }
    else if(i %in% rate_to_conv) {
      #print(as.array(t(gdp[i,5:15])))
      x<-ts(to_rate(as.array(t(gdp[i,5:15]))),start=c(2004))
      #print(x)
      gdp_ts_V<-c(gdp_ts_V,x)
    }
    else if(i %in% rate_bill) {
      data<-as.array(t(gdp[i,5:15]))
      y<-ts(100*diff(data)/data[-length(data)],start=c(2004))
      #print(y)
      gdp_ts_V<-c(gdp_ts_V,y)
    }
    else {
      gdp_ts_V<-c(gdp_ts_V,ts(tgdp[i],start = c(2004)))
    }
  }
  gdp_ts_V
}
l_rate<-as.data.frame(matrix(rates(),nrow=10,ncol=33))
#l_rate
write.table(l_rate, file="Cleaned.csv",row.names=FALSE, col.names=FALSE, sep=",")
#b<-c()
#sorted<-c()
#w_sum<-c()
clean_comm<-c()
#for(i in c(1:33)) {w_sum<-c(w_sum,0)}
for(i in c(1:797)) {
  #print(i)
  row<-mymat[i]
  row[is.na(row)]<-mean(row[-1,], na.rm=TRUE)
  check<-year_avg(row[-1,])
  com_rate<-100*diff(check)/data[-length(check)]
  year_ts<-ts(com_rate,start=c(2004))
clean_comm<-c(clean_comm,ts(com_rate,start=c(2004)))
}
clean_comm<-matrix(clean_comm,ncol = 797,nrow = 10)
write.table(clean_comm, file="Cleaned_prices.csv",row.names=FALSE, col.names=FALSE, sep=",")
  #regr1<-svm(year_ts ~ . ,l_rate,kernel='linear')
  #w <- (t(regr1$coefs) %*% regr1$SV)
  #w_sum<-w_sum+abs(w)
  #sort_w<-sort(w,index.return=TRUE,decreasing = TRUE)
  #sorted<-c(sorted,sort_w$ix)
  #b <- c(b,-1 * regr1$rho)
#}
#plot(ts(as.vector(w_sum)))
#sorted<-t(matrix(sorted,ncol = 797,nrow = 33))
#plot(table(as.vector(t(sorted[,1:5]))))


#row<-as.data.frame(mymat[27])
#print(mean(row[-1,], na.rm=TRUE))
#row[is.na(row)]<-mean(row[-1,], na.rm=TRUE)

#plot(cbind(fitted(regr1),year_ts))
#sort(w,index.return=TRUE,decreasing = TRUE)
#sorted<-sort(abs(w),index.return=TRUE,decreasing = TRUE)
#pred<-predict(regr1,l_rate)
#(pred-year_ts)*100/year_ts
