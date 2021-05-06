
%F-Score
X=csvread('F:\N6-methyladenosine sites\m6A\Fusion\csv\Human.csv');
X1=X(1:5100,:);
X2=X(5101:10200,:);
[n1,m1]=size(X1);
[n2,m2]=size(X2);
aver1=mean(X1);
aver2=mean(X2);
aver3=mean(X);
numrator=(aver1-aver3).^2+(aver2-aver3).^2;
sum_1=zeros(1,m1);
for k=1:n1
    chazhi_1=X1(k,:)-aver1;
    added_1=chazhi_1.^2;
    sum_1=sum_1+added_1;
end
deno_1 = sum_1/(n1-1);
sum_2=zeros(1,m2);

for k=1:n2
    chazhi_2=X2(k,:)-aver2;
    added_2=chazhi_2.^2;
    sum_2=sum_2+added_2;
end
deno_2=sum_2/(n2-1);
deno=deno_1+deno_2;
F_1=numrator./deno;

len=length(F_1);
for k=1:len
    if isnan(F_1(k))
        F_1(k)=-1;
    end
end
 F_2=[F_1;X];
 F_3=F_2';
 F=sortrows(F_3,-1);
 F=F';
 csvwrite('F:\N6-methyladenosine sites\m6A\Fusion\LassoCV\Human.csv',F);


