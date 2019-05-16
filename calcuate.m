clear all;
clc;

load comp4_10_2.txt;
A=comp4_10_2;

M = size(A,1);
N = size(A,2);

m=0;n=0;t=0;k=0;p=0;q=0;
for i=1:M
%     for j=1:2
        if A(i,1)==1 && A(i,2)==1
            m = m + 1;      %正确检测出的正常数
        end
        if A(i,1)==0 && A(i,2)==0
            n = n + 1;      %正确检测出的入侵数
        end
        if A(i,1)==1 && A(i,2)==0
            k = k + 1;   %被误判为入侵的正常数
        end
        if A(i,1)==0 && A(i,2)==1
            t = t + 1;   % 被误判为正常的入侵数
        end
        if A(i,1)==0
            p=p+1;  %入侵数目
        end
        if A(i,1)==1
            q=q+1;  %正常数目
        end
%     end
end
TP=m;FN=k;FP=t;TN=n;
P=TP+FN;
N=FP+TN;
specificity=TN/N;%特效性
precision=TP/(TP+FP); %精度
recall=TP/(TP+FN);

F=(2*precision*recall)/(precision+recall);
G_mean=sqrt(recall*specificity);
Pde=n/p;  %检测率
Pfp=k/q;   % 虚警率
Pfn=(p-n)/p;  % 漏警率
Pi=p/(p+q);  %入侵行为占总样本数据的比率
Pi2=q/(p+q);   %正常行为占样本数据的比率
Pcred =(Pi*Pde)/(Pi*Pde+Pi2*Pfp);            %检测系统报警信息的可信度

DR=n/p;
FPR=k/q;