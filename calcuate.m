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
            m = m + 1;      %��ȷ������������
        end
        if A(i,1)==0 && A(i,2)==0
            n = n + 1;      %��ȷ������������
        end
        if A(i,1)==1 && A(i,2)==0
            k = k + 1;   %������Ϊ���ֵ�������
        end
        if A(i,1)==0 && A(i,2)==1
            t = t + 1;   % ������Ϊ������������
        end
        if A(i,1)==0
            p=p+1;  %������Ŀ
        end
        if A(i,1)==1
            q=q+1;  %������Ŀ
        end
%     end
end
TP=m;FN=k;FP=t;TN=n;
P=TP+FN;
N=FP+TN;
specificity=TN/N;%��Ч��
precision=TP/(TP+FP); %����
recall=TP/(TP+FN);

F=(2*precision*recall)/(precision+recall);
G_mean=sqrt(recall*specificity);
Pde=n/p;  %�����
Pfp=k/q;   % �龯��
Pfn=(p-n)/p;  % ©����
Pi=p/(p+q);  %������Ϊռ���������ݵı���
Pi2=q/(p+q);   %������Ϊռ�������ݵı���
Pcred =(Pi*Pde)/(Pi*Pde+Pi2*Pfp);            %���ϵͳ������Ϣ�Ŀ��Ŷ�

DR=n/p;
FPR=k/q;