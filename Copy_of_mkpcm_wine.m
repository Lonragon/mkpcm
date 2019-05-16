clear all
clc;
%% ��������
load wine.txt;
cluster_n=3;
x=wine;
x(:,1) = []; %�����Ϊɾ������ĵ�5�У������ǩ��
% data(:,size(data,2)) = [];%������size(data,2)=����data������
% data(:,1:7)
% datalength=1000;  %���ݳ���

% %% pca��ά
% [p,princ,egenvalue]=princomp(data);%�������ɷ�
% p=p(:,1:12);%���2��13�е����ɷ�ϵ��
% data=princ(:,1:12);%2��3�����ɷ���
% egenvalue;%���ϵ�����������ֵ���������ɷ���ռ����
% per=100*egenvalue/sum(egenvalue);%�������ɷ���ռ�ٷֱ�

% ��һ��[0,1]����
[y,ps]=mapminmax(x);
ps.ymin=0;
ps.ymax=1;
[y,ps]=mapminmax(x,ps);
ps.ymin=0;
ps.yrange=1;
data=y;

% 
% data=y;
% % datalength=3000; %���ݿ��С
% data_num=10;   %���ݿ����
% 
% Data{1} =  y(1:datalength,:);
% data_n = size(Data{1}, 1); %���ݵĸ��� 
% M = size(Data{1}, 2);% ����ά��
data_n=size(data,1);
M=size(data,2);


%% �������
default_options = [1.5;    % �����Ⱥ������ݴη�m,��pcm��ΪȨ��ָ��
        300;    % ����������
        1e-5;    %����
        1];    % �ж�����

options = default_options;
m = options(1);        % Exponent for U �����Ⱥ������ݴη�
max_iter = options(2);        % Max. iteration ����������
min_impro = options(3);        % Min. improvement  ��С��������
display = options(4);        % Display info or not ��ʾ��Ϣ���
obj_fcn = zeros(max_iter, 1);    % Array for objective function

sigma1=0.5;
sigma2=0.25;
alpha=0.5;

% U= zeros(data_n,cluster_n);
% K1=zeros(data_n,cluster_n);
% K2=zeros(data_n,cluster_n);
% Kcom=zeros(data_n,cluster_n);
% S=zeros(data_n,1);



%% ��ʼ�������Ⱦ��󲢹�һ
% U = rand(cluster_n, data_n); %rand()����������� 
% col_sum = sum(U);
% % mem=col_sum(ones(cluster_n, 1), :);
% U = U./col_sum(ones(cluster_n, 1), :);%��һ��

U0 = rand(data_n,cluster_n); 
S = sum(U0,2);  %����U������ӣ�ÿ�����
for i = 1:data_n
    for k = 1:cluster_n
        U0(i,k) = U0(i,k)/S(i,1);
    end
end
 Um=U0.^m;
center=rand(cluster_n,M);

%  %% Kcom=K1+alpha*K2  (K1,K2���Ǹ�˹�˺���)%%%%%
% for i=1:data_n 
% %     for q=1:data_n
%         for j=1:M
%             for k=1:cluster_n
%                 K1(i,k)=exp(((-1)*(data(i,j)-center(k,j)).^2)./2*sigma1^2);  
% %                 K2(q,k)=exp(((-1)*(data(q,j)-center(k,j)).^2)./2*sigma2^2);
%                 K2(i,k)=exp(((-1)*(data(i,j)-center(k,j)).^2)./2*sigma2^2);
% %                 K3(i,k)=(K1(i,k)./sigma1^2)+alpha*(K2(q,k)./sigma2^2);
% %                 Kcom(i,k)=K1(i,k)+alpha*K2(q,k);
%                 K3(i,k)=(K1(i,k)./sigma1^2)+alpha*(K2(i,k)./sigma2^2);
%                 Kcom(i,k)=K1(i,k)+alpha*K2(i,k);
%             end 
%         end 
% %        s1= ones(size(data,1),1)*center(k,:);
% %          K1(i,j)=exp(((-1)*(data(i,j)-center(k)).^2)/2*sigma^2);
% %          K2(i,j)=exp((-1*(data_avg(i,j)-center(k,:)).^2)/2*sigma^2);
%        
% %     end
% end


%     mf = U.^m;      % MF matrix after exponential modificationָ���������MF����
% %  s1=(ones(size(data, 2), 1)*sum((mf')))';
% %  s2=ones(size(data, 2), 1)*sum((mf'));
% %  center1=mf*data./(ones(size(data, 2), 1)*sum((mf')));
%     center = mf*data./((ones(size(data, 2), 1)*sum((mf')))'); % �����µľ�������
%     

 
%% ��ʼ����
 for l = 1:max_iter,%������������  

%  %% Kcom=K1+alpha*K2  (K1,K2���Ǹ�˹�˺���)%%%%%
% for i=1:data_n 
% %     for q=1:data_n
%        for k=1:cluster_n
%           dummy=0;
%            for j=1:M
%                dummy=dummy+abs(data(i,j)-center(k,j))^2;
%            end 
%                 d(i,k)=sqrt(dummy);
%                 K1(i,k)=exp((-1)*(d(i,k)^2/2*(sigma1^2)));  
% %                 K2(q,k)=exp(((-1)*(data(q,j)-center(k,j)).^2)./2*sigma2^2);
%                 K2(i,k)=exp((-1)*(d(i,k)^2/2*(sigma2^2)));  
% %                 K3(i,k)=(K1(i,k)./sigma1^2)+alpha*(K2(q,k)./sigma2^2);
% %                 Kcom(i,k)=K1(i,k)+alpha*K2(q,k);
%                 K3(i,k)=(K1(i,k)/sigma1^2)+alpha*(K2(i,k)/sigma2^2);
%                 Kcom(i,k)=K1(i,k)+alpha*K2(i,k);
%            
%         end 
% %        s1= ones(size(data,1),1)*center(k,:);
% %          K1(i,j)=exp(((-1)*(data(i,j)-center(k)).^2)/2*sigma^2);
% %          K2(i,j)=exp((-1*(data_avg(i,j)-center(k,:)).^2)/2*sigma^2);
%        
% %     end
% end
   
%% Kcom=K1+alpha*K2  (K1,K2���Ǹ�˹�˺���)%%%%%
for i=1:data_n 
%     for q=1:data_n
       for k=1:cluster_n
           for j=1:M
%                 K1(i,k)=exp(((-1)*(data(i,j)-center(k,j))^2)/2*sigma1^2);  
% %                 K2(q,k)=exp(((-1)*(data(q,j)-center(k,j)).^2)./2*sigma2^2);
%                 K2(i,k)=exp(((-1)*(data(i,j)-center(k,j))^2)/2*sigma2^2);
                   K1(i,k)=exp(((-1)*(data(i,j)-center(k,j))^2)/2*sigma1);       
                   K2(i,k)=exp(((-1)*(data(i,j)-center(k,j))^2)/2*sigma2);
%                 K3(i,k)=(K1(i,k)./sigma1^2)+alpha*(K2(q,k)./sigma2^2);
%                 Kcom(i,k)=K1(i,k)+alpha*K2(q,k);
           end 
%                 K3(i,k)=(K1(i,k)/sigma1^2)+alpha*(K2(i,k)/sigma2^2);
                 K3(i,k)=(K1(i,k)/sigma1)+alpha*(K2(i,k)/sigma2);
                Kcom(i,k)=K1(i,k)+alpha*K2(i,k);
            
        end 
%        s1= ones(size(data,1),1)*center(k,:);
%          K1(i,j)=exp(((-1)*(data(i,j)-center(k)).^2)/2*sigma^2);
%          K2(i,j)=exp((-1*(data_avg(i,j)-center(k,:)).^2)/2*sigma^2);
       
%     end
end


   %% ���¾������� 
  for k=1:cluster_n
      for j=1:M
         s1=zeros(cluster_n,M);
         s2=zeros(cluster_n,M);
       for i=1:data_n
%               s1(k,j)=s1(k,j)+Um(i,k)*data(i,j)*K3(i,k);
              s1(k,j)=s1(k,j)+Um(i,k)*K3(i,k)*data(i,j);  
              s2(k,j)=s2(k,j)+Um(i,k)*K3(i,k);   
             
             
       end
           center(k,j)=s1(k,j)/s2(k,j)
       end
         
   end
      %% �˻���ľ���
    for i=1:data_n
         for k=1:cluster_n
            D(i,k)=2*(1+alpha-Kcom(i,k));
         end
    end
    %% ����eta �Լ������Ⱦ���
  for q=1:cluster_n
       eta=zeros(data_n,1);
       for i=1:data_n
%            dummy=0;
          
           eta_zi=0;
           eta_mu=0;
%            eta=0;
           
            for k=1:cluster_n
               eta_zi=eta_zi+Um(i,k)*(D(i,k));
               eta_mu=eta_mu+Um(i,k);  
               eta(i)=eta_zi/eta_mu;
%                eta=eta';
            end 
             U(i,q)=(1+(D(i,q)/eta(i))^(1/(m-1)))^-1;  
       end
  end
   Um=U.^m;
%     %% ����������
%     for i=1:data_n
%       for k=1:cluster_n
%          U(i,k)=(1+(D(i,k)./eta(i))^(1/(m-1)))^-1;      
%       end
%     end
  
    %% ����Ŀ�꺯��
    s6=0;
    temp1=zeros(data_n,cluster_n);
    temp3=zeros(data_n,cluster_n);
    for k=1:cluster_n
         for i=1:data_n
%              ttttttt(i,k)=(1-U(i,k))^m;
            temp1(i,k)=(1-U(i,k))^m;
%             s5=sum(sum(temp1));
%              s5(i,k)=s5(i,k)+(1-U(i,k)).^m;
            
         end
%          s6=s6+eta(i,k);
          s6=s6+eta(i);
    end 
       s5=sum(sum(temp1));
        temp2=s5*s6;
        
        for i=1:data_n
            for k=1:cluster_n
%               temp3= sum(sum((U(i,k)^m)*D(i,k)^2)); 
               temp3(i,k)=(U(i,k)^m)*D(i,k)^2;
            end
        end
        s7=sum(sum(temp3));
         obj_fcn(l)=s7+temp2;
   
%         U{1}=U{1}';
      
%        out = zeros(size(center, 1), size(data, 1));  %ÿ���㵽ÿ�����ĵľ��룬����Ϊ������
%      
%         if size(center, 2) > 1,%������ά������һִ�����³���
%             for k = 1:size(center, 1),%��K��ֵ
%                 abc = ((data-ones(size(data,1),1) * center(k,:)).^2)';
%                 s3=sqrt(sum(abc));
%                   out(k, :) = sqrt(sum(abc));%�õ�ŷ�Ͼ���
%             end
%        else    % 1-D data
%             for k = 1:size(center, 1),
%                   out(k, :) = abs(center(k)-data)';%absȡ����ֵ
%             end
%         end
   
%     obj_fcn(i) = sum(sum((out.^2).*mf));  % Ŀ�꺯��
%     tmp = out.^(-2/(m-1));      % �����µ������Ⱦ�����⹫ʽ���
%     U_new= tmp./(ones(cluster_n, 1)*sum(tmp));  % �µ������Ⱦ���
%     
% %     
%     [~,label] = max(U_new); %�ҵ���������
% % subplot(1,2,1);
% % gscatter(data(:,1),data(:,4)),title('choose:1,4��,���۽��')
% 
% % % cal accuracy����׼ȷ��
% a_1 = size(find(label(1:50)==1),2);
% a_2 = size(find(label(1:50)==2),2);
% a_3 = size(find(label(1:50)==3),2);
% a = max([a_1,a_2,a_3]);
% b_1 = size(find(label(51:100)==1),2);
% b_2 = size(find(label(51:100)==2),2);
% b_3 = size(find(label(51:100)==3),2);
% b = max([b_1,b_2,b_3]);
% c_1 = size(find(label(101:150)==1),2);
% c_2 = size(find(label(101:150)==2),2);
% c_3 = size(find(label(101:150)==3),2);
% c = max([c_1,c_2,c_3]);
% accuracy = (a+b+c)/150;
% % plot answer
% subplot(1,2,2);
% gscatter(data(:,4),data(:,12),label),title(['choose:4,12�� ʵ�ʽ��,accuracy=',num2str(accuracy)])

    if display, 
        fprintf('Iteration count = %d, obj. fcn = %f\n', l, obj_fcn(l)); %12.7�ս�i������l
        %������������ͺ����Ľ��
    end
    % check termination condition�����ֹ����
    if l> 1,  %������������
        if abs( obj_fcn(l) -  obj_fcn(l-1)) < min_impro, break; end,   %12.7�ս�i������l
    end
    toc
  
end        
toc
% %% center�ķ���һ��
% [y,ps]=mapminmax(center);
% [center,ps]=mapminmax('reverse',y,ps);

 %% ��ͼ
   U=U';
   index1=find(U(1,:)==max(U));%%%%�ҵ�ÿ������Ӧ���м����������ڵ�����
   index2=find(U(2,:)==max(U));
   index3=find(U(3,:)==max(U));
%    index4=find(U(4,:)==max(U));
%    index5=find(U(5,:)==max(U));
%    index6=find(U(6,:)==max(U));
%    index7=find(U(7,:)==max(U));
%    index8=find(U(8,:)==max(U));
   bbbbbbbbbbb=data(index1,1);  %%%%�������һ��Ԫ��

   plot(data(index1,2),data(index1,3),'g*');
   hold on;
   plot(data(index2,2),data(index2,3),'r+');
   hold on;
   plot(data(index3,2),data(index3,3),'bo');
   hold on;
%    plot(data(index4,3),data(index4,6),'m.');
%    hold on;
%    plot(data(index5,3),data(index5,6),'c<');
%    hold on;
%    plot(data(index6,3),data(index6,6),'ysquare');
%    hold on;
%    plot(data(index7,3),data(index7,6),'kdiamond');
%    hold on;
%    plot(data(index8,1),data(index8,7),'mpentagram');
%    hold on;
   title('MKPCM wine-dataset');
   xlabel('Attribute 2');
   ylabel('Attribute 3');
   plot(center(1,2),center(1,3),'xk','MarkerSize',15);
   plot(center(2,2),center(2,3),'xk','MarkerSize',15);
   plot(center(3,2),center(3,3),'xk','MarkerSize',15);
%    plot(center(4,3),center(4,6),'xk','MarkerSize',15);
%    plot(center(5,3),center(5,6),'xk','MarkerSize',15);
%    plot(center(6,3),center(6,6),'xk','MarkerSize',15);
%    plot(center(7,3),center(7,6),'xk','MarkerSize',15);
%    plot(center(8,1),center(8,7),'xk','MarkerSize',15);
%    U=U';
%% ����NMIֵ
U=U';
R=ones(data_n,1);
for i=1:data_n
   for j=1:cluster_n
       aaaaa=R(i,1);
        Uaaaaa=U(i,R(i,1));
      if U(i,R(i,1))<U(i,j)
          R(i,1)=j;
      end
   end
end
R=reshape(R,1,data_n);
Q=wine(:,1);
Q=reshape(Q,1,data_n);

total=size(Q,2);
Q_i=unique(Q);
Q_c=length(Q_i);
R_i=unique(R);
R_c=length(R_i);
idQ=double (repmat(Q,Q_c,1)==repmat(Q_i',1,total));
idR=double (repmat(R,R_c,1)==repmat(R_i',1,total));
idQR=idQ*idR';

Sq=zeros(Q_c,1);
Sr=zeros(R_c,1);
for i=1:Q_c
   for j=1:total
     if idQ(i,j)==1
         Sq(i,1)=Sq(i,1)+1;
     end
   end
end

for i=1:R_c
  for j=1:total
     if idR(i,j)==1
         Sr(i,1)=Sr(i,1)+1;
     end
  end
end

Pq=zeros(Q_c,1);
Pr=zeros(R_c,1);

for i=1:Q_c
    Pq(i,1)=Sq(i,1)/total;
end

for i=1:R_c
   Pr(i,1)=Sr(i,1)/total;   %֮ǰ�д�Srд����Sq;
end

Pqr=idQR/total;

Hq=0;
Hr=0;
for i=1:Q_c
  Hq=Hq+Pq(i,1)*log2(Pq(i,1));
end

for i=1:R_c
  Hr=Hr+Pr(i,1)*log2(Pr(i,1));
end

MI=0;
for i=1:Q_c
  for j=1:R_c
    MI=MI+Pqr(i,j)*log2(Pqr(i,j)/(Pq(i,1)*Pr(j,1))+eps);
  end
end

NMI=MI/((Hq*Hr).^(1/2));
fprintf('NMI=%d\n',NMI);


% t=nmi(Q,R);

% plot(i,j)
%  plot(U(1,:),'-ro');
%  grid on
%  hold on
%  plot(U(2,:),'-g*');
%   plot(U(3,:),'-b+');
% 
% ylabel('Entropy value of data')
% legend('FCM1','FCM2','FCM3','location','northeast');

toc