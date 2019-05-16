clear all
clc;
%% ��������
load seed.txt;
cluster_n=3;
data =seed;
data(:,8) = []; %�����ǩ���ÿ�
% data(:,size(data,2)) = [];%������size(data,2)=����data������
% data(:,1:7)
data_n = size(data, 1); %���ݵĸ��� 
in_n = size(data, 2);% ����ά��
% datalength=1000;  %���ݳ���

%% �������
default_options = [2.5;    % �����Ⱥ������ݴη���ģ��ϵ��
        300;    % ����������
        1e-5;    %����
        1];    % �ж�����

options = default_options;
m = options(1);        % Exponent for U �����Ⱥ������ݴη�
max_iter = options(2);        % Max. iteration ����������
min_impro = options(3);        % Min. improvement  ��С��������
display = options(4);        % Display info or not ��ʾ��Ϣ���
obj_fcn = zeros(max_iter, 1);    % Array for objective function

tic
%% ��ʼ�������Ⱦ��󲢹�һ
U = rand(cluster_n, data_n); %rand()����������� 
col_sum = sum(U);
% col=col_sum(ones(cluster_n,1),:);
U_1 = U./col_sum(ones(cluster_n, 1), :);%��һ��


%% ��ʼ����
for i = 1:max_iter,%������������
    tic
    mf = U_1.^m;      % MF matrix after exponential modificationָ���������MF����
 s1=(ones(size(data, 2), 1)*sum((mf')))';
 s2=ones(size(data, 2), 1)*sum((mf'));
%  center1=mf*data./(ones(size(data, 2), 1)*sum((mf')));
    center = mf*data./((ones(size(data, 2), 1)*sum((mf')))'); % �����µľ�������
   
       out = zeros(size(center, 1), size(data, 1));  %ÿ���㵽ÿ�����ĵľ��룬����Ϊ������
     
        if size(center, 2) > 1,%������ά������һִ�����³���
            for k = 1:size(center, 1),%��K��ֵ
                abc = ((data-ones(size(data,1),1) * center(k,:)).^2)';
                s3=sqrt(sum(abc));
                  out(k, :) = sqrt(sum(abc));%�õ�ŷ�Ͼ���
            end
       else    % 1-D data
            for k = 1:size(center, 1),
                  out(k, :) = abs(center(k)-data)';%absȡ����ֵ
            end
        end
   
    obj_fcn(i) = sum(sum((out.^2).*mf));  % Ŀ�꺯��
    tmp = out.^(-2/(m-1));      % �����µ������Ⱦ�����⹫ʽ���
    U_new= tmp./(ones(cluster_n, 1)*sum(tmp));  % �µ������Ⱦ���
    
    [~,label] = max(U_new); %�ҵ���������
% % subplot(1,2,1);
% % gscatter(data(:,1),data(:,4)),title('choose:1,4��,���۽��')
% 
% cal accuracy����׼ȷ��
a_1 = size(find(label(1:70)==1),2);
a_2 = size(find(label(1:70)==2),2);
a_3 = size(find(label(1:70)==3),2);
a = max([a_1,a_2,a_3]);
b_1 = size(find(label(71:140)==1),2);
b_2 = size(find(label(71:140)==2),2);
b_3 = size(find(label(71:140)==3),2);
b = max([b_1,b_2,b_3]);
c_1 = size(find(label(141:210)==1),2);
c_2 = size(find(label(141:210)==2),2);
c_3 = size(find(label(141:210)==3),2);
c = max([c_1,c_2,c_3]);
accuracy = (a+b+c)/210;
% % plot answer
% subplot(1,2,2);
% gscatter(data(:,4),data(:,12),label),title(['choose:4,12�� ʵ�ʽ��,accuracy=',num2str(accuracy)])




%%������
% fragment=3;
% step=70;
%     for i=1:data_n %��1����6��
%         for j=1:cluster_n %����
%             tempp=U(i,j)*log2(U(i,j));
% %             data11=XNS(tempp,fragment);
% %             last(i,j)=data11;
%         end
%     end
%%    
%      if display,
%          fprintf('Iteration count = %d, obj. fcn = %f, XN_shang=%f\n', i, obj_fcn(i),last(i,j));
% 
    if display, 
        fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
        %������������ͺ����Ľ��
    end
    % check termination condition�����ֹ����
    if i > 1,  %������������
        if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end,
    end
   
%   fragment=3;
%   step=70;
%     for i=1:1:6 %��1����6��
%         for j=1:fragment %����
%             tempp=U(i,((j-1)*step+1):j*step);
%             data11=XNS(tempp,fragment);
%             last(i,j)=data11;
%         end
%     end
%     for i=1:data_n
%         for j=1:cluster_n
%             entroy=U(i,j)*log2(U(i,j));
%             last(i,j)=last(i,j)-entroy;
%         end
%     end
%     
    toc  
end
toc

% plot(i,j)
%  plot(U(1,:),'-ro');
%  grid on
%  hold on
%  plot(U(2,:),'-g*');
%   plot(U(3,:),'-b+');
% 
% ylabel('Entropy value of data')
% legend('FCM1','FCM2','FCM3','location','northeast');
% 
% toc