clear all
clc;
%% 导入数据
load seed.txt;
cluster_n=3;
data =seed;
data(:,8) = []; %将类标签列置空
% data(:,size(data,2)) = [];%所有行size(data,2)=数组data的列数
% data(:,1:7)
data_n = size(data, 1); %数据的个数 
in_n = size(data, 2);% 数据维数
% datalength=1000;  %数据长度

%% 定义变量
default_options = [2.5;    % 隶属度函数的幂次方即模糊系数
        300;    % 最大迭代次数
        1e-5;    %步长
        1];    % 判定条件

options = default_options;
m = options(1);        % Exponent for U 隶属度函数的幂次方
max_iter = options(2);        % Max. iteration 最大迭代次数
min_impro = options(3);        % Min. improvement  最小进化步长
display = options(4);        % Display info or not 显示信息与否
obj_fcn = zeros(max_iter, 1);    % Array for objective function

tic
%% 初始化隶属度矩阵并归一
U = rand(cluster_n, data_n); %rand()产生随机矩阵 
col_sum = sum(U);
% col=col_sum(ones(cluster_n,1),:);
U_1 = U./col_sum(ones(cluster_n, 1), :);%归一化


%% 开始迭代
for i = 1:max_iter,%迭代次数控制
    tic
    mf = U_1.^m;      % MF matrix after exponential modification指数修正后的MF矩阵
 s1=(ones(size(data, 2), 1)*sum((mf')))';
 s2=ones(size(data, 2), 1)*sum((mf'));
%  center1=mf*data./(ones(size(data, 2), 1)*sum((mf')));
    center = mf*data./((ones(size(data, 2), 1)*sum((mf')))'); % 建立新的聚类中心
   
       out = zeros(size(center, 1), size(data, 1));  %每个点到每个中心的距离，行数为中心数
     
        if size(center, 2) > 1,%样本的维数大于一执行以下程序
            for k = 1:size(center, 1),%给K赋值
                abc = ((data-ones(size(data,1),1) * center(k,:)).^2)';
                s3=sqrt(sum(abc));
                  out(k, :) = sqrt(sum(abc));%得到欧氏距离
            end
       else    % 1-D data
            for k = 1:size(center, 1),
                  out(k, :) = abs(center(k)-data)';%abs取绝对值
            end
        end
   
    obj_fcn(i) = sum(sum((out.^2).*mf));  % 目标函数
    tmp = out.^(-2/(m-1));      % 根据新的隶属度矩阵求解公式求出
    U_new= tmp./(ones(cluster_n, 1)*sum(tmp));  % 新的隶属度矩阵
    
    [~,label] = max(U_new); %找到所属的类
% % subplot(1,2,1);
% % gscatter(data(:,1),data(:,4)),title('choose:1,4列,理论结果')
% 
% cal accuracy计算准确率
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
% gscatter(data(:,4),data(:,12),label),title(['choose:4,12列 实际结果,accuracy=',num2str(accuracy)])




%%熵理论
% fragment=3;
% step=70;
%     for i=1:data_n %从1到第6行
%         for j=1:cluster_n %段数
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
        %输出迭代次数和函数的结果
    end
    % check termination condition检查终止条件
    if i > 1,  %进化步长控制
        if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end,
    end
   
%   fragment=3;
%   step=70;
%     for i=1:1:6 %从1到第6行
%         for j=1:fragment %段数
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