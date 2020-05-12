function [w,v,history] = ElasticConjGradTest(rho, lambda, mu,yalmipFile,patientNo,dataName,dirName)


t_start = tic;%设置时间戳

dataFile = [yalmipFile,'\',dataName];
dataAll = load(dataFile);  %从文件读数据

% dataAll = dataAll(1:400,:);  % 根据病人数量动态调整数据数目
% dataAll(401:800,:) = dataAll(1:400,:);
% dataAll(801:1600,:) = dataAll(1:800,:);

[w,v,W_i, V_i, history] = ElasticConjGrad(dataAll, lambda, mu, rho, patientNo);

K = length(history.primObjective);
paramFile = [yalmipFile,'\',dataName,'_featureVector.txt'];
evaluateFile1 = [yalmipFile,'\',dataName,'_MSE_R2.txt'];       %记录MSE bmR2 R2_Adjusted
evaluateFile2 = [yalmipFile,'\',dataName,'_MAE_R2.txt'];       %记录MAE bmR2 R2_Adjusted
evaluateFile3 = [yalmipFile,'\',dataName,'_RMSE_R2.txt'];       %记录RMSE bmR2 R2_Adjusted
evaluateFile4 = [yalmipFile,'\',dataName,'_MAPE_R2.txt'];       %记录MAPE bmR2 R2_Adjusted


%创建目标函数文件夹
objectPath = [yalmipFile,'\目标函数文件'];            %% 创建目标函数文件夹
if ~exist(objectPath)
    mkdir(objectPath)
end;    
strpatientNo = num2str(patientNo,'%3d');                        %UE个数转换为字符串
objectFileName = [objectPath,'\',strpatientNo,'.txt'];              %目标函数文件

%创建迭代次数文件夹
iterFile = [yalmipFile,'\迭代次数文件'];            %% 创建目标函数文件夹
if ~exist(iterFile)
    mkdir(iterFile)
end;    

iterFileName = [iterFile,'\','iter.txt'];              %迭代次数文件

fVector = fopen(paramFile,'a');                 % 将数据保存到文件中
fevaluate1 = fopen(evaluateFile1,'a');          % 均方误差等错误率文件
fevaluate2 = fopen(evaluateFile2,'a');          % 平均绝对误差等错误率文件
fevaluate3 = fopen(evaluateFile3,'a');          %均方根误差等错误率文件
fevaluate4 = fopen(evaluateFile4,'a');          %平均绝对百分比误差等错误率文件

fobject = fopen(objectFileName,'a');            % 目标函数文件
fiter = fopen(iterFileName,'a');                % 迭代次数文件

fprintf(fVector,'\r\n %-5d: ',patientNo);    %写入 UE 个数值
fprintf(fevaluate1,'%-5d ',patientNo);
fprintf(fevaluate2,'%-5d',patientNo);
fprintf(fevaluate3,'%-5d',patientNo);
fprintf(fevaluate4,'%-5d',patientNo);

%迭代次数文件：写入 UE + lambda + rho + iter
fprintf(fiter,'%-5d %-5d %-5d %-5d %-5d\r\n',patientNo, lambda, rho,mu, history.OriginalResidualsIter);

for i = 1:K                                   %目标函数文件：写入目标函数值
    fprintf(fobject,'%-5d ',i);
    fprintf(fobject,'%-5d ',history.beforeADMMObjective(i));
    fprintf(fobject,'%-5d ',history.primObjective(i));
    fprintf(fobject,'%-5d',history.dualObjective(i));
    fprintf(fobject,'%-5d\r\n',history.primObjective(i));
end;

fprintf(fVector,' %-5f ',w);       %特征向量文件：写入特征向量 + 截距
fprintf(fVector,' %-5f\r\n',v);

fprintf(fevaluate1,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.MSE(K), history.R2(K), history.R2_adjusted(K),history.testMSE, history.testR2, history.testR2_adjusted ); %错误率文件：写入 MSE + R2 + R2_Adjusted
fprintf(fevaluate2,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.MAE(K), history.R2(K), history.R2_adjusted(K),history.testMAE, history.testR2, history.testR2_adjusted ); %错误率文件：写入 MAE + R2 + R2_Adjusted
fprintf(fevaluate3,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.RMSE(K), history.R2(K), history.R2_adjusted(K),history.testRMSE, history.testR2, history.testR2_adjusted ); %错误率文件：写入 RMSE + R2 + R2_Adjusted
fprintf(fevaluate4,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.MAPE(K), history.R2(K), history.R2_adjusted(K),history.testMAPE, history.testR2, history.testR2_adjusted ); %错误率文件：写入 MAPE + R2 + R2_Adjusted
fclose(fVector);   %关闭文件
fclose(fevaluate1);
fclose(fevaluate2);
fclose(fevaluate3);
fclose(fevaluate4);
fclose(fobject);
fclose(fiter)

strlambda = num2str(lambda,'%3f');
strrho = num2str(rho,'%3f');
strpatientNo = num2str(patientNo,'%3d');                        %UE个数转换为字符串
saveFile = [yalmipFile,dirName,strpatientNo,'λ',strlambda,'ρ',strrho];

if ~exist(saveFile)
    mkdir(saveFile)
end;

%绘制目标函数值的图像
K = length(history.primObjective);
h = figure;
%plot(1:K, history.beforeADMMObjective, 'k-', 'MarkerSize', 10, 'LineWidth', 2);
plot(1:K, history.primObjective, 'k-', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% t1 = [1, K];
optiobjvalArray = zeros(1,K) + history.optiobjval;
% t2 = [history.optiobjval, history.optiobjval];
plot(1:K, optiobjvalArray, 'r--', 'MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
legend('分布式算法','集中式算法');
%zlabel(['\fontname{宋体}理论值[m]\fontname{Euclid}this is english!\fontname{宋体}又是中文！ '])
ylabel('目标函数值'); xlabel('迭代次数');
grid on
saveObjImg = [saveFile,'\Objective.png'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\图1.tiff'];
saveas(gcf,saveObjImg);


%绘制lasso回归的系数图像：当特征很多的时候能够观察到哪个特征决定性作用比较大
h1 = figure;
plot(w,'k-','MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(history.optW,'r--','MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
legend('ADMM-Coefficient','Centralized-Coefficient');
ylabel('Coefficient'); xlabel('Feature');
saveObjImg = [saveFile,'\Coefficient.png'];
saveas(gcf,saveObjImg);

%绘制R方值、R方校正值
h2 = figure;
plot(1:K, history.R2,'k-','MarkerSize', 10, 'LineWidth', 2);
%hold on;
%plot(1:K, history.R2_adjusted,'r--','MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
%legend('R^2','R^2Adjusted');
xlabel('迭代次数');
ylabel('决定系数R^2值');
saveObjImg = [saveFile,'\R2.png'];
saveas(gcf,saveObjImg);

%绘制训练集MSE误差值
h3 = figure;
plot(1:K, history.MSE,'r-','MarkerSize', 10, 'LineWidth', 2);%ADMM训练集误差
hold on;
plot(1:K, history.centralizeMSE,'k--','MarkerSize', 10, 'LineWidth', 2);%集中式训练集误差
set(gca,'FontSize',16);
legend('ADMM-MSE','Centralized-MSE');
xlabel('iter (k)');
ylabel('Mean Squared Error');
saveObjImg = [saveFile,'\MSE.png'];
saveas(gcf,saveObjImg);


%绘制训练集MAE误差值
h4 = figure;
plot(1:K, history.MAE,'r-','MarkerSize', 10, 'LineWidth', 2);%ADMM训练集误差
hold on;
plot(1:K, history.centralizeMAE,'k--','MarkerSize', 10, 'LineWidth', 2);%集中式训练集误差
set(gca,'FontSize',16);
legend('ADMM-MAE','Centralized-MAE');
xlabel('iter (k)');
ylabel('Mean Absolute Error');
saveObjImg = [saveFile,'\MAE.png'];
saveas(gcf,saveObjImg);

%绘制训练集RMSE误差值
h5 = figure;
plot(1:K, history.RMSE,'k-','MarkerSize', 10, 'LineWidth', 2);%ADMM训练集误差
hold on;
plot(1:K, history.centralizeRMSE,'r--','MarkerSize', 10, 'LineWidth', 2);%集中式训练集误差
set(gca,'FontSize',16);
legend('分布式算法','集中式算法');
xlabel('迭代次数');
ylabel('RMSE');
grid on
saveObjImg = [saveFile,'\RMSE.png'];
saveas(gcf,saveObjImg);

%绘制训练集MAE误差值
h6= figure;
plot(1:K, history.MAPE,'r-','MarkerSize', 10, 'LineWidth', 2);%ADMM训练集误差
hold on;
plot(1:K, history.centralizeMAPE,'k--','MarkerSize', 10, 'LineWidth', 2);%集中式训练集误差
set(gca,'FontSize',16);
legend('ADMM-MAPE','Centralized-MAPE');
xlabel('iter (k)');
ylabel('Mean Absolute Percent Error');
saveObjImg = [saveFile,'\MAPE.png'];
saveas(gcf,saveObjImg);

%绘制原始残差、对偶残差的图像
g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k-', ...
    1:K, history.eps_pri, 'r--',  'LineWidth', 2);
set(gca,'FontSize',16);
legend('原始残差','停止条件');
ylabel('||r||_2');
grid on

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k-', ...
    1:K, history.eps_dual, 'r--', 'LineWidth', 2);
set(gca,'FontSize',16);
legend('对偶残差','停止条件');
ylabel('||s||_2'); xlabel('迭代次数');
grid on
saveResidualImg = [saveFile,'\Residual.png'];
saveas(gcf,saveResidualImg);
% system('shutdown -s');

%绘制R方值、R方校正值
h7 = figure;
plot(1:K, history.R2_adjusted,'k-','MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
legend('校正复相关系数R_a^2值');
xlabel('迭代次数');
ylabel('校正复相关系数R_a^2值');
grid on
saveObjImg = [saveFile,'\R2.png'];
saveas(gcf,saveObjImg);