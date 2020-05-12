function [w,v,history] = ElasticConjGradTest(rho, lambda, mu,yalmipFile,patientNo,dataName,dirName)


t_start = tic;%����ʱ���

dataFile = [yalmipFile,'\',dataName];
dataAll = load(dataFile);  %���ļ�������

% dataAll = dataAll(1:400,:);  % ���ݲ���������̬����������Ŀ
% dataAll(401:800,:) = dataAll(1:400,:);
% dataAll(801:1600,:) = dataAll(1:800,:);

[w,v,W_i, V_i, history] = ElasticConjGrad(dataAll, lambda, mu, rho, patientNo);

K = length(history.primObjective);
paramFile = [yalmipFile,'\',dataName,'_featureVector.txt'];
evaluateFile1 = [yalmipFile,'\',dataName,'_MSE_R2.txt'];       %��¼MSE bmR2 R2_Adjusted
evaluateFile2 = [yalmipFile,'\',dataName,'_MAE_R2.txt'];       %��¼MAE bmR2 R2_Adjusted
evaluateFile3 = [yalmipFile,'\',dataName,'_RMSE_R2.txt'];       %��¼RMSE bmR2 R2_Adjusted
evaluateFile4 = [yalmipFile,'\',dataName,'_MAPE_R2.txt'];       %��¼MAPE bmR2 R2_Adjusted


%����Ŀ�꺯���ļ���
objectPath = [yalmipFile,'\Ŀ�꺯���ļ�'];            %% ����Ŀ�꺯���ļ���
if ~exist(objectPath)
    mkdir(objectPath)
end;    
strpatientNo = num2str(patientNo,'%3d');                        %UE����ת��Ϊ�ַ���
objectFileName = [objectPath,'\',strpatientNo,'.txt'];              %Ŀ�꺯���ļ�

%�������������ļ���
iterFile = [yalmipFile,'\���������ļ�'];            %% ����Ŀ�꺯���ļ���
if ~exist(iterFile)
    mkdir(iterFile)
end;    

iterFileName = [iterFile,'\','iter.txt'];              %���������ļ�

fVector = fopen(paramFile,'a');                 % �����ݱ��浽�ļ���
fevaluate1 = fopen(evaluateFile1,'a');          % �������ȴ������ļ�
fevaluate2 = fopen(evaluateFile2,'a');          % ƽ���������ȴ������ļ�
fevaluate3 = fopen(evaluateFile3,'a');          %���������ȴ������ļ�
fevaluate4 = fopen(evaluateFile4,'a');          %ƽ�����԰ٷֱ����ȴ������ļ�

fobject = fopen(objectFileName,'a');            % Ŀ�꺯���ļ�
fiter = fopen(iterFileName,'a');                % ���������ļ�

fprintf(fVector,'\r\n %-5d: ',patientNo);    %д�� UE ����ֵ
fprintf(fevaluate1,'%-5d ',patientNo);
fprintf(fevaluate2,'%-5d',patientNo);
fprintf(fevaluate3,'%-5d',patientNo);
fprintf(fevaluate4,'%-5d',patientNo);

%���������ļ���д�� UE + lambda + rho + iter
fprintf(fiter,'%-5d %-5d %-5d %-5d %-5d\r\n',patientNo, lambda, rho,mu, history.OriginalResidualsIter);

for i = 1:K                                   %Ŀ�꺯���ļ���д��Ŀ�꺯��ֵ
    fprintf(fobject,'%-5d ',i);
    fprintf(fobject,'%-5d ',history.beforeADMMObjective(i));
    fprintf(fobject,'%-5d ',history.primObjective(i));
    fprintf(fobject,'%-5d',history.dualObjective(i));
    fprintf(fobject,'%-5d\r\n',history.primObjective(i));
end;

fprintf(fVector,' %-5f ',w);       %���������ļ���д���������� + �ؾ�
fprintf(fVector,' %-5f\r\n',v);

fprintf(fevaluate1,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.MSE(K), history.R2(K), history.R2_adjusted(K),history.testMSE, history.testR2, history.testR2_adjusted ); %�������ļ���д�� MSE + R2 + R2_Adjusted
fprintf(fevaluate2,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.MAE(K), history.R2(K), history.R2_adjusted(K),history.testMAE, history.testR2, history.testR2_adjusted ); %�������ļ���д�� MAE + R2 + R2_Adjusted
fprintf(fevaluate3,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.RMSE(K), history.R2(K), history.R2_adjusted(K),history.testRMSE, history.testR2, history.testR2_adjusted ); %�������ļ���д�� RMSE + R2 + R2_Adjusted
fprintf(fevaluate4,'%-5f %-5f %-5f %-5f %-5f %-5f\r\n',history.MAPE(K), history.R2(K), history.R2_adjusted(K),history.testMAPE, history.testR2, history.testR2_adjusted ); %�������ļ���д�� MAPE + R2 + R2_Adjusted
fclose(fVector);   %�ر��ļ�
fclose(fevaluate1);
fclose(fevaluate2);
fclose(fevaluate3);
fclose(fevaluate4);
fclose(fobject);
fclose(fiter)

strlambda = num2str(lambda,'%3f');
strrho = num2str(rho,'%3f');
strpatientNo = num2str(patientNo,'%3d');                        %UE����ת��Ϊ�ַ���
saveFile = [yalmipFile,dirName,strpatientNo,'��',strlambda,'��',strrho];

if ~exist(saveFile)
    mkdir(saveFile)
end;

%����Ŀ�꺯��ֵ��ͼ��
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
legend('�ֲ�ʽ�㷨','����ʽ�㷨');
%zlabel(['\fontname{����}����ֵ[m]\fontname{Euclid}this is english!\fontname{����}�������ģ� '])
ylabel('Ŀ�꺯��ֵ'); xlabel('��������');
grid on
saveObjImg = [saveFile,'\Objective.png'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\ͼ1.tiff'];
saveas(gcf,saveObjImg);


%����lasso�ع��ϵ��ͼ�񣺵������ܶ��ʱ���ܹ��۲쵽�ĸ��������������ñȽϴ�
h1 = figure;
plot(w,'k-','MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(history.optW,'r--','MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
legend('ADMM-Coefficient','Centralized-Coefficient');
ylabel('Coefficient'); xlabel('Feature');
saveObjImg = [saveFile,'\Coefficient.png'];
saveas(gcf,saveObjImg);

%����R��ֵ��R��У��ֵ
h2 = figure;
plot(1:K, history.R2,'k-','MarkerSize', 10, 'LineWidth', 2);
%hold on;
%plot(1:K, history.R2_adjusted,'r--','MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
%legend('R^2','R^2Adjusted');
xlabel('��������');
ylabel('����ϵ��R^2ֵ');
saveObjImg = [saveFile,'\R2.png'];
saveas(gcf,saveObjImg);

%����ѵ����MSE���ֵ
h3 = figure;
plot(1:K, history.MSE,'r-','MarkerSize', 10, 'LineWidth', 2);%ADMMѵ�������
hold on;
plot(1:K, history.centralizeMSE,'k--','MarkerSize', 10, 'LineWidth', 2);%����ʽѵ�������
set(gca,'FontSize',16);
legend('ADMM-MSE','Centralized-MSE');
xlabel('iter (k)');
ylabel('Mean Squared Error');
saveObjImg = [saveFile,'\MSE.png'];
saveas(gcf,saveObjImg);


%����ѵ����MAE���ֵ
h4 = figure;
plot(1:K, history.MAE,'r-','MarkerSize', 10, 'LineWidth', 2);%ADMMѵ�������
hold on;
plot(1:K, history.centralizeMAE,'k--','MarkerSize', 10, 'LineWidth', 2);%����ʽѵ�������
set(gca,'FontSize',16);
legend('ADMM-MAE','Centralized-MAE');
xlabel('iter (k)');
ylabel('Mean Absolute Error');
saveObjImg = [saveFile,'\MAE.png'];
saveas(gcf,saveObjImg);

%����ѵ����RMSE���ֵ
h5 = figure;
plot(1:K, history.RMSE,'k-','MarkerSize', 10, 'LineWidth', 2);%ADMMѵ�������
hold on;
plot(1:K, history.centralizeRMSE,'r--','MarkerSize', 10, 'LineWidth', 2);%����ʽѵ�������
set(gca,'FontSize',16);
legend('�ֲ�ʽ�㷨','����ʽ�㷨');
xlabel('��������');
ylabel('RMSE');
grid on
saveObjImg = [saveFile,'\RMSE.png'];
saveas(gcf,saveObjImg);

%����ѵ����MAE���ֵ
h6= figure;
plot(1:K, history.MAPE,'r-','MarkerSize', 10, 'LineWidth', 2);%ADMMѵ�������
hold on;
plot(1:K, history.centralizeMAPE,'k--','MarkerSize', 10, 'LineWidth', 2);%����ʽѵ�������
set(gca,'FontSize',16);
legend('ADMM-MAPE','Centralized-MAPE');
xlabel('iter (k)');
ylabel('Mean Absolute Percent Error');
saveObjImg = [saveFile,'\MAPE.png'];
saveas(gcf,saveObjImg);

%����ԭʼ�в��ż�в��ͼ��
g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k-', ...
    1:K, history.eps_pri, 'r--',  'LineWidth', 2);
set(gca,'FontSize',16);
legend('ԭʼ�в�','ֹͣ����');
ylabel('||r||_2');
grid on

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k-', ...
    1:K, history.eps_dual, 'r--', 'LineWidth', 2);
set(gca,'FontSize',16);
legend('��ż�в�','ֹͣ����');
ylabel('||s||_2'); xlabel('��������');
grid on
saveResidualImg = [saveFile,'\Residual.png'];
saveas(gcf,saveResidualImg);
% system('shutdown -s');

%����R��ֵ��R��У��ֵ
h7 = figure;
plot(1:K, history.R2_adjusted,'k-','MarkerSize', 10, 'LineWidth', 2);
set(gca,'FontSize',16);
legend('У�������ϵ��R_a^2ֵ');
xlabel('��������');
ylabel('У�������ϵ��R_a^2ֵ');
grid on
saveObjImg = [saveFile,'\R2.png'];
saveas(gcf,saveObjImg);