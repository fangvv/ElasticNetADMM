clear all;
clc;
%% ��������
data = load('sto.tab.v2.txt');
%data = load('scaledSKlearnData.txt');
x = data(:,1:9);
y = data(:,10);

%% ��������
yMean = mean(y);
yDeal = y-yMean;
xMean = mean(x);
xVar = var(x,1);
[m,n] = size(x);
xDeal = zeros(m,n);
for i = 1:m 
    for j = 1:n
        xDeal(i,j) = (x(i,j)-xMean(j))/xVar(j);
    end
end
 
%% ѵ��
runtime  = 400000;%�����Ĳ���
eps = 0.01;%��������
for Block = [1]
    %% �������ݼ�
    
    [examples,Col] = size(data);
    %% Ϊģ�ͼ��Ͻؾ�
    dataTmp = zeros(examples,Col+1);
    dataTmp(:,1) = ones(examples,1);  %���õ�һ��Ϊ1��Ϊ�ؾ��������
    dataTmp(:,2:Col+1) = data(:,:);
    data = dataTmp;

    trainingNum = floor(examples*0.7);
    testExamples = data(trainingNum+1:examples,:);
    data = data(1:trainingNum,:);

    x = data(:, 1:Col);
    y = data(:, Col+1);
    features = Col;
    oneBlock = floor(trainingNum/Block);% ÿ�����ṩ������
    %% ���ѵ����ƽ��
    his.wResult = zeros(runtime, features);
    his.lossResult = zeros(runtime, 1);
    myIter = 0;
    MSE = 0;
    R2 = 0;
    R2Adjusted = 0;
    testMSE = 0;
    testR2 = 0;
    testR2Adjusted = 0;
    for index = 1:Block
        xTrain = x((index-1)*oneBlock + 1 : index*oneBlock, :);
        yTrain = y((index-1)*oneBlock + 1 : index*oneBlock,:);
        [wResult1,lossResult,history] = stageWise(xTrain, yTrain, eps, runtime, testExamples);
        his.wResult = his.wResult + wResult1;
        his.lossResult = his.lossResult + lossResult;
        %myIter = myIter + history.myIter;
        MSE = MSE + history.MSE;
        mse = MSE
        R2 = R2 + history.R2;
        R2Adjusted = R2Adjusted + history.R2Adjusted;
        testMSE = testMSE + history.testMSE;
        testmse = testMSE
        testR2 = testR2 + history.testR2;
        testR2Adjusted = testR2Adjusted + history.testR2Adjusted;
    end
    wResult = his.wResult/Block;
    lossResult = his.lossResult/Block;
    
    myIter = floor(myIter/Block);
    MSE = MSE/Block;
    R2 = R2/Block;
    R2Adjusted = R2Adjusted/Block;
    testMSE = testMSE/Block;
    testR2 = testR2/Block;
    testR2Adjusted = testR2Adjusted/Block;
    
    %% �ҵ���ǰ·��
    p = mfilename('fullpath');
    i = findstr(p,'\');
    saveFile = p(1:i(end));

    %% ��������д�ļ�
    iterFileName = [saveFile, 'iter.txt'];              %Ŀ�꺯���ļ�
    fiter = fopen(iterFileName,'a');                % ���������ļ�
    fprintf(fiter,'%-5d %-5d\r\n',Block, myIter);
    fclose(fiter);
    %% ���д���ļ�
    evaluateFile = [saveFile,'MSE_R2.txt'];       %��¼MSE R2 R2_Adjusted
    fevaluate1 = fopen(evaluateFile,'a');          % �������ȴ������ļ�
    fprintf(fevaluate1,'%-5d %-5f %-5f %-5f %-5f %-5f %-5f\r\n',Block, MSE, R2, R2Adjusted, testMSE ,testR2,testR2Adjusted ); %�������ļ���д�� MSE + R2 + R2_Adjusted
    fclose(fevaluate1);

    %% ��������ֵ
    paramFile = [saveFile,'featureVector.txt'];
    fVector = fopen(paramFile,'a');                % �����ݱ��浽�ļ���
    fprintf(fVector,'%-5d ',Block);       %���������ļ���д����������
    fprintf(fVector,'%-5f ',his.wResult(runtime,:)/Block);       %���������ļ���д����������
    fprintf(fVector,'\r\n');
    fclose(fVector);

    %% ����wResult������������
    hold on 
    xAxis = 1:runtime;
    for i = 1:n
        plot(xAxis, wResult(:,i));
    end
    %% ����ͼ���ļ�
%     saveObjImg = [saveFile,'\coefficient.png'];
%     saveas(gcf,saveObjImg);

    %%Ŀ�꺯��ͼ
    h = figure;
    plot(xAxis, lossResult);
    %% ����ͼ���ļ�
    saveObjImg = [saveFile,'\Objective.png'];
    saveas(gcf,saveObjImg);
    
    %%MSE ͼ
    h = figure;
    plot(xAxis, history.MSEN);
    %% ����ͼ���ļ�
%     saveObjImg = [saveFile,'\MSE.png'];
%     saveas(gcf,saveObjImg);
end